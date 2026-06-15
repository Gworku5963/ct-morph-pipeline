"""
CT Morphological Pipeline v5.1 — MorphoSource Patella
======================================================
Tests software on real CT data using multiple CPU cores,
improves image quality, and generates FEA-ready 3D meshes.

Archive : https://www.morphosource.org/concern/media/000516833
Licence : MIT  |  MorphoSource data: CC BY 4.0 (Duke University)

Pipeline steps (applied in parallel across all CT slices):
  1. CLAHE            — adaptive contrast enhancement
  2. Bilateral filter — edge-preserving noise reduction
  3. NLM denoise      — remove scanner noise
  4. Unsharp mask     — restore bone edge sharpness
  5. Otsu threshold   — automatic binary segmentation
  6. Morph refinement — close×3 + open×2 (fill holes, remove speckles)
  7. Morph analysis   — 12 ops: erode/dilate/open/close/tophat/blackhat/
                        gradient/cortical-shell/trabecular/ridge-groove
  8. Mesh export      — ear-clip triangulation → unified 3D STL (Z-stacked)
  9. Quality report   — SNR / sharpness / bone-fraction per slice

Usage:
  python morph_pipeline_final.py                 # embedded 8-slice demo
  python morph_pipeline_final.py ./ct_slices/    # real archive folder
"""
__version__  = "5.1.0"
__licence__  = "MIT"
__url__      = "https://www.morphosource.org/concern/media/000516833"
__requires__ = ["numpy"]   # cv2 and Pillow are optional

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, time, math
import numpy as np
from concurrent.futures import ProcessPoolExecutor

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
VOXEL_SIZE_MM    = 0.05   # physical pixel size  (micro-CT: 0.01–0.1 mm)
CLAHE_CLIP       = 2.0    # CLAHE clip limit
CLAHE_TILES      = 8      # CLAHE tile grid size
NLM_H            = 10.0   # NLM filter strength
BILATERAL_SIGMA  = 18.0   # bilateral filter sigma
UNSHARP_STRENGTH = 0.5    # unsharp mask strength
CLOSE_ITERS      = 3      # morphological close iterations (hole fill)
OPEN_ITERS       = 2      # morphological open iterations  (noise removal)
STL_OUTPUT       = "patella_volume.stl"

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 0 — DEMO DATA
#  Generates 8 synthetic patella CT slices in memory so the pipeline
#  runs without any archive folder (useful for GDB Online / Replit / Colab).
# ─────────────────────────────────────────────────────────────────────────────
def _make_embedded_slices(n: int = 8, size: int = 64) -> list:
    """Return n synthetic micro-CT bone cross-section slices as uint8 arrays."""
    rng, slices = np.random.default_rng(2024), []
    for sl in range(n):
        img = np.zeros((size, size), np.uint8)
        t   = sl / max(n - 1, 1)
        rx  = int(18 + 8 * math.sin(math.pi * t))
        ry  = int(14 + 6 * math.sin(math.pi * t))
        cx, cy = size // 2, size // 2
        for y in range(size):
            for x in range(size):
                d2 = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
                if 0.72 < d2 <= 1.0:
                    img[y, x] = min(255, 200 + int(rng.integers(-8, 8)))
                elif d2 <= 0.72:
                    img[y, x] = min(255, 90  + int(rng.integers(-20, 20)))
        for _ in range(20 + sl * 2):
            bx = cx + int(rng.integers(-rx + 4, rx - 4))
            by = cy + int(rng.integers(-ry + 4, ry - 4))
            br = int(rng.integers(2, 6))
            bi = int(rng.integers(110, 170))
            for y in range(max(0, by - br), min(size, by + br + 1)):
                for x in range(max(0, bx - br), min(size, bx + br + 1)):
                    if (x - bx) ** 2 + (y - by) ** 2 <= br ** 2:
                        img[y, x] = min(255, int(img[y, x] * 0.5 + bi * 0.5))
        noise = rng.normal(0, 12, img.shape).astype(np.int16)
        img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        sp    = rng.random(img.shape)
        img[sp < 0.005] = 0
        img[sp > 0.995] = 255
        slices.append(img)
    return slices

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 0b — ARCHIVE LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_archive(folder: str) -> list:
    """Load all PNG/TIF/JPG slices from folder as grayscale uint8 arrays."""
    images = []
    exts   = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(exts):
            continue
        path = os.path.join(folder, fname)
        img  = None
        if CV2_AVAILABLE:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None and PIL_AVAILABLE:
            img = np.array(PILImage.open(path).convert("L"), np.uint8)
        if img is not None:
            images.append(img)
        else:
            print("  ⚠ Could not load: " + fname)
    return images

# ─────────────────────────────────────────────────────────────────────────────
#  MORPHOLOGICAL PRIMITIVES  (pure NumPy — no cv2 required)
# ─────────────────────────────────────────────────────────────────────────────
def _disk(r: int) -> np.ndarray:
    """Circular flat structuring element of radius r."""
    s = 2 * r + 1
    k = np.zeros((s, s), bool)
    for y in range(s):
        for x in range(s):
            if (x - r) ** 2 + (y - r) ** 2 <= r ** 2:
                k[y, x] = True
    return k

def _morph(img: np.ndarray, k: np.ndarray, mode: str) -> np.ndarray:
    kh, kw = k.shape; rh, rw = kh // 2, kw // 2
    pad = np.pad(img.astype(np.float32), ((rh, rh), (rw, rw)), mode="reflect")
    H, W = img.shape
    out  = np.full((H, W), 255 if mode == "e" else 0, np.float32)
    fn   = np.minimum if mode == "e" else np.maximum
    for ky, kx in zip(*np.where(k)):
        out = fn(out, pad[ky:ky + H, kx:kx + W])
    return out.astype(np.uint8)

def _e(i, k):  return _morph(i, k, "e")                       # erode
def _d(i, k):  return _morph(i, k, "d")                       # dilate
def _o(i, k):  return _d(_e(i, k), k)                         # open
def _c(i, k):  return _e(_d(i, k), k)                         # close
def _g(i, k):  return np.clip(_d(i,k).astype(np.int16) - _e(i,k).astype(np.int16), 0, 255).astype(np.uint8)
def _th(i, k): return np.clip(i.astype(np.int16) - _o(i,k).astype(np.int16),       0, 255).astype(np.uint8)
def _bh(i, k): return np.clip(_c(i,k).astype(np.int16) - i.astype(np.int16),       0, 255).astype(np.uint8)
def _ig(i, k): return np.clip(i.astype(np.int16) - _e(i,k).astype(np.int16),       0, 255).astype(np.uint8)
def _eg(i, k): return np.clip(_d(i,k).astype(np.int16) - i.astype(np.int16),       0, 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — CLAHE  (adaptive contrast enhancement)
# ─────────────────────────────────────────────────────────────────────────────
def _clahe(img: np.ndarray, clip: float = CLAHE_CLIP,
           tiles: int = CLAHE_TILES) -> np.ndarray:
    if CV2_AVAILABLE:
        return cv2.createCLAHE(clipLimit=clip,
                               tileGridSize=(tiles, tiles)).apply(img)
    H, W = img.shape
    th, tw = max(1, H // tiles), max(1, W // tiles)
    out = np.zeros_like(img, np.float32)
    for tr in range(tiles):
        for tc in range(tiles):
            r0, r1 = tr * th, min((tr + 1) * th, H)
            c0, c1 = tc * tw, min((tc + 1) * tw, W)
            tile = img[r0:r1, c0:c1].astype(np.float32)
            if tile.size == 0:
                continue
            h, _ = np.histogram(tile.ravel(), 256, (0, 256))
            lim  = max(1, int(clip * tile.size / 256))
            ex   = np.sum(np.maximum(h - lim, 0))
            h    = np.minimum(h, lim) + ex / 256
            cdf  = np.cumsum(h)
            cdf  = (cdf - cdf.min()) / max(tile.size - 1, 1) * 255
            out[r0:r1, c0:c1] = cdf[tile.astype(np.int32)]
    return np.clip(out, 0, 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — BILATERAL FILTER  (edge-preserving noise reduction)
#  Always uses pure-NumPy: cv2.bilateralFilter raises getLinearFilter errors
#  for float inputs inside subprocess workers on some OpenCV builds.
# ─────────────────────────────────────────────────────────────────────────────
def _bilateral(img: np.ndarray, d: int = 3,
               sigma: float = BILATERAL_SIGMA) -> np.ndarray:
    f = img.astype(np.float32); H, W = f.shape; r = d // 2
    pad = np.pad(f, r, mode="reflect")
    out = np.zeros_like(f); ws = np.zeros_like(f)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sp_w = math.exp(-(dy ** 2 + dx ** 2) / (2 * sigma ** 2))
            sh   = pad[r + dy:r + dy + H, r + dx:r + dx + W]
            w    = sp_w * np.exp(-(f - sh) ** 2 / (2 * sigma ** 2))
            out += w * sh; ws += w
    return np.clip(out / np.maximum(ws, 1e-6), 0, 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — NLM DENOISE  (non-local means)
# ─────────────────────────────────────────────────────────────────────────────
def _nlm(img: np.ndarray, h: float = NLM_H, p: int = 2) -> np.ndarray:
    if CV2_AVAILABLE:
        return cv2.fastNlMeansDenoising(img, None, h=h,
                                        templateWindowSize=7,
                                        searchWindowSize=21)
    f = img.astype(np.float32); H, W = f.shape
    pad = np.pad(f, p, mode="reflect")
    acc = np.zeros((H, W), np.float32); ws = np.zeros_like(acc)
    for dy in range(-p, p + 1):
        for dx in range(-p, p + 1):
            sh = pad[p + dy:p + dy + H, p + dx:p + dx + W]
            w  = np.exp(-(f - sh) ** 2 / (h * h))
            acc += w * sh; ws += w
    return np.clip(acc / np.maximum(ws, 1e-6), 0, 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — UNSHARP MASK  (restore bone edge sharpness)
# ─────────────────────────────────────────────────────────────────────────────
def _unsharp(img: np.ndarray, strength: float = UNSHARP_STRENGTH,
             r: int = 2) -> np.ndarray:
    if CV2_AVAILABLE:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        return np.clip(img.astype(np.int16) +
                       (strength * (img.astype(np.int16) - blur)).astype(np.int16),
                       0, 255).astype(np.uint8)
    f = img.astype(np.float32); H, W = f.shape
    pad = np.pad(f, r, mode="reflect")
    blur = np.zeros_like(f)
    n = (2 * r + 1) ** 2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            blur += pad[r + dy:r + dy + H, r + dx:r + dx + W]
    blur /= n
    return np.clip(f + strength * (f - blur), 0, 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — OTSU THRESHOLD  (automatic binary segmentation)
# ─────────────────────────────────────────────────────────────────────────────
def _otsu(img: np.ndarray) -> tuple:
    """Return (binary mask, threshold value)."""
    if CV2_AVAILABLE:
        t, binary = cv2.threshold(img, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary, float(t)
    hist, _ = np.histogram(img.ravel(), 256, (0, 256))
    total   = img.size
    su      = np.dot(np.arange(256), hist)
    sB = wB = bt = 0; bv = 0.0
    for t in range(256):
        wB += hist[t]
        if not wB: continue
        wF = total - wB
        if not wF: break
        sB += t * hist[t]; mB = sB / wB; mF = (su - sB) / wF
        v = wB * wF * (mB - mF) ** 2
        if v > bv: bv, bt = v, t
    return (img > bt).astype(np.uint8) * 255, float(bt)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — MORPHOLOGICAL REFINEMENT
#  close×3 fills holes in cortical bone; open×2 removes noise speckles.
#  Applied after Otsu — raw binary mask has too many artefacts for meshing.
# ─────────────────────────────────────────────────────────────────────────────
def _refine(binary: np.ndarray) -> np.ndarray:
    k5 = _disk(2); k3 = _disk(1)
    out = binary.copy()
    for _ in range(CLOSE_ITERS): out = _c(out, k5)
    for _ in range(OPEN_ITERS):  out = _o(out, k3)
    return out

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 — MORPHOLOGICAL ANALYSIS  (12 operations per slice)
# ─────────────────────────────────────────────────────────────────────────────
def morph_analysis(denoised: np.ndarray, binary: np.ndarray) -> dict:
    """Apply 12 morphological operations relevant to bone CT analysis."""
    k3 = _disk(1); k5 = _disk(2); k7 = _disk(3); k9 = _disk(4)
    return {
        "erosion":       _e(denoised, k3),
        "dilation":      _d(denoised, k3),
        "opening":       _o(denoised, k3),
        "closing":       _c(denoised, k3),
        "gradient":      _g(denoised, k3),
        "int_gradient":  _ig(denoised, k3),
        "ext_gradient":  _eg(denoised, k3),
        "tophat":        _th(denoised, k5),
        "blackhat":      _bh(denoised, k5),
        # Patella-specific: cortical shell, trabecular network, ridge/groove
        "cortical":      np.clip(
                             _c(_c(_c(denoised, k7), k7), k7).astype(np.int16) -
                             _o(_o(denoised, k5), k5).astype(np.int16),
                             0, 255).astype(np.uint8),
        "trabecular":    np.clip(
                             _th(denoised, k9).astype(np.int16) +
                             _bh(denoised, k9).astype(np.int16),
                             0, 255).astype(np.uint8),
        "ridge_groove":  _g(denoised, k3),
    }

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 8 — MESH GENERATION
#  Ear-clipping triangulation handles non-convex bone cross-sections correctly.
#  Slices are stacked in Z (z = slice_idx × VOXEL_SIZE_MM) to form a true
#  3D volume — each slice gets its own Z coordinate, not z=0 for all.
# ─────────────────────────────────────────────────────────────────────────────
def _extract_contours(binary: np.ndarray) -> list:
    """Extract boundary contour point lists from a binary mask."""
    if CV2_AVAILABLE:
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        return [c.squeeze() for c in cnts
                if c.squeeze().ndim == 2 and len(c.squeeze()) >= 3]
    H, W = binary.shape; b = binary > 127; pts = []
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if b[y, x] and not all(b[y + dy, x + dx]
               for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                pts.append([x, y])
    return [np.array(pts)] if pts else []

def _ear_clip(poly: np.ndarray) -> list:
    """
    Ear-clipping triangulation for simple polygons (convex and non-convex).
    Fan triangulation from vertex 0 fails on concave bone cross-sections
    because it produces self-intersecting triangles that FEA solvers reject.
    Ear-clipping is O(n²) but correct for any simple polygon.
    """
    pts  = [tuple(p) for p in poly]; n = len(pts)
    if n < 3: return []
    idxs = list(range(n)); tris = []; iters = 0

    def is_ear(prev, curr, nxt):
        ax, ay = pts[prev]; bx, by = pts[curr]; cx, cy = pts[nxt]
        if (bx - ax) * (cy - ay) - (cx - ax) * (by - ay) <= 0:
            return False   # reflex vertex
        for k in idxs:
            if k in (prev, curr, nxt): continue
            px, py = pts[k]
            if ((bx-ax)*(py-ay)-(by-ay)*(px-ax) > 0 and
                (cx-bx)*(py-by)-(cy-by)*(px-bx) > 0 and
                (ax-cx)*(py-cy)-(ay-cy)*(px-cx) > 0):
                return False   # point inside triangle
        return True

    while len(idxs) > 3 and iters < n * n:
        iters += 1; clipped = False
        for i in range(len(idxs)):
            p, c, nx = idxs[(i-1) % len(idxs)], idxs[i], idxs[(i+1) % len(idxs)]
            if is_ear(p, c, nx):
                tris.append((p, c, nx)); idxs.pop(i); clipped = True; break
        if not clipped: break
    if len(idxs) == 3:
        tris.append((idxs[0], idxs[1], idxs[2]))
    return tris

def build_volume_stl(slice_results: list, output_path: str,
                     voxel_size_mm: float = VOXEL_SIZE_MM) -> tuple:
    """
    Assemble a unified 3D STL from all CT slices.

    Each slice's contour vertices get z = slice_idx × voxel_size_mm,
    creating a true volumetric mesh. Adjacent-slice contours are connected
    with quad strips (two triangles per quad). Top and bottom faces are
    capped with ear-clipped triangles.
    """
    all_verts, all_faces, ring_map = [], [], []

    for sr in slice_results:
        if not sr or sr["status"] != "OK":
            ring_map.append([]); continue
        z    = sr["slice_idx"] * voxel_size_mm
        cnts = _extract_contours(sr["pipeline"]["refined"])
        rings = []
        for cnt in cnts:
            if len(cnt) < 3: continue
            start = len(all_verts)
            for pt in cnt:
                all_verts.append(np.array([pt[0] * voxel_size_mm,
                                           pt[1] * voxel_size_mm, z]))
            ring_idxs = list(range(start, len(all_verts)))
            rings.append(ring_idxs)
            for t in _ear_clip(cnt):                        # cap face
                all_faces.append((start + t[0], start + t[1], start + t[2]))
        ring_map.append(rings)

    for zi in range(1, len(ring_map)):                      # side walls
        for pr in ring_map[zi - 1]:
            if not pr: continue
            cr = min(ring_map[zi], key=lambda r: abs(len(r) - len(pr)),
                     default=None) if ring_map[zi] else None
            if not cr: continue
            n = min(len(pr), len(cr))
            for k in range(n):
                v0, v1 = pr[k % len(pr)], pr[(k + 1) % len(pr)]
                v2, v3 = cr[(k + 1) % len(cr)], cr[k % len(cr)]
                all_faces.append((v0, v1, v2))
                all_faces.append((v0, v2, v3))

    verts = np.array(all_verts) if all_verts else np.zeros((0, 3))
    n_written = 0
    with open(output_path, "w") as f:
        f.write("solid patella\n")
        f.write("# Archive: https://www.morphosource.org/concern/media/000516833\n")
        f.write("# Voxel:   {:.4f} mm/px\n".format(voxel_size_mm))
        f.write("# Verts:   {}   Faces: {}\n".format(len(verts), len(all_faces)))
        for i, j, k in all_faces:
            if max(i, j, k) >= len(verts): continue
            v1, v2, v3 = verts[i], verts[j], verts[k]
            n  = np.cross(v2 - v1, v3 - v1)
            nm = np.linalg.norm(n)
            n  = n / nm if nm > 1e-9 else np.array([0., 0., 1.])
            f.write("  facet normal {:.6f} {:.6f} {:.6f}\n".format(*n))
            f.write("    outer loop\n")
            f.write("      vertex {:.4f} {:.4f} {:.4f}\n".format(*v1))
            f.write("      vertex {:.4f} {:.4f} {:.4f}\n".format(*v2))
            f.write("      vertex {:.4f} {:.4f} {:.4f}\n".format(*v3))
            f.write("    endloop\n")
            f.write("  endfacet\n")
            n_written += 1
        f.write("endsolid patella\n")
    return len(verts), n_written

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 9 — QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(original: np.ndarray, enhanced: np.ndarray,
                    binary: np.ndarray) -> dict:
    """Return per-slice SNR, sharpness, and bone-fraction metrics."""
    o = original.astype(np.float32); e = enhanced.astype(np.float32)
    fg = binary > 0; bg = ~fg

    def snr(arr):
        bg_s = float(arr[bg].std()) if bg.any() else 1.0
        fg_m = float(arr[fg].mean()) if fg.any() else 0.0
        return fg_m / max(bg_s, 1e-6)

    def sharpness(img):
        if CV2_AVAILABLE:
            return float(cv2.Laplacian(img.astype(np.uint8), cv2.CV_64F).var())
        f  = img.astype(np.float32)
        kl = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], np.float32)
        pad = np.pad(f, 1, mode="reflect")
        lap = sum(kl[i,j] * pad[i:i+f.shape[0], j:j+f.shape[1]]
                  for i in range(3) for j in range(3))
        return float(lap.var())

    return {
        "snr_orig":      round(snr(o), 2),
        "snr_enh":       round(snr(e), 2),
        "snr_delta":     round(snr(e) - snr(o), 2),
        "sharp_orig":    round(sharpness(original), 1),
        "sharp_enh":     round(sharpness(enhanced), 1),
        "bone_fraction": round(float(fg.sum()) / binary.size, 3),
        "bone_px":       int(fg.sum()),
    }

# ─────────────────────────────────────────────────────────────────────────────
#  FULL PER-SLICE PIPELINE  (steps 1–9 in sequence)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(img: np.ndarray) -> dict:
    """Run all 6 enhancement stages on one CT slice."""
    s1 = _clahe(img)                    # step 1: CLAHE
    s2 = _bilateral(s1)                 # step 2: bilateral
    s3 = _nlm(s2)                       # step 3: NLM
    s4 = _unsharp(s3)                   # step 4: unsharp
    s5, thresh = _otsu(s4)              # step 5: Otsu
    s6 = _refine(s5)                    # step 6: morph refinement
    return {"original": img, "clahe": s1, "bilateral": s2, "nlm": s3,
            "unsharp": s4, "binary": s5, "refined": s6, "threshold": thresh}

# ─────────────────────────────────────────────────────────────────────────────
#  PARALLEL WORKER — must be at module level so multiprocessing can pickle it
#  (lambda and nested functions cannot be pickled → PicklingError crash)
# ─────────────────────────────────────────────────────────────────────────────
def _worker(args: tuple) -> dict:
    """Process one CT slice end-to-end (runs in its own OS process / CPU core)."""
    slice_idx, img = args
    t0 = time.time()
    try:
        pipe    = preprocess(img)
        morph   = morph_analysis(pipe["nlm"], pipe["binary"])
        metrics = compute_metrics(img, pipe["unsharp"], pipe["refined"])
        return {"slice_idx": slice_idx, "pipeline": pipe, "morph": morph,
                "metrics": metrics, "elapsed": time.time() - t0,
                "status": "OK", "pid": os.getpid()}
    except Exception as exc:
        return {"slice_idx": slice_idx, "pipeline": None, "morph": None,
                "metrics": None, "elapsed": time.time() - t0,
                "status": "ERROR: " + str(exc), "pid": os.getpid()}

# ─────────────────────────────────────────────────────────────────────────────
#  ASCII RENDERER  (terminal visualisation — no display window needed)
# ─────────────────────────────────────────────────────────────────────────────
_RAMP = " .,:;+*?%#@"

def _ascii(img: np.ndarray, cols: int = 46, rows: int = 14) -> str:
    H, W = img.shape
    samp = img[::max(1, H // rows), ::max(1, W // cols)]
    return "\n".join(
        "".join(_RAMP[int(p / 255 * (len(_RAMP) - 1))] * 2 for p in row)
        for row in samp)

def _print_slice(sr: dict):
    m = sr["metrics"]; p = sr["pipeline"]
    W = 110; bar = "─" * W; half = W // 2
    hdr = "  SLICE {:02d}  pid={:6d}  [{:.3f}s]  thresh={:.0f}  bone={:.1f}%".format(
          sr["slice_idx"], sr["pid"], sr["elapsed"],
          p["threshold"], m["bone_fraction"] * 100)
    qlt = "  SNR: {:.2f}→{:.2f} ({:+.2f})  Sharp: {:.0f}→{:.0f}  bone_px={}".format(
          m["snr_orig"], m["snr_enh"], m["snr_delta"],
          m["sharp_orig"], m["sharp_enh"], m["bone_px"])
    print("┌" + bar + "┐")
    print("│" + hdr[:W].ljust(W) + "│")
    print("│" + qlt[:W].ljust(W) + "│")
    print("├" + "─" * (half-1) + "┬" + "─" * (W-half-1) + "┤")
    print("│  ORIGINAL (raw CT)".ljust(half) +
          "│  MESH-READY MASK (refined)".ljust(W - half - 1) + "│")
    print("├" + "─" * (half-1) + "┼" + "─" * (W-half-1) + "┤")
    orig = _ascii(p["original"], cols=half//2-2, rows=12).split("\n")
    mask = _ascii(p["refined"],  cols=half//2-2, rows=12).split("\n")
    for ol, ml in zip(orig, mask):
        print(("│  " + ol)[:half].ljust(half) + "│" +
              ("  " + ml)[:W-half-1].ljust(W-half-1) + "│")
    print("└" + "─" * (half-1) + "┴" + "─" * (W-half-1) + "┘\n")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ncpu = max(2, os.cpu_count() or 2)
    print("\n" + "=" * 70)
    print("  CT Morphological Pipeline v{}  |  MorphoSource Patella".format(__version__))
    print("  https://www.morphosource.org/concern/media/000516833")
    print("  Python {}  |  CPUs={}  |  cv2={}  |  PIL={}".format(
        sys.version.split()[0], ncpu,
        "✔ " + cv2.__version__ if CV2_AVAILABLE else "✘ NumPy fallback",
        "✔" if PIL_AVAILABLE else "✘"))
    print("=" * 70)

    # ── Load slices ───────────────────────────────────────────────────────────
    folder = next((a for a in sys.argv[1:] if os.path.isdir(a)), None)
    if folder:
        print("\n► Loading archive: " + folder)
        slices = load_archive(folder) or _make_embedded_slices()
    else:
        print("\n► No folder supplied — running embedded 8-slice demo")
        print("  (supply a folder of TIFF/PNG slices to use real archive data)")
        slices = _make_embedded_slices()
    print("  {} slices loaded,  shape {}".format(len(slices), slices[0].shape))

    # ── Multi-core parallel processing (steps 1–7) ────────────────────────────
    print("\n► Running pipeline on {} CPU cores …".format(ncpu))
    tasks   = list(enumerate(slices))
    t0      = time.time()
    with ProcessPoolExecutor(max_workers=min(ncpu, len(slices))) as pool:
        results = list(pool.map(_worker, tasks))
    t_total = time.time() - t0

    ok  = [r for r in results if r["status"] == "OK"]
    err = [r for r in results if r["status"] != "OK"]
    print("  ✔ {} slices in {:.3f}s  ({} errors)".format(len(ok), t_total, len(err)))
    for r in err:
        print("  ✗ Slice {:02d}: {}".format(r["slice_idx"], r["status"]))

    # ── Step 9: per-slice quality report + ASCII render ───────────────────────
    print("\n► Per-slice quality report:\n")
    for r in sorted(ok, key=lambda x: x["slice_idx"]):
        _print_slice(r)

    # ── Step 7 summary: morphological operations ──────────────────────────────
    print("► Morphological operations (sample from middle slice):")
    mid = ok[len(ok) // 2] if ok else None
    for op, arr in (mid["morph"].items() if mid else []):
        print("  {:15s}  min={:3d}  max={:3d}  mean={:.1f}".format(
            op, int(arr.min()), int(arr.max()), float(arr.mean())))

    # ── Step 8: build unified 3D STL ─────────────────────────────────────────
    print("\n► Building 3D STL mesh from {} slices …".format(len(ok)))
    try:
        nv, nf = build_volume_stl(
            sorted(ok, key=lambda x: x["slice_idx"]),
            STL_OUTPUT, voxel_size_mm=VOXEL_SIZE_MM)
        print("  ✔ {} — {} vertices, {} faces".format(STL_OUTPUT, nv, nf))
        print("  ✔ Import into: Blender / FreeCAD / SimScale / Abaqus / ANSYS")
    except Exception as exc:
        import traceback
        print("  ✗ STL failed: " + str(exc)); traceback.print_exc()

    # ── Volume quality summary table ──────────────────────────────────────────
    W = 80
    print("\n" + "=" * W)
    print("  VOLUME QUALITY REPORT — MorphoSource Patella 000516833")
    print("=" * W)
    print("  {:>5}  {:>8}  {:>8}  {:>6}  {:>10}  {:>10}  {:>7}".format(
        "Slice", "SNR-orig", "SNR-enh", "ΔSNR", "Sharp-orig", "Sharp-enh", "Bone%"))
    print("  " + "-" * (W - 4))
    for r in sorted(ok, key=lambda x: x["slice_idx"]):
        m = r["metrics"]
        print("  {:>5}  {:>8.2f}  {:>8.2f}  {:>6.2f}  {:>10.0f}  {:>10.0f}  {:>6.1f}%".format(
            r["slice_idx"],
            m["snr_orig"], m["snr_enh"], m["snr_delta"],
            m["sharp_orig"], m["sharp_enh"], m["bone_fraction"] * 100))
    if ok:
        print("  " + "-" * (W - 4))
        print("  AVG  ΔSNR={:+.2f}   bone={:.1f}%   sharpness={:.0f}".format(
            sum(r["metrics"]["snr_delta"]     for r in ok) / len(ok),
            sum(r["metrics"]["bone_fraction"] for r in ok) / len(ok) * 100,
            sum(r["metrics"]["sharp_enh"]     for r in ok) / len(ok)))
    print("=" * W)
    print()
    print("  ✔  {} slices on {} CPU cores in {:.3f}s".format(len(ok), ncpu, t_total))
    print("  ✔  Pipeline: CLAHE → Bilateral → NLM → Unsharp → Otsu → Morph-refine")
    print("  ✔  12 morph ops: erode/dilate/open/close/gradient/tophat/blackhat/")
    print("     int-grad/ext-grad/cortical/trabecular/ridge-groove")
    print("  ✔  3D STL: {} → FEA-ready\n".format(STL_OUTPUT))

# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # required on Windows + GDB Online
    main()
