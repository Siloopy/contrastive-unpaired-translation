#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Color card ΔE2000 evaluation for a fixed small subset (5 images).

Pipeline:
1) Detect color card quad on reference image (testB) -> perspective warp
2) Warp each test image (testA + model outputs fake_B) with the same quad-detection strategy
3) Split warped image into ROWS x COLS patches
4) For each patch, compute representative Lab color using:
   - mean
   - median
   - trimmed mean (robust to highlights/text)
5) Compute ΔE2000 to reference patch Lab, average across patches -> per-image score
6) Summarize mean ± std per model and save CSV

Usage examples:
  python run_colorcard_eval_cli.py --agg mean
  python run_colorcard_eval_cli.py --agg median
  python run_colorcard_eval_cli.py --agg trim --trim_ratio 0.1

Notes:
- Filenames are FIXED to your 5 testA images and 1 testB reference.
- Model directories are set to your v1 results.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

# Prefer scikit-image for Lab + CIEDE2000
try:
    from skimage.color import rgb2lab, deltaE_ciede2000
except Exception as e:
    raise ImportError(
        "This script requires scikit-image. Please install it:\n"
        "  pip install scikit-image\n"
        f"Original error: {e}"
    )


# -----------------------------
# 0) Fixed configuration (YOU CAN EDIT PATHS HERE)
# -----------------------------
ROWS = 6
COLS = 7

# Output warp size (should keep your previous settings)
WARP_W = 980
WARP_H = 840

# Default sampling geometry (can be overridden by CLI args)
DEFAULT_MARGIN = 0.06      # exclude outer border of the warped card
DEFAULT_CELL_INNER = 0.55  # sample central region of each cell (fraction of cell size)

# Your fixed subset (5 testA) and reference (testB)
TESTA_FILES = [
    "L1000784.png",
    "L1000811.png",
    "L1000844.png",
    "L1000854.png",
    "L1000874.png",
]
REF_FILE = "L1000757.png"  # testB reference

# Dataset roots (edit if your dataset folder differs)
DATASET_ROOT = Path("./pic_dataset")
TESTA_DIR = DATASET_ROOT / "testA"
TESTB_DIR = DATASET_ROOT / "testB"

# Your v1 results (fake_B folders)
MODELS = [
    ("CUT_baseline", Path("./results/cut_base_e150_lr1e4_v1/test_latest/images/fake_B")),
    ("CUT_VEF",      Path("./results/cut_vef_e150_lr1e4_v1/test_latest/images/fake_B")),
    ("CUT_VEF_stable", Path("./results/cut_e150_lr1e4_v1/test_latest/images/fake_B")),
]

# Output / debug
OUT_CSV_DEFAULT = Path("./de2000_subset.csv")
DEBUG_DIR_DEFAULT = Path("./debug_ref")


# -----------------------------
# 1) Utilities
# -----------------------------
def imread_bgr(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {p}")
    return img


def bgr_to_lab_image(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 -> Lab float (skimage, D65)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb01 = (rgb.astype(np.float32) / 255.0).clip(0, 1)
    lab = rgb2lab(rgb01)  # HxWx3 float
    return lab


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    pts: (4,2)
    """
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)


def four_point_warp(bgr: np.ndarray, quad: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    quad = order_quad_points(quad)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(bgr, M, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return warped


# -----------------------------
# 2) Robust quad detection (reference / each image)
# -----------------------------
@dataclass
class QuadDebug:
    gray: np.ndarray
    edges: np.ndarray
    mask: np.ndarray
    vis: np.ndarray


def find_colorcard_quad_debug(bgr: np.ndarray) -> Tuple[np.ndarray | None, QuadDebug]:
    """
    More robust:
    - gray -> blur -> CLAHE
    - Canny -> close
    - try largest 4-point contour
    - fallback: dilate/close -> take largest contour -> minAreaRect
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Slight denoise + contrast enhance (important for low-contrast / bright haze)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_blur)

    # Canny
    v = np.median(gray_eq)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray_eq, lower, upper)

    # Close gaps
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges_close = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)

    # 1) Try find a good 4-pt contour
    contours, _ = cv2.findContours(edges_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    quad = None
    for cnt in contours[:20]:
        area = cv2.contourArea(cnt)
        if area < 0.02 * (h * w):
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break

    # 2) Fallback: binarize edges into big blob -> minAreaRect
    if quad is None:
        # thicken edges and connect structures
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        mask = cv2.dilate(edges_close, k2, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=2)

        contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)

        if len(contours2) > 0:
            cnt = contours2[0]
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)  # 4x2
            quad = box.astype(np.float32)
    else:
        mask = edges_close.copy()

    # Debug visualization
    vis = bgr.copy()
    if quad is not None:
        q = order_quad_points(quad).astype(np.int32)
        cv2.polylines(vis, [q], isClosed=True, color=(0, 255, 0), thickness=3)

    dbg = QuadDebug(
        gray=gray_eq,
        edges=edges_close,
        mask=mask,
        vis=vis,
    )
    return quad, dbg


def save_debug(prefix: str, dbg: QuadDebug, debug_dir: Path) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{prefix}_gray.png"), dbg.gray)
    cv2.imwrite(str(debug_dir / f"{prefix}_edges.png"), dbg.edges)
    cv2.imwrite(str(debug_dir / f"{prefix}_mask.png"), dbg.mask)
    cv2.imwrite(str(debug_dir / f"{prefix}_vis.png"), dbg.vis)


# -----------------------------
# 3) Patch sampling + aggregation
# -----------------------------
def trimmed_mean(x: np.ndarray, trim_ratio: float) -> np.ndarray:
    """
    x: (N, C) or (N,)
    returns: (C,) or scalar
    Trim both tails by trim_ratio.
    """
    if x.ndim == 1:
        x_sorted = np.sort(x)
        n = len(x_sorted)
        k = int(np.floor(trim_ratio * n))
        if n - 2 * k <= 1:
            return np.mean(x_sorted)
        return np.mean(x_sorted[k : n - k])
    else:
        # per-channel trim
        x_sorted = np.sort(x, axis=0)
        n = x_sorted.shape[0]
        k = int(np.floor(trim_ratio * n))
        if n - 2 * k <= 1:
            return np.mean(x_sorted, axis=0)
        return np.mean(x_sorted[k : n - k, :], axis=0)


def agg_lab(patch_lab: np.ndarray, agg: str, trim_ratio: float) -> np.ndarray:
    """
    patch_lab: (h, w, 3) Lab
    returns: (3,) representative Lab
    """
    pix = patch_lab.reshape(-1, 3)
    if agg == "mean":
        return pix.mean(axis=0)
    if agg == "median":
        return np.median(pix, axis=0)
    if agg == "trim":
        return trimmed_mean(pix, trim_ratio=trim_ratio)
    raise ValueError(f"Unknown agg method: {agg}")


def iter_cell_patches(warp_bgr: np.ndarray, rows: int, cols: int, margin: float, cell_inner: float):
    """
    Yield (r, c, patch_bgr, rect) for each cell.
    rect: (x0, y0, x1, y1) in warp coords
    """
    H, W = warp_bgr.shape[:2]

    # usable region excluding outer margin
    x0 = int(round(W * margin))
    y0 = int(round(H * margin))
    x1 = int(round(W * (1 - margin)))
    y1 = int(round(H * (1 - margin)))

    usable_w = x1 - x0
    usable_h = y1 - y0
    cell_w = usable_w / cols
    cell_h = usable_h / rows

    for r in range(rows):
        for c in range(cols):
            cx0 = x0 + int(round(c * cell_w))
            cy0 = y0 + int(round(r * cell_h))
            cx1 = x0 + int(round((c + 1) * cell_w))
            cy1 = y0 + int(round((r + 1) * cell_h))

            # inner sampling region (center box)
            iw = int(round((cx1 - cx0) * cell_inner))
            ih = int(round((cy1 - cy0) * cell_inner))
            mx = (cx0 + cx1) // 2
            my = (cy0 + cy1) // 2
            px0 = max(0, mx - iw // 2)
            py0 = max(0, my - ih // 2)
            px1 = min(W, mx + iw // 2)
            py1 = min(H, my + ih // 2)

            patch = warp_bgr[py0:py1, px0:px1].copy()
            yield r, c, patch, (px0, py0, px1, py1)


def compute_image_de2000(
    img_bgr: np.ndarray,
    ref_patch_labs: np.ndarray,
    agg: str,
    trim_ratio: float,
    margin: float,
    cell_inner: float,
) -> Tuple[float, np.ndarray]:
    """
    returns:
      - per-image mean ΔE2000 across all patches
      - per-patch ΔE array shape (rows, cols)
    """
    warp_quad, _ = find_colorcard_quad_debug(img_bgr)
    if warp_quad is None:
        raise RuntimeError("Failed to detect quad on current image.")

    warp = four_point_warp(img_bgr, warp_quad, WARP_W, WARP_H)
    warp_lab = bgr_to_lab_image(warp)

    de_map = np.zeros((ROWS, COLS), dtype=np.float32)

    # iterate patches in the same order as ref_patch_labs
    for r, c, patch_bgr, _ in iter_cell_patches(warp, ROWS, COLS, margin, cell_inner):
        patch_lab = bgr_to_lab_image(patch_bgr)
        rep = agg_lab(patch_lab, agg=agg, trim_ratio=trim_ratio)  # (3,)
        ref = ref_patch_labs[r, c]  # (3,)

        # deltaE_ciede2000 expects shape (..., 3)
        d = float(deltaE_ciede2000(ref.reshape(1, 1, 3), rep.reshape(1, 1, 3))[0, 0])
        de_map[r, c] = d

    score = float(de_map.mean())
    return score, de_map


def build_reference_patch_labs(
    ref_bgr: np.ndarray,
    agg: str,
    trim_ratio: float,
    margin: float,
    cell_inner: float,
    debug_dir: Path,
) -> np.ndarray:
    """
    Detect quad on reference, warp, then compute ref patch Lab reps.
    Returns array (ROWS, COLS, 3)
    """
    quad, dbg = find_colorcard_quad_debug(ref_bgr)
    save_debug("ref", dbg, debug_dir)

    if quad is None:
        raise RuntimeError(f"Failed to detect color card quad on reference. See {debug_dir}/ref_*.png")

    warp = four_point_warp(ref_bgr, quad, WARP_W, WARP_H)
    cv2.imwrite(str(debug_dir / "ref_crop.png"), warp)

    ref_patch_labs = np.zeros((ROWS, COLS, 3), dtype=np.float32)

    for r, c, patch_bgr, _ in iter_cell_patches(warp, ROWS, COLS, margin, cell_inner):
        patch_lab = bgr_to_lab_image(patch_bgr)
        ref_patch_labs[r, c] = agg_lab(patch_lab, agg=agg, trim_ratio=trim_ratio)

    return ref_patch_labs


# -----------------------------
# 4) Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", type=str, default="median", choices=["mean", "median", "trim"],
                    help="Patch statistic method.")
    ap.add_argument("--trim_ratio", type=float, default=0.10,
                    help="Trim ratio for trimmed-mean (only used when --agg=trim).")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                    help="Outer margin ratio excluded from warped card.")
    ap.add_argument("--cell_inner", type=float, default=DEFAULT_CELL_INNER,
                    help="Inner sampling ratio of each cell.")
    ap.add_argument("--out_csv", type=str, default=str(OUT_CSV_DEFAULT),
                    help="Output CSV path.")
    ap.add_argument("--debug_dir", type=str, default=str(DEBUG_DIR_DEFAULT),
                    help="Debug directory for reference detection images.")
    return ap.parse_args()


def main():
    args = parse_args()
    agg = args.agg
    trim_ratio = float(args.trim_ratio)
    margin = float(args.margin)
    cell_inner = float(args.cell_inner)
    out_csv = Path(args.out_csv)
    debug_dir = Path(args.debug_dir)

    # Load reference
    ref_path = TESTB_DIR / REF_FILE
    ref_bgr = imread_bgr(ref_path)

    # Build reference patch Lab reps (depends on agg method)
    ref_patch_labs = build_reference_patch_labs(
        ref_bgr=ref_bgr,
        agg=agg,
        trim_ratio=trim_ratio,
        margin=margin,
        cell_inner=cell_inner,
        debug_dir=debug_dir,
    )

    records: List[Dict] = []

    # Evaluate each model
    for model_name, model_dir in MODELS:
        scores = []
        missing = []
        failed = 0

        for fn in TESTA_FILES:
            test_path = model_dir / fn
            if not test_path.exists():
                missing.append(str(test_path))
                continue
            try:
                img_bgr = imread_bgr(test_path)
                score, _ = compute_image_de2000(
                    img_bgr=img_bgr,
                    ref_patch_labs=ref_patch_labs,
                    agg=agg,
                    trim_ratio=trim_ratio,
                    margin=margin,
                    cell_inner=cell_inner,
                )
                scores.append(score)
                records.append({
                    "model": model_name,
                    "filename": fn,
                    "de2000_mean_over_patches": score,
                    "agg": agg,
                    "trim_ratio": trim_ratio,
                    "margin": margin,
                    "cell_inner": cell_inner,
                })
            except Exception as e:
                failed += 1
                records.append({
                    "model": model_name,
                    "filename": fn,
                    "de2000_mean_over_patches": np.nan,
                    "error": str(e),
                    "agg": agg,
                    "trim_ratio": trim_ratio,
                    "margin": margin,
                    "cell_inner": cell_inner,
                })

        scores_np = np.array(scores, dtype=np.float32)
        if len(scores_np) > 0:
            mean = float(scores_np.mean())
            std = float(scores_np.std(ddof=0))
        else:
            mean, std = float("nan"), float("nan")

        print("================================================")
        print(f"[{model_name}] evaluated {len(scores)}/{len(TESTA_FILES)}")
        if missing:
            print(f"[{model_name}] missing files: {len(missing)}")
        if failed:
            print(f"[{model_name}] failed detections: {failed}")
        print(f"[{model_name}] Mean ΔE2000 = {mean:.4f} ± {std:.4f}")

    # Save CSV (per-image rows)
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("================================================")
    print(f"[SAVED] {out_csv}")
    print(f"[INFO] AGG_METHOD={agg}, TRIM_RATIO={trim_ratio}, MARGIN={margin}, CELL_INNER={cell_inner}")
    print(f"[INFO] Reference debug: {debug_dir}/ref_*.png and {debug_dir}/ref_crop.png")


if __name__ == "__main__":
    main()
