#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run colorcard ΔE2000 evaluation for a fixed 5-image subset, sweeping agg methods:
  - mean
  - median
  - trim (trimmed mean)

Outputs:
  1) per-image CSV:   de2000_sweep_per_image.csv
  2) summary CSV:     de2000_sweep_summary.csv  (mean±std across 5 images per model)
  3) rank CSV:        de2000_sweep_rank.csv     (rank consistency across aggs)

Usage:
  python run_colorcard_eval_sweep.py
  python run_colorcard_eval_sweep.py --trim_ratio 0.10 --margin 0.10 --cell_inner 0.45
  python run_colorcard_eval_sweep.py --out_dir ./color_eval_out

Notes:
- Filenames are FIXED (5 testA + 1 testB reference).
- Model dirs are set to your v1 results.
- Reference quad is detected per agg run (because ref patch aggregation depends on agg),
  but quad detection itself is identical; only patch representative changes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

try:
    from skimage.color import rgb2lab, deltaE_ciede2000
except Exception as e:
    raise ImportError(
        "This script requires scikit-image. Install:\n"
        "  pip install scikit-image\n"
        f"Original error: {e}"
    )


# -----------------------------
# Fixed configuration (EDIT PATHS IF NEEDED)
# -----------------------------
ROWS = 6
COLS = 7
WARP_W = 980
WARP_H = 840

DEFAULT_MARGIN = 0.10
DEFAULT_CELL_INNER = 0.45

TESTA_FILES = [
    "L1000784.png",
    "L1000811.png",
    "L1000844.png",
    "L1000854.png",
    "L1000874.png",
]
REF_FILE = "L1000757.png"

DATASET_ROOT = Path("./pic_dataset")
TESTA_DIR = DATASET_ROOT / "testA"
TESTB_DIR = DATASET_ROOT / "testB"

MODELS = [
    ("CUT_baseline",   Path("./results/cut_base_e150_lr1e4_v1/test_latest/images/fake_B")),
    ("CUT_VEF",        Path("./results/cut_vef_e150_lr1e4_v1/test_latest/images/fake_B")),
    ("CUT_VEF_stable", Path("./results/cut_e150_lr1e4_v1/test_latest/images/fake_B")),
]

AGG_SWEEP = ["mean", "median", "trim"]


# -----------------------------
# IO / conversions
# -----------------------------
def imread_bgr(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {p}")
    return img


def bgr_to_lab_image(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb01 = (rgb.astype(np.float32) / 255.0).clip(0, 1)
    return rgb2lab(rgb01)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
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
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(bgr, M, (out_w, out_h), flags=cv2.INTER_LINEAR)


# -----------------------------
# Quad detection (robust)
# -----------------------------
@dataclass
class QuadDebug:
    gray: np.ndarray
    edges: np.ndarray
    mask: np.ndarray
    vis: np.ndarray


def find_colorcard_quad_debug(bgr: np.ndarray) -> Tuple[Optional[np.ndarray], QuadDebug]:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_blur)

    v = np.median(gray_eq)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray_eq, lower, upper)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges_close = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(edges_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    quad = None
    for cnt in contours[:25]:
        area = cv2.contourArea(cnt)
        if area < 0.02 * (h * w):
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break

    if quad is None:
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        mask = cv2.dilate(edges_close, k2, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=2)

        contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)
        if len(contours2) > 0:
            rect = cv2.minAreaRect(contours2[0])
            quad = cv2.boxPoints(rect).astype(np.float32)
    else:
        mask = edges_close.copy()

    vis = bgr.copy()
    if quad is not None:
        q = order_quad_points(quad).astype(np.int32)
        cv2.polylines(vis, [q], isClosed=True, color=(0, 255, 0), thickness=3)

    dbg = QuadDebug(gray=gray_eq, edges=edges_close, mask=mask, vis=vis)
    return quad, dbg


def save_debug(prefix: str, dbg: QuadDebug, debug_dir: Path) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{prefix}_gray.png"), dbg.gray)
    cv2.imwrite(str(debug_dir / f"{prefix}_edges.png"), dbg.edges)
    cv2.imwrite(str(debug_dir / f"{prefix}_mask.png"), dbg.mask)
    cv2.imwrite(str(debug_dir / f"{prefix}_vis.png"), dbg.vis)


# -----------------------------
# Patch iteration + aggregation
# -----------------------------
def iter_cell_patches(warp_bgr: np.ndarray, rows: int, cols: int, margin: float, cell_inner: float):
    H, W = warp_bgr.shape[:2]
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

            iw = int(round((cx1 - cx0) * cell_inner))
            ih = int(round((cy1 - cy0) * cell_inner))
            mx = (cx0 + cx1) // 2
            my = (cy0 + cy1) // 2

            px0 = max(0, mx - iw // 2)
            py0 = max(0, my - ih // 2)
            px1 = min(W, mx + iw // 2)
            py1 = min(H, my + ih // 2)

            patch = warp_bgr[py0:py1, px0:px1].copy()
            yield r, c, patch


def trimmed_mean(x: np.ndarray, trim_ratio: float) -> np.ndarray:
    if x.ndim == 1:
        xs = np.sort(x)
        n = len(xs)
        k = int(np.floor(trim_ratio * n))
        if n - 2 * k <= 1:
            return np.mean(xs)
        return np.mean(xs[k:n - k])
    xs = np.sort(x, axis=0)
    n = xs.shape[0]
    k = int(np.floor(trim_ratio * n))
    if n - 2 * k <= 1:
        return np.mean(xs, axis=0)
    return np.mean(xs[k:n - k, :], axis=0)


def agg_lab(patch_bgr: np.ndarray, agg: str, trim_ratio: float) -> np.ndarray:
    lab = bgr_to_lab_image(patch_bgr)
    pix = lab.reshape(-1, 3)
    if agg == "mean":
        return pix.mean(axis=0)
    if agg == "median":
        return np.median(pix, axis=0)
    if agg == "trim":
        return trimmed_mean(pix, trim_ratio=trim_ratio)
    raise ValueError(f"Unknown agg: {agg}")


def build_reference_patch_labs(
    ref_bgr: np.ndarray,
    agg: str,
    trim_ratio: float,
    margin: float,
    cell_inner: float,
    debug_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - quad (4,2)
      - ref_patch_labs (ROWS, COLS, 3)
    """
    quad, dbg = find_colorcard_quad_debug(ref_bgr)
    save_debug(f"ref_{agg}", dbg, debug_dir)

    if quad is None:
        raise RuntimeError(
            f"Failed to detect quad on reference for agg={agg}. "
            f"See {debug_dir}/ref_{agg}_*.png"
        )

    warp = four_point_warp(ref_bgr, quad, WARP_W, WARP_H)
    cv2.imwrite(str(debug_dir / f"ref_{agg}_crop.png"), warp)

    ref_patch_labs = np.zeros((ROWS, COLS, 3), dtype=np.float32)
    for r, c, patch in iter_cell_patches(warp, ROWS, COLS, margin, cell_inner):
        ref_patch_labs[r, c] = agg_lab(patch, agg=agg, trim_ratio=trim_ratio)
    return quad, ref_patch_labs


def compute_image_score(
    img_bgr: np.ndarray,
    ref_patch_labs: np.ndarray,
    agg: str,
    trim_ratio: float,
    margin: float,
    cell_inner: float,
) -> float:
    quad, _ = find_colorcard_quad_debug(img_bgr)
    if quad is None:
        raise RuntimeError("Failed to detect quad on current image.")

    warp = four_point_warp(img_bgr, quad, WARP_W, WARP_H)

    de_map = np.zeros((ROWS, COLS), dtype=np.float32)
    for r, c, patch in iter_cell_patches(warp, ROWS, COLS, margin, cell_inner):
        rep = agg_lab(patch, agg=agg, trim_ratio=trim_ratio)          # (3,)
        ref = ref_patch_labs[r, c].astype(np.float32).reshape(1, 1, 3)
        cur = rep.astype(np.float32).reshape(1, 1, 3)
        de_map[r, c] = float(deltaE_ciede2000(ref, cur)[0, 0])

    return float(de_map.mean())


# -----------------------------
# Rank consistency
# -----------------------------
def rankdata_small(values: List[float]) -> np.ndarray:
    """
    Simple ranking: smaller value => better (rank 1).
    Handles ties by average rank.
    """
    v = np.array(values, dtype=np.float64)
    order = np.argsort(v)
    ranks = np.empty_like(order, dtype=np.float64)

    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        # average rank for ties, ranks start at 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(a: List[float], b: List[float]) -> float:
    ra = rankdata_small(a)
    rb = rankdata_small(b)
    # Pearson on ranks
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def ordering_string(model_names: List[str], scores: List[float]) -> str:
    idx = np.argsort(np.array(scores, dtype=np.float64))  # smaller is better
    return " < ".join([model_names[i] for i in idx])


# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trim_ratio", type=float, default=0.10, help="Trim ratio for trimmed mean.")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN, help="Outer margin ratio excluded.")
    ap.add_argument("--cell_inner", type=float, default=DEFAULT_CELL_INNER, help="Inner sampling ratio per cell.")
    ap.add_argument("--out_dir", type=str, default="./color_eval_sweep_out", help="Output directory.")
    ap.add_argument("--debug_dir", type=str, default="./debug_ref_sweep", help="Debug directory.")
    return ap.parse_args()


def main():
    args = parse_args()
    trim_ratio = float(args.trim_ratio)
    margin = float(args.margin)
    cell_inner = float(args.cell_inner)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    ref_path = TESTB_DIR / REF_FILE
    ref_bgr = imread_bgr(ref_path)

    per_image_records: List[Dict] = []
    summary_records: List[Dict] = []
    rank_records: List[Dict] = []

    # Cache reference patch labs per agg
    ref_cache: Dict[str, np.ndarray] = {}

    # Sweep aggs
    for agg in AGG_SWEEP:
        # Build reference patch labs (depends on agg)
        _, ref_patch_labs = build_reference_patch_labs(
            ref_bgr=ref_bgr,
            agg=agg,
            trim_ratio=trim_ratio,
            margin=margin,
            cell_inner=cell_inner,
            debug_dir=debug_dir,
        )
        ref_cache[agg] = ref_patch_labs

        # Evaluate each model
        model_means: Dict[str, float] = {}
        model_stds: Dict[str, float] = {}

        for model_name, model_dir in MODELS:
            scores: List[float] = []
            failed = 0
            missing = 0

            for fn in TESTA_FILES:
                p = model_dir / fn
                if not p.exists():
                    missing += 1
                    per_image_records.append({
                        "agg": agg,
                        "trim_ratio": trim_ratio,
                        "margin": margin,
                        "cell_inner": cell_inner,
                        "model": model_name,
                        "filename": fn,
                        "de2000": np.nan,
                        "status": "missing",
                        "path": str(p),
                    })
                    continue

                try:
                    img_bgr = imread_bgr(p)
                    score = compute_image_score(
                        img_bgr=img_bgr,
                        ref_patch_labs=ref_patch_labs,
                        agg=agg,
                        trim_ratio=trim_ratio,
                        margin=margin,
                        cell_inner=cell_inner,
                    )
                    scores.append(score)
                    per_image_records.append({
                        "agg": agg,
                        "trim_ratio": trim_ratio,
                        "margin": margin,
                        "cell_inner": cell_inner,
                        "model": model_name,
                        "filename": fn,
                        "de2000": score,
                        "status": "ok",
                        "path": str(p),
                    })
                except Exception as e:
                    failed += 1
                    per_image_records.append({
                        "agg": agg,
                        "trim_ratio": trim_ratio,
                        "margin": margin,
                        "cell_inner": cell_inner,
                        "model": model_name,
                        "filename": fn,
                        "de2000": np.nan,
                        "status": f"failed: {e}",
                        "path": str(p),
                    })

            s = np.array(scores, dtype=np.float32)
            mean = float(np.mean(s)) if len(s) else float("nan")
            std = float(np.std(s, ddof=0)) if len(s) else float("nan")
            model_means[model_name] = mean
            model_stds[model_name] = std

            summary_records.append({
                "agg": agg,
                "trim_ratio": trim_ratio,
                "margin": margin,
                "cell_inner": cell_inner,
                "model": model_name,
                "n_ok": int(len(scores)),
                "n_total": int(len(TESTA_FILES)),
                "n_missing": int(missing),
                "n_failed": int(failed),
                "de2000_mean": mean,
                "de2000_std": std,
            })

        # Print console summary for this agg
        print("================================================")
        print(f"[AGG={agg}] trim_ratio={trim_ratio}, margin={margin}, cell_inner={cell_inner}")
        for mn in [m[0] for m in MODELS]:
            print(f"  {mn:14s}  mean±std = {model_means[mn]:.4f} ± {model_stds[mn]:.4f}")

    # Rank consistency across aggs (using per-model mean)
    # Build a table: agg -> list of model means in MODELS order
    models_order = [m[0] for m in MODELS]
    agg_to_means = {
        agg: [next(r["de2000_mean"] for r in summary_records if r["agg"] == agg and r["model"] == mn)
              for mn in models_order]
        for agg in AGG_SWEEP
    }
    agg_to_order = {agg: ordering_string(models_order, agg_to_means[agg]) for agg in AGG_SWEEP}

    # pairwise Spearman
    for i in range(len(AGG_SWEEP)):
        for j in range(i + 1, len(AGG_SWEEP)):
            a = AGG_SWEEP[i]
            b = AGG_SWEEP[j]
            rho = spearman_corr(agg_to_means[a], agg_to_means[b])
            same_order = (agg_to_order[a] == agg_to_order[b])
            rank_records.append({
                "trim_ratio": trim_ratio,
                "margin": margin,
                "cell_inner": cell_inner,
                "agg_a": a,
                "agg_b": b,
                "spearman_rho": rho,
                "order_a": agg_to_order[a],
                "order_b": agg_to_order[b],
                "same_ordering": bool(same_order),
            })

    # Save outputs
    per_image_csv = out_dir / "de2000_sweep_per_image.csv"
    summary_csv = out_dir / "de2000_sweep_summary.csv"
    rank_csv = out_dir / "de2000_sweep_rank.csv"

    pd.DataFrame(per_image_records).to_csv(per_image_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_records).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(rank_records).to_csv(rank_csv, index=False, encoding="utf-8-sig")

    print("================================================")
    print(f"[SAVED] {per_image_csv}")
    print(f"[SAVED] {summary_csv}")
    print(f"[SAVED] {rank_csv}")
    print("[RANK] model ordering by agg (smaller ΔE is better):")
    for agg in AGG_SWEEP:
        print(f"  {agg:6s}: {agg_to_order[agg]}")
    print(f"[DEBUG] reference debug images in: {debug_dir}")


if __name__ == "__main__":
    main()
