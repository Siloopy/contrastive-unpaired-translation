#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Color card ΔE2000 evaluation for a small subset.
- Detect the color card quad on a reference image (testB) -> perspective warp
- Warp each test image (testA and model outputs fake_B) using the same quad detection strategy
- Split into ROWS x COLS patches
- For each patch, compute Lab representative color using:
    * median (default) OR
    * trimmed mean (robust to highlights/text)
- Compute ΔE2000 to reference patch color, average across all patches -> per-image score
- Summarize mean ± std per model, save CSV.

Author: (replace if needed)
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

# ============================================================
# 0) 配置区：按你的工程实际改这里
# ============================================================

# 你要评估的模型输出 fake_B 路径（你说只用 v1 结尾的）
MODELS: List[Tuple[str, Path]] = [
    ("CUT_baseline", Path("./results/cut_base_e150_lr1e4_v1/test_latest/images/fake_B")),
    ("CUT_VEF", Path("./results/cut_vef_e150_lr1e4_v1/test_latest/images/fake_B")),
    ("CUT_VEF_stable", Path("./results/cut_e150_lr1e4_v1/test_latest/images/fake_B")),
]

# 你的色卡数据集位置（按你当前结构）
TESTA_DIR = Path("./pic_dataset/testA")  # 原始弱光/不同光照图（同一色卡）
TESTB_DIR = Path("./pic_dataset/testB")  # reference 图所在目录

# 你只想评估的 testA 子集（5张）
SELECT_A = [
    "L1000784.png",
    "L1000811.png",
    "L1000844.png",
    "L1000854.png",
    "L1000874.png",
]

# reference 文件名（testB里那张）
REF_FILENAME = "L1000757.png"

# 色卡网格尺寸
ROWS = 6
COLS = 7

# 透视矫正后的统一尺寸（影响每个patch像素数量，但不影响网格逻辑）
WARP_W = 980
WARP_H = 840

# patch 内采样区域控制（你之前在做稳健性分析）
# MARGIN 越大：越远离色块边界（更保守）；CELL_INNER 越小：采样窗口越小（更保守）
MARGIN = 0.08
CELL_INNER = 0.50

# ---------- 核心：patch统计方式 ----------
# 1) "median"：中位数（默认，最抗离群）
# 2) "trimmed_mean"：截尾均值（对反光/文字也很稳）
AGG_METHOD = "median"         # "median" or "trimmed_mean"
TRIM_RATIO = 0.10             # 仅 trimmed_mean 生效：两端各丢弃10%

# 输出
OUT_CSV = Path("./de2000_subset.csv")
DEBUG_DIR = Path("./debug_ref")
DEBUG_DIR.mkdir(exist_ok=True, parents=True)

# 图像读入方式：OpenCV默认BGR，这里统一转RGB/Lab时会处理
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ============================================================
# 1) 工具函数：排序四点、ΔE2000、鲁棒统计
# ============================================================

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        pts = pts.reshape(4, 2)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0)


def rgb_to_lab_uint8(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> Lab float32 (OpenCV Lab: L in [0,255], a,b in [0,255] offset)."""
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab.astype(np.float32)


def agg_color_lab(lab_patch: np.ndarray, method: str = "median", trim_ratio: float = 0.1) -> np.ndarray:
    """
    lab_patch: HxWx3 float32
    return: (3,) float32 representative Lab (OpenCV Lab space)
    """
    flat = lab_patch.reshape(-1, 3)  # Nx3
    if flat.shape[0] == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    if method == "median":
        return np.median(flat, axis=0).astype(np.float32)

    if method == "trimmed_mean":
        # sort each channel and trim both ends
        n = flat.shape[0]
        k = int(n * trim_ratio)
        if 2 * k >= n:
            # 太极端就退化为median
            return np.median(flat, axis=0).astype(np.float32)

        out = []
        for c in range(3):
            vec = np.sort(flat[:, c])
            vec = vec[k:n - k]
            out.append(np.mean(vec))
        return np.array(out, dtype=np.float32)

    raise ValueError(f"Unknown AGG_METHOD: {method}")


# ---- ΔE2000 (CIEDE2000) 实现 ----
# 参考标准实现写法（数值稳定版）
def delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """
    lab1, lab2: shape (3,) in OpenCV Lab space.
    Convert OpenCV Lab to standard Lab:
      L*: lab[...,0] * 100/255
      a*: lab[...,1] - 128
      b*: lab[...,2] - 128
    Return ΔE00
    """
    L1 = lab1[0] * 100.0 / 255.0
    a1 = lab1[1] - 128.0
    b1 = lab1[2] - 128.0

    L2 = lab2[0] * 100.0 / 255.0
    a2 = lab2[1] - 128.0
    b2 = lab2[2] - 128.0

    # Weighting factors
    kL = kC = kH = 1.0

    C1 = np.sqrt(a1 * a1 + b1 * b1)
    C2 = np.sqrt(a2 * a2 + b2 * b2)
    Cbar = (C1 + C2) / 2.0

    Cbar7 = Cbar ** 7
    G = 0.5 * (1.0 - np.sqrt(Cbar7 / (Cbar7 + (25.0 ** 7))))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2

    C1p = np.sqrt(a1p * a1p + b1 * b1)
    C2p = np.sqrt(a2p * a2p + b2 * b2)

    def hp(ap, b):
        if ap == 0 and b == 0:
            return 0.0
        h = np.degrees(np.arctan2(b, ap))
        return h + 360.0 if h < 0 else h

    h1p = hp(a1p, b1)
    h2p = hp(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    if C1p * C2p == 0:
        dHp = 0.0
    else:
        if dhp > 180:
            dhp -= 360
        elif dhp < -180:
            dhp += 360
        dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lbarp = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0

    if C1p * C2p == 0:
        hbarp = h1p + h2p
    else:
        hsum = h1p + h2p
        hdiff = abs(h1p - h2p)
        if hdiff > 180:
            hbarp = (hsum + 360.0) / 2.0 if hsum < 360.0 else (hsum - 360.0) / 2.0
        else:
            hbarp = hsum / 2.0

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hbarp - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hbarp))
        + 0.32 * np.cos(np.radians(3.0 * hbarp + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hbarp - 63.0))
    )

    dtheta = 30.0 * np.exp(-(((hbarp - 275.0) / 25.0) ** 2))
    RC = 2.0 * np.sqrt((Cbarp ** 7) / ((Cbarp ** 7) + (25.0 ** 7)))
    SL = 1.0 + (0.015 * ((Lbarp - 50.0) ** 2)) / np.sqrt(20.0 + ((Lbarp - 50.0) ** 2))
    SC = 1.0 + 0.045 * Cbarp
    SH = 1.0 + 0.015 * Cbarp * T
    RT = -np.sin(np.radians(2.0 * dtheta)) * RC

    dE = np.sqrt(
        (dLp / (kL * SL)) ** 2
        + (dCp / (kC * SC)) ** 2
        + (dHp / (kH * SH)) ** 2
        + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )
    return float(dE)


# ============================================================
# 2) 色卡四边形检测 + 透视矫正（带debug输出）
# ============================================================

def find_colorcard_quad_debug(bgr: np.ndarray, dbg_prefix: str = "ref"):
    """
    更鲁棒版本：
    1) Canny + close + 找最大四边形（原思路）
    2) 失败则：对 edges 做 dilate/close 形成连通块，取最大轮廓 -> minAreaRect 兜底
    输出 debug：*_gray, *_edges, *_bin, *_vis
    """
    dbg = {}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dbg[f"{dbg_prefix}_gray"] = gray.copy()

    # 轻微降噪 + 提升对比（对你这种亮、灰、对比低的图更重要）
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_blur)

    # Canny 阈值下调（更容易把外框抓出来）
    edges = cv2.Canny(gray_eq, 20, 80)
    dbg[f"{dbg_prefix}_edges"] = edges.copy()

    # 先 close 连接断边
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_close, iterations=2)

    # 再 dilate 让外框更粗、更连通（兜底非常关键）
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thick = cv2.dilate(closed, k_dil, iterations=2)

    _, bin_img = cv2.threshold(thick, 0, 255, cv2.THRESH_BINARY)
    dbg[f"{dbg_prefix}_bin"] = bin_img.copy()

    h, w = gray.shape[:2]
    img_area = float(h * w)

    # ---------- (A) 优先：找“最大四边形轮廓” ----------
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.01 * img_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if area > best_area:
                best_area = area
                best_quad = approx.reshape(4, 2).astype(np.float32)

    # 可视化底图
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if best_quad is not None:
        quad = order_points(best_quad)
        cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 0), 3)
        dbg[f"{dbg_prefix}_vis"] = vis
        return quad, dbg

    # ---------- (B) 兜底：minAreaRect（即使外框不闭合也能抓到矩形区域） ----------
    if len(contours) == 0:
        dbg[f"{dbg_prefix}_vis"] = vis
        return None, dbg

    # 取最大轮廓做 minAreaRect
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 0.005 * img_area:
        dbg[f"{dbg_prefix}_vis"] = vis
        return None, dbg

    rect = cv2.minAreaRect(cnt)  # ((cx,cy),(rw,rh),angle)
    box = cv2.boxPoints(rect)    # 4x2
    box = box.astype(np.float32)

    quad = order_points(box)
    cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 255), 3)  # 黄框表示兜底模式
    dbg[f"{dbg_prefix}_vis"] = vis
    return quad, dbg


def warp_by_quad(bgr: np.ndarray, quad: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    quad = order_points(quad)
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(bgr, M, (out_w, out_h))
    return warped


# ============================================================
# 3) 从 warp 后的色卡图中提取每个patch Lab代表色，并算 ΔE00
# ============================================================

def get_patch_boxes(w: int, h: int, rows: int, cols: int, margin: float, inner: float) -> List[Tuple[int, int, int, int]]:
    """
    Return list of (x1,y1,x2,y2) for inner sampling boxes of each patch.
    margin: relative margin from patch boundary (0~0.5)
    inner: relative size of inner box compared to patch size (0~1)
    """
    patch_w = w / cols
    patch_h = h / rows
    boxes = []

    for r in range(rows):
        for c in range(cols):
            x0 = c * patch_w
            y0 = r * patch_h
            x1 = (c + 1) * patch_w
            y1 = (r + 1) * patch_h

            # apply margin then take inner box
            pw = (x1 - x0)
            ph = (y1 - y0)

            mx = pw * margin
            my = ph * margin

            # inner box size
            iw = pw * inner
            ih = ph * inner

            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0

            ix0 = int(round(cx - iw / 2.0))
            iy0 = int(round(cy - ih / 2.0))
            ix1 = int(round(cx + iw / 2.0))
            iy1 = int(round(cy + ih / 2.0))

            # clamp
            ix0 = max(0, min(w - 1, ix0))
            iy0 = max(0, min(h - 1, iy0))
            ix1 = max(1, min(w, ix1))
            iy1 = max(1, min(h, iy1))

            boxes.append((ix0, iy0, ix1, iy1))
    return boxes


def compute_de2000_image(
    bgr_img: np.ndarray,
    ref_lab_patches: List[np.ndarray],
    rows: int,
    cols: int,
    margin: float,
    inner: float,
    agg_method: str,
    trim_ratio: float,
) -> Tuple[float, List[float]]:
    """
    Compute mean ΔE00 for one image vs reference (patch-wise).
    Return (mean_de, de_list_per_patch).
    """
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    lab = rgb_to_lab_uint8(rgb)

    h, w = lab.shape[:2]
    boxes = get_patch_boxes(w, h, rows, cols, margin, inner)

    de_list = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        patch = lab[y0:y1, x0:x1, :]
        lab_rep = agg_color_lab(patch, method=agg_method, trim_ratio=trim_ratio)
        de = delta_e_ciede2000(lab_rep, ref_lab_patches[i])
        de_list.append(de)

    return float(np.mean(de_list)), de_list


# ============================================================
# 4) 主流程
# ============================================================

def main() -> None:
    ref_path = TESTB_DIR / REF_FILENAME
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference not found: {ref_path}")

    ref_bgr = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    if ref_bgr is None:
        raise RuntimeError(f"Failed to read reference image: {ref_path}")

    # detect quad on reference + debug dump
    quad, dbg = find_colorcard_quad_debug(ref_bgr, dbg_prefix="ref")
    # write debug images
    for k, im in dbg.items():
        outp = DEBUG_DIR / f"{k}.png"
        cv2.imwrite(str(outp), im)

    if quad is None:
        raise RuntimeError(f"Failed to detect color card quad on reference. See {DEBUG_DIR}/ref_vis.png / ref_edges.png / ref_th.png")

    # warp reference
    ref_warp = warp_by_quad(ref_bgr, quad, WARP_W, WARP_H)
    cv2.imwrite(str(DEBUG_DIR / "ref_crop.png"), ref_warp)

    # build reference patch Lab reps
    ref_rgb = cv2.cvtColor(ref_warp, cv2.COLOR_BGR2RGB)
    ref_lab = rgb_to_lab_uint8(ref_rgb)
    boxes = get_patch_boxes(WARP_W, WARP_H, ROWS, COLS, MARGIN, CELL_INNER)

    ref_lab_patches: List[np.ndarray] = []
    for (x0, y0, x1, y1) in boxes:
        patch = ref_lab[y0:y1, x0:x1, :]
        ref_rep = agg_color_lab(patch, method=AGG_METHOD, trim_ratio=TRIM_RATIO)
        ref_lab_patches.append(ref_rep)

    # evaluate each model
    rows_out = []

    for model_name, fake_dir in MODELS:
        # sanity check
        if not fake_dir.exists():
            print(f"[WARN] {model_name} fake_B dir not found: {fake_dir}")
            continue

        per_img_scores = []
        evaluated = 0

        for fn in SELECT_A:
            # model output name is same as testA (你说训练后色卡文件名一致)
            img_path = fake_dir / fn
            if not img_path.exists():
                # 兜底：有些框架可能是 .jpg
                alt = None
                stem = Path(fn).stem
                for ext in [".png", ".jpg", ".jpeg"]:
                    cand = fake_dir / f"{stem}{ext}"
                    if cand.exists():
                        alt = cand
                        break
                if alt is None:
                    rows_out.append({
                        "model": model_name,
                        "filename": fn,
                        "mean_de2000": np.nan,
                        "note": "missing_generated",
                    })
                    continue
                img_path = alt

            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                rows_out.append({
                    "model": model_name,
                    "filename": fn,
                    "mean_de2000": np.nan,
                    "note": "read_failed",
                })
                continue

            # detect quad on this image, warp
            q, _ = find_colorcard_quad_debug(bgr, dbg_prefix="tmp")
            if q is None:
                rows_out.append({
                    "model": model_name,
                    "filename": fn,
                    "mean_de2000": np.nan,
                    "note": "quad_failed",
                })
                continue

            warp = warp_by_quad(bgr, q, WARP_W, WARP_H)

            mean_de, _ = compute_de2000_image(
                warp,
                ref_lab_patches,
                ROWS,
                COLS,
                MARGIN,
                CELL_INNER,
                AGG_METHOD,
                TRIM_RATIO,
            )

            per_img_scores.append(mean_de)
            evaluated += 1

            rows_out.append({
                "model": model_name,
                "filename": fn,
                "mean_de2000": mean_de,
                "note": "",
            })

        if evaluated > 0:
            m = float(np.mean(per_img_scores))
            s = float(np.std(per_img_scores, ddof=0))
            print("=" * 55)
            print(f"[{model_name}] evaluated {evaluated}/{len(SELECT_A)}")
            print(f"[{model_name}] Mean ΔE2000 = {m:.4f} ± {s:.4f}")
        else:
            print("=" * 55)
            print(f"[{model_name}] evaluated 0/{len(SELECT_A)}  (all failed/missing)")

    # save csv
    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("=" * 55)
    print(f"[SAVED] {OUT_CSV.resolve()}")
    print(f"[INFO] AGG_METHOD={AGG_METHOD}, TRIM_RATIO={TRIM_RATIO}, MARGIN={MARGIN}, CELL_INNER={CELL_INNER}")


if __name__ == "__main__":
    main()
