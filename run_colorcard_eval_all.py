# run_colorcard_eval_all.py
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# =============================
# 0) 配置区
# =============================

SELECT_A = [
    "L1000784.png",
    "L1000811.png",
    "L1000844.png",
    "L1000854.png",
    "L1000874.png",
]

REF_PATH = Path("./pic_dataset/testB/L1000757.png")

MODELS = [
    ("CUT_baseline",   Path("./results/cut_base_e150_lr1e4_v1/test_latest/images/fake_B")),
    ("CUT_VEF",        Path("./results/cut_vef_e150_lr1e4_v1/test_latest/images/fake_B")),
    ("CUT_VEF_stable", Path("./results/cut_e150_lr1e4_v1/test_latest/images/fake_B")),
]

ROWS = 6
COLS = 7

WARP_W = 980
WARP_H = 840

MARGIN = 0.08
CELL_INNER = 0.50

OUT_CSV = Path("./de2000_subset.csv")
DEBUG_DIR = Path("./debug_ref")
DEBUG_DIR.mkdir(exist_ok=True, parents=True)

# =============================
# 1) 基础工具
# =============================

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # tl
    rect[2] = pts[np.argmax(s)]      # br
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]   # tr
    rect[3] = pts[np.argmax(diff)]   # bl
    return rect

def warp_to_canvas(img_bgr: np.ndarray, quad: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img_bgr, M, (out_w, out_h))

def bgr_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    return np.stack([L, a, b], axis=-1)

# =============================
# 2) ΔE2000
# =============================

def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.sqrt(a1 * a1 + b1 * b1)
    C2 = np.sqrt(a2 * a2 + b2 * b2)
    avg_C = (C1 + C2) / 2.0

    G = 0.5 * (1 - np.sqrt((avg_C**7) / (avg_C**7 + 25**7 + 1e-12)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p * a1p + b1 * b1)
    C2p = np.sqrt(a2p * a2p + b2 * b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dHp = 2 * np.sqrt(C1p * C2p + 1e-12) * np.sin(np.radians(dhp) / 2.0)

    avg_Lp = (L1 + L2) / 2.0
    avg_Cpp = (C1p + C2p) / 2.0

    hsum = h1p + h2p
    havg = np.where(np.abs(h1p - h2p) > 180, hsum + 360, hsum)
    avg_hp = (havg / 2.0) % 360

    T = (
        1
        - 0.17 * np.cos(np.radians(avg_hp - 30))
        + 0.24 * np.cos(np.radians(2 * avg_hp))
        + 0.32 * np.cos(np.radians(3 * avg_hp + 6))
        - 0.20 * np.cos(np.radians(4 * avg_hp - 63))
    )

    d_ro = 30 * np.exp(-((avg_hp - 275) / 25) ** 2)
    R_C = 2 * np.sqrt((avg_Cpp**7) / (avg_Cpp**7 + 25**7 + 1e-12))

    S_L = 1 + (0.015 * (avg_Lp - 50) ** 2) / np.sqrt(20 + (avg_Lp - 50) ** 2)
    S_C = 1 + 0.045 * avg_Cpp
    S_H = 1 + 0.015 * avg_Cpp * T

    R_T = -np.sin(np.radians(2 * d_ro)) * R_C

    dE = np.sqrt(
        (dLp / S_L) ** 2
        + (dCp / S_C) ** 2
        + (dHp / S_H) ** 2
        + R_T * (dCp / S_C) * (dHp / S_H)
    )
    return dE

# =============================
# 3) 采样色块（6x7）
# =============================

def sample_grid_means(warped_bgr: np.ndarray, rows: int, cols: int, margin: float, cell_inner: float) -> np.ndarray:
    h, w = warped_bgr.shape[:2]
    x0, y0 = int(w * margin), int(h * margin)
    x1, y1 = int(w * (1 - margin)), int(h * (1 - margin))
    roi = warped_bgr[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    cell_w = rw / cols
    cell_h = rh / rows

    lab = bgr_to_lab(roi)
    feats = []

    for r in range(rows):
        for c in range(cols):
            cx0 = int(c * cell_w)
            cy0 = int(r * cell_h)
            cx1 = int((c + 1) * cell_w)
            cy1 = int((r + 1) * cell_h)

            iw = int((cx1 - cx0) * cell_inner)
            ih = int((cy1 - cy0) * cell_inner)
            ix0 = cx0 + ((cx1 - cx0) - iw) // 2
            iy0 = cy0 + ((cy1 - cy0) - ih) // 2
            ix1 = ix0 + iw
            iy1 = iy0 + ih

            patch = lab[iy0:iy1, ix0:ix1, :]
            feats.append(patch.reshape(-1, 3).mean(axis=0))

    return np.array(feats, dtype=np.float32)

# =============================
# 4) 关键：用“连通块 mask”找色卡外接矩形
# =============================

def find_quad_by_mask(img_bgr: np.ndarray, debug_prefix: str = None):
    """
    思路：
    1) 中心裁剪减少背景
    2) Canny 得到碎边缘
    3) 大尺度 dilate + close，把色卡区域连接成一个大连通块
    4) 选中心附近、面积最大的连通块
    5) minAreaRect -> quad
    """
    H, W = img_bgr.shape[:2]

    # 中心裁剪（你的色卡位置很稳定，居中）
    x0 = int(W * 0.10)
    x1 = int(W * 0.92)
    y0 = int(H * 0.06)
    y1 = int(H * 0.92)
    crop = img_bgr[y0:y1, x0:x1].copy()

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 20, 100)

    # ★关键：用大 kernel 把碎边缘连起来
    dil = cv2.dilate(edges, np.ones((15, 15), np.uint8), iterations=2)
    mask = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), iterations=2)

    # 找连通块轮廓
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        if debug_prefix:
            cv2.imwrite(str(DEBUG_DIR / f"{debug_prefix}_crop.png"), crop)
            cv2.imwrite(str(DEBUG_DIR / f"{debug_prefix}_edges.png"), edges)
            cv2.imwrite(str(DEBUG_DIR / f"{debug_prefix}_mask.png"), mask)
        return None

    # 选“中心附近 + 面积大”的轮廓（避免砂石/边缘干扰）
    ch, cw = mask.shape[:2]
    center = np.array([cw / 2.0, ch / 2.0], dtype=np.float32)

    best = None
    best_score = -1e18

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.03 * (ch * cw):
            continue

        x, y, wbox, hbox = cv2.boundingRect(c)
        ar = wbox / (hbox + 1e-6)
        if ar < 0.6 or ar > 2.2:  # 色卡大致接近横向矩形
            continue

        cx = x + wbox / 2.0
        cy = y + hbox / 2.0
        dist = np.linalg.norm(np.array([cx, cy]) - center)

        # score：面积大、且越靠近中心越好
        score = area - 2000.0 * dist
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        if debug_prefix:
            cv2.imwrite(str(DEBUG_DIR / f"{debug_prefix}_crop.png"), crop)
            cv2.imwrite(str(DEBUG_DIR / f"{debug_prefix}_edges.png"), edges)
            cv2.imwrite(str(DEBUG_DIR / f"{debug_prefix}_mask.png"), mask)
        return None

    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect).astype(np.float32)  # crop coords

    # 映射回原图坐标
    box[:, 0] += x0
    box[:, 1] += y0
    quad = order_points(box)

    if debug_prefix:
        vis = img_bgr.copy()
        p = quad.astype(np.int32)
        cv2.polylines(vis, [p], True, (0, 255, 0), 2)
        cv2.imwrite(str(DEBUG_DIR / f"{debug_prefix}_vis.png"), vis)
        cv2.imwrite(str(DEBUG_DIR / f"{debug_prefix}_crop.png"), crop)
        cv2.imwrite(str(DEBUG_DIR / f"{debug_prefix}_edges.png"), edges)
        cv2.imwrite(str(DEBUG_DIR / f"{debug_prefix}_mask.png"), mask)

    return quad

# =============================
# 5) 主流程
# =============================

def main():
    if not REF_PATH.exists():
        raise FileNotFoundError(f"Reference not found: {REF_PATH}")

    ref_img = cv2.imread(str(REF_PATH))
    if ref_img is None:
        raise RuntimeError(f"Failed to read reference: {REF_PATH}")

    quad_r = find_quad_by_mask(ref_img, debug_prefix="ref")
    if quad_r is None:
        raise RuntimeError("Failed to detect quad on reference. Check ./debug_ref/ref_mask.png and ref_vis.png")

    ref_warp = warp_to_canvas(ref_img, quad_r, WARP_W, WARP_H)
    ref_feat = sample_grid_means(ref_warp, ROWS, COLS, MARGIN, CELL_INNER)

    out_rows = []

    for model_name, fake_dir in MODELS:
        if not fake_dir.exists():
            print(f"[WARN] fake_dir not found: {fake_dir}")
            continue

        per_img = []

        for a_name in SELECT_A:
            fake_path = fake_dir / a_name
            if not fake_path.exists():
                print(f"[WARN] [{model_name}] missing fake: {fake_path}")
                continue

            fake_img = cv2.imread(str(fake_path))
            if fake_img is None:
                print(f"[WARN] [{model_name}] failed read: {fake_path}")
                continue

            quad_f = find_quad_by_mask(fake_img, debug_prefix=None)
            if quad_f is None:
                print(f"[WARN] [{model_name}] failed quad: {a_name}")
                continue

            fake_warp = warp_to_canvas(fake_img, quad_f, WARP_W, WARP_H)
            fake_feat = sample_grid_means(fake_warp, ROWS, COLS, MARGIN, CELL_INNER)

            de = float(ciede2000(fake_feat, ref_feat).mean())
            per_img.append(de)

            out_rows.append({
                "model": model_name,
                "image": a_name,
                "ref": REF_PATH.name,
                "de2000": de
            })

        if len(per_img) > 0:
            per_img = np.array(per_img, dtype=np.float32)
            print("========================================")
            print(f"[{model_name}] evaluated {len(per_img)}/{len(SELECT_A)}")
            print(f"[{model_name}] Mean ΔE2000 = {per_img.mean():.4f} ± {per_img.std():.4f}")
            print("========================================")
        else:
            print(f"[{model_name}] No valid samples (missing images or quad detection failed).")

    df = pd.DataFrame(out_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[SAVED] {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
