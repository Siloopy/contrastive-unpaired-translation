import os
import shutil
import tempfile
from glob import glob

import cv2
import numpy as np
import torch
from pytorch_fid import fid_score


# ==============================
# 1. 真实GT路径
# ==============================
GT_DIR = "./dataset/datasets/underwater/testB"

# ==============================
# 2. 你的模型输出路径
# ==============================
MODEL_DIRS = {
    "CUT_baseline_seed0": "./results/underwater_cut_baseline_seed0/test_latest/images/fake_B",
    "CUT_baseline_seed1": "./results/underwater_cut_baseline_seed1/test_latest/images/fake_B",
    "CUT_baseline_seed2": "./results/underwater_cut_baseline_seed2/test_latest/images/fake_B",

    "CUT_vef_seed0": "./results/underwater_cut_vef_seed0/test_latest/images/fake_B",
    "CUT_vef_seed1": "./results/underwater_cut_vef_seed1/test_latest/images/fake_B",
    "CUT_vef_seed2": "./results/underwater_cut_vef_seed2/test_latest/images/fake_B",

    "CUT_vef_stable_seed0": "./results/underwater_cut_vef_stable_seed0/test_latest/images/fake_B",
    "CUT_vef_stable_seed1": "./results/underwater_cut_vef_stable_seed1/test_latest/images/fake_B",
    "CUT_vef_stable_seed2": "./results/underwater_cut_vef_stable_seed2/test_latest/images/fake_B",
}


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def list_images(folder):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)


def check_directory(path):
    if not os.path.exists(path):
        raise ValueError(f"Path not found: {path}")
    files = list_images(path)
    if len(files) == 0:
        raise ValueError(f"No images found in: {path}")
    return len(files)


def scan_sizes(folder, max_show=10):
    """返回 size->count，并打印部分异常尺寸"""
    files = list_images(folder)
    size_count = {}
    bad = 0
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            bad += 1
            continue
        h, w = img.shape[:2]
        size_count[(h, w)] = size_count.get((h, w), 0) + 1

    if bad > 0:
        print(f"⚠ Warning: {bad} unreadable images in {folder}")

    if len(size_count) > 1:
        print(f"⚠ Mixed image sizes in {folder}: {len(size_count)} kinds")
        # show some sizes
        for i, (k, v) in enumerate(sorted(size_count.items(), key=lambda x: -x[1])):
            if i >= max_show:
                break
            print(f"  - {k[0]}x{k[1]}: {v}")
    return size_count


def make_resized_copy(src_dir, dst_dir, target_hw=(256, 256)):
    os.makedirs(dst_dir, exist_ok=True)
    files = list_images(src_dir)
    keep = 0
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            continue
        # resize (w,h) for cv2
        img_r = cv2.resize(img, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
        out = os.path.join(dst_dir, os.path.basename(f))
        # ensure png to avoid jpeg artifacts (optional)
        cv2.imwrite(out, img_r)
        keep += 1
    if keep == 0:
        raise RuntimeError(f"No valid images after resize copy from {src_dir}")
    return keep


def compute_fid(fake_dir, real_dir, batch_size=32, dims=2048):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=batch_size,
        device=device,
        dims=dims
    )
    return float(fid_value)


if __name__ == "__main__":

    print("=" * 70)
    print("Underwater GAN FID Evaluation (Robust Resize Version)")
    print("=" * 70)

    # 检查GT
    gt_num = check_directory(GT_DIR)
    print(f"\nGT directory: {GT_DIR}")
    print(f"GT image count: {gt_num}")
    scan_sizes(GT_DIR)

    results_summary = {}

    # 临时目录：自动清理
    with tempfile.TemporaryDirectory() as tmp_root:
        real_tmp = os.path.join(tmp_root, "real_256")
        print(f"\n[INFO] Creating resized GT in: {real_tmp}")
        real_keep = make_resized_copy(GT_DIR, real_tmp, target_hw=(256, 256))
        print(f"[INFO] Resized GT count: {real_keep}")

        for name, path in MODEL_DIRS.items():

            print("\n" + "-" * 60)
            print(f"Model: {name}")

            fake_num = check_directory(path)
            print(f"Fake path: {path}")
            print(f"Fake image count: {fake_num}")
            scan_sizes(path)

            fake_tmp = os.path.join(tmp_root, f"{name}_256")
            print(f"[INFO] Creating resized Fake in: {fake_tmp}")
            fake_keep = make_resized_copy(path, fake_tmp, target_hw=(256, 256))
            print(f"[INFO] Resized Fake count: {fake_keep}")

            if fake_keep != real_keep:
                print("⚠ Warning: Resized Fake and GT counts differ!")

            fid = compute_fid(fake_tmp, real_tmp, batch_size=32, dims=2048)
            print(f"FID = {fid:.4f}")

            model_type = name.split("_seed")[0]
            results_summary.setdefault(model_type, []).append(fid)

    # 统计 mean ± std
    print("\n" + "=" * 70)
    print("Final Summary (Mean ± Std over seeds)")
    print("=" * 70)

    for model_type, values in results_summary.items():
        mean = float(np.mean(values))
        std = float(np.std(values))
        print(f"{model_type}: {mean:.4f} ± {std:.4f}")
