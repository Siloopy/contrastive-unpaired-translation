import os
import cv2
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Your folders
# ---------------------------
base_dir   = "./results/cut_base_e150_lr1e4_v1/test_latest/images/fake_B"
vef_dir    = "./results/cut_vef_e150_lr1e4_v1/test_latest/images/fake_B"
stable_dir = "./results/cut_e150_lr1e4_v1/test_latest/images/fake_B"

# ---------------------------
# Build NIQE function with fallback
# ---------------------------
def build_niqe_fn(device):
    """
    Returns a callable: score = niqe_fn(img_tensor)
    img_tensor: NCHW, RGB, float32 in [0,1]
    """
    # 1) Try PIQ (PyTorch Image Quality)
    try:
        import piq  # noqa
        # Some versions expose piq.niqe, some expose from piq import niqe
        if hasattr(piq, "niqe") and callable(getattr(piq, "niqe")):
            def niqe_fn(x):
                return piq.niqe(x)
            return niqe_fn, "piq.niqe"
        else:
            try:
                from piq import niqe as _niqe  # type: ignore
                def niqe_fn(x):
                    return _niqe(x)
                return niqe_fn, "from piq import niqe"
            except Exception:
                pass
    except Exception:
        pass

    # 2) Fallback: pyiqa
    try:
        import pyiqa  # type: ignore
        metric = pyiqa.create_metric("niqe", device=device)
        metric.eval()

        @torch.no_grad()
        def niqe_fn(x):
            # pyiqa returns a tensor
            return metric(x)
        return niqe_fn, "pyiqa.create_metric('niqe')"
    except Exception:
        pass

    # 3) Nothing available
    raise ImportError(
        "找不到可用的 NIQE 实现：\n"
        "1) 你当前的 piq 包没有 niqe（可能装错包/版本不对）。\n"
        "2) 也没有安装 pyiqa。\n\n"
        "建议二选一：\n"
        "A) pip install -U piq\n"
        "B) pip install -U pyiqa\n"
    )

niqe_fn, backend = build_niqe_fn(device)
print(f"[INFO] NIQE backend: {backend}")

# ---------------------------
# Compute NIQE for a folder
# ---------------------------
@torch.no_grad()
def compute_niqe_folder(folder, min_hw=192, resize_if_small=True):
    files = sorted(glob(os.path.join(folder, "*.png")))
    if len(files) == 0:
        raise FileNotFoundError(f"Folder has no .png files: {folder}")

    scores = []
    skipped = 0

    for f in tqdm(files, desc=os.path.basename(folder)):
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        # NIQE 对太小的图经常不稳定/直接报错；给一个稳妥处理
        if (h < min_hw or w < min_hw) and resize_if_small:
            scale = max(min_hw / h, min_hw / w)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        img = img.astype(np.float32) / 255.0  # [0,1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        score = niqe_fn(img_tensor)
        # score may be tensor on GPU
        score_val = float(score.detach().cpu().item())
        scores.append(score_val)

    scores = np.array(scores, dtype=np.float64)
    if scores.size == 0:
        raise RuntimeError(f"All images failed/empty in folder: {folder}")

    return float(scores.mean()), float(scores.std(ddof=0)), skipped, len(files)

# ---------------------------
# Run
# ---------------------------
for name, path in [
    ("CUT baseline", base_dir),
    ("CUT + VEF", vef_dir),
    ("CUT + VEF Stable", stable_dir)
]:
    mean, std, skipped, total = compute_niqe_folder(path)
    print(f"\n{name}")
    print(f"Path: {path}")
    print(f"NIQE = {mean:.4f} ± {std:.4f}")
    print(f"Processed: {total - skipped}/{total} (skipped {skipped})")
    print("-" * 40)
