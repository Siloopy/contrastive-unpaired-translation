import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# 修改路径
base_dir   = "./results/cut_base_e150_lr1e4_v1/test_latest/images/fake_B"
vef_dir    = "./results/cut_vef_e150_lr1e4_v1/test_latest/images/fake_B"
stable_dir = "./results/cut_e150_lr1e4_v1/test_latest/images/fake_B"
# UCIQE
# =========================
def compute_uciqe(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = img_lab[:,:,0]
    a = img_lab[:,:,1]
    b = img_lab[:,:,2]

    chroma = np.sqrt(a**2 + b**2)
    sigma_c = np.std(chroma)
    con_l = np.std(L)
    mu_s = np.mean(chroma)

    return 0.4680*sigma_c + 0.2745*con_l + 0.2576*mu_s

# UIQM
def compute_uiqm(img):
    img = img.astype(np.float32) / 255.0
    r, g, b = img[:,:,2], img[:,:,1], img[:,:,0]

    rg = r - g
    yb = 0.5*(r + g) - b

    uicm = -0.0268*np.mean(rg) + 0.1586*np.std(rg) \
           -0.0268*np.mean(yb) + 0.1586*np.std(yb)

    gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1)
    uism = np.mean(np.abs(sobel))

    uiconm = np.std(gray)

    return 0.0282*uicm + 0.2953*uism + 3.5753*uiconm

# 批量评估
def evaluate(folder):
    files = sorted(glob(os.path.join(folder, "*.png")))
    uciqe_list = []
    uiqm_list = []

    for f in tqdm(files):
        img = cv2.imread(f)
        img = cv2.resize(img, (320, 320))

        uciqe_list.append(compute_uciqe(img))
        uiqm_list.append(compute_uiqm(img))

    return (
        np.mean(uciqe_list), np.std(uciqe_list),
        np.mean(uiqm_list), np.std(uiqm_list)
    )

print("\nEvaluating UCIQE & UIQM\n")

for label, folder in [
    ("CUT baseline", base_dir),
    ("CUT + VEF", vef_dir),
    ("CUT + VEF (Stable)", stable_dir)
]:
    mu_u, std_u, mu_q, std_q = evaluate(folder)
    print(f"\n{label}")
    print(f"UCIQE = {mu_u:.4f} ± {std_u:.4f}")
    print(f"UIQM  = {mu_q:.4f} ± {std_q:.4f}")
