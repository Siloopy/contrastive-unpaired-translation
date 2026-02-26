import cv2
import numpy as np
import os

def load_and_resize_to_320(img, target_size=320):
    """
    将图像 resize 到 target_size x target_size
    """
    return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def mean_rgb(image_path, roi):
    """
    image_path: 图像路径
    roi: (x, y, w, h)
    return: (R_mean, G_mean, B_mean)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # =========================
    # 如果是 input.png，先 resize 到 320x320
    # =========================
    if os.path.basename(image_path) == "input.png":
        img = load_and_resize_to_320(img, target_size=320)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x, y, w, h = roi
    patch = img[y:y+h, x:x+w]

    mean = patch.mean(axis=(0, 1))
    return mean


# =========================
# 1. 图像路径（相对于 color_analysis 目录）
# =========================
images = {
    "Input (A)": "input.png",
    "CUT": "cut.png",
    "CUT + VEF": "cut_vef.png",
    "CUT + VEF (Stable)": "cut_vef_stable.png"
}

# =========================
# 2. ROI 定义（不做任何修改）
# 格式：(x, y, width, height)
# =========================
roi_blue   = (120, 60, 24, 24)
roi_yellow = (40, 200, 24, 24)
roi_gray   = (235, 200, 24, 24)


rois = {
    "Blue patch": roi_blue,
    "Yellow patch": roi_yellow,
    "Gray patch": roi_gray
}


# =========================
# 3. 计算并打印结果
# =========================
for roi_name, roi in rois.items():
    print(f"\n=== {roi_name} ===")
    for method, path in images.items():
        r, g, b = mean_rgb(path, roi)
        print(f"{method:20s} | R={r:6.1f}  G={g:6.1f}  B={b:6.1f}")
