import cv2

# =========================
# 1. 读取原始 input.png
# =========================
img = cv2.imread("input2.png")
if img is None:
    raise FileNotFoundError("Cannot load input2.png")

# =========================
# 2. resize 到 320 × 320
# =========================
img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)

h, w, _ = img.shape
assert h == 320 and w == 320

# =========================
# 3. 画网格（用于 ROI 定位）
# =========================
step = 40  # 每 40 像素一条线（你可以改成 20 更细）

for x in range(0, w, step):
    cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)

for y in range(0, h, step):
    cv2.line(img, (0, y), (w, y), (255, 255, 255), 1)

# =========================
# 4. 保存结果
# =========================
out_path = "input_320_with_grid_2.png"
cv2.imwrite(out_path, img)

print(f"[DONE] Saved resized + grid image to {out_path}")


