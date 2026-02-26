import os
from PIL import Image

# 配置路径 
src_dir = "picture"                 # JPG 文件夹
dst_dir = "selfdataset/trainA"      # 输出 PNG 文件夹

os.makedirs(dst_dir, exist_ok=True)

valid_ext = (".jpg", ".jpeg", ".JPG", ".JPEG")

count = 0
for fname in sorted(os.listdir(src_dir)):
    if fname.endswith(valid_ext):
        src_path = os.path.join(src_dir, fname)
        dst_name = os.path.splitext(fname)[0] + ".png"
        dst_path = os.path.join(dst_dir, dst_name)

        img = Image.open(src_path).convert("RGB")
        img.save(dst_path, format="PNG")

        count += 1

print(f"✔ Converted {count} images to PNG, saved in {dst_dir}")
