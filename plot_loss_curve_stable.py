import re
import matplotlib.pyplot as plt

# ====== 1. 配置路径 ======
LOG_PATH = "checkpoints/underwater_cut_vef_stable/loss_log.txt"
OUT_PATH = "loss_curve_stable.png"

# ====== 2. 正则：匹配 CUT 打印的 loss 行 ======
pattern = re.compile(
    r"\(epoch:\s*(\d+),.*?"
    r"G_GAN:\s*([\d\.]+).*?"
    r"G:\s*([\d\.]+).*?"
    r"NCE:\s*([\d\.]+)"
)

epochs = []
g_gan = []
g_total = []
nce = []

# ====== 3. 解析日志 ======
with open(LOG_PATH, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            g_gan.append(float(m.group(2)))
            g_total.append(float(m.group(3)))
            nce.append(float(m.group(4)))

assert len(epochs) > 0, "❌ 没有解析到任何 loss，请检查日志路径或格式"

print(f"✅ Parsed {len(epochs)} loss points")

# ====== 4. 画图 ======
plt.figure(figsize=(8, 5))

plt.plot(epochs, g_gan, label="G_GAN", linewidth=1.2)
plt.plot(epochs, nce, label="PatchNCE", linewidth=1.2)
plt.plot(epochs, g_total, label="G total", linewidth=1.5)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CUT + VEF (Stable) Training Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()

# ====== 5. 保存（不 show，避免后端问题） ======
plt.savefig(OUT_PATH, dpi=300)
plt.close()

print(f"✅ Loss curve saved to: {OUT_PATH}")
