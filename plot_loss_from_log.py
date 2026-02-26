import re
import matplotlib.pyplot as plt

log_path = "checkpoints/underwater_cut_vef_l2_reg/loss_log.txt"

epochs = []
g_gan = []
nce = []
g_total = []

current_epoch = None

with open(log_path, "r") as f:
    for line in f:
        # ① 先读 epoch 行
        if "(epoch:" in line:
            m = re.search(r"epoch:\s*(\d+)", line)
            if m:
                current_epoch = int(m.group(1))

        # ② 再读 loss 行（下一行）
        if "G_GAN:" in line and current_epoch is not None:
            ggan = float(re.search(r"G_GAN:\s*([0-9.]+)", line).group(1))
            nce_loss = float(re.search(r"NCE:\s*([0-9.]+)", line).group(1))
            g = float(re.search(r"G:\s*([0-9.]+)", line).group(1))

            epochs.append(current_epoch)
            g_gan.append(ggan)
            nce.append(nce_loss)
            g_total.append(g)

# 🔍 防御性检查
print(f"Parsed {len(epochs)} points")
assert len(epochs) > 0, "❌ 没有解析到任何 loss，请检查日志格式"

# ===== 画图 =====
plt.figure(figsize=(8, 5))
plt.plot(epochs, g_gan, label="G_GAN")
plt.plot(epochs, nce, label="PatchNCE")
plt.plot(epochs, g_total, label="G total")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CUT + VEF Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("loss_cut_vef.png", dpi=300)
plt.show()

print("✅ Saved loss_cut_vef.png")
