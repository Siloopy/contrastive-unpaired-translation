import re
import matplotlib.pyplot as plt

log_path = "checkpoints/underwater_cut_baseline/loss_log.txt"

epochs = []
g_gan_list = []
nce_list = []

with open(log_path, "r") as f:
    for line in f:
        if "G_GAN" in line and "NCE:" in line:
            epoch_match = re.search(r"epoch:\s*(\d+)", line)
            g_gan_match = re.search(r"G_GAN:\s*([0-9.]+)", line)
            nce_match = re.search(r"NCE:\s*([0-9.]+)", line)

            if epoch_match and g_gan_match and nce_match:
                epochs.append(int(epoch_match.group(1)))
                g_gan_list.append(float(g_gan_match.group(1)))
                nce_list.append(float(nce_match.group(1)))

print(f"Parsed {len(epochs)} points")

plt.figure(figsize=(8, 5))
plt.plot(epochs, g_gan_list, label="G_GAN Loss")
plt.plot(epochs, nce_list, label="PatchNCE Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CUT Training Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("cut_loss_curve.png", dpi=300)
print("Saved figure as cut_loss_curve.png")
