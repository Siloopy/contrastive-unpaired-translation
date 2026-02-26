import torch
import torch.nn as nn
import torch.nn.functional as F


class VEF(nn.Module):
    """
    Paper-ready Minimal VEF (Level-2, stability-first):

    1) Bounded channel-wise compensation (physical prior):
       - Learnable per-channel gain alpha in a reasonable range.
       - Avoids exploding/vanishing enhancement during GAN training.

    2) Monotonic visibility enhancement:
       - Use a smooth monotonic nonlinearity to boost mid/low intensities
         without introducing unstable global normalization.

    3) No batch/global max normalization:
       - Removes cross-image scaling jitter that often destabilizes GAN losses.

    Input/Output:
        x:  (B, 3, H, W), assumed in [-1, 1]
        out:(B, 3, H, W), returned in [-1, 1]
    """

    def __init__(
        self,
        alpha_min: float = 0.5,
        alpha_max: float = 2.0,
        gamma: float = 1.0,
        boost: float = 2.0,
        eps: float = 1e-6
    ):
        super().__init__()

        # ---- 1) bounded alpha (3 params) ----
        # raw -> sigmoid -> [alpha_min, alpha_max]
        self.alpha_raw = nn.Parameter(torch.zeros(3))
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

        # ---- 2) monotonic visibility enhancement ----
        # boost controls enhancement strength; gamma controls curve shape.
        self.gamma = float(gamma)
        self.boost = float(boost)
        self.eps = float(eps)

    def _alpha(self):
        # alpha in [alpha_min, alpha_max]
        s = torch.sigmoid(self.alpha_raw)  # (3,)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * s  # (3,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"VEF expects x shape (B,3,H,W), got {tuple(x.shape)}")

        # [-1,1] -> [0,1]
        x01 = (x + 1.0) * 0.5
        x01 = x01.clamp(0.0, 1.0)

        # 1) channel-wise compensation (physical-ish prior)
        alpha = self._alpha().view(1, 3, 1, 1)
        x_c = x01 * alpha
        x_c = x_c.clamp(0.0, 1.0)

        # 2) visibility enhancement (stable & monotonic)
        # Use a smooth curve: y = (1 + boost * x)^gamma, then rescale back to [0,1]
        # We avoid per-batch max normalization; instead use a deterministic denominator.
        y = (1.0 + self.boost * x_c).clamp(min=1.0)  # >= 1
        y = y.pow(self.gamma)

        # Normalize to [0,1] with known bounds:
        # when x_c in [0,1], y in [1, (1+boost)^gamma]
        denom = (1.0 + self.boost) ** self.gamma
        x_v = (y - 1.0) / (denom - 1.0 + self.eps)
        x_v = x_v.clamp(0.0, 1.0)

        # [0,1] -> [-1,1]
        out = x_v * 2.0 - 1.0
        return out
