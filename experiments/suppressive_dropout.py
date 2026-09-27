import torch
import torch.nn as nn

class SuppressiveDropout(nn.Module):
    """
    Channel-wise Suppressive Dropout for CNNs (N,C,H,W).
    Paper formula:
      S_j ∝ (sum_{k≠j} x_k) * x_j^2 / (1 + b * sum_i x_i^2)^{c+1}
    where x_j is the spatially averaged activation of channel j.
    Zeroes the drop_ratio fraction of channels with the largest S (top-k per sample).
    """
    def __init__(self, drop_ratio=0.2, b=1.0, c=1.0, eps=1e-8):
        super().__init__()
        self.drop_ratio = float(drop_ratio)
        self.b = float(b)
        self.c = float(c)
        self.eps = eps

    def forward(self, x):
        if (not self.training) or self.drop_ratio <= 0.0:
            return x
        assert x.dim() == 4, "Expect N,C,H,W"
        N, C, H, W = x.shape

        # channel mean activation: x̄_j
        xm = x.mean(dim=(2, 3))                    # (N, C)
        x2_sum = (xm ** 2).sum(dim=1, keepdim=True)  # (N,1)
        sum_all = xm.sum(dim=1, keepdim=True)        # (N,1)
        neighbor_sum = sum_all - xm                  # (N,C)

        denom = (1.0 + self.b * x2_sum).pow(self.c + 1.0)  # (N,1)
        S = neighbor_sum * (xm ** 2) / (denom + self.eps)  # (N,C)

        k = max(1, int(round(self.drop_ratio * C)))
        _, idx = torch.topk(S, k=k, dim=1, largest=True, sorted=False)  # (N,k)

        mask = torch.ones((N, C), device=x.device, dtype=x.dtype)
        mask.scatter_(1, idx, 0.0)  # zero the top-S channels
        mask = mask.view(N, C, 1, 1)

        return x * mask
