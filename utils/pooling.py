import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_length_feature(real_len: int, max_expected_len: float = 2000.0) -> float:
    return float(math.log(float(real_len) + 1.0) / math.log(max_expected_len + 1.0))


def masked_max_pool1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_2d = mask.unsqueeze(1).bool()
    min_val = torch.finfo(x.dtype).min
    x_masked = x.masked_fill(~mask_2d, min_val)
    return torch.max(x_masked, dim=-1)[0]


def masked_avg_pool1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_2d = mask.unsqueeze(1).to(dtype=x.dtype)
    lengths = torch.sum(mask_2d, dim=-1).clamp(min=1.0)
    return torch.sum(x * mask_2d, dim=-1) / lengths


class MaskedPowerAveragePooling1d(nn.Module):
    def __init__(self, init_p: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(init_p)))
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        p_clamped = self.p.float().clamp(min=1.0, max=10.0)
        x_clamped = torch.clamp(x_fp32, min=self.eps)
        x_pow = torch.pow(x_clamped, p_clamped)

        mask_2d = mask.unsqueeze(1).float()
        lengths = torch.sum(mask_2d, dim=-1).clamp(min=1.0)
        avg_pow = torch.sum(x_pow * mask_2d, dim=-1) / lengths

        res = torch.pow(torch.clamp(avg_pow, min=self.eps), 1.0 / p_clamped)
        return res.to(orig_dtype)