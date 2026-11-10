import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class HelixPR(nn.Module):
    """
    HelixPR (Shi et al.): helix-aware periodic representation CNN.
    Input: one-hot encoding plus a sinusoidal positional term with the helical
    period (~10 bp) and a global composition (CLS) token; 4-layer 1D CNN.
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 1,
        hidden_channels: tuple = (128, 256, 128, 64),
        kernel_sizes: tuple = (3, 9, 4, 4),
        use_length_feature: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_length_feature = use_length_feature
        h1, h2, h3, h4 = hidden_channels
        k1, k2, k3, k4 = kernel_sizes

        self.conv = nn.Sequential(
            Block(in_channels, h1, k1),
            Block(h1, h2, k2),
            Block(h2, h3, k3),
            Block(h3, h4, k4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        fc_in = h4 + (1 if use_length_feature else 0)
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, h4 // 2),
            nn.ReLU(),
            nn.Linear(h4 // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        # (B, L, C) -> (B, C, L) if needed
        if x.dim() == 3 and x.shape[-1] in (4, 5):
            x = x.permute(0, 2, 1)

        c = self.conv(x)

        if mask is None or not self.use_length_feature:
            pooled = self.pool(c).squeeze(-1)
        else:
            m = mask.unsqueeze(1).float()
            real_l = m.sum(dim=-1).clamp(min=1.0)
            pooled = torch.sum(c * m, dim=-1) / real_l

        if self.use_length_feature:
            if lengths is None and mask is not None:
                lengths = mask.sum(dim=-1).float()
            if lengths is not None:
                log_l = torch.log(lengths.unsqueeze(1).clamp(min=1.0) + 1.0)
            else:
                log_l = torch.zeros((x.shape[0], 1), dtype=pooled.dtype, device=pooled.device)
            pooled = torch.cat([pooled, log_l], dim=1)

        out = self.classifier(pooled)
        return out.squeeze(-1) if self.num_classes == 1 else out

