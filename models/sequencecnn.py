import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pooling import masked_max_pool1d, masked_avg_pool1d


class SequenceCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        conv1_filters: int = 128,
        conv1_kernel: int = 13,
        conv2_filters: int = 128,
        conv2_kernel: int = 7,
        dropout: float = 0.2,
        use_length_feature: bool = False,
    ):
        super().__init__()
        self.use_length_feature = use_length_feature

        self.conv1 = nn.Conv1d(in_channels, conv1_filters, kernel_size=conv1_kernel, padding="same")
        self.bn1 = nn.BatchNorm1d(conv1_filters)
        self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=conv2_kernel, padding="same")
        self.bn2 = nn.BatchNorm1d(conv2_filters)
        self.dropout = nn.Dropout(dropout)

        fc_in = conv2_filters * 2 + (1 if use_length_feature else 0)
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        if x.dim() == 3 and x.shape[-1] in (4, 5):
            x = x.permute(0, 2, 1)

        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))

        if mask is None:
            max_pool = F.adaptive_max_pool1d(h, 1).squeeze(-1)
            avg_pool = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        else:
            max_pool = masked_max_pool1d(h, mask)
            avg_pool = masked_avg_pool1d(h, mask)

        pooled = torch.cat([max_pool, avg_pool], dim=1)
        pooled = self.dropout(pooled)

        if self.use_length_feature:
            if lengths is None and mask is not None:
                lengths = mask.sum(dim=-1).float()
            if lengths is not None:
                log_l = torch.log(lengths.unsqueeze(1).clamp(min=1.0) + 1.0)
            else:
                log_l = torch.zeros((x.shape[0], 1), dtype=pooled.dtype, device=pooled.device)
            pooled = torch.cat([pooled, log_l], dim=1)

        return self.classifier(pooled)