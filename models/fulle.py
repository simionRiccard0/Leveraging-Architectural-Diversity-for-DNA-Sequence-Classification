"""
FullE: 10-Subnetwork Convolutional Neural Network Ensemble with Parameterized Lp Pooling.
"""

from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class PowerAveragePooling1d(nn.Module):
    """L_p Power-Average Pooling smoothly interpolating between Mean (p=1) and Max Pooling (p -> inf)."""

    def __init__(self, init_p: float = 2.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(init_p)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(self.p, min=1.0, max=15.0)
        x_clamped = torch.clamp(x, min=1e-6)
        x_pow = torch.pow(x_clamped, p)
        avg_pow = torch.mean(x_pow, dim=-1)
        return torch.pow(avg_pow, 1.0 / p)


class FullEBranch(nn.Module):
    """Single Convolutional Sub-Network Branch (Frequency, Pattern, or Lp-pooling)."""

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1200,
        kernel_size: int = 11,
        pool_type: str = "max",
        dropout: float = 0.2,
        num_classes: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same")
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool_type = pool_type.lower()
        self.num_classes = num_classes

        if self.pool_type == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif self.pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif self.pool_type == "lp":
            self.pool = PowerAveragePooling1d(init_p=2.0)
        else:
            raise ValueError(f"Unsupported pooling type: {pool_type}")

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_channels, 1000)
        self.classifier = nn.Linear(1000, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[-1] in (4, 5):
            x = x.permute(0, 2, 1)
        h = self.relu(self.conv(x))
        h = self.bn(h)
        if self.pool_type == "lp":
            h = self.pool(h)
        else:
            h = self.pool(h).squeeze(-1)
        h = self.dropout(h)
        return self.relu(self.fc(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        out = self.classifier(feat)
        return out.squeeze(-1) if self.num_classes == 1 else out


class FullEMerger(nn.Module):
    """Multi-Branch Merger concatenating feature outputs from distinct branches."""

    def __init__(self, branches: List[FullEBranch], dropout: float = 0.2, num_classes: int = 1):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.num_classes = num_classes
        total_dim = 1000 * len(branches)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_dim, 500),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(500, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b.extract_features(x) for b in self.branches]
        h_fused = torch.cat(feats, dim=1)
        out = self.classifier(h_fused)
        return out.squeeze(-1) if self.num_classes == 1 else out


class FullE10Ensemble(nn.Module):
    """
    Complete FullE Ensemble Framework combining 10 heterogeneous subnetworks via Sum-Rule (Eq. 3 in Paper).
    """

    def __init__(self, branches: Dict[str, FullEBranch], mergers: Dict[str, FullEMerger], num_classes: int = 1):
        super().__init__()
        self.branches = nn.ModuleDict(branches)
        self.mergers = nn.ModuleDict(mergers)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = []
        is_ce = (self.num_classes == 2)
        all_models = list(self.branches.values()) + list(self.mergers.values())

        for m in all_models:
            out = m(x)
            p = F.softmax(out, dim=1)[:, 1] if is_ce else torch.sigmoid(out)
            probs.append(p)

        # Unweighted average decision fusion (Sum-Rule: 1/10 sum P_m)
        avg_prob = torch.stack(probs, dim=0).mean(dim=0)
        eps = 1e-7
        avg_prob = torch.clamp(avg_prob, min=eps, max=1.0 - eps)
        # Convert to logit space for metric consistency
        logit = torch.log(avg_prob / (1.0 - avg_prob))
        return logit