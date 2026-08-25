"""
FullE: Multi-Branch Convolutional Neural Network Ensemble with Parameterized Lp Pooling[cite: 1, 5].
"""

from typing import List
import torch
import torch.nn as nn


class PowerAveragePooling1d(nn.Module):
    """L_p Power-Average Pooling smoothly interpolating between Mean Pooling (p=1) and Max Pooling (p -> inf)[cite: 1, 5]."""

    def __init__(self, init_p: float = 2.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(init_p)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected shape: (Batch, Channels, Length)
        p = torch.clamp(self.p, min=1.0, max=15.0)
        x_clamped = torch.clamp(x, min=1e-6)
        x_pow = torch.pow(x_clamped, p)
        avg_pow = torch.mean(x_pow, dim=-1)
        return torch.pow(avg_pow, 1.0 / p)


class FullEBranch(nn.Module):
    """Single Convolutional Sub-Network Branch (Frequency, Pattern, or Lp-pooling)[cite: 1, 5]."""

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1200,
        kernel_size: int = 11,
        pool_type: str = "max",
        dropout: float = 0.2
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same")
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool_type = pool_type.lower()
        
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
        self.classifier = nn.Linear(1000, 1)

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
        return self.classifier(feat)


class FullEMerger(nn.Module):
    """Multi-Branch Merger: Concatenates extracted feature embeddings from pre-trained branches[cite: 1, 5]."""

    def __init__(self, branches: List[FullEBranch], dropout: float = 0.2):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        total_dim = 1000 * len(branches)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_dim, 500),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(500, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b.extract_features(x) for b in self.branches]
        h_fused = torch.cat(feats, dim=1)
        return self.classifier(h_fused)


class FullEEnsemble(nn.Module):
    """FullE 10-Subnetwork Decision Fusion Ensemble using unweighted sum rule[cite: 1, 5]."""

    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = [torch.sigmoid(m(x)) for m in self.models]
        return torch.stack(probs, dim=0).mean(dim=0)