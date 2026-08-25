"""
GenExplorer: Hybrid Multi-Branch Neural Architecture for DNA Sequence Classification.
Combines multi-scale convolutions (Pattern & Frequency branches) with a self-attention Transformer branch[cite: 1, 3].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatternBranch(nn.Module):
    """Multi-scale 1D Convolutional branch with Adaptive Max Pooling for local motif discovery[cite: 1, 3]."""

    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.conv6 = nn.Conv1d(4, 512, kernel_size=6, padding="same")
        self.conv12 = nn.Conv1d(4, 512, kernel_size=12, padding="same")
        self.conv18 = nn.Conv1d(4, 512, kernel_size=18, padding="same")
        self.conv24 = nn.Conv1d(4, 512, kernel_size=24, padding="same")
        self.bn = nn.BatchNorm1d(2048)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, Length, 4) -> Permute to (Batch, 4, Length)
        x = x.permute(0, 2, 1)
        c = torch.cat([
            F.gelu(self.conv6(x)),
            F.gelu(self.conv12(x)),
            F.gelu(self.conv18(x)),
            F.gelu(self.conv24(x)),
        ], dim=1)
        pooled = self.pool(c).squeeze(-1)
        return self.fc(self.bn(pooled))


class FrequencyBranch(nn.Module):
    """Multi-scale 1D Convolutional branch with Adaptive Average Pooling for global sequence/k-mer composition[cite: 1, 3]."""

    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.conv6 = nn.Conv1d(4, 256, kernel_size=6, padding="same")
        self.conv12 = nn.Conv1d(4, 256, kernel_size=12, padding="same")
        self.conv18 = nn.Conv1d(4, 256, kernel_size=18, padding="same")
        self.conv24 = nn.Conv1d(4, 256, kernel_size=24, padding="same")
        self.bn = nn.BatchNorm1d(1024)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        c = torch.cat([
            F.gelu(self.conv6(x)),
            F.gelu(self.conv12(x)),
            F.gelu(self.conv18(x)),
            F.gelu(self.conv24(x)),
        ], dim=1)
        pooled = self.pool(c).squeeze(-1)
        return self.fc(self.bn(pooled))


class TransformerBranch(nn.Module):
    """Transformer Encoder branch to model non-local, long-range contextual dependencies[cite: 1, 3]."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Linear(4, d_model)
        self.pos_enc = nn.Embedding(max_len + 1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        positions = torch.arange(1, l + 1, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_enc(positions)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        h = torch.cat([cls_tokens, h], dim=1)
        h_enc = self.transformer(h)
        cls_out = self.norm(h_enc[:, 0, :])
        return self.out(cls_out)


class GenExplorer(nn.Module):
    """Full GenExplorer architecture fusing Pattern, Frequency, and Transformer representations[cite: 1, 3]."""

    def __init__(
        self,
        cnn_drop: float = 0.25,
        trans_drop: float = 0.1,
        out_features: int = 1,
        max_len: int = 512
    ):
        super().__init__()
        self.pattern = PatternBranch(dropout=cnn_drop)
        self.frequency = FrequencyBranch(dropout=cnn_drop)
        self.transformer = TransformerBranch(dropout=trans_drop, max_len=max_len)
        self.fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.GELU(),
            nn.Dropout(cnn_drop),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(cnn_drop),
            nn.Linear(128, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_pat = self.pattern(x)
        h_freq = self.frequency(x)
        h_trans = self.transformer(x)
        h_fused = torch.cat([h_pat, h_freq, h_trans], dim=1)
        return self.fc(h_fused)