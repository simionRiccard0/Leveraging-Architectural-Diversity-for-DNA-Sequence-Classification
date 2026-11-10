#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ablation variants of GenExplorer.
Supports dynamic branch selection:
  - 'F': Frequency branch only (Conv1D + AdaptiveAvgPool)
  - 'P': Pattern branch only (Conv1D + AdaptiveMaxPool)
  - 'FP': Frequency + Pattern branches (ViraMiner-like dual branch)
  - 'FPT': Full GenExplorer (Frequency + Pattern + Transformer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.genexplorer import PatternBranch, FrequencyBranch, TransformerBranch


class GenExplorerAblation(nn.Module):
    def __init__(
        self,
        branches: str = "FPT",
        cnn_drop: float = 0.25,
        trans_drop: float = 0.1,
        out_features: int = 1,
        max_len: int = 512
    ):
        super().__init__()
        self.branches = branches.upper()
        self.has_freq = "F" in self.branches
        self.has_pat = "P" in self.branches
        self.has_trans = "T" in self.branches

        if not (self.has_freq or self.has_pat or self.has_trans):
            raise ValueError(f"At least one branch must be active. Provided: {branches}")

        if self.has_freq:
            self.frequency = FrequencyBranch(dropout=cnn_drop)
        if self.has_pat:
            self.pattern = PatternBranch(dropout=cnn_drop)
        if self.has_trans:
            self.transformer = TransformerBranch(dropout=trans_drop, max_len=max_len)

        num_active = int(self.has_freq) + int(self.has_pat) + int(self.has_trans)
        fused_dim = 512 * num_active

        self.fc = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(cnn_drop),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(cnn_drop),
            nn.Linear(128, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = []
        if self.has_pat:
            pooled.append(self.pattern(x))
        if self.has_freq:
            pooled.append(self.frequency(x))
        if self.has_trans:
            pooled.append(self.transformer(x))

        h_fused = torch.cat(pooled, dim=1) if len(pooled) > 1 else pooled[0]
        return self.fc(h_fused)
