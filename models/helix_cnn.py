#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class HelixCNN(nn.Module):
    """
    4-layer multi-scale 1D CNN (Shi et al., 2026):
    Block 1: k=3  (4 -> 128 ch)
    Block 2: k=9  (128 -> 256 ch)
    Block 3: k=4  (256 -> 128 ch)
    Block 4: k=4  (128 -> 64 ch)
    AdaptiveAvgPool1d(1) rende la rete agnostica rispetto alla lunghezza della sequenza.
    """
    def __init__(self, in_dim: int = 4, hidden_dim: int = 128, out_features: int = 1):
        super().__init__()
        h = hidden_dim
        self.b1 = nn.Sequential(nn.Conv1d(in_dim, h, kernel_size=3, padding="same"), nn.ReLU(), nn.BatchNorm1d(h))
        self.b2 = nn.Sequential(nn.Conv1d(h, 2 * h, kernel_size=9, padding="same"), nn.ReLU(), nn.BatchNorm1d(2 * h))
        self.b3 = nn.Sequential(nn.Conv1d(2 * h, h, kernel_size=4, padding="same"), nn.ReLU(), nn.BatchNorm1d(h))
        self.b4 = nn.Sequential(nn.Conv1d(h, h // 2, kernel_size=4, padding="same"), nn.ReLU(), nn.BatchNorm1d(h // 2))
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(h // 2, 32),
            nn.ReLU(),
            nn.Linear(32, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 4) -> Permute in (B, 4, L)
        if x.dim() == 3 and x.shape[-1] == 4:
            x = x.permute(0, 2, 1)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        z = self.global_pool(x).squeeze(-1)
        return self.classifier(z).squeeze(-1)