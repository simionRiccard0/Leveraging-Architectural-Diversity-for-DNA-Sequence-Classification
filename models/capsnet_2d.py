#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

def squash(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm + 1e-8)
    return scale * (tensor / torch.sqrt(squared_norm + 1e-8))

class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels: int = 256, out_caps: int = 32, cap_dim: int = 8, kernel_size: int = 9, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_caps * cap_dim, kernel_size=kernel_size, stride=stride)
        self.out_caps = out_caps
        self.cap_dim = cap_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.conv(x)
        B, _, H, W = u.shape
        u = u.view(B, self.out_caps, self.cap_dim, H, W)
        u = u.permute(0, 1, 3, 4, 2).contiguous()
        return squash(u.view(B, -1, self.cap_dim))

class TypeCapsules(nn.Module):
    def __init__(self, in_caps: int, in_dim: int = 8, num_classes: int = 2, out_dim: int = 32, num_routing: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_routing = num_routing
        self.W = nn.Parameter(torch.randn(num_classes, in_caps, out_dim, in_dim) * 0.01)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B = u.shape[0]
        u_exp = u.unsqueeze(1).unsqueeze(-1)
        W_exp = self.W.unsqueeze(0).expand(B, -1, -1, -1, -1)
        u_hat = torch.matmul(W_exp, u_exp).squeeze(-1)  # (B, num_classes, in_caps, out_dim)

        b_ij = torch.zeros(B, self.num_classes, u.shape[1], device=u.device)
        for r in range(self.num_routing):
            c_ij = F.softmax(b_ij, dim=1)
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=2)
            v_j = squash(s_j, dim=-1)
            if r < self.num_routing - 1:
                b_ij = b_ij + (u_hat * v_j.unsqueeze(2)).sum(dim=-1)
        return v_j

class CapsEnhancerModel(nn.Module):
    """
    Rete a capsule per FCGR 64x64:
    Conv2D(1 -> 256, k=9, s=1) -> PrimaryCaps(32 canali x 8 dim, k=9, s=2) -> TypeCaps(2 capsule x 32 dim).
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=9, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        # Input 64x64 -> Conv1 k=9, s=1 -> 56x56 -> PrimaryCaps k=9, s=2 -> 24x24
        self.primary = PrimaryCapsules(256, out_caps=32, cap_dim=8, kernel_size=9, stride=2)
        total_primary = 32 * 24 * 24  # 18.432 capsule
        self.type_caps = TypeCapsules(in_caps=total_primary, in_dim=8, num_classes=num_classes, out_dim=32, num_routing=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Restituisce probabilità positive (norma euclidea della capsula di classe)
        h = self.relu(self.conv1(x))
        u = self.primary(h)
        v = self.type_caps(u)
        probs = torch.norm(v, dim=-1)  # (B, 2)
        return probs[:, 1]