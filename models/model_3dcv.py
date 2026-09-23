#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class ViTBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        x_norm = self.norm1(x)
        attn_out, weights = self.attn(x_norm, x_norm, x_norm, need_weights=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        if return_attn:
            return x, weights
        return x

class SingleViewViT(nn.Module):
    def __init__(self, img_size: int = 128, patch_size: int = 16, embed_dim: int = 128,
                 depth: int = 3, heads: int = 4, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([ViTBlock(embed_dim, heads, embed_dim * 2, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B = x.shape[0]
        p = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, p], dim=1) + self.pos_embed
        x = self.dropout(x)

        attns = []
        for l in self.layers:
            if return_attn:
                x, a = l(x, return_attn=True)
                attns.append(a)
            else:
                x = l(x)
        cls_out = self.norm(x[:, 0])
        feat = self.proj(cls_out)
        return (feat, attns) if return_attn else feat

class Model3DCV(nn.Module):
    """
    3DCV completo con supporto alle 7 modalità di ablazione:
    'full', 'p1', 'p2', 'p3', 'p1_p2', 'p1_p3', 'p2_p3'.
    """
    MODES = {
        'full':  ['p1', 'p2', 'p3'],
        'p1':    ['p1'],
        'p2':    ['p2'],
        'p3':    ['p3'],
        'p1_p2': ['p1', 'p2'],
        'p1_p3': ['p1', 'p3'],
        'p2_p3': ['p2', 'p3'],
    }

    def __init__(self, mode: str = 'full', img_size: int = 128, patch_size: int = 16,
                 embed_dim: int = 128, depth: int = 3, heads: int = 4, num_classes: int = 1):
        super().__init__()
        assert mode in self.MODES, f"Modalità non valida: {mode}"
        self.mode = mode
        self.active_views = self.MODES[mode]
        self.feat_dim = 256

        self.branch_p1 = SingleViewViT(img_size, patch_size, embed_dim, depth, heads, self.feat_dim)
        self.branch_p2 = SingleViewViT(img_size, patch_size, embed_dim, depth, heads, self.feat_dim)
        self.branch_p3 = SingleViewViT(img_size, patch_size, embed_dim, depth, heads, self.feat_dim)

        fused_in = self.feat_dim * len(self.active_views)
        self.classifier = nn.Sequential(
            nn.Linear(fused_in, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor, return_attns: bool = False):
        # x: (B, 3, H, W)
        feats, attns = [], {}
        if 'p1' in self.active_views:
            res = self.branch_p1(x[:, 0:1], return_attn=return_attns)
            f, a = res if return_attns else (res, None)
            feats.append(f)
            if return_attns: attns['p1'] = a
        if 'p2' in self.active_views:
            res = self.branch_p2(x[:, 1:2], return_attn=return_attns)
            f, a = res if return_attns else (res, None)
            feats.append(f)
            if return_attns: attns['p2'] = a
        if 'p3' in self.active_views:
            res = self.branch_p3(x[:, 2:3], return_attn=return_attns)
            f, a = res if return_attns else (res, None)
            feats.append(f)
            if return_attns: attns['p3'] = a

        h = torch.cat(feats, dim=1)
        logits = self.classifier(h)
        return (logits, attns) if return_attns else logits