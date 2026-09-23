#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_attention_rollout(attentions: list, discard_ratio: float = 0.1) -> np.ndarray:
    """
    Calcola l'Attention Rollout iterativo: A_aug = 0.5 * A + 0.5 * I.
    attentions: lista di tensori (B, num_heads, tokens, tokens) per ogni layer.
    """
    result = None
    for attn in attentions:
        attn_mean = attn.mean(dim=1)  # media sulle teste -> (B, N, N)
        B, N, _ = attn_mean.shape

        if discard_ratio > 0.0:
            flat = attn_mean.view(B, -1)
            k = int(flat.size(1) * discard_ratio)
            val, _ = torch.kthvalue(flat, k, dim=1, keepdim=True)
            attn_mean = torch.where(attn_mean < val.unsqueeze(-1), torch.zeros_like(attn_mean), attn_mean)
            attn_mean = attn_mean / (attn_mean.sum(dim=-1, keepdim=True) + 1e-8)

        I = torch.eye(N, device=attn.device).unsqueeze(0).expand(B, -1, -1)
        aug_attn = 0.5 * attn_mean + 0.5 * I
        aug_attn = aug_attn / aug_attn.sum(dim=-1, keepdim=True)

        result = aug_attn if result is None else torch.bmm(aug_attn, result)

    # Estrae l'attenzione dal token [CLS] verso tutti i patch di sequenza
    cls_attention = result[:, 0, 1:]  # (B, N - 1)
    grid_size = int(np.sqrt(cls_attention.shape[1]))
    return cls_attention.reshape(B, grid_size, grid_size).cpu().numpy()

def save_rollout_figure(planes: np.ndarray, rollout_dict: dict, title: str, out_path: str, resolution: int = 128):
    """Salva il pannello comparativo con le proiezioni e le rispettive mappe di attenzione."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"3DCV Attention Rollout — {title}", fontsize=14, fontweight='bold')
    names = ['P1 (XY)', 'P2 (XZ)', 'P3 (YZ)']

    for i in range(3):
        axes[0, i].imshow(planes[i], cmap='bone')
        axes[0, i].set_title(f"Proiezione {names[i]}")
        axes[0, i].axis('off')

    fused_cgr = np.clip(planes.sum(axis=0), 0, 1)
    axes[0, 3].imshow(fused_cgr, cmap='bone')
    axes[0, 3].set_title("CGR Sovrapposta")
    axes[0, 3].axis('off')

    fused_rollout = np.zeros((resolution, resolution), dtype=np.float32)
    keys = ['p1', 'p2', 'p3']

    for i, k in enumerate(keys):
        if k in rollout_dict:
            r_map = rollout_dict[k][0]
            r_t = torch.tensor(r_map).unsqueeze(0).unsqueeze(0)
            r_resized = F.interpolate(r_t, size=(resolution, resolution), mode='bilinear', align_corners=False).squeeze().numpy()
            fused_rollout += r_resized

            axes[1, i].imshow(planes[i], cmap='gray', alpha=0.5)
            axes[1, i].imshow(r_resized, cmap='jet', alpha=0.6)
            axes[1, i].set_title(f"Rollout {names[i]}")
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')

    fused_norm = (fused_rollout - fused_rollout.min()) / (fused_rollout.max() - fused_rollout.min() + 1e-8)
    axes[1, 3].imshow(fused_cgr, cmap='gray', alpha=0.5)
    axes[1, 3].imshow(fused_norm, cmap='jet', alpha=0.6)
    axes[1, 3].set_title("Rollout Combinato (Somma)")
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()