#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Attention Rollout and High-Resolution ViT Saliency Visualizations for 3DCV.
Computes recursive attention flow: A_aug = 0.5 * A + 0.5 * I.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def compute_attention_rollout(attentions: list, discard_ratio: float = 0.1) -> np.ndarray:
    """Calcola l'Attention Rollout cumulativo attraverso tutti i layer del ViT.

    Gestisce:
    - Tensori a 4 dimensioni: (Batch, Num_Heads, N, N) tramite media sulle teste.
    - Tensori a 3 dimensioni: (Batch, N, N).
    Restituisce:
    - Mappa spaziale CLS-to-Patches risagomata a (Batch, Grid_Size, Grid_Size).
    """
    result = None

    for attn in attentions:
        if attn.dim() == 4:
            attn_mean = attn.mean(dim=1)
        elif attn.dim() == 3:
            attn_mean = attn
        else:
            raise ValueError(f"Dimensione inattesa per i pesi di attenzione: {attn.shape}")

        B, N, _ = attn_mean.shape

        # Filtraggio del rumore tramite soglia di percentile
        if discard_ratio > 0.0:
            flat = attn_mean.view(B, -1)
            k = int(flat.size(1) * discard_ratio)
            val, _ = torch.kthvalue(flat, k, dim=1, keepdim=True)
            attn_mean = torch.where(
                attn_mean < val.unsqueeze(-1),
                torch.zeros_like(attn_mean),
                attn_mean,
            )
            attn_mean = attn_mean / (attn_mean.sum(dim=-1, keepdim=True) + 1e-8)

        # Matrice identità per self-residual connection
        I = torch.eye(N, device=attn.device).unsqueeze(0).expand(B, -1, -1)
        aug_attn = 0.5 * attn_mean + 0.5 * I
        aug_attn = aug_attn / aug_attn.sum(dim=-1, keepdim=True)

        result = aug_attn if result is None else torch.bmm(aug_attn, result)

    # Estrazione dell'attenzione tra il token CLS (indice 0) e tutti i patch spaziali (1:)
    cls_attention = result[:, 0, 1:]  # (Batch, N - 1)
    num_patches = cls_attention.shape[1]
    grid_size = int(np.sqrt(num_patches))

    if grid_size * grid_size != num_patches:
        raise ValueError(
            f"Il numero di token patch ({num_patches}) non forma una griglia quadrata perfetta."
        )

    spatial_map = cls_attention.reshape(B, grid_size, grid_size).detach().cpu().numpy()
    return spatial_map


def save_rollout_figure(
    planes: np.ndarray,
    rollout_dict: dict,
    title: str,
    out_path: str,
    resolution: int = 128,
):
    """Genera e salva il pannello multi-vista comparativo a 2 righe e 4 colonne:

    - Riga 1: P1 (XY), P2 (XZ), P3 (YZ), Sovrapposizione CGR 3D
    - Riga 2: Rollout P1, Rollout P2, Rollout P3, Rollout Combinato (Somma pesata).
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor="white")
    fig.suptitle(f"3DCV Attention Rollout — {title}", fontsize=16, fontweight="bold", y=0.98)

    plane_labels = ["Proiezione P1 (XY)", "Proiezione P2 (XZ)", "Proiezione P3 (YZ)"]

    # --- Riga 1: Piani CGR ---
    for i in range(3):
        axes[0, i].imshow(planes[i], cmap="bone", origin="lower")
        axes[0, i].set_title(plane_labels[i], fontsize=12, fontweight="medium")
        axes[0, i].axis("off")

    fused_cgr = np.clip(planes.sum(axis=0), 0.0, 1.0)
    axes[0, 3].imshow(fused_cgr, cmap="bone", origin="lower")
    axes[0, 3].set_title("CGR Sovrapposta (Fusione 3D)", fontsize=12, fontweight="medium")
    axes[0, 3].axis("off")

    # --- Riga 2: Attention Rollout bilineare sovrapposto ---
    fused_rollout = np.zeros((resolution, resolution), dtype=np.float32)
    view_keys = ["p1", "p2", "p3"]

    for i, k in enumerate(view_keys):
        if k in rollout_dict:
            r_map = rollout_dict[k][0]  # Primo elemento del batch
            r_tensor = torch.tensor(r_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            r_upsampled = (
                F.interpolate(
                    r_tensor,
                    size=(resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .numpy()
            )

            # Normalizzazione locale
            r_norm = (r_upsampled - r_upsampled.min()) / (r_upsampled.max() - r_upsampled.min() + 1e-8)
            fused_rollout += r_norm

            axes[1, i].imshow(planes[i], cmap="gray", alpha=0.45, origin="lower")
            axes[1, i].imshow(r_norm, cmap="jet", alpha=0.60, origin="lower")
            axes[1, i].set_title(f"Rollout {plane_labels[i].split()[-1]}", fontsize=12)
            axes[1, i].axis("off")
        else:
            axes[1, i].axis("off")

    # Mappa combinata finale normalizzata
    comb_norm = (fused_rollout - fused_rollout.min()) / (fused_rollout.max() - fused_rollout.min() + 1e-8)
    axes[1, 3].imshow(fused_cgr, cmap="gray", alpha=0.45, origin="lower")
    axes[1, 3].imshow(comb_norm, cmap="jet", alpha=0.65, origin="lower")
    axes[1, 3].set_title("Rollout Combinato (Fusione Multi-Vista)", fontsize=12, fontweight="bold")
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()