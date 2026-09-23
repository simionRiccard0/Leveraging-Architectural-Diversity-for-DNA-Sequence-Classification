#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sequence representations and geometric coordinate transforms:
- Exact HelixPR (B-DNA helical pitch + CLS composition token)
- Standard One-Hot encoding
- 3D Tetrahedral Chaos Game Representation (CGR) and orthogonal plane projections
"""

import math
import numpy as np
import torch

# ==============================================================================
# 1. CODIFICHE 1D (ONE-HOT E HELIXPR ORIGINALE)
# ==============================================================================

NUCLEOTIDE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot_encode(sequence: str, max_len: int = None, padding_mode: str = "zero") -> np.ndarray:
    """Codifica standard One-Hot su 4 canali ortogonali (A, C, G, T)."""
    seq = sequence.upper()
    L = len(seq)
    arr = np.zeros((L, 4), dtype=np.float32)
    for i, char in enumerate(seq):
        idx = NUCLEOTIDE_TO_INDEX.get(char, -1)
        if idx != -1:
            arr[i, idx] = 1.0
        else:
            arr[i, :] = 0.25

    if max_len is not None and max_len > L:
        pad_len = max_len - L
        if padding_mode == "circular":
            reps = int(np.ceil(max_len / L))
            arr = np.tile(arr, (reps, 1))[:max_len]
        else:
            pad = np.zeros((pad_len, 4), dtype=np.float32)
            arr = np.vstack([arr, pad])
    return arr


def helixpr_encode(
    sequence: str,
    alpha: float = 0.05,
    period: float = 10.0,
    max_len: int = None,
    padding_mode: str = "zero",
) -> np.ndarray:
    """Implementazione fedele al 100% all'algoritmo originale di HelixPR.

    Caratteristiche conformi all'autore:
    - Token CLS in posizione 0 con composizione globale delle basi: [#A/L, #C/L, #G/L, #T/L].
    - Coordinate trigonometriche [sin, cos, -sin, cos] su periodo T=10.0 bp.
    - Indicizzazione posizionale 1-based per i nucleotidi.
    Output: matrice numpy di shape (L + 1, 4) oppure (max_len, 4).
    """
    seq = sequence.upper()
    L = len(seq)
    if L == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # 1. CLS Token compositivo (riga 0)
    cls_token = np.array(
        [
            seq.count("A") / L,
            seq.count("C") / L,
            seq.count("G") / L,
            seq.count("T") / L,
        ],
        dtype=np.float32,
    )

    # 2. Matrice One-Hot (L, 4)
    one_hot = np.zeros((L, 4), dtype=np.float32)
    for i, char in enumerate(seq):
        idx = NUCLEOTIDE_TO_INDEX.get(char, -1)
        if idx != -1:
            one_hot[i, idx] = 1.0
        else:
            one_hot[i, :] = 0.25

    # 3. Fourier Positional Encoding conforme: [sin, cos, -sin, cos]
    positions = np.arange(1, L + 1, dtype=np.float32)
    angles = 2.0 * np.pi * positions / period
    sin_vals = np.sin(angles)
    cos_vals = np.cos(angles)

    pos_enc = np.stack([sin_vals, cos_vals, -sin_vals, cos_vals], axis=1)

    # Somma pesata e concatenazione del CLS token
    body = one_hot + alpha * pos_enc
    encoded = np.vstack([cls_token.reshape(1, 4), body])  # (L + 1, 4)

    # 4. Gestione padding se richiesta
    total_len = encoded.shape[0]
    if max_len is not None and max_len > total_len:
        pad_len = max_len - total_len
        if padding_mode == "circular":
            # Replica ciclica preservando il CLS token all'indice 0
            body_only = encoded[1:]
            reps = int(np.ceil((max_len - 1) / len(body_only)))
            tiled_body = np.tile(body_only, (reps, 1))[: max_len - 1]
            encoded = np.vstack([encoded[0:1], tiled_body])
        else:
            pad = np.zeros((pad_len, 4), dtype=np.float32)
            encoded = np.vstack([encoded, pad])

    return encoded


# ==============================================================================
# 2. PROIEZIONE 3D CGR TETRAEDRICA (REBUTTAL 3DCV)
# ==============================================================================

# Vertici del tetraedro regolare nello spazio euclideo tridimensionale
CANONICAL_TETRAHEDRON = {
    "A": np.array([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3.0),
    "C": np.array([1.0, -1.0, -1.0], dtype=np.float32) / np.sqrt(3.0),
    "G": np.array([-1.0, 1.0, -1.0], dtype=np.float32) / np.sqrt(3.0),
    "T": np.array([-1.0, -1.0, 1.0], dtype=np.float32) / np.sqrt(3.0),
}


def get_shuffled_tetrahedron(seed: int) -> dict:
    """Restituisce un mapping dei vertici del tetraedro permutato casualmente per i controlli di sensitività geometrica."""
    rng = np.random.RandomState(seed)
    keys = ["A", "C", "G", "T"]
    coords = [CANONICAL_TETRAHEDRON[k] for k in keys]
    permuted_indices = rng.permutation(4)
    return {keys[i]: coords[permuted_indices[i]] for i in range(4)}


def sequence_to_3d_cgr(sequence: str, vertex_map: dict = None) -> np.ndarray:
    """Calcola la traiettoria Chaos Game Representation in 3D: X_t = 0.5 * (X_{t-1} + V_{n_t})."""
    if vertex_map is None:
        vertex_map = CANONICAL_TETRAHEDRON

    current_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    points = []

    for char in sequence.upper():
        if char in vertex_map:
            target = vertex_map[char]
            current_pos = 0.5 * (current_pos + target)
            points.append(current_pos.copy())

    return np.array(points, dtype=np.float32) if points else np.zeros((1, 3), dtype=np.float32)


def project_to_orthogonal_planes(points_3d: np.ndarray, resolution: int = 128) -> np.ndarray:
    """Proietta la traiettoria 3D CGR sui tre piani ortogonali:

    P1 (XY), P2 (XZ), P3 (YZ), normalizzati in istogrammi 2D a risoluzione fissa.
    Output: tensore numpy di shape (3, resolution, resolution).
    """
    planes = np.zeros((3, resolution, resolution), dtype=np.float32)
    if len(points_3d) == 0:
        return planes

    # Coordinate limitate a [-1, 1] e convertite in pixel [0, resolution - 1]
    def to_pixel(coords_2d):
        clamped = np.clip(coords_2d, -1.0, 1.0)
        pixel_indices = ((clamped + 1.0) / 2.0 * (resolution - 1)).astype(np.int64)
        return pixel_indices

    # P1: XY
    p1_idx = to_pixel(points_3d[:, [0, 1]])
    np.add.at(planes[0], (p1_idx[:, 1], p1_idx[:, 0]), 1.0)

    # P2: XZ
    p2_idx = to_pixel(points_3d[:, [0, 2]])
    np.add.at(planes[1], (p2_idx[:, 1], p2_idx[:, 0]), 1.0)

    # P3: YZ
    p3_idx = to_pixel(points_3d[:, [1, 2]])
    np.add.at(planes[2], (p3_idx[:, 1], p3_idx[:, 0]), 1.0)

    # Normalizzazione Min-Max per piano
    for i in range(3):
        max_val = planes[i].max()
        if max_val > 0.0:
            planes[i] = planes[i] / max_val

    return planes