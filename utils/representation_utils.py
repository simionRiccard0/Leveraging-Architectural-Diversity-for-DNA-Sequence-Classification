#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import numpy as np
import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. HELIXPR ENCODING (Shi et al., BMC Bioinformatics 2026)
# -----------------------------------------------------------------------------
class HelixPREncoder:
    """
    Helix-aware Periodic Representation:
    - Token CLS composizionale globale prepended: [f_A, f_T, f_C, f_G]
    - One-hot encoding per nucleotide v_k
    - Fourier Positional Encoding periodico p_k con T=10 bp e ampiezza alpha=0.05
    """
    MAPPING = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    def __init__(self, T: float = 10.0, alpha: float = 0.05):
        self.T = float(T)
        self.alpha = float(alpha)

    def encode(self, sequence: str, target_len: int = None, padding_mode: str = "zero") -> np.ndarray:
        seq = str(sequence).upper().strip()
        L_orig = len(seq)
        
        # Conteggio frequenze per il token CLS globale
        counts = {'A': 0, 'T': 0, 'C': 0, 'G': 0}
        for b in seq:
            if b in counts:
                counts[b] += 1
        denom = max(1, L_orig)
        cls_token = np.array([counts['A']/denom, counts['T']/denom, counts['C']/denom, counts['G']/denom], dtype=np.float32)

        # Gestione lunghezza variabile / padding
        if target_len is not None and len(seq) < target_len:
            if padding_mode == "circular":
                repeats = (target_len // len(seq)) + 1
                seq = (seq * repeats)[:target_len]
            # per 'zero' padding, il riempimento avviene a livello di matrice

        L = len(seq) if target_len is None else max(len(seq), target_len)
        matrix = np.zeros((L + 1, 4), dtype=np.float32)
        matrix[0] = cls_token

        for k in range(1, len(seq) + 1):
            base = seq[k - 1]
            v_k = np.zeros(4, dtype=np.float32)
            if base in self.MAPPING:
                v_k[self.MAPPING[base]] = 1.0

            theta_k = (2.0 * math.pi * k) / self.T
            sin_t = math.sin(theta_k)
            cos_t = math.cos(theta_k)
            # Layout canali conforme al paper: [sin, cos, cos, sin]
            p_k = np.array([sin_t, cos_t, cos_t, sin_t], dtype=np.float32)
            matrix[k] = v_k + self.alpha * p_k

        if target_len is not None:
            if matrix.shape[0] < target_len + 1:
                pad_size = (target_len + 1) - matrix.shape[0]
                matrix = np.pad(matrix, ((0, pad_size), (0, 0)), mode='constant')
            else:
                matrix = matrix[:target_len + 1]

        return matrix


# -----------------------------------------------------------------------------
# 2. 2D FCGR ENCODING (CapsEnhancer / Yao et al., JCIM 2024)
# -----------------------------------------------------------------------------
class FCGR2DEncoder:
    """
    Frequency Chaos Game Representation 2D discretizzata su griglia N x N (default 64x64).
    Mappatura su quadrato unitario: A=(-1,1), C=(-1,-1), G=(1,-1), T=(1,1).
    """
    VERTICES = {
        'A': np.array([-1.0,  1.0], dtype=np.float32),
        'C': np.array([-1.0, -1.0], dtype=np.float32),
        'G': np.array([ 1.0, -1.0], dtype=np.float32),
        'T': np.array([ 1.0,  1.0], dtype=np.float32),
    }

    def __init__(self, resolution: int = 64):
        self.k = resolution

    def encode(self, sequence: str) -> np.ndarray:
        seq = str(sequence).upper().strip()
        pos = np.array([0.0, 0.0], dtype=np.float32)
        grid = np.zeros((self.k, self.k), dtype=np.float32)
        coords = []

        for b in seq:
            if b in self.VERTICES:
                pos = 0.5 * (pos + self.VERTICES[b])
                coords.append(pos.copy())

        if len(coords) == 0:
            return grid[np.newaxis, ...]

        coords = np.array(coords)
        px = np.clip(((coords[:, 0] + 1.0) * 0.5 * (self.k - 1)).astype(np.int32), 0, self.k - 1)
        py = np.clip(((coords[:, 1] + 1.0) * 0.5 * (self.k - 1)).astype(np.int32), 0, self.k - 1)
        np.add.at(grid, (py, px), 1.0)

        max_val = grid.max()
        if max_val > 0:
            grid /= max_val
        return grid[np.newaxis, ...]  # Shape: (1, resolution, resolution)


# -----------------------------------------------------------------------------
# 3. 3DCV TRAJECTORY & PROJECTION ENCODER (Ablations & Shuffle Controls)
# -----------------------------------------------------------------------------
DEFAULT_TETRAHEDRON = {
    'A': np.array([-1.0, -1.0, -1.0], dtype=np.float32),
    'T': np.array([ 1.0,  1.0, -1.0], dtype=np.float32),
    'C': np.array([ 1.0, -1.0,  1.0], dtype=np.float32),
    'G': np.array([-1.0,  1.0,  1.0], dtype=np.float32),
}

def get_shuffled_tetrahedron(seed: int = None) -> dict:
    """Restituisce una permutazione casuale dei vertici tetraedrici per i controlli."""
    rng = random.Random(seed)
    bases = ['A', 'T', 'C', 'G']
    coords = [
        np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        np.array([ 1.0,  1.0, -1.0], dtype=np.float32),
        np.array([ 1.0, -1.0,  1.0], dtype=np.float32),
        np.array([-1.0,  1.0,  1.0], dtype=np.float32),
    ]
    rng.shuffle(coords)
    return {b: coords[i] for i, b in enumerate(bases)}

class CGR3DProjector:
    """Generatore di traiettoria CGR 3D e proiezioni ortogonali sui piani P1 (XY), P2 (XZ), P3 (YZ)."""
    def __init__(self, resolution: int = 128, vertices: dict = None):
        self.resolution = resolution
        self.vertices = vertices if vertices is not None else DEFAULT_TETRAHEDRON

    def encode(self, sequence: str) -> np.ndarray:
        seq = str(sequence).upper().strip()
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        traj = []
        for b in seq:
            if b in self.vertices:
                pos = 0.5 * (pos + self.vertices[b])
                traj.append(pos.copy())

        planes = np.zeros((3, self.resolution, self.resolution), dtype=np.float32)
        if len(traj) == 0:
            return planes

        traj = np.array(traj)
        scaled = (traj + 1.0) * 0.5 * (self.resolution - 1)
        px = np.clip(scaled[:, 0].astype(np.int32), 0, self.resolution - 1)
        py = np.clip(scaled[:, 1].astype(np.int32), 0, self.resolution - 1)
        pz = np.clip(scaled[:, 2].astype(np.int32), 0, self.resolution - 1)

        np.add.at(planes[0], (py, px), 1.0)  # P1: XY
        np.add.at(planes[1], (pz, px), 1.0)  # P2: XZ
        np.add.at(planes[2], (pz, py), 1.0)  # P3: YZ

        for i in range(3):
            m = planes[i].max()
            if m > 0:
                planes[i] /= m
        return planes  # Shape: (3, resolution, resolution)