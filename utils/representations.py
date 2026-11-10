"""
Alternative input representations used in the comparison with the literature:
- HelixPR: one-hot encoding with a helix-aware periodic positional term and a
  global composition (CLS) token (Shi et al.).
- Komlós-Hadamard: 3-channel tetrahedral mapping (Kabbani et al.).
"""

import math
from typing import Dict, Optional
import numpy as np

KOMLOS_MAPPING: Dict[str, list] = {
    "A": [ 1.0,  1.0, -1.0],
    "C": [-1.0,  1.0,  1.0],
    "T": [ 1.0, -1.0,  1.0],
    "U": [ 1.0, -1.0,  1.0],
    "G": [-1.0, -1.0, -1.0],
    "N": [ 0.0,  0.0,  0.0]
}

BASES = "ACGT"
_LUT = np.full(256, 4, dtype=np.int64)
for _i, _b in enumerate(BASES):
    _LUT[ord(_b)] = _i


TASK_OPTIMAL_PERIODS: Dict[str, float] = {
    "enhancer": 10.07,
    "viral": 10.00,
    "ecoli": 10.50,
    "anellovirus": 10.00,
}


def get_dataset_period(task: str) -> float:
    return TASK_OPTIMAL_PERIODS.get(task.lower(), 10.07)


def encode_komlos(seq: str, target_len: Optional[int] = None) -> np.ndarray:
    seq = seq.upper().replace("U", "T")
    if target_len is not None:
        seq = seq[:target_len]
    enc = [KOMLOS_MAPPING.get(b, [0.0, 0.0, 0.0]) for b in seq]
    if target_len is not None:
        while len(enc) < target_len:
            enc.append([0.0, 0.0, 0.0])
    return np.array(enc, dtype=np.float32)


def encode_helixpr(
    seq: str,
    period_T: float = 10.0,
    alpha: float = 0.05,
    include_cls: bool = True,
    target_len: Optional[int] = None,
) -> np.ndarray:
    seq = seq.upper().replace("U", "T")
    if target_len is not None and len(seq) > target_len:
        seq = seq[:target_len]

    L = len(seq)
    if L == 0:
        out_len = (target_len + 1) if (target_len is not None and include_cls) else (target_len or 1)
        return np.zeros((out_len, 4), dtype=np.float32)

    BASES = "ACGT"

    cls_token = np.array([[seq.count(b) / L for b in BASES]], dtype=np.float32)

    indices = _LUT[np.frombuffer(seq.encode(), dtype=np.uint8)]
    valid = indices < 4

    oh = np.zeros((L, 4), dtype=np.float32)
    oh[np.flatnonzero(valid), indices[valid]] = 1.0
    oh[~valid, :] = 0.25

    pos = np.arange(1, L + 1, dtype=np.float32)[:, None]
    angle = 2.0 * math.pi * pos / float(max(period_T, 1e-4))
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    pos_enc = np.concatenate([sin_a, cos_a, -sin_a, cos_a], axis=1)

    body = oh + alpha * pos_enc

    if include_cls:
        mat = np.vstack([cls_token, body]).astype(np.float32)
    else:
        mat = body.astype(np.float32)

    if target_len is not None:
        expected_len = target_len + (1 if include_cls else 0)
        if mat.shape[0] < expected_len:
            pad = np.zeros((expected_len - mat.shape[0], 4), dtype=np.float32)
            mat = np.vstack([mat, pad])

    return mat
