"""
PhysHelical: Bio-physically grounded 4-channel input representation.
- Parametri canonici Olson et al. (1998) a 6 dimensioni (Shift, Slide, Rise, Tilt, Roll, Twist).
- Normalizzazione Z-Score offline sulla matrice dei 16 dinucleotidi.
- Modulatore strutturale a 4 canali: g(i) = tanh(m(i)).
- Portante armonica 4D: h(i) = [sin, cos, -sin, cos].
- Modulazione additiva controllata: X(i) = OneHot + alpha * (h * (1 + lambda_phys * g)).
- Token CLS composizionale di frequenza globale in posizione 0 (L + 1 canali).
"""

import math
from typing import Dict, Optional
import numpy as np

OLSON_CANONICAL: Dict[str, list] = {
    "AA": [ 0.04, -0.30, 3.04, -0.11, -5.44, 35.62],
    "AC": [ 0.02, -0.47, 3.24, -0.62, -2.24, 34.40],
    "AG": [-0.04, -0.21, 3.32,  0.08,  8.40, 27.79],
    "AT": [ 0.00, -0.42, 3.32,  0.00, -6.02, 31.50],
    "CA": [ 0.11,  0.89, 3.38,  0.38,  6.70, 34.50],
    "CC": [-0.01, -0.07, 3.37, -0.14,  4.50, 33.67],
    "CG": [ 0.00,  0.32, 3.61,  0.00,  6.70, 29.80],
    "CT": [ 0.04, -0.21, 3.32, -0.08,  8.40, 27.79],
    "GA": [-0.02, -0.02, 3.17, -2.40, -5.00, 36.90],
    "GC": [ 0.00,  0.00, 3.40,  0.00, -5.00, 40.00],
    "GG": [ 0.01, -0.07, 3.37,  0.14,  4.50, 33.67],
    "GT": [-0.02, -0.47, 3.24,  0.62, -2.24, 34.40],
    "TA": [ 0.00, -0.45, 3.60,  0.00,  0.90, 36.00],
    "TC": [ 0.02, -0.02, 3.17,  2.40, -5.00, 36.90],
    "TG": [-0.11,  0.89, 3.38, -0.38,  6.70, 34.50],
    "TT": [-0.04, -0.30, 3.04,  0.11, -5.44, 35.62],
}

BASES = "ACGT"
_LUT = np.full(256, 4, dtype=np.int64)
for _i, _b in enumerate(BASES):
    _LUT[ord(_b)] = _i

_O_MAT = np.array([OLSON_CANONICAL[a + b] for a in BASES for b in BASES], dtype=np.float32)
_O_MEAN = _O_MAT.mean(axis=0)
_O_STD = _O_MAT.std(axis=0)
_O_STD[_O_STD == 0.0] = 1.0
_O_Z = (_O_MAT - _O_MEAN) / _O_STD

TASK_OPTIMAL_PERIODS: Dict[str, float] = {
    "enhancer": 10.07,
    "cohn": 9.97,
    "basith": 10.05,
    "viral": 10.00,
    "ecoli": 10.50,
    "anellovirus": 10.00,
    "modomics": 10.00,
}


def get_dataset_period(task: str) -> float:
    return TASK_OPTIMAL_PERIODS.get(task.lower(), 10.07)


def encode_physhelical(
    seq: str,
    period_T: float = 10.07,
    alpha: float = 0.05,
    lambda_phys: float = 0.50,
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

    indices = _LUT[np.frombuffer(seq.encode(), dtype=np.uint8)]
    valid = indices < 4

    # 1. Base One-Hot (L, 4)
    oh = np.zeros((L, 4), dtype=np.float32)
    oh[np.flatnonzero(valid), indices[valid]] = 1.0
    oh[~valid, :] = 0.25

    # 2. Parametri fisico-chimici interpolati per-base (L, 6)
    if L > 1:
        dinuc_idx = 4 * indices[:-1] + indices[1:]
        pair_valid = valid[:-1] & valid[1:]
        steps = np.zeros((L - 1, 6), dtype=np.float32)
        steps[pair_valid] = _O_Z[dinuc_idx[pair_valid]]
        steps_ext = np.vstack([steps[:1], steps, steps[-1:]])
        p = 0.5 * (steps_ext[:-1] + steps_ext[1:])
    else:
        p = np.zeros((L, 6), dtype=np.float32)

    # 3. Modulatore conformazionale a 4 canali: m(i)
    m1 = p[:, 5] + p[:, 2]
    m2 = p[:, 4] + p[:, 3]
    m3 = p[:, 4] - p[:, 3]
    m4 = p[:, 5] - p[:, 2]
    g = np.tanh(np.stack([m1, m2, m3, m4], axis=1))

    # 4. Portante armonica 4D
    pos = np.arange(1, L + 1, dtype=np.float32)[:, None]
    angle = 2.0 * math.pi * pos / float(max(period_T, 1e-4))
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    h = np.concatenate([sin_a, cos_a, -sin_a, cos_a], axis=1)

    # 5. Modulazione additiva
    body = oh + alpha * (h * (1.0 + lambda_phys * g))

    # 6. Token CLS composizionale
    if include_cls:
        cls_token = np.array([[seq.count(b) / L for b in BASES]], dtype=np.float32)
        mat = np.vstack([cls_token, body]).astype(np.float32)
    else:
        mat = body.astype(np.float32)

    if target_len is not None:
        expected_len = target_len + (1 if include_cls else 0)
        if mat.shape[0] < expected_len:
            pad = np.zeros((expected_len - mat.shape[0], 4), dtype=np.float32)
            mat = np.vstack([mat, pad])

    return mat