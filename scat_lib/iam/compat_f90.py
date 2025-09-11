from __future__ import annotations
import numpy as np
from typing import Sequence, List, Tuple, Optional
from .form_factors import fx_cromer_mann

def sinc_sin_over_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-12
    out[small] = 1.0
    xs = x[~small]
    out[~small] = np.sin(xs) / xs
    return out

def gas_intensity_xray_f90(positions: np.ndarray, elements: Sequence[str], q_vals: Sequence[float]) -> np.ndarray:
    positions = np.asarray(positions, float)
    q_vals = np.asarray(q_vals, float)
    N = positions.shape[0]
    diffs = positions[:, None, :] - positions[None, :, :]
    rij = np.linalg.norm(diffs, axis=2)
    I = np.zeros_like(q_vals, float)
    for k, q in enumerate(q_vals):
        s = q/(4.0*np.pi)
        f = np.array([fx_cromer_mann(el, s) for el in elements], float)
        W = np.outer(f, f)
        x = q * rij
        S = sinc_sin_over_x(x)
        I[k] = float(np.sum(W * S))
    return I

def read_xyz_simple(path: str) -> Tuple[np.ndarray, List[str]]:
    with open(path, 'r', encoding='utf-8') as fh:
        first = fh.readline().strip()
        n = int(first.split()[0])
        _comment = fh.readline()
        elems = []
        coords = []
        for _ in range(n):
            parts = fh.readline().split()
            elems.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(coords, float), elems
