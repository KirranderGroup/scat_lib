from __future__ import annotations
from typing import Tuple, List
import numpy as np

def read_xyz(path: str) -> Tuple[np.ndarray, List[str]]:
    """Read a minimal XYZ file: first line N, second comment, then 'label x y z' per atom.
    Returns positions [Å] as (N,3) float array and the list of labels (strings) as-is.
    Labels are not canonicalized so they can match CM keys like 'Cval', 'Siv'.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    N = int(lines[0].split()[0])
    body = lines[2:2+N]
    labels: List[str] = []
    pos = np.zeros((N,3), dtype=float)
    for i, ln in enumerate(body):
        parts = ln.split()
        labels.append(parts[0])
        pos[i,0] = float(parts[1])
        pos[i,1] = float(parts[2])
        pos[i,2] = float(parts[3])
    return pos, labels

def pair_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """Return NxN matrix of pair distances r_ij [Å]."""
    R = np.asarray(positions, float)
    diffs = R[:,None,:] - R[None,:,:]
    rij = np.linalg.norm(diffs, axis=2)
    return rij
