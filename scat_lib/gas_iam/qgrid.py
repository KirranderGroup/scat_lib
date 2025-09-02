from __future__ import annotations
import numpy as np
from .constants import A0_ANG

def q_grid_f90(nq: int = 1000, qmin: float = 0.0, qmax: float | None = None) -> np.ndarray:
    """Default q-grid used by the original F90: [0, 8 / a0] Å^-1 with nq points.
    a0 = 0.529177210903 Å (Bohr radius).
    """
    if qmax is None:
        qmax = 8.0 / A0_ANG
    return np.linspace(qmin, qmax, int(nq))
