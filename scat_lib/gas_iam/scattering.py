from __future__ import annotations
import numpy as np
from typing import List, Tuple
from .cm import CromerMannTable, fx_cromer_mann
from .constants import PI

def _sinc(x: np.ndarray) -> np.ndarray:
    # Return sin(x)/x with the correct limit at x=0
    out = np.empty_like(x, dtype=float)
    small = np.abs(x) < 1e-12
    out[small] = 1.0
    xs = x[~small]
    out[~small] = np.sin(xs)/xs
    return out

def intensity_components_xray(positions: np.ndarray, labels: List[str], q: np.ndarray, cm: CromerMannTable) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Return (I_total, I_self, I_cross) for reference, though the F90 prints only I_total==I_molecular.
    This follows the Debye expression:
        I_total(q) = sum_{i,j} f_i(s) f_j(s) sinc(q r_ij)
    with s = q / (4π).
    """
    R = np.asarray(positions, float)
    q = np.asarray(q, float)
    N = R.shape[0]
    # distances
    diffs = R[:,None,:] - R[None,:,:]
    rij = np.linalg.norm(diffs, axis=2)  # (N,N)
    # Precompute diagonal mask
    diag_mask = np.eye(N, dtype=bool)
    # Allocate outputs
    I_tot = np.zeros(q.shape, float)
    I_self = np.zeros(q.shape, float)
    I_cross = np.zeros(q.shape, float)
    # Loop over q to avoid gigantic memory
    for k, qk in enumerate(q):
        s = qk / (4.0*PI)
        # per-atom form factors at this s
        w = np.array([fx_cromer_mann(sym, s, cm) for sym in labels], float)  # (N,)
        # self and cross separated only for reporting; math below computes total via quadratic form
        I_self[k] = float(np.sum(w*w))
        # sinc matrix
        S = _sinc(qk * rij)                 # (N,N); S_ii = 1 by our small-limit handling
        # Total
        I_tot[k] = float(w @ (S @ w))
        # Cross part (subtract diagonal contributions)
        I_cross[k] = I_tot[k] - I_self[k]
    return I_tot, I_self, I_cross

def intensity_molecular_xray(positions: np.ndarray, labels: List[str], q: np.ndarray, cm: CromerMannTable) -> np.ndarray:
    """Return I(q) exactly as printed by the F90 code for X-rays:
    I(q) = sum_{i,j} f_i(s) f_j(s) sinc(q r_ij), with s = q/(4π).
    No Debye–Waller, no damping, and includes i==j terms (limit 1).
    """
    I_tot, _, _ = intensity_components_xray(positions, labels, q, cm)
    return I_tot
