from __future__ import annotations
import numpy as np
from typing import Sequence, List, Optional, Tuple
from .form_factors import fx_cromer_mann

def gas_intensity_xray(structure, q_vals: Sequence[float], *, groups: Optional[List[List[int]]] = None,
                       pair_sigmas: Optional[List[np.ndarray]] = None, group_weights: Optional[Sequence[float]] = None,
                       return_components: bool = False, use_B_for_sigma_if_missing: bool = True) -> Tuple[np.ndarray,np.ndarray,np.ndarray] | np.ndarray:
    q_vals = np.asarray(q_vals, float)
    sites = structure.sites
    N = len(sites)
    R = np.array([s.position for s in sites], float)
    occ = np.array([s.occupancy for s in sites], float)
    B   = np.array([s.debye_waller.B_iso for s in sites], float)
    charges = [s.charge for s in sites]
    elements = [s.element for s in sites]

    if groups is None:
        groups = [list(range(N))]
    group_arrays = [np.array(g, dtype=int) for g in groups]

    def sinc_sin_over_x(x):
        x = np.asarray(x, float)
        out = np.empty_like(x)
        small = np.abs(x) < 1e-12
        out[small] = 1.0
        xs = x[~small]
        out[~small] = np.sin(xs)/xs
        return out

    # Precompute pair index lists and distances per group
    pair_lists = []
    for g in group_arrays:
        I, J = np.triu_indices(g.size, k=1)
        pair_lists.append((g[I], g[J]))

    I_total = np.zeros_like(q_vals)
    I_self  = np.zeros_like(q_vals)
    I_cross = np.zeros_like(q_vals)

    for k, q in enumerate(q_vals):
        s = q/(4.0*np.pi)
        f = np.array([fx_cromer_mann(elem, s, charge=chg) for elem, chg in zip(elements, charges)], float)
        T = np.exp(-B * s*s)
        w = occ * f * T

        Is = 0.0
        Ic = 0.0
        for g in group_arrays:
            wg = w[g]
            Is += float(np.sum(wg*wg))
        for (I,J) in pair_lists:
            rij = np.linalg.norm(R[I]-R[J], axis=1)
            x = q * rij
            sinc = sinc_sin_over_x(x)
            Ic += 2.0 * float(np.sum( (w[I]*w[J]) * sinc ))
        I_self[k] = Is
        I_cross[k] = Ic
        I_total[k] = Is + Ic

    return (I_total, I_self, I_cross) if return_components else I_total

def gas_structure_function_Sq(Iq: np.ndarray, q_vals: np.ndarray, structure, *, groups: Optional[List[List[int]]] = None) -> np.ndarray:
    q_vals = np.asarray(q_vals, float)
    if groups is None:
        groups = [list(range(len(structure.sites)))]
    group_arrays = [np.array(g, dtype=int) for g in groups]
    Sq = np.empty_like(q_vals, float)
    for k, q in enumerate(q_vals):
        s = q/(4.0*np.pi)
        f = np.array([fx_cromer_mann(site.element, s, charge=site.charge) for site in structure.sites], float)
        B = np.array([site.debye_waller.B_iso for site in structure.sites], float)
        T = np.exp(-B * s*s)
        w = f * T
        favg_sq = 0.0; f2avg = 0.0
        for g in group_arrays:
            fg = w[g]
            favg_sq += (np.mean(fg))**2
            f2avg += np.mean(fg*fg)
        favg_sq /= len(group_arrays)
        f2avg /= len(group_arrays)
        Sq[k] = (Iq[k] - f2avg) / max(favg_sq, 1e-30)
    return Sq

def gas_Fq(Sq: np.ndarray, q_vals: np.ndarray) -> np.ndarray:
    return np.asarray(q_vals, float) * (np.asarray(Sq, float) - 1.0)
