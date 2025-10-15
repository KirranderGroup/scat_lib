from __future__ import annotations
import numpy as np
from typing import List, Tuple, Callable, Optional, Mapping
from .cm import CromerMannTable, fx_cromer_mann
from .constants import PI

# type of a provider function: fx(symbol: str, s: float) -> float
FXFunc = Callable[[str, float], float]

def _sinc(x: np.ndarray) -> np.ndarray:
    """Return sinc(x) = sin(x)/x, handling x=0 case."""
    out = np.empty_like(x, dtype=float)
    small = np.abs(x) < 1e-12
    out[small] = 1.0
    xs = x[~small]
    out[~small] = np.sin(xs)/xs
    return out

def _fx_from_backend(backend: str | FXFunc, cm: Optional[CromerMannTable] = None, ion_map: Optional[Mapping[str, str]] = None) -> FXFunc:
    """Return a function fx(symbol: str, s: float) -> float according to backend."""
    if callable(backend):
        return backend
    be = (backend or 'affl').lower()
    if be == 'xraydb':
        from .xraydb_provider import fx_xraydb as _fx
        return lambda sym, s: _fx(sym, s, ion_map=ion_map)
    elif be in ('affl','cm','cromer-mann'):
        cm = cm or CromerMannTable()
        return lambda sym, s: fx_cromer_mann(sym, s, cm)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'affl' or 'xraydb' or pass a callable.")


def intensity_components_xray(positions: np.ndarray, labels: List[str], q: np.ndarray, cm: Optional[CromerMannTable] = None,
                              *, backend: str | FXFunc = 'affl', ion_map: Optional[Mapping[str, str]] = None) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Return total, self, and cross I(q) from atomic positions and labels.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) with atomic positions in Angstrom.
    labels : List[str]
        List of length N with atomic symbols or labels.
    q : np.ndarray
        1D array of q values (in 1/Angstrom) at which to
        compute the scattering intensity.
    cm : Optional[CromerMannTable], optional
        Optional Cromer-Mann table to use if backend is 'affl'.
        If None, a default table will be used. Default is None.
    backend : str | FXFunc, optional
        Backend to use for atomic form factors.
        'affl' (default) for internal Cromer–Mann table,
        or 'xraydb' to use xraydb.f0,
        or a callable function of signature fx(symbol: str, s: float) -> float.
        Default is 'affl'.
    ion_map : Optional[Mapping[str, str]], optional
        Optional mapping of labels to standard symbols,
        e.g., {'Cval':'C', 'Siv':'Si4+'}.
        Only used if backend is 'xraydb'. Default is None.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Three 1D arrays of I(q) values corresponding to input q values:
        total, self, and cross terms.
    """
    R = np.asarray(positions, float)
    q = np.asarray(q, float)
    N = R.shape[0]
    diffs = R[:,None,:] - R[None,:,:]
    rij = np.linalg.norm(diffs, axis=2)  # (N,N)

    I_tot = np.zeros(q.shape, float); I_self = np.zeros(q.shape, float); I_cross = np.zeros(q.shape, float)

    fx = _fx_from_backend(backend, cm=cm, ion_map=ion_map)

    # unique labels to avoid repeated lookups at each q
    labels = list(labels)
    uniq = sorted(set(labels))
    # For each q, compute f0 for each unique symbol once
    for k, qk in enumerate(q):
        s = qk / (4.0*PI)
        f_by_sym = {sym: float(fx(sym, s)) for sym in uniq}
        w = np.array([f_by_sym[sym] for sym in labels], float)
        I_self[k] = float(np.sum(w*w))
        S = _sinc(qk * rij)
        I_tot[k] = float(w @ (S @ w))
        I_cross[k] = I_tot[k] - I_self[k]
    return I_tot, I_self, I_cross

def intensity_molecular_xray(positions: np.ndarray, labels: List[str], q: np.ndarray, cm: Optional[CromerMannTable] = None,
                             *, backend: str | FXFunc = 'affl', ion_map: Optional[Mapping[str, str]] = None) -> np.ndarray:
    """
    Return I(q) from atomic positions and labels.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) with atomic positions in Angstrom.
    labels : List[str]
        List of length N with atomic symbols or labels.
    q : np.ndarray
        1D array of q values (in 1/Angstrom) at which to
        compute the scattering intensity.
    cm : Optional[CromerMannTable], optional
        Optional Cromer-Mann table to use if backend is 'affl'.
        If None, a default table will be used. Default is None.
    backend : str | FXFunc, optional
        Backend to use for atomic form factors.
        'affl' (default) for internal Cromer–Mann table,
        or 'xraydb' to use xraydb.f0,
        or a callable function of signature fx(symbol: str, s: float) -> float.
        Default is 'affl'.
    ion_map : Optional[Mapping[str, str]], optional
        Optional mapping of labels to standard symbols,
        e.g., {'Cval':'C', 'Siv':'Si4+'}.
        Only used if backend is 'xray
    db'. Default is None.

    Returns
    -------
    np.ndarray
        1D array of I(q) values corresponding to input q values.
    """
    I_tot, _, _ = intensity_components_xray(positions, labels, q, cm, backend=backend, ion_map=ion_map)
    return I_tot

def intensity_pyscf(mol: "gto.Mole", q: np.ndarray, cm: Optional[CromerMannTable] = None,
                     *, backend: str | FXFunc = 'affl', ion_map: Optional[Mapping[str, str]] = None) -> np.ndarray:
    """Return I(q) from a PySCF gto.Mole object."""
    from .pyscf_bridge import positions_and_labels_from_mole
    positions, labels = positions_and_labels_from_mole(mol)
    return intensity_molecular_xray(positions, labels, q, cm, backend=backend, ion_map=ion_map)