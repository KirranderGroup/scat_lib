"""IAM structure factors for sets of atoms.

We provide minimalist classes to define atom sites and compute structure factors
for either X-rays (standard IAM with f_x) or electrons (IAM using Mott–Bethe).

Conventions
-----------
- Positions are in ångströms in Cartesian coordinates unless `fractional=True` and
  a `cell` (a, b, c, alpha, beta, gamma) is passed (degrees). In that case we convert
  to Cartesian using standard crystallographic conventions.
- Phase factor uses `exp(i * Q·r)` with Q in Å^-1 and r in Å. If you prefer fractional
  Miller indices hkl, construct `Q = 2π * B * hkl`, but that's outside this simple helper.

Debye–Waller factor
-------------------
We implement isotropic DW via B_iso: T(s) = exp(-B_iso * s^2), where s = |q|/(4π).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Sequence, List, Tuple
import numpy as np
import math

from .kinematics import s_from_q
from .form_factors import fx_cromer_mann
from .electron import fe_mott_bethe, fe_mott_bethe_relativistic

@dataclass
class DebyeWaller:
    B_iso: float = 0.0  # Å^2

    def factor_from_q(self, q_mag: float) -> float:
        s = s_from_q(q_mag)
        return float(np.exp(-self.B_iso * s**2))

@dataclass
class AtomSite:
    element: str
    position: Sequence[float]  # Cartesian Å
    occupancy: float = 1.0
    debye_waller: DebyeWaller = field(default_factory=DebyeWaller)
    charge: str | None = None  # e.g. '1-', '2+', '4+' to select ionized CM coefficients

@dataclass
class Structure:
    sites: List[AtomSite]

def _phase(q_vec: np.ndarray, r: Sequence[float]) -> complex:
    return complex(np.exp(1j * float(np.dot(q_vec, np.asarray(r)))))

def structure_factor_xray(structure: Structure, q_vecs: np.ndarray) -> np.ndarray:
    """Compute F(q) for X-rays (IAM) at each q-vector.

    Parameters
    ----------
    structure : Structure
        List of atom sites with positions (Å), occupancies, and Debye–Waller (B_iso).
    q_vecs : array (N, 3)
        Scattering vectors in Å^-1.

    Returns
    -------
    F : complex ndarray (N,)
    """
    q_vecs = np.asarray(q_vecs, dtype=float)
    F = np.zeros(q_vecs.shape[0], dtype=complex)
    for i, q in enumerate(q_vecs):
        qmag = float(np.linalg.norm(q))
        s = qmag / (4.0 * np.pi)
        acc = 0.0 + 0.0j
        for site in structure.sites:
            fx = fx_cromer_mann(site.element, s, charge=site.charge)
            T = site.debye_waller.factor_from_q(qmag)
            acc += site.occupancy * fx * T * _phase(q, site.position)
        F[i] = acc
    return F

def structure_factor_electron(structure: Structure, q_vecs: np.ndarray, beam_energy_keV: float | None = None) -> np.ndarray:
    """Compute electron F(q) (IAM via Mott–Bethe) at each q-vector.

    By default uses non-relativistic Mott–Bethe; pass `beam_energy_keV` to apply a simple relativistic scaling.
    Returns values in ångströms.

    Notes
    -----
    The electron scattering factor diverges like 1/s^2 as s->0. In practice, avoid q=0.
    """
    q_vecs = np.asarray(q_vecs, dtype=float)
    F = np.zeros(q_vecs.shape[0], dtype=complex)
    for i, q in enumerate(q_vecs):
        qmag = float(np.linalg.norm(q))
        s = qmag / (4.0 * np.pi)
        acc = 0.0 + 0.0j
        for site in structure.sites:
            if beam_energy_keV is None:
                fe = fe_mott_bethe(site.element, s)
            else:
                fe = fe_mott_bethe_relativistic(site.element, s, beam_energy_keV=beam_energy_keV)
            T = site.debye_waller.factor_from_q(qmag)
            acc += site.occupancy * fe * T * _phase(q, site.position)
        F[i] = acc
    return F

