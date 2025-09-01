"""Electron scattering factors via the Mott–Bethe relation.

fe(s) = (1 / (8 pi^2 a0)) * (Z - f_x(s)) / s^2      [Å]

We also provide an optional relativistic correction that replaces `a0` by `a0/gamma`, i.e.
multiplies the prefactor by `gamma` (see notes). This is a common pragmatic correction used
in the electron-diffraction literature; see references in the docs.

Parameters
----------
- `s` is sin(theta)/lambda in Å^-1
- If you already have g = 2*s (Å^-1), pass `s = g/2`.

Caveat: as s -> 0, fe diverges like 1/s^2. Practical calculations avoid the singularity by using
a small but finite `s_min` or by using tabulated electron scattering factors specifically fitted
for g -> 0 behaviour. We expose `s_min` in the helper below.
"""
from __future__ import annotations
import math
from .constants import INV_8PI2_A0_ANG, ATOMIC_NUMBERS
from .form_factors import fx_cromer_mann, z_minus_fx

def fe_mott_bethe(element: str, s: float, s_min: float = 1e-3) -> float:
    """Electron scattering factor fe(s) [Å] using the (non-relativistic) Mott–Bethe relation.

    A tiny floor `s_min` avoids divergence at s=0.
    """
    s_eff = max(abs(s), s_min)
    val = INV_8PI2_A0_ANG * (z_minus_fx(element, s_eff)) / (s_eff**2)
    return float(val)

def fe_mott_bethe_relativistic(element: str, s: float, beam_energy_keV: float, s_min: float = 1e-3) -> float:
    """Relativistically corrected fe(s) [Å] via Mott–Bethe.

    Uses a multiplicative factor gamma = 1 + E / (m_e c^2) with E the kinetic energy.
    This effectively replaces a0 -> a0/gamma in the prefactor, so fe is scaled by gamma.

    Parameters
    ----------
    element : str
        Element symbol (neutral only in the tabulated Z-f; ions fall back to Cromer–Mann).
    s : float
        s = sin(theta)/lambda in Å^-1.
    beam_energy_keV : float
        Electron kinetic energy in keV, e.g. 200.0 for 200 keV.
    s_min : float
        Lower bound on s to avoid singularity at s=0.
    """
    # gamma = 1 + E / (m c^2); m c^2 = 511 keV
    gamma = 1.0 + (beam_energy_keV / 511.0)
    return gamma * fe_mott_bethe(element, s, s_min=s_min)

