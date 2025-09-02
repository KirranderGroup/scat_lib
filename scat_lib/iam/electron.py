from __future__ import annotations
from .constants import INV_8PI2_A0_ANG
from .form_factors import z_minus_fx
def fe_mott_bethe(element: str, s: float, s_min: float = 1e-3) -> float:
    s_eff = max(abs(s), s_min)
    return float(INV_8PI2_A0_ANG * (z_minus_fx(element, s_eff)) / (s_eff*s_eff))
def fe_mott_bethe_relativistic(element: str, s: float, beam_energy_keV: float, s_min: float = 1e-3) -> float:
    gamma = 1.0 + (beam_energy_keV/511.0)
    return gamma * fe_mott_bethe(element, s, s_min=s_min)
