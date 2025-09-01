"""Physical constants and basic element tables used by iam_scattering.

Notes
-----
- Units are SI unless stated otherwise.
- For convenience, `BOHR_ANG` is the Bohr radius in ångströms.
- `E_CHARGE` is the elementary charge.
- `ATOMIC_NUMBERS` maps element symbols to atomic numbers (Z).
"""

from __future__ import annotations

# --- Physical constants (CODATA 2018/2019 recommended values) ---
# Values chosen to be stable across Python installs.
E_CHARGE = 1.602176634e-19        # Coulomb (exact)
EPS0     = 8.8541878128e-12       # F/m
H        = 6.62607015e-34         # J*s (exact)
HBAR     = 1.054571817e-34        # J*s
M_E      = 9.1093837015e-31       # kg
C        = 299_792_458.0          # m/s (exact)
A0_M     = 5.29177210903e-11      # m
A0_ANG   = A0_M * 1e10            # Å

# Classical electron radius r_e = e^2/(4 pi eps0 m_e c^2) [m]
import math as _math
R_E = E_CHARGE**2 / (4*_math.pi*EPS0 * M_E * C**2)

# Useful constants for electron scattering factors
# 1 / (8 pi^2 a0) in Å^-1 gives fe(s) (Å) when multiplied by (Z - fx)/s^2
INV_8PI2_A0_ANG = 1.0 / (8.0 * _math.pi**2 * A0_ANG)

# Map of element symbols to atomic numbers. Includes main block + lanthanides/actinides up to Cf.
ATOMIC_NUMBERS = {
    # 1–10
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    # 11–18
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    # 19–36
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28,
    "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    # 37–54
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46,
    "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
    # 55–86
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72,
    "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
    "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
    # 87–98
    "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96,
    "Bk": 97, "Cf": 98,
}

