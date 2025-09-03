"""gas_iam_f90 — Gas-phase IAM scattering (X-ray). Now with optional xraydb backend.

Backends for f0(s):
- 'affl' (default): bundled Cromer–Mann coefficients (affl.txt).
- 'xraydb': use xraydb.f0(ion, s) (Waasmaier–Kirfel).

Usage:
    Iq = intensity_molecular_xray(pos, labels, q, backend='xraydb')
"""

from .constants import A0_ANG, PI
from .cm import CromerMannTable, load_cm_table, fx_cromer_mann
from .geometry import read_xyz, pair_distance_matrix
from .qgrid import q_grid_f90
from .scattering import intensity_molecular_xray, intensity_components_xray

__all__ = [
    "A0_ANG", "PI",
    "CromerMannTable", "load_cm_table", "fx_cromer_mann",
    "read_xyz", "pair_distance_matrix",
    "q_grid_f90",
    "intensity_molecular_xray", "intensity_components_xray",
]
