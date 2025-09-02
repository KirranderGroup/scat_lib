"""gas_iam_f90 — Gas-phase IAM scattering (X-ray) mirroring the user's F90 code.

Core API
--------
- load_cm_table(path=None) -> CromerMannTable
- read_xyz(path) -> (positions[N,3], labels[N])
- q_grid_f90(nq=1000, qmin=0.0, qmax=None) -> q[Nq]
- intensity_molecular_xray(positions, labels, q, cm=None) -> I(q)

Notes:
- This reproduces the F90 output: the **molecular/interference** signal only,
  i.e. I(q) = sum_{i,j} f_i(s) f_j(s) sinc(q r_ij). No Debye–Waller or damping.
- Element labels must match the affl.txt keys exactly (e.g., 'C', 'Cval', 'Siv', 'O1-').
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
