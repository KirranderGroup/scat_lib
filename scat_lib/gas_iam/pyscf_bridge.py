from __future__ import annotations
from typing import Tuple, List, TYPE_CHECKING
import numpy as np

from .geometry import parse_xyz_string

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from pyscf import gto

def positions_and_labels_from_mole(mol: "gto.Mole") -> Tuple[np.ndarray, List[str]]:
    """Return positions [Å] and element labels from a PySCF gto.Mole object."""
    if mol is None:
        raise TypeError("Expected a pyscf.gto.Mole instance, got None")
    try:
        xyz = mol.tostring(format="xyz")
    except AttributeError as exc:
        raise TypeError("Expected a pyscf.gto.Mole instance with tostring(format='xyz')") from exc
    if not isinstance(xyz, str):
        raise ValueError("mol.tostring(format='xyz') must return an XYZ string")
    return parse_xyz_string(xyz)

__all__ = ["positions_and_labels_from_mole"]
