from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@dataclass(frozen=True)
class CMCoeffs:
    a: np.ndarray  # (4,)
    b: np.ndarray  # (4,)
    c: float

class CromerMannTable:
    """Table of Cromer–Mann coefficients loaded from affl.txt (4 Gaussians + constant)."""
    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(DATA_DIR, "affl.txt")
        self._d: Dict[str, CMCoeffs] = {}
        with open(self.path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split()
                sym = parts[0]
                vals = [float(x) for x in parts[1:]]
                if len(vals) != 9:
                    raise ValueError(f"affl.txt line for {sym} has {len(vals)} numeric fields; expected 9")
                a = np.array(vals[0:8:2], dtype=float)
                b = np.array(vals[1:8:2], dtype=float)
                c = float(vals[8])
                self._d[sym] = CMCoeffs(a=a, b=b, c=c)

    def __contains__(self, k: str) -> bool:
        return k in self._d

    def get(self, k: str) -> CMCoeffs:
        try:
            return self._d[k]
        except KeyError:
            keys = ", ".join(sorted(self._d.keys())[:10]) + ("..." if len(self._d) > 10 else "")
            raise KeyError(f"Element label '{k}' not found in affl table at {self.path}. Example keys: {keys}")

    @property
    def keys(self) -> List[str]:
        return list(self._d.keys())

def load_cm_table(path: str | None = None) -> CromerMannTable:
    return CromerMannTable(path)

def fx_cromer_mann(symbol: str, s: float, table: CromerMannTable) -> float:
    """Evaluate f_x(s) from CM coefficients for `symbol` (e.g., 'C', 'Cval', 'Si', 'Siv', 'O1-').
    s is sin(theta)/lambda in Å^-1.
    """
    coeffs = table.get(symbol)
    # f(s) = sum_i a_i * exp(-b_i s^2) + c
    return float(np.sum(coeffs.a * np.exp(-coeffs.b * (s*s))) + coeffs.c)
