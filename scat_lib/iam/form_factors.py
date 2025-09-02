from __future__ import annotations
import os, numpy as np
from dataclasses import dataclass
from typing import Dict, List
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@dataclass
class CromerMannCoeffs:
    a: np.ndarray
    b: np.ndarray
    c: float

class CromerMannTable:
    def __init__(self, path: str | None = None) -> None:
        path = path or os.path.join(_DATA_DIR, "affl.txt")
        self._coeffs: Dict[str, CromerMannCoeffs] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): 
                    continue
                parts = line.split()
                sym = parts[0]
                vals = list(map(float, parts[1:]))
                if len(vals) != 9:
                    raise ValueError(f"Unexpected columns in affl.txt for {sym}: {len(vals)}")
                a = np.array(vals[0:8:2], float)
                b = np.array(vals[1:8:2], float)
                c = float(vals[8])
                self._coeffs[sym] = CromerMannCoeffs(a=a, b=b, c=c)

    def __contains__(self, k: str) -> bool: return k in self._coeffs
    def get(self, k: str) -> CromerMannCoeffs: return self._coeffs[k]
    @property
    def symbols(self) -> List[str]: return list(self._coeffs.keys())

class ZMinusFxTable:
    def __init__(self, path: str | None = None, cm_table: CromerMannTable | None = None) -> None:
        path = path or os.path.join(_DATA_DIR, "isfl.txt")
        self._grid_s = np.array([0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00,1.50,2.00], float)
        self._data: Dict[str, np.ndarray] = {}
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                if not line.strip(): continue
                parts = line.split()
                sym = parts[0]
                vals = np.array(list(map(float, parts[1:])), float)
                self._data[sym] = vals
        self._cm = cm_table or CromerMannTable()

    def __contains__(self, k: str) -> bool: return k in self._data

    def z_minus_fx_interp(self, element: str, s: float) -> float:
        import numpy as np
        if (element in self._data) and (self._grid_s[0] <= s <= self._grid_s[-1]):
            y = self._data[element]
            return float(np.interp(s, self._grid_s, y))
        coeffs = self._cm.get(element)
        fx = float(np.sum(coeffs.a * np.exp(-coeffs.b * (s*s))) + coeffs.c)
        # For neutral fallback, approximate Z as fx(s->0) ~ sum(a) + c
        Z_approx = float(np.sum(coeffs.a) + coeffs.c)
        return float(Z_approx - fx)

def fx_cromer_mann(element: str, s: float, charge: str | None = None, table: CromerMannTable | None = None) -> float:
    table = table or CromerMannTable()
    key = element
    if charge:
        ch_key = f"{element}{charge}"
        if ch_key in table:
            key = ch_key
    c = table.get(key)
    return float(np.sum(c.a * np.exp(-c.b * (s*s))) + c.c)

def z_minus_fx(element: str, s: float, table: ZMinusFxTable | None = None) -> float:
    table = table or ZMinusFxTable()
    return table.z_minus_fx_interp(element, s)
