"""Atomic form factors: X-ray (Cromer–Mann) and Z - f_x(s) tables.

This module provides:

- `CromerMannTable`: loads Cromer–Mann coefficients (4 Gaussians + constant) from an embedded file.
- `ZMinusFxTable`: loads a tabulated grid of (Z - f_x) vs s for neutral atoms.
- `fx_cromer_mann(element, s, charge=None)`: compute f_x(s) using Cromer–Mann coefficients.
- `z_minus_fx(element, s)`: compute Z - f_x(s), either from the tabulated grid (with interpolation)
  when in bounds 0.10 <= s <= 2.00 Å^-1 and the element is neutral, or fall back to Z - f_x from CM.

Data sources
------------
- Cromer–Mann coefficients file `data/affl.txt` (as provided by the user; typical format is 4 Gaussians + c)
- Tabulated `data/isfl.txt` grid with columns s = 0.10, 0.20, ..., 2.00 Å^-1

"""
from __future__ import annotations
import os, math
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
from .constants import ATOMIC_NUMBERS

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@dataclass
class CromerMannCoeffs:
    a: np.ndarray  # shape (4,)
    b: np.ndarray  # shape (4,)
    c: float

class CromerMannTable:
    """Load and serve Cromer–Mann coefficients from affl.txt."""
    def __init__(self, path: str | None = None) -> None:
        path = path or os.path.join(_DATA_DIR, "affl.txt")
        self._coeffs: Dict[str, CromerMannCoeffs] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): 
                    continue
                parts = line.split()
                symbol = parts[0]
                vals = list(map(float, parts[1:]))
                if len(vals) != 9:
                    # Format is expected to be 8 numbers for (a1,b1,...,a4,b4) + c
                    raise ValueError(f"Unexpected number of columns in affl.txt line for {symbol}: {len(vals)}")
                a = np.array(vals[0:8:2], dtype=float)
                b = np.array(vals[1:8:2], dtype=float)
                c = float(vals[8])
                self._coeffs[symbol] = CromerMannCoeffs(a=a, b=b, c=c)

    def __contains__(self, key: str) -> bool:
        return key in self._coeffs

    def get(self, key: str) -> CromerMannCoeffs:
        return self._coeffs[key]

    @property
    def symbols(self) -> List[str]:
        return list(self._coeffs.keys())

class ZMinusFxTable:
    """Tabulated grid of (Z - f_x) at s = 0.10, 0.20, ..., 2.00 Å^-1 for neutral atoms.

    Provides linear interpolation across the grid. For s outside [0.10, 2.00], falls back to
    `Z - f_x` via Cromer–Mann coefficients.
    """
    def __init__(self, path: str | None = None, cm_table: CromerMannTable | None = None) -> None:
        path = path or os.path.join(_DATA_DIR, "isfl.txt")
        self._grid_s = np.array([
            0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00,1.50,2.00
        ], dtype=float)
        self._data: Dict[str, np.ndarray] = {}
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline()  # "Element 0.10 0.20 ..."
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                sym = parts[0]
                vals = np.array(list(map(float, parts[1:])), dtype=float)
                if vals.shape[0] != self._grid_s.shape[0]:
                    raise ValueError(f"isfl row for {sym} has {vals.shape[0]} columns, expected {self._grid_s.shape[0]}")
                self._data[sym] = vals
        self._cm = cm_table or CromerMannTable()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def z_minus_fx_interp(self, element: str, s: float) -> float:
        """Interpolate (Z - f_x) for a neutral element symbol at a given s.

        If element not in the table or s out of bounds, fall back to CM evaluation.
        """
        if (element in self._data) and (self._grid_s[0] <= s <= self._grid_s[-1]):
            y = self._data[element]
            return float(np.interp(s, self._grid_s, y))
        # fallback
        Z = ATOMIC_NUMBERS.get(element)
        if Z is None:
            raise KeyError(f"Unknown element symbol: {element}")
        coeffs = self._cm.get(element)
        fx = np.sum(coeffs.a * np.exp(-coeffs.b * (s**2))) + coeffs.c
        return float(Z - fx)

def fx_cromer_mann(element: str, s: float, charge: str | None = None, table: CromerMannTable | None = None) -> float:
    """Compute X-ray normal scattering factor f_x(s) via Cromer–Mann.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'C', 'Ni'). For ions, pass the charged label present in the table
        (e.g., 'O1-', 'Fe2+', 'Si4+'). If `charge` is given, the function tries `f"{element}{charge}"` first.
    s : float
        s = sin(theta)/lambda in Å^-1.
    charge : str, optional
        Charge suffix like '1-' or '3+'; only if present in the CM table.
    table : CromerMannTable, optional
        Preloaded table to avoid re-reading the file.
    """
    table = table or CromerMannTable()
    key = element
    if charge:
        ch_key = f"{element}{charge}"
        if ch_key in table:
            key = ch_key
    coeffs = table.get(key)
    return float(np.sum(coeffs.a * np.exp(-coeffs.b * (s**2))) + coeffs.c)

def z_minus_fx(element: str, s: float, table: ZMinusFxTable | None = None) -> float:
    """Return (Z - f_x(s)). Uses tabulated grid where applicable, or Cromer–Mann otherwise."""
    table = table or ZMinusFxTable()
    return table.z_minus_fx_interp(element, s)

