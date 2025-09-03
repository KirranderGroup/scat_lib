from __future__ import annotations
from typing import Optional, Mapping
import re

def _normalize_label(label: str, ion_map: Optional[Mapping[str, str]] = None) -> str:
    """Normalize label to a form xraydb understands. 
    - Applies user-supplied ion_map first.
    - Accepts 'Fe2+' / 'O1-' style ionic labels.
    - Falls back to bare element symbol if parsing succeeds but charge is unsupported.
    """
    if ion_map and label in ion_map:
        return ion_map[label]
    # direct acceptance (Fe, Fe2+, O1-, etc.)
    if re.match(r'^[A-Z][a-z]?([0-9]+[+-])?$', label):
        return label
    # Try to strip 'val'/'v' suffixes (e.g., Cval, Siv, Sival) => base element (C, Si)
    m = re.match(r'^([A-Z][a-z]?)(?:val|v|valence)?$', label)
    if m:
        return m.group(1)
    # default: return as-is to let xraydb error out
    return label

def fx_xraydb(symbol: str, s: float, *, ion_map: Optional[Mapping[str, str]] = None) -> float:
    """Return f0(s) using xraydb (Waasmaier–Kirfel). Expects s = sin(theta)/lambda [Å^-1].
    Requires 'pip install xraydb'. Raises ImportError if missing.
    """
    try:
        import xraydb
    except Exception as exc:
        raise ImportError("xraydb is not installed. Install with: pip install xraydb") from exc
    sym = _normalize_label(symbol, ion_map=ion_map)
    return float(xraydb.f0(sym, s))
