"""Optional provider using the `xraydb` Python package (if installed).

This gives access to energy-dependent corrections f1, f2 and alternative f0 coefficients.
We only import at call-time to avoid a hard dependency.
"""
from __future__ import annotations
from typing import Optional

def xraydb_fx(element: str, s: float) -> Optional[float]:
    """Return f0(s) using xraydb if available, else None."""
    try:
        import xraydb
        print('using xraydb')
    except Exception:
        return None
    # xraydb.f0 returns (f0, res) where res includes source; argument is sin(theta)/lambda
    f0, _ = xraydb.f0(element, s)
    return float(f0)

