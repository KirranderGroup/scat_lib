"""
Compatibility wrapper for scat_lib.pyscf_scat.scat_calc

This module is kept to preserve backward-compatible imports like
`from scat_lib.scat_calc import ...`. New code should import from
`scat_lib.pyscf_scat.scat_calc`.
"""

from .pyscf_scat.scat_calc import *  # re-export

