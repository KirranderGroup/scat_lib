"""
Scattering Library

A Python library for calculating X-ray scattering from ab initio electronic 
structure calculations using PySCF.

It also allows the calculation of scattering using configuration state functions (CSF).
"""

__version__ = "0.1.0"
__author__ = "Patrick Wang"

# Import main modules (with error handling for missing dependencies)
try:
    from . import ci_to_2rdm
    from . import fit_utils
    from . import makerdm
    from . import molecule
    from . import molden_reader_nikola_pyscf
    from . import rdm_tools
    from . import reduced_ci
    from . import scat_calc
    from . import sine_transform
except ImportError as e:
    # Handle missing dependencies gracefully during documentation build
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}")

__all__ = [
    'ci_to_2rdm',
    'fit_utils', 
    'makerdm',
    'molecule',
    'molden_reader_nikola_pyscf',
    'rdm_tools',
    'reduced_ci',
    'scat_calc',
    'sine_transform'
]
