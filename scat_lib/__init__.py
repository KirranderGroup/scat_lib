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
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import ci_to_2rdm: {e}")

try:
    from . import fit_utils
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import fit_utils: {e}")

try:
    from . import makerdm
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import makerdm: {e}")

try:
    from . import molecule
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import molecule: {e}")

try:
    from . import molden_reader_nikola_pyscf
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import molden_reader_nikola_pyscf: {e}")

try:
    from . import rdm_tools
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import rdm_tools: {e}")

try:
    from . import reduced_ci
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import reduced_ci: {e}")

try:
    from . import scat_calc
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import scat_calc: {e}")

try:
    from . import iam
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import iam: {e}")

try:
    from . import gas_iam
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import gas_iam: {e}")

__all__ = [
    'ci_to_2rdm',
    'fit_utils', 
    'makerdm',
    'molecule',
    'molden_reader_nikola_pyscf',
    'rdm_tools',
    'reduced_ci',
    'scat_calc',
    'iam',
    'gas_iam',
]
