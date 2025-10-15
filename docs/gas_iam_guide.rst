Gas IAM Guide
=============

The ``scat_lib.gas_iam`` subpackage provides a minimal, fast implementation of gas-phase
independent-atom model (IAM) X-ray scattering.

Key characteristics
-------------------

- No Debye–Waller factors or damping.
- Uses Cromer–Mann form factors with elemental/ionic labels that must match the bundled table.
- Computes the molecular (total) intensity via the Debye expression with a ``sinc`` kernel.

Quickstart
----------

.. code-block:: python

   import numpy as np
   from scat_lib.gas_iam import intensity_molecular_xray, intensity_pyscf
   from pyscf import gto
   
   mol = gto.M(atom = 'Li 0 0 0; F 0 0 1.5639', basis = 'sto-3g')
   q = np.linspace(0, 10, 100)  # q in Ang
   Iq = intensity_pyscf(mol, q)  # I(q) in arbitrary units


One can also specify the positions and labels directly with the intensity_molecular_xray method.


Notes
-----

- Element/ion labels must exactly match the keys in the bundled table (e.g., ``'C'``, ``'Cval'``, ``'Siv'``, ``'O1-'``).
- Units: Positions are in Ångstrom, ``q`` is in inverse Å, and the Debye expression uses ``s = q / (4π)``.
- Pass ``inelastic=True`` (or ``'table'`` / ``'xraydb'``) to ``intensity_molecular_xray`` /
  ``intensity_components_xray`` to obtain the incoherent contribution alongside the coherent terms.

See also
--------

- :mod:`scat_lib.iam` for a structured IAM implementation with crystals, Debye–Waller factors, and electron scattering.
