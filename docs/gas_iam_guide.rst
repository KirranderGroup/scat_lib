Gas IAM Guide
=============

The ``scat_lib.gas_iam`` subpackage provides a minimal, fast implementation of gas-phase
independent-atom model (IAM) X-ray scattering that mirrors an existing Fortran-90 code.

Key characteristics
-------------------

- No Debye–Waller factors or damping.
- Uses Cromer–Mann form factors with elemental/ionic labels that must match the bundled table.
- Computes the molecular (total) intensity via the Debye expression with a ``sinc`` kernel.

Quickstart
----------

.. code-block:: python

   import numpy as np
   from scat_lib import gas_iam as gi

   # 1) Load Cromer–Mann coefficients table
   cm = gi.load_cm_table()   # uses bundled data in scat_lib/gas_iam/data

   # 2) Read a simple XYZ geometry (Å) with element labels matching the table
   positions, labels = gi.read_xyz("molecule.xyz")

   # 3) Build a q-grid comparable to the F90 implementation
   q = gi.q_grid_f90(nq=1000, qmin=0.0, qmax=25.0)  # Å^-1

   # 4) Compute the gas-phase molecular intensity I(q)
   Iq = gi.intensity_molecular_xray(positions, labels, q, cm)

   # Optional: components (self vs cross) for reference/debugging
   I_total, I_self, I_cross = gi.intensity_components_xray(positions, labels, q, cm)

API highlights
--------------

- ``gi.load_cm_table(path=None) -> CromerMannTable``: Loads Cromer–Mann parameters.
- ``gi.read_xyz(path) -> (positions[N,3], labels[N])``: Reads XYZ coordinates and element labels.
- ``gi.q_grid_f90(nq=1000, qmin=0.0, qmax=None) -> q[Nq]``: q-grid utility mirroring the F90 workflow.
- ``gi.intensity_molecular_xray(positions, labels, q, cm) -> I(q)``: Total molecular intensity.
- ``gi.intensity_components_xray(...) -> (I_total, I_self, I_cross)``: Split for diagnostic use.

Notes
-----

- Element/ion labels must exactly match the keys in the bundled table (e.g., ``'C'``, ``'Cval'``, ``'Siv'``, ``'O1-'``).
- Units: Positions are in Ångstrom, ``q`` is in Å⁻¹, and the Debye expression uses ``s = q / (4π)``.

See also
--------

- :mod:`scat_lib.iam` for a structured IAM implementation with crystals, Debye–Waller factors, and electron scattering.
