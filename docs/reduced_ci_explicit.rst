scat\_lib.reduced\_ci module
============================

.. currentmodule:: scat_lib.reduced_ci

This module provides classes and functions for performing reduced CASSCF calculations 
with the ability to write CI coefficients and CSF strings to files.

Classes
-------

.. autoclass:: ReducedCASSCF
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. automethod:: __init__
   .. automethod:: write_ci
   .. automethod:: write_csf
   .. automethod:: get_reduced_csf_params
   .. automethod:: calc_etot
   .. automethod:: update_ci
   .. automethod:: run_scattering
   .. automethod:: run_scattering_csf
   .. automethod:: generate_comparison_plot

Properties
----------

.. autoattribute:: ReducedCASSCF.nparams
.. autoattribute:: ReducedCASSCF.ncsf
.. autoattribute:: ReducedCASSCF.csf
.. autoattribute:: ReducedCASSCF.transformer
.. autoattribute:: ReducedCASSCF.reduced_csf
.. autoattribute:: ReducedCASSCF.reduced_csf_inverse