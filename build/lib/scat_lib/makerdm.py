import numpy as np
from pyscf import gto, scf, mcscf, tools

def get_dms(casscf, state=0):
    """
    Calculate the 1- and 2-RDMs for a CASCI calculation.

    Parameters
    ----------
    casscf : pyscf.mcscf.casci.CASCI
        The CASCI object.    
    state : int (optional)
        The state for which to calculate the RDMs. Default is 0.    
    """
    # calculates the dms for the CASCI calculation
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    print(ncore)
#    if casscf.nstates > 1:
#        ci = casscf.ci[state]
#    else:
    ci = casscf.ci
    mo_coeff = casscf.mo_coeff
    nmo = mo_coeff.shape[1]
    casdm1, casdm2 = casscf.fcisolver.make_rdm12(ci, ncas, nelecas)
    dm1, dm2 = _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo)

    return dm1, dm2


def _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo):
    '''
    Transform the 1- and 2-RDMs from the active space to the MO basis.

    Parameters
    ----------
    casdm1 : np.ndarray
        The 1-RDM in the active space.
    casdm2 : np.ndarray
        The 2-RDM in the active space.
    ncore : int
        The number of core orbitals.
    ncas : int
        The number of active orbitals.
    nmo : int
        The number of molecular orbitals.
    
    Returns
    -------
    dm1 : np.ndarray
        The 1-RDM in the MO basis.
    dm2 : np.ndarray
        The 2-RDM in the MO basis.
    '''
    # script to add the frozen section to density matrices
    nocc = ncas + ncore
    dm1 = np.zeros((nmo, nmo))
    idx = np.arange(ncore)
    dm1[idx, idx] = 2
    dm1[ncore:nocc, ncore:nocc] = casdm1

    dm2 = np.zeros((nmo, nmo, nmo, nmo))
    dm2[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc] = casdm2
    for i in range(ncore):
        for j in range(ncore):
            dm2[i, i, j, j] += 4
            dm2[i, j, j, i] += -2
        dm2[i, i, ncore:nocc, ncore:nocc] = dm2[ncore:nocc, ncore:nocc, i,
                                                i] = 2 * casdm1
        dm2[i, ncore:nocc, ncore:nocc, i] = dm2[ncore:nocc, i, i,
                                                ncore:nocc] = -casdm1
    return dm1, dm2