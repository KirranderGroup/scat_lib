import numpy as np
from pyscf import gto, scf, mcscf, tools
from pyscf.scf import addons
from . import mo2ao


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


def make_rdm2_on_ROHF(mf, mol, separate_components=False):
    '''
    Makes the 2-RDM for a ROHF calculation in AO basis.

    Parameters
    ----------
    mf : pyscf.scf.rohf.ROHF
        The ROHF mean-field object.
    mol : pyscf.gto.Mole
        The PySCF molecule object.
    Returns
    -------
    dm2 : np.ndarray
        The 2-RDM in AO basis.
    '''
    uhf = addons.convert_to_uhf(mf)
    occ_a, occ_b = uhf.mo_occ
    na = np.asarray(occ_a, dtype=float)
    nb = np.asarray(occ_b, dtype=float)
    n = na + nb
    nmo = n.size
    Gamma = np.zeros((nmo, nmo, nmo, nmo))
    Na = np.diag(na)
    Nb = np.diag(nb)
    N = np.diag(n)
    Gamma_el = (np.einsum('ik,jl->ijkl', N, N, optimize=True))
    Gamma_inel = (-np.einsum('il,jk->ijkl', Na, Na, optimize=True) - np.einsum('il,jk->ijkl', Nb, Nb, optimize=True))
    Gamma_tot = Gamma_el + Gamma_inel

    if separate_components:
        Gamma_el = np.transpose(Gamma_el, (0, 2, 1, 3)) # to chemists notation
        Gamma_inel = np.transpose(Gamma_inel, (0, 2, 1, 3)) # to chemists notation
        dm_el = mo2ao.create_Zcotr(mf, mol, Gamma_el)
        dm_inel = mo2ao.create_Zcotr(mf, mol, Gamma_inel)
        return dm_el, dm_inel

    else:
        Gamma_tot = np.transpose(Gamma_tot, (0, 2, 1, 3)) # to chemists notation
        dm = mo2ao.create_Zcotr(mf, mol, Gamma_tot)
        return dm



