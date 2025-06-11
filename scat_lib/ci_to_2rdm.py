"""
ci_to_2rdm.py

This file generates the 2-RDM from a CI wavefunction using PySCF.

Patrick Wang
patrick.wang@chem.ox.ac.uk

2025-03-11
"""

import numpy as np
from pyscf import gto, scf, mcscf, fci, ci



def write_ci_file(file_name, casscf,nelec, ncas, tol=5E-5):
    """
    Writes CI (Configuration Interaction) coefficients to text files for a given CASSCF calculation.

    This function generates two files:
        1. '{file_name}_ci.txt': Contains all CI coefficients for all possible alpha and beta determinants.
        2. '{file_name}_ci_large.txt': Contains CI coefficients larger than a specified tolerance.

    Parameters
    ----------
    file_name : str
        The base name for the output files (without extension).
    casscf : object
        The CASSCF object containing the CI coefficients and FCISolver.
    nelec : tuple or int
        Number of electrons (alpha, beta) in the active space.
    ncas : int
        Number of active orbitals.
    tol : float, optional
        Tolerance for selecting large CI coefficients (default is 5E-5).

    Notes
    -----
    - The function assumes the presence of `casscf.ci` (CI coefficient matrix) and
        `casscf.fcisolver.large_ci` (method to extract large CI coefficients).
    - The output files list determinants as lists of occupied orbitals for alpha and beta spins.
    """
    nelec = (2, 2)
    occslst = fci.cistring.gen_occslst(range(ncas), 4//2)

    with open(f'{file_name}_ci.txt','w') as f:
        f.write('# det_alpha\tdet_beta\tCI Coeffs\n')
        for i, occs_alpha in enumerate(occslst.tolist()):
            for j, occs_beta in enumerate(occslst.tolist()):
                f.write('%s\t%s\t%.12f\n' % (occs_alpha, occs_beta, casscf.ci[i,j]))

    with open(f'{file_name}_ci_large.txt', 'w') as f:
        f.write('# det_alpha,   det_beta,   CI Coeffs\n')
        for c, ia, ib in casscf.fcisolver.large_ci(casscf.ci, ncas, nelec, tol=tol, return_strs=False):
            f.write('%s\t%s\t%.12f\n' % (ia.tolist(), ib.tolist(), c))



def read_ci_file(file, sort_by_ci = False):
    '''
    Parse the CI coefficients from a file

    Parameters
    ----------
    file : str
        File containing CI coefficients, space delimited
    kwarg sort_by_ci : bool
        Sort the arrays returned by magnitude of CI coefficient
    Returns
    -------
    det_alpha : np.array
        Alpha determinants
    det_beta : np.array
        Beta determinants
    ci : np.array
        CI coefficients
    '''
    with open(file, 'r') as f:
        lines = f.readlines()
    
    alpha_det = []
    beta_det = []
    ci_coeffs = []

    for line in lines[1:]:
        parts = line.split('\t')
        alpha = eval(parts[0])
        beta = eval(parts[1])
        ci = float(parts[2])

        alpha_det.append(alpha)
        beta_det.append(beta)
        ci_coeffs.append(ci)
    
    alpha_det = np.array(alpha_det)
    beta_det = np.array(beta_det)
    ci_coeffs = np.array(ci_coeffs)

    if sort_by_ci:
        ci_ind = ci_coeffs.argsort()
        ci_coeffs = ci_coeffs[ci_ind[::-1]]
        alpha_det = alpha_det[ci_ind[::-1]]
        beta_det = beta_det[ci_ind[::-1]]
    
    else:
        pass

    return alpha_det, beta_det, ci_coeffs

def update_ci_coeffs(alpha_det:np.array, 
                     beta_det: np.array, 
                     ci_coeff:np.array, 
                     casscf:mcscf.CASSCF, 
                     update=True):
    '''
    Updates the CI coefficients given a set of new coefficients
    and their corresponding cofigurations.

    Parameters
    ----------
    alpha_det : numpy.array 
        Alpha determinants
    beta_det : numpy.array
        beta determinants
    ci_coeff : numpy.array
        CI coefficients
    casscf : mcscf.CASSCF
        CASSCF instance
    update : bool
        Defaults to true and overwrite the CI array
    '''
    occ_list = fci.cistring.gen_occslst(range(casscf.ncas), (casscf.nelecas[0]+casscf.nelecas[1]) // 2).tolist()
    find_index = lambda det : occ_list.index(det)
    
    casscf_cis = np.zeros_like(casscf.ci)
    
    for alpha, beta, ci in zip(alpha_det, beta_det, ci_coeff):
        if type(alpha) == list:
            alpha_index = find_index(alpha)
            beta_index = find_index(beta)
        else:
            alpha_index = find_index(alpha.tolist())
            beta_index = find_index(beta.tolist())
        casscf_cis[alpha_index, beta_index] = ci

    if update:
        casscf.ci = casscf_cis
        return casscf_cis
    else:
        return casscf_cis
    



def _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo):
    """
    Constructs the one- and two-particle reduced density matrices (RDM1 and RDM2) in the molecular orbital (MO) basis.
    Parameters
    ----------
    casdm1 : np.ndarray
        The active-space (CAS) one-particle reduced density matrix of shape (ncas, ncas).
    casdm2 : np.ndarray
        The active-space (CAS) two-particle reduced density matrix of shape (ncas, ncas, ncas, ncas).
    ncore : int
        Number of core (fully occupied) orbitals.
    ncas : int
        Number of active (CAS) orbitals.
    nmo : int
        Total number of molecular orbitals.
    Returns
    -------
    dm1 : np.ndarray
        The one-particle RDM in the MO basis, shape (nmo, nmo).
    dm2 : np.ndarray
        The two-particle RDM in the MO basis, shape (nmo, nmo, nmo, nmo).
    Notes
    -----
    - The function embeds the CAS RDMs into the full MO space, filling in the core contributions.
    - Core orbitals are assumed to be doubly occupied.
    """

    nocc = ncas + ncore
    dm1 = np.zeros((nmo,nmo))
    idx = np.arange(ncore)
    dm1[idx,idx] = 2
    dm1[ncore:nocc,ncore:nocc] = casdm1

    dm2 = np.zeros((nmo,nmo,nmo,nmo))
    dm2[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc] = casdm2
    for i in range(ncore):
        for j in range(ncore):
            dm2[i,i,j,j] += 4
            dm2[i,j,j,i] += -2
        dm2[i,i,ncore:nocc,ncore:nocc] = dm2[ncore:nocc,ncore:nocc,i,i] =2*casdm1
        dm2[i,ncore:nocc,ncore:nocc,i] = dm2[ncore:nocc,i,i,ncore:nocc] = -casdm1
    return dm1, dm2



def get_dms(casscf, state=0):
    """
    Compute the one- and two-particle reduced density matrices (RDMs) for a CASCI/CASSCF calculation.
    Parameters
    ----------
    casscf : object
        A CASSCF or CASCI object containing the wavefunction and molecular orbital information.
    state : int, optional
        The state index for which to compute the density matrices (default is 0).
    Returns
    -------
    dm1 : numpy.ndarray
        The one-particle reduced density matrix in the molecular orbital basis.
    dm2 : numpy.ndarray
        The two-particle reduced density matrix in the molecular orbital basis.
    Notes
    -----
    This function extracts the CI vector and molecular orbital coefficients from the provided
    CASSCF/CASCI object, computes the active-space RDMs using the FCI solver, and then
    transforms them to the full molecular orbital basis.
    """

    # calculates the dms for the CASCI calculation
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    ci = casscf.ci
    mo_coeff = casscf.mo_coeff
    nmo = mo_coeff.shape[1]
    casdm1, casdm2 = casscf.fcisolver.make_rdm12(ci, ncas, nelecas)
    dm1, dm2 = _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo)

    return dm1, dm2




def write_rdms(file,casscf,mf):
    """
    This function extracts the active space reduced density matrices from a CASSCF calculation,

    Parameters
    ----------
    file : str
        Base filename for output files. The function will write to '1rdm_{file}.txt' and '2rdm_{file}.txt'.
    casscf : object
        A CASSCF object containing the wavefunction, CI coefficients, and orbital information.
    mf : object
        A mean-field object (e.g., from PySCF) providing molecular orbital energies and coefficients.

    Outputs
    -------
    1rdm_{file}.txt : file
        Contains the non-zero elements of the 1-RDM in the MO basis.
    2rdm_{file}.txt : file
        Contains the non-zero elements of the 2-RDM in the MO basis.
    """


    no_frozen = np.sum(mf.mo_energy < -1e6)
    print(no_frozen)
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore=casscf.ncore
    ci = casscf.ci
    mo_coeff=casscf.mo_coeff
    nmo=mo_coeff.shape[1]

    casdm1, casdm2 = casscf.fcisolver.make_rdm12(ci,ncas,nelecas)
    dm1,dm2 = _make_rdm12_on_mo(casdm1, casdm2, ncore,ncas,nmo)

    no_mos = dm1.shape[0]

    pthresh = 1e-17

    with open(f'1rdm_{file}.txt', 'w') as f:
        for i in range(no_mos):
            for j in range(no_mos):
                if np.abs(dm1[i, j]) > pthresh:
                    f.write(f"{i+1: 3d}  {j+1: 3d}  {dm1[i, j]}\n")

    with open(f'2rdm_{file}.txt', 'w') as f:
        for i in range(no_mos):
            for j in range(no_mos):
                for k in range(no_mos):
                    for l in range(no_mos):
                        if np.abs(dm2[i, j, k, l]) > pthresh:
                            f.write(
                                f"{i+1: 3d}  {j+1: 3d}  {k+1: 3d}  {l+1: 3d}  {dm2[i, j, k, l]}\n"
                            )



def calc_energy(casscf, mol, ci_modified):
    assert np.isclose(np.linalg.norm(ci_modified), 1), 'Wavefunction is not normalised!'
    h1_eff, e_core = casscf.get_h1eff()       # one-electron part (active space) and core energy
    eri_cas = casscf.get_h2eff()             # two-electron integrals in active space (MO basis)
    # Compute energy of the modified CI vector with these integrals
    E_cas = fci.direct_spin1.energy(h1_eff, eri_cas, ci_modified, casscf.ncas, casscf.nelecas)  # electronic CAS energy
    E_total = E_cas + e_core + mol.energy_nuc()  # add core and nuclear repulsion energy
    return E_total



def contract_ci(alpha,beta,ci):
    '''
    Contract the CI coefficients to get the 2-RDM

    Parameters
    ----------
    alpha : np.array
        Alpha determinants
    beta : np.array
        Beta determinants
    ci : np.array
        CI coefficients

    Returns
    -------
    '''
