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
        alpha_index = find_index(alpha.tolist())
        beta_index = find_index(beta.tolist())
        casscf_cis[alpha_index, beta_index] = ci

    if update:
        casscf.ci = casscf_cis
        return casscf_cis
    else:
        return casscf_cis
    



def _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo):
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
