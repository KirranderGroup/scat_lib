import numpy as np
from pyscf import ci, gto, scf, tools, cc, mcscf, lib, ao2mo, fci

mol = gto.M(
    atom='Be 0 0 0',
    basis='3-21G',
    cart=True,
    max_memory=120000,
    charge=0,
    spin=0,
    verbose=4,
    symmetry=False,
)
mf = scf.HF(mol).run()
tools.molden.dump_scf(mf, 'molden.molden')


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

no_frozen = np.sum(mf.mo_energy < -1e6)
print(no_frozen)

# Note that the line following these comments could be replaced by
#mycc = cc.CCSD(mf)
#mycc.kernel()
#mycc = ci.CISD(mf, frozen=no_frozen).run()
# Ncas, nelec
casscf = mcscf.CASSCF(mf,9,4,ncore=0)
casscf.kernel()
tools.molden.from_mcscf(casscf, 'molden_casscf.molden')        
#print(mycc.ci)
#print('CCSD total energy', mycc.e_tot)
#temp = 1*mycc.ci[0]
#mycc.ci[0] = mycc.ci[1]
#mycc.ci[1] = temp

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

with open('1rdm_pyscf_CASCI.txt', 'w') as f:
    for i in range(no_mos):
        for j in range(no_mos):
            if np.abs(dm1[i, j]) > pthresh:
                f.write(f"{i+1: 3d}  {j+1: 3d}  {dm1[i, j]}\n")

with open('2rdm_pyscf_CASCI.txt', 'w') as f:
    for i in range(no_mos):
        for j in range(no_mos):
            for k in range(no_mos):
                for l in range(no_mos):
                    if np.abs(dm2[i, j, k, l]) > pthresh:
                        f.write(
                            f"{i+1: 3d}  {j+1: 3d}  {k+1: 3d}  {l+1: 3d}  {dm2[i, j, k, l]}\n"
                        )

occslst = fci.cistring.gen_occslst(range(ncas), 4//2)

with open('321gci_CASCI.txt','w') as f:
    f.write('# det_alpha\tdet_beta\tCI Coeffs\n')
    for i, occs_alpha in enumerate(occslst.tolist()):
        for j, occs_beta in enumerate(occslst.tolist()):
            f.write('%s\t%s\t%.12f\n' % (occs_alpha, occs_beta, casscf.ci[i,j]))
nelec = (2, 2)

with open('321gci_large_CASCI.txt', 'w') as f:
    f.write('# det_alpha,   det_beta,   CI Coeffs\n')
    for c, ia, ib in casscf.fcisolver.large_ci(casscf.ci, ncas, nelec, tol=5E-5, return_strs=False):
        f.write('%s\t%s\t%.12f\n' % (ia.tolist(), ib.tolist(), c))


