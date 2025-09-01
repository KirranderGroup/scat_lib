import numpy as np

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

def save_rdm(casscf, name):

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

    with open(f'1rdm_{name}.txt', 'w') as f:
        for i in range(no_mos):
            for j in range(no_mos):
                if np.abs(dm1[i, j]) > pthresh:
                    f.write(f"{i+1: 3d}  {j+1: 3d}  {dm1[i, j]}\n")

    with open(f'2rdm_{name}.txt', 'w') as f:
        for i in range(no_mos):
            for j in range(no_mos):
                for k in range(no_mos):
                    for l in range(no_mos):
                        if np.abs(dm2[i, j, k, l]) > pthresh:
                            f.write(
                                f"{i+1: 3d}  {j+1: 3d}  {k+1: 3d}  {l+1: 3d}  {dm2[i, j, k, l]}\n"
                            )

    # print('done saving!')

def update_options(name):
    with open('options.dat', 'w') as f:
        f.write(f'''
1
4
0. 0. 0.
0.01
1e-20
1e-20
False
False
False
1e-10 250.0 1000
1
1 1 1
total_{name}.dat
False
False
False
readrdm
1rdm_{name}.txt
2rdm_{name}.txt
''')
