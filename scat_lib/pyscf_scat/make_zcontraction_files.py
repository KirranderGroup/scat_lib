import numpy as np
import molden_reader_nikola_morder as mldreader
import os
from scat_calc import types

def _make_zcontraction_option(
        atoms,
        geom,
        file_name,
        one_rdm_file,
        two_rdm_file,
        molden_file,
        type='total',
        log_file='scat.log',
        q_range = (1E-10,250),
        q_points = 1000,
        cutoffcentre = 1E-2,
        cutoffz = 1e-20,
        cutoffmd = 1e-20,
        state1 = 1,
        state2 = 1,
        state3 = 1,
        path= None):


    jeremyR = False
    mcci = False
    hf = False
    molpro = False
    molcas = False
    bagel = False
    readtwordm = True
    confs = 0
    civs = 0
    Nmo_max = 100
    civs = np.array(civs)
    

    with open(os.path.join(path, 'options.dat'), 'w') as f:
        f.write(str(np.size(atoms.atomic_numbers())) + '\n')
        for i in atoms.atomic_numbers():
            f.write(str(i) + ' ')
        f.write('\n')
        for i in range(np.size(atoms.atomic_numbers())):
            f.write(str(geom[i, :])[1:-1] + '\n')
        f.write(str(cutoffcentre) + '\n')
        f.write(str(cutoffz) + '\n')
        f.write(str(cutoffmd) + '\n')
        f.write(str(jeremyR) + '\n')
        f.write(str(mcci) + '\n')
        f.write(str(hf) + '\n')
        f.write(str(q_range[0]) + ' ' + str(q_range[1]) + ' ' + str(q_points) + ' \n')
        f.write(str(types[type]) + '\n')
        f.write(str(state1) + ' ' + str(state2) + '\n')
        f.write(file_name + '\n')
        f.write(str(molpro) + '\n')
        f.write(str(molcas) + '\n')
        f.write(str(bagel) + '\n')
        f.write('readtwordm' + '\n')
        f.write(two_rdm_file)


def _make_zcontraction_files(mldfile, path='./'):
    """
    Create files needed for Z-contraction from a Molden file.
    """
    Nmo_max = 600
    gtos, atoms, coeffs, mos, groupC,contr = mldreader.read_orbitals(mldfile, N=Nmo_max, decontract=False)
    xx = gtos.x
    yy = gtos.y
    zz = gtos.z
    l = gtos.l
    m = gtos.m
    n = gtos.n
    ga = gtos.ga
    group = gtos.group

    mmod = np.transpose(gtos.mo)

    l = np.asarray(l)
    m = np.asarray(m)
    n = np.asarray(n)
    ga = np.asarray(ga)
    mmod = np.asarray(mmod, dtype=np.float64)
    with open(os.path.join(path, 'basis.dat'), 'w') as f:
        f.write(str(np.size(l)) + '\n')
        for i in range(np.size(l)):
            f.write(str(xx[i]) + ' ' + str(yy[i]) + ' ' + str(zz[i]) + ' ' + str(ga[i]) + ' ' + str(l[i]) + ' ' + str(
                m[i]) + ' ' + str(n[i]) + ' ' + str(group[i]) + ' '+ str(contr[i])+'\n')
    with open(os.path.join(path, 'MOs.dat'), 'w') as f:
        f.write(str(np.size(mmod[:, 0])) + ' ' + str(np.size(mmod[0, :])) + '\n')
        for i in range(np.size(mmod[:, 0])):
            for j in range(np.size(mmod[0, :])):
                f.write(str(mmod[i, j]) + ' ')
            f.write('\n')
    with open(os.path.join(path, 'MOs2.dat'), 'w') as f:
        f.write(str(np.size(mos[:, 0])) + ' ' + str(np.size(mos[0, :])) + '\n')
        for i in range(np.size(mos[:, 0])):
            for j in range(np.size(mos[0, :])):
                f.write(str(mos[i, j]) + ' ')
            f.write('\n')
    with open(os.path.join(path, 'coeffs.dat'), 'w') as f:
        f.write(str(np.size(l)) + '\n')
        for i in range(np.size(l)):
            f.write(str(coeffs[i]) + '\n')
        f.write(str(np.size(groupC)) + '\n')
        count = 1
        for i in range(np.size(groupC)):
            f.write(str(count) + ' ' + str(count + groupC[i] - 1) + ' ' + str(groupC[i]) + '\n')
            count = count + groupC[i]
    return