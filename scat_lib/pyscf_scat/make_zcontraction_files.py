import numpy as np
import molden_reader_nikola_morder as mldreader
import os



def _make_zcontraction_files(mldfile, path='./'):
    """
    Create files needed for Z-contraction from a Molden file.
    """
    Nmo_max = 600
    gtos, _, coeffs, mos, groupC,contr = mldreader.read_orbitals(mldfile, N=Nmo_max, decontract=False)
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