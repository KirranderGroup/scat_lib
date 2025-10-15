import os
import sys
import subprocess
from pyscf import gto, mcscf, scf, fci, ci, tools
sys.path.append('./')
from . import molden_reader_nikola_pyscf as pymldreader
import numpy as np
from copy import deepcopy 
scat_dir = '/u/ajmk/sann8252/PyXSCAT_Patrick/src'
if scat_dir not in sys.path:
    sys.path.append(scat_dir)

from . import ci_to_2rdm
from pyscf.csf_fci import CSFTransformer
from . import makerdm 



types = {'total': '1', 
         'elastic':'2',
         'total_aligned': '3',
         'elastic_aligned' : '4',
         'total_electron' : '5',
         'elastic_electron' : '6',
         'total_j2' : '7',
         'elastic_j2' :'8',
         'resolved_cms' : '9'}

def prepare_files(
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
    """
    Prepares the files needed to run scattering.

    Parameters
    ----------
    file_name : str
        The output scattering file name
    one_rdm_file : str
        Path to the 1rdm file
    two_rdm_file : str
        Path to the 2rdm file
    molden_file : str
        Path to the Molden File for orbitals
    
    **kwargs
    type : str, (total, elastic)
        type of scattering to be computed, defaults total
    log_file : str
        Path to the log file for scattering calculation
    
    Returns
    -------
    q : array_like
        An array of q wave vector values, in a.u.
    intensity : array_like
        An array of intensity values at the corresponding q
    """
    global types

    gtos, atoms = pymldreader.read_orbitals(molden_file, N=100, decontract=True)
    geom = atoms.geometry()
    
    # if path is not None:
    #     os.chdir(path)
    # if not os.path.exists('options.dat'):
    #     subprocess.run(['rm', 'options.dat'])
    # if not os.path.exists('basis.dat'):
    #     subprocess.run(['rm', 'basis.dat'])
    # if not os.path.exists('MOs.dat'):
    #     subprocess.run(['rm', 'MOs.dat'])
    # if not os.path.exists('1rdm_' + file_name + '.txt'):
    #     subprocess.run(['rm', '1rdm_' + file_name + '.txt'])
    # if not os.path.exists('2rdm_' + file_name + '.txt'):
    #     subprocess.run(['rm', '2rdm_' + file_name + '.txt'])
    # if not os.path.exists('2rdm.txt'):
    #     subprocess.run(['rm', '2rdm.txt'])
    # if not os.path.exists(file_name + '.molden'):
    #     subprocess.run(['rm', file_name + '.molden'])
    # if not os.path.exists(file_name):
    #     subprocess.run(['rm', file_name])
    
    with open('options.dat', 'w') as f:
        f.write(str(np.size(atoms.atomic_numbers())) + '\n')
        for i in atoms.atomic_numbers():
            f.write(str(i) + ' ')
        f.write('\n')
        for i in range(np.size(atoms.atomic_numbers())):
            f.write(str(geom[i, :])[1:-1] + '\n')
        f.write(str(cutoffcentre) + '\n')
        f.write(str(cutoffz) + '\n')
        f.write(str(cutoffmd) + '\n')
        f.write('False' + '\n') # JeremyR
        f.write('False' + '\n') # MCCI Boolean
        f.write('False' + '\n') # HF Boolean
        f.write(str(q_range[0]) + ' ' + str(q_range[1]) + ' ' + str(q_points) + ' \n')
        f.write(str(types[type]) + '\n')
        f.write(str(state1) + ' ' + str(state2) + ' ' + str(state3) + '\n')
        f.write(file_name + '\n')
        f.write('False' + '\n') #Molpro Bool Flag
        f.write('False' + '\n') #Molcas Bool Flag
        f.write('False' + '\n') #Bagel Bool Flag

        f.write('readrdm' + '\n')
        f.write(one_rdm_file + '\n')
        f.write(two_rdm_file)

    #print(geom)
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

    with open('basis.dat', 'w') as f:
        f.write(str(np.size(l)) + '\n')
        for i in range(np.size(l)):
            f.write(str(xx[i]) + ' ' + str(yy[i]) + ' ' + str(zz[i]) + ' ' + str(ga[i]) + ' ' + str(l[i]) + ' ' + str(
                m[i]) + ' ' + str(n[i]) + ' ' + str(group[i]) + '\n')
    with open('MOs.dat', 'w') as f:
        f.write(str(np.size(mmod[:, 0])) + ' ' + str(np.size(mmod[0, :])) + '\n')
        for i in range(np.size(mmod[:, 0])):
            for j in range(np.size(mmod[0, :])):
                f.write(str(mmod[i, j]) + ' ')
            f.write('\n')
            

    return 

def run_scattering(
        file_name,
        one_rdm_file,
        two_rdm_file,
        molden_file,
        type='total',
        log_file='scat.log',
        q_range = (0,250),
        q_points = 1000,
        cutoffcentre = 1E-2,
        cutoffz = 1e-20,
        cutoffmd = 1e-20,
        state1 = 1,
        state2 = 1,
        state3 = 1,
        clean_files=True
):
    """
    Runs scattering calculation on a given one_rdm and two_rdm file.

    Parameters
    ----------
    file_name : str
        The output scattering file name
    one_rdm_file : str
        Path to the 1rdm file
    two_rdm_file : str
        Path to the 2rdm file
    molden_file : str
        Path to the Molden File for orbitals
    
    **kwargs
    type : str, (total, elastic)
        type of scattering to be computed, defaults total
    log_file : str
        Path to the log file for scattering calculation
    
    Returns
    -------
    q : array_like
        An array of q wave vector values, in a.u.
    intensity : array_like
        An array of intensity values at the corresponding q
    """
    prepare_files(
        file_name,
        one_rdm_file,
        two_rdm_file,
        molden_file,
        type=type,
        log_file='scat.log',
        q_range = q_range,
        q_points = q_points,
        cutoffcentre = cutoffcentre,
        cutoffz = cutoffz,
        cutoffmd = cutoffmd,
        state1 = state1,
        state2 = state2,
        state3 = state3)

    # with open(log_file, 'w') as f:
        #subprocess.run(['Main.exe'], stdout=f)
    #print(f'Running scattering with type {type} and file {file_name}')

    try:
        os.system(f'LD_LIBRARY_PATH=/opt/intel/oneapi/vpl/2022.0.0/lib:/opt/intel/oneapi/tbb/2021.5.1/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.5.1//libfabric/lib:/opt/intel/oneapi/mpi/2021.5.1//lib/release:/opt/intel/oneapi/mpi/2021.5.1//lib:/opt/intel/oneapi/mkl/2022.0.2/lib/intel64:/opt/intel/oneapi/itac/2021.5.0/slib:/opt/intel/oneapi/ipp/2021.5.2/lib/intel64:/opt/intel/oneapi/ippcp/2021.5.1/lib/intel64:/opt/intel/oneapi/ipp/2021.5.2/lib/intel64:/opt/intel/oneapi/dnnl/2022.0.2/cpu_dpcpp_gpu_dpcpp/lib:/opt/intel/oneapi/debugger/2021.5.0/gdb/intel64/lib:/opt/intel/oneapi/debugger/2021.5.0/libipt/intel64/lib:/opt/intel/oneapi/debugger/2021.5.0/dep/lib:/opt/intel/oneapi/dal/2021.5.3/lib/intel64:/opt/intel/oneapi/compiler/2022.0.2/linux/lib:/opt/intel/oneapi/compiler/2022.0.2/linux/lib/x64:/opt/intel/oneapi/compiler/2022.0.2/linux/lib/oclfpga/host/linux64/lib:/opt/intel/oneapi/compiler/2022.0.2/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/ccl/2021.5.1/lib/cpu_gpu_dpcpp:/u/ajmk/sann8252/local/lib Main.exe > {log_file}')
    except Exception as e:
        print(f"Error running Main.exe: {e}")
        return None
    
    result = np.loadtxt(f'{file_name}')
    if clean_files:
        subprocess.run(['rm', 'options.dat'])
        subprocess.run(['rm', 'basis.dat'])
        subprocess.run(['rm', 'MOs.dat'])
        subprocess.run(['rm', '1rdm_' + file_name + '.txt'])
        subprocess.run(['rm', '2rdm_' + file_name + '.txt'])
        subprocess.run(['rm', '2rdm.txt'])
        subprocess.run(['rm', file_name + '.molden'])
        subprocess.run(['rm', file_name])
    
    return result

def run_scattering_pyscf(
        casscf,
        mf,
        file_name,
        orbital_type = 'HF',
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
        **kwargs
        ):
    
    """
    Runs scattering calculation on a given one_rdm and two_rdm file.

    Parameters
    ----------
    casscf : pyscf.mcscf.CASSCF
        The CASSCF object containing the CI coefficients and FCISolver.
    mf : pyscf.scf.hf.SCF
        The mean-field object containing the molecular information.
    file_name : str
        The output scattering file name
    orbital_type : str
        The type of orbitals to be used, either 'HF' or 'CASSCF'.
    type : str, (total, elastic)
        type of scattering to be computed, defaults total
    log_file : str
        Path to the log file for scattering calculation
    q_range : tuple
        The range of q values for the scattering calculation.
    q_points : int
        The number of q points to be used in the calculation.
    cutoffcentre : float
        The cutoff value for the centre of the scattering calculation.
    cutoffz : float
        The cutoff value for the z component of the scattering calculation.
    cutoffmd : float
        The cutoff value for the md component of the scattering calculation.
    state1 : int
        The first state to be considered in the scattering calculation.
    state2 : int
        The second state to be considered in the scattering calculation.
    state3 : int
        The third state to be considered in the scattering calculation.
    
    Returns
    -------
    q : array_like
        An array of q wave vector values, in a.u.
    intensity : array_like
        An array of intensity values at the corresponding q
    """

    if orbital_type == 'HF':
        tools.molden.dump_scf(mf, f'{file_name}.molden')
    elif orbital_type == 'CASSCF':
        tools.molden.from_mcscf(casscf, f'{file_name}.molden')
    
    _ci = casscf.ci
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    nmo = casscf.mo_coeff.shape[1]

    casdm1, casdm2 = casscf.fcisolver.make_rdm12(_ci, ncas, nelecas)
    dm1, dm2 = makerdm._make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo)

    no_mos = dm1.shape[0]

    pthresh=1e-17

    with open(f'1rdm_{file_name}.txt', 'w') as f:
        for i in range(no_mos):
            for j in range(no_mos):
                if np.abs(dm1[i,j]) > pthresh:
                    f.write(f"{i+1: 3d}  {j+1: 3d}  {dm1[i, j]}\n")


    with open(f'2rdm_{file_name}.txt', 'w') as f:
        for i in range(no_mos):
            for j in range(no_mos):
                for k in range(no_mos):
                    for l in range(no_mos):
                        if np.abs(dm2[i, j, k, l]) > pthresh:
                            f.write(
                                f"{i+1: 3d}  {j+1: 3d}  {k+1: 3d}  {l+1: 3d}  {dm2[i, j, k, l]}\n"
                            )

    result = run_scattering(file_name, 
                            f'1rdm_{file_name}.txt', 
                            f'2rdm_{file_name}.txt', 
                            f'{file_name}.molden',
                            type=type,
                            log_file=log_file,
                            q_range = q_range,
                            q_points = q_points,
                            cutoffcentre = cutoffcentre,
                            cutoffz = cutoffz,
                            cutoffmd = cutoffmd,
                            state1 = state1,
                            state2 = state2,
                            state3 = state3,
                            **kwargs)
    return result

def run_scattering_csf(
        csf,
        nalpha,
        nbeta,
        norb,
        spin_mult,
        casscf,
        mf,
        file_name,
        orbital_type = 'HF',
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
        **kwargs
        ):
    """
    Run scattering on a given CSF configuration.
    
    Parameters
    ----------
    csf : list
        The CSF configuration to be used for the scattering calculation.
    nalpha : int
        The number of alpha electrons in the active space.
    nbeta : int
        The number of beta electrons in the active space.
    norb : int
        The total number of orbitals in the system.
    spin_mult : int
        The spin multiplicity of the system.

    casscf : pyscf.mcscf.CASSCF
        The CASSCF object containing the CI coefficients and FCISolver.
    mf : pyscf.scf.hf.SCF
        The mean-field object containing the molecular information.
    file_name : str
        The output scattering file name
    orbital_type : str
        The type of orbitals to be used, either 'HF' or 'CASSCF'.
    type : str, (total, elastic)
        type of scattering to be computed, defaults total
    log_file : str
        Path to the log file for scattering calculation
    q_range : tuple
        The range of q values for the scattering calculation.
    q_points : int
        The number of q points to be used in the calculation.
    cutoffcentre : float
        The cutoff value for the centre of the scattering calculation.
    cutoffz : float
        The cutoff value for the z component of the scattering calculation.
    cutoffmd : float
        The cutoff value for the md component of the scattering calculation.
    state1 : int
        The first state to be considered in the scattering calculation.
    state2 : int
        The second state to be considered in the scattering calculation.
    state3 : int
        The third state to be considered in the scattering calculation.
    
    Returns
    -------
    q : array_like
        An array of q wave vector values, in a.u.
    intensity : array_like
        An array of intensity values at the corresponding q
    """

    transformer = CSFTransformer(norb, nalpha, nbeta, spin_mult)
    dets = transformer.vec_csf2det(csf)
    occslst = fci.cistring.gen_occslst(range(casscf.ncas), (nalpha + nbeta) // 2)
    alpha = []
    beta = []

    for i, occs_alpha in enumerate(occslst.tolist()):
        for j, occs_beta in enumerate(occslst.tolist()):
            alpha.append(occs_alpha)
            beta.append(occs_beta)

    casscf_copy = deepcopy(casscf)
    alpha = np.array(alpha)
    beta = np.array(beta)
    ci_to_2rdm.update_ci_coeffs(alpha, beta, dets.flatten(), casscf_copy, update = True)

    result = run_scattering_pyscf(
        casscf_copy,
        mf,
        file_name,
        orbital_type = orbital_type,
        type=type,
        log_file=log_file,
        q_range = q_range,
        q_points = q_points,
        cutoffcentre = cutoffcentre,
        cutoffz = cutoffz,
        cutoffmd = cutoffmd,
        state1 = state1,
        state2 = state2,
        state3 = state3,
        **kwargs
    )
    return result
    
    


if __name__ in "__main__":
    mol = gto.Mole(atom = 'Be 0 0 0', basis = '3-21g', symmetry = False, spin = 0, charge = 0 , cart=True)
    mf = scf.HF(mol)
    mf.kernel()
    casscf = mcscf.CASSCF(mf, 9, 4)
    casscf.kernel()
    pyscf_result = run_scattering_pyscf(
        casscf,
        mf,
        'test_total_pyscf',
        orbital_type = 'CASSCF',
        type='total',
        log_file='scat.log',
        q_range = (1E-10,250),
        q_points = 1000,
        cutoffcentre = 1E-2,
        cutoffz = 1e-20,
        cutoffmd = 1e-20,
        state1 = 1,
        state2 = 1,
        state3 = 1
    )
    transformer = CSFTransformer(9,2,2,1)
    csf_result = run_scattering_csf(
        transformer.vec_det2csf(casscf.ci),
        2,
        2,
        9,
        1,
        casscf,
        mf,
        'csf_test')
    print((csf_result - pyscf_result).sum())
