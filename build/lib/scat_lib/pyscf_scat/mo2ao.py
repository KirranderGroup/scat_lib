"""
Module to transform 2RDM from MO to AO basis and symmetrize it.

Andres Moreno Carrascosa

2025

"""

import numpy as np
from pyscf import tools


def symmetrize8(G):
    '''
    Symmetrize a 4-index tensor G[p,q,r,s] by averaging over all 8 permutations
    of the indices that correspond to the same physical quantity.

    Parameters
    ----------
    G : np.ndarray
        A 4-dimensional numpy array representing the tensor to be symmetrized.

    Returns
    -------
    np.ndarray
        The symmetrized tensor.
    '''
    return (
        G
      + G.transpose(1,0,2,3)
      + G.transpose(0,1,3,2)
      + G.transpose(1,0,3,2)
      + G.transpose(2,3,0,1)
      + G.transpose(3,2,0,1)
      + G.transpose(2,3,1,0)
      + G.transpose(3,2,1,0)
    ) * 0.125


def mo2ao_2rdm_halftrans(dm2, C):
    '''
    Transform a 2-RDM from MO basis to AO basis using half transformation.
    Parameters
    ----------
    dm2 : np.ndarray
        4-dimensional array representing the 2-RDM in MO basis.
    C : np.ndarray
        2-dimensional array representing the MO coefficients (shape: nao x nmo).
    Returns
    -------
    G : np.ndarray
        4-dimensional array representing the 2-RDM in AO basis.
    '''
    X = np.tensordot(C, dm2, axes=(1,0))      # (nao, nmo, nmo, nmo)
    X = np.tensordot(C, X,   axes=(1,1))      # (nao, nao, nmo, nmo)
    X = np.tensordot(C, X,   axes=(1,2))      # (nao, nao, nao, nmo)
    G = np.tensordot(C, X,   axes=(1,3))      # (nao, nao, nao, nao)
    return G



def norm_reorder_MOs(mos,mol):
    '''
    Normalize and reorder MOs to match the order in Molden files.
    Parameters
    ----------
    mos : np.ndarray
        2-dimensional array representing the MO coefficients (shape: nao x nmo).
    mol : pyscf.gto.Mole
        PySCF molecule object.
    Returns
    -------
    mos_ord : np.ndarray
        2-dimensional array of normalized and reordered MO coefficients.
    '''

    idx=tools.molden.order_ao_index(mol)
    norm = abs(mol.intor('int1e_ovlp').diagonal() ** .5)
    mos_norm= np.einsum('i,ij->ij', norm, mos)
    mos_ord=mos_norm[idx,:]
    return mos_ord



def create_Zcotr(mf,mol,dm2): 
    '''
    Create and save the symmetrized 2RDM in AO basis to a binary file 'Zcotr.dat'.
    Parameters
    ----------
    mf : pyscf.scf.SCF
        PySCF mean-field object containing MO coefficients.
    mol : pyscf.gto.Mole
        PySCF molecule object.
    dm2 : np.ndarray
        4-dimensional array representing the 2-RDM in MO basis.
    Returns
    -------
    dm3 : np.ndarray
        4-dimensional array representing the symmetrized 2-RDM in AO basis.
    '''

    dm2=dm2.transpose(1,0,3,2)
    mos = norm_reorder_MOs(mf.mo_coeff,mol)
    dm2_sym = symmetrize8(dm2)           # fast in MO space
    dm3 = mo2ao_2rdm_halftrans(dm2_sym, mos)   # single AO transform
    
    return dm3
    
      
