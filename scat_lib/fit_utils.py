import numpy as np
from .ci_to_2rdm import *
from pyscf import mcscf, scf, fci, ci, ao2mo
import matplotlib.pyplot as plt
from .rdm_tools import *
import os 
import sys
from scipy.optimize import minimize
import colorcet as cc
import matplotlib as mpl
plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=cc.glasbey_bw)
os.environ['OMP_NUM_THREADS']='32'
from .sine_transform import *
from pickle import dump

def fitting(x, casscf, ref, hf_dat, name='fit'):
    '''
    Fit the CI coefficients to the scattering data
    
    Parameters
    ----------
    x : np.ndarray
        CI coefficients
    casscf : mcscf.CASSCF
        CASSCF object
    ref : np.ndarray
        Reference scattering intensities
    hf_dat : np.ndarray
        HF scattering intensities
    name : str, optional
        Name of the fit
    
    Returns
    -------
    cost : float
        Cost of the fit
    
    '''
    x = x/np.sum(x**2)
    alpha, beta, ci = read_ci_file('pyscf_civectors_large.txt', sort_by_ci=True)
    update_ci_coeffs(alpha, beta, x, casscf, update=True)
    save_rdm(casscf,name)
    update_options(name)
    _ = os.system('~/PyXSCAT_Patrick/src/Main.exe')
    calc = np.loadtxt(f'total_{name}.dat')
    N = 12
    
    Itarget = ref[:,1] + N  # ref has Itarget(q)
    Ifit    = calc[:,1] + N  # calc has Ifit(q)
    Ihf     = hf_dat[:,1] + N# hf_dat has Ihf(q)
    
    # Implement the residual: sum of ((Itarget(q) - Ifit(q)) / Ihf(q))**2
    r = np.sum( ((Itarget - Ifit)*0.5*ref[:,0] / Ihf) ** 2 )
    Iinf = Ifit[-1]     # or whichever index is "large q"

    # define N how you like
    alpha_c = 1
    beta_c = 1
    penalty = alpha_c * (calc[:,1][0] - 12)**2 + beta_c * (Iinf - 12)**2

    # total objective
    cost = r + penalty

    # log it for reference
    #r=np.sum((calc[:,1]*(ref[:,0]+5) - ref[:,1]*((ref[:,0]+5)))**2)
    _ = os.system(f'echo {r}\t {penalty} \t {cost} >> ./runs/residual_{name}.dat')
    return cost


def plot(name, 
         ref, 
         title=None, 
         saveto=None, 
         display_energy=False,
         mol=None,
         casscf=None,
         ci_modified=None,
         return_fig_ax=False):
    '''
    Plot the reference, HF, and fit scattering intensities
    and their percentage differences

    Parameters
    ----------
    name : str
        Name of the fit
    ref : np.ndarray
        Reference scattering intensities
    title : str, optional
        Title of the plot
    saveto : str, optional
        Save the plot to a file

    Returns
    -------
    None
    
    '''
    fig,ax = plt.subplots()
    ax.set_xlim(0,8)
    ax.plot(ref[:,0], ref[:,1], label='AVQZ Reference')
    hf = np.loadtxt('total_hf.dat')
    ax.plot(hf[:,0], hf[:,1], label='HF')
    #ax.legend()
    axt = ax.twinx()
    axt.scatter(hf[:,0], -(ref[:,1] - hf[:,1])/ref[:,1]*100, s=15, marker='x', label='HF % Diff')
    axt.set_ylim(-50,50)
    axt.axhline(0, c='grey', ls='--')
    fit = np.loadtxt(f'./runs/total_{name}.dat')
    ax.plot(fit[:,0], fit[:,1], label='Fit')
    axt.scatter(fit[:,0], -(ref[:,1] - fit[:,1])/ref[:,1]*100, s=15, marker='x', label='Fit % Diff')
    ls1, lb1 = ax.get_legend_handles_labels()
    ls2, lb2 = axt.get_legend_handles_labels() 
    axt.legend(ls1 + ls2, lb1 + lb2)
    ax.set_title(f"{name if title is None else title}")
    ax.set_xlabel('$q$ / a.u.')
    ax.set_ylabel('Scattering Intensity')
    axt.set_ylabel('Percentage Difference')


    if display_energy is True:
        E_total = calc_energy(mol, casscf, ci_modified)
        print(f'Total energy: {E_total}')
        ax.annotate(f'Fit Total energy: {E_total:.4f} a.u.', xy=(0.3,0.9), xycoords='axes fraction', fontsize=8)



    if saveto is not None:
        fig.savefig(f'{saveto}.png',dpi=300)     
    if return_fig_ax:
        return fig, ax, axt, E_total
    

def calc_energy(mol, casscf, ci_modified):
    '''
    Calculate the total energy of the system

    Parameters
    ----------  
    mol : pyscf.gto.Mole
        Molecule object
    casscf : mcscf.CASSCF
        CASSCF object
    ci_modified : np.ndarray
        Modified CI vector

    Returns
    -------
    E_total : float
        Total energy of the system
    '''
    h1_eff, e_core = casscf.get_h1eff()       # one-electron part (active space) and core energy
    eri_cas = casscf.get_h2eff()             # two-electron integrals in active space (MO basis)
    E_cas = fci.direct_spin1.energy(h1_eff, eri_cas, ci_modified, casscf.ncas, casscf.nelecas)  # electronic CAS energy
    E_total = E_cas + e_core + mol.energy_nuc()  # add core and nuclear repulsion energy
    return E_total




def generate_comparison_plot(file, ref_file='reference.dat', hf_file = None,save_path=None):
    """
    Generate comprehensive comparison plot.
    
    Args:
        name: Name of the optimization run
        ref_file: Reference data file
        save_path: Path to save the plot
    """
    # Load data
    ref = np.loadtxt(ref_file)
    if hf_file is not None: hf = np.loadtxt(hf_file)
    fit = np.loadtxt(file)
    
    # Create main figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Upper plot: Scattering intensities
    ax1.plot(ref[:,0], ref[:,1], 'k-', linewidth=2, label='AVQZ Reference')
    if hf_file is not None: ax1.plot(hf[:,0], hf[:,1], 'b--', linewidth=1.5, label='HF')
    ax1.plot(fit[:,0], fit[:,1], 'r-', linewidth=1.5, label='CSF Optimized Fit')
    
    ax1.set_xlabel('$q$ / a.u.')
    ax1.set_ylabel('Scattering Intensity')
    ax1.set_xlim(0, 8)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Lower plot: Percentage differences
    ax2.axhline(0, color='k', linestyle='-', alpha=0.5)
    if hf_file is not None: ax2.scatter(hf[:,0], -(ref[:,1] - hf[:,1])/ref[:,1]*100, 
               s=20, marker='x', color='blue', label='HF % Diff')
    ax2.scatter(fit[:,0], -(ref[:,1] - fit[:,1])/ref[:,1]*100, 
               s=20, marker='o', color='red', label='CSF Optimized % Diff')
    
    ax2.set_xlabel('$q$ / a.u.')
    ax2.set_ylabel('% Difference')
    ax2.set_xlim(0, 8)
    ax2.set_ylim(-50, 50)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Set title
    fig.suptitle(f'CSF-Based Optimized Fit: {file}', fontsize=16)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    return fig