from pyscf import gto, scf, mcscf, fci
from mrh.my_pyscf.fci.csfstring import CSFTransformer
import numpy as np
from scat_lib.ci_to_2rdm import write_ci_file, read_ci_file, update_ci_coeffs, calc_energy
import sys
sys.path.append('./')
from .scat_calc import run_scattering, run_scattering_csf, run_scattering_pyscf
from fit_utils import generate_comparison_plot
import matplotlib.pyplot as plt
import colorcet as cc
import seaborn as sns
sns.set_palette(cc.glasbey_bw)


class ReducedCASSCF(mcscf.mc1step.CASSCF):
    """
    A class to perform reduced CASSCF calculations with the ability to write CI coefficients and CSF strings to a file.

    Attributes
    ----------
    mf_or_mol : object
        The mean-field object or molecule object.
    ncas : int
        The number of active orbitals.
    nalpha : int
        The number of alpha electrons.
    nbeta : int
        The number of beta electrons.
    ncore : int, optional
        The number of core orbitals.
    frozen : int, optional
        The number of frozen orbitals.
    nelec : tuple
        The number of electrons (alpha, beta).
    occslst : list
        List of occupied orbitals.
    transformer : CSFTransformer
        The transformer object for CSF transformations.
    csf : numpy.ndarray
        The CSF coefficients.
    reduced_csf : numpy.ndarray
        The reduced CSF coefficients after applying a tolerance.
    reduced_csf_inverse : numpy.ndarray
        The inverse mapping of the reduced CSF coefficients.
    ncsf : int
        The number of CSFs.
    ci : numpy.ndarray
        The CI coefficients.
    fcisolver : object
        The FCI solver object.
    """

    def __init__(self, mf_or_mol, ncas=0, nalpha = 0 , nbeta = 0, ncore=None, frozen=None):
        super().__init__(mf_or_mol, ncas = ncas, nelecas = (nalpha + nbeta), ncore = ncore, frozen = frozen)
        self._mf = mf_or_mol
        self.nelec = (nalpha, nbeta)
        self.nalpha = nalpha
        self.nbeta = nbeta        
        self._transformer = CSFTransformer(ncas, nalpha, nbeta, 1)
        self.occslst = fci.cistring.gen_occslst(range(self.ncas), (self.nelec[0] + self.nelec[1]) // 2)
        _ = self.kernel()
        self._csf = self.transformer.vec_det2csf(self.ci)

        self.alpha_dets = [a for a in self.occslst.tolist() for _ in range(len(self.occslst))]
        self.beta_dets = [b for _ in range(len(self.occslst)) for b in self.occslst.tolist()]


    def write_ci(self, file_name, tol = None):
        
        if tol is not None:
            with open(f'{file_name}.txt', 'w') as f:
                f.write('# det_alpha,   det_beta,   CI Coeffs\n')
                for c, ia, ib in self.fcisolver.large_ci(self.ci, self.ncas, self.nelec, tol=tol, return_strs=False):
                    f.write('%s\t%s\t%.12f\n' % (ia.tolist(), ib.tolist(), c))

        else:
            with open(f'{file_name}.txt','w') as f:
                f.write('# det_alpha\tdet_beta\tCI Coeffs\n')
                for i, occs_alpha in enumerate(self.occslst.tolist()):
                    for j, occs_beta in enumerate(self.occslst.tolist()):
                        f.write('%s\t%s\t%.12f\n' % (occs_alpha, occs_beta, self.ci[i,j]))
        
        print('CI Written')
        return
    

    def write_csf(self, file_name):
        with open(f'{file_name}.txt', 'w') as f:
            f.write('CSF String,    Coeffs\n')

            for i, coeff in enumerate(self.csf):
                try:
                    csf_str = self.transformer.printable_csfstring(i)
                    f.write(f"{i}\t{csf_str}\t{coeff:.12f}\n")
                except:
                    f.write(f"{i}\t{coeff:.12f}\n")

    
    def get_reduced_csf_params(self, decimals=10):
        unique_csf, unique_locs, inverse, counts = np.unique(self.csf.round(decimals=decimals), return_inverse = True, return_index = True, return_counts = True)
        reduction_factors = {a : np.sqrt(b) for a, b in enumerate(counts)}

        rescaled_csf = np.array([val * reduction_factors[i] for i, val in enumerate(unique_csf)])

        print(20*'*')
        print(f"Dimensions before reduction: {self.transformer.ncsf}")
        print(f"Dimensions after reduction: {unique_csf.shape}")
        print(20*'*')
        self._reduced_csf = rescaled_csf
        self._reduced_csf_inverse = inverse
        self._reduced_csf_reduction_factors = reduction_factors
        self._reduced_csf_unique_locs = unique_locs
        self._nparams = len(unique_csf)
        return rescaled_csf, inverse
    

    @property
    def nparams(self):
        return self._nparams

    @property
    def ncsf(self):
        return self._transformer.ncsf

    @property
    def csf(self):
        return self._csf
    
    @property
    def transformer(self):
        return self._transformer
    


    @transformer.setter
    def transformer(self, value):
        if not isinstance(value, CSFTransformer):
            raise TypeError("transformer must be an instance of CSFTransformer.")
        self.transformer = value
        self.transformer.norb = self.ncas
        self.transformer.nalpha = self.nalpha
        self.transformer.nbeta = self.nbeta

    @csf.setter
    def csf(self, value):
        norm = np.sqrt(np.sum(value**2))
        if np.isclose(norm, 1):
            if len(value.flatten()) == self.transformer.ncsf:
                self._csf = value
                full_det = self.transformer.vec_csf2det(value)
                self.update_ci(full_det.flatten())           
        else:
            raise ValueError("CSF coefficients are not normalized. Please normalize them before setting.")
    
    @property
    def reduced_csf(self):
        return self._reduced_csf
    @property
    def reduced_csf_inverse(self):
        return self._reduced_csf_inverse
    
    @reduced_csf.setter
    def reduced_csf(self, value):

        assert len(value) == self.nparams, f'Dimension of supplied array is not {self.nparams}'

        if not isinstance(value, np.ndarray):
            raise TypeError("reduced_csf must be a numpy array.")
        
        if np.isclose(np.sum(value**2),1):
            self._reduced_csf = value
        else:
            raise ValueError("reduced_csf coefficients are not normalized. Please normalize them before setting.")
        
        unscaled = np.array([val *1/ self._reduced_csf_reduction_factors[i] for i, val in enumerate(self._reduced_csf)])

        expanded_csf = unscaled[self._reduced_csf_inverse]
        print(np.sum(expanded_csf**2))
        assert np.isclose(np.sum(expanded_csf**2), 1), 'Expanded CSF coefficients are not normalized!'
        self.csf = expanded_csf


    def calc_etot(self):
        self.e_tot = calc_energy(self, self._mf, self.ci)
        return self.e_tot

    def update_ci(self, value):
        norm = (value**2).sum()
        assert np.isclose(norm, 1), f'Supplied CI Vector is not normalised! (Norm = {norm:.10f})'
        _ = update_ci_coeffs(self.alpha_dets, self.beta_dets, value, self, update = True)
        _ = self.calc_etot()
    

    def run_scattering(self, file_name, **kwargs):
        """
        Run the scattering calculation using the CI coefficients and CSF strings.
        
        Parameters
        ----------
        file_name : str
            The name of the file containing the CI coefficients and CSF strings.
        **kwargs : dict
            Additional arguments to pass to the scattering function.
        """
        self._result = run_scattering_pyscf(self, self._mf, file_name, **kwargs)
        return self._result

    def run_scattering_csf(self, file_name, **kwargs):
        """
        Run the scattering calculation using the CI coefficients and CSF strings.
        
        Parameters
        ----------
        file_name : str
            The name of the file containing the CI coefficients and CSF strings.
        **kwargs : dict
            Additional arguments to pass to the scattering function.
        """
        self._result = run_scattering_csf(self.csf, self.nalpha, self.nbeta, self.ncas, 1, self, self._mf, file_name, **kwargs)

        
        return self._result

    def generate_comparison_plot(self, ref_file, hf_file=None, save_path=None, **kwargs):
        """
        Generate a comparison plot of the scattering results.
        
        Parameters
        ----------
        file_name : str
            The name of the file containing the CI coefficients and CSF strings.
        **kwargs : dict
            Additional arguments to pass to the plotting function.
        """
        # Load data
        ref = np.loadtxt(ref_file)
        if hf_file is not None: hf = np.loadtxt(hf_file)
        
        fit = self._result
        
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
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to {save_path}")
        
        return fig
