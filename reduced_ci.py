from pyscf import gto, scf, mcscf, fci
from mrh.my_pyscf.fci.csfstring import CSFTransformer
import numpy as np
from ci_to_2rdm import write_ci_file, read_ci_file, update_ci_coeffs
from scat_lib import run_scattering, run_scattering_csf, run_scattering_pyscf

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

        return rescaled_csf, inverse
    
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
                self.csf = value           
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
        result = run_scattering_pyscf(self, self._mf, file_name, **kwargs)
        return result

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
        result = run_scattering_csf(self.csf, self.nalpha, self.nbeta, self.ncas, 1, self, self._mf, file_name, **kwargs)
        
        
        return result

