from pyscf import gto, scf, mcscf, fci
from mrh.my_pyscf.fci.csfstring import CSFTransformer
import numpy as np
from ci_to_2rdm import write_ci_file, read_ci_file, update_ci_coeffs


class ReducedCASSCF(mcscf.mc1step.CASSCF):

    def __init__(self, mf_or_mol, ncas=0, nalpha = 0 , nbeta = 0, ncore=None, frozen=None):
        super().__init__(mf_or_mol, ncas = ncas, nelecas = (nalpha + nbeta), ncore = ncore, frozen = frozen)
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
        unique_csf, unique_locs = np.unique(self.csf.round(decimals=decimals), return_inverse = True)
        print(20*'*')
        print(f"Dimensions before reduction: {self.transformer.ncsf}")
        print(f"Dimensions after reduction: {unique_csf.shape}")
        print(20*'*')
        self.reduced_csf = unique_csf
        self.reduced_csf_inverse = unique_locs
        
        return unique_csf, unique_locs
    
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