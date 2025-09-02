import numpy as np
import sys
sys.path.append('../../..')
from scat_lib.iam import gas_intensity_xray_f90, Structure, AtomSite, gas_intensity_xray
import matplotlib.pyplot as plt

pos = np.array([[0.0,0.0,0.0],[1.5639,0.0,0.0]])
elems = ["Li","F"]
q = np.linspace(0.0, 8, 1000)  # IAM.f90 default

I_f90 = gas_intensity_xray_f90(pos, elems, q)

# Library gas with B=0 equals the same double sum (self + 2*cross)
joe = np.loadtxt('li_f.dat')  # Joe's data for LiF gas
new = np.loadtxt('/u/ajmk/sann8252/scat_lib/gas_iam_f90-0.1.0/Iq.txt')
# plt.plot(q, I_f90, label='I_f90')
plt.plot(new[:,0], new[:,1], label='I_gas_lib')
#plt.plot(q, joe, label='Joe\'s data')
plt.xlabel('Q (1/Angstrom)')
plt.ylabel('Intensity')
plt.legend()
plt.savefig('example_f90_parity.png', dpi=300)