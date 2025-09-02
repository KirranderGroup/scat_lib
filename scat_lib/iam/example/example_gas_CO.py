import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../../..')
au2bohr = 0.52917721067  # Angstrom
from scat_lib.iam  import Structure, AtomSite, DebyeWaller, gas_intensity_xray

# CO example
mol = Structure([
    AtomSite("Li", (0.0, 0.0, 0.0), debye_waller=DebyeWaller(B_iso=0)),
    AtomSite("F", (0, 0.0, 1.5639), debye_waller=DebyeWaller(B_iso=0))
])

q = np.linspace(1E-10, 8, 1000)
Iq, Iself, Icross = gas_intensity_xray(mol, q, return_components=True)

plt.plot(q / au2bohr, Iq, label='I(q)')
plt.xlabel('q (1/Å)')
plt.ylabel('Intensity')
plt.legend()
plt.savefig('example_gas_LiF.png', dpi=300)

joe = np.loadtxt('li_f.dat')
plt.plot(q / au2bohr, joe)
plt.savefig('joe.png', dpi=300)