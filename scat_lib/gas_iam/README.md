# gas_iam_f90 (v0.2.0)

Gas-phase IAM scattering (X-ray) reimplemented from your F90 code.

**New:** optional `xraydb` backend for form factors. Use Waasmaier–Kirfel `f0(ion, s)` directly from the `xraydb` Python package.

## Install

```bash
python -m pip install ./gas_iam_f90-0.2.0.zip
# optional dependency for the xraydb backend:
python -m pip install xraydb
```

## Use with xraydb

```python
from gas_iam_f90 import read_xyz, q_grid_f90, intensity_molecular_xray

pos, labels = read_xyz("molecule.xyz")
q = q_grid_f90(nq=1000)
Iq = intensity_molecular_xray(pos, labels, q, backend="xraydb")   # uses xraydb.f0(ion, s)
```

If your XYZ uses labels like `Cval`, `Siv`, or `Sival`, supply a mapping to valid xraydb ion names (for example, `Cval -> C`, `Siv -> Si4+` or `Si`, depending on the physics you want):

```bash
python -m gas_iam_f90 --xyz molecule.xyz --backend xraydb --ion-map '{"Cval":"C", "Siv":"Si"}' --out Iq.txt
```

## Notes

- The bundled `data/affl.txt` still provides Cromer–Mann coefficients for the classic 4-Gaussian + constant parameterization (including special labels such as `Cval`, `Siv`, and ionic forms).  For the `xraydb` backend the coefficients come from Waasmaier–Kirfel (1995) and are queried at run time. 
- Units: positions in Å; q in Å⁻¹; `s = q/(4π)` in Å⁻¹.
- Output matches the F90 program's molecular signal `I(q) = Σ_{ij} f_i(s) f_j(s) sinc(q r_ij)` (no Debye–Waller or damping).
