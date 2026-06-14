# Julia compute engine (XScattering.jl) for scat_lib

The scattering compute can now run on the Julia port **XScattering.jl** instead of
the legacy Fortran executable, with **no change to your workflow** beyond one
keyword. The deck preparation (`prepare_files` / `prepare_zcotr_files`) is reused
unchanged; only the compute step is swapped, via `juliacall`.

## Setup (once)

```bash
pip install juliacall
export XSCATTERING_PROJECT=/path/to/XScattering          # the XScattering.jl package dir
julia --project=$XSCATTERING_PROJECT -e 'using Pkg; Pkg.instantiate()'
export JULIA_NUM_THREADS=16                              # optional: parallel quartet loop
```

## Use it

Add `engine="julia"` to your existing `run_scattering_pyscf` call:

```python
from scat_lib.pyscf_scat.scat_calc import run_scattering_pyscf

# total scattering (uses the zcotr deck + 2rdmAO)
q, I = run_scattering_pyscf(mc, mf, 'water_total.dat',
                           backend='zcotr', engine='julia', type='total')

# elastic scattering (1RDM path)
q, I = run_scattering_pyscf(mc, mf, 'water_elastic.dat',
                           backend='normal', engine='julia', type='elastic')
```

- `engine='fortran'` (default) → unchanged legacy behaviour.
- `fortran_pi=True` → reproduce the legacy Fortran's float32-π value bit-for-bit.
  Default uses true double π (~8e-8 more accurate).

`engine='julia'` is also threaded through the lower-level `run_scattering` and
`run_scattering_zcotr`.

## What changed in scat_lib

- `pyscf_scat/julia_backend.py` (new): the `juliacall` bridge.
- `pyscf_scat/scat_calc.py`: added an `engine` (and `fortran_pi`) keyword to
  `run_scattering`, `run_scattering_zcotr` and `run_scattering_pyscf`; when
  `engine="julia"`, the compute step calls `julia_backend.run_julia` after the
  deck is written. The Fortran path is otherwise untouched.

## Validation

The Julia engine reproduces the Fortran/oracle curves to machine precision
(total and elastic, single-atom and multi-centre); see the XScattering.jl test
suite and docs.
