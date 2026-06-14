"""
Julia compute backend for scat_lib: runs the XScattering.jl engine via juliacall
instead of the legacy Fortran executable, while reusing the exact same deck files
(basis.dat / MOs.dat / coeffs.dat / options.dat + 2rdmAO / onerdm.dat) that
``prepare_files`` already writes. This keeps the public ``run_scattering*`` API
unchanged: pass ``engine="julia"`` to opt in.

Setup (once):
    pip install juliacall
    # point at the XScattering.jl package and instantiate it:
    export XSCATTERING_PROJECT=/path/to/XScattering
    julia --project=$XSCATTERING_PROJECT -e 'using Pkg; Pkg.instantiate()'

Threading: set ``JULIA_NUM_THREADS`` (or ``PYTHON_JULIACALL_THREADS``) before the
first juliacall import.
"""
import os
import numpy as np

# Map scat_lib ``type`` strings -> XScattering ``kind`` strings.
_KIND = {
    "total": "total",
    "elastic": "elastic",
    "total_j1": "total_j1",
    "total_j2": "total_j2",
    "elastic_j2": "elastic_j2",
    "total_aligned": "total_aligned",
    "elastic_aligned": "elastic_aligned",
}

_JL = None  # cached juliacall Main


def _project_dir():
    p = os.environ.get("XSCATTERING_PROJECT")
    if p:
        return os.path.abspath(p)
    # fall back to a sibling "XScattering" of the repo root (…/<repo>/XScattering)
    here = os.path.dirname(os.path.abspath(__file__))
    for up in range(2, 8):
        cand = os.path.join(here, *([".."] * up), "XScattering")
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    raise RuntimeError(
        "Cannot locate the XScattering.jl package; set XSCATTERING_PROJECT to its path."
    )


def _jl():
    """Lazily import juliacall, activate the XScattering project, and load it."""
    global _JL
    if _JL is not None:
        return _JL
    try:
        from juliacall import Main as jl
    except ImportError as e:
        raise ImportError(
            "engine='julia' needs juliacall (`pip install juliacall`)."
        ) from e
    proj = _project_dir()
    jl.seval("import Pkg")
    jl.seval(f'Pkg.activate(raw"{proj}"; io=devnull)')
    jl.seval("using XScattering")
    _JL = jl
    return jl


def _ensure_onerdm(one_rdm_file, norbs):
    """Elastic kinds read a dense onerdm.dat; build it from the sparse 1RDM txt."""
    if os.path.exists("onerdm.dat"):
        return
    dm1 = np.zeros((norbs, norbs))
    with open(one_rdm_file) as fh:
        for line in fh:
            t = line.split()
            if len(t) >= 3:
                i, j = int(t[0]) - 1, int(t[1]) - 1
                dm1[i, j] = float(t[2])
    with open("onerdm.dat", "w") as fh:
        for i in range(norbs):
            fh.write(" ".join(repr(x) for x in dm1[i, :]) + "\n")


def _norbs_from_mos():
    with open("MOs.dat") as fh:
        return int(fh.readline().split()[0])


def run_julia(file_name, type="total", one_rdm_file=None, fortran_pi=False):
    """
    Compute the curve with XScattering.jl. The deck (basis/MOs/coeffs/options and
    2rdmAO/1RDM) must already be written in the current directory by
    ``prepare_files`` (+ the zcotr 2rdmAO step for total kinds). Returns the
    ``(npoints, 2)`` array of ``(q, I)`` and also writes it to ``file_name``.
    """
    kind = _KIND.get(type)
    if kind is None:
        raise ValueError(f"engine='julia' does not support type='{type}' yet "
                         f"(supported: {sorted(_KIND)}).")
    if kind.startswith("elastic"):
        if one_rdm_file is None:
            raise ValueError("elastic kinds need one_rdm_file to build onerdm.dat")
        _ensure_onerdm(one_rdm_file, _norbs_from_mos())
    jl = _jl()
    jl.XScattering.scatter(".", kind=kind, fortran_pi=fortran_pi, outfile=file_name)
    return np.loadtxt(file_name)
