from __future__ import annotations
import argparse, sys, numpy as np
from .cm import load_cm_table
from .geometry import read_xyz
from .qgrid import q_grid_f90
from .scattering import intensity_molecular_xray

def main(argv=None):
    p = argparse.ArgumentParser(description="Gas-phase IAM (X-ray) intensity, F90-compatible")
    p.add_argument("--xyz", required=True, help="Input XYZ file (labels must match affl.txt keys, e.g. 'Cval', 'Siv')")
    p.add_argument("--nq", type=int, default=1000, help="Number of q points (default 1000)")
    p.add_argument("--qmin", type=float, default=0.0, help="q min [Å^-1] (default 0.0)")
    p.add_argument("--qmax", type=float, default=None, help="q max [Å^-1] (default 8/a0 ≈ 15.12)")
    p.add_argument("--affl", type=str, default=None, help="Path to affl.txt (Cromer–Mann table). Defaults to bundled copy.")
    p.add_argument("--out", type=str, default=None, help="Output .txt file (columns: q  I(q)). Defaults to stdout.")
    args = p.parse_args(argv)

    cm = load_cm_table(args.affl)
    pos, labels = read_xyz(args.xyz)
    q = q_grid_f90(nq=args.nq, qmin=args.qmin, qmax=args.qmax)

    Iq = intensity_molecular_xray(pos, labels, q, cm)

    if args.out:
        import numpy as np
        np.savetxt(args.out, np.column_stack([q, Iq]), header="q[1/Å]    I(q) (molecular)")
    else:
        for qi, Ii in zip(q, Iq):
            print(f"{qi:12.6f}  {Ii:16.8f}")

if __name__ == "__main__":
    main()
