"""Write a benchmark preset of ``python/problems.py`` to a ``.mat`` file.

The Julia benchmark (``bench_aha.jl``) reads the problem from this file so that
it times *exactly* the same trajectory, Gram blocks, sensitivity maps and
coefficients as the Python benchmark of MR_AHA (the random numbers come from
NumPy's generator and cannot be reproduced in Julia).

The file is written by :func:`problems.save`, i.e. in the column-major layout
of the MR_AHA documentation, which is also Julia's native layout:

* ``SEs``  ``[Nx, Ny, (Nz,) Ncoils]``
* ``phi2`` ``[Ntrajs, (Nkz,) L, L]`` (Gram blocks ``Phi_rt(:,idx) * Phi_rt(:,idx)'``)
* ``traj`` ``[Ntrajs, Nkx, ndim]`` in radians, ``[-pi, pi)``
* ``x``    ``[Nx, Ny, (Nz,) L]``
* ``traj_type`` ``'2D'``, ``'3D'`` or ``'stack'``

Usage::

    python julia/save_problem.py 2d julia/data/2d.mat [--seed 0]
        [--n NX NY [NZ]] [--L L] [--coils C] [--spokes S] [--samples K]

Only NumPy and SciPy are needed (no GPU, no ``mr_aha``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import problems  # noqa: E402


def main(argv=None) -> int:
    """Parse the command line, build the problem and save it.

    Returns:
        0 on success.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("preset", choices=sorted(problems.PRESETS))
    ap.add_argument("out", help="output .mat (or .npz) file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, nargs="+", default=None)
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--coils", type=int, default=None)
    ap.add_argument("--spokes", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None)
    a = ap.parse_args(argv)
    p = problems.override(problems.PRESETS[a.preset], a.n, a.L, a.coils,
                          a.spokes, a.samples)
    prob = problems.synthetic(p, a.seed)
    problems.save(prob, a.out)
    print(f"{a.preset}: {prob['desc']} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
