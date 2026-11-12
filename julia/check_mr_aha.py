"""Compare the ``A^H A x`` of MRISubspaceRecon.jl with that of MR_AHA.

Reads a problem file written by ``julia/save_problem.py`` and the result of
``julia bench_aha.jl --data <problem> --save-result <y>``, applies MR_AHA's
Toeplitz and NUFFT operators to the same ``x`` (Python interface, one GPU) and
prints the relative 2-norm differences (also of MR_AHA's masked Toeplitz
operator), both as is and after the best-fitting
complex scalar ``c = <y_ref, y> / <y, y>`` (to expose a pure normalisation
difference).

Usage::

    MR_AHA_ROOT=~/code/MR_AHA python julia/check_mr_aha.py PROBLEM.mat Y_JULIA.mat

The library is found like ``python/bench_aha.py`` finds it (``$MR_AHA_ROOT``,
else the ``third_party/MR_AHA`` submodule).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

import bench_aha  # noqa: E402
import problems   # noqa: E402


def compare(y: np.ndarray, ref: np.ndarray) -> tuple:
    """Relative differences of ``y`` to ``ref``.

    Returns:
        ``(raw, fitted, c)``: ``||y - ref|| / ||ref||``, the same after
        scaling ``y`` by the least-squares complex scalar ``c``, and ``c``.
    """
    y, ref = y.ravel().astype(np.complex128), ref.ravel().astype(np.complex128)
    c = np.vdot(y, ref) / np.vdot(y, y)
    nr = np.linalg.norm(ref)
    return (np.linalg.norm(y - ref) / nr, np.linalg.norm(c * y - ref) / nr, c)


def main(argv=None) -> int:
    """Run the comparison; returns 0."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    prob = problems.load(argv[0])
    # The Julia result is column-major [Nx, Ny, (Nz,) L]. MAT.jl writes v7.3
    # (HDF5) files, whose reader returns it row-major, (L, (Nz,) Ny, Nx), as
    # the Python API wants; a v5 file read by loadmat needs a transpose.
    try:
        y_jl = np.transpose(sio.loadmat(argv[1])["y"])
    except NotImplementedError:
        y_jl = problems._mat_v73(Path(argv[1]))["y"]

    bench_aha.setup()
    mr_aha, xp = bench_aha.mr_aha, bench_aha.xp
    dev = bench_aha.to_device(prob)
    cfg = mr_aha.make_config(prob["traj_type"], prob["n_dim"], prob["L"],
                             n_coils=prob["n_coils"], eps=1e-6)
    print(f"{argv[0]}: {prob['desc']}")
    refs = {}
    for method in ("toeplitz", "nufft"):
        op = mr_aha.create_op(cfg, dev["SEs"], dev["phi2"], dev["traj"],
                              method=method)
        refs[method] = xp.asnumpy(op(dev["x"]))
        op.close()
    # MR_AHA's masked Toeplitz operator, the closest analogue of the package's
    # operator (which also keeps the kernel only near the samples).
    cfg_m = mr_aha.make_config(prob["traj_type"], prob["n_dim"], prob["L"],
                               n_coils=prob["n_coils"], eps=1e-6,
                               is_masked=True)
    mask = mr_aha.calc_mask(cfg_m, dev["traj"])
    op = mr_aha.create_op(cfg_m, dev["SEs"], dev["phi2"], dev["traj"], mask,
                          method="toeplitz")
    refs["masked"] = xp.asnumpy(op(dev["x"]))
    op.close()
    for method in ("toeplitz", "masked"):
        raw, _, _ = compare(refs[method], refs["nufft"])
        print(f"  MR_AHA {method:<9s} vs MR_AHA nufft : rel diff {raw:.2e}")
    for method, ref in refs.items():
        raw, fit, c = compare(y_jl, ref)
        print(f"  Julia vs MR_AHA {method:<9s}: rel diff {raw:.2e}, after "
              f"scalar fit {fit:.2e} (c = {c.real:.6g}{c.imag:+.2g}i)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
