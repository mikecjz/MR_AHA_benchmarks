#!/usr/bin/env python
"""Time BART's low-rank (subspace) :math:`A^H A` on the GPU next to MR_AHA's.

`BART <https://codeberg.org/mrirecon/bart>`_ (the Berkeley Advanced
Reconstruction Toolbox, pinned here as the submodule ``bart/bart``) evaluates
the normal operator of a subspace non-Cartesian SENSE model the same way as
MR_AHA's Toeplitz method:

* the temporal basis is passed with ``-B`` (``nufft_create2()`` with a basis,
  ``src/noncart/nufft.c``);
* with Toeplitz embedding (BART's default inside ``pics``) the operator is an
  FFT, a multiplication by an :math:`L \\times L` point-spread function (PSF)
  per grid point, and an inverse FFT (``toeplitz_mult()``);
* ``--nufft-conf upper-triag-psf`` stores only the upper triangle of the
  Hermitian PSF, :math:`(L^2+L)/2` blocks, as MR_AHA's triangular
  ``phi2_toep``.

Differences that matter for speed: BART processes all coils at once (one grid
of ``Ncoils x L`` images instead of one coil at a time), it splits the
2x-oversampled grid into :math:`2^d` shifted grids of the image size
("decomposed" Toeplitz) and loops over them only in ``--lowmem`` mode, and it
has no stack-of-stars mode, so a stack is a full 3D problem with 2x
oversampling along z too.

Methods (``--methods``)
-----------------------
``bart_toep``
    BART Toeplitz normal operator, full Hermitian :math:`L \\times L` PSF.
``bart_toep_triag``
    The same with the upper-triangular PSF (``upper-triag-psf``).
``bart_toep_triag_lowmem``
    The same in BART's low-memory mode (one of the :math:`2^d` PSF factors at
    a time).
``bart_nufft``
    No Toeplitz embedding: BART's gridding NUFFT forward and adjoint with the
    basis.
``bart_pics``
    End-to-end through BART's own Python interface (``bart.py``): the time per
    conjugate-gradient iteration of ``bart pics -g``, from two runs with
    different iteration counts (start-up, file I/O and PSF cancel out). An
    iteration is one :math:`A^H A` plus a few vector operations.
``mr_aha_toeplitz``, ``mr_aha_nufft``
    MR_AHA's Toeplitz and NUFFT operators (``mr_aha.create_op``) on the same
    problem and GPU, for reference.

The ``bart_*`` operator methods (all but ``bart_pics``) are timed by the
small C program ``bart_aha`` (``bart_aha.c``), linked against BART's
libraries: BART's tools have no command that applies only the normal
operator, and ``bart.py`` runs tools as separate processes that exchange
``.cfl`` files, so an operator cannot stay alive between calls. ``bart_aha``
builds exactly the model of ``bart pics`` (``sense_nc_init()``), synchronises
the GPU around every timed call and prints JSON; this script writes its
inputs with BART's ``cfl.py`` and reads its result.

Problem mapping
---------------
The problems are those of ``python/problems.py`` (same presets, same
trajectories, same Gram blocks). MR_AHA takes one Gram block
:math:`\\Phi_2 = \\Phi(:,\\mathrm{idx})\\Phi(:,\\mathrm{idx})^H` per readout
(per readout and kz for a stack); BART takes a temporal basis with one row per
time frame, each frame with its own trajectory. Every Gram block is therefore
factorised (eigen-decomposition, :math:`\\Phi_2 = \\sum_k u_k u_k^H`) and each
readout becomes ``r`` frames (``r`` = the largest rank, 2 for the synthetic
blocks) that share its trajectory, with basis rows :math:`u_k` (with this
choice BART's full-PSF Toeplitz operator agrees with MR_AHA to single
precision). The trajectory is converted from radians to BART's units (cycles
per FOV).

.. note::
   In BART v1.0.01 the upper-triangular PSF (``upper-triag-psf``) applies the
   conjugate of the Gram blocks that the full PSF and the plain NUFFT apply,
   i.e. it is wrong for a complex basis (``--check`` shows a large error for
   ``bart_toep_triag`` with the synthetic complex Gram blocks) and right for a
   real basis, the usual case in BART. The run time does not depend on this.
A stack-of-stars readout at kz index ``p`` gets the 3D coordinate
``kz = p - Nz/2``.

Run ``--check`` to compare BART's and MR_AHA's :math:`A^H A x`: BART's NUFFT is
scaled differently, so the relative error is reported after the best complex
scalar fit, and the scalar itself.

Examples
--------
::

    python bart/bench_bart.py                                   # preset 2d
    python bart/bench_bart.py --problem 2d 2d72 --check
    python bart/bench_bart.py --problem stack --n 160 160 28 --methods bart_toep_triag mr_aha_toeplitz
    python bart/bench_bart.py --problem 3d --methods bart_toep_triag_lowmem mr_aha_toeplitz

Requirements: BART built by ``bart/build_bart.sh`` (``bart/bart/bart`` and
``bart/bart_aha``), and the MR_AHA Python interface with CuPy (as for
``python/bench_aha.py``; ``MR_AHA_ROOT`` selects the checkout). One GPU: the
first of ``CUDA_VISIBLE_DEVICES``. Results are printed and written to
``results/<gpu>/bench_bart_<time>.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BART_DIR = HERE / "bart"
BART_AHA = HERE / "bart_aha"

sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(BART_DIR / "python"))
os.environ.setdefault("BART_TOOLBOX_PATH", str(BART_DIR))

import problems  # noqa: E402

#: Terminal colour codes in BART's messages.
ANSI = re.compile(r"\x1b\[[0-9;]*m")

#: Methods in report order.
METHODS = ("bart_toep", "bart_toep_triag", "bart_toep_triag_lowmem",
           "bart_nufft", "bart_pics", "mr_aha_toeplitz", "mr_aha_nufft")

#: Default methods.
DEFAULT_METHODS = ("bart_toep", "bart_toep_triag", "bart_toep_triag_lowmem",
                   "bart_pics", "mr_aha_toeplitz")

#: Options of ``bart_aha`` for every BART operator method.
BART_FLAGS = {
    "bart_toep": [],
    "bart_toep_triag": ["--upper-triag"],
    "bart_toep_triag_lowmem": ["--upper-triag", "--lowmem"],
    "bart_nufft": ["--no-toeplitz"],
}


# --------------------------------------------------------------------------
# Converting a problem to BART's layout
# --------------------------------------------------------------------------

def gram_factors(phi2: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Factorise Hermitian PSD Gram blocks, :math:`\\Phi_2 = \\sum_k u_k u_k^H`.

    Args:
        phi2: Row-major Gram blocks of :mod:`problems`, ``(L, L, n)``:
            element ``[j, i, t]`` is entry ``(i, j)`` of block ``t``.
        tol: Eigenvalues below ``tol`` times the largest one are dropped.

    Returns:
        ``complex64`` array ``(n, r, L)`` with the vectors :math:`u_k` of every
        block, ``r`` the largest numerical rank (zero vectors pad blocks of
        lower rank).
    """
    L, n = phi2.shape[0], phi2.shape[-1]
    M = np.transpose(phi2.reshape(L, L, n), (2, 1, 0)).astype(np.complex128)
    M = 0.5 * (M + np.conj(np.transpose(M, (0, 2, 1))))     # exact Hermitian
    w, V = np.linalg.eigh(M)                                  # ascending
    w = np.clip(w, 0.0, None)
    keep = w > tol * max(float(w.max()), 1e-30)
    r = max(int(keep.sum(axis=1).max()), 1)
    w, V = w[:, -r:], V[:, :, -r:]
    U = V * np.sqrt(w)[:, None, :]                            # (n, L, r)
    return np.ascontiguousarray(np.transpose(U, (0, 2, 1)), dtype=np.complex64)


def to_bart(prob: dict) -> dict:
    """Convert a :mod:`problems` problem to BART arrays.

    The returned arrays are NumPy arrays whose index order is BART's
    dimension order (``cfl.writecfl`` writes them column-major):

    * ``traj`` ``[3, Nkx, 1, 1, 1, T]`` in cycles per FOV;
    * ``basis`` ``[1, 1, 1, 1, 1, T, L]``;
    * ``maps`` ``[Nx, Ny, Nz, Ncoils]``;
    * ``x`` ``[Nx, Ny, Nz, 1, 1, 1, L]``.

    Args:
        prob: Problem ``dict`` of :mod:`problems` (row-major NumPy arrays).

    Returns:
        ``dict`` with the four arrays and ``frames`` (T) and ``rank`` (frames
        per readout).
    """
    tt = prob["traj_type"].lower()
    n_dim = list(prob["n_dim"]) + [1] * (3 - len(prob["n_dim"]))
    Nx, Ny, Nz = n_dim
    L = prob["L"]
    traj = np.asarray(prob["traj"], dtype=np.float64)        # (ndim, Nkx, Ntrajs)
    nkx, ntr = traj.shape[1], traj.shape[2]

    # Readout coordinates in cycles per FOV: radians * N / (2 pi).
    k = np.zeros((3, nkx, ntr))
    for d in range(traj.shape[0]):
        k[d] = traj[d] * n_dim[d] / (2 * np.pi)

    if tt == "stack":
        # One readout per (spoke, kz); kz in centred order p - Nz/2.
        phi2 = prob["phi2"]                                   # (L, L, Nkz, Ntrajs)
        nkz = phi2.shape[2]
        U = gram_factors(phi2.reshape(L, L, nkz * ntr))       # (nkz*ntr, r, L): t = kz*ntr + s
        kk = np.repeat(k[:, :, None, :], nkz, axis=2)         # (3, Nkx, Nkz, Ntrajs)
        kk[2] = (np.arange(nkz) - nkz // 2)[None, :, None]
        kk = kk.reshape(3, nkx, nkz * ntr)
    else:
        U = gram_factors(prob["phi2"])                        # (ntr, r, L)
        kk = k
    n_read, r = U.shape[0], U.shape[1]

    # Frame f = readout * r + k: trajectory of the readout, basis row u_k.
    traj_b = np.repeat(kk, r, axis=2)                         # (3, Nkx, n_read*r)
    T = n_read * r
    basis = U.reshape(T, L)
    out = dict(
        traj=traj_b.reshape(3, nkx, 1, 1, 1, T).astype(np.complex64),
        basis=basis.reshape(1, 1, 1, 1, 1, T, L),
        maps=np.transpose(prob["SEs"]).reshape(Nx, Ny, Nz, -1),
        x=np.transpose(prob["x"]).reshape(Nx, Ny, Nz, 1, 1, 1, L),
        frames=T, rank=r)
    return out


def from_bart_image(y: np.ndarray, prob: dict) -> np.ndarray:
    """BART image ``[Nx, Ny, Nz, 1, 1, 1, L]`` -> row-major ``(L, (Nz,) Ny, Nx)``."""
    shape = tuple(reversed(prob["n_dim"]))
    y = np.asarray(y).reshape(-1, prob["L"], order="F")      # (Nx*Ny*Nz, L)
    return np.ascontiguousarray(y.T.reshape((prob["L"],) + shape))


# --------------------------------------------------------------------------
# BART runs
# --------------------------------------------------------------------------

def write_inputs(b: dict, folder: Path) -> dict:
    """Write the BART arrays as ``.cfl``/``.hdr`` files with BART's ``cfl.py``.

    Returns:
        ``dict`` name -> file stem.
    """
    import cfl  # noqa: PLC0415  (bart/bart/python/cfl.py)

    names = {}
    for key in ("traj", "basis", "maps", "x"):
        stem = str(folder / key)
        cfl.writecfl(stem, b[key])
        names[key] = stem
    return names


def run_bart_op(method: str, files: dict, args, out_stem=None) -> dict:
    """Time one BART operator with ``bart_aha``.

    Args:
        method: A key of :data:`BART_FLAGS`.
        files: Input file stems (:func:`write_inputs`).
        args: Parsed command line (``reps``, ``warmup``, ``timeout``).
        out_stem: If given, ``A^H A x`` is written there.

    Returns:
        The JSON of ``bart_aha`` (``setup_s``, ``apply_ms``, ``mem_gb``) or
        ``{"error": ...}``.
    """
    cmd = [str(BART_AHA), "-g", *BART_FLAGS[method],
           "--reps", str(args.reps), "--warmup", str(args.warmup)]
    if out_stem:
        cmd += ["--out", out_stem]
    cmd += [files[k] for k in ("traj", "basis", "maps", "x")]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return dict(error=f"timeout after {args.timeout} s")
    if p.returncode != 0:
        msg = ANSI.sub("", p.stderr or p.stdout)
        if "BART_GPU_GLOBAL_MEMORY" in msg or "out of memory" in msg.lower():
            return dict(error="GPU out of memory")
        tail = msg.strip().splitlines()[-3:]
        return dict(error=f"exit {p.returncode}: " + " | ".join(tail))
    return json.loads(p.stdout.strip().splitlines()[-1])


def run_bart_pics(b: dict, prob: dict, args) -> dict:
    """Time a conjugate-gradient iteration of ``bart pics -g`` through ``bart.py``.

    ``pics`` is run twice, with ``--pics-iters`` ``K1`` and ``K2`` iterations,
    on random k-space data, with the upper-triangular Toeplitz PSF computed on
    the GPU (``--gpu-gridding``) and the low-memory NUFFT (``-U``), i.e. the
    operator of ``bart_toep_triag_lowmem`` (the only one that fits the larger
    problems on a 24 GB GPU). The time per iteration is
    ``(t(K2) - t(K1)) / (K2 - K1)``; ``t`` is the wall-clock time of the
    ``bart.bart`` call, which includes writing and reading the ``.cfl`` files,
    starting the process and computing the PSF, all of which cancel.

    Returns:
        ``dict`` with ``iter_ms``, ``t1_s``, ``t2_s``, ``k1``, ``k2`` or
        ``{"error": ...}``.
    """
    import bart as bartpy  # noqa: PLC0415  (bart/bart/python/bart.py)

    rng = np.random.default_rng(1)
    nkx, T = b["traj"].shape[1], b["frames"]
    nc = b["maps"].shape[3]
    ksp = (rng.standard_normal((1, nkx, 1, nc, 1, T), dtype=np.float32)
           + 1j * rng.standard_normal((1, nkx, 1, nc, 1, T), dtype=np.float32))
    k1, k2 = args.pics_iters
    times = []
    for K in (k1, k2):
        cmd = (f"pics -g --gpu-gridding -U -l2 -r 0.001 -i {K} "
               f"--nufft-conf upper-triag-psf")
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):   # bart.py echoes the tool's output
            out = bartpy.bart(1, cmd, ksp, b["maps"], t=b["traj"],
                              B=b["basis"])
        times.append(time.perf_counter() - t0)
        if bartpy.bart.ERR or out is None:
            msg = ANSI.sub("", bartpy.bart.stderr or "")
            if "BART_GPU_GLOBAL_MEMORY" in msg or "out of memory" in msg.lower():
                return dict(error="GPU out of memory")
            err = msg.strip().splitlines()[-3:]
            return dict(error="bart pics failed: " + " | ".join(err))
    return dict(iter_ms=1e3 * (times[1] - times[0]) / (k2 - k1),
                t1_s=times[0], t2_s=times[1], k1=k1, k2=k2)


# --------------------------------------------------------------------------
# MR_AHA reference
# --------------------------------------------------------------------------

def run_mr_aha(method: str, prob: dict, args):
    """Time MR_AHA's operator on the current GPU with ``python/bench_aha.py``.

    Returns:
        ``(row, y)``: ``dict`` with ``setup_s``, ``apply_ms``, ``mem_gb``,
        and the result ``A^H A x`` as a NumPy array (or ``None``).
    """
    import bench_aha as ba  # noqa: PLC0415  (python/bench_aha.py)

    if ba.mr_aha is None:
        ba.setup()
    xp = ba.xp
    dev = [int(xp.cuda.Device().id)]
    d = ba.to_device(prob)
    ns = argparse.Namespace(eps=args.eps, toep_dist="auto", upsampfac=None,
                            gpu_method=None)
    ba.free_pools(dev)
    free0 = ba.free_bytes(dev)
    ba.sync_all(dev)
    t0 = time.perf_counter()
    op, close, _ = ba.make_op("toeplitz" if method == "mr_aha_toeplitz"
                              else "nufft_c", d, dev, False, ns)
    ba.sync_all(dev)
    setup_s = time.perf_counter() - t0
    for _ in range(args.warmup):
        y = op(d["x"])
    ba.sync_all(dev)
    mem = (free0 - ba.free_bytes(dev)) / 2**30
    ms = []
    for _ in range(args.reps):
        ba.sync_all(dev)
        t1 = time.perf_counter()
        y = op(d["x"])
        ba.sync_all(dev)
        ms.append(1e3 * (time.perf_counter() - t1))
    y = xp.asnumpy(y)
    close()
    del op, d
    ba.free_pools(dev)
    return dict(setup_s=setup_s, apply_ms=ms, mem_gb=mem), y


def compare(a: np.ndarray, ref: np.ndarray) -> dict:
    """Relative difference of ``a`` to ``ref`` after the best complex scale.

    Returns:
        ``dict`` with ``rel_err`` (``|s a - ref| / |ref|`` for the least-squares
        scalar ``s``), ``scale`` (``s`` as ``[re, im]``) and ``rel_err_raw``
        (without scaling).
    """
    a = a.ravel().astype(np.complex128)
    r = ref.ravel().astype(np.complex128)
    s = np.vdot(a, r) / np.vdot(a, a)
    nr = np.linalg.norm(r)
    return dict(rel_err=float(np.linalg.norm(s * a - r) / nr),
                scale=[float(s.real), float(s.imag)],
                rel_err_raw=float(np.linalg.norm(a - r) / nr))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def gpu_name() -> str:
    """Name of the first visible GPU (``nvidia-smi``), or ``'gpu'``."""
    try:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
        out = subprocess.run(["nvidia-smi", "-i", vis, "--query-gpu=name",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, check=True).stdout.strip()
        return out.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "gpu"


def bart_version() -> str:
    """Output of ``bart version`` (the pinned submodule)."""
    try:
        return subprocess.run([str(BART_DIR / "bart"), "version"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def fmt(row: dict) -> str:
    """One line of the report table."""
    name = row["method"]
    if "error" in row:
        return f"  {name:<24s} FAILED: {row['error'][:110]}"
    if name == "bart_pics":
        return (f"  {name:<24s} {'':>9s} {row['iter_ms']:10.2f}   "
                f"(per CG iteration; {row['k1']}/{row['k2']} it: "
                f"{row['t1_s']:.2f}/{row['t2_s']:.2f} s)")
    med = float(np.median(row["apply_ms"]))
    s = f"  {name:<24s} {row['setup_s']:9.3f} {med:10.2f} {row['mem_gb']:8.2f}"
    if "check" in row:
        c = row["check"]
        s += (f"   err {c['rel_err']:.1e} (scale {c['scale'][0]:.3g}"
              f"{c['scale'][1]:+.1e}i)")
    return s


def run_problem(name: str, prob: dict, methods, args) -> dict:
    """Run all methods on one problem; print and return the rows."""
    print(f"\n== {name}: {prob['desc']}")
    b = to_bart(prob)
    print(f"   BART: {b['frames']} frames ({b['rank']} per readout), "
          f"basis {b['frames']} x {prob['L']}")
    print(f"  {'method':<24s} {'set-up s':>9s} {'A^H A ms':>10s} {'mem GB':>8s}")
    rows, ref = [], None
    with tempfile.TemporaryDirectory(prefix="bench_bart_",
                                     dir=args.tmpdir) as tmp:
        tmp = Path(tmp)
        files = write_inputs(b, tmp)
        # MR_AHA first when checking, so that it is the reference.
        order = sorted(methods, key=lambda m: (not m.startswith("mr_aha"),
                                               METHODS.index(m)))
        for m in order:
            if m.startswith("mr_aha"):
                try:
                    row, y = run_mr_aha(m, prob, args)
                except Exception as e:  # noqa: BLE001  (report and go on)
                    row, y = dict(error=f"{type(e).__name__}: {e}"), None
                if ref is None and y is not None:
                    ref = y
                elif args.check and y is not None:
                    row["check"] = compare(y, ref)
            elif m == "bart_pics":
                row = run_bart_pics(b, prob, args)
            else:
                out = str(tmp / f"y_{m}") if args.check else None
                row = run_bart_op(m, files, args, out)
                if args.check and "error" not in row and ref is not None:
                    import cfl  # noqa: PLC0415

                    y = from_bart_image(cfl.readcfl(out), prob)
                    row["check"] = compare(y, ref)
            row["method"] = m
            rows.append(row)
            print(fmt(row), flush=True)
    return dict(problem=name, desc=prob["desc"], traj_type=prob["traj_type"],
                n_dim=list(prob["n_dim"]), L=prob["L"],
                n_coils=prob["n_coils"], frames=b["frames"], rows=rows)


def parse_args(argv=None) -> argparse.Namespace:
    """Command line; see the module docstring."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", nargs="+", default=["2d"],
                   choices=sorted(problems.PRESETS), help="presets (default 2d)")
    p.add_argument("--n", nargs="+", type=int, help="matrix size Nx Ny [Nz]")
    p.add_argument("--L", type=int)
    p.add_argument("--coils", type=int)
    p.add_argument("--spokes", type=int)
    p.add_argument("--samples", type=int)
    p.add_argument("--data", help=".mat or .npz problem file (python/README.md)")
    p.add_argument("--traj-type", help="2D, 3D or stack, for --data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--methods", nargs="+", choices=METHODS,
                   default=list(DEFAULT_METHODS))
    p.add_argument("--reps", type=int, default=10, help="timed applications")
    p.add_argument("--warmup", type=int, default=2, help="untimed applications")
    p.add_argument("--pics-iters", nargs=2, type=int, default=[5, 55],
                   metavar=("K1", "K2"), help="bart_pics iteration counts")
    p.add_argument("--eps", type=float, default=1e-6, help="MR_AHA NUFFT tolerance")
    p.add_argument("--check", action="store_true",
                   help="compare every A^H A x with MR_AHA's Toeplitz result")
    p.add_argument("--timeout", type=float, default=3600, help="per BART run, s")
    p.add_argument("--tmpdir", help="folder for the .cfl files (default: system temp)")
    p.add_argument("--out", help="JSON output file")
    a = p.parse_args(argv)
    if a.check and "mr_aha_toeplitz" not in a.methods:
        a.methods = ["mr_aha_toeplitz"] + list(a.methods)
    return a


def main(argv=None) -> int:
    """Entry point."""
    args = parse_args(argv)
    for exe in (BART_DIR / "bart", BART_AHA):
        if not exe.exists():
            raise SystemExit(f"{exe} is missing: run bart/build_bart.sh first")
    gpu = gpu_name()
    print(f"GPU: {gpu}   BART {bart_version()}")
    if args.data:
        probs = [(Path(args.data).name,
                  problems.load(args.data, args.traj_type, args.seed))]
    else:
        probs = []
        for name in args.problem:
            pr = problems.override(problems.PRESETS[name], args.n, args.L,
                                   args.coils, args.spokes, args.samples)
            probs.append((name, problems.synthetic(pr, args.seed)))
    results = [run_problem(n, pr, args.methods, args) for n, pr in probs]

    sys.path.insert(0, str(REPO / "python"))
    import bench_aha as ba  # noqa: PLC0415

    out = Path(args.out) if args.out else (
        REPO / "results" / ba.gpu_folder(gpu)
        / time.strftime("bench_bart_%Y%m%d_%H%M%S.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(gpu=gpu, bart=bart_version(), argv=sys.argv[1:],
                reps=args.reps, warmup=args.warmup)
    out.write_text(json.dumps(dict(meta=meta, results=results), indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
