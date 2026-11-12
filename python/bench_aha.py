#!/usr/bin/env python
"""Time the :math:`A^H A` operator of MR_AHA: Toeplitz vs NUFFT, 1 or more GPUs.

One command builds a test problem, creates the forward normal operator with
each selected method on each selected number of GPUs, and reports how long the
set-up and one application take.

Methods (``--methods``)
-----------------------
``toeplitz``
    :func:`mr_aha.create_op` with ``method='toeplitz'``. The Toeplitz kernel
    is computed at creation; one application is, per coil, a zero-padded FFT,
    a block-diagonal multiplication and an inverse FFT. One or several GPUs.
``nufft_c``
    :func:`mr_aha.create_op` with ``method='nufft'``: per coil a type-2
    NUFFT, the Gram multiplication and a type-1 NUFFT, all inside the C/CUDA
    library (cufinufft's C API). One or several GPUs.
``nufft_native``
    :class:`mr_aha.aha_nufft_plan`: the same NUFFT operator written in Python
    with CuPy and cufinufft's own Python interface, without the library.
    Always one GPU. On Apple silicon it runs on MLX and mlx-nufft.

Problems (``--problem``)
------------------------
Synthetic presets (see :mod:`problems`): ``2d`` (radial 160 x 160, L = 6),
``2d72`` (radial 320 x 320, L = 72), ``stack`` (stack-of-stars
320 x 320 x 56, L = 12) and ``3d`` (kooshball 160^3, L = 12). ``--n``,
``--L``, ``--coils``, ``--spokes`` and ``--samples`` change the sizes of the
selected presets. ``--data FILE`` times the problem stored in a ``.mat`` or
``.npz`` file instead (layout: :func:`problems.load`).

What is reported
----------------
For every method and GPU count:

``set-up``
    Wall-clock time of creating the operator (kernel, plans, copies to the
    other GPUs), all GPUs synchronised. With ``--setup-reps N > 1`` the
    operator is created ``N`` times: the first creation is reported as
    ``cold`` and the median of the others as ``set-up``. The first creation
    on a GPU also pays for one-off initialisation (CUDA context, cuFFT,
    cufinufft); a tiny warm-up problem takes that cost on the current GPU
    before the first table. ``kernel_s`` in the JSON is the part spent on the
    Toeplitz kernel.
``A^H A``
    Median wall-clock time of one application over ``--reps`` runs after
    ``--warmup`` untimed runs.
``speed-up``
    Time of the same method on one GPU divided by this row's time (when one
    GPU was also run).
``mem``
    Device memory, summed over the GPUs, in use after the warm-up runs minus
    before the operator was created: what the operator holds plus the
    temporaries and the result of one application.
``CG`` (``--cg N``)
    Time of ``N`` plain conjugate-gradient iterations with the operator.
``diff`` (``--check``)
    Relative difference of this row's ``A^H A x`` to the first row's
    (marked ``ref``). Unmasked rows agree to about the NUFFT tolerance
    (``--eps``). A masked Toeplitz operator is an approximation and differs
    more, about 1e-2 on the random ``x`` used here (far less on images);
    with ``--mask on`` it is the first row and so the reference.

Examples
--------
::

    python bench_aha.py                                # preset "2d", all methods, 1 GPU
    python bench_aha.py --problem stack 3d --gpus 1 2 4
    python bench_aha.py --problem stack --methods toeplitz --mask both
    python bench_aha.py --problem 3d --n 96 96 96 --L 8 --check --cg 10
    python bench_aha.py --data my_scan.mat --methods toeplitz nufft_c

The library is taken from ``$MR_AHA_ROOT`` if that is set, otherwise from the
``third_party/MR_AHA`` submodule of this repository, otherwise from an
installed ``mr_aha`` package; ``$MR_AHA_LIBRARY`` names the shared library
explicitly. Results are printed and written as JSON to
``results/<gpu>/bench_aha_<time>.json`` (or ``--out``), where ``<gpu>`` is
the GPU model, e.g. ``titan_rtx``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import problems

#: Root of the benchmarks repository.
REPO = Path(__file__).resolve().parent.parent

#: Methods in report order.
METHODS = ("toeplitz", "nufft_c", "nufft_native")

#: The ``mr_aha`` package, its backend (``'cuda'`` or ``'metal'``) and the
#: array module of that backend (CuPy or NumPy); set by :func:`setup`.
mr_aha = None
BACKEND = None
xp = None


def setup() -> None:
    """Import ``mr_aha`` and select the array module of its backend.

    The package is looked for in ``$MR_AHA_ROOT/python``, then in the
    ``third_party/MR_AHA`` submodule, then among the installed packages.
    Sets the module globals :data:`mr_aha`, :data:`BACKEND` and :data:`xp`.

    Raises:
        SystemExit: With directions if the package or its shared library
            cannot be loaded.
    """
    global mr_aha, BACKEND, xp  # noqa: PLW0603
    root = os.environ.get("MR_AHA_ROOT")
    cand = Path(root).expanduser() if root else REPO / "third_party" / "MR_AHA"
    if (cand / "python" / "mr_aha").is_dir():
        sys.path.insert(0, str(cand / "python"))
    elif root:
        raise SystemExit(f"MR_AHA_ROOT={root}: no python/mr_aha in there")
    try:
        import mr_aha as _m  # noqa: PLC0415
    except (ImportError, OSError) as e:
        raise SystemExit(
            f"cannot load mr_aha: {e}\n"
            "Build the library first (see the README of this repository):\n"
            "  git submodule update --init --recursive\n"
            "  cmake -S third_party/MR_AHA -B third_party/MR_AHA/build "
            "-DMR_AHA_BUILD_PYTHON=ON\n"
            "  cmake --build third_party/MR_AHA/build -j\n"
            "or point MR_AHA_ROOT at an existing checkout with a build/ folder."
        ) from e
    mr_aha = _m
    BACKEND = mr_aha.backend()
    if BACKEND == "cuda":
        import cupy  # noqa: PLC0415

        xp = cupy
    else:
        xp = np


# --------------------------------------------------------------------------
# Devices, synchronisation, memory
# --------------------------------------------------------------------------

def visible_gpus() -> int:
    """Number of GPUs this process can use (1 on the Metal backend)."""
    if BACKEND != "cuda":
        return 1
    return int(xp.cuda.runtime.getDeviceCount())


def pick_devices(n: int) -> list:
    """The ``n`` CUDA devices a run uses: the current one first.

    Args:
        n: Number of GPUs.

    Returns:
        CUDA ordinals, the current device (which the library always uses as
        the primary one) followed by the lowest-numbered others.
    """
    if BACKEND != "cuda":
        return [0]
    cur = int(xp.cuda.Device().id)
    return [cur] + [d for d in range(visible_gpus()) if d != cur][: n - 1]


def sync_all(devices) -> None:
    """Wait until all queued work on ``devices`` is done (no-op on Metal)."""
    if BACKEND != "cuda":
        return
    for d in devices:
        xp.cuda.Device(d).synchronize()


def free_pools(devices) -> None:
    """Return CuPy's cached memory blocks on ``devices`` to the driver."""
    if BACKEND != "cuda":
        return
    for d in devices:
        with xp.cuda.Device(d):
            xp.get_default_memory_pool().free_all_blocks()


def free_bytes(devices):
    """Free device memory summed over ``devices``.

    Returns:
        Bytes, or ``None`` on the Metal backend (unified memory, no
        per-process figure).
    """
    if BACKEND != "cuda":
        return None
    return sum(int(xp.cuda.Device(d).mem_info[0]) for d in devices)


def gpu_info() -> dict:
    """Describe the machine: GPUs and the versions that matter for timing.

    Returns:
        ``dict`` with ``backend``, ``gpus`` (name and memory of
        every visible GPU) and the versions of ``mr_aha`` and of the array and
        NUFFT packages in use.
    """
    out = dict(backend=BACKEND,
               mr_aha=getattr(mr_aha, "__version__", "unknown"),
               numpy=np.__version__)
    if BACKEND != "cuda":
        import platform  # noqa: PLC0415

        name = "Apple GPU"
        try:
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True,
                                  check=True).stdout.strip() or name
        except (OSError, subprocess.CalledProcessError):
            pass
        out["gpus"] = [dict(name=name or platform.processor())]
        return out
    gpus = []
    for d in range(visible_gpus()):
        props = xp.cuda.runtime.getDeviceProperties(d)
        gpus.append(dict(id=d, name=props["name"].decode(),
                         mem_gb=round(props["totalGlobalMem"] / 2**30, 1)))
    out.update(gpus=gpus, cupy=xp.__version__,
               cuda_runtime=int(xp.cuda.runtime.runtimeGetVersion()))
    try:
        import cufinufft  # noqa: PLC0415

        out["cufinufft"] = cufinufft.__version__
    except ImportError:
        out["cufinufft"] = None
    return out


def gpu_folder(name: str) -> str:
    """Folder name for a GPU model: ``"NVIDIA TITAN RTX"`` -> ``"titan_rtx"``.

    The vendor and the memory size and type (``80GB``, ``HBM3``, ``PCIe``,
    ``SXM``) are dropped, so ``"NVIDIA H100 80GB HBM3"`` -> ``"h100"`` and
    ``"Apple M3 Ultra"`` -> ``"m3_ultra"``.
    """
    words = re.findall(r"[a-z0-9]+", name.lower())
    drop = re.compile(r"nvidia|apple|\d+gb|hbm\d*|pcie|sxm\d*")
    return "_".join(w for w in words if not drop.fullmatch(w)) or "gpu"


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------

def native_available() -> bool:
    """Whether the ``nufft_native`` method can run here.

    It needs cufinufft's Python package on CUDA and mlx-nufft on Metal.
    """
    try:
        if BACKEND == "cuda":
            import cufinufft  # noqa: F401, PLC0415
        else:
            import mlx_nufft  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def make_op(method: str, prob: dict, devices, masked: bool, args):
    """Create one operator.

    Args:
        method: ``'toeplitz'``, ``'nufft_c'`` or ``'nufft_native'``.
        prob: Problem on the device (see :func:`to_device`).
        devices: CUDA devices to use; one entry means the current GPU only.
        masked: Toeplitz only: build the masked operator, with the mask of
            :func:`mr_aha.calc_mask`.
        args: Parsed command line (``eps``, ``toep_dist``, ``upsampfac``,
            ``gpu_method``).

    Returns:
        ``(apply, close, info)``: a callable ``x -> A^H A x``, a callable
        releasing the operator, and a ``dict`` of facts about it (``devices``
        actually used, ``kernel_s``, ``mask_s``, ``toep_source``).
    """
    kw = dict(n_coils=prob["n_coils"], eps=args.eps)
    multi = dict(devices=list(devices)) if len(devices) > 1 else {}
    nopts = {}
    if args.upsampfac is not None:
        nopts["upsampfac"] = float(args.upsampfac)
    if args.gpu_method is not None:
        nopts["gpu_method"] = int(args.gpu_method)
    SEs, phi2, traj = prob["SEs"], prob["phi2"], prob["traj"]

    if method == "toeplitz":
        cfg = mr_aha.make_config(prob["traj_type"], prob["n_dim"], prob["L"],
                                 is_masked=masked, **kw)
        mask, mask_s = None, 0.0
        if masked:
            t0 = time.perf_counter()
            mask = mr_aha.calc_mask(cfg, traj)
            sync_all(devices[:1])
            mask_s = time.perf_counter() - t0
        op = mr_aha.create_op(cfg, SEs, phi2, traj, mask, method="toeplitz",
                              toep_dist=args.toep_dist, **multi)
        inf = op.info
        return op, op.close, dict(
            devices=list(inf["devices"]), kernel_s=inf["toep_ms"] / 1e3,
            mask_s=mask_s, toep_source=inf["toep_source"],
            mask_keep=(float(mask.size) / float(np.prod(
                [cfg.phi2_dim[i] for i in range(len(prob["n_dim"]))]))
                if masked else None))

    cfg = mr_aha.make_config(prob["traj_type"], prob["n_dim"], prob["L"], **kw)
    if method == "nufft_c":
        op = mr_aha.create_op(cfg, SEs, phi2, traj, method="nufft",
                              nufft_opts=nopts or None, **multi)
        return op, op.close, dict(devices=list(op.info["devices"]))

    plan = mr_aha.aha_nufft_plan(cfg, traj, **nopts)
    return (lambda v: plan.apply(v, SEs, phi2)), plan.close, dict(
        devices=list(devices[:1]))


def cg_time(apply, b, n_iter: int, devices) -> float:
    """Wall-clock time of ``n_iter`` plain conjugate-gradient iterations.

    Solves ``A^H A u = b`` from ``u = 0`` without a stopping test, so that
    every operator does the same amount of work.

    Args:
        apply: The operator.
        b: Right-hand side.
        n_iter: Number of iterations.
        devices: Devices to synchronise before and after.

    Returns:
        Seconds.
    """
    if BACKEND == "cuda":
        from cupy import cublas  # noqa: PLC0415

        def dot(a, c):
            return complex(cublas.dotc(a.ravel(), c.ravel())).real
    else:
        def dot(a, c):
            return float(np.vdot(a.ravel(), c.ravel()).real)

    dot(b, b)                       # loads cuBLAS before the clock starts
    sync_all(devices)
    t0 = time.perf_counter()
    u = xp.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = dot(r, r)
    for _ in range(n_iter):
        Ap = apply(p)
        alpha = rs / dot(p, Ap)
        u += np.complex64(alpha) * p
        r -= np.complex64(alpha) * Ap
        rs_new = dot(r, r)
        p *= np.complex64(rs_new / rs)
        p += r
        rs = rs_new
        del Ap
    sync_all(devices)
    return time.perf_counter() - t0


def rel_diff(a, b) -> float:
    """Relative 2-norm difference ``||a - b|| / ||b||``."""
    return float(xp.linalg.norm((a - b).ravel()) / xp.linalg.norm(b.ravel()))


def run_one(method: str, prob: dict, n_gpus: int, masked: bool, args,
            ref) -> tuple:
    """Create, time and release one operator.

    Args:
        method: The method.
        prob: Problem on the device.
        n_gpus: Number of GPUs to ask for.
        masked: Toeplitz only: the masked operator.
        args: Parsed command line.
        ref: ``A^H A x`` of the first row of this problem, or ``None``.

    Returns:
        ``(row, y)``: the result row (a ``dict`` ready for JSON) and, with
        ``--check``, this operator's ``A^H A x`` (else ``None``).
    """
    devices = pick_devices(n_gpus)
    x = prob["x"]
    row = dict(method=method, masked=bool(masked), gpus=n_gpus)
    free_pools(devices)
    mem0 = free_bytes(devices)

    # Set-up: the first creation is "cold" within this configuration.
    times, close = [], None
    for i in range(args.setup_reps):
        if close is not None:
            close()
            free_pools(devices)
        sync_all(devices)
        t0 = time.perf_counter()
        apply, close, info = make_op(method, prob, devices, masked, args)
        sync_all(devices)
        times.append(time.perf_counter() - t0)
    try:
        row.update(info)
        row["gpus_used"] = len(info["devices"])
        row["setup_cold_s"] = times[0]
        row["setup_s"] = float(np.median(times[1:])) if len(times) > 1 else times[0]

        # One application.
        y = None
        for _ in range(args.warmup):
            del y
            y = apply(x)
        sync_all(devices)
        mem1 = free_bytes(devices)
        if mem0 is not None:
            row["mem_gb"] = (mem0 - mem1) / 1e9
        ts = []
        for _ in range(args.reps):
            del y
            sync_all(devices)
            t0 = time.perf_counter()
            y = apply(x)
            sync_all(devices)
            ts.append(time.perf_counter() - t0)
        ts = np.array(ts)
        row.update(aha_ms=float(np.median(ts)) * 1e3,
                   aha_min_ms=float(ts.min()) * 1e3,
                   aha_mean_ms=float(ts.mean()) * 1e3, reps=int(args.reps))

        if args.check and ref is not None:
            row["rel_diff"] = rel_diff(y, ref)
        if not args.check:
            y = None
        if args.cg:
            row.update(cg_s=cg_time(apply, x, args.cg, devices),
                       cg_iters=int(args.cg))
    finally:
        close()
        free_pools(devices)
    return row, y


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def label(row: dict) -> str:
    """Method name of a row as printed (``toeplitz masked`` for the mask)."""
    return row["method"] + (" masked" if row.get("masked") else "")


def fmt_row(row: dict, args) -> str:
    """Format one result row of the table."""
    if "error" in row:
        return f"  {label(row):<16s} {row['gpus']:>4d}   failed: {row['error']}"
    used = row["gpus_used"]
    gpus = f"{used}" if used == row["gpus"] else f"{row['gpus']}>{used}"
    s = (f"  {label(row):<16s} {gpus:>4s} {row['setup_s']:9.3f} "
         f"{row['aha_ms']:10.2f}")
    show = row.get("speedup") and row["gpus"] > 1
    s += f" {row['speedup']:8.2f}" if show else f" {'':>8s}"
    s += f" {row['mem_gb']:8.2f}" if "mem_gb" in row else f" {'n/a':>8s}"
    if args.cg:
        s += f" {row['cg_s']:8.3f}"
    if args.check:
        s += f" {row['rel_diff']:9.1e}" if "rel_diff" in row else f" {'ref':>9s}"
    return s


def header(args) -> str:
    """Column headings matching :func:`fmt_row`."""
    s = (f"  {'method':<16s} {'GPUs':>4s} {'set-up s':>9s} {'A^H A ms':>10s} "
         f"{'speed-up':>8s} {'mem GB':>8s}")
    if args.cg:
        s += f" {'CG s':>8s}"
    if args.check:
        s += f" {'diff':>9s}"
    return s


def add_speedups(rows: list) -> None:
    """Add ``speedup`` (one-GPU time / this time) to the rows, in place."""
    base = {(r["method"], r["masked"]): r["aha_ms"] for r in rows
            if "aha_ms" in r and r["gpus"] == 1}
    for r in rows:
        b = base.get((r["method"], r["masked"]))
        if b and "aha_ms" in r:
            r["speedup"] = b / r["aha_ms"]


def warm_up(methods, args) -> None:
    """Pay the one-off start-up costs of the process before anything is timed.

    Creates and applies every selected method once on a tiny 2D problem on
    the current GPU, which loads cuFFT, cuBLAS and cufinufft and compiles
    CuPy's kernels. Without it the first row of the first table would carry
    these costs in its set-up time. The other GPUs of a multi-GPU run are
    still initialised by their first operator; use ``--setup-reps`` to
    separate that.

    Args:
        methods: Methods about to be run.
        args: Parsed command line.
    """
    tiny = problems.Preset("2D", (32, 32), 2, 2, 16, 32)
    dev = to_device(problems.synthetic(tiny))
    devices = pick_devices(1)
    for method in methods:
        try:
            apply, close, _ = make_op(method, dev, devices, False, args)
            try:
                apply(dev["x"])
                if args.cg:
                    cg_time(apply, dev["x"], 1, devices)
                sync_all(devices)
            finally:
                close()
        except Exception:  # noqa: BLE001 - the real run reports failures
            pass
    del dev
    free_pools(devices)


def to_device(prob: dict) -> dict:
    """Copy the arrays of a problem to the current GPU (CUDA backend).

    Args:
        prob: Problem with NumPy arrays (:mod:`problems`).

    Returns:
        The same ``dict`` with ``SEs``, ``phi2``, ``traj`` and ``x`` as
        arrays of the backend (CuPy on CUDA, unchanged NumPy on Metal).
    """
    out = dict(prob)
    for k in ("SEs", "phi2", "traj", "x"):
        out[k] = xp.asarray(prob[k])
    return out


def run_problem(name: str, prob: dict, methods, gpus, masks, args) -> dict:
    """Run every selected configuration on one problem and print its table.

    Args:
        name: Name for the report (preset or file name).
        prob: Problem with NumPy arrays.
        methods: Methods to run.
        gpus: GPU counts to run.
        masks: Toeplitz variants, a subset of ``(False, True)`` (unmasked,
            masked).
        args: Parsed command line.

    Returns:
        ``dict`` for the JSON report: ``name``, ``desc``, sizes, ``kernel_gb``
        (size of the unmasked Toeplitz kernel) and ``rows``.
    """
    cfg = mr_aha.make_config(prob["traj_type"], prob["n_dim"], prob["L"],
                             n_coils=prob["n_coils"])
    n_sp = len(prob["n_dim"])
    kernel_gb = (mr_aha.num_tri(prob["L"])
                 * int(np.prod([cfg.phi2_dim[i] for i in range(n_sp)])) * 8 / 1e9)
    print(f"\n== {name}: {prob['desc']}\n   unmasked Toeplitz kernel "
          f"{kernel_gb:.2f} GB", flush=True)
    print(header(args), flush=True)
    dev = to_device(prob)
    rows, ref = [], None
    for method in methods:
        variants = masks if method == "toeplitz" else (False,)
        counts = gpus if method != "nufft_native" else [1]
        for masked in variants:
            for g in counts:
                row = dict(method=method, masked=bool(masked), gpus=g)
                try:
                    row, y = run_one(method, dev, g, masked, args, ref)
                    if args.check and ref is None:
                        ref = y
                    del y
                except Exception as e:  # noqa: BLE001 - report and go on
                    row["error"] = f"{type(e).__name__}: {e}".splitlines()[0]
                    free_pools(pick_devices(g))
                rows.append(row)
                add_speedups(rows)
                print(fmt_row(row, args), flush=True)
    del dev, ref
    free_pools(pick_devices(visible_gpus()))
    return dict(name=name, desc=prob["desc"], traj_type=prob["traj_type"],
                n_dim=list(prob["n_dim"]), L=prob["L"], n_coils=prob["n_coils"],
                kernel_gb=kernel_gb, rows=rows)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    """Parse the command line (see the module docstring for the meaning)."""
    ap = argparse.ArgumentParser(
        description="Time the A^H A operator of MR_AHA: Toeplitz vs NUFFT, "
                    "one or several GPUs.",
        epilog="presets: " + "; ".join(
            f"{k} = {p.desc} {'x'.join(map(str, p.n_dim))}, L={p.L}, "
            f"{p.coils} coils" for k, p in problems.PRESETS.items()))
    g = ap.add_argument_group("problem")
    g.add_argument("--problem", nargs="+", default=["2d"],
                   choices=list(problems.PRESETS), metavar="NAME",
                   help="synthetic presets to run: %(choices)s (default: 2d)")
    g.add_argument("--n", type=int, nargs="+", default=None, metavar="N",
                   help="matrix size Nx Ny [Nz] instead of the preset's")
    g.add_argument("--L", type=int, default=None, help="number of ranks")
    g.add_argument("--coils", type=int, default=None, help="number of coils")
    g.add_argument("--spokes", type=int, default=None,
                   help="number of readouts")
    g.add_argument("--samples", type=int, default=None,
                   help="samples per readout")
    g.add_argument("--data", default=None, metavar="FILE",
                   help=".mat or .npz file with SEs, phi2, traj (and "
                        "traj_type, x) to time instead of a preset")
    g.add_argument("--traj-type", default=None, choices=["2D", "3D", "stack"],
                   help="trajectory type of --data when the file has none")
    g.add_argument("--save-data", default=None, metavar="FILE",
                   help="also write the (first) problem to this .mat or .npz "
                        "file, e.g. for the MATLAB benchmark")
    g.add_argument("--seed", type=int, default=0,
                   help="seed of the random inputs (default: 0)")

    g = ap.add_argument_group("what to run")
    g.add_argument("--methods", nargs="+", default=None, choices=METHODS,
                   metavar="METHOD",
                   help="%(choices)s (default: all that are available)")
    g.add_argument("--gpus", nargs="+", default=["1"], metavar="G",
                   help="GPU counts to run, e.g. 1 2 4, or 'all' "
                        "(default: 1); nufft_native always uses one")
    g.add_argument("--mask", default="off", choices=["off", "on", "both"],
                   help="Toeplitz operator: unmasked (off, default), masked "
                        "(on) or both")
    g.add_argument("--toep-dist", default="auto",
                   choices=["auto", "split", "replicate", "broadcast"],
                   help="Toeplitz on several GPUs: how the kernel reaches "
                        "every GPU (default: auto)")
    g.add_argument("--upsampfac", type=float, default=None,
                   help="NUFFT methods: cufinufft upsampling factor "
                        "(default: 2)")
    g.add_argument("--gpu-method", type=int, default=None, choices=[1, 2],
                   help="NUFFT methods: cufinufft gpu_method for both "
                        "transforms (default: the library's choice)")
    g.add_argument("--eps", type=float, default=1e-6,
                   help="NUFFT tolerance (default: 1e-6)")

    g = ap.add_argument_group("measurement")
    g.add_argument("--reps", type=int, default=10,
                   help="timed applications (default: 10)")
    g.add_argument("--warmup", type=int, default=2,
                   help="untimed applications first (default: 2)")
    g.add_argument("--setup-reps", type=int, default=1,
                   help="operator creations; above 1 the first is reported "
                        "as cold and the median of the rest as set-up "
                        "(default: 1)")
    g.add_argument("--cg", type=int, default=0, metavar="N",
                   help="also time N conjugate-gradient iterations")
    g.add_argument("--check", action="store_true",
                   help="report the relative difference of every A^H A x to "
                        "the first row's")
    g.add_argument("--out", default=None, metavar="FILE",
                   help="JSON output file (default: "
                        "results/<gpu>/bench_aha_<time>.json)")
    args = ap.parse_args(argv)
    if args.reps < 1 or args.warmup < 1 or args.setup_reps < 1:
        ap.error("--reps, --warmup and --setup-reps must be at least 1")
    return args


def main(argv=None) -> int:
    """Run the benchmark described by the command line.

    Returns:
        0 if every configuration ran, 1 if any failed.
    """
    args = parse_args(argv)
    setup()

    # Methods: the requested ones, minus those that cannot run here.
    methods = list(args.methods or METHODS)
    if "nufft_native" in methods and not native_available():
        need = "cufinufft" if BACKEND == "cuda" else "mlx-nufft"
        print(f"skipping nufft_native: the Python package {need} is not "
              "installed")
        methods.remove("nufft_native")

    # GPU counts: integers or 'all', limited to the visible devices.
    n_vis = visible_gpus()
    gpus = []
    for g in args.gpus:
        n = n_vis if str(g).lower() == "all" else int(g)
        if n < 1:
            raise SystemExit("--gpus: counts must be at least 1")
        if n > n_vis:
            print(f"skipping {n} GPUs: only {n_vis} visible")
        elif n not in gpus:
            gpus.append(n)
    if not gpus:
        raise SystemExit("no usable GPU count in --gpus")
    masks = {"off": (False,), "on": (True,), "both": (False, True)}[args.mask]

    # Problems.
    todo = []
    if args.data:
        todo.append((Path(args.data).stem,
                     problems.load(args.data, args.traj_type, args.seed)))
    else:
        for name in args.problem:
            try:
                p = problems.override(problems.PRESETS[name], args.n, args.L,
                                      args.coils, args.spokes, args.samples)
            except ValueError as e:
                raise SystemExit(f"--problem {name}: {e}") from e
            todo.append((name, p))

    info = gpu_info()
    names = ", ".join(sorted({g["name"] for g in info["gpus"]}))
    print(f"backend {BACKEND}, {n_vis} GPU(s) visible: "
          f"{names}")
    for var in ("MR_AHA_TOEP_SPLIT", "MR_AHA_MGPU_SPLIT", "MR_AHA_MGPU_MIN_WORK"):
        if os.environ.get(var):
            print(f"note: {var}={os.environ[var]} is set and overrides the "
                  "library's defaults")

    report = dict(info, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                  args={k: v for k, v in vars(args).items()},
                  env={k: v for k, v in os.environ.items()
                       if k.startswith("MR_AHA_") or k == "CUDA_VISIBLE_DEVICES"},
                  problems=[])
    out = Path(args.out) if args.out else (
        REPO / "results" / gpu_folder(info["gpus"][0]["name"])
        / f"bench_aha_{time.strftime('%Y%m%d-%H%M%S')}.json")

    warm_up(methods, args)
    for i, (name, p) in enumerate(todo):
        prob = problems.synthetic(p, args.seed) if isinstance(
            p, problems.Preset) else p
        if args.save_data and i == 0:
            problems.save(prob, args.save_data)
            print(f"problem written to {args.save_data}")
        report["problems"].append(run_problem(name, prob, methods, gpus,
                                              masks, args))
        del prob
        # Saved after every problem, so that a later failure keeps the rest.
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
    print(f"\nresults written to {out}")
    failed = sum("error" in r for pr in report["problems"] for r in pr["rows"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
