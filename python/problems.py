"""Benchmark problems for ``bench_aha.py``: presets, synthetic inputs, files.

A problem is the set of arrays the :math:`A^H A` operator of MR_AHA is built
from and applied to:

``SEs``
    Sensitivity maps, ``complex64``, ``(Ncoils, Nz, Ny, Nx)`` or, in 2D,
    ``(Ncoils, Ny, Nx)``.
``phi2``
    Gram blocks :math:`\\Phi\\Phi^H` of the temporal basis (MR-Multitasking
    convention), ``complex64``, ``(L, L, Ntrajs)`` or, for a stack,
    ``(L, L, Nkz, Ntrajs)`` with kz in centred order.
``traj``
    Trajectory in radians, ``float32``, ``(ndim, Nkx, Ntrajs)``.
``x``
    The spatial coefficients the operator is applied to, ``complex64``,
    ``(L, Nz, Ny, Nx)`` or ``(L, Ny, Nx)``.

All arrays in this module are **row-major** NumPy arrays, i.e. the
column-major shapes of the MR_AHA documentation reversed, which is what the
Python interface of the library expects.

Two sources are supported:

* :func:`synthetic` builds a problem from a few sizes (:data:`PRESETS` holds
  four ready-made ones). The run time of the operator depends on the sizes and
  on the geometry of the trajectory, not on the values of the arrays, so the
  maps, Gram blocks and coefficients are simply smooth or random.
* :func:`load` reads a problem from a ``.mat`` or ``.npz`` file, to time a real
  acquisition. :func:`save` writes one, which also documents the layout.

Nothing here touches the GPU or imports ``mr_aha``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Preset:
    """Sizes that define a synthetic problem.

    Attributes:
        traj_type: ``'2D'``, ``'3D'`` or ``'stack'`` (stack of identical 2D
            trajectories along kz).
        n_dim: Image matrix size in column-major order, ``(Nx, Ny)`` or
            ``(Nx, Ny, Nz)``, as in the MR_AHA documentation.
        L: Number of ranks.
        coils: Number of receive coils.
        spokes: Number of readouts (radial spokes), ``Ntrajs``.
        samples: Samples per readout, ``Nkx``.
        desc: One-line description for the report.
    """

    traj_type: str
    n_dim: tuple
    L: int
    coils: int
    spokes: int
    samples: int
    desc: str = ""


#: Ready-made problems (name -> :class:`Preset`). The sizes are those of the
#: MR_AHA benchmark study: a small 2D radial problem, a high-rank 2D radial
#: problem (8.6 GB Toeplitz kernel), a stack-of-stars (14 GB) and a 3D
#: kooshball (20 GB).
PRESETS = {
    "2d": Preset("2D", (160, 160), 6, 16, 377, 160,
                 "2D golden-angle radial"),
    "2d72": Preset("2D", (320, 320), 72, 12, 754, 320,
                   "2D golden-angle radial, high rank"),
    "stack": Preset("stack", (320, 320, 56), 12, 12, 280, 320,
                    "stack-of-stars"),
    "3d": Preset("3D", (160, 160, 160), 12, 12, 15680, 160,
                 "3D golden-means kooshball"),
}


def override(p: Preset, n_dim: Optional[Sequence[int]] = None,
             L: Optional[int] = None, coils: Optional[int] = None,
             spokes: Optional[int] = None,
             samples: Optional[int] = None) -> Preset:
    """Return ``p`` with some of its sizes replaced.

    Args:
        p: The preset to start from.
        n_dim: New matrix size, ``(Nx, Ny)`` or ``(Nx, Ny, Nz)``; must have as
            many entries as the preset's.
        L: New number of ranks.
        coils: New number of coils.
        spokes: New number of readouts.
        samples: New number of samples per readout.

    Returns:
        The modified :class:`Preset`.

    Raises:
        ValueError: If ``n_dim`` has the wrong number of entries.
    """
    kw = {}
    if n_dim is not None:
        n_dim = tuple(int(v) for v in n_dim)
        if len(n_dim) != len(p.n_dim):
            raise ValueError(f"n needs {len(p.n_dim)} values for a "
                             f"{p.traj_type} problem, got {len(n_dim)}")
        kw["n_dim"] = n_dim
    for key, val in (("L", L), ("coils", coils), ("spokes", spokes),
                     ("samples", samples)):
        if val is not None:
            kw[key] = int(val)
    return replace(p, **kw)


def radial_traj(ndim: int, n_traj: int, n_kx: int) -> np.ndarray:
    """Golden-angle (2D) or golden-means (3D) radial trajectory.

    Every readout is a full diameter through the centre of k-space. In 2D the
    angle advances by the golden angle; in 3D the directions follow the
    two-dimensional golden means of Chan et al. (kooshball).

    Args:
        ndim: 2 or 3.
        n_traj: Number of readouts.
        n_kx: Samples per readout.

    Returns:
        ``float32`` array ``(ndim, Nkx, Ntrajs)`` in radians, ``[-pi, pi)``.
    """
    k = np.pi * (np.arange(n_kx) - n_kx // 2) / (n_kx // 2)
    m = np.arange(n_traj)
    if ndim == 2:
        a = m * np.pi * (np.sqrt(5) - 1) / 2
        d = np.stack([np.cos(a), np.sin(a)])
    else:
        phi1, phi2 = 0.4656, 0.6823
        z = (m * phi1) % 1.0 * 2 - 1
        az = 2 * np.pi * ((m * phi2) % 1.0)
        r = np.sqrt(1 - z ** 2)
        d = np.stack([r * np.cos(az), r * np.sin(az), z])
    return (d[:, None, :] * k[None, :, None]).astype(np.float32)


def gram_blocks(rng: np.random.Generator, L: int, n: int) -> np.ndarray:
    """Random Hermitian positive semi-definite Gram blocks.

    Each block is :math:`\\Phi\\Phi^H` for a random rank x time basis
    :math:`\\Phi` with two time points, the MR-Multitasking convention of the
    library (``Phi2 = Phi_rt(:,idx) * Phi_rt(:,idx)'``).

    Args:
        rng: Random generator.
        L: Number of ranks.
        n: Number of blocks.

    Returns:
        ``complex64`` array ``(L, L, n)``, row-major: element ``[j, i, t]`` is
        entry ``(i, j)`` of block ``t``.
    """
    a = rng.standard_normal((n, 2, L)) + 1j * rng.standard_normal((n, 2, L))
    g = np.einsum("nki,nkj->jin", a, a.conj())
    return np.ascontiguousarray(g, dtype=np.complex64)


def coil_maps(n_coils: int, shape: Sequence[int]) -> np.ndarray:
    """Smooth synthetic sensitivity maps.

    Coil ``c`` has a Gaussian magnitude profile centred on a ring around the
    object (alternating between two rings along z in 3D) and a linear phase
    whose direction depends on the coil. The maps are not meant to be
    realistic, only smooth, complex and different from each other.

    Args:
        n_coils: Number of coils.
        shape: Row-major image shape, ``(Ny, Nx)`` or ``(Nz, Ny, Nx)``.

    Returns:
        ``complex64`` array ``(n_coils,) + shape``.
    """
    shape = tuple(int(v) for v in shape)
    axes = [np.linspace(-1.0, 1.0, n, dtype=np.float32) for n in shape]
    y = axes[-2].reshape((1,) * (len(shape) - 2) + (-1, 1))
    x = axes[-1].reshape((1,) * (len(shape) - 1) + (-1,))
    z = axes[0].reshape((-1, 1, 1)) if len(shape) == 3 else None
    out = np.empty((n_coils,) + shape, dtype=np.complex64)
    for c in range(n_coils):
        ang = 2 * np.pi * c / n_coils
        cx, cy = 1.2 * np.cos(ang), 1.2 * np.sin(ang)
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        phase = 0.5 * np.pi * (np.cos(ang) * x + np.sin(ang) * y)
        if z is not None:
            d2 = d2 + (z - (0.5 if c % 2 else -0.5)) ** 2
        out[c] = np.exp(-d2 / 2.0) * np.exp(1j * phase)
    return out


def synthetic(p: Preset, seed: int = 0) -> dict:
    """Build a synthetic problem from its sizes.

    Args:
        p: The sizes (a :class:`Preset`, possibly modified by
            :func:`override`).
        seed: Seed of the random Gram blocks and coefficients.

    Returns:
        ``dict`` with ``traj_type``, ``n_dim`` (column-major order), ``L``,
        ``n_coils``, the row-major NumPy arrays ``SEs``, ``phi2``, ``traj``
        and ``x`` described in the module docstring, and ``desc``.
    """
    rng = np.random.default_rng(seed)
    n_dim = tuple(p.n_dim)
    shape = tuple(reversed(n_dim))
    tt = p.traj_type.lower()
    traj = radial_traj(3 if tt == "3d" else 2, p.spokes, p.samples)
    if tt == "stack":
        nkz = n_dim[2]
        phi2 = gram_blocks(rng, p.L, nkz * p.spokes)
        phi2 = phi2.reshape(p.L, p.L, nkz, p.spokes)
    else:
        phi2 = gram_blocks(rng, p.L, p.spokes)
    x = (rng.standard_normal((p.L,) + shape, dtype=np.float32)
         + 1j * rng.standard_normal((p.L,) + shape, dtype=np.float32))
    size = "x".join(str(v) for v in n_dim)
    desc = (f"{p.desc or p.traj_type} {size}, L={p.L}, {p.coils} coils, "
            f"{p.spokes}x{p.samples} samples"
            + (f" x {n_dim[2]} kz" if tt == "stack" else ""))
    return dict(traj_type=p.traj_type, n_dim=n_dim, L=p.L, n_coils=p.coils,
                SEs=coil_maps(p.coils, shape), phi2=phi2, traj=traj,
                x=x.astype(np.complex64), desc=desc)


def _finish(traj_type: str, SEs, phi2, traj, x, desc: str,
            seed: int) -> dict:
    """Validate loaded arrays and complete them into a problem ``dict``.

    Args:
        traj_type: ``'2D'``, ``'3D'`` or ``'stack'``.
        SEs: Row-major sensitivity maps.
        phi2: Row-major Gram blocks.
        traj: Row-major trajectory.
        x: Row-major coefficients, or ``None`` to draw random ones.
        desc: Description for the report.
        seed: Seed for the random ``x``.

    Returns:
        The problem ``dict`` of :func:`synthetic`.

    Raises:
        ValueError: If the shapes do not fit the trajectory type.
    """
    tt = str(traj_type).strip().lower()
    names = {"2d": "2D", "3d": "3D", "stack": "stack"}
    if tt not in names:
        raise ValueError(f"traj_type must be 2D, 3D or stack, got {traj_type!r}")
    SEs = np.ascontiguousarray(SEs, dtype=np.complex64)
    phi2 = np.ascontiguousarray(phi2, dtype=np.complex64)
    traj = np.ascontiguousarray(traj, dtype=np.float32)
    n_sp = 2 if tt == "2d" else 3
    if SEs.ndim != n_sp + 1:
        raise ValueError(f"SEs must have {n_sp + 1} dimensions for a "
                         f"{names[tt]} problem, got shape {SEs.shape}")
    want = 4 if tt == "stack" else 3
    if phi2.ndim != want or phi2.shape[0] != phi2.shape[1]:
        raise ValueError(f"phi2 must be (L, L, {'Nkz, ' if tt == 'stack' else ''}"
                         f"Ntrajs), got shape {phi2.shape}")
    ndim_traj = 3 if tt == "3d" else 2
    if traj.ndim != 3 or traj.shape[0] != ndim_traj:
        raise ValueError(f"traj must be ({ndim_traj}, Nkx, Ntrajs), "
                         f"got shape {traj.shape}")
    shape = SEs.shape[1:]
    L = int(phi2.shape[0])
    if x is None:
        rng = np.random.default_rng(seed)
        x = (rng.standard_normal((L,) + shape, dtype=np.float32)
             + 1j * rng.standard_normal((L,) + shape, dtype=np.float32))
    x = np.ascontiguousarray(x, dtype=np.complex64)
    if x.shape != (L,) + shape:
        raise ValueError(f"x must be {(L,) + shape}, got {x.shape}")
    n_dim = tuple(reversed(shape))
    size = "x".join(str(v) for v in n_dim)
    full = (f"{desc}: {names[tt]} {size}, L={L}, {SEs.shape[0]} coils, "
            f"{traj.shape[2]}x{traj.shape[1]} samples")
    return dict(traj_type=names[tt], n_dim=n_dim, L=L, n_coils=int(SEs.shape[0]),
                SEs=SEs, phi2=phi2, traj=traj, x=x, desc=full)


def _mat_v73(path: Path) -> dict:
    """Read the variables of a MATLAB v7.3 (HDF5) file.

    HDF5 lists the dimensions of a MATLAB array in reverse, so the arrays
    come out row-major without any transposition.

    Args:
        path: The ``.mat`` file.

    Returns:
        ``dict`` of NumPy arrays (complex where MATLAB stored complex data)
        and, for character arrays, ``str``.
    """
    import h5py  # noqa: PLC0415

    out = {}
    with h5py.File(path, "r") as h:
        for key, ds in h.items():
            if not isinstance(ds, h5py.Dataset):
                continue
            a = ds[()]
            if a.dtype.names and "real" in a.dtype.names:
                a = a["real"] + 1j * a["imag"]
            if ds.attrs.get("MATLAB_class", b"") == b"char":
                a = "".join(chr(int(c)) for c in np.asarray(a).ravel())
            out[key] = a
    return out


def load(path, traj_type: Optional[str] = None, seed: int = 0) -> dict:
    """Load a problem from a ``.mat`` or ``.npz`` file.

    ``.mat`` (any version): variables ``SEs``, ``phi2``, ``traj`` in the
    **column-major** layout of the MR_AHA documentation and of its MATLAB
    interface, ``SEs`` ``[Nx, Ny, (Nz,) Ncoils]``, ``phi2``
    ``[Ntrajs, (Nkz,) L, L]``, ``traj`` ``[Ntrajs, Nkx, ndim]``; they are
    converted to row-major here. ``.npz``: the same names in the
    **row-major** layout of the Python interface (module docstring).

    Both may also hold ``traj_type`` (``'2D'``, ``'3D'`` or ``'stack'``) and
    ``x`` (coefficients to apply the operator to; random when absent).

    Args:
        path: The file.
        traj_type: Trajectory type; overrides the one stored in the file and
            is required when the file has none.
        seed: Seed of the random ``x`` when the file has none.

    Returns:
        The problem ``dict`` of :func:`synthetic`.

    Raises:
        ValueError: If a variable is missing or has the wrong shape.
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        rev = False
    else:
        try:
            import scipy.io as sio  # noqa: PLC0415

            d = {k: v for k, v in sio.loadmat(path).items()
                 if not k.startswith("__")}
            rev = True          # loadmat keeps MATLAB's dimension order
        except NotImplementedError:
            d = _mat_v73(path)  # v7.3: dimensions already reversed
            rev = False
    for key in ("SEs", "phi2", "traj"):
        if key not in d:
            raise ValueError(f"{path}: variable {key!r} is missing")
    tt = traj_type
    if tt is None:
        if "traj_type" not in d:
            raise ValueError(f"{path} has no traj_type; pass --traj-type")
        tt = d["traj_type"]
        if isinstance(tt, np.ndarray):
            tt = "".join(str(v) for v in tt.ravel())

    def arr(key):
        a = d.get(key)
        if a is None:
            return None
        return np.transpose(a) if rev else a

    return _finish(tt, arr("SEs"), arr("phi2"), arr("traj"), arr("x"),
                   path.name, seed)


def save(problem: dict, path) -> None:
    """Write a problem to a ``.mat`` or ``.npz`` file readable by :func:`load`.

    A ``.mat`` file (MATLAB v7 format, each variable below 2 GB) holds the
    arrays in the column-major layout, so the MATLAB benchmark and the MATLAB
    interface of the library can use it directly.

    Args:
        problem: A problem ``dict``.
        path: Output file, ``.mat`` or ``.npz``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ("SEs", "phi2", "traj", "x")
    if path.suffix == ".npz":
        np.savez(path, traj_type=problem["traj_type"],
                 **{k: problem[k] for k in keys})
        return
    import scipy.io as sio  # noqa: PLC0415

    sio.savemat(path, dict(traj_type=problem["traj_type"],
                           **{k: np.transpose(problem[k]) for k in keys}))
