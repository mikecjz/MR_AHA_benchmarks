# BART benchmark

`bench_bart.py` times the low-rank (subspace) normal operator $A^H A$ of
[BART](https://codeberg.org/mrirecon/bart) (the Berkeley Advanced Reconstruction Toolbox) on the GPU,
next to MR_AHA's on the same problems, GPU and process.

## What BART implements

Checked in the pinned source (`bart/bart`, tag **v1.0.01**):

| Question | Answer | Where |
|---|---|---|
| Low-rank / subspace $A^H A$ | Yes: NUFFT with a temporal basis (`-B` in `pics`, `nufft`), basis $[T, L]$ in dims `TE`×`COEFF` | `src/noncart/nufft.c` `nufft_create2()`, `src/sense/modelnc.c` `sense_nc_init()` |
| Toeplitz embedding | Yes, the default for the normal operator: FFT → $L\times L$ PSF per grid point → IFFT | `toeplitz_mult()` in `src/noncart/nufft.c` |
| Triangular storage | Yes, optional: `--nufft-conf upper-triag-psf` stores $(L^2+L)/2$ PSF blocks | `nufft_conf_opts`, `compute_square_basis()`, `md_ztenmul_upper_triag()` |
| GPU | Yes (`-g`, `--gpu-gridding` for the PSF on the GPU), CUDA build | `Makefile` `CUDA=1` |
| MATLAB / Python interface | Yes, but both (`matlab/bart.m`, `python/bart.py`) run the `bart` executable as a separate process and exchange `.cfl` files; no in-memory operator | `python/bart.py` `bart()` |

How it differs from MR_AHA's Toeplitz operator:

* **All coils at once.** BART's grid holds `Ncoils × L` images; MR_AHA loops over the coils with
  one coil's `L` images in flight.
* **Decomposed oversampling.** The 2×-oversampled grid is split into $2^d$ shifted grids of the
  image size (BART's `decomp`, default). Without `--lowmem` all $2^d$ are in memory at once.
* **No stack-of-stars mode.** A stack is a full 3D problem, oversampled 2× along z as well, so its
  PSF is twice MR_AHA's `phi2_toep`.
* **Gram blocks vs basis.** BART takes a basis with one row per time frame, each frame with its
  own trajectory; MR_AHA takes a Gram block $\Phi_2$ per readout. The benchmark factorises each
  $\Phi_2 = \sum_k u_k u_k^H$ and gives each readout `r` frames (2 for the synthetic blocks) with
  basis rows $u_k$. With that, BART's full-PSF Toeplitz operator equals MR_AHA's up to a real
  scale (BART's FFT normalisation) to ~1e-5.
* **Upper-triangular PSF bug (v1.0.01).** `upper-triag-psf` applies $\overline{\Phi_2}$ where
  the full PSF and the plain NUFFT apply $\Phi_2$: with a complex basis its result is wrong
  (`--check` shows ~0.5 relative error for `bart_toep_triag*`), with a real basis (the usual BART
  case) it agrees to ~1e-5. Timing is unaffected.

## Build BART with CUDA

```bash
git submodule update --init bart/bart
bart/build_bart.sh            # GPU archs "75 80 86 89 90"; or e.g.  bart/build_bart.sh "75"
```

The script builds `bart/bart/bart` with `CUDA=1 OPENBLAS=1 OMP=1` and the timing helper
`bart/bart_aha`. Requirements (Ubuntu names): `build-essential libfftw3-dev liblapacke-dev
libopenblas-dev libpng-dev`, a CUDA toolkit, and **GCC ≥ 12** (BART refuses older ones; the
script picks the newest `gcc-NN` on `PATH`, or set `CC`). `CUDA_BASE` overrides the toolkit found
from `nvcc`.

## Run

Uses the same Python environment as `python/bench_aha.py` (NumPy, CuPy, the MR_AHA library; see
the [top-level README](../README.md); `MR_AHA_ROOT` selects a checkout).

```bash
python bart/bench_bart.py                                   # preset 2d
python bart/bench_bart.py --problem 2d 2d72 stack 3d --check
python bart/bench_bart.py --problem stack --methods bart_toep_triag_lowmem mr_aha_toeplitz
```

One GPU: the first of `CUDA_VISIBLE_DEVICES`. Results are printed and written to
`results/<gpu>/bench_bart_<time>.json`.

| Method | What is timed |
|---|---|
| `bart_toep` | BART Toeplitz $A^H A$, full Hermitian $L\times L$ PSF (BART's default) |
| `bart_toep_triag` | the same with `upper-triag-psf` |
| `bart_toep_triag_lowmem` | the same in BART's `lowmem` mode (one of the $2^d$ PSF factors at a time) |
| `bart_nufft` | no Toeplitz: BART's gridding NUFFT forward + adjoint with the basis |
| `bart_pics` | through `bart.py`: time per CG iteration of `bart pics -g --gpu-gridding -U --nufft-conf upper-triag-psf` (the operator of `bart_toep_triag_lowmem`), from runs with 5 and 55 iterations (process start, file I/O and PSF cancel) |
| `mr_aha_toeplitz`, `mr_aha_nufft` | MR_AHA's operators (`mr_aha.create_op`) for reference |

The `bart_*` operator rows are timed by `bart_aha` (`bart_aha.c`), a ~200-line C program linked
against BART's libraries. It builds exactly the model of `bart pics` (`sense_nc_init()`: maps ∘
NUFFT-with-basis), computes the PSF on the GPU, and times `linop_normal()` with the GPU synchronised
around each call. This is needed because BART's tools have no "apply $A^H A$" command and
`bart.py` cannot keep an operator alive between calls. `bart_pics` is the cross-check through the
official Python interface: one CG iteration costs one $A^H A$ plus a few vector operations.

Options: `--problem`, `--n`, `--L`, `--coils`, `--spokes`, `--samples`, `--data`, `--methods`,
`--reps` (10), `--warmup` (2), `--pics-iters K1 K2` (5 55), `--check`, `--tmpdir` (where the
`.cfl` files go; the 3D presets need a few GB), `--out`. `--help` lists them.

## Results

TITAN RTX (24 GB), CUDA 12.6, BART v1.0.01, one GPU, `--check`. Times are the best median of
2–3 runs; on the shared machine the larger problems varied by up to 2.5× between runs for both
libraries (ranges in brackets). Set-up includes the PSF / Toeplitz kernel.

| Problem | MR_AHA Toeplitz | BART Toeplitz (full PSF) | BART upper-triag | BART upper-triag lowmem | `bart pics` per CG iteration | MR_AHA NUFFT |
|---|---|---|---|---|---|---|
| `2d` 160², L=6, 16 coils | **2.20 ms** (0.05 GB) | 4.70 ms (0.32 GB) | 5.14 ms (0.32 GB) | 5.22 ms (0.17 GB) | 5.1 ms [5.1–5.6] | 21 ms |
| `2d72` with L=24 (320², 12 coils) | **44 ms** (1.3 GB) | 148 ms [148–405] (12.3 GB) | 156 ms [156–198] (7.1 GB) | 155 ms [155–440] (3.9 GB) | 159 ms [159–338] | 296 ms |
| `stack` 160×160×28, L=12, 12 coils, 140×160 | **133 ms** [133–192] (2.8 GB) | out of memory | out of memory | 751 ms [751–1556] (10.9 GB) | 673 ms [673–1562] | 541 ms |
| `3d` 96³, L=12, 12 coils, 5645×96 | **387 ms** [387–985] (6.8 GB) | out of memory | out of memory | 829 ms [829–2104] (13.1 GB) | 1338 ms [1338–2110] | 3619 ms |
| full `2d72` (L=72), `stack`, `3d` presets | 673 / 1012 ms / OOM | out of memory | out of memory | out of memory | out of memory | 722 / 2880 / 11713 ms |

* BART's Toeplitz operator is **2.1–5.6× slower** than MR_AHA's here and needs **2–10× the GPU
  memory** (all coils at once, $2^d$ decomposed grids). On 24 GB, BART fits the stack and 3D
  problems only in lowmem mode, and none of the full-size presets except `2d`. MR_AHA's own full
  `3d` preset (20 GB kernel) does not fit either.
* The upper-triangular PSF saves memory but not time in BART (its triangular multiply kernel is
  slower than the full one).
* `bart pics` per CG iteration, measured through `bart.py`, agrees with the operator timing of
  `bart_aha`, so the helper measures what BART users get.
* Accuracy (`--check`, relative to MR_AHA Toeplitz after the best scalar): BART full PSF 7e-6 (2D) to
  1.6e-5. BART upper-triag 0.3–0.8 with these complex Gram blocks (the conjugation bug above);
  with a real basis 5e-6 to 7e-6 (2D, stack, 3D checked on small problems).

The JSON files are under `results/titan_rtx/bench_bart_*` (not tracked, like all results).

## Files

* `build_bart.sh`: builds BART with CUDA and `bart_aha`.
* `bart_aha.c`: the timing helper.
* `bench_bart.py`: the benchmark.
* `bart/`: BART, pinned submodule (v1.0.01, BSD-3-Clause).
