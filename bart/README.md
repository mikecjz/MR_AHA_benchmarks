# BART benchmark

`bench_bart.py` times the low-rank (subspace) normal operator $A^H A$ of
[BART](https://codeberg.org/mrirecon/bart) (the Berkeley Advanced Reconstruction Toolbox) on the GPU,
next to MR_AHA's on the same problems, GPU and process.

## What BART implements

Checked in the pinned source (`bart/bart`, tag **v1.0.01**). The submodule is the fork
[mikecjz/bart](https://github.com/mikecjz/bart), branch `v1.0.01-container`: BART v1.0.01 with
unchanged C sources, plus a `Dockerfile` and wrappers (`bart.py`, `bart.m`) that can run BART
from a container image.

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

## Get BART: native build or container image

`bench_bart.py` can run BART in two ways. Both need the submodule's sources (`cfl.py` and
`bart.py` are used from it):

```bash
git submodule update --init bart/bart
```

### Native build (workstation)

```bash
bart/build_bart.sh            # GPU archs "75 80 86 89 90"; or e.g.  bart/build_bart.sh "75"
```

The script builds `bart/bart/bart` with `CUDA=1 OPENBLAS=1 OMP=1` and the timing helper
`bart/bart_aha`. Requirements (Ubuntu names): `build-essential libfftw3-dev liblapacke-dev
libopenblas-dev libpng-dev`, a CUDA toolkit, and **GCC ≥ 12** (BART refuses older ones; the
script picks the newest `gcc-NN` on `PATH`, or set `CC`). `CUDA_BASE` overrides the toolkit found
from `nvcc`.

### Container image (HPC cluster)

Where GCC 12 is missing, BART runs from an Apptainer / Singularity image instead. The image is
[`mikecjz/bart`](https://hub.docker.com/r/mikecjz/bart) on Docker Hub, built from the
`Dockerfile` of the submodule (BART v1.0.01, CUDA 12.6, the same build options as above).

```bash
module load apptainer                       # or singularity
bart/build_bart.sh --sif                    # image -> bart/bart.sif, helper -> bart/bart_aha_sif
bart/build_bart.sh --sif /path/bart.sif     # the same with an image somewhere else
```

The script downloads the image (`docker://mikecjz/bart:v1.0.01-devel`, 3.9 GB; `BART_IMAGE`
names another one) unless the file exists, and compiles the timing helper inside it against
BART's libraries in `/opt/bart`. No root rights are needed. The `devel` image is required because
of the helper; the smaller `mikecjz/bart:v1.0.01` holds only the `bart` binary.

The SIF file is itself a `bart` executable (`bart/bart.sif version`, and
`APPTAINER_NV=1 bart/bart.sif pics -g ...` on a GPU), and BART's Python and MATLAB wrappers use
it when `BART_SIF` names it (`doc/docker.md` in the submodule).

## Run

Uses the same Python environment as `python/bench_aha.py` (NumPy, CuPy, the MR_AHA library; see
the [top-level README](../README.md); `MR_AHA_ROOT` selects a checkout).

```bash
python bart/bench_bart.py                                   # preset 2d
python bart/bench_bart.py --problem 2d 2d72 stack 3d --check
python bart/bench_bart.py --problem stack --methods bart_toep_triag_lowmem mr_aha_toeplitz
python bart/bench_bart.py --problem 2d --methods bart_pics_cg mr_aha_toeplitz --cg 20
```

With a container image, add `--sif` (or set `BART_SIF`):

```bash
python bart/bench_bart.py --sif bart/bart.sif --problem 2d --check
```

`bart_aha_sif` and `bart pics` (through `bart.py`) then run inside the container with `--nv`;
MR_AHA runs natively in the Python process. The operator times are taken inside `bart_aha` and
do not contain the start of the container. The end-to-end time of `bart_pics_cg` does: a few
tenths of a second per `pics` call.

### Out of memory, and problems that fit the GPU

A BART method that cannot allocate GPU memory is reported as `GPU out of memory` with the
evidence that the limit is the GPU of that run: the allocation BART failed on (from its own
message), the memory of the GPU, and what other processes held when BART started. The numbers
are also in the JSON (`oom`).

`--fit` adapts a preset to the GPU instead: if BART cannot fit it even in its most frugal mode
(upper-triangular PSF, low-memory mode), the preset is shrunk step by step until it does, and
all methods run on that problem. The stack loses slices first (56, 52, ... 28) and in-plane
matrix after that, the kooshball shrinks from 160³ in steps of 8 with its spokes per matrix
area kept, and `2d72` loses ranks (72, 64, ... 24). The reduced problem is therefore the
largest of these that the GPU at hand holds; the sizes that failed are kept in the JSON (`fit`).

```bash
python bart/bench_bart.py --problem 2d72 stack 3d --fit --check
```

On a Slurm cluster, [`hpc/bench_bart.sbatch`](hpc/bench_bart.sbatch) does everything in one job
(MR_AHA for the GPU, the image, the helper, and four sets of runs: the problems that fit a
24 GB GPU, the presets as they are, the presets fitted to the GPU, and 20 CG iterations):

```bash
mkdir -p bart/hpc/logs
sbatch --gres=gpu:l40s:1 --export=ALL,ARCH=89 bart/hpc/bench_bart.sbatch
sbatch --gres=gpu:h100:1 --export=ALL,ARCH=90 bart/hpc/bench_bart.sbatch
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
| `bart_pics_cg` | through `bart.py`: `--cg N` (default 20) CG iterations of `bart pics -g --gpu-gridding` with BART's default full PSF, or `-U` (lowmem) if that runs out of memory, or `-U` with the upper-triangular PSF if that does too. Runs with 1 and N+1 iterations, fastest of `--pics-reps` (3) each; `cg_s` is the difference, `total_s` the whole `pics` call with N iterations (process start, `.cfl` I/O, PSF, scaling, adjoint, CG). BART's CG has no stopping tolerance by default, so every run does all its iterations |
| `mr_aha_toeplitz`, `mr_aha_nufft` | MR_AHA's operators (`mr_aha.create_op`) for reference; with `--cg N` also N plain CG iterations (`bench_aha.cg_time`, after one untimed iteration) and `total_s` = inputs to the GPU + operator creation + CG |

The `bart_*` operator rows are timed by `bart_aha` (`bart_aha.c`), a ~200-line C program linked
against BART's libraries. It builds exactly the model of `bart pics` (`sense_nc_init()`: maps ∘
NUFFT-with-basis), computes the PSF on the GPU, and times `linop_normal()` with the GPU synchronised
around each call. This is needed because BART's tools have no "apply $A^H A$" command and
`bart.py` cannot keep an operator alive between calls. `bart_pics` is the cross-check through the
official Python interface: one CG iteration costs one $A^H A$ plus a few vector operations.

Options: `--problem`, `--n`, `--L`, `--coils`, `--spokes`, `--samples`, `--data`, `--methods`,
`--reps` (10), `--warmup` (2), `--pics-iters K1 K2` (5 55), `--cg N`, `--pics-reps` (3), `--check`, `--fit`, `--tmpdir` (where the
`.cfl` files go; the 3D presets need a few GB), `--sif`, `--out`. `--help` lists them.

## Results

MR_AHA [`c60db28`](https://github.com/mikecjz/MR_AHA/tree/c60db28), BART v1.0.01, one GPU,
`--check`. On the TITAN RTX (24 GB) BART is the native build with CUDA 12.6. On the L40S (45 GB
usable) and the H100 (80 GB) BART runs from the container image and MR_AHA is built on the node
with CUDA 12.9; those numbers come from `hpc/bench_bart.sbatch`, one job per GPU. The TITAN RTX
numbers are the faster of two runs on a shared machine. Set-up includes the PSF / Toeplitz
kernel. Memory is the GPU memory a method holds after its warm-up.

### Problems fitted to each GPU

BART cannot hold all presets on any of the three GPUs. `--fit` shrinks a preset only as far as
BART needs in its most frugal mode (upper-triangular PSF, low memory), so each GPU gets the
largest problem of the ladder that BART fits on it (BART's memory in brackets):

| Preset | TITAN RTX (24 GB) | L40S (45 GB) | H100 (80 GB) |
|---|---|---|---|
| `2d72` (320², L=72) | **L=64** (22.8 GB) | as it is (28.4 GB) | as it is (28.4 GB) |
| `stack` (320²×56, L=12) | **256²×28** (17.2 GB) | **320²×32** (30.7 GB) | **320²×48** (46.0 GB) |
| `3d` (160³, L=12) | **128³** (19.6 GB) | **144³** (27.9 GB) | as it is (38.3 GB) |

### One $A^H A$ on the fitted problems

**TITAN RTX**

| Problem | MR_AHA Toeplitz | BART full PSF | BART upper-triag | BART upper-triag, low memory | Best BART / MR_AHA | `bart pics` per CG iteration | MR_AHA NUFFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2d` 160², L=6 | **2.16 ms** (0.05 GB) | 4.75 ms (0.32 GB) | 5.13 ms (0.32 GB) | 5.23 ms (0.17 GB) | 2.2× | 4.7 ms | 20 ms |
| `2d72` 320², L=64 | **175 ms** (7.2 GB) | out of memory | out of memory | 985 ms (22.8 GB) | 5.6× | 988 ms | 627 ms |
| `stack` 256²×28 | **302 ms** (7.2 GB) | out of memory | out of memory | 1714 ms (17.2 GB) | 5.7× | 1717 ms | 828 ms |
| `3d` 128³ | **740 ms** (16.1 GB) | out of memory | out of memory | 1913 ms (19.6 GB) | 2.6× | 1920 ms | 3795 ms |

**L40S**

| Problem | MR_AHA Toeplitz | BART full PSF | BART upper-triag | BART upper-triag, low memory | Best BART / MR_AHA | `bart pics` per CG iteration | MR_AHA NUFFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2d` 160², L=6 | **0.44 ms** (0.05 GB) | 1.52 ms (0.36 GB) | 2.42 ms (0.36 GB) | 1.24 ms (0.21 GB) | 2.8× | 5.1 ms | 13 ms |
| `2d72` 320², L=72 | **90 ms** (9.0 GB) | out of memory | out of memory | 705 ms (28.4 GB) | 7.8× | 710 ms | 434 ms |
| `stack` 320²×32 | **417 ms** (12.9 GB) | out of memory | out of memory | 2521 ms (30.7 GB) | 6.0× | 2537 ms | 913 ms |
| `3d` 144³ | **792 ms** (23.0 GB) | out of memory | out of memory | 2332 ms (27.9 GB) | 2.9× | 2339 ms | 4210 ms |

**H100**

| Problem | MR_AHA Toeplitz | BART full PSF | BART upper-triag | BART upper-triag, low memory | Best BART / MR_AHA | `bart pics` per CG iteration | MR_AHA NUFFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2d` 160², L=6 | **0.53 ms** (0.05 GB) | 1.31 ms (0.38 GB) | 2.26 ms (0.39 GB) | 1.64 ms (0.23 GB) | 2.5× | 1.8 ms | 6.62 ms |
| `2d72` 320², L=72 | **36 ms** (9.0 GB) | out of memory | out of memory | 277 ms (28.4 GB) | 7.7× | 278 ms | 201 ms |
| `stack` 320²×48 | **154 ms** (19.3 GB) | out of memory | out of memory | 912 ms (46.0 GB) | 5.9× | 919 ms | 685 ms |
| `3d` 160³ | **253 ms** (31.5 GB) | out of memory | out of memory | 762 ms (38.3 GB) | 3.0× | 767 ms | 2052 ms |

* MR_AHA's Toeplitz operator is **2.1–7.8× faster** than BART's fastest mode on every problem and
  GPU, and holds less GPU memory.
* The gap is largest on `2d72` and on the stack. For ranks above 30 MR_AHA `c60db28` has tiled
  block-diagonal kernels; before them `2d72` took 335 ms on the H100, more than BART's 277 ms.
  BART has no stack-of-stars mode, so it treats a stack as a 3D problem oversampled along z.
* On these problems BART runs only with the upper-triangular PSF in low-memory mode. The
  upper-triangular PSF saves memory but not time (its multiply kernel is slower than the full
  one where both run).
* `bart pics` per CG iteration, measured through `bart.py`, agrees with the operator timing of
  `bart_aha`, so the helper measures what BART users get. On the tiny `2d` problem the
  per-iteration time through the container is not reliable: the start of the container varies
  by more than the iterations take.
* Accuracy (`--check`, relative to MR_AHA Toeplitz after the best scalar): BART full PSF 7e-6
  (`2d`) and 1.6e-5 (`2d72`, L=24), the same on all three GPUs. BART upper-triag 0.2–0.9 with
  these complex Gram blocks (the conjugation bug above). MR_AHA's NUFFT method 8e-7 to 2e-6.

### The same problems on all three GPUs

The problems that BART fits on 24 GB, for a comparison between GPUs. MR_AHA Toeplitz / BART's
fastest mode (ratio):

| Problem | TITAN RTX | L40S | H100 |
|---|---:|---:|---:|
| `2d` 160², L=6 | 2.16 ms / 4.75 ms (2.2×) | 0.44 ms / 1.24 ms (2.8×) | 0.53 ms / 1.31 ms (2.5×) |
| `2d72` 320², L=24 | 43 ms / 150 ms (3.5×) | 24 ms / 94 ms (4.0×) | 9.00 ms / 25 ms (2.8×) |
| `stack` 160²×28 | 121 ms / 671 ms (5.6×) | 90 ms / 557 ms (6.2×) | 22 ms / 138 ms (6.3×) |
| `3d` 96³ | 313 ms / 829 ms (2.6×) | 232 ms / 684 ms (3.0×) | 50 ms / 169 ms (3.4×) |

The three modes of BART on these problems:

**TITAN RTX**

| Problem | MR_AHA Toeplitz | BART full PSF | BART upper-triag | BART upper-triag, low memory | `bart pics` per CG iteration | MR_AHA NUFFT |
|---|---:|---:|---:|---:|---:|---:|
| `2d72` 320², L=24 | **43 ms** (1.3 GB) | 150 ms (12.3 GB) | 157 ms (7.1 GB) | 155 ms (3.9 GB) | 156 ms | 224 ms |
| `stack` 160²×28 | **121 ms** (2.8 GB) | out of memory | out of memory | 671 ms (10.9 GB) | 677 ms | 351 ms |
| `3d` 96³ | **313 ms** (6.8 GB) | out of memory | out of memory | 829 ms (13.1 GB) | 821 ms | 1694 ms |

**L40S**

| Problem | MR_AHA Toeplitz | BART full PSF | BART upper-triag | BART upper-triag, low memory | `bart pics` per CG iteration | MR_AHA NUFFT |
|---|---:|---:|---:|---:|---:|---:|
| `2d72` 320², L=24 | **24 ms** (1.3 GB) | 95 ms (12.4 GB) | 103 ms (7.2 GB) | 94 ms (4.0 GB) | 94.4 ms | 143 ms |
| `stack` 160²×28 | **90 ms** (2.8 GB) | out of memory | out of memory | 557 ms (11.0 GB) | 561 ms | 225 ms |
| `3d` 96³ | **232 ms** (6.8 GB) | out of memory | out of memory | 684 ms (13.2 GB) | 690 ms | 1051 ms |

**H100**

| Problem | MR_AHA Toeplitz | BART full PSF | BART upper-triag | BART upper-triag, low memory | `bart pics` per CG iteration | MR_AHA NUFFT |
|---|---:|---:|---:|---:|---:|---:|
| `2d72` 320², L=24 | **9.00 ms** (1.3 GB) | 25 ms (12.4 GB) | 41 ms (7.2 GB) | 34 ms (4.0 GB) | 34.9 ms | 66 ms |
| `stack` 160²×28 | **22 ms** (2.8 GB) | out of memory | 139 ms (75.1 GB) | 138 ms (11.0 GB) | 134 ms | 151 ms |
| `3d` 96³ | **50 ms** (6.8 GB) | out of memory | 170 ms (21.6 GB) | 169 ms (13.2 GB) | 170 ms | 435 ms |

### The presets as they are

What happens without `--fit`. A failed entry gives the first allocation BART could not get on
that GPU (from BART's own message; nothing else was on the GPU). A smaller GPU fails on an
earlier, smaller request, so the sizes differ between GPUs:

| Preset | GPU | MR_AHA Toeplitz | BART full PSF | BART upper-triag | BART upper-triag, low memory |
|---|---|---:|---:|---:|---:|
| `2d72` | TITAN RTX | 229 ms (9.0 GB) | fails on a 18 GB request | fails on a 32 GB request | fails on a 8 GB request |
| `2d72` | L40S | 90 ms (9.0 GB) | fails on a 63 GB request | fails on a 32 GB request | 705 ms (28.4 GB) |
| `2d72` | H100 | 36 ms (9.0 GB) | fails on a 63 GB request | fails on a 32 GB request | 278 ms (28.4 GB) |
| `stack` | TITAN RTX | 1011 ms (18.5 GB) | fails on a 49 GB request | fails on a 26 GB request | fails on a 26 GB request |
| `stack` | L40S | 753 ms (22.6 GB) | fails on a 49 GB request | fails on a 213 GB request | fails on a 26 GB request |
| `stack` | H100 | 178 ms (22.6 GB) | fails on a 393 GB request | fails on a 213 GB request | fails on a 26 GB request |
| `3d` | TITAN RTX | out of memory | fails on a 35 GB request | fails on a 152 GB request | fails on a 19 GB request |
| `3d` | L40S | 1062 ms (31.5 GB) | fails on a 281 GB request | fails on a 152 GB request | fails on a 19 GB request |
| `3d` | H100 | 253 ms (31.5 GB) | fails on a 281 GB request | fails on a 152 GB request | 761 ms (38.3 GB) |

* BART's full PSF and its upper-triangular PSF without low-memory mode ask for single blocks of
  tens to hundreds of GB on the stack and 3D presets. They run only on the small 2D problems.
* MR_AHA runs every preset on every GPU except the unmasked `3d` preset on 24 GB inside this
  script (its 20 GB kernel fits only without the other arrays the comparison keeps; the masked
  operator of `python/bench_aha.py` fits).

### 20 conjugate-gradient iterations

`--methods bart_pics_cg mr_aha_toeplitz mr_aha_nufft --cg 20`, on the fitted problems. "End to
end" is MR_AHA's inputs-to-GPU + operator creation + CG, and BART's whole `pics` call through
`bart.py`. The BART mode is the first of full PSF, low memory, and upper-triangular PSF with low
memory that fits.

| GPU | Problem | MR_AHA Toeplitz CG | BART `pics` CG | Ratio | MR_AHA end to end | BART end to end | Ratio | MR_AHA NUFFT CG |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TITAN RTX | `2d72` 320², L=64 | **3.54 s** | 19.7 s (upper-triag, low memory) | 5.6× | **5.33 s** | 29.5 s | 5.5× | 13.1 s |
| TITAN RTX | `stack` 256²×28 | **6.16 s** | 34.7 s (upper-triag, low memory) | 5.6× | **7.84 s** | 62.0 s | 7.9× | 16.7 s |
| TITAN RTX | `3d` 128³ | **15.0 s** | 38.6 s (upper-triag, low memory) | 2.6× | **24.1 s** | 68.2 s | 2.8× | 76.7 s |
| L40S | `2d72` 320², L=72 | **1.83 s** | 14.1 s (upper-triag, low memory) | 7.7× | **3.97 s** | 23.3 s | 5.9× | 8.71 s |
| L40S | `stack` 320²×32 | **8.52 s** | 51.1 s (upper-triag, low memory) | 6.0× | **10.3 s** | 87.5 s | 8.5× | 18.6 s |
| L40S | `3d` 144³ | **16.0 s** | 47.2 s (upper-triag, low memory) | 2.9× | **22.3 s** | 78.8 s | 3.5× | 83.8 s |
| H100 | `2d72` 320², L=72 | **0.73 s** | 3.93 s (low memory) | 5.4× | **2.57 s** | 12.3 s | 4.8× | 4.03 s |
| H100 | `stack` 320²×48 | **3.13 s** | 18.3 s (upper-triag, low memory) | 5.8× | **5.27 s** | 58.9 s | 11.2× | 13.8 s |
| H100 | `3d` 160³ | **5.10 s** | 15.3 s (upper-triag, low memory) | 3.0× | **9.60 s** | 48.0 s | 5.0× | 41.0 s |

On the problems that fit 24 GB:

| GPU | Problem | MR_AHA Toeplitz CG | BART `pics` CG | Ratio | MR_AHA end to end | BART end to end | Ratio | MR_AHA NUFFT CG |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TITAN RTX | `2d72` 320², L=24 | **0.88 s** | 3.05 s (full PSF) | 3.5× | **1.27 s** | 6.58 s | 5.2× | 4.90 s |
| TITAN RTX | `stack` 160²×28 | **2.46 s** | 13.7 s (low memory) | 5.5× | **3.40 s** | 27.5 s | 8.1× | 7.39 s |
| TITAN RTX | `3d` 96³ | **6.32 s** | 16.9 s (low memory) | 2.7× | **10.5 s** | 32.2 s | 3.1× | 34.9 s |
| L40S | `2d72` 320², L=24 | **0.48 s** | 1.85 s (full PSF) | 3.9× | **1.02 s** | 5.55 s | 5.4× | 2.89 s |
| L40S | `stack` 160²×28 | **1.84 s** | 9.33 s (low memory) | 5.1× | **2.82 s** | 20.6 s | 7.3× | 4.51 s |
| L40S | `3d` 96³ | **4.68 s** | 11.6 s (low memory) | 2.5× | **6.73 s** | 24.4 s | 3.6× | 21.1 s |
| H100 | `2d72` 320², L=24 | **0.18 s** | 0.51 s (full PSF) | 2.8× | **0.60 s** | 3.55 s | 5.9× | 1.32 s |
| H100 | `stack` 160²×28 | **0.45 s** | 2.28 s (low memory) | 5.1× | **1.38 s** | 10.9 s | 7.9× | 3.04 s |
| H100 | `3d` 96³ | **1.01 s** | 2.81 s (low memory) | 2.8× | **2.21 s** | 12.6 s | 5.7× | 8.75 s |

* CG time is almost all $A^H A$ in both, so the CG ratio follows the operator ratio above.
* BART's end-to-end time adds a process start, `.cfl` I/O and the PSF on every call, because
  `bart.py` cannot keep an operator alive; with the container image also the start of the
  container.
* The `2d` problem is left out: its 20 iterations take less than the run-to-run variation of
  the two `pics` calls they are the difference of.

The JSON files are under `results/<gpu>/bench_bart_*` (not tracked, like all results).

## Files

* `build_bart.sh`: builds BART with CUDA and `bart_aha`; with `--sif`, gets the container image
  and builds `bart_aha_sif` inside it.
* `hpc/bench_bart.sbatch`: Slurm job that runs the benchmark with the container image.
* `bart_aha.c`: the timing helper.
* `bench_bart.py`: the benchmark.
* `bart/`: BART, pinned submodule ([mikecjz/bart](https://github.com/mikecjz/bart), v1.0.01
  plus the container files; BSD-3-Clause).
