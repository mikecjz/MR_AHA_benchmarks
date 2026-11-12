# MR_AHA_benchmarks

Speed benchmarks for [MR_AHA](https://github.com/mikecjz/MR_AHA), the GPU library that evaluates
the forward normal operator $A^H A$ for non-Cartesian MRI with low-rank (subspace /
MR-Multitasking) modelling.

One script per language builds a test problem, creates the operator with each method on one or
several GPUs, and prints how long the set-up and one application take:

| Method | What runs | GPUs |
|---|---|---|
| `toeplitz` | the library's Toeplitz operator: kernel computed once, then two FFTs per coil | 1 or several |
| `nufft_c` | the library's NUFFT operator: two NUFFTs per coil with cufinufft's C API, inside the C/CUDA library | 1 or several |
| `nufft_native` | the same NUFFT operator written in Python (CuPy + cufinufft's Python package) or MATLAB (gpuArray + cufinufft's MATLAB interface), without the library | 1 |

* [`python/`](python/): `bench_aha.py`, a command-line script. [How to run it](python/README.md).
* [`matlab/`](matlab/): `bench_aha.m`, a function with the same options.
  [How to run it](matlab/README.md).

Both take the same options: the problem (four synthetic presets, custom sizes, or your own data
file), the methods, the GPU counts, masked or unmasked Toeplitz, NUFFT settings, repetitions, an
optional conjugate-gradient run and an optional accuracy check.

```
$ python python/bench_aha.py
backend cuda, 1 GPU(s) visible: NVIDIA TITAN RTX

== 2d: 2D golden-angle radial 160x160, L=6, 16 coils, 377x160 samples
   unmasked Toeplitz kernel 0.02 GB
  method           GPUs  set-up s   A^H A ms speed-up   mem GB
  toeplitz            1     0.015       2.22              0.01
  nufft_c             1     0.002      27.27              0.05
  nufft_native        1     0.003      21.54              0.05
```

## Getting the library

The library is a pinned submodule, `third_party/MR_AHA`. Clone with submodules and build it once
(CUDA toolkit 11.x or 12.x, CMake ≥ 3.24, a C++17 compiler):

```bash
git clone --recurse-submodules https://github.com/mikecjz/MR_AHA_benchmarks.git
cd MR_AHA_benchmarks

cmake -S third_party/MR_AHA -B third_party/MR_AHA/build -DCMAKE_BUILD_TYPE=Release \
      -DMR_AHA_BUILD_PYTHON=ON \
      -DMR_AHA_BUILD_MATLAB=ON -DMR_AHA_BUILD_CUFINUFFT_MATLAB=ON   # MATLAB benchmark only
cmake --build third_party/MR_AHA/build -j
```

To benchmark a checkout you already have, including local changes, set `MR_AHA_ROOT` instead:

```bash
export MR_AHA_ROOT=~/code/MR_AHA      # uses its python/, matlab/ and build/
```

Build details and requirements, including Apple silicon, are in the
[MR_AHA documentation](https://mikecjz.github.io/MR_AHA_docs/getting-started/install.html).

## Problems

| Preset | Trajectory | Matrix | L | Coils | Readouts × samples | Unmasked Toeplitz kernel |
|---|---|---|---|---|---|---|
| `2d` | 2D golden-angle radial | 160 × 160 | 6 | 16 | 377 × 160 | 0.02 GB |
| `2d72` | 2D golden-angle radial | 320 × 320 | 72 | 12 | 754 × 320 | 8.6 GB |
| `stack` | stack-of-stars | 320 × 320 × 56 | 12 | 12 | 280 × 320, 56 kz | 14 GB |
| `3d` | 3D golden-means kooshball | 160 × 160 × 160 | 12 | 12 | 15680 × 160 | 20 GB |

Every size can be changed from the command line. The inputs are synthetic: the run time depends
on the sizes and on the geometry of the trajectory, not on the values, so the sensitivity maps
are smooth profiles and the Gram blocks and coefficients are random. To time a real acquisition,
pass a `.mat` file with `SEs`, `phi2`, `traj` and `traj_type` in the library's documented layout
(`--data` / `'data'`).

The three large presets need a GPU with enough memory for the kernel plus the work buffers. A
configuration that does not fit is reported as failed and the run continues. The masked Toeplitz
operator (`--mask on`) stores less.

## Reading the output

| Column | Meaning |
|---|---|
| `GPUs` | GPUs asked for. `4>3` means the library used 3 of the 4: it uses the fewest GPUs that give the same number of coils per GPU, and one GPU for problems too small to gain. |
| `set-up s` | Creating the operator: Toeplitz kernel, plans, copies to the other GPUs. |
| `A^H A ms` | Median time of one application. |
| `speed-up` | One-GPU time of the same method divided by this row's time. |
| `mem GB` | Device memory in use after the first applications minus before the operator was created. Python: summed over the GPUs. MATLAB: the primary GPU only. |
| `CG s` | With `--cg N`: time of N plain conjugate-gradient iterations. |
| `diff` | With `--check`: relative difference of this row's $A^H A x$ to the first row's. |

Each run also writes a JSON file under `results/<gpu>/` (the GPU model, e.g. `results/titan_rtx/`) with these numbers and more (cold set-up
time, kernel time, mask time, minimum and mean application time, GPU model, versions, options).

Timing notes:

* All GPUs are synchronised before and after every timed section.
* A tiny warm-up problem runs first so that loading cuFFT, cuBLAS and cufinufft is not charged to
  the first row. Other GPUs still initialise on their first operator; `--setup-reps 3` reports
  that first, cold, creation separately from the median of the rest.
* On a shared machine, pick idle GPUs with `CUDA_VISIBLE_DEVICES`.

## Published results

Measured results and their discussion (TITAN RTX, L40S, H100 and Apple M3 Ultra; masking,
multi-GPU scaling, NUFFT settings) are in the
[benchmarks section of the documentation](https://mikecjz.github.io/MR_AHA_docs/benchmarks/).

## License

Copyright (C) 2026 Junzhou Chen. Released under the [GNU AGPL v3 only](LICENSE), like MR_AHA
itself; see
[MR_AHA's licensing notes](https://github.com/mikecjz/MR_AHA/blob/main/licenses/LICENSING.md).
