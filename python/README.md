# Python benchmark

`bench_aha.py` times the $A^H A$ operator of MR_AHA with the Toeplitz method, the library's
NUFFT method and a pure-Python NUFFT operator, on one or several GPUs.

## Requirements

* The MR_AHA shared library, built as described in the [top-level README](../README.md)
  (`third_party/MR_AHA/build`, or `MR_AHA_ROOT` pointing at another checkout).
* Python ≥ 3.9 with NumPy and CuPy for your CUDA toolkit:

  ```bash
  pip install numpy cupy-cuda12x
  pip install cufinufft          # only for the nufft_native method
  pip install scipy h5py         # only for --data / --save-data with .mat files
  ```

  The cufinufft wheel needs `libcudart.so.12` on `LD_LIBRARY_PATH`. Without cufinufft the
  `nufft_native` method is skipped.

On Apple silicon the arrays are NumPy arrays and `nufft_native` uses mlx-nufft; only one GPU
exists there.

## Run

```bash
cd python
python bench_aha.py                                   # preset 2d, all methods, 1 GPU
python bench_aha.py --problem stack 3d                # the two large 3D presets
python bench_aha.py --problem stack --gpus 1 2 4      # multi-GPU scaling
python bench_aha.py --problem stack --methods toeplitz --mask both
python bench_aha.py --problem 3d --n 96 96 96 --L 8 --check --cg 10
python bench_aha.py --data my_scan.mat --methods toeplitz nufft_c
```

`CUDA_VISIBLE_DEVICES` selects the GPUs. The current CuPy device (device 0 of the visible ones)
is the primary GPU; `--gpus 4` adds the next three.

## Options

| Option | Meaning | Default |
|---|---|---|
| `--problem NAME ...` | presets: `2d`, `2d72`, `stack`, `3d` | `2d` |
| `--n Nx Ny [Nz]` | matrix size instead of the preset's | |
| `--L`, `--coils`, `--spokes`, `--samples` | ranks, coils, readouts, samples per readout | |
| `--data FILE` | `.mat` or `.npz` problem file instead of a preset | |
| `--traj-type` | `2D`, `3D` or `stack`, when the file does not say | |
| `--save-data FILE` | write the problem to a `.mat` or `.npz` file | |
| `--seed` | seed of the random inputs | 0 |
| `--methods METHOD ...` | `toeplitz`, `nufft_c`, `nufft_native` | all available |
| `--gpus G ...` | GPU counts, or `all`; `nufft_native` always uses one | `1` |
| `--mask off\|on\|both` | Toeplitz: unmasked, masked, or both | `off` |
| `--toep-dist` | Toeplitz on several GPUs: `auto`, `split`, `replicate`, `broadcast` | `auto` |
| `--upsampfac` | NUFFT methods: cufinufft upsampling factor | 2 |
| `--gpu-method 1\|2` | NUFFT methods: cufinufft `gpu_method` for both transforms | library's choice |
| `--eps` | NUFFT tolerance | 1e-6 |
| `--reps`, `--warmup` | timed and untimed applications | 10, 2 |
| `--setup-reps` | operator creations (first reported as cold) | 1 |
| `--cg N` | also time N conjugate-gradient iterations | off |
| `--check` | relative difference of every result to the first row's | off |
| `--out FILE` | JSON output | `results/<host>/bench_aha_<time>.json` |

`python bench_aha.py --help` prints the same list.

## Data files

`--data` reads `SEs`, `phi2`, `traj` and optionally `traj_type` and `x`:

* `.mat`: the column-major layout of the MR_AHA documentation and MATLAB interface,
  `SEs [Nx, Ny, (Nz,) Ncoils]`, `phi2 [Ntrajs, (Nkz,) L, L]`, `traj [Ntrajs, Nkx, ndim]`.
* `.npz`: the row-major layout of the Python interface, i.e. the same shapes reversed.

`--save-data` writes a synthetic problem in either format, which is the easiest way to see the
layout, and lets the MATLAB benchmark time exactly the same arrays:

```bash
python bench_aha.py --problem stack --n 64 64 12 --save-data small.mat
```

## Files

* `bench_aha.py`: the benchmark.
* `problems.py`: presets, synthetic inputs, reading and writing problem files. No GPU needed.
