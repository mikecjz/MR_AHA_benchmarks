# MATLAB benchmark

`bench_aha.m` times the $A^H A$ operator of MR_AHA with the Toeplitz method, the library's NUFFT
method and a pure-MATLAB NUFFT operator, on one or several GPUs. It takes the same options as
the [Python benchmark](../python/README.md), as name-value pairs.

## Requirements

* MATLAB with the Parallel Computing Toolbox (not needed on Apple silicon).
* The MR_AHA MEX files, built with `-DMR_AHA_BUILD_MATLAB=ON` as described in the
  [top-level README](../README.md).
* For the `nufft_native` method: cufinufft's MATLAB interface, built with
  `-DMR_AHA_BUILD_CUFINUFFT_MATLAB=ON` (or any other installation, see `'cufinufft_dir'`).
  Without it that method is skipped.

`bench_aha` puts the library on the MATLAB path itself. It looks for the checkout in
`'mr_aha_root'`, then the environment variable `MR_AHA_ROOT`, then `third_party/MR_AHA`; for the
MEX files in `'mex_dir'`, then `MR_AHA_MEX_DIR`, then `<root>/build/matlab`.

## Run

```matlab
cd matlab
bench_aha                                                   % preset 2d, all methods, 1 GPU
bench_aha('problem', {'stack', '3d'})                       % the two large 3D presets
bench_aha('problem', 'stack', 'gpus', [1 2 4])              % multi-GPU scaling
bench_aha('problem', 'stack', 'methods', {'toeplitz'}, 'mask', 'both')
bench_aha('problem', '3d', 'n', [96 96 96], 'L', 8, 'check', true, 'cg', 10)
bench_aha('data', 'my_scan.mat', 'methods', {'toeplitz', 'nufft_c'})
report = bench_aha('problem', 'stack');                     % results as a struct
```

From a shell:

```bash
matlab -batch "cd matlab; bench_aha('problem', 'stack', 'gpus', [1 2])"
```

The current `gpuDevice` is the primary GPU; `'gpus', 4` adds the next three in `gpuDevice`
order. Select the primary GPU with `gpuDevice(k)` before the call, or limit the visible GPUs
with `CUDA_VISIBLE_DEVICES`.

## Options

| Name | Meaning | Default |
|---|---|---|
| `'problem'` | preset or cell array of presets: `'2d'`, `'2d72'`, `'stack'`, `'3d'` | `'2d'` |
| `'n'` | matrix size `[Nx Ny]` or `[Nx Ny Nz]` instead of the preset's | |
| `'L'`, `'coils'`, `'spokes'`, `'samples'` | ranks, coils, readouts, samples per readout | |
| `'data'` | `.mat` problem file instead of a preset | |
| `'traj_type'` | `'2D'`, `'3D'` or `'stack'`, when the file does not say | |
| `'seed'` | seed of the random inputs | 0 |
| `'methods'` | cell array of `'toeplitz'`, `'nufft_c'`, `'nufft_native'` | all available |
| `'gpus'` | vector of GPU counts, or `'all'`; `nufft_native` always uses one | 1 |
| `'mask'` | Toeplitz: `'off'`, `'on'` or `'both'` | `'off'` |
| `'toep_dist'` | Toeplitz on several GPUs: `'auto'`, `'split'`, `'replicate'`, `'broadcast'` | `'auto'` |
| `'upsampfac'` | NUFFT methods: cufinufft upsampling factor | 2 |
| `'gpu_method'` | NUFFT methods: cufinufft `gpu_method` for both transforms, 1 or 2 | library's choice |
| `'eps'` | NUFFT tolerance | 1e-6 |
| `'reps'`, `'warmup'` | timed and untimed applications | 10, 2 |
| `'setup_reps'` | operator creations (first reported as cold) | 1 |
| `'cg'` | also time this many conjugate-gradient iterations | 0 |
| `'check'` | relative difference of every result to the first row's | false |
| `'out'` | JSON output | `results/<gpu>/bench_aha_matlab_<time>.json` (`<gpu>`: the GPU model, e.g. `titan_rtx`) |
| `'mr_aha_root'`, `'mex_dir'`, `'cufinufft_dir'` | where the library is | see above |

`help bench_aha` prints the full description.

## Data files

`'data'` reads a `.mat` file with `SEs [Nx, Ny, (Nz,) Ncoils]`, `phi2 [Ntrajs, (Nkz,) L, L]`,
`traj [Ntrajs, Nkx, ndim]`, `traj_type` and optionally `x`, the layout of the MR_AHA MATLAB
interface. The Python benchmark writes such a file with `--save-data`, so both languages can
time the same arrays. `bench_problem` returns a preset as a struct if you want to save one from
MATLAB:

```matlab
d = bench_problem('stack', 'n', [64 64 12]);
save('small.mat', '-struct', 'd', 'SEs', 'phi2', 'traj', 'traj_type', 'x');
```

## Notes

* `mem GB` is the primary GPU only: MATLAB cannot query another GPU without resetting it.
* `nufft_native` runs last and the results are saved after every row. On some machines
  cufinufft's MEX file aborts MATLAB on certain trajectories; the earlier rows are then still in
  the JSON file.

## Files

* `bench_aha.m`: the benchmark.
* `bench_problem.m`: presets, synthetic inputs, reading problem files.
