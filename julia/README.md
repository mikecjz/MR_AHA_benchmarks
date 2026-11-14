# Julia benchmark: MRISubspaceRecon.jl

`bench_aha.jl` times the GPU low-rank (subspace) normal operator $A^H A$ of
[MRISubspaceRecon.jl](https://github.com/MagneticResonanceImaging/MRISubspaceRecon.jl) on the same
problems as the MR_AHA benchmark (`../python/bench_aha.py`), to compare MR_AHA with another
package. The package is a pinned submodule, `julia/MRISubspaceRecon.jl` (tag **v0.10.1**).

## What MRISubspaceRecon.jl implements

Citations are to the submodule at v0.10.1; `ext/…` is `ext/MRISubspaceReconCUDAExt/NFFTNormalOp.jl`.

* **Toeplitz embedding.** The only non-Cartesian normal operator is `NFFTNormalOp`. Its kernel is
  computed once on a grid oversampled by a fixed factor 2 in every dimension
  (`calculate_kernel_noncartesian(2 .* img_shape, …)`, `ext/…:10`, `src/NFFTNormalOp.jl:39`): for
  every pair of basis functions, a type-1 NUFFT of the products `conj(U[t,i]) * U[t,j]` (one per
  sample, `ext/…:142`, kernel `multiply_basis_vectors!` at `ext/…:227-242`) followed by an FFT
  (`ext/…:144-145`). Each application then loops over the coils (`ext/…:211`) and, per coil,
  zero-fills two work arrays of `2N × L`, writes `S_c · x`, does one batched FFT over all L
  coefficients (`ext/…:212-215`), multiplies by the kernel (`ext/…:219`), does one batched inverse
  FFT and accumulates `conj(S_c) ·` the cropped result (`ext/…:221-222`). There is **no
  NUFFT-based (non-Toeplitz) normal operator**; `FFTNormalOp` (`src/FFTNormalOp.jl`) is for
  Cartesian (or GROG-gridded) data. So only the Toeplitz operator is benchmarked.
* **Triangular (packed Hermitian) storage on the GPU.** Only the upper triangle `ic2 >= ic1` is
  computed (`ext/…:140-141`) and stored in LAPACK packed form, `Λ` of size
  `L(L+1)/2 × Nmask` (`ext/…:130`, packed index `ic1 + ic2(ic2-1)/2` at `ext/…:249`). The multiply
  kernel `kernel_mul!` reconstructs the lower triangle as the conjugate through a lookup table
  (`ext/…:43`, `ext/…:257-280`). One thread computes one output (k-space point, coefficient) with
  a loop over the L inputs. (The CPU version stores the full `L × L × Nmask` array,
  `src/NFFTNormalOp.jl:137`.) For a real basis `U`, a real kernel of half the size is used
  (`ext/…:154-198`); the Gram blocks here are complex, so that path does not apply.
* **Always masked.** The kernel is kept only at the grid points within the support of a Gaussian
  spreading kernel (σ = 1) around the samples (`calculate_kmask_indcs`, `ext/…:75-91`) and zero
  elsewhere. This is the same kind of approximation as MR_AHA's *masked* Toeplitz operator
  (`--mask on`); unlike MR_AHA it cannot be switched off.
* **GPU via CUDA.jl.** All of the above runs on the GPU when the trajectory and basis are
  `CuArray`s (package extension `MRISubspaceReconCUDAExt`, weak dependency on CUDA.jl). The NUFFTs
  come from NonuniformFFTs.jl (`ext/…:107`, `gpu_method=:shared_memory`), the FFTs from cuFFT.
  One GPU only; the coil loop is sequential.
* **Inputs.** A temporal basis `U` (`Nframes × L`) and a trajectory per time frame, in cycles,
  `[-1/2, 1/2)`, shape `(ndim, Nsamples_per_frame, Nframes)`; it does not take Gram blocks.
  `img_shape` times 2 must factor into 2, 3 and 5 (assertion at `ext/…:76`), and there is no
  stack-of-stars mode (a stack is a 3D trajectory, oversampled by 2 in z as well).

## How the problems are translated

The problems are those of `python/problems.py`, saved to `julia/data/<preset>.mat` by
`save_problem.py` (run automatically when the file is missing, with `$PYTHON`, default `python`,
which needs NumPy and SciPy) and read by `bench_aha.jl`, so the trajectory, Gram blocks, maps and
coefficients are bit-identical to those MR_AHA is timed with.

* **Gram blocks → basis.** Each Gram block $\Phi_2 = \sum_k \lambda_k v_k v_k^H$ (rank 2 here: two
  time points per readout, as in `gram_blocks`) is eigen-decomposed and becomes `r = 2` time frames
  that share the readout's k-space positions, with basis rows `sqrt(λ_k) * transpose(v_k)`. The
  package applies $v_i \leftarrow \sum_j \sum_t \overline{U_{ti}} U_{tj} v_j$ per k-space point,
  MR_AHA applies $v_l \leftarrow \sum_m v_m \Phi_2(m,l)$; the two agree for $U = \Phi_{rt}^T$
  (rank × time basis transposed, no conjugation). The operator is therefore the same; the only cost
  is twice as many non-uniform points in the kernel computation (set-up), not in the application.
  This conversion is not timed (`prep_s` in the JSON).
* **Trajectory.** Radians in `[-π, π)` divided by 2π. A stack becomes a 3D trajectory with
  `kz = (iz - Nz÷2) / Nz` for the centred kz index `iz`.
* **Coefficients / maps.** Column-major `[Nx, Ny, (Nz,) L]` and one `[Nx, Ny, (Nz)]` map per coil,
  as the package wants.

### Correctness check

`check_mr_aha.py` applies MR_AHA's operators (Python interface) to the same `x` and compares with
the Julia result (`bench_aha.jl --data P.mat --save-result Y.mat`). On a TITAN RTX:

| Problem | MR_AHA Toeplitz vs MR_AHA NUFFT | MR_AHA masked vs NUFFT | **Julia vs MR_AHA NUFFT** | Julia vs MR_AHA Toeplitz | best-fit scalar |
|---|---|---|---|---|---|
| `2d` preset | 9.4e-7 | 3.0e-3 | **1.9e-3** | 1.9e-3 | 0.99999 |
| stack 64×64×16, L=4, 4 coils, 50×64 | 1.2e-6 | 1.1e-2 | **4.6e-3** | 4.6e-3 | 0.99999 |
| 3D 48³, L=4, 4 coils, 800×48 | 1.4e-6 | 1.0e-2 | **3.4e-3** | 3.4e-3 | 0.99999 |

(relative 2-norm differences; the random `x` of the benchmark is a worst case for masking.)

* **No scaling difference**: the least-squares scalar is 1 to 5 digits, so both packages compute
  the same, unnormalised $A^H A$ (type-2 NUFFT with $e^{-i k\cdot r}$, adjoint with $e^{+i k\cdot r}$).
* **Same sign and Gram-block convention**: with `U = transpose(Φ_rt)` and `k/(2π)` the results
  agree; a sign or conjugation error would give O(1) differences.
* The remaining 2-5e-3 is the package's kernel mask: it is of the same order as MR_AHA's own masked
  operator, and the package's mask keeps slightly more of the grid (82 % / 81 % / 57 % for `2d` /
  `2d72` / `3d`, against 81 % / 80 % / 54 % for `mr_aha.calc_mask`).

## Results (one TITAN RTX, 24 GB)

`--reps 10 --warmup 2 --setup-reps 3` for both, set-up = median of the warm creations, A^H A =
median application. MR_AHA: `python/bench_aha.py --methods toeplitz nufft_c --mask both
--setup-reps 3` (library `c60db28`, cufinufft eps 1e-6), measured on 2026-11-14. Julia 1.11.7,
CUDA.jl 6.5.1, NonuniformFFTs 0.9.10, 8 threads, measured on 2026-11-12 and not repeated (the
package is unchanged). JSON in `results/titan_rtx/` (`bench_aha_julia_*.json`, `bench_aha_*.json`).

| Preset | | set-up s | A^H A ms | mem GB |
|---|---|---:|---:|---:|
| `2d` 160², L=6, 16 coils | **MRISubspaceRecon.jl** (masked, packed) | 0.030 | **3.05** | 0.03 |
|  | MR_AHA toeplitz | 0.011 | **2.14** | 0.04 |
|  | MR_AHA toeplitz masked | 0.012 | 2.09 | 0.04 |
|  | MR_AHA nufft_c | 0.002 | 20.5 | 0.05 |
| `2d72` 320², L=72, 12 coils | **MRISubspaceRecon.jl** | 5.68 | **802** | 7.48 |
|  | MR_AHA toeplitz | 2.62 | **230** | 9.62 |
|  | MR_AHA toeplitz masked | 2.17 | 202 | 7.88 |
|  | MR_AHA nufft_c | 0.003 | 709 | 0.48 |
| `stack` 320²×56, L=12, 12 coils | **MRISubspaceRecon.jl** | – | **fails**: 2·56 = 112 is not a product of 2, 3, 5 | – |
|  | MR_AHA toeplitz | 2.99 | **1012** | 19.3 |
|  | MR_AHA toeplitz masked | 3.05 | 979 | 16.7 |
|  | MR_AHA nufft_c | 0.005 | 2625 | 2.13 |
| `stack`, Nz = 32 instead of 56 | **MRISubspaceRecon.jl** (640×640×64 grid) | 31.7 | **1489** | 18.7 |
| (substitute that the package accepts) | MR_AHA toeplitz (640×640×32 grid) | 1.98 | **574** | 13.5 |
|  | MR_AHA toeplitz masked | 1.85 | 555 | 12.1 |
|  | MR_AHA nufft_c | 0.004 | 1471 | 1.24 |
| `3d` 160³, L=12, 12 coils | **MRISubspaceRecon.jl** | 36.1 | **1685** | 18.5 |
|  | MR_AHA toeplitz | 18.7 | **1603** | 24.0 |
|  | MR_AHA toeplitz masked | 18.9 | 1387 | 15.1 |
|  | MR_AHA nufft_c | 0.005 | 7891 | 1.40 |

Observations:

* **Application.** MRISubspaceRecon.jl's packed, masked Toeplitz operator is 1.2-4.0× slower than
  MR_AHA's masked Toeplitz operator on the 2D and 3D presets (1.5× / 4.0× / 1.2× for `2d` /
  `2d72` / `3d`) and 1.05-3.5× slower than MR_AHA's unmasked one. At L = 12 and below both are
  dominated by the same per-coil batched FFTs; the package zero-fills both `2N × L` work arrays
  per coil. At L = 72 the block multiply decides: the package uses one thread per output that
  gathers a packed triangle, MR_AHA its tiled kernels for ranks above 30.
* **Stack-of-stars.** The package has no stack mode: the preset fails its 2-3-5 size check, and
  where it runs (Nz = 32) it oversamples z by 2 and so does twice the FFT work and stores twice
  the kernel; MR_AHA's stack operator is 2.7× faster there.
* **Set-up** (kernel computation) is 2-17× slower in the package (one type-1 NUFFT + FFT per
  triangular entry, with twice the points because Gram blocks become two frames; MR_AHA batches
  the entries). The first, cold creation in Julia additionally includes compilation (≈ 1.7 s for
  `2d`), which `--setup-reps 3` excludes from the median.
* **Memory.** Because the kernel is always masked, its memory is close to MR_AHA's masked
  operator (`3d`: 18.5 GB, against 15.1 GB masked and 24.0 GB unmasked for MR_AHA; the package
  also keeps two `2N × L` work arrays).

### L40S and H100

Same settings, one GPU, on the HPC with `hpc/bench_julia.sbatch` (MR_AHA `c60db28` built for sm_89 /
sm_90, Julia 1.11.7, CUDA.jl 6.5.1 with its own CUDA 13.4 runtime), both libraries in the same
job on the same node. A^H A in ms (set-up in s):

| Preset | L40S Julia | L40S MR_AHA masked | L40S MR_AHA toeplitz | H100 Julia | H100 MR_AHA masked | H100 MR_AHA toeplitz |
|---|---:|---:|---:|---:|---:|---:|
| `2d` | 0.82 (0.02) | **0.43** (0.28) | 0.43 | 0.88 (0.02) | **0.54** (0.21) | 0.53 |
| `2d72` | 362 (4.10) | **86** (2.31) | 90 | 76 (4.66) | **33** (2.02) | 36 |
| `stack` | fails (2-3-5) | **744** (2.48) | 752 | fails (2-3-5) | **170** (1.75) | 178 |
| `stack`, Nz = 32 | 1142 (19.38) | **412** (1.67) | 417 | 272 (5.27) | **96** (1.07) | 100 |
| `3d` | 1347 (21.44) | **1016** (7.37) | 1064 | 317 (6.15) | **232** (4.78) | 252 |

* The package is 1.3-4.2x slower than MR_AHA's masked operator on the 2D and 3D presets (2.8x
  on the stack, which it treats as 3D). Its set-up is 1.3-12x slower, except on the small `2d`
  preset, where MR_AHA's set-up takes 0.2-0.3 s against 0.02 s.
* `2d72` (L = 72): MR_AHA is 4.2x faster on the L40S and 2.3x faster on the H100. This is the
  effect of the tiled block-diagonal kernels for ranks above 30 in MR_AHA `c60db28`. With the
  generic kernels before them (`711003d`) the masked operator took 288 ms on the L40S and 165 ms
  on the H100, and the package was 2.2x faster on the H100.
* Correctness (`check_mr_aha.py`) is the same as on the TITAN RTX on both GPUs: 1.9e-3 / 3.4e-3 /
  4.6e-3 against MR_AHA's NUFFT, scalar 0.99999.

Run on a Slurm cluster from the repository root (`third_party/MR_AHA` initialised with
`--recursive`, problem files in `julia/data/`): `mkdir -p julia/hpc/logs`, then
`sbatch julia/hpc/setup_julia.sbatch` once (installs the official Julia 1.11.7 tarball in `~/opt`
and instantiates `julia/` on a compute node), then
`sbatch --gres=gpu:l40s:1 --export=ALL,ARCH=89 julia/hpc/bench_julia.sbatch` (or `gpu:h100:1`,
`ARCH=90`). The job script gives each GPU type its own Julia compile cache (`~/.julia_sm<ARCH>`):
node types with different CPUs sharing `~/.julia/compiled` invalidate each other's native code and
wait on each other's precompile locks.

## Requirements and installation

* Julia ≥ 1.11. Without root, e.g. the official tarball (or [juliaup](https://github.com/JuliaLang/juliaup)):

  ```bash
  mkdir -p ~/opt && cd ~/opt
  curl -L https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.7-linux-x86_64.tar.gz | tar xz
  export PATH=~/opt/julia-1.11.7/bin:$PATH
  ```

* An NVIDIA GPU and driver. CUDA.jl downloads its own CUDA runtime; no toolkit is needed.
  If `LD_LIBRARY_PATH` contains a system CUDA (`/usr/local/cuda/lib64`), CUDA.jl warns that it
  loaded libraries from there; run with `env -u LD_LIBRARY_PATH` to avoid mixing versions.
* Python with NumPy and SciPy to write the problem files (`save_problem.py`), and the MR_AHA
  Python environment for `check_mr_aha.py`.

From the repository root:

```bash
git submodule update --init julia/MRISubspaceRecon.jl
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

`Project.toml` uses the submodule through `Pkg.develop(path="MRISubspaceRecon.jl")`;
`Manifest.toml` pins every other package.

## Run

```bash
cd julia
export CUDA_VISIBLE_DEVICES=3 PYTHON=python   # one GPU; Python for save_problem.py
env -u LD_LIBRARY_PATH julia -t 8 --project=. bench_aha.jl                      # preset 2d
env -u LD_LIBRARY_PATH julia -t 8 --project=. bench_aha.jl --problem 2d 2d72 stack 3d
python save_problem.py stack data/stack_nz32.mat --n 320 320 32                # custom sizes
env -u LD_LIBRARY_PATH julia -t 8 --project=. bench_aha.jl --data data/stack_nz32.mat
```

| Option | Meaning | Default |
|---|---|---|
| `--problem NAME ...` | presets `2d`, `2d72`, `stack`, `3d` (files `data/<name>.mat`) | `2d` |
| `--data FILE.mat` | a problem file from `save_problem.py` or `python/bench_aha.py --save-data` | |
| `--reps`, `--warmup` | timed and untimed applications | 10, 2 |
| `--setup-reps` | operator creations; the first is reported as cold | 3 |
| `--out FILE.json` / `--no-json` | JSON report (default `results/<gpu>/bench_aha_julia_<time>.json`) | |
| `--save-result FILE.mat` | write `A^H A x` of the (last) problem, for `check_mr_aha.py` | |

Correctness check against MR_AHA:

```bash
env -u LD_LIBRARY_PATH julia --project=. bench_aha.jl --data data/2d.mat --no-json --save-result data/y.mat
MR_AHA_ROOT=~/code/MR_AHA python check_mr_aha.py data/2d.mat data/y.mat
```

The JSON rows hold `setup_s`, `setup_cold_s`, `kernel_s`, `aha_ms` (median), `aha_min_ms`,
`aha_mean_ms`, `mem_gb`, `kernel_gb` (packed `Λ`), `mask_keep` (fraction of the Toeplitz grid kept),
`frames_per_readout`, `prep_s`, and on failure `error`.
