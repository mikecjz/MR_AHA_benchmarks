#!/bin/bash
# MRISubspaceRecon.jl vs MR_AHA (711003d) A^H A on one GPU.
#   sbatch --gres=gpu:l40s:1 --export=ALL,ARCH=89 julia/hpc/bench_julia.sbatch
#   sbatch --gres=gpu:h100:1 --export=ALL,ARCH=90 julia/hpc/bench_julia.sbatch
#SBATCH --job-name=julia_aha
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=julia/hpc/logs/%x_%j.out
set -uo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$ROOT"
ARCH=${ARCH:?set ARCH=89 or 90}
JL=$HOME/opt/julia-1.11.7/bin/julia
PY=$HOME/.conda/envs/mr_aha/bin/python
module load cuda12.9/toolkit/12.9.1
export PATH=$(dirname "$PY"):$PATH
echo "== $(hostname) $(date)"; nvidia-smi -L

# ---- MR_AHA library for this GPU ----
LIB=third_party/MR_AHA
B=$LIB/build_sm$ARCH
cmake -S $LIB -B $B -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$ARCH \
    -DMR_AHA_BUILD_MATLAB=OFF -DMR_AHA_BUILD_PYTHON=ON -DMR_AHA_BUILD_EXAMPLES=OFF > /dev/null || exit 1
cmake --build $B -j "${SLURM_CPUS_PER_TASK:-8}" --target mr_aha_c 2>&1 | grep -E " error" || true
export MR_AHA_ROOT=$ROOT/$LIB MR_AHA_LIBRARY=$ROOT/$B/lib/libmr_aha_c.so
test -f "$MR_AHA_LIBRARY" || { echo "no $MR_AHA_LIBRARY"; exit 1; }

# ---- Julia: one compile cache per GPU type ----
# Jobs on node types with different CPUs that share ~/.julia/compiled keep
# invalidating each other's native code caches (and wait on each other's
# precompile locks). The first depot gets this node type's compiled cache;
# packages and artifacts still come from ~/.julia.
export JULIA_DEPOT_PATH=$HOME/.julia_sm$ARCH:$HOME/.julia
mkdir -p "$HOME/.julia_sm$ARCH"
env -u LD_LIBRARY_PATH "$JL" --project=julia -e 'using Pkg; Pkg.precompile(); using CUDA; CUDA.versioninfo()'

# ---- MRISubspaceRecon.jl ----
cd julia
env -u LD_LIBRARY_PATH "$JL" -t 8 --project=. bench_aha.jl --problem 2d 2d72 stack 3d --setup-reps 3
env -u LD_LIBRARY_PATH "$JL" -t 8 --project=. bench_aha.jl --data data/stack_nz32.mat --setup-reps 3
for p in 2d 3d_small stack_small; do
    env -u LD_LIBRARY_PATH "$JL" -t 8 --project=. bench_aha.jl --data data/$p.mat --no-json \
        --save-result data/y_${p}_julia_sm$ARCH.mat > /dev/null &&
    "$PY" check_mr_aha.py data/$p.mat data/y_${p}_julia_sm$ARCH.mat
done
cd ..

# ---- MR_AHA on the same problems ----
"$PY" -u python/bench_aha.py --problem 2d 2d72 stack 3d --methods toeplitz nufft_c --mask both --setup-reps 3 2>&1 | grep -v "^\[mr_aha\]"
"$PY" -u python/bench_aha.py --problem stack --n 320 320 32 --methods toeplitz nufft_c --mask both --setup-reps 3 2>&1 | grep -v "^\[mr_aha\]"
echo "== done $(date)"
