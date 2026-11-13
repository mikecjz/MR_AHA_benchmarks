#!/bin/bash
# Install Julia 1.11.7 (official tarball) in ~/opt and instantiate + precompile
# julia/ on a compute node (CPU only). The HPC module has only Julia 1.6.6,
# too old for CUDA.jl 6.5 / MRISubspaceRecon.jl v0.10.1.
#SBATCH --job-name=julia_setup
#SBATCH --partition=preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=julia/hpc/logs/%x_%j.out
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
JV=1.11.7
JDIR=$HOME/opt/julia-$JV
echo "== $(hostname) $(date)"
if [ ! -x "$JDIR/bin/julia" ]; then
    mkdir -p "$HOME/opt"
    curl -fsSL "https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-$JV-linux-x86_64.tar.gz" \
        | tar xz -C "$HOME/opt"
fi
"$JDIR/bin/julia" --version
cd "$ROOT/julia"
env -u LD_LIBRARY_PATH "$JDIR/bin/julia" -t 8 --project=. \
    -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); Pkg.status()'
echo "== done $(date)"
