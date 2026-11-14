#!/bin/bash
# BART (from an Apptainer / Singularity image) vs MR_AHA: A^H A and 20 CG
# iterations on one GPU of an HPC cluster. Submit from the repository root:
#
#   mkdir -p bart/hpc/logs
#   sbatch --gres=gpu:l40s:1 --export=ALL,ARCH=89 bart/hpc/bench_bart.sbatch
#   sbatch --gres=gpu:h100:1 --export=ALL,ARCH=90 bart/hpc/bench_bart.sbatch
#
# Variables (--export=ALL,NAME=value,...):
#   ARCH     CUDA compute capability of the GPU, for the MR_AHA build (89 = L40S, 90 = H100)
#   SIF      BART image (default bart/bart.sif; downloaded by bart/build_bart.sh --sif if missing)
#   PY       Python with NumPy and CuPy (default ~/.conda/envs/mr_aha/bin/python)
#   MODULES  environment modules to load (default: CUDA toolkit and Apptainer of this cluster)
#   SETS     which runs: any of "small full cg" (default: all three)
#
# BART cannot be compiled on this cluster (it needs GCC >= 12), so it runs from
# the image; MR_AHA is compiled here for the GPU and runs natively.
#SBATCH --job-name=bart_aha
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=bart/hpc/logs/%x_%j.out
set -uo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$ROOT"
ARCH=${ARCH:?set ARCH=89 (L40S) or 90 (H100)}
SIF=${SIF:-$ROOT/bart/bart.sif}
PY=${PY:-$HOME/.conda/envs/mr_aha/bin/python}
SETS=${SETS:-small full cg}
module load ${MODULES:-cuda12.9/toolkit/12.9.1 apptainer/1.0.1}
export PATH=$(dirname "$PY"):$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "== $(hostname) $(date)"; nvidia-smi -L; apptainer --version

# ---- MR_AHA library for this GPU ----
LIB=third_party/MR_AHA
B=$LIB/build_sm$ARCH
cmake -S $LIB -B $B -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$ARCH \
    -DMR_AHA_BUILD_MATLAB=OFF -DMR_AHA_BUILD_PYTHON=ON -DMR_AHA_BUILD_EXAMPLES=OFF > /dev/null || exit 1
cmake --build $B -j "${SLURM_CPUS_PER_TASK:-8}" --target mr_aha_c 2>&1 | grep -E " error" || true
export MR_AHA_ROOT=$ROOT/$LIB MR_AHA_LIBRARY=$ROOT/$B/lib/libmr_aha_c.so
test -f "$MR_AHA_LIBRARY" || { echo "no $MR_AHA_LIBRARY"; exit 1; }

# ---- BART image and the timing helper (compiled inside the image) ----
bart/build_bart.sh --sif "$SIF" || exit 1

# The .cfl files go to a job folder on the node's local disk.
TMP=$(mktemp -d "${TMPDIR:-/tmp}/bench_bart_${SLURM_JOB_ID:-$$}_XXXX")
trap 'rm -rf "$TMP"' EXIT
run() {
    echo; echo "## bench_bart.py $*"
    "$PY" -u bart/bench_bart.py --sif "$SIF" --tmpdir "$TMP" "$@" 2>&1 | grep -v "^\[mr_aha\]"
}

OPS="bart_toep bart_toep_triag bart_toep_triag_lowmem bart_pics mr_aha_toeplitz mr_aha_nufft"
CG="bart_pics_cg mr_aha_toeplitz mr_aha_nufft"
# The problems that BART fits on a 24 GB GPU (the TITAN RTX table of bart/README.md).
SMALL=("--problem 2d"
       "--problem 2d72 --L 24"
       "--problem stack --n 160 160 28 --spokes 140 --samples 160"
       "--problem 3d --n 96 96 96 --spokes 5645 --samples 96")
# The full presets.
FULL=("--problem 2d72" "--problem stack" "--problem 3d")

for set in $SETS; do
    case $set in
    small) for p in "${SMALL[@]}"; do run $p --check --methods $OPS; done ;;
    full)  for p in "${FULL[@]}";  do run $p --check --methods $OPS; done ;;
    cg)    for p in "${SMALL[@]}" "${FULL[@]}"; do run $p --methods $CG --cg 20; done ;;
    *)     echo "unknown set $set" ;;
    esac
done
echo "== done $(date)"
