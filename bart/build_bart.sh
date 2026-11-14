#!/usr/bin/env bash
# Build BART with CUDA support and the timing helper bart_aha.
#
# Two ways, for the two ways bench_bart.py can run BART:
#
#   bart/build_bart.sh [GPU arch list]     native: compile the pinned submodule bart/bart
#                                          and bart/bart_aha.   e.g.  bart/build_bart.sh "75 89 90"
#   bart/build_bart.sh --sif [FILE]        container: BART comes ready-built in an Apptainer /
#                                          Singularity image (FILE, default bart/bart.sif; it is
#                                          downloaded from Docker Hub if it does not exist). Only
#                                          bart/bart_aha_sif is compiled, inside the container.
#                                          For machines without GCC >= 12 (HPC clusters).
#
# Environment:
#   CC          C compiler, GCC >= 12 (default: newest gcc-NN on PATH)            [native]
#   CUDA_BASE   CUDA toolkit root (default: dirname of nvcc's bin/, else /usr/local/cuda) [native]
#   JOBS        parallel make jobs (default: nproc)                                 [native]
#   BART_IMAGE  image that --sif downloads (default: docker://mikecjz/bart:v1.0.01-devel,
#               built from the Dockerfile of the submodule bart/bart; the "devel" image has
#               BART's static libraries, which bart_aha links against)
#   BART_SIF_RUNTIME  apptainer or singularity (default: whichever is on the PATH)
#
# The native build needs (Debian/Ubuntu package names): build-essential libfftw3-dev
# liblapacke-dev libopenblas-dev libpng-dev, and a CUDA toolkit (11.x/12.x).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# bart_aha: the timing helper (bart_aha.c), linked against BART's static
# libraries with the same flags as the bart binary itself.
#   build_helper <BART tree> <compiler> <CUDA root> <CUDA lib folder> <output>
# ---------------------------------------------------------------------------
build_helper() {
    local B="$1" cc="$2" cuda="$3" cudalib="$4" out="$5" libs="" l
    for l in sense noncart linops iter num misc; do libs+=" $B/lib/lib$l.a"; done
    echo "building $out"
    $cc -O2 -std=gnu17 -fopenmp -DUSE_CUDA -DUSE_OPENBLAS -I"$cuda/include" \
        -iquote "$B/src" -include "$B/src/main.h" \
        "$here/bart_aha.c" $libs $libs \
        -lfftw3f -lfftw3f_threads \
        -L"$cuda/$cudalib" -lcufft -lcudart -lcublas -lstdc++ -Wl,-rpath,"$cuda/$cudalib" \
        -llapacke -lopenblas -lpng -lm -lrt -Wl,-no-warn-execstack \
        -o "$out.tmp$$"
    mv "$out.tmp$$" "$out"      # atomic: jobs that build at the same time do not clash
}

# Inside the container (called by the --sif branch below): BART is in /opt/bart.
if [ "${1:-}" = "--helper-in-container" ]; then
    build_helper /opt/bart gcc /usr/local/cuda lib64 "$here/bart_aha_sif"
    exit 0
fi

# ---------------------------------------------------------------------------
# Container: download the image if needed, compile bart_aha_sif inside it.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--sif" ]; then
    sif="${2:-$here/bart.sif}"
    image="${BART_IMAGE:-docker://mikecjz/bart:v1.0.01-devel}"
    rt="${BART_SIF_RUNTIME:-$(command -v apptainer || command -v singularity || true)}"
    [ -n "$rt" ] || { echo "neither apptainer nor singularity found (HPC: module load apptainer)" >&2; exit 1; }
    if [ ! -f "$sif" ]; then
        echo "building $sif from $image"
        "$rt" build "$sif" "$image"
    fi
    sif="$(cd "$(dirname "$sif")" && pwd)/$(basename "$sif")"
    echo "BART in $sif: $("$rt" run "$sif" version)"
    "$rt" exec --bind "$here" "$sif" bash "$here/build_bart.sh" --helper-in-container
    echo "done: $here/bart_aha_sif; run  python bart/bench_bart.py --sif $sif"
    exit 0
fi

# ---------------------------------------------------------------------------
# Native: compile the submodule bart/bart and bart_aha.
# ---------------------------------------------------------------------------
src="$here/bart"
[ -f "$src/Makefile" ] || { echo "bart/bart is empty: git submodule update --init bart/bart" >&2; exit 1; }

archs="${1:-75 80 86 89 90}"
gencode=""
for a in $archs; do gencode+=" -gencode arch=compute_${a},code=sm_${a}"; done

if [ -z "${CUDA_BASE:-}" ]; then
    if command -v nvcc >/dev/null; then CUDA_BASE="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
    else CUDA_BASE=/usr/local/cuda; fi
fi
# The CUDA libraries live in lib64 (or targets/x86_64-linux/lib) of a toolkit.
CUDA_LIB=lib64
[ -d "$CUDA_BASE/lib64" ] || CUDA_LIB=lib

# BART needs GCC >= 12. Use $CC if set, else the newest gcc-1[2-9] found, else gcc.
if [ -z "${CC:-}" ]; then
    CC=gcc
    for v in 15 14 13 12; do command -v gcc-$v >/dev/null && { CC=gcc-$v; break; }; done
fi
echo "C compiler: $CC ($($CC -dumpversion))"

echo "building BART in $src with CUDA_BASE=$CUDA_BASE, archs: $archs"
make -C "$src" -j "${JOBS:-$(nproc)}" \
    CUDA=1 CUDA_BASE="$CUDA_BASE" CUDA_LIB="$CUDA_LIB" \
    CC="$CC" CUDA_CC="$CC" GPUARCH_FLAGS="$gencode" OPENBLAS=1 OMP=1 \
    bart
"$src/bart" version

build_helper "$src" "$CC" "$CUDA_BASE" "$CUDA_LIB" "$here/bart_aha"
echo "done: $src/bart and $here/bart_aha"
