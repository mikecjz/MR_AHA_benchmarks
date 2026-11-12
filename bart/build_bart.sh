#!/usr/bin/env bash
# Build the pinned BART submodule (bart/bart) with CUDA support.
#
# Usage:  bart/build_bart.sh [GPU arch list]      e.g.  bart/build_bart.sh "75 89 90"
#
# Environment:
#   CC         C compiler, GCC >= 12 (default: newest gcc-NN on PATH)
#   CUDA_BASE  CUDA toolkit root (default: dirname of nvcc's bin/, else /usr/local/cuda)
#   JOBS       parallel make jobs (default: nproc)
#
# Needs (Debian/Ubuntu package names): build-essential libfftw3-dev liblapacke-dev
# libopenblas-dev libpng-dev, and a CUDA toolkit (11.x/12.x).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
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

# ---------------------------------------------------------------------------
# bart_aha: the timing helper (bart_aha.c), linked against BART's static
# libraries with the same flags as the bart binary itself.
# ---------------------------------------------------------------------------
B="$src"
libs=""
for l in sense noncart linops iter num misc; do libs+=" $B/lib/lib$l.a"; done
echo "building $here/bart_aha"
$CC -O2 -std=gnu17 -fopenmp -DUSE_CUDA -DUSE_OPENBLAS -I"$CUDA_BASE/include" \
    -iquote "$B/src" -include "$B/src/main.h" \
    "$here/bart_aha.c" $libs $libs \
    -lfftw3f -lfftw3f_threads \
    -L"$CUDA_BASE/$CUDA_LIB" -lcufft -lcudart -lcublas -lstdc++ -Wl,-rpath,"$CUDA_BASE/$CUDA_LIB" \
    -llapacke -lopenblas -lpng -lm -lrt -Wl,-no-warn-execstack \
    -o "$here/bart_aha"
echo "done: $src/bart and $here/bart_aha"
