/**
 * @file bart_aha.c
 * @brief Time BART's low-rank (subspace) normal operator A^H A on the GPU.
 *
 * BART's command-line tools have no command that applies only the normal
 * operator, and its Python interface (`bart.py`) runs those tools as separate
 * processes that exchange `.cfl` files. To time exactly the operator that
 * `bart pics` uses inside its conjugate-gradient iterations, this small
 * program links against BART's static libraries and builds the same model as
 * `pics` (`sense_nc_init()` in `src/sense/modelnc.c`):
 *
 *     A = NUFFT_basis o maps,      A^H A = maps^H o NUFFT_basis^H NUFFT_basis o maps
 *
 * where `NUFFT_basis` is BART's NUFFT linear operator with a temporal basis
 * (`nufft_create2()` with `basis != NULL`). With Toeplitz embedding (BART's
 * default for the normal operator) the inner part is evaluated as
 * FFT -> multiplication by the L x L point-spread function (PSF) -> IFFT
 * (`toeplitz_mult()` in `src/noncart/nufft.c`), optionally with the PSF
 * stored as its upper triangle (`--upper-triag`, BART's
 * `--nufft-conf --upper-triag-psf`).
 *
 * It is driven by `bench_bart.py`, which writes the inputs with BART's
 * `cfl.py`, so all arrays use BART's 16-dimensional column-major layout:
 *
 * | file   | BART dims                                   | meaning |
 * |--------|---------------------------------------------|---------|
 * | traj   | [3, Nkx, 1, 1, 1, T]                         | k-space coordinates (BART units: cycles/FOV, i.e. -N/2..N/2) |
 * | basis  | [1, 1, 1, 1, 1, T, L]                        | temporal basis, one row per time frame |
 * | maps   | [Nx, Ny, Nz, Ncoils]                         | coil sensitivities |
 * | x      | [Nx, Ny, Nz, 1, 1, 1, L]                     | spatial coefficients |
 * | out    | [Nx, Ny, Nz, 1, 1, 1, L]                     | A^H A x (written only with `--out`) |
 *
 * Usage:
 * @code
 *   bart_aha [options] <traj> <basis> <maps> <x>
 *     -g               run on the GPU (default: CPU)
 *     --upper-triag    store only the upper triangle of the PSF
 *     --lowmem         BART's low-memory NUFFT mode (loops over the 2^d PSF factors)
 *     --no-toeplitz    no Toeplitz embedding: gridding NUFFT forward + adjoint
 *     --reps N         timed applications (default 10)
 *     --warmup N       untimed applications first (default 2)
 *     --out NAME       write A^H A x to NAME.{cfl,hdr}
 * @endcode
 *
 * It prints one JSON object on stdout:
 * `setup_s` (operator creation, PSF included), `apply_ms` (all timed
 * applications, milliseconds), `mem_gb` (device memory in use after the
 * warm-up minus before creation; BART keeps freed GPU blocks in its own
 * cache, so this is close to the peak).
 */

#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include "num/gpuops.h"
#endif

#include "linops/linop.h"
#include "noncart/nufft.h"
#include "sense/modelnc.h"

/** Wall-clock time in seconds (monotonic). */
static double now_s(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/** Wait for all queued GPU work (no-op on the CPU). */
static void sync_dev(bool gpu)
{
#ifdef USE_CUDA
	if (gpu)
		cuda_sync_device();
#else
	(void)gpu;
#endif
}

/** Device memory in use, in bytes (0 on the CPU). */
static double used_bytes(bool gpu)
{
#ifdef USE_CUDA
	if (gpu) {

		size_t fr = 0, tot = 0;
		cudaMemGetInfo(&fr, &tot);
		return (double)tot - (double)fr;
	}
#else
	(void)gpu;
#endif
	return 0.;
}

/** Copy an array to the GPU when `gpu` is set; otherwise return it unchanged. */
static complex float* to_dev(bool gpu, const long dims[DIMS], complex float* x)
{
#ifdef USE_CUDA
	if (gpu)
		return md_gpu_move(DIMS, dims, x, CFL_SIZE);
#else
	(void)gpu; (void)dims;
#endif
	return x;
}

int main(int argc, char* argv[])
{
	bool gpu = false;
	bool upper = false;
	bool lowmem = false;
	bool toeplitz = true;
	int reps = 10;
	int warmup = 2;
	const char* out_name = NULL;
	const char* pos[4];
	int npos = 0;

	for (int i = 1; i < argc; i++) {

		if (0 == strcmp(argv[i], "-g"))
			gpu = true;
		else if (0 == strcmp(argv[i], "--upper-triag"))
			upper = true;
		else if (0 == strcmp(argv[i], "--lowmem"))
			lowmem = true;
		else if (0 == strcmp(argv[i], "--no-toeplitz"))
			toeplitz = false;
		else if ((0 == strcmp(argv[i], "--reps")) && (i + 1 < argc))
			reps = atoi(argv[++i]);
		else if ((0 == strcmp(argv[i], "--warmup")) && (i + 1 < argc))
			warmup = atoi(argv[++i]);
		else if ((0 == strcmp(argv[i], "--out")) && (i + 1 < argc))
			out_name = argv[++i];
		else if ((npos < 4) && ('-' != argv[i][0]))
			pos[npos++] = argv[i];
		else {

			fprintf(stderr, "usage: %s [-g] [--upper-triag] [--lowmem] [--no-toeplitz] "
					"[--reps N] [--warmup N] [--out NAME] <traj> <basis> <maps> <x>\n", argv[0]);
			return 1;
		}
	}

	if (4 != npos) {

		fprintf(stderr, "need 4 input files: traj basis maps x\n");
		return 1;
	}

	debug_level = DP_WARN;
	num_init();

	if (gpu) {
#ifdef USE_CUDA
		bart_use_gpu = true;
		num_init_gpu_support();
#else
		fprintf(stderr, "BART was built without CUDA\n");
		return 1;
#endif
	}

	// ---- inputs (host memory, mapped from the .cfl files) ----

	long trj_dims[DIMS], bas_dims[DIMS], map_dims[DIMS], img_dims[DIMS];
	complex float* traj = load_cfl(pos[0], DIMS, trj_dims);
	complex float* basis = load_cfl(pos[1], DIMS, bas_dims);
	complex float* maps = load_cfl(pos[2], DIMS, map_dims);
	complex float* x = load_cfl(pos[3], DIMS, img_dims);

	// k-space: [1, Nkx, (spokes), Ncoils, 1, T] -- as pics derives it from the data.
	long ksp_dims[DIMS];
	md_select_dims(DIMS, ~(COEFF_FLAG | READ_FLAG), ksp_dims, trj_dims);
	ksp_dims[READ_DIM] = 1;
	ksp_dims[COIL_DIM] = map_dims[COIL_DIM];

	// max_dims of pics_model(): image dims + coils + coefficient dimension.
	long max_dims[DIMS];
	md_copy_dims(DIMS, max_dims, ksp_dims);
	md_copy_dims(5, max_dims, map_dims);
	max_dims[COEFF_DIM] = bas_dims[COEFF_DIM];
	max_dims[TE_DIM] = 1;

	struct nufft_conf_s conf = nufft_conf_defaults;
	conf.toeplitz = toeplitz;
	conf.upper_triag = upper;
	conf.lowmem = lowmem;

	double mem0 = used_bytes(gpu);

	// ---- operator creation: PSF on the GPU when traj is on the GPU ----

	sync_dev(gpu);
	double t0 = now_s();

	complex float* traj_d = to_dev(gpu, trj_dims, traj);
	complex float* maps_d = to_dev(gpu, map_dims, maps);

	const struct linop_s* op = sense_nc_init(max_dims, map_dims, maps_d, ksp_dims,
						trj_dims, traj_d, &conf,
						NULL, NULL, bas_dims, basis, NULL, 0UL);
	sync_dev(gpu);
	double setup_s = now_s() - t0;

	// ---- application ----

	complex float* x_d = to_dev(gpu, img_dims, x);
	complex float* y_d = md_alloc_sameplace(DIMS, img_dims, CFL_SIZE, x_d);

	for (int i = 0; i < warmup; i++)
		linop_normal(op, DIMS, img_dims, y_d, x_d);

	sync_dev(gpu);
	double mem1 = used_bytes(gpu);

	double* ms = calloc((size_t)(reps > 0 ? reps : 1), sizeof(double));

	for (int i = 0; i < reps; i++) {

		sync_dev(gpu);
		double t1 = now_s();
		linop_normal(op, DIMS, img_dims, y_d, x_d);
		sync_dev(gpu);
		ms[i] = 1e3 * (now_s() - t1);
	}

	if (NULL != out_name) {

		complex float* out = create_cfl(out_name, DIMS, img_dims);
		md_copy(DIMS, img_dims, out, y_d, CFL_SIZE);
		unmap_cfl(DIMS, img_dims, out);
	}

	printf("{\"setup_s\": %.6f, \"mem_gb\": %.4f, \"apply_ms\": [", setup_s, (mem1 - mem0) / 1073741824.);

	for (int i = 0; i < reps; i++)
		printf("%s%.4f", i ? ", " : "", ms[i]);

	printf("]}\n");

	free(ms);
	linop_free(op);
	md_free(y_d);

	if (x_d != x)
		md_free(x_d);
	if (traj_d != traj)
		md_free(traj_d);
	if (maps_d != maps)
		md_free(maps_d);

	unmap_cfl(DIMS, trj_dims, traj);
	unmap_cfl(DIMS, bas_dims, basis);
	unmap_cfl(DIMS, map_dims, maps);
	unmap_cfl(DIMS, img_dims, x);

	return 0;
}
