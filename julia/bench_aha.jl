#!/usr/bin/env julia
"""
    bench_aha.jl

GPU speed benchmark of the low-rank (subspace) normal operator ``A^H A`` of
[MRISubspaceRecon.jl](https://github.com/MagneticResonanceImaging/MRISubspaceRecon.jl)
on the same problems as the MR_AHA benchmark (`python/bench_aha.py`).

# What is timed

MRISubspaceRecon.jl offers one non-Cartesian normal operator, `NFFTNormalOp`, a
Toeplitz embedding (see `julia/README.md` for the details and source citations):

* **set-up**: `calculate_kernel_noncartesian` (the k-space mask of the Toeplitz
  grid and the packed upper-triangular kernel `Λ`, one type-1 NUFFT + FFT per
  triangular entry) followed by `NFFTNormalOp(img_shape, Λ, kmask_indcs; cmaps)`
  (work buffers, cuFFT plans). This is exactly what the public constructor
  `NFFTNormalOp(img_shape, trj, U; cmaps)` does; the two steps are called
  separately only to report the kernel time on its own.
* **A^H A**: one `mul!(y, A, x)`, median over `--reps` runs after `--warmup`
  runs.
* **mem GB**: free device memory before creating the operator minus free device
  memory after the warm-up applications (CUDA.jl's pool is reclaimed first), as
  in the Python benchmark.

Every timed section is bracketed by `CUDA.synchronize()`. Before the first
problem of each dimensionality a tiny problem of that dimensionality is created
and applied once, so that Julia's compilation of the GPU kernels and the loading
of cuFFT are not charged to the first measurement (`--setup-reps 3` additionally
separates the first, cold, creation from the median of the others).

The package has no NUFFT-based (non-Toeplitz) normal operator, so only the
Toeplitz operator is timed.

# Problem translation

The problems are those of `python/problems.py`, written to a `.mat` file by
`julia/save_problem.py` (done automatically when a preset file is missing) and
read here, so the trajectory, Gram blocks, maps and coefficients are
bit-identical to the ones MR_AHA is timed with. MRISubspaceRecon.jl does not take
Gram blocks but a temporal basis `U` (`Nframes x L`) and one trajectory per time
frame; its kernel is ``Λ_{ij}(k) = Σ_t conj(U[t,i]) U[t,j] psf_t(k)``. Each
Gram block ``Φ_2 = Σ_k λ_k v_k v_k^H`` (Hermitian, here of rank 2: two time points
per readout) is therefore factored by an eigen-decomposition into `r` time
frames that share the readout's k-space positions, with
`U[frame_k, :] = sqrt(λ_k) * transpose(v_k)`. This reproduces the operator of
MR_AHA (`v <- v * Φ_2` per k-space point) exactly, at the price of `r` times as
many non-uniform points in the kernel computation (`r = 2` for the presets).
See [`gram_to_basis`](@ref).

Trajectories are converted from radians in ``[-π, π)`` to cycles in
``[-1/2, 1/2)``. A stack-of-stars becomes a 3D trajectory with Cartesian kz
(the package has no stack mode), so its Toeplitz grid is oversampled by 2 in z as
well.

# Usage

    julia --project=julia julia/bench_aha.jl [--problem 2d 2d72 stack 3d]
          [--data FILE.mat] [--reps 10] [--warmup 2] [--setup-reps 3]
          [--out FILE.json] [--save-result FILE.mat] [--no-json]

Run with one GPU only, e.g. `CUDA_VISIBLE_DEVICES=3`.
"""

using CUDA
using LinearAlgebra
using LinearOperators
using MAT
using JSON
using Printf
using Statistics
using Dates
using MRISubspaceRecon
import NonuniformFFTs
import Pkg

"""The CUDA extension of MRISubspaceRecon.jl, which holds the GPU methods."""
const EXT = Base.get_extension(MRISubspaceRecon, :MRISubspaceReconCUDAExt)

"""Root of the benchmarks repository (parent of `julia/`)."""
const REPO = dirname(@__DIR__)

"""Preset names understood by `--problem` (those of `python/problems.py`)."""
const PRESETS = ("2d", "2d72", "stack", "3d")

# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

"""
    parse_args(argv) -> Dict{String,Any}

Parse the command line (see the module docstring). Unknown options are an error.
"""
function parse_args(argv)
    a = Dict{String,Any}("problem" => String[], "data" => nothing, "reps" => 10,
                         "warmup" => 2, "setup-reps" => 3, "out" => nothing,
                         "save-result" => nothing, "json" => true)
    i = 1
    while i <= length(argv)
        opt = argv[i]
        if opt == "--problem"
            i += 1
            while i <= length(argv) && !startswith(argv[i], "--")
                push!(a["problem"], argv[i]); i += 1
            end
            continue
        elseif opt in ("--data", "--out", "--save-result")
            a[opt[3:end]] = argv[i+1]; i += 1
        elseif opt in ("--reps", "--warmup", "--setup-reps")
            a[opt[3:end]] = parse(Int, argv[i+1]); i += 1
        elseif opt == "--no-json"
            a["json"] = false
        elseif opt in ("-h", "--help")
            # Print the file's docstring (between its first two triple quotes).
            src = read(@__FILE__, String); q = findall("\"\"\"", src)
            println(src[q[1].stop+2:q[2].start-1]); exit(0)
        else
            error("unknown option $opt")
        end
        i += 1
    end
    isempty(a["problem"]) && a["data"] === nothing && push!(a["problem"], "2d")
    for p in a["problem"]
        p in PRESETS || error("--problem: unknown preset $p (have $(join(PRESETS, ", ")))")
    end
    return a
end

# --------------------------------------------------------------------------
# Problems
# --------------------------------------------------------------------------

"""
    preset_file(name) -> String

Path of the `.mat` file of a preset (`julia/data/<name>.mat`), written by
`julia/save_problem.py` with the Python of `\$PYTHON` (default `python`) if it
does not exist yet.
"""
function preset_file(name)
    f = joinpath(@__DIR__, "data", "$name.mat")
    if !isfile(f)
        py = get(ENV, "PYTHON", "python")
        run(`$py $(joinpath(@__DIR__, "save_problem.py")) $name $f`)
    end
    return f
end

"""
    load_problem(path) -> NamedTuple

Read a problem written by `python/problems.py:save` (column-major MR_AHA layout,
see `julia/save_problem.py`) and return `(traj_type, n_dim, L, n_coils, SEs,
phi2, traj, x, desc)` with Julia arrays in that same layout.
"""
function load_problem(path)
    d = matread(path)
    tt = lowercase(strip(string(d["traj_type"])))
    SEs = ComplexF32.(d["SEs"])
    phi2 = ComplexF32.(d["phi2"])
    traj = Float32.(d["traj"])
    x = ComplexF32.(d["x"])
    n_dim = size(SEs)[1:end-1]
    L = size(phi2, ndims(phi2))
    ns = size(traj)
    desc = "$(tt == "2d" ? "2D" : tt == "3d" ? "3D" : "stack") $(join(n_dim, "x")), " *
           "L=$L, $(size(SEs)[end]) coils, $(ns[1])x$(ns[2]) samples" *
           (tt == "stack" ? " x $(n_dim[3]) kz" : "")
    return (traj_type=tt, n_dim=n_dim, L=L, n_coils=size(SEs)[end], SEs=SEs,
            phi2=phi2, traj=traj, x=x, desc=desc)
end

"""
    gram_to_basis(G; rtol=1e-5) -> (U, r)

Factor Gram blocks into a temporal basis.

`G` is `[nb, L, L]` (block `b` is `G[b, :, :]`, the MR-Multitasking
``Φ_2 = Φ_{rt}(:,idx) Φ_{rt}(:,idx)'``). Each Hermitian block is
eigen-decomposed, ``Φ_2 = Σ_k λ_k v_k v_k^H``, and the `r` largest
eigenpairs become `r` time frames with basis rows `sqrt(λ_k) * transpose(v_k)`.

Why this is the same operator: MR_AHA maps the rank coefficients `v` of one k-space point to
``v_l ← Σ_m v_m Φ_2(m, l) = Σ_k conj(Φ(l,t_k)) Σ_m Φ(m,t_k) v_m``, and
MRISubspaceRecon.jl maps them to ``v_i ← Σ_j Σ_t conj(U[t,i]) U[t,j] v_j``;
the two agree for `U[t_k, :] = Φ(:, t_k)`, i.e. the rank x time basis
transposed without conjugation, and ``Φ(:, t_k) = sqrt(λ_k) v_k``.

`r` is the largest number of eigenvalues above `rtol` times the block's largest
one, over all blocks (2 for the synthetic presets, whose blocks come from two
time points); blocks of lower rank get zero rows, which changes nothing.

Returns `U` as a `Matrix{ComplexF32}` `[r*nb, L]` with frame `k + r*(b-1)`
belonging to block `b`, and `r`.
"""
function gram_to_basis(G; rtol=1e-5)
    nb, L = size(G, 1), size(G, 2)
    eig = Vector{Any}(undef, nb)
    r = 0
    Threads.@threads for b in 1:nb
        e = eigen(Hermitian(ComplexF64.(G[b, :, :])))
        eig[b] = e
    end
    for b in 1:nb
        λ = eig[b].values
        r = max(r, count(>(rtol * max(maximum(λ), eps())), λ))
    end
    U = zeros(ComplexF32, r * nb, L)
    for b in 1:nb
        λ, V = eig[b].values, eig[b].vectors   # ascending order
        for k in 1:r
            j = L - k + 1
            λ[j] > 0 || continue
            U[k+r*(b-1), :] .= ComplexF32.(sqrt(λ[j]) .* V[:, j])
        end
    end
    return U, r
end

"""
    to_package(p) -> (trj, U, cmaps_host, img_shape, r)

Translate an MR_AHA problem into the inputs of MRISubspaceRecon.jl:

* `trj`: `Float32` `[ndim, Nkx, Nframes]` in cycles, ``[-1/2, 1/2)``, one
  trajectory per time frame (readouts repeated `r` times, see
  [`gram_to_basis`](@ref); for a stack, frames over (readout, kz)).
* `U`: `ComplexF32` `[Nframes, L]`.
* `cmaps_host`: vector of the coil maps.
"""
function to_package(p)
    Ntr, Nkx, nd = size(p.traj)
    k = permutedims(p.traj, (3, 2, 1)) ./ Float32(2π)       # [nd, Nkx, Ntr], cycles
    if p.traj_type == "stack"
        Nz = p.n_dim[3]
        L = p.L
        # Blocks ordered (readout fastest, kz), as in phi2 [Ntr, Nkz, L, L].
        U, r = gram_to_basis(reshape(p.phi2, Ntr * Nz, L, L))
        trj = Array{Float32}(undef, 3, Nkx, r * Ntr * Nz)
        for z in 1:Nz, t in 1:Ntr
            # kz in centred order: index z (1-based) is frequency z-1-Nz÷2,
            # i.e. (z-1-Nz÷2)/Nz cycles per sample.
            kz = Float32((z - 1 - Nz ÷ 2) / Nz)
            b = t + Ntr * (z - 1)
            for j in 1:r
                f = j + r * (b - 1)
                trj[1:2, :, f] .= k[:, :, t]
                trj[3, :, f] .= kz
            end
        end
    else
        U, r = gram_to_basis(p.phi2)
        trj = Array{Float32}(undef, nd, Nkx, r * Ntr)
        for t in 1:Ntr, j in 1:r
            trj[:, :, j+r*(t-1)] .= k[:, :, t]
        end
    end
    # NonuniformFFTs wants points in [-1/2, 1/2); the radial presets hit -1/2
    # exactly and stay below +1/2, so this only guards against round-off.
    clamp!(trj, -0.5f0, prevfloat(0.5f0))
    cm = [p.SEs[(Colon() for _ in p.n_dim)..., c] for c in 1:p.n_coils]
    return trj, U, cm, Tuple(p.n_dim), r
end

"""
    tiny_problem(ndim) -> NamedTuple

A tiny random problem (32^ndim, L=2, 2 coils) for the warm-up.
"""
function tiny_problem(ndim)
    n = ntuple(_ -> 16, ndim)
    Ntr, Nkx = 16, 16
    trj = rand(Float32, ndim, Nkx, 2Ntr) .- 0.5f0
    U = randn(ComplexF32, 2Ntr, 2)
    cm = [randn(ComplexF32, n...) for _ in 1:2]
    return trj, U, cm, n
end

# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

"""Free device memory in bytes, after releasing CUDA.jl's cached blocks."""
function free_bytes()
    GC.gc(true)
    CUDA.reclaim()
    CUDA.synchronize()
    return CUDA.free_memory()
end

"""
    make_op(trj, U, cmaps, img_shape) -> (A, kernel_s, n_mask)

Create the GPU Toeplitz normal operator from device arrays. Equivalent to
`NFFTNormalOp(img_shape, trj, U; cmaps)` (CUDA extension,
`ext/MRISubspaceReconCUDAExt/NFFTNormalOp.jl:19-31`), split into its two steps to
time the kernel separately.
"""
function make_op(trj, U, cmaps, img_shape)
    CUDA.synchronize()
    t0 = time_ns()
    Λ, kmask = EXT.calculate_kernel_noncartesian(2 .* img_shape, trj, U)
    CUDA.synchronize()
    kernel_s = (time_ns() - t0) / 1e9
    A = NFFTNormalOp(img_shape, Λ, kmask; cmaps=cmaps)
    return A, kernel_s, length(kmask)
end

"""
    timed_create(trj, U, cmaps, img_shape; keep) -> (t, kernel_s[, A, n_mask])

Create the operator with [`make_op`](@ref), timing the whole creation
(synchronised). With `keep=false` the operator is dropped before returning, so
that it can be garbage collected.
"""
function timed_create(trj, U, cmaps, img_shape; keep)
    CUDA.synchronize()
    t = time_ns()
    A, kernel_s, nmask = make_op(trj, U, cmaps, img_shape)
    CUDA.synchronize()
    t = (time_ns() - t) / 1e9
    return keep ? (t, kernel_s, A, nmask) : (t, kernel_s)
end

"""
    warm_up(ndim)

Create and apply an operator once on a tiny problem of dimensionality `ndim`,
so that compiling the GPU kernels for this array rank is not timed.
"""
function warm_up(ndim)
    trj, U, cm, n = tiny_problem(ndim)
    A, _, _ = make_op(CuArray(trj), CuArray(U), CuArray.(cm), n)
    x = CUDA.randn(ComplexF32, prod(n) * 2)
    y = similar(x)
    mul!(y, A, x)
    CUDA.synchronize()
    return nothing
end

"""
    run_problem(name, p, args) -> Dict

Translate, create, time and release the operator for one problem; print a table
row and return the row for the JSON report (with the result `y` on the host
under `"_y"` when `--save-result` is given).
"""
function run_problem(name, p, args)
    t0 = time()
    trj, U, cm, img_shape, r = to_package(p)
    prep_s = time() - t0
    ntri = p.L * (p.L + 1) ÷ 2
    grid = 2 .* img_shape
    row = Dict{String,Any}("method" => "mrisubspacerecon_toeplitz", "gpus" => 1,
                           "frames_per_readout" => r, "n_frames" => size(U, 1),
                           "toeplitz_grid" => collect(grid), "prep_s" => prep_s)
    @printf("\n== %s: %s\n   unmasked packed kernel would be %.2f GB on the %s grid; %d time frames per readout (Gram blocks -> basis in %.1f s)\n",
            name, p.desc, ntri * prod(grid) * 8 / 1e9, join(grid, "x"), r, prep_s)
    println(header())
    A = nothing; x = nothing; y = nothing
    try
        d_trj, d_U, d_cm = CuArray(trj), CuArray(U), CuArray.(cm)
        x = CuArray(vec(p.x))
        mem0 = free_bytes()
        times = Float64[]; ks = Float64[]
        # All creations but the last are timed in their own function frame and
        # discarded there, so that the operator is unreachable (and its device
        # memory reclaimable) before the next one is built; holding it in this
        # frame kept it alive and made the large presets run out of memory.
        for i in 1:args["setup-reps"]-1
            t, kernel_s = timed_create(d_trj, d_U, d_cm, img_shape; keep=false)
            push!(times, t); push!(ks, kernel_s)
            free_bytes()
        end
        t, kernel_s, A, nmask = timed_create(d_trj, d_U, d_cm, img_shape; keep=true)
        push!(times, t); push!(ks, kernel_s)
        row["setup_cold_s"] = times[1]
        row["setup_s"] = length(times) > 1 ? median(times[2:end]) : times[1]
        row["kernel_s"] = length(ks) > 1 ? median(ks[2:end]) : ks[1]
        row["mask_keep"] = nmask / prod(grid)
        row["kernel_gb"] = ntri * nmask * 8 / 1e9
        y = similar(x)
        for _ in 1:args["warmup"]
            mul!(y, A, x)
        end
        CUDA.synchronize()
        row["mem_gb"] = (mem0 - free_bytes()) / 1e9
        ts = Float64[]
        for _ in 1:args["reps"]
            CUDA.synchronize()
            t = time_ns()
            mul!(y, A, x)
            CUDA.synchronize()
            push!(ts, (time_ns() - t) / 1e9)
        end
        row["aha_ms"] = median(ts) * 1e3
        row["aha_min_ms"] = minimum(ts) * 1e3
        row["aha_mean_ms"] = mean(ts) * 1e3
        row["reps"] = args["reps"]
        if args["save-result"] !== nothing
            row["_y"] = reshape(Array(y), size(p.x))
        end
    catch e
        msg = sprint(showerror, e)
        row["error"] = first(split(msg, '\n'))
    finally
        A = nothing; x = nothing; y = nothing
        free_bytes()
    end
    println(fmt_row(row))
    return row
end

"""Column headings of the result table (as in `python/bench_aha.py`)."""
header() = @sprintf("  %-26s %4s %9s %9s %10s %8s %8s", "method", "GPUs",
                    "set-up s", "kernel s", "A^H A ms", "mem GB", "Λ GB")

"""Format one result row."""
function fmt_row(r)
    haskey(r, "error") && return @sprintf("  %-26s %4d   failed: %s", r["method"], 1, r["error"])
    return @sprintf("  %-26s %4d %9.3f %9.3f %10.2f %8.2f %8.2f", r["method"], 1,
                    r["setup_s"], r["kernel_s"], r["aha_ms"], r["mem_gb"], r["kernel_gb"])
end

"""
    gpu_folder(name) -> String

Folder name for a GPU model, as `gpu_folder` in `python/bench_aha.py`:
`"NVIDIA TITAN RTX"` -> `"titan_rtx"`, `"NVIDIA H100 80GB HBM3"` -> `"h100"`.
"""
function gpu_folder(name)
    words = [m.match for m in eachmatch(r"[a-z0-9]+", lowercase(name))]
    drop = r"^(nvidia|apple|\d+gb|hbm\d*|pcie|sxm\d*)$"
    w = filter(s -> !occursin(drop, s), words)
    return isempty(w) ? "gpu" : join(w, "_")
end

"""Version of a package in the active project, or `"unknown"`."""
function pkg_version(name)
    for (_, info) in Pkg.dependencies()
        info.name == name && return string(info.version)
    end
    return "unknown"
end

"""
    main(argv) -> Int

Run the benchmark; return 0 if every problem ran, 1 otherwise.
"""
function main(argv=ARGS)
    args = parse_args(argv)
    CUDA.functional() || error("CUDA.jl finds no usable GPU")
    gname = CUDA.name(CUDA.device())
    println("julia $(VERSION), CUDA.jl $(pkg_version("CUDA")), MRISubspaceRecon " *
            "$(pkg_version("MRISubspaceRecon")), $(Threads.nthreads()) CPU thread(s); " *
            "GPU: $gname")
    todo = Tuple{String,String}[]
    args["data"] !== nothing && push!(todo, (splitext(basename(args["data"]))[1], args["data"]))
    for p in args["problem"]
        push!(todo, (p, preset_file(p)))
    end
    report = Dict{String,Any}(
        "backend" => "julia-cuda", "julia" => string(VERSION),
        "cuda_jl" => pkg_version("CUDA"),
        "mrisubspacerecon" => pkg_version("MRISubspaceRecon"),
        "nonuniformffts" => pkg_version("NonuniformFFTs"),
        "cuda_runtime" => string(CUDA.runtime_version()),
        "gpus" => [Dict("id" => 0, "name" => gname,
                        "mem_gb" => round(CUDA.total_memory() / 2^30; digits=1))],
        "time" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
        "args" => Dict(k => v for (k, v) in args),
        "env" => Dict(k => v for (k, v) in ENV if k == "CUDA_VISIBLE_DEVICES"),
        "problems" => Any[])
    out = args["out"] !== nothing ? args["out"] :
          joinpath(REPO, "results", gpu_folder(gname),
                   "bench_aha_julia_$(Dates.format(now(), "yyyymmdd-HHMMSS")).json")
    warmed = Set{Int}()
    failed = 0
    for (name, file) in todo
        p = load_problem(file)
        nd = p.traj_type == "2d" ? 2 : 3
        if !(nd in warmed)
            warm_up(nd); push!(warmed, nd)
        end
        row = run_problem(name, p, args)
        failed += haskey(row, "error")
        if haskey(row, "_y")
            matwrite(args["save-result"], Dict("y" => pop!(row, "_y")))
            println("A^H A x written to $(args["save-result"])")
        end
        push!(report["problems"], Dict("name" => name, "desc" => p.desc,
              "traj_type" => p.traj_type, "n_dim" => collect(p.n_dim), "L" => p.L,
              "n_coils" => p.n_coils, "rows" => [row]))
        if args["json"]
            mkpath(dirname(out))
            open(out, "w") do io
                JSON.print(io, report, 2)
            end
        end
        p = nothing; GC.gc(true)
    end
    args["json"] && println("\nresults written to $out")
    return failed > 0 ? 1 : 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
