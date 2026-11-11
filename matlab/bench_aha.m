function report = bench_aha(varargin)
%BENCH_AHA  Time the A'A operator of MR_AHA: Toeplitz vs NUFFT, 1 or more GPUs.
%
%   BENCH_AHA()                                   % preset '2d', all methods, 1 GPU
%   BENCH_AHA('Name', Value, ...)
%   report = BENCH_AHA(...)
%
%   Builds a test problem, creates the forward normal operator with each
%   selected method on each selected number of GPUs, and reports how long the
%   set-up and one application take.
%
%   METHODS ('methods')
%     'toeplitz'      MR_AHA_OP.CREATE_OP(..., 'method', 'toeplitz'). The
%                     Toeplitz kernel is computed at creation; one
%                     application is, per coil, a zero-padded FFT, a
%                     block-diagonal multiplication and an inverse FFT. One
%                     or several GPUs.
%     'nufft_c'       MR_AHA_OP.CREATE_OP(..., 'method', 'nufft'): per coil a
%                     type-2 NUFFT, the Gram multiplication and a type-1
%                     NUFFT, all inside the C/CUDA library. One or several
%                     GPUs.
%     'nufft_native'  MR_NUFFT_PLAN + AHA_CUFINUFFT: the same NUFFT operator
%                     written in MATLAB with gpuArrays and cufinufft's own
%                     MATLAB interface, without the library. Always one GPU.
%                     Needs cufinufft's MATLAB interface (CUFINUFFT_PLAN) on
%                     the path, see 'cufinufft_dir'; skipped otherwise. It
%                     runs last: on some machines that MEX file aborts
%                     MATLAB, and the other results are saved before it.
%
%   NAME-VALUE PAIRS: THE PROBLEM
%     'problem'     preset name or cell array of names: '2d' (default),
%                   '2d72', 'stack', '3d'; see BENCH_PROBLEM.
%     'n'           matrix size [Nx Ny] or [Nx Ny Nz] instead of the preset's
%     'L'           number of ranks
%     'coils'       number of coils
%     'spokes'      number of readouts
%     'samples'     samples per readout
%     'data'        .mat file with SEs, phi2, traj (and traj_type, x) to time
%                   instead of a preset; layout in BENCH_PROBLEM
%     'traj_type'   trajectory type of 'data' when the file has none
%     'seed'        seed of the random inputs (default 0)
%
%   NAME-VALUE PAIRS: WHAT TO RUN
%     'methods'     cell array, subset of the three methods (default: all
%                   that are available)
%     'gpus'        vector of GPU counts, e.g. [1 2 4], or 'all' (default 1).
%                   The current gpuDevice is always the primary GPU; the
%                   others are added in gpuDevice order. 'nufft_native'
%                   always uses one.
%     'mask'        Toeplitz operator: 'off' (unmasked, default), 'on'
%                   (masked, with the mask of CALC_MASK) or 'both'
%     'toep_dist'   Toeplitz on several GPUs: 'auto' (default), 'split',
%                   'replicate' or 'broadcast'
%     'upsampfac'   NUFFT methods: cufinufft upsampling factor (default 2)
%     'gpu_method'  NUFFT methods: cufinufft gpu_method for both transforms,
%                   1 or 2 (default: the library's choice)
%     'eps'         NUFFT tolerance (default 1e-6)
%
%   NAME-VALUE PAIRS: THE MEASUREMENT
%     'reps'        timed applications (default 10)
%     'warmup'      untimed applications first (default 2)
%     'setup_reps'  operator creations; above 1 the first is reported as
%                   cold and the median of the rest as set-up (default 1)
%     'cg'          also time this many conjugate-gradient iterations
%                   (default 0)
%     'check'       true: report the relative difference of every A'A x to
%                   the first row's (default false)
%     'out'         JSON output file (default
%                   results/<host>/bench_aha_matlab_<time>.json)
%
%   NAME-VALUE PAIRS: WHERE THE LIBRARY IS
%     'mr_aha_root'    MR_AHA checkout. Default: the environment variable
%                      MR_AHA_ROOT, else third_party/MR_AHA of this
%                      repository. Its matlab/ folder is put on the path.
%     'mex_dir'        folder with the MEX files. Default: the environment
%                      variable MR_AHA_MEX_DIR, else <mr_aha_root>/build/matlab.
%     'cufinufft_dir'  folder with cufinufft's MATLAB interface. Default:
%                      the finufft/matlab folder next to mex_dir (built with
%                      -DMR_AHA_BUILD_CUFINUFFT_MATLAB=ON), if it exists.
%
%   WHAT IS REPORTED, for every method and GPU count
%     set-up    wall-clock time of creating the operator (kernel, plans,
%               copies to the other GPUs). The first creation on a GPU also
%               pays for one-off initialisation; a tiny warm-up problem takes
%               that cost on the primary GPU before the first table.
%     A'A       median wall-clock time of one application
%     speed-up  time of the same method on one GPU divided by this row's
%     mem       memory in use on the primary GPU after the warm-up runs
%               minus before the operator was created (MATLAB cannot query
%               the other GPUs without resetting them)
%     CG        time of 'cg' plain conjugate-gradient iterations
%     diff      relative difference of A'A x to the first row's (marked
%               ref). Unmasked rows agree to about the NUFFT tolerance; a
%               masked Toeplitz operator is an approximation and differs
%               more, about 1e-2 on the random x used here.
%
%   OUTPUT
%     report   struct with the machine, the options and, per problem, one
%              struct per row; the same content is written as JSON.
%
%   EXAMPLES
%     bench_aha
%     bench_aha('problem', {'stack', '3d'}, 'gpus', [1 2 4])
%     bench_aha('problem', 'stack', 'methods', {'toeplitz'}, 'mask', 'both')
%     bench_aha('problem', '3d', 'n', [96 96 96], 'L', 8, 'check', true, 'cg', 10)
%     bench_aha('data', 'my_scan.mat', 'methods', {'toeplitz', 'nufft_c'})
%
%   On Apple silicon (Metal backend) the same call works with ordinary
%   MATLAB arrays on the one GPU.
%
%   See also BENCH_PROBLEM, MR_AHA_OP, AHA_CUFINUFFT, MR_NUFFT_PLAN.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    all_methods = {'toeplitz', 'nufft_c', 'nufft_native'};
    ip = inputParser;
    ip.addParameter('problem', '2d');
    ip.addParameter('n', []);
    ip.addParameter('L', []);
    ip.addParameter('coils', []);
    ip.addParameter('spokes', []);
    ip.addParameter('samples', []);
    ip.addParameter('data', '');
    ip.addParameter('traj_type', '');
    ip.addParameter('seed', 0);
    ip.addParameter('methods', {});
    ip.addParameter('gpus', 1);
    ip.addParameter('mask', 'off');
    ip.addParameter('toep_dist', 'auto');
    ip.addParameter('upsampfac', []);
    ip.addParameter('gpu_method', []);
    ip.addParameter('eps', 1e-6);
    ip.addParameter('reps', 10);
    ip.addParameter('warmup', 2);
    ip.addParameter('setup_reps', 1);
    ip.addParameter('cg', 0);
    ip.addParameter('check', false);
    ip.addParameter('out', '');
    ip.addParameter('mr_aha_root', '');
    ip.addParameter('mex_dir', '');
    ip.addParameter('cufinufft_dir', '');
    ip.parse(varargin{:});
    o = ip.Results;
    o.mask = validatestring(o.mask, {'off', 'on', 'both'});
    o.toep_dist = validatestring(o.toep_dist, ...
                                 {'auto', 'split', 'replicate', 'broadcast'});
    if o.reps < 1 || o.warmup < 1 || o.setup_reps < 1
        error('bench_aha:args', ...
              '''reps'', ''warmup'' and ''setup_reps'' must be at least 1.');
    end

    here = fileparts(mfilename('fullpath'));
    repo = fileparts(here);
    setup_paths(o, repo);
    backend = mr_backend();
    is_cuda = strcmp(backend, 'cuda');

    % ---- methods: the requested ones that can run here ------------------
    methods = o.methods;
    if isempty(methods), methods = all_methods; end
    if ischar(methods) || isstring(methods), methods = cellstr(methods); end
    methods = cellfun(@(m) validatestring(m, all_methods), methods, ...
                      'UniformOutput', false);
    methods = all_methods(ismember(all_methods, methods));    % report order
    if ismember('nufft_native', methods) && is_cuda && ...
            exist('cufinufft_plan', 'file') ~= 2
        fprintf(['skipping nufft_native: cufinufft''s MATLAB interface ' ...
                 '(cufinufft_plan) is not on the path\n']);
        methods = setdiff(methods, {'nufft_native'}, 'stable');
    end

    % ---- GPU counts -----------------------------------------------------
    if is_cuda
        g = gpuDevice();
        n_vis = gpuDeviceCount();
        gpu_names = {g.Name};
    else
        g = [];
        n_vis = 1;
        gpu_names = {'Apple GPU'};
    end
    if ischar(o.gpus) || isstring(o.gpus)
        if ~strcmpi(o.gpus, 'all')
            error('bench_aha:args', '''gpus'' must be a vector or ''all''.');
        end
        gpus = n_vis;
    else
        gpus = unique(double(o.gpus(:)).', 'stable');
    end
    for n = gpus(gpus > n_vis)
        fprintf('skipping %d GPUs: only %d visible\n', n, n_vis);
    end
    gpus = gpus(gpus >= 1 & gpus <= n_vis);
    if isempty(gpus), error('bench_aha:args', 'No usable GPU count.'); end
    switch o.mask
        case 'off',  masks = false;
        case 'on',   masks = true;
        otherwise,   masks = [false true];
    end

    % ---- problems -------------------------------------------------------
    if ~isempty(o.data)
        [~, base] = fileparts(char(o.data));
        todo = {base, {char(o.data), 'traj_type', o.traj_type, 'seed', o.seed}};
    else
        names = o.problem;
        if ischar(names) || isstring(names), names = cellstr(names); end
        todo = cell(0, 2);
        for i = 1:numel(names)
            todo(end + 1, :) = {names{i}, {names{i}, 'n', o.n, 'L', o.L, ...
                'coils', o.coils, 'spokes', o.spokes, ...
                'samples', o.samples, 'seed', o.seed}}; %#ok<AGROW>
        end
    end

    [~, host] = system('hostname -s');
    host = strtrim(host);
    fprintf('host %s, backend %s, %d GPU(s) visible: %s\n', host, backend, ...
            n_vis, gpu_names{1});
    for v = {'MR_AHA_TOEP_SPLIT', 'MR_AHA_MGPU_SPLIT', 'MR_AHA_MGPU_MIN_WORK'}
        if ~isempty(getenv(v{1}))
            fprintf(['note: %s=%s is set and overrides the library''s ' ...
                     'defaults\n'], v{1}, getenv(v{1}));
        end
    end

    report = struct('host', host, 'backend', backend, 'gpu', gpu_names{1}, ...
                    'gpus_visible', n_vis, 'matlab', version, ...
                    'time', datestr(now, 'yyyy-mm-dd HH:MM:SS'), ...
                    'options', rmfield(o, {'mr_aha_root', 'mex_dir', ...
                                           'cufinufft_dir'}), ...
                    'problems', {{}}); %#ok<DATST,TNOW1>
    out = char(o.out);
    if isempty(out)
        out = fullfile(repo, 'results', host, ['bench_aha_matlab_' ...
                       datestr(now, 'yyyymmdd-HHMMSS') '.json']); %#ok<DATST,TNOW1>
    end
    if ~exist(fileparts(out), 'dir') && ~isempty(fileparts(out))
        mkdir(fileparts(out));
    end

    warm_up(methods, o, g);

    for i = 1:size(todo, 1)
        d = bench_problem(todo{i, 2}{:});
        report.problems{end + 1} = run_problem(todo{i, 1}, d, methods, ...
                                               gpus, masks, o, g, out, report);
        clear d
    end
    save_json(report, out);
    fprintf('\nresults written to %s\n', out);
    if nargout == 0, clear report; end
end

% ======================================================================
% Paths
% ======================================================================

function setup_paths(o, repo)
%SETUP_PATHS  Put the MR_AHA MATLAB interface and its MEX files on the path.
%   The checkout is o.mr_aha_root, else $MR_AHA_ROOT, else the submodule
%   third_party/MR_AHA of this repository. Errors with build directions if
%   the MEX files are not there.
    root = char(o.mr_aha_root);
    if isempty(root), root = getenv('MR_AHA_ROOT'); end
    if isempty(root), root = fullfile(repo, 'third_party', 'MR_AHA'); end
    if ~isfolder(fullfile(root, 'matlab'))
        error('bench_aha:paths', ['No MR_AHA checkout in %s.\nRun ' ...
              '"git submodule update --init --recursive" in this ' ...
              'repository, or set MR_AHA_ROOT.'], root);
    end
    mex_dir = char(o.mex_dir);
    if isempty(mex_dir), mex_dir = getenv('MR_AHA_MEX_DIR'); end
    if isempty(mex_dir), mex_dir = fullfile(root, 'build', 'matlab'); end
    addpath(fullfile(root, 'matlab'));
    if isfolder(mex_dir), addpath(mex_dir); end
    if exist('mr_aha_op_mex', 'file') ~= 3
        error('bench_aha:paths', ['The MR_AHA MEX files are not in %s.\n' ...
              'Build them (see the README of this repository):\n' ...
              '  cmake -S %s -B %s -DMR_AHA_BUILD_MATLAB=ON ' ...
              '-DMR_AHA_BUILD_CUFINUFFT_MATLAB=ON\n' ...
              '  cmake --build %s -j\n' ...
              'or pass ''mex_dir''.'], mex_dir, root, ...
              fullfile(root, 'build'), fullfile(root, 'build'));
    end
    cdir = char(o.cufinufft_dir);
    if isempty(cdir)
        cdir = fullfile(fileparts(mex_dir), 'finufft', 'matlab');
    end
    if isfile(fullfile(cdir, 'cufinufft_plan.m')), addpath(cdir); end
end

% ======================================================================
% Running
% ======================================================================

function pr = run_problem(name, d, methods, gpus, masks, o, g, out, report)
%RUN_PROBLEM  Every selected configuration on one problem; prints its table.
%   Returns the struct stored in report.problems: name, desc, sizes,
%   kernel_gb (size of the unmasked Toeplitz kernel) and rows (cell array of
%   row structs). The report is saved after every row, so that a crash of a
%   later configuration keeps the earlier results.
    cfg = mr_default_config(d.traj_type, d.n_dim, d.L);
    kernel_gb = d.L * (d.L + 1) / 2 * prod(cfg.phi2_dim) * 8 / 1e9;
    fprintf('\n== %s: %s\n   unmasked Toeplitz kernel %.2f GB\n', name, ...
            d.desc, kernel_gb);
    print_header(o);
    d = to_device(d, g);
    pr = struct('name', name, 'desc', d.desc, 'traj_type', d.traj_type, ...
                'n_dim', d.n_dim, 'L', d.L, 'n_coils', d.n_coils, ...
                'kernel_gb', kernel_gb, 'rows', {{}});
    ref = [];
    for m = 1:numel(methods)
        method = methods{m};
        variants = false;
        if strcmp(method, 'toeplitz'), variants = masks; end
        counts = gpus;
        if strcmp(method, 'nufft_native'), counts = 1; end
        for masked = variants
            for n_gpus = counts
                row = struct('method', method, 'masked', masked, ...
                             'gpus', n_gpus);
                try
                    [row, y] = run_one(method, d, n_gpus, masked, o, g, ref);
                    if o.check && isempty(ref), ref = y; end
                    clear y
                catch err
                    row.error = strtok(err.message, newline);
                end
                pr.rows{end + 1} = row;
                pr.rows = add_speedups(pr.rows);
                print_row(pr.rows{end}, o);
                partial = report;
                partial.problems{end + 1} = pr;
                save_json(partial, out);
            end
        end
    end
end

function [row, y] = run_one(method, d, n_gpus, masked, o, g, ref)
%RUN_ONE  Create, time and release one operator.
%   Returns the result row and, with 'check', the operator's A'A x.
    row = struct('method', method, 'masked', masked, 'gpus', n_gpus);
    mem0 = avail(g);

    % Set-up: the first creation is "cold" within this configuration.
    ts = zeros(1, o.setup_reps);
    cleanup = [];
    for i = 1:o.setup_reps
        if ~isempty(cleanup), cleanup(); end
        sync(g);
        t = tic;
        [f, cleanup, info] = make_op(method, d, n_gpus, masked, o, g);
        sync(g);
        ts(i) = toc(t);
    end
    finish = onCleanup(cleanup);
    for fn = fieldnames(info).', row.(fn{1}) = info.(fn{1}); end
    row.setup_cold_s = ts(1);
    row.setup_s = ts(1);
    if o.setup_reps > 1, row.setup_s = median(ts(2:end)); end

    % One application.
    for i = 1:o.warmup
        y = f(d.x);
    end
    sync(g);
    if ~isempty(mem0), row.mem_gb = (mem0 - avail(g)) / 1e9; end
    ts = zeros(1, o.reps);
    for i = 1:o.reps
        sync(g);
        t = tic;
        y = f(d.x);
        sync(g);
        ts(i) = toc(t);
    end
    row.aha_ms = median(ts) * 1e3;
    row.aha_min_ms = min(ts) * 1e3;
    row.aha_mean_ms = mean(ts) * 1e3;
    row.reps = o.reps;

    if o.check && ~isempty(ref)
        row.rel_diff = double(gather(norm(y(:) - ref(:)) / norm(ref(:))));
    end
    if ~o.check, y = []; end
    if o.cg > 0
        row.cg_s = cg_time(f, d.x, o.cg, g);
        row.cg_iters = o.cg;
    end
end

function [f, cleanup, info] = make_op(method, d, n_gpus, masked, o, g)
%MAKE_OP  Create one operator.
%   f        handle x -> A'A x
%   cleanup  handle releasing the operator
%   info     struct of facts about it (gpus_used, kernel_s, mask_s,
%            toep_source, mask_keep)
    cfg = mr_default_config(d.traj_type, d.n_dim, d.L, 'eps', o.eps, ...
                            'is_masked', masked && strcmp(method, 'toeplitz'));
    multi = {};
    if n_gpus > 1
        % The current gpuDevice is the primary GPU; add the others in order.
        others = setdiff(1:gpuDeviceCount(), g.Index, 'stable');
        multi = {'devices', [g.Index, others(1:n_gpus - 1)]};
    end
    nopts = struct();
    if ~isempty(o.upsampfac),  nopts.upsampfac  = double(o.upsampfac);  end
    if ~isempty(o.gpu_method), nopts.gpu_method = double(o.gpu_method); end
    info = struct('gpus_used', 1);

    switch method
        case 'toeplitz'
            args = [{'method', 'toeplitz', 'toep_dist', o.toep_dist}, multi];
            if masked
                t = tic;
                mask = calc_mask(d.traj, cfg);
                sync(g);
                info.mask_s = toc(t);
                info.mask_keep = numel(mask) / prod(cfg.phi2_dim);
                args = [args, {'mask', mask}];
            end
            op = mr_aha_op.create_op(d.SEs, d.phi2, d.traj, cfg, args{:});
            s = op.info();
            info.gpus_used = double(s.n_devices);
            info.kernel_s = double(s.toep_ms) / 1e3;
            info.toep_source = s.toep_source;
            f = @(x) op.apply(x);
            cleanup = @() delete(op);
        case 'nufft_c'
            args = [{'method', 'nufft'}, multi];
            if ~isempty(fieldnames(nopts))
                args = [args, {'nufft_opts', nopts}];
            end
            op = mr_aha_op.create_op(d.SEs, d.phi2, d.traj, cfg, args{:});
            s = op.info();
            info.gpus_used = double(s.n_devices);
            f = @(x) op.apply(x);
            cleanup = @() delete(op);
        otherwise                                   % nufft_native
            nv = [fieldnames(nopts), struct2cell(nopts)].';
            plan = mr_nufft_plan(d.traj, cfg, nv{:});
            f = @(x) AhA_cufinufft(x, d.SEs, d.phi2, d.traj, cfg, plan);
            cleanup = @() free_plan(plan);
    end
end

function free_plan(plan)
%FREE_PLAN  Release the cufinufft plans of MR_NUFFT_PLAN (CUDA backend).
    for f = {'t1', 't2'}
        if isfield(plan, f{1}) && isobject(plan.(f{1})) && ...
                ismethod(plan.(f{1}), 'delete') && isvalid(plan.(f{1}))
            delete(plan.(f{1}));
        end
    end
end

function warm_up(methods, o, g)
%WARM_UP  Pay the one-off start-up costs before anything is timed.
%   Creates and applies every selected method once on a tiny 2D problem on
%   the primary GPU, which loads the MEX files, cuFFT and cufinufft. Without
%   it the first row of the first table would carry these costs.
    d = to_device(bench_problem('2d', 'n', [32 32], 'L', 2, 'coils', 2, ...
                                'spokes', 16, 'samples', 32), g);
    for m = 1:numel(methods)
        try
            [f, cleanup] = make_op(methods{m}, d, 1, false, o, g);
            f(d.x);
            if o.cg > 0, cg_time(f, d.x, 1, g); end
            sync(g);
            cleanup();
        catch
            % The real run reports failures.
        end
    end
end

function t = cg_time(f, b, n_iter, g)
%CG_TIME  Wall-clock time of n_iter plain conjugate-gradient iterations.
%   Solves A'A u = b from u = 0 without a stopping test, so that every
%   operator does the same amount of work.
    sync(g);
    t0 = tic;
    u  = zeros(size(b), 'like', b);
    r  = b;
    p  = r;
    rs = real(r(:)' * r(:));
    for it = 1:n_iter
        Ap = f(p);
        alpha = rs / real(p(:)' * Ap(:));
        u = u + alpha * p;
        r = r - alpha * Ap;
        rs_new = real(r(:)' * r(:));
        p = r + (rs_new / rs) * p;
        rs = rs_new;
    end
    sync(g);
    t = toc(t0);
end

% ======================================================================
% Devices
% ======================================================================

function d = to_device(d, g)
%TO_DEVICE  Move the arrays of a problem to the primary GPU (CUDA backend).
    if isempty(g), return, end
    for f = {'SEs', 'phi2', 'traj', 'x'}
        d.(f{1}) = gpuArray(d.(f{1}));
    end
end

function sync(g)
%SYNC  Wait until the queued GPU work is done (no-op on the Metal backend).
%   The results of a multi-GPU operator are summed on the primary GPU, so
%   waiting for the primary GPU waits for all of them.
    if ~isempty(g), wait(g); end
end

function b = avail(g)
%AVAIL  Free memory of the primary GPU in bytes ([] on the Metal backend).
    b = [];
    if ~isempty(g), b = double(g.AvailableMemory); end
end

% ======================================================================
% Reporting
% ======================================================================

function rows = add_speedups(rows)
%ADD_SPEEDUPS  Add speedup (one-GPU time / this time) to multi-GPU rows.
    for i = 1:numel(rows)
        r = rows{i};
        if ~isfield(r, 'aha_ms'), continue, end
        for j = 1:numel(rows)
            b = rows{j};
            if isfield(b, 'aha_ms') && b.gpus == 1 && ...
                    strcmp(b.method, r.method) && b.masked == r.masked
                rows{i}.speedup = b.aha_ms / r.aha_ms;
            end
        end
    end
end

function print_header(o)
%PRINT_HEADER  Column headings matching PRINT_ROW.
    s = sprintf('  %-16s %4s %9s %10s %8s %8s', 'method', 'GPUs', ...
                'set-up s', 'A''A ms', 'speed-up', 'mem GB');
    if o.cg > 0, s = [s sprintf(' %8s', 'CG s')]; end
    if o.check,  s = [s sprintf(' %9s', 'diff')]; end
    fprintf('%s\n', s);
end

function print_row(r, o)
%PRINT_ROW  One line of the table.
    name = r.method;
    if r.masked, name = [name ' masked']; end
    if isfield(r, 'error')
        fprintf('  %-16s %4d   failed: %s\n', name, r.gpus, r.error);
        return
    end
    gpus = sprintf('%d', r.gpus);
    if r.gpus_used ~= r.gpus, gpus = sprintf('%d>%d', r.gpus, r.gpus_used); end
    s = sprintf('  %-16s %4s %9.3f %10.2f', name, gpus, r.setup_s, r.aha_ms);
    if isfield(r, 'speedup') && r.gpus > 1
        s = [s sprintf(' %8.2f', r.speedup)];
    else
        s = [s sprintf(' %8s', '')];
    end
    if isfield(r, 'mem_gb')
        s = [s sprintf(' %8.2f', r.mem_gb)];
    else
        s = [s sprintf(' %8s', 'n/a')];
    end
    if o.cg > 0, s = [s sprintf(' %8.3f', r.cg_s)]; end
    if o.check
        if isfield(r, 'rel_diff')
            s = [s sprintf(' %9.1e', r.rel_diff)];
        else
            s = [s sprintf(' %9s', 'ref')];
        end
    end
    fprintf('%s\n', s);
end

function save_json(report, out)
%SAVE_JSON  Write the report as JSON.
    fid = fopen(out, 'w');
    if fid < 0, error('bench_aha:out', 'Cannot write %s.', out); end
    closer = onCleanup(@() fclose(fid));
    try
        txt = jsonencode(report, 'PrettyPrint', true);
    catch
        txt = jsonencode(report);                 % before R2021a
    end
    fwrite(fid, txt);
end
