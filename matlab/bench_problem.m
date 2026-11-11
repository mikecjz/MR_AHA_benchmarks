function d = bench_problem(name, varargin)
%BENCH_PROBLEM  Test problem for BENCH_AHA: preset, custom sizes or a file.
%
%   d = BENCH_PROBLEM(name)
%   d = BENCH_PROBLEM(name, 'Name', Value, ...)
%   d = BENCH_PROBLEM(file)
%   d = BENCH_PROBLEM(file, 'traj_type', tt)
%   BENCH_PROBLEM()                          % list the presets
%
%   Returns the arrays the A'A operator of MR_AHA is built from and applied
%   to, as host arrays in the column-major layout of the MR_AHA documentation
%   and of its MATLAB interface.
%
%   SYNTHETIC PROBLEMS
%     name is one of the presets
%       '2d'     2D golden-angle radial, 160 x 160, L = 6, 16 coils,
%                377 spokes x 160 samples
%       '2d72'   2D golden-angle radial, 320 x 320, L = 72, 12 coils,
%                754 x 320 samples (8.6 GB Toeplitz kernel)
%       'stack'  stack-of-stars, 320 x 320 x 56, L = 12, 12 coils,
%                280 x 320 samples x 56 kz (14 GB)
%       '3d'     3D golden-means kooshball, 160^3, L = 12, 12 coils,
%                15680 x 160 samples (20 GB)
%     and the name-value pairs change its sizes:
%       'n'        matrix size [Nx Ny] or [Nx Ny Nz]
%       'L'        number of ranks
%       'coils'    number of coils
%       'spokes'   number of readouts
%       'samples'  samples per readout
%       'seed'     seed of the random inputs (default 0)
%     The run time of the operator depends on the sizes and on the geometry
%     of the trajectory, not on the values of the arrays, so the sensitivity
%     maps are smooth synthetic profiles and the Gram blocks and the
%     coefficients are random.
%
%   PROBLEMS FROM A FILE
%     file is a .mat file holding
%       SEs        [Nx, Ny, Ncoils] or [Nx, Ny, Nz, Ncoils]
%       phi2       [Ntrajs, L, L] or, for a stack, [Ntrajs, Nkz, L, L]
%                  (MR-Multitasking Gram blocks, kz in centred order)
%       traj       [Ntrajs, Nkx, ndim] in radians
%       traj_type  '2D', '3D' or 'stack' (or pass 'traj_type')
%       x          optional coefficients [Nx, Ny, (Nz,) L] to apply the
%                  operator to; random when absent.
%     The Python benchmark writes such a file with --save-data.
%
%   OUTPUT
%     d   struct with fields
%           traj_type  '2D', '3D' or 'stack'
%           n_dim      [Nx Ny] or [Nx Ny Nz]
%           L, n_coils
%           SEs, phi2, traj, x   complex single (traj: real single)
%           desc       one-line description
%
%   See also BENCH_AHA.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    presets = struct( ...
        'name',    {'2d', '2d72', 'stack', '3d'}, ...
        'traj_type', {'2D', '2D', 'stack', '3D'}, ...
        'n',       {[160 160], [320 320], [320 320 56], [160 160 160]}, ...
        'L',       {6, 72, 12, 12}, ...
        'coils',   {16, 12, 12, 12}, ...
        'spokes',  {377, 754, 280, 15680}, ...
        'samples', {160, 320, 320, 160}, ...
        'desc',    {'2D golden-angle radial', ...
                    '2D golden-angle radial, high rank', ...
                    'stack-of-stars', '3D golden-means kooshball'});
    if nargin == 0
        for p = presets
            fprintf('  %-6s %s %s, L=%d, %d coils, %dx%d samples\n', p.name, ...
                    p.desc, size_str(p.n), p.L, p.coils, p.spokes, p.samples);
        end
        return
    end

    ip = inputParser;
    ip.addParameter('n', []);
    ip.addParameter('L', []);
    ip.addParameter('coils', []);
    ip.addParameter('spokes', []);
    ip.addParameter('samples', []);
    ip.addParameter('seed', 0);
    ip.addParameter('traj_type', '');
    ip.parse(varargin{:});
    o = ip.Results;

    name = char(name);
    k = find(strcmpi(name, {presets.name}), 1);
    if isempty(k)
        if ~isfile(name)
            error('bench_problem:name', ...
                  '''%s'' is neither a preset (%s) nor a file.', name, ...
                  strjoin({presets.name}, ', '));
        end
        d = from_file(name, o);
        return
    end

    % ---- sizes: the preset, with the overrides ---------------------------
    p = presets(k);
    if ~isempty(o.n)
        if numel(o.n) ~= numel(p.n)
            error('bench_problem:n', ...
                  '''n'' needs %d values for a %s problem, got %d.', ...
                  numel(p.n), p.traj_type, numel(o.n));
        end
        p.n = double(o.n(:)).';
    end
    for f = {'L', 'coils', 'spokes', 'samples'}
        if ~isempty(o.(f{1})), p.(f{1}) = double(o.(f{1})); end
    end

    % ---- arrays ---------------------------------------------------------
    rng(o.seed);
    is_stack = strcmpi(p.traj_type, 'stack');
    ndim = 2 + strcmpi(p.traj_type, '3D');
    d = struct('traj_type', p.traj_type, 'n_dim', p.n, 'L', p.L, ...
               'n_coils', p.coils);
    d.traj = radial_traj(ndim, p.spokes, p.samples);
    if is_stack
        d.phi2 = reshape(gram_blocks(p.L, p.spokes * p.n(3)), ...
                         [p.spokes, p.n(3), p.L, p.L]);
    else
        d.phi2 = gram_blocks(p.L, p.spokes);
    end
    d.SEs = coil_maps(p.coils, p.n);
    d.x = complex(randn([p.n p.L], 'single'), randn([p.n p.L], 'single'));
    d.desc = sprintf('%s %s, L=%d, %d coils, %dx%d samples', p.desc, ...
                     size_str(p.n), p.L, p.coils, p.spokes, p.samples);
    if is_stack
        d.desc = sprintf('%s x %d kz', d.desc, p.n(3));
    end
end

% ======================================================================
% Local functions
% ======================================================================

function s = size_str(n)
%SIZE_STR  '320x320x56' from [320 320 56].
    s = strjoin(arrayfun(@num2str, n, 'UniformOutput', false), 'x');
end

function traj = radial_traj(ndim, n_traj, n_kx)
%RADIAL_TRAJ  Golden-angle (2D) or golden-means (3D) radial trajectory.
%   Every readout is a full diameter through the centre of k-space. Returns
%   real single [Ntrajs, Nkx, ndim] in radians, [-pi, pi).
    k = pi * ((0:n_kx - 1) - floor(n_kx / 2)) / floor(n_kx / 2);
    m = (0:n_traj - 1).';
    if ndim == 2
        a = m * pi * (sqrt(5) - 1) / 2;
        dir = [cos(a), sin(a)];
    else                                   % golden means (Chan et al.)
        z  = mod(m * 0.4656, 1) * 2 - 1;
        az = 2 * pi * mod(m * 0.6823, 1);
        r  = sqrt(1 - z .^ 2);
        dir = [r .* cos(az), r .* sin(az), z];
    end
    traj = single(reshape(dir, [n_traj, 1, ndim]) .* reshape(k, [1, n_kx, 1]));
end

function phi2 = gram_blocks(L, n)
%GRAM_BLOCKS  Random Hermitian positive semi-definite Gram blocks [n, L, L].
%   Block t is Phi * Phi' for a random rank-by-time basis Phi with two time
%   points (MR-Multitasking convention): phi2(t, i, j) = sum_k Phi(i, k) *
%   conj(Phi(j, k)).
    A = complex(randn(n, L, 1, 2, 'single'), randn(n, L, 1, 2, 'single'));
    phi2 = sum(A .* conj(permute(A, [1 3 2 4])), 4);
end

function SEs = coil_maps(n_coils, n)
%COIL_MAPS  Smooth synthetic sensitivity maps [Nx, Ny, (Nz,) Ncoils].
%   Coil c has a Gaussian magnitude profile centred on a ring around the
%   object (alternating between two rings along z in 3D) and a linear phase
%   whose direction depends on the coil. Not realistic, only smooth, complex
%   and different from each other.
    x = reshape(linspace(-1, 1, n(1)), [], 1);
    y = reshape(linspace(-1, 1, n(2)), 1, []);
    is3 = numel(n) == 3;
    if is3, z = reshape(linspace(-1, 1, n(3)), 1, 1, []); end
    SEs = zeros([n n_coils], 'like', complex(single(0)));
    for c = 1:n_coils
        ang = 2 * pi * (c - 1) / n_coils;
        d2 = (x - 1.2 * cos(ang)) .^ 2 + (y - 1.2 * sin(ang)) .^ 2;
        ph = 0.5 * pi * (cos(ang) * x + sin(ang) * y);
        if is3
            d2 = d2 + (z - 0.5 * (2 * mod(c - 1, 2) - 1)) .^ 2;
            SEs(:, :, :, c) = single(exp(-d2 / 2) .* exp(1i * ph));
        else
            SEs(:, :, c) = single(exp(-d2 / 2) .* exp(1i * ph));
        end
    end
end

function d = from_file(file, o)
%FROM_FILE  Load and validate a problem stored in a .mat file.
    s = load(file);
    for f = {'SEs', 'phi2', 'traj'}
        if ~isfield(s, f{1})
            error('bench_problem:file', '%s: variable ''%s'' is missing.', ...
                  file, f{1});
        end
    end
    tt = o.traj_type;
    if isempty(tt)
        if ~isfield(s, 'traj_type')
            error('bench_problem:file', ...
                  '%s has no traj_type; pass ''traj_type''.', file);
        end
        tt = s.traj_type;
    end
    tt = strtrim(char(tt));
    names = {'2D', '3D', 'stack'};
    k = find(strcmpi(tt, names), 1);
    if isempty(k)
        error('bench_problem:file', ...
              'traj_type must be 2D, 3D or stack, got ''%s''.', tt);
    end
    tt = names{k};
    n_sp = 2 + ~strcmp(tt, '2D');
    sz = size(s.SEs);
    sz(end + 1:n_sp + 1) = 1;              % a single coil drops the last dim
    if numel(sz) ~= n_sp + 1
        error('bench_problem:file', ...
              'SEs must have %d dimensions for a %s problem.', n_sp + 1, tt);
    end
    L = size(s.phi2, ndims(s.phi2));
    d = struct('traj_type', tt, 'n_dim', sz(1:n_sp), 'L', L, ...
               'n_coils', sz(end));
    d.traj = single(real(s.traj));
    d.phi2 = complex(single(s.phi2));
    d.SEs  = complex(single(s.SEs));
    if isfield(s, 'x')
        d.x = complex(single(s.x));
    else
        rng(o.seed);
        d.x = complex(randn([d.n_dim L], 'single'), ...
                      randn([d.n_dim L], 'single'));
    end
    [~, base, ext] = fileparts(file);
    d.desc = sprintf('%s%s: %s %s, L=%d, %d coils, %dx%d samples', base, ...
                     ext, tt, size_str(d.n_dim), L, d.n_coils, ...
                     size(d.traj, 1), size(d.traj, 2));
end
