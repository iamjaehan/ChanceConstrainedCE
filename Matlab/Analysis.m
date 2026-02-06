% ============================================================
% VQ Monte Carlo results plotting (BOXCHART version)
% CSV schema (confirmed):
%   mc, solver, status, ..., coord_sigma, ..., obj, ..., n_dev, dev_rate,
%   solver_time_sec, wall_ms, num_pne
% Case metadata is encoded in filename: *_6a_10s_90c.csv (a,sigma,alpha)
%
% Figures:
% 1) Scalability (wall_ms) : (6/0/90 vs 8/0/90) x {CE_FULL, RRCE_PNE}
% 2) Efficiency-to-scale (obj): (6/0/90 vs 8/0/90) x {all except CE_NAIVE}
% 3) Efficiency-to-error (obj): (6/0/90, 6/10/90, 6/30/90) x {FCFS, CE_NAIVE, RRCE_PNE}
% 4) Alpha sweep: (6/10/70, 6/10/90, 6/10/99) x {CE_NAIVE, RRCE_PNE}
%    -> stacked boxcharts for obj and dev_rate
% ============================================================

clear; close all; clc;
% ============================================================
% Global plotting defaults (LaTeX + font size)
% ============================================================

set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

set(groot, 'defaultAxesFontSize', 18);
set(groot, 'defaultTextFontSize', 18);
set(groot, 'defaultLegendFontSize', 18);

% Optional but recommended for papers
set(groot, 'defaultAxesLineWidth', 1.2);
set(groot, 'defaultLineLineWidth', 2.0);
set(groot, 'defaultFigureColor', 'w');

%% ---------------------------
% USER CONFIG
% ----------------------------
dataDir = ".."; % folder containing all CSVs
fileGlob = fullfile(dataDir, "*.csv"); % or "mc_epoch_results_*.csv"
onlyOK  = true;

% If you want to filter "invalid RRCE runs" (optional; default off)
% Example: require num_pne > 0 for RRCE_PNE
filterInvalidRRCE = true;

%% ---------------------------
% Load all CSV files
% ----------------------------
files = dir(fileGlob);
assert(~isempty(files), "No CSV files found: %s", fileGlob);

allRows = table();

for k = 1:numel(files)
    fname = files(k).name;
    fpath = fullfile(files(k).folder, fname);

    meta = parse_case_from_filename(fname);
    if ~meta.ok
        fprintf("Skipping (unmatched filename): %s\n", fname);
        continue;
    end

    T = readtable(fpath, 'Delimiter', ',', 'PreserveVariableNames', true);
    T.Properties.VariableNames = lower(string(T.Properties.VariableNames));

    required = ["solver","status","obj","wall_ms","dev_rate","coord_sigma"];
    for c = required
        assert(ismember(c, string(T.Properties.VariableNames)), ...
            "Missing column '%s' in %s", c, fname);
    end

    if onlyOK
        T = T(string(T.status)=="OK", :);
    end

    % Attach filename meta
    T.a     = repmat(meta.a, height(T), 1);
    T.alpha = repmat(meta.alpha, height(T), 1);

    % Standardize algorithm name column
    T.algorithm = string(T.solver);

    % Keep tidy subset
    keep = ["mc","algorithm","status","a","coord_sigma","alpha", ...
            "obj","wall_ms","dev_rate","n_dev","solver_time_sec","num_pne"];
    keep = keep(ismember(keep, string(T.Properties.VariableNames)));
    T = T(:, keep);

    allRows = [allRows; T]; %#ok<AGROW>
end

assert(height(allRows) > 0, "No usable rows loaded. Check dataDir + filename pattern.");

if filterInvalidRRCE && ismember("num_pne", string(allRows.Properties.VariableNames))
    rrce = allRows.algorithm=="RRCE_PNE";
    allRows = allRows(~rrce | (allRows.num_pne > 0), :);
end

%% ---------------------------
% Algorithm labels (your actual solver strings)
% ----------------------------
ALG.CE_FULL   = "CE_FULL";
ALG.RRCE_PNE  = "RRCE_PNE";
ALG.CE_NAIVE  = "CE_NAIVE";
ALG.FCFS      = "AGG_ORACLE_FCFS";
ALG.CENTRAL   = "GREEDY_CENTRALIZED";

% ------------------------------------------------------------
% Additive "normalized" cost relative to GREEDY_CENTRALIZED
% obj_norm := obj - obj_greedy
% group key: (a, coord_sigma, alpha, mc)
% ------------------------------------------------------------

keyVars = ["a","coord_sigma","alpha","mc"];

[~,~,gid] = unique(allRows(:, keyVars), 'rows');
isGreedy = (allRows.algorithm == ALG.CENTRAL);

obj_greedy_by_group = splitapply( ...
    @(obj, isG) local_greedy_median(obj, isG), ...
    allRows.obj, isGreedy, gid);

allRows.obj_greedy = obj_greedy_by_group(gid);

% ★ 핵심: obj_norm 이름은 유지, 의미는 additive
allRows.obj_norm = allRows.obj - allRows.obj_greedy;

% Guardrails
bad = isnan(allRows.obj_greedy);
allRows.obj_norm(bad) = NaN;

% Sanity check
gmask = isGreedy & ~isnan(allRows.obj_norm);
if any(gmask)
    fprintf("Sanity: max |obj_norm(greedy)| = %.3e\n", ...
        max(abs(allRows.obj_norm(gmask))));
end


% -------- helper ----------
function m = local_greedy_median(obj, isG)
    x = obj(isG);
    if isempty(x)
        m = NaN;
    else
        m = median(x);
    end
end


%% ============================================================
% 1) Scalability: wall_ms for 6/0/90 vs 8/0/90
% alg: CE_FULL, RRCE_PNE
% x: a, y: wall_ms
% ============================================================
plot_box_byA( ...
    allRows, struct("a",[4 5 6 7 8], "sigma",0, "alpha",90), ...
    [ALG.CE_FULL, ALG.RRCE_PNE], ...
    "wall_ms", "wall time (ms) [log]", ...
    "Scalability: wall time vs aircraft count");
exportgraphics(gcf, "Scalability.pdf","Resolution",300);

%% ============================================================
% 2) Efficiency-to-scale: obj for 6/0/90 vs 8/0/90, exclude CE_NAIVE
% alg: all except CE_NAIVE (within that slice)
% ============================================================
slice2 = allRows(allRows.coord_sigma==0 & allRows.alpha==90 & ismember(allRows.a,[4 5 6 7 8]), :);
algs2 = unique(slice2.algorithm, 'stable');
algs2(algs2 == ALG.CE_NAIVE) = [];

plot_box_byA2( ...
    allRows, struct("a",[4 5 6 7 8], "sigma",0, "alpha",90), ...
    algs2, ...
    "obj", "cost", ...
    "Efficiency vs scale");
exportgraphics(gcf, "eff_scale.pdf","Resolution",300);

%% ============================================================
% 3) Efficiency-to-error: obj for 6/0/90, 6/10/90, 6/30/90
% x: coord_sigma, y: obj, alg: FCFS, CE_NAIVE, RRCE_PNE
% ============================================================
plot_box_bySigma( ...
    allRows, struct("a",6, "sigma",[0 5 15 30], "alpha",90), ...
    [ALG.CENTRAL, ALG.FCFS, ALG.CE_NAIVE, ALG.RRCE_PNE], ...
    "obj", "cost", ...
    "Efficiency vs uncertainty");
exportgraphics(gcf, "eff_uncertainty.pdf","Resolution",300);

%%
% plot_errorbar_bySigma( ...
%     allRows, struct("a",6, "sigma",[0 5 15 30], "alpha",90), ...
%     [ALG.CE_FULL, ALG.CE_NAIVE, ALG.RRCE_PNE], ...
%     "obj", "cost", ...
%     "Efficiency vs uncertainty");
% exportgraphics(gcf, "eff_uncertainty.pdf","Resolution",300);

plot_errorbar_devfreq_bySigma( ...
    allRows, struct("a",6,"sigma",[0 5 15 30],"alpha",90), ...
    [ALG.CE_FULL, ALG.CE_NAIVE, ALG.RRCE_PNE], ...
    "Deviation frequency vs uncertainty");
exportgraphics(gcf, "devfreq_uncertainty.pdf","Resolution",300);

plot_conditional_cost_sigma_box( ...
    allRows, struct("a",6,"sigma",[0 5 15 30],"alpha",90), ...
    [ALG.CE_NAIVE, ALG.RRCE_PNE], ...
    "obj_norm", ...
    "normalized cost $J/J_{\mathrm{greedy}}$", ...
    "Sigma sweep: conditional cost diagnostics");

%% ============================================================
% 4) Alpha sweep: (6/10/70, 6/10/90, 6/10/99)
% alg: CE_NAIVE, RRCE_PNE
% two metrics: obj and dev_rate (stacked)
% ============================================================
% plot_two_metric_alpha_box( ...
%     allRows, struct("a",6, "sigma",15, "alpha",[50 75 90 95 99]), ...
%     [ALG.CE_NAIVE, ALG.RRCE_PNE], ...
%     "obj", "cost", ...
%     "dev_rate", "deviation rate", ...
%     "Alpha sweep");
% exportgraphics(gcf, "alpha.pdf","Resolution",300);

plot_two_metric_alpha_box( ...
    allRows, struct("a",6,"sigma",15,"alpha",[50 75 90 95 99]), ...
    [ALG.CE_NAIVE, ALG.RRCE_PNE], ...
    "obj_norm", "normalized cost $J/J_{\mathrm{greedy}}$", ...
    "Deviation frequency $\Pr(n_{\mathrm{dev}}>0)$", ...
    "Alpha sweep (cost vs deviation)");

plot_conditional_cost_alpha_box( ...
    allRows, struct("a",6,"sigma",15,"alpha",[50 75 90 95 99]), ...
    [ALG.CE_NAIVE, ALG.RRCE_PNE], ...
    "obj_norm", "normalized cost $J/J_{\mathrm{greedy}}$", ...
    "Alpha sweep: conditional cost diagnostics");
exportgraphics(gcf, "alpha_conditional_cost.pdf","Resolution",300);



%% ============================================================
% ==================== Local functions ========================
% ============================================================

function meta = parse_case_from_filename(fname)
    meta = struct("ok",false,"a",NaN,"sigma",NaN,"alpha",NaN);
    pat = "(?<a>\d+)a_(?<s>\d+)s_(?<c>\d+)c";
    m = regexp(fname, pat, 'names');
    if isempty(m), return; end
    meta.ok    = true;
    meta.a     = str2double(m.a);
    meta.sigma = str2double(m.s);
    meta.alpha = str2double(m.c);
end

function plot_box_byA(T, spec, algs, yfield, ylab, ttl)
    algs = string(algs);

    X = T( ismember(T.a, spec.a) & T.coord_sigma==spec.sigma & T.alpha==spec.alpha ...
         & ismember(T.algorithm, algs), :);
    assert(height(X)>0, "No data found for plot: %s", ttl);

    % x-axis grouping
    aOrder = sort(spec.a(:))';
    X.xcat = categorical(string(X.a), string(aOrder), 'Ordinal', true);

    % FIX: enforce stable algorithm order for GroupByColor
    X.algorithm = categorical(string(X.algorithm), algs, 'Ordinal', true);
    X.algorithm = removecats(X.algorithm);

    figure('Name', ttl);
    ax = axes(); hold(ax,'on');
    boxchart(ax, X.xcat, log(X.(yfield)), 'GroupByColor', X.algorithm);

    xlabel("aircraft count (a)");
    ylabel(ylab);
    title(ttl);
    grid(ax,'on');

    % FIX: legend must match categories actually used
    legend(ax, categories(X.algorithm), 'Location','best');
end

function plot_box_byA2(T, spec, algs, yfield, ylab, ttl)
    algs = string(algs);

    X = T( ismember(T.a, spec.a) & T.coord_sigma==spec.sigma & T.alpha==spec.alpha ...
         & ismember(T.algorithm, algs), :);
    assert(height(X)>0, "No data found for plot: %s", ttl);

    % x-axis grouping
    aOrder = sort(spec.a(:))';
    X.xcat = categorical(string(X.a), string(aOrder), 'Ordinal', true);

    % FIX: enforce stable algorithm order for GroupByColor
    X.algorithm = categorical(string(X.algorithm), algs, 'Ordinal', true);
    X.algorithm = removecats(X.algorithm);

    figure('Name', ttl);
    ax = axes(); hold(ax,'on');
    boxchart(ax, X.xcat, (X.(yfield)), 'GroupByColor', X.algorithm);

    xlabel("aircraft count (a)");
    ylabel(ylab);
    title(ttl);
    grid(ax,'on');

    % FIX: legend must match categories actually used
    legend(ax, categories(X.algorithm), 'Location','best');
end

function plot_box_bySigma(T, spec, algs, yfield, ylab, ttl)
    algs = string(algs);

    X = T( T.a==spec.a & ismember(T.coord_sigma, spec.sigma) & T.alpha==spec.alpha ...
         & ismember(T.algorithm, algs), :);
    assert(height(X)>0, "No data found for plot: %s", ttl);

    % x-axis grouping
    sOrder = sort(spec.sigma(:))';
    X.xcat = categorical(string(X.coord_sigma), string(sOrder), 'Ordinal', true);

    % FIX: enforce stable algorithm order
    X.algorithm = categorical(string(X.algorithm), algs, 'Ordinal', true);
    X.algorithm = removecats(X.algorithm);

    figure('Name', ttl);
    ax = axes(); hold(ax,'on');
    boxchart(ax, X.xcat, X.(yfield), 'GroupByColor', X.algorithm);

    xlabel("Utility uncertainty");
    ylabel(ylab);
    title(ttl);
    grid(ax,'on');

    legend(ax, categories(X.algorithm), 'Location','best');
end

function plot_errorbar_bySigma(T, spec, algs, yfield, ylab, ttl)
    algs = string(algs);

    % Slice data
    X = T( T.a==spec.a & ismember(T.coord_sigma, spec.sigma) & T.alpha==spec.alpha ...
         & ismember(T.algorithm, algs), :);
    assert(height(X)>0, "No data found for plot: %s", ttl);

    % Enforce ordering
    sigmaOrder = spec.sigma(:)';
    algOrder   = algs(:)';

    X.coord_sigma = categorical(string(X.coord_sigma), ...
                                string(sigmaOrder), 'Ordinal', true);
    X.algorithm   = categorical(string(X.algorithm), ...
                                algOrder, 'Ordinal', true);

    % Prepare figure
    figure('Name', ttl);
    ax = axes(); hold(ax,'on');

    % Slight horizontal offsets to avoid overlap
    nAlg = numel(algOrder);
    offsets = linspace(-0.15, 0.15, nAlg);

    % x positions
    xBase = 1:numel(sigmaOrder);

    % Loop over algorithms
    for k = 1:nAlg
        alg = algOrder(k);
        Xk = X(X.algorithm==alg, :);

        mu  = nan(size(xBase));
        err = nan(size(xBase));

        for i = 1:numel(sigmaOrder)
            Xi = Xk(Xk.coord_sigma==string(sigmaOrder(i)), yfield);
            Xi = Xi.(yfield);
            Xi = Xi(~isnan(Xi));

            if isempty(Xi), continue; end

            mu(i)  = mean(Xi);
            % 95% confidence interval
            err(i) = 1.96 * std(Xi) / sqrt(numel(Xi));
        end

        errorbar(ax, xBase + offsets(k), mu, err, ...
            'o-', 'CapSize', 8, 'LineWidth', 1.8, ...
            'DisplayName', alg);
    end

    % Axes cosmetics
    ax.XTick = xBase;
    ax.XTickLabel = string(sigmaOrder);
    xlabel("Utility uncertainty $\sigma$");
    ylabel(ylab);
    title(ttl);
    grid(ax,'on');
    legend(ax, 'Location','best');
end


function plot_two_metric_alpha_box(T, spec, algs, y1, y1lab, y2lab, ttl)
    % NOTE:
    % y1  : cost or obj_norm (boxchart)
    % y2  : ignored (we compute deviation frequency from n_dev)
    % y2lab : label for deviation frequency axis

    algs = string(algs);

    X = T( T.a==spec.a & T.coord_sigma==spec.sigma & ismember(T.alpha, spec.alpha) ...
         & ismember(T.algorithm, algs), :);
    assert(height(X)>0, "No data found for plot: %s", ttl);

    assert(ismember("n_dev", string(X.Properties.VariableNames)), ...
        "'n_dev' column required for deviation frequency plot.");

    % x-axis grouping
    aOrder = sort(spec.alpha(:))';
    X.xcat = categorical(string(X.alpha), string(aOrder), 'Ordinal', true);

    % stable algorithm order
    X.algorithm = categorical(string(X.algorithm), algs, 'Ordinal', true);
    X.algorithm = removecats(X.algorithm);

    figure('Name', ttl);
    tl = tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

    %% --------------------
    % (1) cost / efficiency (boxchart)
    % ---------------------
    ax1 = nexttile(tl,1); hold(ax1,'on');
    boxchart(ax1, X.xcat, X.(y1), 'GroupByColor', X.algorithm);
    ylabel(ax1, y1lab);
    title(ax1, ttl);
    grid(ax1,'on');
    legend(ax1, categories(X.algorithm), 'Location','best');

    %% --------------------
    % (2) deviation frequency (errorbar)
    % ---------------------
    ax2 = nexttile(tl,2); hold(ax2,'on');

    nAlg = numel(algs);
    offsets = linspace(-0.15, 0.15, nAlg);
    xBase = 1:numel(aOrder);

    for k = 1:nAlg
        alg = algs(k);
        Xk = X(X.algorithm==alg, :);

        p  = nan(size(xBase));
        lo = nan(size(xBase));
        hi = nan(size(xBase));

        for i = 1:numel(aOrder)
            Xi = Xk(Xk.xcat==string(aOrder(i)), :);
            if height(Xi)==0, continue; end

            yi = Xi.n_dev > 0;   % deviation occurred?
            n  = numel(yi);
            phat = mean(yi);

            % Wilson 95% CI
            z = 1.96;
            denom = 1 + z^2/n;
            center = (phat + z^2/(2*n)) / denom;
            half = (z/denom) * sqrt((phat*(1-phat) + z^2/(4*n)) / n);

            p(i)  = phat;
            lo(i) = max(0, center - half);
            hi(i) = min(1, center + half);
        end

        errorbar(ax2, xBase + offsets(k), p, p-lo, hi-p, ...
            'o-', 'CapSize', 8, 'LineWidth', 1.8, ...
            'DisplayName', alg);
    end

    ax2.XTick = xBase;
    ax2.XTickLabel = string(aOrder);
    xlabel(ax2, "confidence $\alpha$ [\%]");
    ylabel(ax2, y2lab);
    ylim(ax2, [0 1]);
    grid(ax2,'on');
    legend(ax2, 'Location','best');
end


function plot_errorbar_devfreq_bySigma(T, spec, algs, ttl)
    algs = string(algs);

    % Slice
    X = T( T.a==spec.a & ismember(T.coord_sigma, spec.sigma) & T.alpha==spec.alpha ...
         & ismember(T.algorithm, algs), :);
    assert(height(X)>0, "No data found for plot: %s", ttl);

    % Need n_dev
    assert(ismember("n_dev", string(X.Properties.VariableNames)), ...
        "Column 'n_dev' is required for deviation frequency plot.");

    sigmaOrder = spec.sigma(:)';
    algOrder   = algs(:)';

    % Categories for consistent ordering
    X.coord_sigma = categorical(string(X.coord_sigma), string(sigmaOrder), 'Ordinal', true);
    X.algorithm   = categorical(string(X.algorithm), algOrder, 'Ordinal', true);

    figure('Name', ttl);
    ax = axes(); hold(ax,'on');

    nAlg = numel(algOrder);
    offsets = linspace(-0.15, 0.15, nAlg);
    xBase = 1:numel(sigmaOrder);

    for k = 1:nAlg
        alg = algOrder(k);
        Xk = X(X.algorithm==alg, :);

        p  = nan(size(xBase));
        lo = nan(size(xBase));
        hi = nan(size(xBase));

        for i = 1:numel(sigmaOrder)
            Xi = Xk(Xk.coord_sigma==string(sigmaOrder(i)), :);
            if height(Xi)==0, continue; end

            % Event: any deviation in epoch
            yi = Xi.n_dev > 0;

            n = numel(yi);
            phat = mean(yi);

            % Wilson 95% CI (robust for small n / extremes)
            z = 1.96;
            denom = 1 + z^2/n;
            center = (phat + z^2/(2*n)) / denom;
            half = (z/denom) * sqrt( (phat*(1-phat) + z^2/(4*n)) / n );

            p(i)  = phat;
            lo(i) = max(0, center - half);
            hi(i) = min(1, center + half);
        end

        % convert to symmetric error for errorbar
        errLow = p - lo;
        errHigh = hi - p;

        errorbar(ax, xBase + offsets(k), p, errLow, errHigh, ...
            'o-', 'CapSize', 8, 'LineWidth', 1.8, ...
            'DisplayName', alg);
    end

    ax.XTick = xBase;
    ax.XTickLabel = string(sigmaOrder);
    xlabel("Utility uncertainty $\sigma$");
    ylabel("Deviation frequency $\Pr(n_{\mathrm{dev}}>0)$");
    title(ttl);
    ylim([0 1]);
    grid(ax,'on');
    legend(ax, 'Location','best');
end

function plot_conditional_cost_alpha_box(T, spec, algs, yfield, ylab, ttl)
    % Shows 3 panels:
    % (1) unconditional cost
    % (2) cost | (n_dev > 0)
    % (3) cost | (n_dev == 0)

    algs = string(algs);

    X = T( T.a==spec.a & T.coord_sigma==spec.sigma & ismember(T.alpha, spec.alpha) ...
         & ismember(T.algorithm, algs), :);
    assert(height(X)>0, "No data found for plot: %s", ttl);
    assert(ismember("n_dev", string(X.Properties.VariableNames)), "'n_dev' required.");

    % x-axis order
    aOrder = sort(spec.alpha(:))';
    X.xcat = categorical(string(X.alpha), string(aOrder), 'Ordinal', true);

    % stable algorithm order
    X.algorithm = categorical(string(X.algorithm), algs, 'Ordinal', true);
    X.algorithm = removecats(X.algorithm);

    figure('Name', ttl);
    tl = tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

    % -------- (1) unconditional
    ax1 = nexttile(tl,1); hold(ax1,'on');
    boxchart(ax1, X.xcat, X.(yfield), 'GroupByColor', X.algorithm);
    ylabel(ax1, ylab);
    title(ax1, ttl + " (unconditional)");
    grid(ax1,'on');
    legend(ax1, categories(X.algorithm), 'Location','best');

    % -------- (2) conditional dev>0
    ax2 = nexttile(tl,2); hold(ax2,'on');
    Xp = X(X.n_dev > 0, :);
    if height(Xp)==0
        text(ax2, 0.1, 0.5, "No samples with n\_dev>0 in this slice.", 'Units','normalized');
        axis(ax2,'off');
    else
        boxchart(ax2, Xp.xcat, Xp.(yfield), 'GroupByColor', Xp.algorithm);
        ylabel(ax2, ylab);
        title(ax2, "conditional: $n_{\mathrm{dev}}>0$");
        grid(ax2,'on');
        legend(ax2, categories(Xp.algorithm), 'Location','best');
    end

    % -------- (3) conditional dev==0
    ax3 = nexttile(tl,3); hold(ax3,'on');
    Xz = X(X.n_dev == 0, :);
    if height(Xz)==0
        text(ax3, 0.1, 0.5, "No samples with n\_dev=0 in this slice.", 'Units','normalized');
        axis(ax3,'off');
    else
        boxchart(ax3, Xz.xcat, Xz.(yfield), 'GroupByColor', Xz.algorithm);
        xlabel(ax3, "confidence $\alpha$ [\%]");
        ylabel(ax3, ylab);
        title(ax3, "conditional: $n_{\mathrm{dev}}=0$");
        grid(ax3,'on');
        legend(ax3, categories(Xz.algorithm), 'Location','best');
    end
end

function plot_conditional_cost_sigma_box(T, spec, algs, yfield, ylab, ttl)
    algs = string(algs);

    X = T( T.a==spec.a & ismember(T.coord_sigma, spec.sigma) & T.alpha==spec.alpha ...
         & ismember(T.algorithm, algs), :);
    assert(height(X)>0, "No data found for plot: %s", ttl);
    assert(ismember("n_dev", string(X.Properties.VariableNames)), "'n_dev' required.");

    sOrder = sort(spec.sigma(:))';
    X.xcat = categorical(string(X.coord_sigma), string(sOrder), 'Ordinal', true);

    X.algorithm = categorical(string(X.algorithm), algs, 'Ordinal', true);
    X.algorithm = removecats(X.algorithm);

    figure('Name', ttl);
    tl = tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

    ax1 = nexttile(tl,1); hold(ax1,'on');
    boxchart(ax1, X.xcat, X.(yfield), 'GroupByColor', X.algorithm);
    ylabel(ax1, ylab);
    title(ax1, ttl + " (unconditional)");
    grid(ax1,'on');
    legend(ax1, categories(X.algorithm), 'Location','best');

    ax2 = nexttile(tl,2); hold(ax2,'on');
    Xp = X(X.n_dev > 0, :);
    if height(Xp)==0
        text(ax2, 0.1, 0.5, "No samples with n\_dev>0 in this slice.", 'Units','normalized');
        axis(ax2,'off');
    else
        boxchart(ax2, Xp.xcat, Xp.(yfield), 'GroupByColor', Xp.algorithm);
        ylabel(ax2, ylab);
        title(ax2, "conditional: $n_{\mathrm{dev}}>0$");
        grid(ax2,'on');
        legend(ax2, categories(Xp.algorithm), 'Location','best');
    end

    ax3 = nexttile(tl,3); hold(ax3,'on');
    Xz = X(X.n_dev == 0, :);
    if height(Xz)==0
        text(ax3, 0.1, 0.5, "No samples with n\_dev=0 in this slice.", 'Units','normalized');
        axis(ax3,'off');
    else
        boxchart(ax3, Xz.xcat, Xz.(yfield), 'GroupByColor', Xz.algorithm);
        xlabel(ax3, "Utility uncertainty $\sigma$");
        ylabel(ax3, ylab);
        title(ax3, "conditional: $n_{\mathrm{dev}}=0$");
        grid(ax3,'on');
        legend(ax3, categories(Xz.algorithm), 'Location','best');
    end
end
