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

%% ============================================================
% 1) Scalability: wall_ms for 6/0/90 vs 8/0/90
% alg: CE_FULL, RRCE_PNE
% x: a, y: wall_ms
% ============================================================
plot_box_byA( ...
    allRows, struct("a",[6 8], "sigma",0, "alpha",90), ...
    [ALG.CE_FULL, ALG.RRCE_PNE], ...
    "wall_ms", "wall time (ms) [log]", ...
    "Scalability: wall time vs aircraft count");
exportgraphics(gcf, "Scalability.pdf","Resolution",300);

%% ============================================================
% 2) Efficiency-to-scale: obj for 6/0/90 vs 8/0/90, exclude CE_NAIVE
% alg: all except CE_NAIVE (within that slice)
% ============================================================
slice2 = allRows(allRows.coord_sigma==0 & allRows.alpha==90 & ismember(allRows.a,[6 8]), :);
algs2 = unique(slice2.algorithm, 'stable');
algs2(algs2 == ALG.CE_NAIVE) = [];

plot_box_byA2( ...
    allRows, struct("a",[6 8], "sigma",0, "alpha",90), ...
    algs2, ...
    "obj", "cost", ...
    "Efficiency vs scale");
exportgraphics(gcf, "eff_scale.pdf","Resolution",300);

%% ============================================================
% 3) Efficiency-to-error: obj for 6/0/90, 6/10/90, 6/30/90
% x: coord_sigma, y: obj, alg: FCFS, CE_NAIVE, RRCE_PNE
% ============================================================
plot_box_bySigma( ...
    allRows, struct("a",6, "sigma",[0 10 30], "alpha",90), ...
    [ALG.FCFS, ALG.CE_NAIVE, ALG.RRCE_PNE], ...
    "obj", "cost", ...
    "Efficiency vs uncertainty");
exportgraphics(gcf, "eff_uncertainty.pdf","Resolution",300);

%% ============================================================
% 4) Alpha sweep: (6/10/70, 6/10/90, 6/10/99)
% alg: CE_NAIVE, RRCE_PNE
% two metrics: obj and dev_rate (stacked)
% ============================================================
plot_two_metric_alpha_box( ...
    allRows, struct("a",6, "sigma",10, "alpha",[99 90 70]), ...
    [ALG.CE_NAIVE, ALG.RRCE_PNE], ...
    "obj", "cost", ...
    "dev_rate", "deviation rate", ...
    "Alpha sweep");
exportgraphics(gcf, "alpha.pdf","Resolution",300);

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

function plot_two_metric_alpha_box(T, spec, algs, y1, y1lab, y2, y2lab, ttl)
    algs = string(algs);

    X = T( T.a==spec.a & T.coord_sigma==spec.sigma & ismember(T.alpha, spec.alpha) ...
         & ismember(T.algorithm, algs), :);
    assert(height(X)>0, "No data found for plot: %s", ttl);

    % x-axis grouping
    aOrder = sort(spec.alpha(:))';
    X.xcat = categorical(string(X.alpha), string(aOrder), 'Ordinal', true);

    % FIX: stable algorithm order
    X.algorithm = categorical(string(X.algorithm), algs, 'Ordinal', true);
    X.algorithm = removecats(X.algorithm);

    figure('Name', ttl);
    tl = tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

    % --- metric 1 ---
    ax1 = nexttile(tl,1); hold(ax1,'on');
    boxchart(ax1, X.xcat, X.(y1), 'GroupByColor', X.algorithm);
    ylabel(ax1, y1lab);
    title(ax1, ttl);
    grid(ax1,'on');
    legend(ax1, categories(X.algorithm), 'Location','best');

    % --- metric 2 (dev_rate optional) ---
    ax2 = nexttile(tl,2); hold(ax2,'on');

    if ~ismember(y2, string(X.Properties.VariableNames))
        text(ax2, 0.1, 0.5, sprintf("'%s' column not found for this slice.", y2), 'Units','normalized');
        axis(ax2,'off');
        return;
    end

    mask = ~ismissing(X.(y2)) & ~isnan(X.(y2));
    X2 = X(mask,:);
    if height(X2)==0
        text(ax2, 0.1, 0.5, sprintf("No valid '%s' values after filtering.", y2), 'Units','normalized');
        axis(ax2,'off');
        return;
    end

    boxchart(ax2, X2.xcat, X2.(y2), 'GroupByColor', X2.algorithm);
    xlabel(ax2, "alpha [%]");
    ylabel(ax2, y2lab);
    grid(ax2,'on');
    legend(ax2, categories(X2.algorithm), 'Location','best');
end