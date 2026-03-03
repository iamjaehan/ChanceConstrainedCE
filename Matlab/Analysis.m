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
set(groot, 'defaultTextFontWeight', 'normal');
set(groot, 'defaultAxesFontWeight', 'normal');

% Optional but recommended for papers
set(groot, 'defaultAxesLineWidth', 1.2);
set(groot, 'defaultLineLineWidth', 2.0);
set(groot, 'defaultFigureColor', 'w');

%% ---------------------------
% USER CONFIG
% ----------------------------
dataDir = ".."; % folder containing all CSVs
fileGlob = fullfile(dataDir, "*.csv"); % or "mc_epoch_results_*.csv"
onlyOK  = false;

% If you want to filter "invalid RRCE runs" (optional; default off)
% Example: require num_pne > 0 for RRCE_PNE
filterInvalidRRCE = true;
filterCEFullNoProgress = true;   % <-- on/off switch

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

if filterCEFullNoProgress
    % robust string normalization
    st = lower(strtrim(string(allRows.status)));
    isCEfull = (string(allRows.algorithm) == "CE_FULL");

    % Treat any of these as "non-progress"
    badStatus = (st == "mcp_noprogress") | (st == "mcp_noprogress" ) | (st == "mcp_no_progress");
    % (include both spellings to be safe)

    allRows = allRows(~(isCEfull & badStatus), :);
end
allRows.wall_ms = allRows.wall_ms / 1000;

%% ---------------------------
% Algorithm labels (your actual solver strings)
% ----------------------------
ALG.CE_FULL   = "CE_FULL";
ALG.RRCE_PNE  = "RRCE_PNE";
ALG.CE_NAIVE  = "CE_NAIVE";
ALG.FCFS      = "AGG_ORACLE_FCFS";
ALG.CENTRAL   = "GREEDY_CENTRALIZED";

% ---- Display names (legend labels) ----
global DISP
DISP = containers.Map();
DISP("GREEDY_CENTRALIZED") = "CENT";
DISP("AGG_ORACLE_FCFS")    = "FCFS";
DISP("CE_FULL")            = "Full-CCCE";
DISP("CE_NAIVE")           = "RR-Nominal";
DISP("RRCE_PNE")           = "RR-CCCE";

% ---- Fixed colors per algorithm (RGB in [0,1]) ----
global COL
COL = containers.Map();
COL("GREEDY_CENTRALIZED") = [0.2 0.2 0.2];
COL("AGG_ORACLE_FCFS")    = [0.49 0.18 0.56];
COL("CE_FULL")            = [0.85 0.33 0.10];
COL("CE_NAIVE")           = [0.93 0.69 0.13];
COL("RRCE_PNE")           = [0.0 0.45 0.74];

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
% plot_box_byA( ...
%     allRows, struct("a",[4 5 6 7 8 9], "sigma",0, "alpha",90), ...
%     [ALG.CE_FULL, ALG.RRCE_PNE], ...
%     "wall_ms", "Computation time [ms] (log)", ...
%     "");
% exportgraphics(gcf, "1_Scalability.pdf","Resolution",300);

plot_scalability_overlay( ...
    allRows, ...
    struct("a",4:14,"sigma",0,"alpha",90), ...
    ALG.CE_FULL, ALG.RRCE_PNE, ...
    "wall_ms", "Computation time [s] (log)", "" ...
);
text(7,400,"Epoch duration (4 min)",'FontWeight','bold')
set(gcf, 'Position', [1000 818 560  350]);
exportgraphics(gcf, "1_Scalability.pdf","Resolution",300);

%% ============================================================
% 2) Efficiency-to-scale: obj for 6/0/90 vs 8/0/90, exclude CE_NAIVE
% alg: all except CE_NAIVE (within that slice)
% ============================================================
slice2 = allRows(allRows.coord_sigma==0 & allRows.alpha==90 & ismember(allRows.a,[4 5 6 7 8]), :);
algs2 = unique(slice2.algorithm, 'stable');
algs2(algs2 == ALG.CE_NAIVE) = [];

plot_box_byA2( ...
    allRows, struct("a",[4 5 6 7 8 9], "sigma",0, "alpha",90), ...
    algs2, ...
    "obj", "Delay Cost", ...
    "");
set(gcf, 'Position', [1000 818 560  350]);
exportgraphics(gcf, "2_OverallCost.pdf","Resolution",300);

%% ==========
% 3) Deviation rate [\%] by sigma
% ============
plot_errorbar_devfreq_bySigma( ...
    allRows, struct("a",6,"sigma",[0 5 20 45],"alpha",90), ...
    [ALG.CE_FULL, ALG.CE_NAIVE, ALG.RRCE_PNE], ...
    "");
set(gcf, 'Position', [1000 818 560  350]);
exportgraphics(gcf, "3_devfreq_uncertainty.pdf","Resolution",300);

%% ============================================================
% 4) Efficiency-to-error: obj for 6/0/90, 6/10/90, 6/30/90
% x: coord_sigma, y: obj, alg: FCFS, CE_NAIVE, RRCE_PNE
% ============================================================
plot_box_bySigma( ...
    allRows, struct("a",6, "sigma",[0 5 20 45], "alpha",90), ...
    [ALG.CE_FULL, ALG.CE_NAIVE, ALG.RRCE_PNE], ...
    "obj", "Delay Cost", ...
    "");
set(gcf, 'Position', [1000 818 560  350]);
exportgraphics(gcf, "4_eff_uncertainty.pdf","Resolution",300);

%% ============================================================
% 5) Efficiency vs confidence (alpha sweep)  [separate figure]
% ============================================================
temp = plot_box_byAlpha( ...
    allRows, struct("a",6,"sigma",20,"alpha",[30 50 75 90 95 99]), ...
    [ALG.CE_FULL, ALG.CE_NAIVE, ALG.RRCE_PNE], ...
    "obj", "Delay Cost", ...
    "");
set(gcf, 'Position', [1000 818 560  350]);
exportgraphics(gcf, "eff_confidence.pdf","Resolution",300);

%% ============================================================
% 6) Deviation rate [\%] vs confidence (alpha sweep) [separate figure]
% ============================================================
plot_errorbar_devfreq_byAlpha( ...
    allRows, struct("a",6,"sigma",20,"alpha",[30 50 75 90 95 99]), ...
    [ALG.CE_FULL, ALG.CE_NAIVE, ALG.RRCE_PNE], ...
    "");
set(gcf, 'Position', [1000 818 560  350]);
exportgraphics(gcf, "devfreq_confidence.pdf","Resolution",300);

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

function plot_box_byA2(T, spec, algs, yfield, ylab, ttl)
    global DISP COL
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
    h = boxchart(ax, X.xcat, (X.(yfield)), 'GroupByColor', X.algorithm,'MarkerStyle','none');
    apply_fixed_colors_boxchart(h, categories(X.algorithm), COL);

    algCats = categories(X.algorithm);
    nGroups = numel(algCats);
    
    for iG = 1:nGroups
        alg = string(algCats{iG});
        mask = (X.algorithm == algCats{iG});
        overlay_group_median_star(ax, X.xcat(mask), X.(yfield)(mask), nGroups, iG, COL(alg));
    end

    xlabel("Number of eligible aircraft per epoch");
    ylabel(ylab);
    title(ttl);
    grid(ax,'on');
    ax.XGrid = 'off';
    
    yl = ylim(ax);
    nG = numel(categories(X.xcat));
    for i = 1:(nG-1)
        xline(ax, i+0.5, ':', ...
            'Color', [0.2 0.2 0.2], ...
            'LineWidth', 1.8, ...
            'HandleVisibility','off');
    end

    % FIX: legend must match categories actually used
    legend(ax, map_display_names(categories(X.algorithm), DISP), 'Location','northwest','Interpreter', 'latex','FontName','times');
end

function plot_box_bySigma(T, spec, algs, yfield, ylab, ttl)
    global DISP COL
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
    h = boxchart(ax, X.xcat, X.(yfield), 'GroupByColor', X.algorithm,'MarkerStyle','none');
    apply_fixed_colors_boxchart(h, categories(X.algorithm), COL);

    % after boxchart(...)
    nAlg = numel(algs);
    offsets = linspace(-0.33, 0.33, nAlg);   % boxchart보다 조금 넓게/좁게 조절 가능
    
    xBase = 1:numel(sOrder);
    
    for k = 1:nAlg
        alg = algs(k);
        Xk = X(X.algorithm==alg, :);
    
        mu = nan(size(xBase));
        for i = 1:numel(sOrder)
            Xi = Xk(Xk.xcat==string(sOrder(i)), :);
            yi = Xi.(yfield);
            yi = yi(~isnan(yi));
            if isempty(yi), continue; end
            mu(i) = mean(yi);
        end
    
        c = get_alg_color(alg, COL);
        % 점만 (권장)
        % plot(ax, xBase + offsets(k), mu, 'o', 'HandleVisibility','off');  % legend 지저분해지면 숨김
    
        % 선까지 연결하려면 아래로 교체
        plot(ax, xBase + offsets(k), mu, '-x', 'HandleVisibility','off','MarkerEdgeColor',c, 'MarkerFaceColor',c, 'Color',c);
    end

    ylim([7 35])


    xlabel("Utility uncertainty $\sigma$ ");
    ylabel(ylab);
    title(ttl);
    grid(ax,'on');
    ax.XGrid = 'off';
    ax = gca;

    yl = ylim(ax);
    nG = numel(categories(X.xcat));
    for i = 1:(nG-1)
        xline(ax, i+0.5, ':', ...
            'Color', [0.2 0.2 0.2], ...
            'LineWidth', 1.8, ...
            'HandleVisibility','off');
    end

    legend(ax, map_display_names(categories(X.algorithm), DISP), 'Location','northeast','Interpreter', 'latex','FontName','times');
end

function X = plot_box_byAlpha(T, spec, algs, yfield, ylab, ttl)
    global DISP COL
    algs = string(algs);

    X = T( T.a==spec.a & T.coord_sigma==spec.sigma & ismember(T.alpha, spec.alpha) ...
         & ismember(T.algorithm, algs), :);
    assert(height(X)>0, "No data found for plot: %s", ttl);

    % x-axis grouping (alpha order)
    cOrder = sort(spec.alpha(:))';
    X.xcat = categorical(string(X.alpha), string(cOrder), 'Ordinal', true);

    % stable algorithm order
    X.algorithm = categorical(string(X.algorithm), algs, 'Ordinal', true);
    X.algorithm = removecats(X.algorithm);

    figure('Name', ttl);
    ax = axes(); hold(ax,'on');
    h = boxchart(ax, X.xcat, X.(yfield), 'GroupByColor', X.algorithm,'MarkerStyle','none');
    apply_fixed_colors_boxchart(h, categories(X.algorithm), COL);

    nAlg = numel(algs);
    offsets = linspace(-0.33, 0.33, nAlg);   % boxchart보다 조금 넓게/좁게 조절 가능
    
    xBase = 1:numel(cOrder);
    
    for k = 1:nAlg
        alg = algs(k);
        Xk = X(X.algorithm==alg, :);
    
        mu = nan(size(xBase));
        length(cOrder)
        for i = 1:numel(cOrder)
            Xi = Xk(Xk.xcat==string(cOrder(i)), :);
            yi = Xi.(yfield);
            yi = yi(~isnan(yi));
            if isempty(yi), continue; end
            mu(i) = mean(yi);
        end
    
        c = get_alg_color(alg, COL);
        % 점만 (권장)
        % plot(ax, xBase + offsets(k), mu, 'o', 'HandleVisibility','off');  % legend 지저분해지면 숨김
    
        % 선까지 연결하려면 아래로 교체
        plot(ax, xBase + offsets(k), mu, '-*', 'HandleVisibility','off', 'MarkerEdgeColor',c, 'MarkerFaceColor', c, 'Color',c);
    end
    
    yl = ylim(ax);
    nG = numel(categories(X.xcat));
    for i = 1:(nG-1)
        xline(ax, i+0.5, ':', ...
            'Color', [0.2 0.2 0.2], ...
            'LineWidth', 1.8, ...
            'HandleVisibility','off');
    end
    ylim([7 35])

    xlabel("Confidence level $\alpha$ [\%]");
    ylabel(ylab);
    title(ttl);
    grid(ax,'on');
    ax.XGrid = 'off';
    legend(ax, map_display_names(categories(X.algorithm), DISP), 'Location','northeast','Interpreter', 'latex','FontName','times');
end


function plot_errorbar_devfreq_byAlpha(T, spec, algs, ttl)
    global DISP COL
    algs = string(algs);

    X = T( T.a==spec.a & T.coord_sigma==spec.sigma & ismember(T.alpha, spec.alpha) ...
         & ismember(T.algorithm, algs), :);
    assert(height(X)>0, "No data found for plot: %s", ttl);

    assert(ismember("n_dev", string(X.Properties.VariableNames)) || ...
           ismember("dev_rate", string(X.Properties.VariableNames)), ...
        "Need 'n_dev' or 'dev_rate' for Deviation rate [\%] plot.");

    % use dev_rate > 0 as event (matches your sigma-plot)
    if ~ismember("dev_rate", string(X.Properties.VariableNames))
        % fallback if only n_dev exists
        X.dev_rate = double(X.n_dev > 0);
    end

    cOrder = spec.alpha(:)';
    algOrder = algs(:)';

    X.alpha     = categorical(string(X.alpha), string(cOrder), 'Ordinal', true);
    X.algorithm = categorical(string(X.algorithm), algOrder, 'Ordinal', true);

    figure('Name', ttl);
    ax = axes(); hold(ax,'on');

    nAlg = numel(algOrder);
    offsets = linspace(-0.15, 0.15, nAlg);
    xBase = 1:numel(cOrder);

    for k = 1:nAlg
        alg = algOrder(k);
        Xk = X(X.algorithm==alg, :);

        p  = nan(size(xBase));
        lo = nan(size(xBase));
        hi = nan(size(xBase));

        for i = 1:numel(cOrder)
            Xi = Xk(Xk.alpha==string(cOrder(i)), :);
            if height(Xi)==0, continue; end

            yi = (Xi.dev_rate > 0); % event
            n  = numel(yi);
            phat = mean(yi);

            % Wilson 95% CI
            z = 1.645;
            denom  = 1 + z^2/n;
            center = (phat + z^2/(2*n)) / denom;
            half   = (z/denom) * sqrt((phat*(1-phat) + z^2/(4*n)) / n);

            p(i)  = phat;
            lo(i) = max(0, center - half/2);
            hi(i) = min(1, center + half/2);
        end

        c = COL(string(alg));
        errorbar(ax, xBase + offsets(k), p, p-lo, hi-p, ...
            'o-', 'CapSize', 8, 'LineWidth', 1.8, ...
            'DisplayName', map_display_names(string(alg), DISP), 'Color', c);
    end

    yl = ylim(ax);
    nG = numel(xBase);
    for i = 1:(nG-1)
        xline(ax, i+0.5, ':', ...
            'Color', [0.2 0.2 0.2], ...
            'LineWidth', 1.8, ...
            'HandleVisibility','off');
    end

    ax.XTick = xBase;
    ax.XTickLabel = string(cOrder);
    xlabel("Confidence level $\alpha$ [\%]");
    ylabel("Deviation rate [\%]");
    title(ttl);
    ylim([0 0.6]);
    yl = ylim(ax);    
    yt = ax.YTick;
    ax.YTickLabel = compose('%.0f', 100*yt);
    grid(ax,'on');
    ax.XGrid = 'off';
    legend(ax, map_display_names(categories(X.algorithm), DISP), 'Location','northeast','Interpreter', 'latex','FontName','times','FontName','times');
end

function plot_errorbar_devfreq_bySigma(T, spec, algs, ttl)
    global DISP COL
    algs = string(algs);

    % Slice
    X = T( T.a==spec.a & ismember(T.coord_sigma, spec.sigma) & T.alpha==spec.alpha ...
         & ismember(T.algorithm, algs), :);
    assert(height(X)>0, "No data found for plot: %s", ttl);

    % Need n_dev
    assert(ismember("dev_rate", string(X.Properties.VariableNames)), ...
        "Column 'n_dev' is required for Deviation rate [\%] plot.");

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
            yi = Xi.dev_rate > 0;

            n = numel(yi);
            phat = mean(yi);

            % Wilson 95% CI (robust for small n / extremes)
            z = 1.645;
            denom = 1 + z^2/n;
            center = (phat + z^2/(2*n)) / denom;
            half = (z/denom) * sqrt( (phat*(1-phat) + z^2/(4*n)) / n );

            p(i)  = phat;
            lo(i) = max(0, center - half/2);
            hi(i) = min(1, center + half/2);
        end

        % convert to symmetric error for errorbar
        errLow = p - lo;
        errHigh = hi - p;

        c = COL(string(alg));
        errorbar(ax, xBase + offsets(k), p, errLow, errHigh, ...
            'o-', 'CapSize', 8, 'LineWidth', 1.8, ...
            'DisplayName', map_display_names(string(alg), DISP), 'Color', c);
    end

    yl = ylim(ax);
    nG = numel(xBase);
    for i = 1:(nG-1)
        xline(ax, i+0.5, ':', ...
            'Color', [0.2 0.2 0.2], ...
            'LineWidth', 1.8, ...
            'HandleVisibility','off');
    end

    ax.XTick = xBase;
    ax.XTickLabel = string(sigmaOrder);
    xlabel("Utility uncertainty $\sigma$");
    ylabel("Deviation rate [\%]");
    title(ttl);
    ylim([0 0.6]);
    yl = ylim(ax);    
    yt = ax.YTick;
    ax.YTickLabel = compose('%.0f', 100*yt);
    grid(ax,'on');
    legend(ax, map_display_names(categories(X.algorithm), DISP), 'Location','northeast','Interpreter', 'latex','FontName','times');
end

function out = map_display_names(cats, DISP)
    out = string(cats);
    for i = 1:numel(out)
        key = string(cats{i});
        if isKey(DISP, key)
            out(i) = DISP(key);
        end
    end
end

function apply_fixed_colors_boxchart(h, algCats, COL)
    % h can be scalar or array (one per group). Handle both.
    if ~isscalar(h)
        hs = h;
    else
        hs = h; % sometimes MATLAB returns scalar; still OK
    end

    % When grouped, MATLAB typically returns an array of BoxChart objects
    % in the same order as categories(X.algorithm).
    for i = 1:numel(hs)
        key = string(algCats{i});
        if isKey(COL, key)
            c = COL(key);
            try
                hs(i).BoxFaceColor = c;
            catch
            end
            try
                hs(i).BoxEdgeColor = c;
            catch
            end
            try
                hs(i).WhiskerLineColor = c;
            catch
            end
            try
                hs(i).MedianLineColor = c;
            catch
            end
        end
    end
end

function c = get_alg_color(alg, COL)
    key = string(alg);
    if isKey(COL, key)
        c = COL(key);
    else
        c = [0 0 0]; % fallback
    end
end

function plot_scalability_overlay(T, spec, alg_ce, alg_rr, yfield, ylab, ttl)
    global DISP COL

    aOrder = sort(spec.a(:))';
    aCats  = categorical(string(aOrder), string(aOrder), 'Ordinal', true);

    X = T( ismember(T.a, spec.a) & T.coord_sigma==spec.sigma & T.alpha==spec.alpha ...
         & ismember(T.algorithm, [alg_ce, alg_rr]), :);
    assert(height(X)>0, "No data for scalability overlay.");

    % Make a categorical x with all a shown (even if data missing)
    X.xcat = categorical(string(X.a), string(aOrder), 'Ordinal', true);

    figure('Name', ttl);
    ax = axes(); hold(ax,'on');

    % --- CE_FULL (only where it exists) ---
    Xce = X(string(X.algorithm)==string(alg_ce), :);
    if height(Xce) > 0
        hce = boxchart(ax, Xce.xcat, Xce.(yfield), 'MarkerStyle','none');
        c = COL(string(alg_ce));
        try hce.BoxFaceColor = c; end
        try hce.BoxEdgeColor = c; end
        try hce.WhiskerLineColor = c; end
        try hce.MedianLineColor = c; end
        hce.DisplayName = DISP(string(alg_ce));
    end

    [g, xlev] = findgroups(Xce.xcat);
    med = splitapply(@median, Xce.(yfield), g);
    plot(ax, double(xlev), med, 'o', 'MarkerSize', 5, 'LineWidth', 1.4, ...
         'Color', COL(string(alg_ce)), 'HandleVisibility','off');

    hold on

    % --- RRCE_PNE (includes extra a=10..13) ---
    Xrr = X(string(X.algorithm)==string(alg_rr), :);
    if height(Xrr) > 0
        hrr = boxchart(ax, Xrr.xcat, Xrr.(yfield), 'MarkerStyle','none');
        c = COL(string(alg_rr));
        try hrr.BoxFaceColor = c; end
        try hrr.BoxEdgeColor = c; end
        try hrr.WhiskerLineColor = c; end
        try hrr.MedianLineColor = c; end
        hrr.DisplayName = DISP(string(alg_rr));
    end

    [g, xlev] = findgroups(Xrr.xcat);
    med = splitapply(@median, Xrr.(yfield), g);
    plot(ax, double(xlev), med, 'o', 'MarkerSize', 5, 'LineWidth', 1.4, ...
         'Color', COL(string(alg_rr)), 'HandleVisibility','off');

    xlabel("Number of eligible aircraft per epoch");
    ylabel(ylab);
    title(ttl);
    grid(ax,'on');
    ax.XGrid = 'off';

    % separators
    nG = numel(categories(aCats));
    for i = 1:(nG-1)
        xline(ax, i+0.5, ':', 'Color',[0.2 0.2 0.2], 'LineWidth',1.8, 'HandleVisibility','off');
    end

    % your existing styling
    yline(ax, 240, 'HandleVisibility','off','LineWidth',2,'LineStyle','-.');
    ax.YScale = 'log';

    legend(ax, 'Location','northwest','Interpreter','latex','FontName','times');
end

function overlay_group_median_star(ax, xcat, y, nGroups, iGroup, c)
    % Overlay median as a star marker (keep existing median line)
    % xcat: categorical (Ordinal recommended)
    % nGroups: number of groups in GroupByColor
    % iGroup:  1..nGroups (order consistent with categories used in GroupByColor)
    % c:       RGB color for the star marker (e.g., COL(alg))

    dx = 0.372 * ( (2*iGroup - (nGroups+1)) / max(1,(nGroups-1)) ); % symmetric offsets

    % median per x category
    [g, xlev] = findgroups(xcat);
    med = splitapply(@median, y, g);

    xCenters = double(xlev) + dx;

    plot(ax, xCenters, med, 'o', ...
        'MarkerSize', 5, 'LineWidth', 1.4, ...
        'Color', c, ...
        'HandleVisibility','off');
end