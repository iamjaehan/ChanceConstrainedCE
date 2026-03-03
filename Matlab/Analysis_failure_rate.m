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
filterInvalidRRCE = false;
filterCEFullNoProgress = true;   % <-- on/off switch

yLimVal = 0.6;

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
% PNE statistics (case-level) for RRCE_PNE
%  - how often PNE enumeration fails (num_pne == 0 or NaN)
%  - distribution of num_pne by (a, coord_sigma, alpha)
% ============================================================

% ---- USER SWITCHES ----
enablePneStats = true;
pneFailMode = "zero_or_nan";   % "zero", "nan", "zero_or_nan"
exportPneTables = true;

if enablePneStats
    assert(ismember("num_pne", string(allRows.Properties.VariableNames)), ...
        "'num_pne' column is required for PNE stats.");

    Xp = allRows(allRows.algorithm == ALG.RRCE_PNE, :);
    if height(Xp)==0
        warning("No RRCE_PNE rows found. Skipping PNE stats.");
    else
        caseVars = ["a","coord_sigma","alpha"];

        % normalize num_pne
        np = Xp.num_pne;

        switch pneFailMode
            case "zero"
                isFail = (np == 0);
            case "nan"
                isFail = isnan(np);
            case "zero_or_nan"
                isFail = (np == 0) | isnan(np);
            otherwise
                error("Unknown pneFailMode: %s", pneFailMode);
        end

        % case grouping
        [caseKeys,~,gid] = unique(Xp(:, caseVars), 'rows');

        % aggregate stats
        n_mc        = splitapply(@numel, np, gid);
        fail_rate   = splitapply(@(x) mean((x==0) | isnan(x)), np, gid);  % conservative
        mean_pne    = splitapply(@(x) mean(x(~isnan(x))), np, gid);
        median_pne  = splitapply(@(x) median(x(~isnan(x))), np, gid);
        min_pne     = splitapply(@(x) min(x(~isnan(x))), np, gid);
        max_pne     = splitapply(@(x) max(x(~isnan(x))), np, gid);
        % ---- NEW: std/se for PNE count (ignore NaN) ----
        std_pne = splitapply(@(x) std(x(~isnan(x))), np, gid);
        n_eff   = splitapply(@(x) sum(~isnan(x)),    np, gid);
        se_pne  = std_pne ./ sqrt(n_eff);

        % handle "all NaN" cases (std/se too)
        std_pne(n_eff==0) = NaN;
        se_pne(n_eff==0)  = NaN;

        % handle "all NaN" cases
        mean_pne(isnan(mean_pne))     = NaN;
        median_pne(isnan(median_pne)) = NaN;
        min_pne(isnan(min_pne))       = NaN;
        max_pne(isnan(max_pne))       = NaN;

        pneStats = caseKeys;
        pneStats.n_mc          = n_mc;
        pneStats.fail_rate     = fail_rate;
        pneStats.mean_num_pne  = mean_pne;
        pneStats.median_num_pne= median_pne;
        pneStats.min_num_pne   = min_pne;
        pneStats.max_num_pne   = max_pne;
        pneStats.n_eff_pne     = n_eff;      % NEW
        pneStats.std_num_pne   = std_pne;    % NEW
        pneStats.se_num_pne    = se_pne;     % NEW

        % overall summary
        overall = table();
        overall.n_rows = height(Xp);
        overall.fail_rate = mean(isFail);
        overall.mean_num_pne = mean(np(~isnan(np)));
        overall.median_num_pne = median(np(~isnan(np)));
        overall.min_num_pne = min(np(~isnan(np)));
        overall.max_num_pne = max(np(~isnan(np)));

        fprintf("[PNE] RRCE_PNE rows=%d, fail_rate=%.3f\n", overall.n_rows, overall.fail_rate);

        % show worst cases
        [~,idx] = sort(pneStats.fail_rate, 'descend');
        topK = min(10, height(pneStats));
        disp("Top failure-rate cases (up to 10):");
        disp(pneStats(idx(1:topK), :));

        if exportPneTables
            writetable(pneStats, "pne_case_stats.csv");
            writetable(overall,  "pne_overall_stats.csv");
        end
    end
end


%% ============================================================
% PNE plots (dual y-axis, errorbar) - NO local functions
%   Left  y-axis: mean #PNE (± 1.96*SE)
%   Right y-axis: failure rate (± 1.96*SE), in [0,1]
% ============================================================
enablePnePlots = true;

if exist("pneStats","var") && enablePnePlots && height(pneStats)>0

    z = 1.645;

    % ------------------------------------------------------------
    % (1) vs a : sigma=0, alpha=90
    % ------------------------------------------------------------
    sliceA = pneStats(pneStats.coord_sigma==0 & pneStats.alpha==90, :);
    assert(height(sliceA)>0, "No PNE stats for (sigma=0, alpha=90).");

    aOrder = sort(unique(sliceA.a));
    n = numel(aOrder);
    mu = NaN(n,1); se_mu = NaN(n,1);
    pf = NaN(n,1); se_pf = NaN(n,1);

    if ~ismember("se_fail", string(sliceA.Properties.VariableNames))
        assert(ismember("n_mc", string(sliceA.Properties.VariableNames)), "Need se_fail or n_mc in pneStats.");
        sliceA.se_fail = sqrt(sliceA.fail_rate .* (1 - sliceA.fail_rate) ./ sliceA.n_mc);
    end

    for i = 1:n
        R = sliceA(sliceA.a==aOrder(i), :);
        mu(i)    = mean(R.mean_num_pne, 'omitnan');
        se_mu(i) = mean(R.se_num_pne,   'omitnan');
        pf(i)    = mean(R.fail_rate,    'omitnan');
        se_pf(i) = mean(R.se_fail,      'omitnan');
    end

    x = (1:n)'; xlab = string(aOrder);

    figure('Name','RRCE_PNE: PNE stats vs a');
    ax = axes(); hold(ax,'on');

    yyaxis(ax,'left');
    errorbar(ax, x, mu, z*se_mu, 'o-', 'CapSize',8, 'LineWidth',1.8);
    ylabel(ax, "Average \# of PNE  ");

    yyaxis(ax,'right');
    errorbar(ax, x, pf, z*se_pf, 's--', 'CapSize',8, 'LineWidth',1.8);
    ylabel(ax, "Failure rate [\%]");
    ylim(ax,[0 yLimVal]);
    yl = ylim(ax);    
    yt = ax.YTick;
    ax.YTickLabel = compose('%.0f', 100*yt);

    ax.XTick = x; ax.XTickLabel = xlab;
    xlabel(ax, "Number of eligible aircraft per epoch");
    grid(ax,'on');
    legend(ax, {"Average \#PNE","Failure rate"}, 'Location','northwest');
    set(gcf,'Position',[1000 818 560  280])

    exportgraphics(gcf, "pne_ac.pdf","Resolution",300);

    % ------------------------------------------------------------
    % (2) vs sigma : a=6, alpha=90
    % ------------------------------------------------------------
    sliceS = pneStats(pneStats.a==6 & pneStats.alpha==90, :);
    assert(height(sliceS)>0, "No PNE stats for (a=6, alpha=90).");

    sOrder = sort(unique(sliceS.coord_sigma));
    n = numel(sOrder);
    mu = NaN(n,1); se_mu = NaN(n,1);
    pf = NaN(n,1); se_pf = NaN(n,1);

    if ~ismember("se_fail", string(sliceS.Properties.VariableNames))
        assert(ismember("n_mc", string(sliceS.Properties.VariableNames)), "Need se_fail or n_mc in pneStats.");
        sliceS.se_fail = sqrt(sliceS.fail_rate .* (1 - sliceS.fail_rate) ./ sliceS.n_mc);
    end

    for i = 1:n
        R = sliceS(sliceS.coord_sigma==sOrder(i), :);
        mu(i)    = mean(R.mean_num_pne, 'omitnan');
        se_mu(i) = mean(R.se_num_pne,   'omitnan');
        pf(i)    = mean(R.fail_rate,    'omitnan');
        se_pf(i) = mean(R.se_fail,      'omitnan');
    end

    x = (1:n)'; xlab = string(sOrder);

    figure('Name','RRCE_PNE: PNE stats vs sigma');
    ax = axes(); hold(ax,'on');

    yyaxis(ax,'left');
    errorbar(ax, x, mu, z*se_mu, 'o-', 'CapSize',8, 'LineWidth',1.8);
    ylabel(ax, "Average \# of PNE  ");

    yyaxis(ax,'right');
    errorbar(ax, x, pf, z*se_pf, 's--', 'CapSize',8, 'LineWidth',1.8);
    ylabel(ax, "Failure rate [\%]");
    ylim(ax,[0 yLimVal]);
    yl = ylim(ax);    
    yt = ax.YTick;
    ax.YTickLabel = compose('%.0f', 100*yt);

    ax.XTick = x; ax.XTickLabel = xlab;
    xlabel(ax, "Utility uncertainty $\sigma$");
    grid(ax,'on');
    legend(ax, {"Average \#PNE","Failure rate"}, 'Location','northeast');
    set(gcf,'Position',[1000 818 560  280])

    exportgraphics(gcf, "pne_sigma.pdf","Resolution",300);

    % ------------------------------------------------------------
    % (3) vs alpha : a=6, sigma=20
    % ------------------------------------------------------------
    sliceC = pneStats(pneStats.a==6 & pneStats.coord_sigma==20, :);
    assert(height(sliceC)>0, "No PNE stats for (a=6, sigma=20).");

    cOrder = sort(unique(sliceC.alpha));
    n = numel(cOrder);
    mu = NaN(n,1); se_mu = NaN(n,1);
    pf = NaN(n,1); se_pf = NaN(n,1);

    if ~ismember("se_fail", string(sliceC.Properties.VariableNames))
        assert(ismember("n_mc", string(sliceC.Properties.VariableNames)), "Need se_fail or n_mc in pneStats.");
        sliceC.se_fail = sqrt(sliceC.fail_rate .* (1 - sliceC.fail_rate) ./ sliceC.n_mc);
    end

    for i = 1:n
        R = sliceC(sliceC.alpha==cOrder(i), :);
        mu(i)    = mean(R.mean_num_pne, 'omitnan');
        se_mu(i) = mean(R.se_num_pne,   'omitnan');
        pf(i)    = mean(R.fail_rate,    'omitnan');
        se_pf(i) = mean(R.se_fail,      'omitnan');
    end

    x = (1:n)'; xlab = string(cOrder);

    figure('Name','RRCE_PNE: PNE stats vs alpha');
    ax = axes(); hold(ax,'on');

    yyaxis(ax,'left');
    errorbar(ax, x, mu, z*se_mu, 'o-', 'CapSize',8, 'LineWidth',1.8);
    ylabel(ax, "Average \# of PNE  ");

    yyaxis(ax,'right');
    errorbar(ax, x, pf, z*se_pf, 's--', 'CapSize',8, 'LineWidth',1.8);
    ylabel(ax, "Failure rate [\%]");
    ylim(ax,[0 yLimVal]);

    ax.XTick = x; ax.XTickLabel = xlab;
    xlabel(ax, "confidence $\alpha$ [\%]");
    grid(ax,'on');
    legend(ax, {"Average \#PNE","Failure rate"}, 'Location','northeast');
    yl = ylim(ax);    
    yt = ax.YTick;
    ax.YTickLabel = compose('%.0f', 100*yt);
    set(gcf,'Position',[1000 818 560  280])

    exportgraphics(gcf, "pne_alpha.pdf","Resolution",300);

end



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

    xlabel("Number of eligible aircraft per epoch");
    ylabel(ylab);
    title(ttl);
    grid(ax,'on');

    % FIX: legend must match categories actually used
    legend(ax, categories(X.algorithm), 'Location','best');
end
