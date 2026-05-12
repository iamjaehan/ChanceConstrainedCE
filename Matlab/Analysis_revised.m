% =============================================================================
% Analysis_revised.m
% Plots for Experiment 1 (alpha sweep) and Experiment 2 (InfoGain strategy)
% Data files: ../exp1_results.mat, ../exp2_results.mat
% =============================================================================

clear; clc; close all;

% ── Global style ──────────────────────────────────────────────────────────────
set(groot, 'defaultTextInterpreter',          'tex');
set(groot, 'defaultAxesTickLabelInterpreter', 'tex');
set(groot, 'defaultLegendInterpreter',        'tex');
set(groot, 'defaultAxesFontSize',    23);
set(groot, 'defaultTextFontSize',    23);
set(groot, 'defaultLegendFontSize',  23);
set(groot, 'defaultAxesLineWidth',   1.2);
set(groot, 'defaultLineLineWidth',   1.5);
set(groot, 'defaultFigureColor',     'w');
set(groot, 'defaultAxesFontName',    'Times New Roman');
set(groot, 'defaultTextFontName',    'Times New Roman');

% ─────────────────────────────────────────────────────────────────────────────
% CONFIG  (toggle here as needed)
% ─────────────────────────────────────────────────────────────────────────────

% Deviation metric:
%   'deviation_rate'     → fraction of rollouts with ≥1 deviation
%   'mean_num_deviators' → average # agents deviating per rollout
DEV_METRIC = 'deviation_rate';
% DEV_METRIC = 'mean_num_deviators';
DEV_LABEL  = 'Deviation rate';
% DEV_METRIC = 'mean_num_deviators';
% DEV_LABEL  = 'Mean \# deviators';

% Exp2 normalization mode:
%   'none'     → absolute cost
%   'baseline' → divide by Baseline (Baseline=1)
%   'minmax'   → (val-min)/(baseline-min)  →  Baseline=1, best method=0
NORMALIZE_E2 = 'minmax';

% ─────────────────────────────────────────────────────────────────────────────
% Load data
% ─────────────────────────────────────────────────────────────────────────────
E1 = load('../exp1_results.mat');
E2 = load('../exp2_results.mat');

% =============================================================================
%% EXPERIMENT 1 — Alpha sweep (NaiveCE vs CC-CE)
% =============================================================================

alphas_e1   = double(E1.alpha(:));
exp_cost_e1 = double(E1.expected_cost(:));
real_mean_e1= double(E1.realized_mean(:));
trials_e1   = double(E1.trial(:));

if strcmp(DEV_METRIC, 'deviation_rate')
    dev_vals_e1 = double(E1.deviation_rate(:));
else
    dev_vals_e1 = double(E1.mean_num_deviators(:));
end

% Unique alpha values in ascending order → defines x-axis order
unique_alphas = sort(unique(alphas_e1));
nA = length(unique_alphas);

% Build display labels (flexible to any alpha list)
labels_e1 = cell(nA, 1);
for k = 1:nA
    a = unique_alphas(k);
    if a == 0
        labels_e1{k} = 'NaiveCE';
    else
        labels_e1{k} = sprintf('CC-CE (\\alpha=%.2f)', a);
    end
end

% Collect per-method data vectors, normalized by NaiveCE within each trial
data_exp_e1  = cell(nA, 1);
data_real_e1 = cell(nA, 1);
data_dev_e1  = cell(nA, 1);

for k = 1:nA
    idx = (alphas_e1 == unique_alphas(k));

    ec = exp_cost_e1(idx);
    rm = real_mean_e1(idx);
    dv = dev_vals_e1(idx);

    % Normalize expected/realized cost by NaiveCE value within same trial
    trial_ids = trials_e1(idx);
    for t = unique(trial_ids)'
        naive_idx = (trials_e1 == t) & (alphas_e1 == 0);
        base_exp  = exp_cost_e1(naive_idx);
        base_real = real_mean_e1(naive_idx);
        if isempty(base_exp) || base_exp == 0; continue; end
        t_loc = (trial_ids == t);
        ec(t_loc) = ec(t_loc) ./ base_exp;
        rm(t_loc) = rm(t_loc) ./ base_real;
    end

    data_exp_e1{k}  = ec(~isnan(ec));
    data_real_e1{k} = rm(~isnan(rm));
    data_dev_e1{k}  = dv(~isnan(dv));
end

% ── Figure 1 ─────────────────────────────────────────────────────────────────
fig1 = figure('Name', 'Exp1', 'Position', [50 50 900 550]);
ax_l = axes(fig1);

colorExp  = [0.20 0.45 0.75];   % blue  – expected cost
colorReal = [0.85 0.33 0.10];   % orange – realized cost
colorDev  = [0.47 0.67 0.19];   % green  – deviation rate

mu_exp  = cellfun(@(v) mean(v(~isnan(v))), data_exp_e1)';
mu_real = cellfun(@(v) mean(v(~isnan(v))), data_real_e1)';
mu_dev  = cellfun(@(v) mean(v(~isnan(v))), data_dev_e1)';

se_exp  = cellfun(@(v) 1.96 * std(v(~isnan(v))) / sqrt(sum(~isnan(v))), data_exp_e1)';
se_real = cellfun(@(v) 1.96 * std(v(~isnan(v))) / sqrt(sum(~isnan(v))), data_real_e1)';
se_dev  = cellfun(@(v) 1.96 * std(v(~isnan(v))) / sqrt(sum(~isnan(v))), data_dev_e1)';

xs = 1:nA;

hold(ax_l, 'on');

fill(ax_l, [xs fliplr(xs)], ...
    [mu_exp+se_exp fliplr(mu_exp-se_exp)], ...
    colorExp, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
fill(ax_l, [xs fliplr(xs)], ...
    [mu_real+se_real fliplr(mu_real-se_real)], ...
    colorReal, 'FaceAlpha', 0.15, 'EdgeColor', 'none');

h_exp  = plot(ax_l, xs, mu_exp,  '-o', 'Color', colorExp,  ...
    'LineWidth', 2, 'MarkerSize', 7, ...
    'MarkerFaceColor', colorExp,  'MarkerEdgeColor', 'k');
h_real = plot(ax_l, xs, mu_real, '-s', 'Color', colorReal, ...
    'LineWidth', 2, 'MarkerSize', 7, ...
    'MarkerFaceColor', colorReal, 'MarkerEdgeColor', 'k');

yline(ax_l, 1.0, '--k', 'LineWidth', 2.0, 'Alpha', 0.5);

set(ax_l, 'XTick', xs, 'XTickLabel', {}, ...
    'FontName', 'Times New Roman', 'FontSize', 23);
ylabel(ax_l, 'Normalized cost', 'FontName', 'Times New Roman', 'FontSize', 23);
xlim(ax_l, [0.5  nA + 0.5]);
ylim(ax_l, [0.5  Inf]);
grid(ax_l, 'on');  box(ax_l, 'on');

% Right y-axis for deviation rate
ax_r = axes(fig1, 'Position', ax_l.Position, ...
    'YAxisLocation', 'right', 'Color', 'none', ...
    'FontName', 'Times New Roman', 'FontSize', 23);
ax_r.XTick = [];
hold(ax_r, 'on');

fill(ax_r, [xs fliplr(xs)], ...
    [mu_dev+se_dev fliplr(mu_dev-se_dev)], ...
    colorDev, 'FaceAlpha', 0.15, 'EdgeColor', 'none');

h_dev = plot(ax_r, xs, mu_dev, '-^', 'Color', colorDev, ...
    'LineWidth', 2, 'MarkerSize', 7, ...
    'MarkerFaceColor', colorDev, 'MarkerEdgeColor', 'k');

ax_r.YAxis.Color = colorDev;
ylabel(ax_r, 'Deviation rate', 'FontName', 'Times New Roman', 'FontSize', 23, 'Color', colorDev);
xlim(ax_r, [0.5  nA + 0.5]);
ylim(ax_r, [0  Inf]);

% Link x-axes so zoom/pan stays in sync
linkaxes([ax_l ax_r], 'x');

set(gcf,'Position',[1000 818 650 350])
lg = legend(ax_l, [h_exp h_real h_dev], ...
    {'Norm. expected cost', 'Norm. realized cost', 'Deviation rate'}, ...
    'Location', 'east', 'FontName', 'Times New Roman', 'FontSize', 23, ...
    'Interpreter', 'tex');
lg.Position(2) = lg.Position(2) + 0.03;

exportgraphics(fig1, 'exp1_results.pdf', 'Resolution', 300);

% =============================================================================
%% EXPERIMENT 2 — InfoGain strategy comparison across γ values
% =============================================================================

gammas_e2    = double(E2.gamma(:));
methods_e2   = string(E2.method(:));
trials_e2    = double(E2.trial(:));
exp_cost_e2  = double(E2.expected_cost(:));
real_mean_e2 = double(E2.realized_mean(:));

if strcmp(DEV_METRIC, 'deviation_rate')
    dev_vals_e2 = double(E2.deviation_rate(:));
else
    dev_vals_e2 = double(E2.mean_num_deviators(:));
end

% Unique gamma values (ascending) and method names
unique_gammas   = sort(unique(gammas_e2));
nG              = length(unique_gammas);
unique_methods  = unique(methods_e2);

% Preferred method order (Baseline first)
pref_order = ["Baseline", ...
              "Random-5",        "Top5-"  + char(963), "Top5-"  + char(955), "Top5-IG",  ...
              "Half-Random-5",   "Half-Top5-" + char(963), "Half-Top5-" + char(955), "Half-Top5-IG"];
method_order = strings(0);
for k = 1:length(pref_order)
    if any(unique_methods == pref_order(k))
        method_order(end+1) = pref_order(k);
    end
end
for k = 1:length(unique_methods)          % append any unlisted methods
    if ~any(method_order == unique_methods(k))
        method_order(end+1) = unique_methods(k);
    end
end
nM = length(method_order);

% tex display labels for methods
labels_e2 = cell(nM, 1);
for k = 1:nM
    s = char(method_order(k));
    s = strrep(s, char(963), '\sigma');
    s = strrep(s, char(955), '\lambda');
    labels_e2{k} = s;
end

% Collect expected and realized cost: data{method_idx, gamma_idx}
% Raw (unnormalized) arrays first — normalization applied below
data_exp_e2_raw  = cell(nM, nG);
data_real_e2_raw = cell(nM, nG);
for ki = 1:nM
    for kj = 1:nG
        idx = (methods_e2 == method_order(ki)) & (gammas_e2 == unique_gammas(kj));
        data_exp_e2_raw{ki,kj}  = exp_cost_e2(idx);
        data_real_e2_raw{ki,kj} = real_mean_e2(idx);
    end
end

% Apply normalization per (gamma, trial)
data_exp_e2  = cell(nM, nG);
data_real_e2 = cell(nM, nG);
for ki = 1:nM
    for kj = 1:nG
        idx = (methods_e2 == method_order(ki)) & (gammas_e2 == unique_gammas(kj));
        ec  = exp_cost_e2(idx);
        rm  = real_mean_e2(idx);

        if ~strcmp(NORMALIZE_E2, 'none')
            trial_ids = trials_e2(idx);
            for t = unique(trial_ids)'
                tidx_all  = (gammas_e2 == unique_gammas(kj)) & (trials_e2 == t);
                base_idx  = tidx_all & (methods_e2 == "Baseline");
                base_exp  = exp_cost_e2(base_idx);
                base_real = real_mean_e2(base_idx);
                if isempty(base_exp) || base_exp == 0; continue; end
                t_loc = (trial_ids == t);

                if strcmp(NORMALIZE_E2, 'baseline')
                    ec(t_loc) = ec(t_loc) ./ base_exp;
                    rm(t_loc) = rm(t_loc) ./ base_real;

                elseif strcmp(NORMALIZE_E2, 'minmax')
                    % min over all methods for this (gamma, trial)
                    all_ec   = arrayfun(@(mi) exp_cost_e2(tidx_all & (methods_e2 == method_order(mi))), ...
                                        1:nM, 'UniformOutput', false);
                    all_ec   = [all_ec{:}];
                    min_exp  = min(all_ec(~isnan(all_ec)));
                    denom_exp = base_exp - min_exp;

                    all_rm   = arrayfun(@(mi) real_mean_e2(tidx_all & (methods_e2 == method_order(mi))), ...
                                        1:nM, 'UniformOutput', false);
                    all_rm   = [all_rm{:}];
                    min_real  = min(all_rm(~isnan(all_rm)));
                    denom_real = base_real - min_real;

                    if denom_exp > 0
                        ec(t_loc) = (ec(t_loc) - min_exp) ./ denom_exp;
                    end
                    if denom_real > 0
                        rm(t_loc) = (rm(t_loc) - min_real) ./ denom_real;
                    end
                end
            end
        end

        data_exp_e2{ki,kj}  = ec(~isnan(ec));
        data_real_e2{ki,kj} = rm(~isnan(rm));
    end
end

% ── Figure 2 (disabled) ───────────────────────────────────────────────────────
% fig2 = figure('Name', 'Exp2', 'Position', [100 50 900 550]);
% ...
% exportgraphics(fig2, 'exp2_results.pdf', 'Resolution', 300);

% gamma_labels needed by Figure 3
gamma_labels = arrayfun(@(g) sprintf('\\gamma=%.2f', g), ...
                        unique_gammas, 'UniformOutput', false);
markers = {'o', 's', '^', 'd', 'v'};

% =============================================================================
%% EXPERIMENT 2 — Figure 3: Half strategies only (partial info acquisition)
% =============================================================================

half_methods = ["Baseline", "Half-Random-5", ...
                "Half-Top5-" + char(963), ...
                "Half-Top5-" + char(955), ...
                "Half-Top5-IG"];

% Find indices into method_order for the half subset
half_idx = zeros(1, length(half_methods));
for k = 1:length(half_methods)
    pos = find(method_order == half_methods(k), 1);
    if ~isempty(pos)
        half_idx(k) = pos;
    end
end
half_idx = half_idx(half_idx > 0);
nH = length(half_idx);

% tex labels for the half subset
labels_e3 = cell(nH, 1);
for k = 1:nH
    s = char(method_order(half_idx(k)));
    s = strrep(s, char(963), '\sigma');
    s = strrep(s, char(955), '\lambda');
    labels_e3{k} = s;
end

fig3 = figure('Name', 'Exp2-Half', 'Position', [150 50 900 550]);
ax3  = axes(fig3);
hold(ax3, 'on');

cmap_e3  = lines(nG);
h_gamma3 = gobjects(nG, 1);
xs3 = 1:nH;

for kj = 1:nG
    c  = cmap_e3(kj, :);
    mk = markers{mod(kj-1, numel(markers)) + 1};

    mu_vals = zeros(1, nH);
    ci_vals = zeros(1, nH);
    for ki = 1:nH
        v = data_exp_e2{half_idx(ki), kj};
        v = sort(v(~isnan(v)));
        mu_vals(ki) = median(v);
        ci_vals(ki) = (quantile(v, 0.75) - quantile(v, 0.25)) / 2;
    end

    fill(ax3, [xs3 fliplr(xs3)], ...
        [mu_vals+ci_vals fliplr(mu_vals-ci_vals)], ...
        c, 'FaceAlpha', 0.15, 'EdgeColor', 'none');

    h_gamma3(kj) = plot(ax3, xs3, mu_vals, ['-' mk], 'Color', c, ...
        'LineWidth', 2, 'MarkerSize', 7, ...
        'MarkerFaceColor', c, 'MarkerEdgeColor', 'k');
end

if ~strcmp(NORMALIZE_E2, 'none')
    yline(ax3, 1.0, '--k', 'LineWidth', 2, 'Alpha', 0.5);
end
if strcmp(NORMALIZE_E2, 'minmax')
    yline(ax3, 0.0, ':k', 'LineWidth', 2, 'Alpha', 0.4);
end

set(ax3, 'XTick', [], ...
    'FontName', 'Times New Roman', 'FontSize', 23);

ylabel(ax3, 'Standardized score', 'FontName', 'Times New Roman', 'FontSize', 23);

xlim(ax3, [0.5  nH + 0.5]);
grid(ax3, 'on');  box(ax3, 'on');

set(gcf,'Position',[1000 818 650 350])
ylim([-0.1 inf])
lg = legend(ax3, h_gamma3, gamma_labels, ...
    'Location', 'southwest', 'FontName', 'Times New Roman', 'FontSize', 23, ...
    'Interpreter', 'tex');
lg.Position(2) = lg.Position(2) + 0.1;
exportgraphics(fig3, 'exp3_results.pdf', 'Resolution', 300);