clear; clc; close all;

% load mc_results_ccce_alpha.mat
% load mc_results_ccce_alpha_big.mat
% load mc_results_ccce_alpha_med.mat
% load mc_results_ccce_alpha_small.mat
load mc_results_ccce_alpha_usmall.mat

%% parameter
nSamples = 10;      % number of MC samples per algorithm
confLevel = 0.95;    % confidence level

%% z-value for normal CI
zval = norminv(0.5 + confLevel/2);

method = string(method);
alpha  = double(alpha);
score  = double(score);
mean_num_deviators = double(mean_num_deviators);
mc_iter = double(mc_iter);

N = numel(method);
alg_label = strings(N,1);

for i = 1:N
    if method(i) == "NE"
        alg_label(i) = "NE";
    elseif method(i) == "NaiveCE"
        alg_label(i) = "Naive CE";
    elseif method(i) == "CCCE"
        alg_label(i) = sprintf("CC-CE (%.2f)", alpha(i));
    else
        alg_label(i) = method(i);
    end
end

order = [
    "NE"
    "Naive CE"
    "CC-CE (0.75)"
    "CC-CE (0.90)"
    "CC-CE (0.95)"
    "CC-CE (0.99)"
];

%% normalize score by Naive CE within each MC instance
norm_score = nan(N,1);
all_iters = unique(mc_iter);

for t = all_iters(:)'
    idx_t = (mc_iter == t);
    idx_naive = idx_t & (alg_label == "Naive CE");

    if sum(idx_naive) ~= 1
        continue
    end

    base_score = score(idx_naive);

    if isnan(base_score) || abs(base_score) < 1e-12
        continue
    end

    norm_score(idx_t) = score(idx_t) ./ base_score;
end

alg_cat = categorical(alg_label, order, 'Ordinal', true);
xcat = categorical(order, order, 'Ordinal', true);

%% mean number of deviators + CI
mean_dev_alg = nan(length(order),1);
ci_halfwidth = nan(length(order),1);

for k = 1:length(order)
    if order(k) == "NE"
        continue
    end

    idx = (alg_label == order(k)) & ~isnan(mean_num_deviators);
    vals = mean_num_deviators(idx);

    if ~isempty(vals)
        mean_dev_alg(k) = mean(vals);
        s = std(vals, 0);
        ci_halfwidth(k) = zval * s / sqrt(nSamples);
    end
end

%% mean normalized score for trend line
mean_score_alg = nan(length(order),1);

for k = 1:length(order)
    idx = (alg_label == order(k)) & ~isnan(norm_score);
    vals = norm_score(idx);

    if ~isempty(vals)
        mean_score_alg(k) = mean(vals);
    end
end

%% plot
figure('Position',[100 100 1150 520]);
hold on

yyaxis right
h1 = plot(xcat, mean_dev_alg, '-o', 'LineWidth', 2, 'MarkerSize', 7);
hold on
h2 = errorbar(xcat, mean_dev_alg, ci_halfwidth, ...
    'LineStyle', 'none', ...
    'LineWidth', 1.5, ...
    'CapSize', 10);
ylabel('Mean number of deviators')
ylim([0 3])

% test = [nan, 0.5, 0.75, 0.9, 0.95, 0.99];
% plot(3*(1-test))

yyaxis left
idx_score_valid = ~isnan(norm_score);
boxchart(alg_cat(idx_score_valid), norm_score(idx_score_valid), 'BoxWidth', 0.6)
hold on
h3 = plot(xcat, mean_score_alg, '-s', 'LineWidth', 2, 'MarkerSize', 7);
ylabel('Normalized score')

xlabel('Algorithm')
title('Normalized score and mean number of deviators')
grid on
box on
xtickangle(20)

legend([h1 h3], {'Mean # deviators', 'Mean normalized score'}, 'Location', 'best')