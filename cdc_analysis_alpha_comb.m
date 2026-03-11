clear; clc; close all;

%% files
% cases = {
%     'big',   'mc_results_ccce_alpha_big.mat'
%     'med',   'mc_results_ccce_alpha_med.mat'
%     'small', 'mc_results_ccce_alpha_small.mat'
%     'usmall','mc_results_ccce_alpha_usmall.mat'
% };

cases = {
    'big',   'mc_results_ccce_alpha_big.mat'
    'med',   'mc_results_ccce_alpha_med.mat'
    'small', 'mc_results_ccce_alpha_small.mat'
};

order = [
    "NE"
    "Naive CE"
    "CC-CE (0.75)"
    "CC-CE (0.90)"
    "CC-CE (0.95)"
    "CC-CE (0.99)"
];

nCases = size(cases,1);
nAlg   = length(order);

%% fixed colors for each case
case_colors = [
    0.0000    0.4470    0.7410   % big   : blue
    0.8500    0.3250    0.0980   % med   : orange/red
    0.4940    0.1840    0.5560   % small : purple
    0.4660    0.6740    0.1880   % usmall: green
];

%% store normalized scores
all_norm_score = cell(nCases, nAlg);

for c = 1:nCases
    S = load(cases{c,2});

    method  = string(S.method);
    alpha   = double(S.alpha);
    score   = double(S.score);
    mc_iter = double(S.mc_iter);

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

    % normalize by Naive CE within each MC instance
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

    for k = 1:nAlg
        idx = (alg_label == order(k)) & ~isnan(norm_score);
        all_norm_score{c,k} = norm_score(idx);
    end
end

%% plot
% figure('Position',[1000 818 560  300]);
% hold on

figure('Position',[100 100 1200 500]);
hold on

xCenters = 1:nAlg;
offsets  = [-0.27, -0.09, 0.09, 0.27];

case_names = string(cases(:,1));
h = gobjects(nCases,1);

for c = 1:nCases
    thisColor = case_colors(c,:);

    for k = 1:nAlg
        vals = all_norm_score{c,k};
        if isempty(vals)
            continue
        end

        xc = xCenters(k) + offsets(c);
        x  = ones(size(vals)) * xc;

        % swarm only
        s = swarmchart(x, vals, 22, ...
            'MarkerEdgeColor', thisColor, ...
            'MarkerFaceColor', thisColor, ...
            'MarkerFaceAlpha', 0.30, ...
            'MarkerEdgeAlpha', 0.55, ...
            'XJitter', 'density', ...
            'XJitterWidth', 0.10);
        hold on

        if ~isgraphics(h(c))
            h(c) = s;
        end

        % mean marker
        mu = mean(vals, 'omitnan');
        plot(xc, mu, 'd', ...
            'MarkerSize', 7, ...
            'MarkerFaceColor', thisColor, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.8);
    end
end

set(gca, 'YScale', 'log')
xlim([0.5, nAlg+0.5])
xticks(xCenters)
xticklabels(order)
xtickangle(20)

ylabel('Normalized score')
xlabel('Algorithm')
title('Normalized score by algorithm and case')
grid on
box on

legend(h, case_names, 'Location', 'best')