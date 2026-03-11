clear; clc; close all;

%% files
cases = {
    % 'big',   'mc_results_ccce_alpha_big.mat'
    'med',   'mc_results_ccce_alpha_med.mat'
    'small', 'mc_results_ccce_alpha_small.mat'
    'usmall','mc_results_ccce_alpha_usmall.mat'
};

% cases = {
%     'big',   'mc_results_ccce_alpha_big.mat'
%     'med',   'mc_results_ccce_alpha_med.mat'
%     'small', 'mc_results_ccce_alpha_small.mat'
% };

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
scoreNormFactor = [300,30,30];
scoreNormFactor = [1,1,1];

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

        norm_score(idx_t) = (score(idx_t) ./ base_score-1)./scoreNormFactor(c)+1;
    end

    for k = 1:nAlg
        idx = (alg_label == order(k)) & ~isnan(norm_score);
        all_norm_score{c,k} = norm_score(idx);
    end
end

%% plot
% figure('Position',[1000 818 560  300]);
% hold on

%% plot
figure('Position',[100 100 1200 500]);
hold on

xCenters = 1:nAlg;
offsets  = [-0.18, 0, 0.18];   % case 3개니까 3개만 쓰는 게 맞음
% offsets  = [-0.21 -0.07 0.07 0.21]* 1.1;   % case 3개니까 3개만 쓰는 게 맞음
boxw     = 0.14;

case_names = string(cases(:,1));
h = gobjects(nCases,1);

for c = 1:nCases
    thisColor = case_colors(c,:);
    mean_vals = nan(1,nAlg);   % 각 case의 algorithm별 mean 저장

    for k = 1:nAlg
        vals = all_norm_score{c,k};
        if isempty(vals)
            continue
        end

        xc = xCenters(k) + offsets(c);
        x  = ones(size(vals)) * xc;

        % 1) boxchart
        b = boxchart(x, vals, ...
            'BoxWidth', boxw, ...
            'BoxFaceColor', thisColor, ...
            'BoxEdgeColor', thisColor, ...
            'MarkerColor', thisColor, ...
            'WhiskerLineColor', thisColor, ...
            'LineWidth', 1.2);
        b.BoxFaceAlpha = 0.12;
        b.MarkerStyle = 'o';
        b.JitterOutliers = 'off';

        % legend handle 저장
        if ~isgraphics(h(c))
            h(c) = b;
        end

        % 2) swarmchart overlay
        swarmchart(x, vals, 18, ...
            'MarkerEdgeColor', thisColor, ...
            'MarkerFaceColor', thisColor, ...
            'MarkerFaceAlpha', 0.28, ...
            'MarkerEdgeAlpha', 0.45, ...
            'XJitter', 'density', ...
            'XJitterWidth', boxw*0.55);

        % 3) mean marker
        mu = mean(vals, 'omitnan');
        mean_vals(k) = mu;
        mean_vals(1) = nan;

        plot(xc, mu, 'd', ...
            'MarkerSize', 7, ...
            'MarkerFaceColor', thisColor, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.8);
    end

    % 4) mean끼리 연결
    valid = ~isnan(mean_vals);
    plot(xCenters(valid) + offsets(c), mean_vals(valid), '-', ...
        'Color', thisColor, ...
        'LineWidth', 1.8);
end

set(gca, 'YScale', 'linear')
xlim([0.5, nAlg+0.5])
xticks(xCenters)
xticklabels(order)
xtickangle(20)
% ylim([0.95 1.4])

ylabel('Normalized score')
xlabel('Algorithm')
title('Normalized score by algorithm and case')
grid on
box on

legend(h, case_names, 'Location', 'best')