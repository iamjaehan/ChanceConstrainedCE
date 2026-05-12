clear; clc; close all;
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'tex');
set(groot, 'defaultLegendInterpreter', 'tex');

set(groot, 'defaultAxesFontSize', 18);
set(groot, 'defaultTextFontSize', 18);
set(groot, 'defaultLegendFontSize', 13);
set(groot, 'defaultTextFontWeight', 'normal');
set(groot, 'defaultAxesFontWeight', 'normal');

% Optional but recommended for papers
set(groot, 'defaultAxesLineWidth', 1.2);
set(groot, 'defaultLineLineWidth', 2.0);
set(groot, 'defaultFigureColor', 'w');

%% files
cases = {
    '\gamma =1.50',   'mc_results_ccce_alpha_big_2.mat'
    '\gamma =1.10',   'mc_results_ccce_alpha_med_2.mat'
    '\gamma =1.05', 'mc_results_ccce_alpha_small_2.mat'
    '\gamma =1.02','mc_results_ccce_alpha_usmall_2.mat'
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
scoreNormFactor = [300,30,15,10];
scoreNormFactor = ones(nCases);

for c = 1:nCases
    S = load(cases{c,2});

    method  = string(S.method);
    alpha   = double(S.alpha);
    score   = double(S.score);
    mc_iter = double(S.mc_iter);
    scaleNormFactor = std(score);

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
        idx_naive = idx_t & (alg_label == "CC-CE (0.75)");
        % idx_naive = idx_t & (alg_label == "Naive CE");
        idx_NE = (alg_label == "NE");
        scaleNormFactor = std(score(idx_NE), 0, 'omitnan');

        if sum(idx_naive) ~= 1
            continue
        end

        base_score = score(idx_naive);

        if isnan(base_score) || abs(base_score) < 1e-12
            continue
        end

        % norm_score(idx_t) = (score(idx_t) ./ base_score);
        norm_score(idx_t) = (score(idx_t) - base_score)./scaleNormFactor;
    end

    for k = 1:nAlg
        idx = (alg_label == order(k)) & ~isnan(norm_score);
        all_norm_score{c,k} = norm_score(idx);
    end
end

%% plot
figure('Position',[1000 218 560  300]);
% hold on

% figure('Position',[100 500 500 500]);
hold on

xCenters = 1:nAlg;
% offsets  = [-0.18, 0, 0.18];   % case 3개니까 3개만 쓰는 게 맞음
offsets  = [-0.21 -0.07 0.07 0.21]* 1.3;   % case 3개니까 3개만 쓰는 게 맞음
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
        b.MarkerStyle = 'none';
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
        % mean_vals(1) = nan;

        plot(xc, mu, 'd', ...
            'MarkerSize', 7, ...
            'MarkerFaceColor', thisColor, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.2);
    end

    % 4) mean끼리 연결
    valid = ~isnan(mean_vals);
    plot(xCenters(valid) + offsets(c), mean_vals(valid), '-', ...
        'Color', [thisColor 0.5], ...
        'LineWidth', 2,'LineStyle','-');
end

set(gca, 'YScale', 'linear')
xlim([0.5, nAlg+0.5])
xticks(xCenters)
% xticklabels(alg_label(2:end))
xticklabels([])
% xtickangle(20)
% ylim([0.95 1.4])

ylabel('Standardized score shift')
% xlabel('Algorithm')
% title('')
grid on
box on

legend(h, case_names, 'Location', 'northwest','Interpreter','tex')
% ylim([-0.02 0.05])

%% Plot 2
% figure('Position',[100 100 500 500]);
% hold on
% 
% xCenters = 1:nAlg;
% offsets  = [-0.21 -0.07 0.07 0.21]*1.3;
% 
% case_names = string(cases(:,1));
% h = gobjects(nCases,1);
% 
% for c = 1:nCases
%     thisColor = case_colors(c,:);
%     mean_vals = nan(1,nAlg);
% 
%     for k = 1:nAlg
%         vals = all_norm_score{c,k};
%         if isempty(vals)
%             continue
%         end
% 
%         xc = xCenters(k) + offsets(c);
%         x  = ones(size(vals))*xc;
% 
%         % swarm plot
%         s = swarmchart(x, vals, 20, ...
%             'MarkerEdgeColor', thisColor, ...
%             'MarkerFaceColor', thisColor, ...
%             'MarkerFaceAlpha', 0.35, ...
%             'MarkerEdgeAlpha', 0.6, ...
%             'XJitter','density', ...
%             'XJitterWidth',0.05);
%         hold on
% 
%         % legend handle 저장
%         if ~isgraphics(h(c))
%             h(c) = s;
%         end
% 
%         % mean marker
%         mu = mean(vals,'omitnan');
%         mean_vals(k) = mu;
% 
%         plot(xc, mu, 'd', ...
%             'MarkerSize',7, ...
%             'MarkerFaceColor',thisColor, ...
%             'MarkerEdgeColor','k', ...
%             'LineWidth',0.6);
%     end
% 
%     % baseline 제거 (CC-CE 0.75)
%     % mean_vals(1) = nan;
% 
%     % mean 연결선
%     valid = ~isnan(mean_vals);
%     plot(xCenters(valid)+offsets(c), mean_vals(valid), '-', ...
%         'Color',[thisColor 0.6], ...
%         'LineWidth',2);
% end
% 
% set(gca,'YScale','linear')
% xlim([0.5 nAlg+0.5])
% xticks(xCenters)
% xticklabels([])
% 
% ylabel('Standardized score shift')
% 
% grid on
% box on
% 
% legend(h, case_names,'Location','best')

%% Plot3
figure('Position',[1000 818 560 300])

t = tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

for sp = 1:2
    ax = nexttile;
    hold(ax, 'on')

    xCenters = 1:nAlg;
    % offsets  = [-0.18, 0, 0.18];   % case 3개니까 3개만 쓰는 게 맞음
    offsets  = [-0.21 -0.07 0.07 0.21]* 1.3;   % case 3개니까 3개만 쓰는 게 맞음
    % offsets2 = [-0.1 0.1];
    offsets2 = [-0.18, 0, 0.18];
    boxw     = 0.14;

    case_names = string(cases(:,1));
    h = gobjects(nCases,1);

    for c = 1:nCases
        % 아래 subplot에서는 med(2), small(3)만 표시
        if sp == 2 && ~ismember(c,[2 3 4])
            continue
        end

        thisColor = case_colors(c,:);
        mean_vals = nan(1,nAlg);

        for k = 1:nAlg
            vals = all_norm_score{c,k};
            if isempty(vals)
                continue
            end

            if sp == 1
                xc = xCenters(k) + offsets(c);
            else
                xc = xCenters(k) + offsets2(c-1);
            end
            x  = ones(size(vals)) * xc;

            % boxchart
            b = boxchart(ax, x, vals, ...
                'BoxWidth', boxw, ...
                'BoxFaceColor', thisColor, ...
                'BoxEdgeColor', thisColor, ...
                'MarkerColor', thisColor, ...
                'WhiskerLineColor', thisColor, ...
                'LineWidth', 1.2);

            b.BoxFaceAlpha = 0.12;
            b.MarkerStyle = 'none';
            b.JitterOutliers = 'off';

            if ~isgraphics(h(c))
                h(c) = b;
            end

            % swarm
            swarmchart(ax, x, vals, 18, ...
                'MarkerEdgeColor', thisColor, ...
                'MarkerFaceColor', thisColor, ...
                'MarkerFaceAlpha', 0.28, ...
                'MarkerEdgeAlpha', 0.45, ...
                'XJitter', 'density', ...
                'XJitterWidth', boxw * 0.55);

            % mean marker
            mu = mean(vals, 'omitnan');
            mean_vals(k) = mu;

            plot(ax, xc, mu, 'o', ...
                'MarkerSize', 6, ...
                'MarkerFaceColor', thisColor, ...
                'MarkerEdgeColor', 'k', ...
                'LineWidth', 0.2);
        end

        % mean line
        validMean = ~isnan(mean_vals);
        if sp == 1
            plot(ax, xCenters(validMean) + offsets(c), mean_vals(validMean), '-', ...
                'Color', [thisColor 0.5], ...
                'LineWidth', 4);
        else
            plot(ax, xCenters(validMean) + offsets2(c-1), mean_vals(validMean), '-', ...
            'Color', [thisColor 0.5], ...
            'LineWidth', 4);
        end
        set(gca,'FontName','Times New Roman','FontSize',18)
    end

    set(ax, 'YScale', 'linear')
    xlim(ax, [0.5 nAlg+0.5])
    xticks(ax, xCenters)
    xticklabels(ax, [])
    grid(ax, 'on')
    box(ax, 'on')

    validLegend = isgraphics(h);
    if sp == 1
    lgd = legend(ax, h(validLegend), case_names(validLegend), ...
        'Location', 'southeast', 'Interpreter', 'tex','FontSize',14, Orientation='horizontal');
    else
    lgd = legend(ax, h(validLegend), case_names(validLegend), ...
    'Location', 'southwest', 'Interpreter', 'tex','FontSize',14);
    end
    lgd.ItemTokenSize = [10 10];

    % if sp == 1
    %     % ylim(ax, [-0.15 0.15])
    %     ylim(ax, [-0.18 0.3])
    % else
    %     ylim(ax, [-0.03 0.02])
    % end

    if sp == 1
        ylim(ax, [-0.05 0.08])
    else
        ylim(ax, [-0.006 0.013])
    end
end

ylabel(t, 'Standardized score shift','fontsize',18,'fontname','times')
exportgraphics(gcf, "2_alpha_test_cdc.pdf","Resolution",300);