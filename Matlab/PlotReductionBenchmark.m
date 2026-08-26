% =============================================================================
% PlotReductionBenchmark.m
% Joint-action reduction benchmark for the vertiport CC-CE game.
%
% Data files (written by devel/exp_reduction_to_mat.jl):
%   ../exp_reduction_results.mat          uniform yield cost, all of F certified
%   ../exp_reduction_partial_results.mat  heterogeneous yield cost, ~29% certified
%
% Figures, per case:
%   5  Scaling   total solve time vs. n, log scale, all available methods
%   6  Opt gap   each restricted method vs. the full CC-CE optimum
%
% Both cases are drawn in one MATLAB session, since session startup dominates
% the runtime (the plotting itself is milliseconds).
% =============================================================================

clear; clc; close all;

% -- Global style (matches the other scripts in this folder) -------------------
set(groot, 'defaultTextInterpreter',          'tex');
set(groot, 'defaultAxesTickLabelInterpreter', 'tex');
set(groot, 'defaultLegendInterpreter',        'tex');
set(groot, 'defaultAxesFontSize',    18);
set(groot, 'defaultTextFontSize',    18);
set(groot, 'defaultLegendFontSize',  16);
set(groot, 'defaultAxesLineWidth',   1.2);
set(groot, 'defaultLineLineWidth',   2.0);
set(groot, 'defaultFigureColor',     'w');
set(groot, 'defaultAxesFontName',    'Times New Roman');
set(groot, 'defaultTextFontName',    'Times New Roman');

EXPORT = true;                       % write PDFs next to this script

% -- Cases to draw ------------------------------------------------------------
CASES = { ...
  struct('file','../exp_reduction_results.mat',         'tag','',         ...
         'sub','all F certified'), ...
  struct('file','../exp_reduction_partial_results.mat', 'tag','_partial', ...
         'sub','29% F certified') ...
};

% -- Palette ------------------------------------------------------------------
% Method hues, fixed order, shared with the benchmark write-up so a method is the
% same colour everywhere. Validated for colour-vision-deficiency separation.
C.full    = [0.165 0.471 0.839];     % #2a78d6
C.reduced = [0.922 0.408 0.204];     % #eb6834
C.hullX   = [0.106 0.686 0.478];     % #1baf7a
C.hullF   = [0.929 0.631 0.000];     % #eda100
C.hullD   = [0.761 0.216 0.604];     % #c2379a
INK       = [0.106 0.125 0.157];
MUTED     = [0.420 0.459 0.510];
GRID      = [0.843 0.855 0.878];


for ci = 1:numel(CASES)
    CS = CASES{ci};
    if ~isfile(CS.file)
        fprintf('skipping %s (not found)\n', CS.file); continue;
    end
    D = load(CS.file);
    n = double(D.n(:));

    M(1) = struct('name','Full CC-CE',            'color',C.full,   'mark','o','data',D.full);
    M(2) = struct('name','Reduced joint CC-CE',   'color',C.reduced,'mark','s','data',D.reduced);
    M(3) = struct('name','Hull of PNE / X',       'color',C.hullX,  'mark','^','data',D.hullX);
    M(4) = struct('name','Hull of PNE / F',       'color',C.hullF,  'mark','d','data',D.hullF);
    % Hull direct exists only where Proposition 1's certificate holds, so the
    % partial-certification file has no such field.
    if isfield(D, 'hullDirect')
        M(5) = struct('name','Hull of PNE / No search','color',C.hullD,'mark','v','data',D.hullDirect);
    end

    fprintf('\n=== %s ===\n', CS.sub);
    fprintf('r = %d (m = %d), gamma = %.1f, z_alpha = %.1f, sigma = %.1f\n', ...
            D.params.r, D.params.m, D.params.gamma, D.params.zalpha, D.params.sigma);

    % =========================================================================
    %% FIGURE 5 — Scaling: total solve time vs. n
    % =========================================================================
    f1 = figure('Position', [100 100 780 560]); hold on; box on;

    for k = 1:numel(M)
        y = double(M(k).data.total(:));
        ok = ~isnan(y);
        plot(n(ok), y(ok), '-', 'Color', M(k).color, 'Marker', M(k).mark, ...
             'MarkerSize', 9, 'MarkerFaceColor', M(k).color, ...
             'MarkerEdgeColor', 'w', 'DisplayName', M(k).name);
        xl = n(find(ok, 1, 'last'));  yl = y(find(ok, 1, 'last'));
        if yl >= 1, txt = sprintf('%.1f s', yl); else, txt = sprintf('%.0f ms', yl*1000); end
        text(xl + 0.12, yl, txt, 'Color', M(k).color, 'FontSize', 14, ...
             'VerticalAlignment', 'middle', 'FontWeight', 'bold');
    end

    set(gca, 'YScale','log', 'XTick',n, 'XGrid','on', 'YGrid','on', ...
             'GridColor',GRID, 'GridAlpha',1, 'XColor',MUTED, 'YColor',MUTED, 'Layer','top');
    xlim([min(n)-0.25, max(n)+0.75]);
    ylim([1e-3 1.5e3]);
    xlabel('Number of queues  n', 'Color', INK);
    ylabel('Total solve time  [s]', 'Color', INK);
    title({'Solve time comparison', CS.sub}, 'Color', INK, 'FontWeight', 'normal');
    legend('Location','northwest', 'Box','off', 'TextColor', INK);

    hold off;
    if EXPORT, exportgraphics(f1, ['5_reduction_scaling' CS.tag '.pdf'], 'ContentType','vector'); end

    % =========================================================================
    %% FIGURE 6 — Outcome comparison
    % =========================================================================
    % Solution cost each method reaches, grouped by n. Bars of equal height
    % within a group is the result: the formulations agree on the optimum.
    % A method that did not run at that n simply has no bar (NaN).
    f2 = figure('Position', [100 100 900 560]); hold on; box on;

    Y = nan(numel(n), numel(M));
    for k = 1:numel(M)
        Y(:, k) = double(M(k).data.score(:));
    end

    hb = bar(n, Y, 'grouped', 'EdgeColor', 'w', 'LineWidth', 1.2);
    for k = 1:numel(M)
        hb(k).FaceColor = M(k).color;
        hb(k).DisplayName = M(k).name;
    end

    set(gca, 'XTick',n, 'XGrid','off','YGrid','on', ...
             'GridColor',GRID,'GridAlpha',1,'XColor',MUTED,'YColor',MUTED,'Layer','top');
    xlim([min(n)-0.6, max(n)+0.6]);
    ylim([0, max(Y(:), [], 'omitnan') * 1.18]);
    xlabel('Number of queues  n', 'Color', INK);
    ylabel('Cost', 'Color', INK);
    title({'Solution cost', CS.sub}, 'Color', INK, 'FontWeight','normal');
    legend('Location','northwest', 'Box','off', 'TextColor', INK, 'NumColumns', 2);
    hold off;

    if EXPORT, exportgraphics(f2, ['6_reduction_cost' CS.tag '.pdf'], 'ContentType','vector'); end

    % =========================================================================
    %% Console summary
    % =========================================================================
    fprintf('\n%-5s %12s %6s', 'n', '|X|', '|F|');
    for k = 1:numel(M), fprintf(' %14s', M(k).name); end
    fprintf('   (total s)\n');
    for i = 1:numel(n)
        fprintf('%-5d %12d %6d', n(i), D.X(i), D.F(i));
        for k = 1:numel(M)
            t = double(M(k).data.total(i));
            if isnan(t), fprintf(' %14s', 'n/a'); else, fprintf(' %14.3f', t); end
        end
        fprintf('\n');
    end
    fprintf('%-5s %12s %6s', 'n', '', '');
    for k = 1:numel(M), fprintf(' %14s', M(k).name); end
    fprintf('   (objective)\n');
    for i = 1:numel(n)
        fprintf('%-5d %12s %6s', n(i), '', '');
        for k = 1:numel(M)
            s = double(M(k).data.score(i));
            if isnan(s), fprintf(' %14s', 'n/a'); else, fprintf(' %14.4f', s); end
        end
        fprintf('\n');
    end
    clear M
end
