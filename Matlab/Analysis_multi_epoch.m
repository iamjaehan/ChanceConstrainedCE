% ============================================================
% plot_multi_epoch_totalobj_boxchart.m
% Multi-epoch summary: boxchart of total_obj by solver
% Input: multi-epoch.csv  (summary file from MC_test.jl)
% ============================================================

clear; close all; clc;

% ----------------------------
% Global plotting defaults (LaTeX + font size)  [same as yours]
% ----------------------------
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

set(groot, 'defaultAxesFontSize', 18);
set(groot, 'defaultTextFontSize', 18);
set(groot, 'defaultLegendFontSize', 18);
set(groot, 'defaultTextFontWeight', 'normal');
set(groot, 'defaultAxesFontWeight', 'normal');

set(groot, 'defaultAxesLineWidth', 1.2);
set(groot, 'defaultLineLineWidth', 2.0);
set(groot, 'defaultFigureColor', 'w');

% ----------------------------
% Load data
% ----------------------------
csvPath = "../mc_out/multi-epoch.csv";     % <- change if needed
% csvPath = "../mc_out/multi-epoch-zero.csv";     % <- change if needed
T = readtable(csvPath, 'Delimiter', ',', 'PreserveVariableNames', true);
T.Properties.VariableNames = lower(string(T.Properties.VariableNames));

required = ["solver","total_obj"];
for c = required
    assert(ismember(c, string(T.Properties.VariableNames)), ...
        "Missing column '%s' in %s", c, csvPath);
end

T.algorithm = string(T.solver);

% ----------------------------
% Algorithm display + colors (same as your previous script)
% ----------------------------
global DISP COL
DISP = containers.Map();
DISP("GREEDY_CENTRALIZED") = "CENT";
DISP("AGG_ORACLE_FCFS")    = "FCFS";
DISP("CE_FULL")            = "Full-CCCE";
DISP("CE_NAIVE")           = "RRCE-Nominal";
DISP("RRCE_PNE")           = "RRCE-CCCE";

COL = containers.Map();
COL("GREEDY_CENTRALIZED") = [0.2 0.2 0.2];
COL("AGG_ORACLE_FCFS")    = [0.49 0.18 0.56];
COL("CE_FULL")            = [0.85 0.33 0.10];
COL("CE_NAIVE")           = [0.93 0.69 0.13];
COL("RRCE_PNE")           = [0.0 0.45 0.74];

% Choose plotting order (keep only those present)
algOrderAll = ["GREEDY_CENTRALIZED","AGG_ORACLE_FCFS","CE_NAIVE","CE_FULL","RRCE_PNE"];
present = unique(T.algorithm, 'stable');
algOrder = algOrderAll(ismember(algOrderAll, present));

assert(~isempty(algOrder), "No known solvers found in the CSV.");

% Make x categories = display names (so x-axis looks clean)
dispNames = strings(size(algOrder));
for i = 1:numel(algOrder)
    key = string(algOrder(i));
    if isKey(DISP, key)
        dispNames(i) = string(DISP(key));
    else
        dispNames(i) = key;
    end
end
xCats = categorical(dispNames, dispNames, 'Ordinal', true);

% ----------------------------
% Plot: one boxchart per algorithm (so each can have its own color)
% ----------------------------
figure('Name', 'Total objective by solver');
ax = axes(); hold(ax,'on');

for i = 1:numel(algOrder)
    alg = string(algOrder(i));
    Xi  = T(T.algorithm == alg, :);

    % x values: a constant categorical for this algorithm (display name)
    x = repmat(xCats(i), height(Xi), 1);

    h = boxchart(ax, x, Xi.total_obj, ...
        'MarkerStyle','none', ...
        'BoxWidth', 0.55);

    % fixed color
    if isKey(COL, alg)
        c = COL(alg);
        try h.BoxFaceColor = c; end
        try h.BoxEdgeColor = c; end
        try h.WhiskerLineColor = c; end
        try h.MedianLineColor = c; end
    end
end

xlabel("Algorithm");
ylabel("Total coordinator cost [min]");
% title("Multi-epoch total objective comparison");
grid(ax,'on');
ax.XGrid = 'off';

% Optional: separators between categories (same style as yours)
for k = 1:(numel(categories(xCats))-1)
    xline(ax, k+0.5, ':', ...
        'Color', [0.2 0.2 0.2], ...
        'LineWidth', 1.8, ...
        'HandleVisibility','off');
end

% Optional: show sample sizes above each box
% ytop = max(T.total_obj) * 1.03;
% for i = 1:numel(algOrder)
%     alg = string(algOrder(i));
%     n = sum(T.algorithm == alg);
%     text(ax, i, ytop, sprintf("$n=%d$", n), ...
%         'HorizontalAlignment','center', 'VerticalAlignment','bottom');
% end

% export (optional)
% exportgraphics(gcf, "multi_epoch_totalobj_boxchart.pdf", "Resolution", 300);
