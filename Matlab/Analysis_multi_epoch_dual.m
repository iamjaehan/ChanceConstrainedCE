% ============================================================
% plot_multi_epoch_compare_two_cases.m
% Two-case comparison: (algorithm) x (case) grouped boxcharts
%   - multi-epoch.csv
%   - multi-epoch-zero.csv
% Output: 5 groups (algorithms) x 2 boxes (cases)
% ============================================================

clear; close all; clc;

% ----------------------------
% Global plotting defaults (LaTeX + font size)
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
% Files (two cases)
% ----------------------------
fileA = "../mc_out/multi-epoch.csv";
fileB = "../mc_out/multi-epoch-zero.csv";

TA = readtable(fileA, 'Delimiter', ',', 'PreserveVariableNames', true);
TB = readtable(fileB, 'Delimiter', ',', 'PreserveVariableNames', true);
TA.Properties.VariableNames = lower(string(TA.Properties.VariableNames));
TB.Properties.VariableNames = lower(string(TB.Properties.VariableNames));

required = ["solver","total_obj"];
for c = required
    assert(ismember(c, string(TA.Properties.VariableNames)), "Missing column '%s' in %s", c, fileA);
    assert(ismember(c, string(TB.Properties.VariableNames)), "Missing column '%s' in %s", c, fileB);
end

TA.algorithm = string(TA.solver);
TB.algorithm = string(TB.solver);

TA.case = repmat("Base", height(TA), 1);   % <- label shown in legend
TB.case = repmat("Zero", height(TB), 1);

T = [TA(:, ["algorithm","case","total_obj"]); TB(:, ["algorithm","case","total_obj"])];

% ----------------------------
% Algorithm display + colors (same as your previous script)
% ----------------------------
global DISP COL
DISP = containers.Map();
% DISP("GREEDY_CENTRALIZED") = "CENT";
% DISP("AGG_ORACLE_FCFS")    = "FCFS";
% DISP("CE_FULL")            = "Full-CCCE";
% DISP("CE_NAIVE")           = "RRCE-Nominal";
% DISP("RRCE_PNE")           = "RRCE-CCCE";

COL = containers.Map();
COL("GREEDY_CENTRALIZED") = [0.2 0.2 0.2];
COL("AGG_ORACLE_FCFS")    = [0.49 0.18 0.56];
COL("CE_FULL")            = [0.85 0.33 0.10];
COL("CE_NAIVE")           = [0.93 0.69 0.13];
COL("RRCE_PNE")           = [0.0 0.45 0.74];

algOrderAll = ["GREEDY_CENTRALIZED","AGG_ORACLE_FCFS","CE_NAIVE","CE_FULL","RRCE_PNE"];
present = unique(T.algorithm, 'stable');
algOrder = algOrderAll(ismember(algOrderAll, present));
assert(~isempty(algOrder), "No known solvers found in the CSVs.");

% Display names for x-axis
dispNames = strings(size(algOrder));
for i = 1:numel(algOrder)
    key = string(algOrder(i));
    if isKey(DISP, key)
        dispNames(i) = string(DISP(key));
    else
        dispNames(i) = key;
    end
end

% Categorical x (algorithm groups)
T.alg_disp = strings(height(T),1);
for i = 1:numel(algOrder)
    key = string(algOrder(i));
    mask = (T.algorithm == key);
    T.alg_disp(mask) = dispNames(i);
end
T.alg_disp = categorical(T.alg_disp, dispNames, 'Ordinal', true);

% Categorical case order (two boxes per algorithm)
caseOrder = ["Base","Zero"];
T.case = categorical(string(T.case), caseOrder, 'Ordinal', true);

% ----------------------------
% Plot: grouped boxcharts (2 boxes per alg)
% ----------------------------
figure('Name', 'Total objective: Base vs Zero');
ax = axes(); hold(ax,'on');

for i = 1:numel(algOrder)
    alg = string(algOrder(i));
    Xi  = T(T.algorithm == alg, :);
    if height(Xi)==0, continue; end

    % draw (two boxes via GroupByColor)

    h = boxchart(ax, Xi.alg_disp, Xi.total_obj, ...
        'GroupByColor', Xi.case, ...
        'MarkerStyle','none', ...
        'BoxWidth', 0.65);


    % recolor the two cases using the algorithm base color
    baseColor = COL(alg);

    % make "Base" darker, "Zero" lighter (same hue family)
    cBase = baseColor;
    cZero = lighten(baseColor, 0.45);  % 0.45 -> 꽤 연하게

    % h order corresponds to categories(Xi.case) == ["Base","Zero"]
    % but be safe:
    caseCats = categories(Xi.case);
    
    cBase = [0.85 0.33 0.10];   % red (noise)
    cZero = [0.00 0.45 0.74];   % blue (no variance)
    
    for j = 1:numel(h)
    
        cc = string(caseCats{j});
    
        if cc == "Base"
            % ---- Noise present ----
            h(j).BoxFaceColor = cBase;
            h(j).BoxFaceAlpha = 0.30;
            h(j).BoxEdgeColor = cBase;
            h(j).WhiskerLineColor = cBase;
            h(j).LineWidth = 1.2;
    
        else
            % ---- Zero variance (deterministic) ----
            h(j).BoxFaceColor = cZero;
            h(j).BoxFaceAlpha = 0.30;
            h(j).BoxEdgeColor = cZero;
            h(j).WhiskerLineColor = cZero;
            h(j).LineWidth = 2.5;  
        end
    
    end

% ------------------------------------------------------------
% Overlay median as star markers
% ------------------------------------------------------------
algCats = categories(T.alg_disp);
xCenter = find(strcmp(algCats, string(Xi.alg_disp(1))));

medBase = median(Xi.total_obj(Xi.case=="Base"));
medZero = median(Xi.total_obj(Xi.case=="Zero"));

dx = 0.25;   % left/right offset inside group

plot(ax, xCenter - dx, medBase, ...
    'o', 'MarkerSize', 5, 'LineWidth', 1.6, ...
    'Color', cBase, 'HandleVisibility','off');

plot(ax, xCenter + dx, medZero, ...
    'o', 'MarkerSize', 5, 'LineWidth', 1.6, ...
    'Color', cZero, 'HandleVisibility','off');

end

% xlabel("Algorithm");
ylabel("Accumulated delay cost");
grid(ax,'on'); ax.XGrid = 'off';
ylim([220 320])

% separators between algorithm groups
nG = numel(categories(T.alg_disp));
for k = 1:(nG-1)
    xline(ax, k+0.5, ':', 'Color', [0.2 0.2 0.2], 'LineWidth', 1.8, 'HandleVisibility','off');
end

set(gcf,'Position',[1000 818 560 350])
legend(ax, ["$\sigma = 100$","$\sigma = 0$"], 'Location','northwest', 'Interpreter','latex', 'FontName','times');

% export (optional)
exportgraphics(gcf, "multi_epoch.pdf", "Resolution", 300);


% ----------------------------
% helper: lighten color toward white
% ----------------------------
function c2 = lighten(c, a)
    % a in [0,1]; 0=no change, 1=white
    c2 = c + a*(1 - c);
    c2 = min(max(c2,0),1);
end
