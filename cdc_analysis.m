load mc_results_ccce.mat

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

%%
method = string(method);
score = score(:);
mc_iter = mc_iter(:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% method labeling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

method_label = strings(size(method));

method_label(method=="baseline") = "CE";
method_label(method=="mu") = "CE - top 5 (\mu)";
method_label(method=="mu_sigma") = "CE - top 5 (\mu\sigma)";
method_label(method=="random") = "CE - top 5 (random)";
method_label(method=="ne") = "NE";

keep = method_label ~= "";
method_label = method_label(keep);
score = score(keep);
mc_iter = mc_iter(keep);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plotting order
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

order = [
    "NE"
    "CE"
    "CE - top 5 (\mu)"
    "CE - top 5 (\mu\sigma)"
    "CE - top 5 (random)"
];

method_label = categorical(method_label, order, 'Ordinal', true);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% normalize by CE score per MC run
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

score_norm = zeros(size(score));

unique_iter = unique(mc_iter);

for k = 1:length(unique_iter)

    idx = mc_iter == unique_iter(k);

    ce_score = score(idx & method_label=="CE");

    score_norm(idx) = score(idx) ./ ce_score;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(1);
clf
boxchart(method_label, score_norm)
hold on

means = zeros(numel(order),1);

for i = 1:numel(order)
    means(i) = mean(score_norm(method_label==order(i)));
end

plot(1:numel(order), means, 'ko', 'MarkerSize', 5, 'LineWidth', 1.5)

ylabel('Normalized system cost','FontSize',18)
% xlabel('Method')
% title('Monte Carlo system cost comparison (normalized by CE)')
grid on
set(gca,'XTick',[])
hold off
ylim([0.6 1.8])
set(gcf, 'Position', [1000 818 560  300]);
exportgraphics(gcf, "1_sigma_test_cdc.pdf","Resolution",300);

%%
figure(2);
clf

% NE 제거
mask = method_label ~= "NE";
method_label_f = method_label(mask);
score_norm_f   = score_norm(mask);

% 안 쓰는 categorical category 제거
method_label_f = removecats(method_label_f);

boxchart(method_label_f, score_norm_f)
hold on

order_f = order(order ~= "NE");
means = zeros(numel(order_f),1);

for i = 1:numel(order_f)
    means(i) = mean(score_norm_f(method_label_f == order_f(i)));
end

plot(1:numel(order_f), means, 'ko', 'MarkerSize', 5, 'LineWidth', 1.5)

ylabel('Normalized system cost','FontSize',18)
grid on
set(gca,'XTick',[])
ylim([0.7 1.1])

set(gcf, 'Position', [1000 818 560 220]);
exportgraphics(gcf, "1_sigma_test_cdc.pdf","Resolution",300);

hold off