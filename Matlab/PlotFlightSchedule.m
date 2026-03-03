%%
figure(1)
clf

% Load schedule
tbl = readtable(csv_path);

% sched_t 기준으로 departure count 계산
T_max = max(tbl.sched_t);
dep_count = zeros(T_max+1,1);

for t = 0:T_max
    dep_count(t+1) = sum(tbl.sched_t == t);
end


epoch_len = 4;   % 4분 epoch

% Load schedule
tbl = readtable(csv_path);

T_max = max(tbl.sched_t);
n_epoch = ceil((T_max+1)/epoch_len);

% runway별 dep count 저장
dep_epoch_r1 = zeros(n_epoch,1);
dep_epoch_r2 = zeros(n_epoch,1);

for e = 1:n_epoch
    t_start = (e-1)*epoch_len;
    t_end   = t_start + epoch_len - 1;

    idx_epoch = tbl.sched_t >= t_start & tbl.sched_t <= t_end;

    dep_epoch_r1(e) = sum(idx_epoch & tbl.runway == 1);
    dep_epoch_r2(e) = sum(idx_epoch & tbl.runway == 2);
end

epoch_time = (0:n_epoch-1)*epoch_len;

% Plot
figure;
b = bar(epoch_time, [dep_epoch_r1 dep_epoch_r2], 'grouped');

b(1).FaceColor = [0.8 0.35 0.35];
b(2).FaceColor = [0.35 0.45 0.75];

xlabel('Time [minutes]');
ylabel('Departures per epoch');
legend('Runway 1','Runway 2','location','northwest');
grid on;
ylim([0 7])

set(gcf,'Position',[1000 818 560 350])

exportgraphics(gcf, "scenario_departure.pdf","Resolution",300);