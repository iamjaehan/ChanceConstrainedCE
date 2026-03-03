% csv_path = "../schedule/flight_schedule_1h.csv";
csv_path = "../schedule/flight_schedule_1h_5b.csv";

T_total_min = 60;
delta_t = 1;

n_airlines = 5;
n_runways = 2;

avg_dep_rate = 1.05;  % per minute 

% pax_mean = [150, 120, 90, 110, 100];  % airline별 평균 pax
% pax_std  = 20;
pax_class_ratio = [0.3 0.3 0.4]; % Heavy / Mid / Low

generate_flight_schedule_csv( ...
    csv_path, ...
    T_total_min, delta_t, ...
    n_airlines, n_runways, ...
    avg_dep_rate, ...
    pax_class_ratio ...
);


%%
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

b(1).FaceColor = [0.85 0.2 0.2];   % red for runway 1
b(2).FaceColor = [0.2 0.2 0.85];   % blue for runway 2

xlabel('Time (min)');
ylabel('Number of scheduled flights');
legend('Runway 1','Runway 2','location','best');
grid on;



