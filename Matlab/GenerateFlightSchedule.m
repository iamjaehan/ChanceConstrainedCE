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
