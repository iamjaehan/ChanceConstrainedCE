csv_path = "../schedule/flight_schedule_1h.csv";

T_total_min = 60;
delta_t = 1;

n_airlines = 3;
n_runways = 2;

avg_dep_rate = 0.6;  % 평균 분당 0.6편 → 약 36편/시간

pax_mean = [150, 120, 90];  % airline별 평균 pax
pax_std  = 20;

generate_flight_schedule_csv( ...
    csv_path, ...
    T_total_min, delta_t, ...
    n_airlines, n_runways, ...
    avg_dep_rate, ...
    pax_mean, pax_std ...
);
