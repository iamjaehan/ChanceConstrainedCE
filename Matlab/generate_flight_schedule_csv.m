function generate_flight_schedule_csv( ...
    csv_path, ...
    T_total_min, ...      % total horizon in minutes (e.g., 60)
    delta_t, ...          % time step (e.g., 1)
    n_airlines, ...
    n_runways, ...
    avg_dep_rate, ...     % average departures per minute
    pax_mean, pax_std ... % passenger distribution
)
% ---------------------------------------------------------
% Generate synthetic flight departure schedule and export CSV
% ---------------------------------------------------------

rng(1); % reproducibility

T = T_total_min / delta_t;   % number of epochs
flight_id = 1;

rows = {};

for t = 0:(T-1)

    % number of departures scheduled at time t
    % Poisson is standard for departure modeling
    n_dep = poissrnd(avg_dep_rate * delta_t);

    for k = 1:n_dep

        airline_id = randi(n_airlines);

        runway = randi(n_runways);

        % passenger count (truncate at >=10)
        pax = max(10, round(normrnd(pax_mean(airline_id), pax_std)));

        % ready time: allow small randomness (can be = sched_t)
        ready_offset = randi([0, 3]);  % up to 3 min earlier
        ready_t = max(0, t - ready_offset);

        rows(end+1, :) = { ...
            sprintf("F%04d", flight_id), ...
            airline_id, ...
            t, ...
            ready_t, ...
            runway, ...
            pax ...
        };

        flight_id = flight_id + 1;
    end
end

% Convert to table
schedule_tbl = cell2table(rows, ...
    'VariableNames', {'flight_id','airline_id','sched_t','ready_t','runway','pax'});

% Write CSV
writetable(schedule_tbl, csv_path);

fprintf("Generated %d flights, saved to %s\n", height(schedule_tbl), csv_path);
end
