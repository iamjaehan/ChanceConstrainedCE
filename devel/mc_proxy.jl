mc_runs = 50

rows = run_mc_alpha_experiment(
           r = 2,                       # number of routes per player
           n = 3,                       # number of players
           λ = ones(3),                 # congestion weights
           Δ = 1.0,                   # penalty scale

           alpha_list = [0, 0.75, 0.9, 0.95, 0.99],

           sigma_max = 50,           # sigma randomization range
           mc_runs = mc_runs,               # number of MC games
           noise_runs = 100,            # rollouts per game

           seed = 42,

           include_ne = true,
           ne_pick_mode = :random,
           mult = 1.5
       )
       
save_mc_alpha_rows_mat(rows, "mc_results_ccce_alpha_big.mat")


rows = run_mc_alpha_experiment(
           r = 2,                       # number of routes per player
           n = 3,                       # number of players
           λ = ones(3),                 # congestion weights
           Δ = 1.0,                   # penalty scale

           alpha_list = [0, 0.75, 0.9, 0.95, 0.99],

           sigma_max = 10,           # sigma randomization range
           mc_runs = mc_runs,               # number of MC games
           noise_runs = 100,            # rollouts per game

           seed = 42,

           include_ne = true,
           ne_pick_mode = :random,
           mult = 1.1
       )
       
save_mc_alpha_rows_mat(rows, "mc_results_ccce_alpha_med.mat")

rows = run_mc_alpha_experiment(
           r = 2,                       # number of routes per player
           n = 3,                       # number of players
           λ = ones(3),                 # congestion weights
           Δ = 1.0,                   # penalty scale

           alpha_list = [0, 0.75, 0.9, 0.95, 0.99],

           sigma_max = 5,           # sigma randomization range
           mc_runs = mc_runs,               # number of MC games
           noise_runs = 100,            # rollouts per game

           seed = 42,

           include_ne = true,
           ne_pick_mode = :random,
           mult = 1.05
       )
       
save_mc_alpha_rows_mat(rows, "mc_results_ccce_alpha_small.mat")

rows = run_mc_alpha_experiment(
           r = 2,                       # number of routes per player
           n = 3,                       # number of players
           λ = ones(3),                 # congestion weights
           Δ = 1.0,                   # penalty scale

           alpha_list = [0, 0.75, 0.9, 0.95, 0.99],

           sigma_max = 3,           # sigma randomization range
           mc_runs = mc_runs,               # number of MC games
           noise_runs = 100,            # rollouts per game

           seed = 42,

           include_ne = true,
           ne_pick_mode = :random,
           mult = 1.03
       )
       
save_mc_alpha_rows_mat(rows, "mc_results_ccce_alpha_usmall.mat")

println("TEST DONE")