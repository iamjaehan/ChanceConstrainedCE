rows = run_mc_alpha_experiment(
           r = 2,                       # number of routes per player
           n = 3,                       # number of players
           λ = ones(3),                 # congestion weights
           Δ = 1.0,                   # penalty scale

           alpha_list = [0.5, 0.75, 0.9, 0.95, 0.99],

           sigma_max = 20,           # sigma randomization range
           mc_runs = 5,               # number of MC games
           noise_runs = 100,            # rollouts per game

           seed = 2,

           include_ne = true,
           ne_pick_mode = :random
       )
       
save_mc_alpha_rows_mat(rows, "mc_results_ccce_alpha_big.mat")