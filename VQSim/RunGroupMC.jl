overall_seed = rand(1:10000,1)[1]

### Test case 1 ### 6 / 0 / 0.9
cfgE = MCEpochConfig(
    csv_path = "schedule/flight_schedule_1h_5b.csv",
    params = VQState.SimParams(2, [2,2]),
    max_subset_size = 1024,
    B0_total = 6,
    Q0_runway = [3, 4],
    t_epoch = 0,
    lateness_mean = 0.0,
    lateness_std = 10.0,
    Δ = 1e12,
    lambda_fair = 1.0,
    rho_release = 0.0,
    enable_deviation = true,
    alpha = 0.9,
    coord_sigma_mode = SIGMA_SCALAR,
    coord_sigma_scalar = 0,
    coord_sigma_vec = Float64[],
    real_sigma_mode = SIGMA_SCALAR,
    real_sigma_scalar = 0,
    real_sigma_vec = Float64[],
    N_mc = 100,
    base_seed = overall_seed,
    solver_modes = [GREEDY_CENTRALIZED, AGG_ORACLE_FCFS, CE_FULL, CE_NAIVE, RRCE_PNE]
)

println("Running test case 1")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_6a_0s_90c.csv")

### Test case 2 ### 5 / 0 / 0.9
cfgE.B0_total = 5
println("Running test case 2")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_5a_0s_90c.csv")

### Test case 3 ### 4 / 0 / 0.9
cfgE.B0_total = 4
println("Running test case 3")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_4a_0s_90c.csv")

### Test case 4 ### 7 / 0 / 0.9
cfgE.B0_total = 7
println("Running test case 4")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_7a_0s_90c.csv")

## Test case 5 ### 8 / 0 / 0.9
cfgE.B0_total = 8
println("Running test case 5")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_8a_0s_90c.csv")

### Test case 6 ### 6 / 5 / 0.9
cfgE.B0_total = 6
cfgE.coord_sigma_scalar = 5
cfgE.real_sigma_scalar = 5
println("Running test case 6")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_6a_5s_90c.csv")

### Test case 7 ### 6 / 15 / 0.9
cfgE.coord_sigma_scalar = 20
cfgE.real_sigma_scalar = 20
println("Running test case 7")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_6a_20s_90c.csv")

### Test case 8 ### 6 / 30 / 0.9
cfgE.coord_sigma_scalar = 45
cfgE.real_sigma_scalar = 45
println("Running test case 8")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_6a_45s_90c.csv")

### Test case 9 ### 6 / 15 / 0.5
cfgE.coord_sigma_scalar = 20
cfgE.real_sigma_scalar = 20
cfgE.alpha = 0.5
println("Running test case 9")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_6a_20s_50c.csv")

### Test case 10 ### 6 / 15 / 0.75
cfgE.alpha = 0.75
println("Running test case 10")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_6a_20s_75c.csv")

### Test case 11 ### 6 / 15 / 0.95
cfgE.alpha = 0.95
println("Running test case 11")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_6a_20s_95c.csv")

### Test case 12 ### 6 / 15 / 0.99
cfgE.alpha = 0.99
println("Running test case 12")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_6a_20s_99c.csv")

### Test case 13 ### 6 / 15 / 0.99
cfgE.alpha = 0.3
println("Running test case 13")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_6a_20s_30c.csv")