overall_seed = rand(1:10000,1)[1]

### Test case 1 ### 6 / 0 / 0.9
cfgE = MCEpochConfig(
    csv_path = "schedule/flight_schedule_1h_5b.csv",
    params = VQState.SimParams(2, [2,2]),
    max_subset_size = 1024,
    B0_total = 10,
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
    solver_modes = [RRCE_PNE]
)

println("Running test case 1")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_10a_0s_90c.csv")

### Test case 2 ### 11 / 0 / 0.9
cfgE.B0_total = 11
println("Running test case 2")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_11a_0s_90c.csv")

### Test case 3 ### 12 / 0 / 0.9
cfgE.B0_total = 12
println("Running test case 3")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_12a_0s_90c.csv")

### Test case 4 ### 13 / 0 / 0.9
cfgE.B0_total = 13
println("Running test case 4")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_13a_0s_90c.csv")

### Test case 5 ### 14 / 0 / 0.9
cfgE.B0_total = 14
println("Running test case 5")
df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_14a_0s_90c.csv")