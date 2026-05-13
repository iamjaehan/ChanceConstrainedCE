# =============================================================================
# Experiment 1: NaiveCE vs CC-CE (alpha sweep)
# n=3, r=1 (m=2), γ=1.1, σ_i ~ U(0,1) per trial
# Methods: NaiveCE (α=0), CC-CE (α ∈ {0.66, 0.85, 0.90, 0.95})
# Metrics: expected cost, realized cost (with agent deviation), deviation rate,
#          mean number of deviating agents
# =============================================================================

using correlated
using Statistics, Random, Distributions, MAT, Printf

include("0_GameSetup.jl")
include("CorrBasedOptimizer.jl")
include("MC_ccce_alpha.jl")   # player_cost, alpha_to_z, sanitize_prob_vector,
                               # sample_joint_action_from_x
include("exp_utils.jl")        # simulate_realized, system_cost, evaluate_solution

# ── Experiment parameters ────────────────────────────────────────────────────
const E1_R      = 2            # r=1 → m = 2^1 = 2 actions per agent
const E1_N      = 3
const E1_GAMMA  = 2.0          # congestion multiplier γ
const E1_LAMBDA = ones(E1_N)
const E1_DELTA  = 1.0
const E1_ALPHAS = [0.0, 0.66, 0.80, 0.90, 0.95, 0.99]

# ── Main experiment ──────────────────────────────────────────────────────────

function run_exp1(;
    n_trials   = 10,
    noise_runs = 200,
    seed       = 42,
    log_file   = "exp1_log.txt",
    mat_file   = "exp1_results.mat",
)
    rng  = MersenneTwister(seed)
    rows = NamedTuple[]
    io   = open(log_file, "w")

    function lg(s)
        println(s)
        println(io, s)
        flush(io)
    end

    lg("=" ^ 70)
    lg("Experiment 1 | NaiveCE vs CC-CE (alpha sweep)")
    lg(@sprintf("n=%d  r=%d → m=%d actions  γ=%.2f", E1_N, E1_R, 2^E1_R, E1_GAMMA))
    lg(@sprintf("σ_i ~ U(0,1)  |  n_trials=%d  |  noise_runs=%d", n_trials, noise_runs))
    lg(@sprintf("alpha_list = %s", string(E1_ALPHAS)))
    lg("=" ^ 70)

    for t in 1:n_trials

        sigma = rand(rng, E1_N) * 100   # σ_i ~ U(0,1) independently per trial

        lg(@sprintf("\n─── Trial %d / %d ───────────────────────────────────", t, n_trials))
        lg(@sprintf("σ = [%s]", join([@sprintf("%.4f", s) for s in sigma], ", ")))

        for α in E1_ALPHAS
            label  = α == 0.0 ? "NaiveCE     " : @sprintf("CC-CE(α=%.2f)", α)
            zalpha = alpha_to_z(α)

            res = SearchCorr(E1_R, E1_N, E1_LAMBDA, E1_DELTA;
                             zalpha = zalpha,
                             sigma  = sigma,
                             mult   = E1_GAMMA)

            status_str = string(res.status)

            if !occursin("Solved", status_str) && status_str != "OPTIMAL"
                lg(@sprintf("  [%s]  SOLVER: %s — skipped", label, status_str))
                push!(rows, (
                    trial              = t,
                    method             = label,
                    alpha              = Float64(α),
                    sigma              = copy(sigma),
                    expected_cost      = NaN,
                    realized_mean      = NaN,
                    realized_std       = NaN,
                    deviation_rate     = NaN,
                    mean_num_deviators = NaN,
                    solver_status      = status_str,
                ))
                continue
            end

            x  = sanitize_prob_vector(res.primals[1:res.l])
            ev = evaluate_solution(rng, x, res.C, res.m, res.n, sigma;
                                   noise_runs = noise_runs)

            lg(@sprintf("  [%s]  (solver: %s)", label, status_str))
            lg(@sprintf("    expected_cost      = %8.3f", ev.expected_cost))
            lg(@sprintf("    realized_cost_mean = %8.3f  (±%.3f)",
                        ev.realized_mean, ev.realized_std))
            lg(@sprintf("    deviation_rate     = %8.4f  (frac. rollouts w/ ≥1 dev)",
                        ev.deviation_rate))
            lg(@sprintf("    mean_num_deviators = %8.4f  (avg # agents deviating/rollout)",
                        ev.mean_num_deviators))

            push!(rows, (
                trial              = t,
                method             = label,
                alpha              = Float64(α),
                sigma              = copy(sigma),
                expected_cost      = ev.expected_cost,
                realized_mean      = ev.realized_mean,
                realized_std       = ev.realized_std,
                deviation_rate     = ev.deviation_rate,
                mean_num_deviators = ev.mean_num_deviators,
                solver_status      = status_str,
            ))
        end
    end

    close(io)

    # ── Save to MAT ──────────────────────────────────────────────────────────
    if !isempty(rows)
        sigma_mat = reduce(hcat, [r.sigma for r in rows])'   # (n_rows × E1_N)
        matwrite(mat_file, Dict(
            "trial"              => [r.trial              for r in rows],
            "method"             => [r.method             for r in rows],
            "alpha"              => [r.alpha              for r in rows],
            "expected_cost"      => [r.expected_cost      for r in rows],
            "realized_mean"      => [r.realized_mean      for r in rows],
            "realized_std"       => [r.realized_std       for r in rows],
            "deviation_rate"     => [r.deviation_rate     for r in rows],
            "mean_num_deviators" => [r.mean_num_deviators for r in rows],
            "solver_status"      => [r.solver_status      for r in rows],
            "sigma_mat"          => sigma_mat,
        ))
        println("Saved → $mat_file  |  Log → $log_file")
    end

    return rows
end

# ── Entry point ──────────────────────────────────────────────────────────────
# rows = run_exp1(n_trials = 10, noise_runs = 10)
