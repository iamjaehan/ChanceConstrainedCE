# =============================================================================
# Experiment 2: Information Acquisition Strategy Comparison
# n=3, r=1 (m=2), α=0.9 (fixed), σ_i ~ U(0,1) per trial
# γ ∈ {1.05, 1.2, 1.5, 2.0}
#
# For each (γ, trial):
#   1. Solve baseline CC-CE(α=0.9)
#   2. Select top-5 constraints by each strategy, set their σ→0, re-solve
#   3. Compare expected/realized cost and deviation metrics
#
# Strategies (5):
#   Baseline  — original CC-CE, no info acquisition
#   Random-5  — 5 constraints chosen at random
#   Top5-σ    — top 5 by uncertainty magnitude σ_{i(c)}
#   Top5-λ    — top 5 by dual variable λ_c* (shadow price)
#   Top5-IG   — top 5 by InfoGain_c = λ_c* · π_c · σ_{i(c)}  (paper Eq. 17)
# =============================================================================

using correlated
using Statistics, Random, Distributions, MAT, Printf

include("0_GameSetup.jl")
include("CorrBasedOptimizer.jl")
include("MC_ccce_alpha.jl")   # player_cost, alpha_to_z, sanitize_prob_vector,
                               # sample_joint_action_from_x
include("exp_utils.jl")        # simulate_realized, system_cost, evaluate_solution

# ── Experiment parameters ────────────────────────────────────────────────────
const E2_R      = 2
const E2_N      = 3
const E2_LAMBDA = ones(E2_N)
const E2_DELTA  = 1.0
const E2_ALPHA  = 0.9
const E2_GAMMAS = [1.3, 1.5, 1.8, 2.0]
const E2_K           = 5    # number of constraints to de-noise per strategy
const E2_SCALE_FULL  = 0.0  # full uncertainty removal
const E2_SCALE_HALF  = 0.5  # partial uncertainty removal

# ── Main experiment ──────────────────────────────────────────────────────────

function run_exp2(;
    n_trials   = 10,
    noise_runs = 200,
    seed       = 42,
    log_file   = "exp2_log.txt",
    mat_file   = "exp2_results.mat",
)
    rng    = MersenneTwister(seed)
    rows   = NamedTuple[]
    zalpha = alpha_to_z(E2_ALPHA)
    io     = open(log_file, "w")

    function lg(s)
        println(s); println(io, s); flush(io)
    end

    lg("=" ^ 70)
    lg("Experiment 2 | Information Acquisition Strategy Comparison")
    lg(@sprintf("n=%d  r=%d → m=%d  α=%.2f  k=%d constraints de-noised",
                E2_N, E2_R, 2^E2_R, E2_ALPHA, E2_K))
    lg(@sprintf("σ_i ~ U(0,1)  |  n_trials=%d  |  noise_runs=%d", n_trials, noise_runs))
    lg(@sprintf("γ_list = %s", string(E2_GAMMAS)))
    lg("=" ^ 70)

    for γ in E2_GAMMAS
        lg(@sprintf("\n══ γ = %.2f ══════════════════════════════════════════════════", γ))

        for t in 1:n_trials
            sigma = rand(rng, E2_N) * 100

            lg(@sprintf("\n  ─── Trial %d/%d   σ=[%s]", t, n_trials,
                        join([@sprintf("%.4f", s) for s in sigma], ", ")))

            # ── Baseline CC-CE solve ─────────────────────────────────────────
            res0 = SearchCorr(E2_R, E2_N, E2_LAMBDA, E2_DELTA;
                              zalpha = zalpha,
                              sigma  = sigma,
                              mult   = γ)

            # ── Helper: σ→scale*σ for selected constraints ───────────────────
            function CEScaleKeyList(ce_list, scale)
                Dict{Tuple{Int,Int,Int},Float64}(
                    (e.player, e.rec, e.dev) => scale for e in ce_list
                )
            end

            # ── Select constraint key sets per strategy ──────────────────────
            # (all strategies are derived from the baseline solution)
            # Each entry: (name, scale_keys)  — scale=FULL(0.0) or HALF(0.5)
            no_scale = Dict{Tuple{Int,Int,Int},Float64}()
            strategies = [
                ("Baseline",      no_scale),
                ("Random-5",      CEScaleKeyList(RandomKCE(res0, E2_K; sigma=sigma, rng=rng), E2_SCALE_FULL)),
                ("Top5-σ",        CEScaleKeyList(TopKCEBySigma(res0, E2_K; sigma=sigma),      E2_SCALE_FULL)),
                ("Top5-λ",        CEScaleKeyList(TopKCEByMu(res0, E2_K),                      E2_SCALE_FULL)),
                ("Top5-IG",       CEScaleKeyList(TopKCEByMuPSigma(res0, E2_K; sigma=sigma),   E2_SCALE_FULL)),
                ("Half-Random-5", CEScaleKeyList(RandomKCE(res0, E2_K; sigma=sigma, rng=rng), E2_SCALE_HALF)),
                ("Half-Top5-σ",   CEScaleKeyList(TopKCEBySigma(res0, E2_K; sigma=sigma),      E2_SCALE_HALF)),
                ("Half-Top5-λ",   CEScaleKeyList(TopKCEByMu(res0, E2_K),                      E2_SCALE_HALF)),
                ("Half-Top5-IG",  CEScaleKeyList(TopKCEByMuPSigma(res0, E2_K; sigma=sigma),   E2_SCALE_HALF)),
            ]

            for (name, scale_keys) in strategies

                res = isempty(scale_keys) ? res0 :
                      SearchCorr(E2_R, E2_N, E2_LAMBDA, E2_DELTA;
                                 zalpha           = zalpha,
                                 sigma            = sigma,
                                 mult             = γ,
                                 sigma_scale_keys = scale_keys)

                status_str = string(res.status)
                x  = sanitize_prob_vector(res.primals[1:res.l])
                ev = evaluate_solution(rng, x, res.C, res.m, res.n, sigma;
                                       noise_runs = noise_runs)

                lg(@sprintf("    [%-10s]  exp=%.3f  real=%.3f(±%.3f)  dev_rate=%.3f  mean_devs=%.3f  [%s]",
                            name,
                            ev.expected_cost, ev.realized_mean, ev.realized_std,
                            ev.deviation_rate, ev.mean_num_deviators,
                            status_str))

                push!(rows, (
                    gamma              = γ,
                    trial              = t,
                    method             = name,
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
    end

    close(io)

    # ── Save to MAT ──────────────────────────────────────────────────────────
    if !isempty(rows)
        sigma_mat = reduce(hcat, [r.sigma for r in rows])'   # (n_rows × E2_N)
        matwrite(mat_file, Dict(
            "gamma"              => [r.gamma              for r in rows],
            "trial"              => [r.trial              for r in rows],
            "method"             => [r.method             for r in rows],
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
# rows = run_exp2(n_trials = 10, noise_runs = 10)
