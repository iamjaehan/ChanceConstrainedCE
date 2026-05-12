# =============================================================================
# exp_utils.jl — shared helpers for Experiment 1 & 2
# Requires: 0_GameSetup.jl, CorrBasedOptimizer.jl, MC_ccce_alpha.jl
# =============================================================================

# ── Realized simulation ───────────────────────────────────────────────────────

"""
Given a recommended joint profile a_rec and distribution x, simulate each
agent's deviation decision using the paper's model (Eq. 5–6).

Deviation model:
  Agent i draws η_i ~ N(0, σ_i²) and evaluates the perturbed deviation margin:
    M̃_i(a_i, a_i'; z) = M̄_i(z) + η_i · π_i(a_i)  (paper Eq. 6)
  where M̄_i(z) = CalcPhi(follow) - CalcPhi(deviate) is the EXPECTED
  deviation margin averaged over opponents under z (NOT point-wise).

  Agent i deviates to the best expected action a_i' if:
    M̄_i + η_i · π_i > 0   (paper Eq. 7, strict)
  Ties (best expected deviation == expected follow cost) do NOT count as
  deviation, since following is equally optimal.

  Agents decide simultaneously, holding opponents fixed at a_rec for the
  realized cost calculation.
"""
function simulate_realized(a_rec::Tuple, x, C, m, n, sigma, rng; tol = 1e-8)
    x_f      = reshape(x, ntuple(_ -> m, n))
    a_real   = collect(a_rec)
    deviated = falses(n)

    for i in 1:n
        ai   = a_rec[i]
        η_i  = randn(rng) * sigma[i]
        π_ai = CalcMarginalP(i, ai, x_f, m, n)

        # Expected cost of following (avg over opponents under z | rec = ai)
        exp_follow = CalcPhi(i, ai, ai, x_f, m, n, C)

        # Find best deviation over all alternatives (not restricted to strictly better)
        best_exp_dev = Inf
        best_a       = -1
        for a_prime in 1:m
            a_prime == ai && continue
            ec = CalcPhi(i, ai, a_prime, x_f, m, n, C)
            if ec < best_exp_dev
                best_exp_dev = ec
                best_a       = a_prime
            end
        end

        # No alternative exists (m=1 edge case)
        best_a == -1 && continue

        # M̄_i = E[J_follow] - E[J_best_dev]  (≤ 0 for valid CE; noise can flip it)
        M_bar = exp_follow - best_exp_dev

        # Perturbed deviation condition: M̄_i + η_i · π_i > 0  (paper Eq. 6/7)
        # tol guard handles ties (M_bar + η_i·π_i ≈ 0 → not a deviation)
        if M_bar + η_i * π_ai > tol
            a_real[i]   = best_a
            deviated[i] = true
        end
    end

    return Tuple(a_real), deviated
end

function system_cost(a::Tuple, C, n)
    return sum(player_cost(i, a, C) for i in 1:n)
end

# ── Solution evaluator ────────────────────────────────────────────────────────

"""
Evaluate a CC-CE solution x.

Metrics:
  expected_cost      — E_{a~z}[∑_i J_i(a)]  (paper Eq. 24, unweighted)
  realized_mean/std  — system cost after agents respond to each recommendation
                       under noise (noise_runs Monte Carlo rollouts)
  deviation_rate     — fraction of rollouts where ≥1 agent deviated
  mean_num_deviators — average number of deviating agents per rollout
"""
function evaluate_solution(rng, x, C, m, n, sigma; noise_runs = 200)

    expected_cost = CalcJ(x, C, m, n)

    realized_vals  = Float64[]
    any_dev_flags  = Bool[]
    num_dev_counts = Int[]

    for _ in 1:noise_runs
        a_rec            = sample_joint_action_from_x(rng, x, m, n)
        a_real, deviated = simulate_realized(a_rec, x, C, m, n, sigma, rng)
        push!(realized_vals,   system_cost(a_real, C, n))
        push!(any_dev_flags,   any(deviated))
        push!(num_dev_counts,  count(deviated))
    end

    return (;
        expected_cost      = expected_cost,
        realized_mean      = mean(realized_vals),
        realized_std       = std(realized_vals),
        deviation_rate     = mean(any_dev_flags),
        mean_num_deviators = mean(num_dev_counts),
    )
end
