using Random
using Distributions
using Statistics

include("schedule.jl")
include("state.jl")
include("actions.jl")
include("costs.jl")
include("VQ_BruteRRCETensor.jl")
include("VQ_SearchCorrTensor.jl")
include("VQ_Fcfs.jl")

using .VQSchedule
using .VQState
using .VQActions
using .VQCosts
# ----------------------------
# Config / enums
# ----------------------------
@enum SolverMode begin
    GREEDY_CENTRALIZED
    AGG_ORACLE_FCFS
    CE_NAIVE          # uncertainty ignored in solver (sigma=0 in CE)
    CE_FULL           # uncertainty-aware CE (your full solver)
    RRCE_PNE          # brute RRCE over PNE
end

@enum SigmaMode begin
    SIGMA_ZERO        # all zeros
    SIGMA_SCALAR      # sigma = scalar for all airlines
    SIGMA_VECTOR      # sigma = provided vector (length = #airlines or #players)
end

Base.@kwdef mutable struct SimConfig
    csv_path::String = "schedule/flight_schedule_1h_5b.csv"
    params::SimParams = SimParams(2, [2,2])
    T_sim::Int = 16
    max_subset_size::Int = 1024

    # objective weights used in VQCosts.compute_costs
    lambda_fair::Float64 = 1.0
    rho_release::Float64 = 0.0

    # CE / RRCE params
    Δ::Float64 = 0.0
    α::Float64 = 0.9  # credibility
    zalpha::Float64 = quantile(Normal(), 0.9)

    # solver choice
    solver_mode::SolverMode = RRCE_PNE
    seed::Int = 1

    # --- Uncertainty knobs ---
    # (A) what the coordinator ASSUMES when solving
    coord_sigma_mode::SigmaMode = SIGMA_ZERO
    coord_sigma_scalar::Float64 = 0.0
    coord_sigma_vec::Vector{Float64} = Float64[]   # optional

    # (B) what actually happens in realized choice / deviation
    real_sigma_mode::SigmaMode = SIGMA_ZERO
    real_sigma_scalar::Float64 = 0.0
    real_sigma_vec::Vector{Float64} = Float64[]    # optional

    # deviation model on/off (applies to CE / RRCE recommendations)
    enable_deviation::Bool = true
end

# ----------------------------
# Logging structs
# ----------------------------
Base.@kwdef mutable struct EpochLog
    t::Int
    Q::Vector{Int}
    active_airlines::Vector{Int}
    backlog_by_airline::Dict{Int,Int}

    n_backlog_total::Int
    n_pushed::Int
    pushed_ids::Vector{Int}

    J_coord::Float64
    airline_cost_sum::Dict{Int,Float64}      # sum of costs for that airline (pax-weighted if your cost does that)
    airline_cost_avg::Dict{Int,Float64}      # avg per active flight (pax-weighted cost / count) unless you redefine

    solver_time_sec::Float64
    solver_detail::Dict{String,Any}
end

Base.@kwdef mutable struct SimResult
    cfg::SimConfig
    logs::Vector{EpochLog}
    total_time_sec::Float64

    # post metrics
    fairness_max_gap::Float64
    total_delay_unweighted::Float64
end

# ----------------------------
# FCFS helper
# ----------------------------
function select_fcfs(elig::Vector{Int}, flights, k::Int; rng::AbstractRNG=Random.default_rng())
    k <= 0 && return Int[]
    k = min(k, length(elig))
    k == 0 && return Int[]

    ord = sortperm(elig, by = i -> flights[i].sched_t)

    out = Int[]
    i = 1
    while i <= length(ord) && length(out) < k
        t0 = flights[elig[ord[i]]].sched_t
        j = i
        while j <= length(ord) && flights[elig[ord[j]]].sched_t == t0
            j += 1
        end
        block = elig[ord[i:j-1]]
        shuffle!(rng, block)
        append!(out, block)
        i = j
    end
    return out[1:k]
end

# ----------------------------
# Sigma builders
# ----------------------------
function build_sigma(cfg::SimConfig, mode::SigmaMode, scalar::Float64, vec::Vector{Float64}, n_players::Int)
    if mode == SIGMA_ZERO
        return zeros(n_players)
    elseif mode == SIGMA_SCALAR
        return fill(scalar, n_players)
    elseif mode == SIGMA_VECTOR
        length(vec) == n_players || error("SIGMA_VECTOR requires length(vec)==n_players. Got $(length(vec)) vs $n_players.")
        return vec
    else
        error("Unknown SigmaMode")
    end
end

# ----------------------------
# Epoch metrics (from your cost function)
# ----------------------------
function compute_epoch_metrics(state, pushed, flights, active_airlines, active_by_airline, t, params;
    lambda_fair::Float64, rho_release::Float64)

    info = VQCosts.compute_costs(
        state.Q, pushed, flights, active_airlines, active_by_airline, t, params;
        n_runways=params.n_runways,
        beta_queue=lambda_fair,
        rho_release=rho_release
    )

    # Build per-airline cost summaries (using info.C_air? depends on your compute_costs return)
    # If compute_costs returns airline-specific totals, use that.
    # Otherwise we can recompute per airline by filtering pushed/backlog etc.
    # Here I assume `info` includes `air_costs::Dict{Int,Float64}` or similar.
    # If not, adjust this part once you confirm the return fields.

    airline_sum = Dict{Int,Float64}()
    airline_avg = Dict{Int,Float64}()
    for aid in active_airlines
        n = length(active_by_airline[aid])
        # fallback if field doesn't exist: store NaN
        airline_sum[aid] = hasproperty(info, :J_air) ? info.J_air[aid] : NaN
        airline_avg[aid] = n > 0 ? airline_sum[aid] / n : NaN
    end

    return info.J_coord, airline_sum, airline_avg, info
end

# ----------------------------
# Solver implementations
# ----------------------------
function solve_epoch(cfg::SimConfig, rng::AbstractRNG,
    state, flights, active_airlines, active_by_airline,
    actions_by_player, joint_pushed, joint_choice, params)

    nP = length(active_airlines)
    solver_detail = Dict{String,Any}()

    # helper: greedy best_k by coordinator objective
    function best_k_by_Jcoord()
        best_k = 1
        best_cost = Inf
        for k in eachindex(joint_pushed)
            pushed = joint_pushed[k]
            info = VQCosts.compute_costs(
                state.Q, pushed, flights, active_airlines, active_by_airline, state.t, params;
                n_runways=params.n_runways,
                beta_queue=cfg.lambda_fair,
                rho_release=cfg.rho_release
            )
            if info.J_coord < best_cost
                best_cost = info.J_coord
                best_k = k
            end
        end
        return best_k
    end

    t0 = time()
    pushed_real = Int[]

    if cfg.solver_mode == GREEDY_CENTRALIZED
        k = best_k_by_Jcoord()
        pushed_real = joint_pushed[k]
        solver_detail["k"] = k

    elseif cfg.solver_mode == AGG_ORACLE_FCFS
        # 1) oracle aggregate quota from greedy best_k
        k = best_k_by_Jcoord()
        jc = joint_choice[k]

        quotas = Dict{Int,Int}()
        for p in 1:nP
            aid = active_airlines[p]
            aidx = jc[p]
            quotas[aid] = length(actions_by_player[p][aidx])
        end

        # 2) airline chooses actual flights by FCFS
        pushed_fcfs = Int[]
        for aid in active_airlines
            elig = active_by_airline[aid]
            q = quotas[aid]
            append!(pushed_fcfs, select_fcfs(elig, flights, q; rng=rng))
        end

        pushed_real = pushed_fcfs
        solver_detail["k_oracle"] = k
        solver_detail["quotas"] = quotas

    else
        # Build game tensors once for CE/RRCE
        game = build_epoch_game_tensors(state, flights, active_airlines, active_by_airline,
            actions_by_player, joint_pushed, joint_choice, params; n_runways=params.n_runways)

        # sigma used by coordinator (assumed uncertainty inside solver)
        sigma_coord = build_sigma(cfg, cfg.coord_sigma_mode, cfg.coord_sigma_scalar, cfg.coord_sigma_vec, size(game.C_air,1))

        # sigma used to generate realized deviation noise
        sigma_real  = build_sigma(cfg, cfg.real_sigma_mode,  cfg.real_sigma_scalar,  cfg.real_sigma_vec,  size(game.C_air,1))

        k_rec = 1
        z = nothing

        if cfg.solver_mode == CE_NAIVE
            # naive = same solver but with sigma_coord=0 and (optionally) zalpha=0
            res = SearchCorrTensor(game.C_air, game.joint_choice, game.choice_to_k, game.action_sizes;
                                   Δ=cfg.Δ, zalpha=cfg.zalpha, sigma=zeros(size(game.C_air,1)))
            z = res.z
            k_rec = sample_k(z)

        elseif cfg.solver_mode == CE_FULL
            res = SearchCorrTensor(game.C_air, game.joint_choice, game.choice_to_k, game.action_sizes;
                                   Δ=cfg.Δ, zalpha=cfg.zalpha, sigma=sigma_coord)
            z = res.z
            k_rec = sample_k(z)

        elseif cfg.solver_mode == RRCE_PNE
            pne_ks = find_pne_set(game.C_air, game.joint_choice, game.choice_to_k, game.action_sizes;
                                  tol=1e-3, zalpha=cfg.zalpha, sigma=sigma_coord)
            solver_detail["num_pne"] = length(pne_ks)

            if isempty(pne_ks)
                # robust fallback for MC: do nothing
                solver_detail["fallback"] = "no_pne_push_none"
                pushed_real = Int[]
                t1 = time()
                return pushed_real, (t1 - t0), solver_detail
            end

            rr = solve_rrce_over_pne(game.C_air, pne_ks, length(game.joint_pushed); Δ=cfg.Δ)
            z = pne_to_distribution(rr.pne_ks, rr.λ, length(game.joint_pushed))
            k_rec = sample_k(z)

        else
            error("Unknown solver_mode")
        end

        solver_detail["k_rec"] = k_rec

        # recommendation
        pushed_rec = game.joint_pushed[k_rec]

        # realized deviation model (optional)
        if cfg.enable_deviation
            C_air_noisy = copy(game.C_air)
            for i in 1:size(C_air_noisy,1)
                C_air_noisy[i, :] .+= sigma_real[i] .* randn(rng, size(C_air_noisy,2))
            end
            k_real = realized_choice_conditional_BR(
                C_air_noisy, game.joint_choice, game.choice_to_k, game.action_sizes, k_rec
            )
            solver_detail["k_real"] = k_real
            pushed_real = game.joint_pushed[k_real]
        else
            pushed_real = pushed_rec
        end
    end

    t1 = time()
    return pushed_real, (t1 - t0), solver_detail
end

# ----------------------------
# Main simulation
# ----------------------------
function run_simulation(cfg::SimConfig)::SimResult
    rng = MersenneTwister(cfg.seed)

    flights = VQSchedule.load_schedule(cfg.csv_path)
    state = init_state(cfg.params, length(flights))

    logs = EpochLog[]
    total_t0 = time()

    for step in 1:cfg.T_sim
        # active airlines
        active_by_airline = build_active_by_airline(state, flights)
        active_airlines = sort(collect(keys(active_by_airline)))

        if isempty(active_airlines)
            # log minimal
            push!(logs, EpochLog(
                t=state.t, Q=copy(state.Q),
                active_airlines=Int[],
                backlog_by_airline=Dict{Int,Int}(),
                n_backlog_total=0,
                n_pushed=0,
                pushed_ids=Int[],
                J_coord=0.0,
                airline_cost_sum=Dict{Int,Float64}(),
                airline_cost_avg=Dict{Int,Float64}(),
                solver_time_sec=0.0,
                solver_detail=Dict{String,Any}("note" => "no_active")
            ))
            VQCosts.evolve_state!(state, Int[], flights; mu=cfg.params.mu)
            continue
        end

        # action sets
        actions_by_player = Vector{Vector{Vector{Int}}}()
        for aid in active_airlines
            elig = active_by_airline[aid]
            acts = enumerate_actions_subset(elig; max_subset_size=cfg.max_subset_size)
            push!(actions_by_player, acts)
        end

        joint_pushed, joint_choice = enumerate_joint_actions(actions_by_player)

        # solve one epoch using selected solver_mode
        pushed, solver_time, solver_detail = solve_epoch(cfg, rng,
            state, flights, active_airlines, active_by_airline,
            actions_by_player, joint_pushed, joint_choice, cfg.params)

        # metrics before state evolves
        backlog_counts = Dict(aid => length(active_by_airline[aid]) for aid in active_airlines)
        n_backlog_total = sum(values(backlog_counts))

        J_coord, air_sum, air_avg, info = compute_epoch_metrics(
            state, pushed, flights, active_airlines, active_by_airline, state.t, cfg.params;
            lambda_fair=cfg.lambda_fair,
            rho_release=cfg.rho_release
        )

        push!(logs, EpochLog(
            t=state.t,
            Q=copy(state.Q),
            active_airlines=copy(active_airlines),
            backlog_by_airline=backlog_counts,
            n_backlog_total=n_backlog_total,
            n_pushed=length(pushed),
            pushed_ids=copy(pushed),
            J_coord=J_coord,
            airline_cost_sum=air_sum,
            airline_cost_avg=air_avg,
            solver_time_sec=solver_time,
            solver_detail=solver_detail
        ))

        # evolve
        VQCosts.evolve_state!(state, pushed, flights; mu=cfg.params.mu)
    end

    total_t1 = time()

    # ----------------------------
    # Post metrics
    # ----------------------------
    # Fairness: max gap over airlines of (choose one)
    # Option A: use airline_cost_avg (as proxy for u_i)
    # Option B: use airline_cost_sum
    # I’ll use avg by default.
    all_vals = Float64[]
    for lg in logs
        for (aid, v) in lg.airline_cost_avg
            isfinite(v) && push!(all_vals, v)
        end
    end
    fairness = isempty(all_vals) ? 0.0 : (maximum(all_vals) - minimum(all_vals))

    # Efficiency (unweighted): you said “delay only” for coordinator-side reporting.
    # Your J_coord includes -rho_release*|pushed| and other terms.
    # Here we just sum J_coord as-is. If you want pure Σ delay only, we need a dedicated output from compute_costs.
    total_obj = sum(lg.J_coord for lg in logs)

    return SimResult(
        cfg=cfg,
        logs=logs,
        total_time_sec=(total_t1-total_t0),
        fairness_max_gap=fairness,
        total_delay_unweighted=total_obj
    )
end

# ----------------------------
# Example usage
# ----------------------------
# cfg = SimConfig(
#     csv_path="schedule/flight_schedule_1h_5b.csv",
#     solver_mode=AGG_ORACLE_FCFS,
#     seed=1,
#     T_sim=16,

#     # uncertainty knobs
#     coord_sigma_mode=SIGMA_ZERO,
#     real_sigma_mode=SIGMA_ZERO,

#     enable_deviation=true
# )

# res = run_simulation(cfg)
# println("Done. total_time=$(res.total_time_sec) sec, fairness_max_gap=$(res.fairness_max_gap), total_obj=$(res.total_delay_unweighted)")
