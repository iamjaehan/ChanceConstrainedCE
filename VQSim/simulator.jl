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
    params::VQState.SimParams = VQState.SimParams(2, [2,2])
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

    airline_cost_sum_unw::Dict{Int,Float64} # unweighted sum (J_air_sum_unw)
    airline_cost_avg_unw::Dict{Int,Float64} # unweighted sum (J_air_avg_unw)

    solver_time_sec::Float64
    solver_detail::Dict{String,Any}
end

Base.@kwdef mutable struct SimResult
    cfg::SimConfig
    logs::Vector{EpochLog}
    total_time_sec::Float64

    # post metrics
    fairness_max_gap::Float64          # global max-min over ALL epoch airline avg_unw (legacy-style)
    total_coord_obj::Float64           # sum of J_coord (name fixed)

    # NEW: more interpretable summaries
    fairness_gap_ep_mean::Float64      # mean over epochs of (max-min across airlines) using avg_unw
    fairness_gap_ep_max::Float64       # max over epochs of (max-min across airlines) using avg_unw
    fair_feasible_rate::Float64        # fraction of epochs where solver_detail["fair_feasible"] == true (if present)
    avg_num_pne::Float64               # average num_pne over epochs (RRCE only; NaN if none)
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

    airline_sum = Dict{Int,Float64}()       # weighted
    airline_avg = Dict{Int,Float64}()       # weighted avg

    airline_sum_unw = Dict{Int,Float64}()   # unweighted sum
    airline_avg_unw = Dict{Int,Float64}()   # unweighted avg (fairness)

    for aid in active_airlines
        n = length(active_by_airline[aid])

        # weighted airline objective (IC / deviation reference)
        airline_sum[aid] = hasproperty(info, :J_air) ? info.J_air[aid] : NaN
        airline_avg[aid] = n > 0 ? airline_sum[aid] / n : NaN

        # unweighted metrics for fairness
        airline_sum_unw[aid] = hasproperty(info, :J_air_sum_unw) ? info.J_air_sum_unw[aid] : NaN
        airline_avg_unw[aid] = hasproperty(info, :J_air_avg_unw) ? info.J_air_avg_unw[aid] : NaN
    end

    return info.J_coord, airline_sum, airline_avg, airline_sum_unw, airline_avg_unw, info
end

# Pick k that minimizes c_coord subject to fairness-gap <= Δ.
# fairness-gap(k) := max_i C_air_fair[i,k] - min_i C_air_fair[i,k]
# If no feasible k exists, fall back to pure argmin c_coord.
# Returns (k_best, fair_feasible, gap_best)
function pick_k_min_coord_with_fairness(
    c_coord::AbstractVector,
    C_air_fair::AbstractMatrix,
    Δ::Real
)
    K = length(c_coord)
    @assert size(C_air_fair, 2) == K

    best_k = 1
    best_cost = Inf
    best_gap = Inf
    found = false

    # Feasible search
    for k in 1:K
        col = view(C_air_fair, :, k)
        gap = maximum(col) - minimum(col)
        if gap <= Δ
            found = true
            ck = c_coord[k]
            if ck < best_cost
                best_cost = ck
                best_k = k
                best_gap = gap
            end
        end
    end

    # Fallback: no feasible k
    if !found
        best_k = argmin(c_coord)
        col = view(C_air_fair, :, best_k)
        best_gap = maximum(col) - minimum(col)
    end

    return best_k, found, best_gap
end

# ----------------------------
# Solver implementations
# ----------------------------
function solve_epoch(cfg::SimConfig, rng::AbstractRNG,
    state, flights, active_airlines, active_by_airline,
    actions_by_player, joint_pushed, joint_choice, params)

    nP = length(active_airlines)
    solver_detail = Dict{String,Any}()

    t0 = time()
    pushed_real = Int[]

    # --- Greedy / FCFS도 game 텐서를 만들어서 동일한 (J_coord + fairness-gap) 기준 적용 ---
    if cfg.solver_mode == GREEDY_CENTRALIZED
        game = build_epoch_game_tensors(state, flights, active_airlines, active_by_airline,
            actions_by_player, joint_pushed, joint_choice, params; n_runways=params.n_runways)

        k, fair_ok, gap = pick_k_min_coord_with_fairness(game.c_coord, game.C_air_fair, cfg.Δ)
        pushed_real = game.joint_pushed[k]

        solver_detail["k"] = k
        solver_detail["fair_feasible"] = fair_ok
        solver_detail["fair_gap_rec"] = gap

        t1 = time()
        return pushed_real, (t1 - t0), solver_detail

    elseif cfg.solver_mode == AGG_ORACLE_FCFS
        game = build_epoch_game_tensors(state, flights, active_airlines, active_by_airline,
            actions_by_player, joint_pushed, joint_choice, params; n_runways=params.n_runways)

        # 1) oracle chooses k by (J_coord min subject to fairness-gap<=Δ)
        k_oracle, fair_ok, gap = pick_k_min_coord_with_fairness(game.c_coord, game.C_air_fair, cfg.Δ)

        # 2) quotas from oracle action
        jc = game.joint_choice[k_oracle]
        quotas = Dict{Int,Int}()
        for p in 1:nP
            aid = active_airlines[p]
            aidx = jc[p]
            quotas[aid] = length(actions_by_player[p][aidx])
        end

        # 3) airline chooses flights by FCFS
        pushed_fcfs = Int[]
        for aid in active_airlines
            elig = active_by_airline[aid]
            q = quotas[aid]
            append!(pushed_fcfs, select_fcfs(elig, flights, q; rng=rng))
        end

        pushed_real = pushed_fcfs

        solver_detail["k_oracle"] = k_oracle
        solver_detail["quotas"] = quotas
        solver_detail["fair_feasible"] = fair_ok
        solver_detail["fair_gap_rec"] = gap

        t1 = time()
        return pushed_real, (t1 - t0), solver_detail
    end

    # --- CE / RRCE branch ---
    game = build_epoch_game_tensors(state, flights, active_airlines, active_by_airline,
        actions_by_player, joint_pushed, joint_choice, params; n_runways=params.n_runways)

    # sigma used by coordinator (assumed uncertainty inside solver)
    sigma_coord = build_sigma(cfg, cfg.coord_sigma_mode, cfg.coord_sigma_scalar, cfg.coord_sigma_vec, size(game.C_air_ic,1))

    # sigma used to generate realized deviation noise
    sigma_real  = build_sigma(cfg, cfg.real_sigma_mode,  cfg.real_sigma_scalar,  cfg.real_sigma_vec,  size(game.C_air_ic,1))

    k_rec = 1
    z = nothing

    if cfg.solver_mode == CE_NAIVE
        # naive = sigma ignored in solver (use zeros)
        # res = SearchCorrTensor(game.C_air_ic, game.C_air_fair, game.c_coord,
        #                        game.joint_choice, game.choice_to_k, game.action_sizes;
        #                        Δ=cfg.Δ, zalpha=cfg.zalpha, sigma=zeros(size(game.C_air_ic,1)))
        # z = res.z
        # k_rec = sample_k(z)
        # IMPORTANT: PNE detection must use IC costs (weighted) => game.C_air_ic
        pne_ks = find_pne_set(game.C_air_ic, game.joint_choice, game.choice_to_k, game.action_sizes;
                              tol=1e-3, zalpha=cfg.zalpha, sigma=zeros(size(game.C_air_ic,1)))
        solver_detail["num_pne"] = length(pne_ks)

        if isempty(pne_ks)
            # fallback: do nothing (as you had), but record it
            solver_detail["fallback"] = "no_pne_push_none"
            pushed_real = Int[]
            t1 = time()
            return pushed_real, (t1 - t0), solver_detail
        end

        # IMPORTANT: RRCE must use (objective=c_coord, fairness=C_air_fair)
        rr = solve_rrce_over_pne(game.C_air_ic, pne_ks, length(game.joint_pushed);
                                 Δ=cfg.Δ, C_air_fair=game.C_air_fair, c_coord=game.c_coord)

        solver_detail["rrce_status"] = string(rr.status)
        solver_detail["rrce_time_sec"] = rr.solverTime

        # Use rr.z_use directly (do NOT rebuild via pne_to_distribution)
        z = rr.z_use
        k_rec = sample_k(z)

    elseif cfg.solver_mode == CE_FULL
        res = SearchCorrTensor(game.C_air_ic, game.C_air_fair, game.c_coord,
                               game.joint_choice, game.choice_to_k, game.action_sizes;
                               Δ=cfg.Δ, zalpha=cfg.zalpha, sigma=sigma_coord)
        z = res.z
        k_rec = sample_k(z)

    elseif cfg.solver_mode == RRCE_PNE
        # IMPORTANT: PNE detection must use IC costs (weighted) => game.C_air_ic
        pne_ks = find_pne_set(game.C_air_ic, game.joint_choice, game.choice_to_k, game.action_sizes;
                              tol=1e-3, zalpha=cfg.zalpha, sigma=sigma_coord)
        solver_detail["num_pne"] = length(pne_ks)

        if isempty(pne_ks)
            # fallback: do nothing (as you had), but record it
            solver_detail["fallback"] = "no_pne_push_none"
            pushed_real = Int[]
            t1 = time()
            return pushed_real, (t1 - t0), solver_detail
        end

        # IMPORTANT: RRCE must use (objective=c_coord, fairness=C_air_fair)
        rr = solve_rrce_over_pne(game.C_air_ic, pne_ks, length(game.joint_pushed);
                                 Δ=cfg.Δ, C_air_fair=game.C_air_fair, c_coord=game.c_coord)

        solver_detail["rrce_status"] = string(rr.status)
        solver_detail["rrce_time_sec"] = rr.solverTime

        # Use rr.z_use directly (do NOT rebuild via pne_to_distribution)
        z = rr.z_use
        k_rec = sample_k(z)

    else
        error("Unknown solver_mode")
    end

    solver_detail["k_rec"] = k_rec

    # recommendation
    pushed_rec = game.joint_pushed[k_rec]

    # realized deviation model (optional)
    if cfg.enable_deviation
        # IMPORTANT: deviation must be based on IC costs (weighted)
        C_air_noisy = copy(game.C_air_ic)
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
                airline_cost_sum_unw=Dict{Int,Float64}(),
                airline_cost_avg_unw=Dict{Int,Float64}(),
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

        J_coord, air_sum, air_avg, air_sum_unw, air_avg_unw, info = compute_epoch_metrics(
            state, pushed, flights, active_airlines, active_by_airline, state.t, cfg.params;
            lambda_fair=cfg.lambda_fair,
            rho_release=cfg.rho_release)


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
            airline_cost_sum_unw=air_sum_unw,
            airline_cost_avg_unw=air_avg_unw,
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

    # (A) Epoch-wise fairness gap (unweighted avg 기준)
    gap_eps = Float64[]
    for lg in logs
        vals = Float64[]
        for (_, v) in lg.airline_cost_avg_unw
            isfinite(v) && push!(vals, v)
        end
        push!(gap_eps, isempty(vals) ? 0.0 : (maximum(vals) - minimum(vals)))
    end
    fairness_gap_ep_mean = isempty(gap_eps) ? 0.0 : mean(gap_eps)
    fairness_gap_ep_max  = isempty(gap_eps) ? 0.0 : maximum(gap_eps)

    # (B) Legacy-style global gap (epoch 섞어서 전체 max-min)
    all_vals = Float64[]
    for lg in logs
        for (_, v) in lg.airline_cost_avg_unw
            isfinite(v) && push!(all_vals, v)
        end
    end
    fairness_global = isempty(all_vals) ? 0.0 : (maximum(all_vals) - minimum(all_vals))

    # (C) Coordinator objective (현재는 J_coord 합)
    total_coord_obj = sum(lg.J_coord for lg in logs)

    # (D) Fairness-feasible rate (Greedy/FCFS/CE/RRCE에서 기록한 경우)
    flags = Bool[]
    for lg in logs
        if haskey(lg.solver_detail, "fair_feasible")
            push!(flags, Bool(lg.solver_detail["fair_feasible"]))
        end
    end
    fair_feasible_rate = isempty(flags) ? NaN : mean(flags)

    # (E) Avg #PNE (RRCE epochs만)
    pne_counts = Float64[]
    for lg in logs
        if haskey(lg.solver_detail, "num_pne")
            push!(pne_counts, Float64(lg.solver_detail["num_pne"]))
        end
    end
    avg_num_pne = isempty(pne_counts) ? NaN : mean(pne_counts)

    return SimResult(
        cfg=cfg,
        logs=logs,
        total_time_sec=(total_t1-total_t0),

        fairness_max_gap=fairness_global,
        total_coord_obj=total_coord_obj,

        fairness_gap_ep_mean=fairness_gap_ep_mean,
        fairness_gap_ep_max=fairness_gap_ep_max,
        fair_feasible_rate=fair_feasible_rate,
        avg_num_pne=avg_num_pne
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
# println("Done. total_time=$(res.total_time_sec) sec, fairness_max_gap=$(res.fairness_max_gap), total_obj=$(res.total_coord_obj)")