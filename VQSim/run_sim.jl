using Random
using Distributions

include("schedule.jl")
include("state.jl")
include("actions.jl")
include("costs.jl")

using .VQSchedule
using .VQState
using .VQActions
using .VQCosts

# csv_path = "schedule/flight_schedule_1h.csv"
csv_path = "schedule/flight_schedule_1h_5b.csv"

# === LOAD .csv ===
flights = VQSchedule.load_schedule(csv_path)

# === CONFIG ===
params = SimParams(2, [2, 2])       # 2 runways, 1 departure per epoch (mu)
T_sim = 16                           # Number of epochs (16 for 60 minutes)

lambda_fair = 1.0
rho_release = 0.1
max_subset_size = 1024

# sigma = fill(500.0, length(flights))
sigma = fill(0.0, length(flights)) # Airline action uncertainty
α = 0.9 # Credibility parameter (α = 0.9 → 90% certainty)
Δ = 0.0
zalpha = quantile(Normal(), α)

state = VQState.init_state(params, length(flights))
println("Loaded flights: ", length(flights))

for step in 1:T_sim
    # === Identify active airlines ===
    active_by_airline = build_active_by_airline(state, flights) # Airline to corresponding active flight(s)
    active_airlines = sort(collect(keys(active_by_airline)))

    println("\n=== t=$(state.t) ===")
    println("Q(t) = ", state.Q)
    println("Active airlines: ", active_airlines)
    if isempty(active_airlines)
        println("No active airlines. Advancing time with departures only.")
        # still apply departures and advance time
        VQCosts.evolve_state!(state, Int[], flights; mu=params.mu)
        continue
    end

    # build action sets (subset actions)
    actions_by_player = Vector{Vector{Vector{Int}}}()
    for aid in active_airlines
        elig = active_by_airline[aid]   # Corresponding flights
        acts = enumerate_actions_subset(elig; max_subset_size=max_subset_size)
        push!(actions_by_player, acts)
        println(" airline $aid eligible=$(length(elig)), actions=$(length(acts))")
    end

    joint_pushed, joint_choice = enumerate_joint_actions(actions_by_player)
    println("Joint actions: ", length(joint_pushed))

    """ Greedy selection by coordinator cost (Greedy centralized enforcement) """
    # best_k = 1
    # best_cost = Inf
    # best_info = nothing
    # for k in eachindex(joint_pushed)
    #     pushed = joint_pushed[k]
    #     info = VQCosts.compute_costs(
    #         state.Q, pushed, flights, active_airlines, active_by_airline, state.t, params;
    #         n_runways=params.n_runways,
    #         beta_queue=lambda_fair,
    #         rho_release=rho_release
    #     )
    #     if info.J_coord < best_cost
    #         best_cost = info.J_coord
    #         best_k = k
    #         best_info = info
    #     end
    # end
    # pushed_best = joint_pushed[best_k]
    # println(length(pushed_best),"/",sum(length, values(active_by_airline))," aircraft pushed.")
    # VQCosts.evolve_state!(state, pushed_best, flights; mu=params.mu)

    """ CE-based solver """
    # 1) build tensors for this epoch
    game = build_epoch_game_tensors(state, flights, active_airlines, active_by_airline,
    actions_by_player, joint_pushed, joint_choice, params; n_runways = 2)

    # 2) solve CE (standard CE + epigraph)
    res = SearchCorrTensor(game.C_air, game.joint_choice, game.choice_to_k, game.action_sizes; Δ=Δ, zalpha = zalpha, sigma = sigma)
    z = res.z   # length K

    # 3) sample recommendation
    k_rec = sample_k(z)
    pushed_rec = game.joint_pushed[k_rec]

    # 4) Actual choice
    C_air_noisy = copy(game.C_air)
    for i in 1:size(C_air_noisy,1)
        C_air_noisy[i, :] .+= sigma[i] .* randn(size(C_air_noisy,2))
    end

    k_real = realized_choice_conditional_BR(
        C_air_noisy, game.joint_choice, game.choice_to_k, game.action_sizes, k_rec
    )
    pushed_real = game.joint_pushed[k_real]
    println(length(pushed_real),"/",sum(length, values(active_by_airline))," aircraft pushed.")

    # 5) evolve
    VQCosts.evolve_state!(state, pushed_rec, flights; mu=params.mu)
    # VQCosts.evolve_state!(state, pushed_real, flights; mu=params.mu)

    """ Brute - RRCE solver"""
    # #build tensors for this epoch
    # game = build_epoch_game_tensors(state, flights, active_airlines, active_by_airline,
    # actions_by_player, joint_pushed, joint_choice, params; n_runways = 2)

    # # Find PNE set
    # pne_ks = find_pne_set(game.C_air, game.joint_choice, game.choice_to_k, game.action_sizes; tol=1e-3, zalpha = zalpha, sigma = sigma)
    # println("Number of pnes: ",length(pne_ks))   
    
    # # Debugger
    # # out = max_regret_per_k(game.C_air, game.joint_choice, game.choice_to_k, game.action_sizes; tol=0.0)
    # # summarize_regrets(out; tol=1e-7, top=10)

    # if isempty(pne_ks)
    #     println("No PNE found this epoch")

    # else
    #     # RRCE-like mixing
    #     rr = solve_rrce_over_pne(game.C_air, pne_ks, length(game.joint_pushed); Δ=Δ)
    #     z_use = pne_to_distribution(rr.pne_ks, rr.λ, length(game.joint_pushed))

    #     k_rec = sample_k(z_use)
    # end

    # k_rec = sample_k(z_use)
    # pushed_rec = game.joint_pushed[k_rec]

    # # 노이즈 일탈(기존 그대로)
    # C_air_noisy = copy(game.C_air)
    # for i in 1:size(C_air_noisy,1)
    #     C_air_noisy[i, :] .+= sigma[i] .* randn(size(C_air_noisy,2))
    # end

    # k_real = realized_choice_conditional_BR(
    #     C_air_noisy, game.joint_choice, game.choice_to_k, game.action_sizes, k_rec
    # )
    # pushed_real = game.joint_pushed[k_real]
    # # println(active_by_airline)
    # # println(pushed_real)
    # println(length(pushed_real),"/",sum(length, values(active_by_airline))," aircraft pushed.")

    # VQCosts.evolve_state!(state, pushed_real, flights; mu=params.mu)

    """ Aggregate-oracle + Airline FCFS """
    # # 1) Coordinator picks best joint action by J_coord (oracle-like), but uses it only for airline quotas
    # best_k = 1
    # best_cost = Inf
    # best_info = nothing
    # for k in eachindex(joint_pushed)
    #     pushed = joint_pushed[k]
    #     info = VQCosts.compute_costs(
    #         state.Q, pushed, flights, active_airlines, active_by_airline, state.t, params;
    #         n_runways=params.n_runways,
    #         beta_queue=lambda_fair,
    #         rho_release=rho_release
    #     )
    #     if info.J_coord < best_cost
    #         best_cost = info.J_coord
    #         best_k = k
    #         best_info = info
    #     end
    # end

    # # 2) Extract per-airline quotas from the selected joint_choice
    # # joint_choice[best_k] is length nP vector: local action index for each airline-player
    # quotas = Dict{Int,Int}()  # airline_id -> quota
    # jc = joint_choice[best_k]
    # for p in 1:length(active_airlines)
    #     aid = active_airlines[p]
    #     aidx = jc[p]  # local action index
    #     quota = length(actions_by_player[p][aidx])  # number of flights in that chosen subset
    #     quotas[aid] = quota
    # end

    # # 3) Each airline chooses which flights to push using FCFS (only within its eligible set)
    # rng = Random.default_rng()
    # pushed_fcfs = Int[]
    # for aid in active_airlines
    #     elig = active_by_airline[aid]
    #     q = quotas[aid]
    #     chosen = select_fcfs(elig, flights, q; rng=rng)
    #     append!(pushed_fcfs, chosen)
    # end

    # println("Aggregate-oracle+FCFS: pushed $(length(pushed_fcfs)) / $(sum(length, values(active_by_airline))) flights.")

    # # 4) Evolve state using the realized pushed set
    # VQCosts.evolve_state!(state, pushed_fcfs, flights; mu=params.mu)
end

println("\nDone. Final Q = ", state.Q, ", t=", state.t)
