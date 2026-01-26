using Random

include("schedule.jl")
include("state.jl")
include("actions.jl")
include("costs.jl")

using .VQSchedule
using .VQState
using .VQActions
using .VQCosts

# === CONFIG ===
csv_path = "schedule/flight_schedule_1h.csv"
params = SimParams(2, [2, 2])       # 2 runways, 1 departure per epoch (mu)
T_sim = 16                           # Number of epochs (16 for 64 minutes)

lambda_fair = 1.0
rho_release = 0.1
max_subset_size = 1024

# === LOAD .csv ===
flights = VQSchedule.load_schedule(csv_path)
state = init_state(params, length(flights))

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
    #         state.Q, pushed, flights, active_airlines, active_by_airline, state.t;
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
    # VQCosts.evolve_state!(state, pushed_best, flights; mu=params.mu)

    """ CE-baseed solver """
    # 1) build tensors for this epoch
    game = build_epoch_game_tensors(state, flights, active_airlines, active_by_airline,
    actions_by_player, joint_pushed, joint_choice; n_runways = 2)

    # 2) solve CE (standard CE + epigraph)
    res = SearchCorrTensor(game.C_air, game.joint_choice, game.choice_to_k, game.action_sizes; Δ=0.0)
    z = res.z   # length K

    # 3) sample recommendation
    k_rec = sample_k(z)
    pushed_rec = game.joint_pushed[k_rec]

    # 4) evolve
    VQCosts.evolve_state!(state, pushed_rec, flights; mu=params.mu)

end

println("\nDone. Final Q = ", state.Q, ", t=", state.t)
