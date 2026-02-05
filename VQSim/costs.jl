module VQCosts

using Statistics

export predict_next_queue, compute_costs, evolve_state!

"""
Compute evaluation queue (tildeQ) = Q + arrivals(pushed_flights). No departures.
"""
function predict_next_queue(Q::Vector{Int}, pushed::Vector{Int}, flights, n_runways::Int, params)
    tildeQ = copy(Q)
    mu = params.mu
    for idx in pushed
        r = flights[idx].runway
        tildeQ[r] += 1
    end
    for r = 1:n_runways
        tildeQ[r] = max(0, tildeQ[r] - mu[r])
    end
    return tildeQ
end

function queue_delay(Q, params, r; Q0=1)
    mu = params.mu[r]
    return max(0, Q - Q0)/mu*4
end

"""
Compute nominal airline costs and coordinator cost for a candidate joint pushed set.

active_airlines: Vector{Int} (the player ordering for this epoch)
active_by_airline: Dict airline_id => eligible flight indices (for fairness normalization)

beta_queue: weight for queue/taxi delay (currently used inside d_total construction)
rho_release: regularizer weight on total released count (encourages throughput)
"""
function compute_costs(
    Q::Vector{Int},
    pushed::Vector{Int},
    flights,
    active_airlines::Vector{Int},
    active_by_airline::Dict{Int,Vector{Int}},
    t::Int,
    params;                          # current epoch time index
    n_runways::Int,
    beta_queue::Float64 = 1.0,       # weight for queue/taxi delay
    rho_release::Float64 = 0.0
)
    # Queue prediction for cost evaluation (CE stage): departures ignored, arrivals from `pushed` included
    tildeQ = predict_next_queue(Q, pushed, flights, n_runways, params)

    # --- Backlog set: all eligible flights (ready and not released) ---
    backlog = Int[]
    for aid in active_airlines
        append!(backlog, active_by_airline[aid])
    end

    # --- Airline backlog counts (for avg normalization) ---
    n_backlog_air = Dict{Int,Int}(aid => length(active_by_airline[aid]) for aid in active_airlines)

    # airline costs
    # J_air: pax-weighted (THIS is what airlines optimize; use this for IC constraints)
    J_air = Dict{Int,Float64}(aid => 0.0 for aid in active_airlines)

    # unweighted airline costs (for fairness metric/constraint)
    J_air_sum_unw = Dict{Int,Float64}(aid => 0.0 for aid in active_airlines)

    # coordinator cost: unweighted total delay over backlog flights
    total_delay_unweighted = 0.0

    pushed_set = Set(pushed)

    # taxiway congestion penalty (as in your original)
    tdelay = max(0, length(pushed) - 4.0)
    # tdelay = 0

    for idx in backlog
        f = flights[idx]

        # base lateness so far
        lateness = max(0, t - f.sched_t)

        # opportunity cost of waiting one more timestep
        wait_penalty = (idx in pushed_set) ? 0.0 : 4.0

        # queue delay (depends on joint pushed)
        qdelay = queue_delay(tildeQ[f.runway], params, f.runway; Q0=4.0)

        # NOTE: beta_queue is currently not separately applied; keeping d_total consistent with your old code.
        if lateness + wait_penalty > 10 
            d_total = (lateness+wait_penalty)^2 + qdelay + tdelay^2
        else
            d_total = wait_penalty + qdelay + tdelay^2
        end
        total_delay_unweighted += d_total

        # pax-weighted airline cost (IC)
        J_air[f.airline_id] += f.pax * d_total

        # unweighted airline cost (fairness)
        J_air_sum_unw[f.airline_id] += d_total
    end

    # unweighted avg per airline (recommended primitive for fairness-threshold)
    J_air_avg_unw = Dict{Int,Float64}()
    for aid in active_airlines
        n = n_backlog_air[aid]
        J_air_avg_unw[aid] = (n > 0) ? (J_air_sum_unw[aid] / n) : 0.0
    end

    release_count = length(pushed)

    # coordinator objective (unweighted total delay with optional throughput regularizer)
    J_coord = total_delay_unweighted

    return (
        tildeQ = tildeQ,
        backlog = backlog,

        # airline costs
        J_air = J_air,                     # pax-weighted (IC constraints)
        J_air_sum_unw = J_air_sum_unw,     # unweighted sum (diagnostics)
        J_air_avg_unw = J_air_avg_unw,     # unweighted avg (fairness constraint/metric)

        # coordinator cost
        J_coord = J_coord,
        release_count = release_count,
        total_delay_unweighted = total_delay_unweighted
    )
end

"""
Real state evolution with departures:
Q_next = max(0, Q + arrivals - departures)
departures per runway: d_r = min(Q_r, mu_r)

Also marks pushed flights as released (pushback executed).
"""
function evolve_state!(state, pushed::Vector{Int}, flights; mu::Vector{Int})
    # arrivals add
    for idx in pushed
        r = flights[idx].runway
        state.Q[r] += 1
    end
    # departures remove (service)
    for r in eachindex(state.Q)
        d = min(state.Q[r], mu[r])
        state.Q[r] -= d
    end
    # mark released
    for idx in pushed
        state.released[idx] = true
    end
    # advance time
    state.t += 4
    return state
end

end # module