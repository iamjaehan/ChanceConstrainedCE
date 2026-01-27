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
    for i in 1:n_runways
        tildeQ[i] - mu[i]
    end
    return tildeQ
end

function queue_delay(Q, params, r; Q0=4)
    return max(0, Q - Q0)
    # return exp(Q*0.231)-1
    # return Q/3
    # return max(0, Q - Q0)^2 /36
    # return Q^2 / 4
    mu = params.mu[r]
    # return max(0, Q-Q0)/mu*4.0
    return max(0, Q)/mu*4.0
end

"""
Compute nominal airline costs and coordinator cost for a candidate joint pushed set.

active_airlines: Vector{Int} (the player ordering for this epoch)
active_by_airline: Dict airline_id => eligible flight indices (for fairness normalization)
lambda_fair: weight on fairness
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
    # active_by_airline[aid] is assumed to list eligible flights for that airline at time t
    backlog = Int[]
    for aid in active_airlines
        append!(backlog, active_by_airline[aid])
    end

    # airline costs (pax-weighted, over backlog flights)
    J_air = Dict{Int, Float64}()
    for aid in active_airlines
        J_air[aid] = 0.0
    end

    # coordinator cost: unweighted total delay over backlog flights
    total_delay_unweighted = 0.0

    pushed_set = Set(pushed)

    # taxiway congestion penalty
    tdelay = max(0,length(pushed) - 4.0)

    for idx in backlog
        f = flights[idx]

        # base lateness so far
        lateness = max(0, t - f.sched_t)

        # opportunity cost of waiting one more timestep
        wait_penalty = idx in pushed_set ? 0.0 : 4.0

        # queue delay (depends on joint pushed)
        qdelay = queue_delay(tildeQ[f.runway], params, f.runway; Q0=1)

        d_total = lateness + wait_penalty + qdelay + tdelay

        total_delay_unweighted += d_total
        J_air[f.airline_id] += f.pax * d_total
        # J_air[f.airline_id] += d_total
    end

    release_count = length(pushed)

    # Under (B), J_coord naturally penalizes "do nothing" because backlog lateness grows with t.
    # rho_release can be kept as a mild regularizer if you want.
    J_coord = total_delay_unweighted - rho_release * release_count

    return (
        tildeQ = tildeQ,
        backlog = backlog,
        J_air = J_air,
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
