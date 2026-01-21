module VQState

export SimParams, SimState, init_state, build_active_by_airline, mark_released!

struct SimParams
    n_runways::Int
    mu::Vector{Int}         # departure capacity per runway per epoch (for real state update)
end

mutable struct SimState
    t::Int
    Q::Vector{Int}          # runway queue lengths
    released::BitVector     # per-flight released status
end

function init_state(params::SimParams, n_flights::Int)::SimState
    return SimState(0, zeros(Int, params.n_runways), falses(n_flights))
end

"""
Return Dict(airline_id => Vector{Int flight_idx}) of eligible flights at time t.
Eligible = ready_t <= t and not released.
"""
function build_active_by_airline(state::SimState, flights)::Dict{Int, Vector{Int}}
    active = Dict{Int, Vector{Int}}()
    t = state.t
    for (idx, f) in enumerate(flights)
        if !state.released[idx] && f.ready_t <= t
            v = get!(active, f.airline_id, Int[])
            push!(v, idx)
        end
    end
    return active
end

function mark_released!(state::SimState, pushed::Vector{Int})
    for idx in pushed
        state.released[idx] = true
    end
end

end # module
