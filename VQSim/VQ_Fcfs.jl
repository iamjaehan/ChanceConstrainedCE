using Random
"""
Select `k` flights from `elig` by FCFS (earliest sched_t first).
If ties in sched_t, break ties randomly.
"""
function select_fcfs(elig::Vector{Int}, flights, k::Int; rng::AbstractRNG=Random.default_rng())
    k <= 0 && return Int[]
    k = min(k, length(elig))
    k == 0 && return Int[]

    # Group by sched_t and randomize within ties
    # 1) sort by sched_t
    ord = sortperm(elig, by = i -> flights[i].sched_t)

    # 2) break ties randomly within equal sched_t blocks
    out = Int[]
    i = 1
    while i <= length(ord) && length(out) < k
        t0 = flights[elig[ord[i]]].sched_t
        j = i
        while j <= length(ord) && flights[elig[ord[j]]].sched_t == t0
            j += 1
        end
        block = elig[ord[i:j-1]]
        Random.shuffle!(rng, block)
        append!(out, block)
        i = j
    end

    return out[1:k]
end