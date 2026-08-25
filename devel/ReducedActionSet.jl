using correlated
using BlockArrays: Block

# =============================================================================
# ReducedActionSet.jl
#
# Joint-action-space reduction for the runway/queue CC-CE game defined in
# 0_GameSetup.jl (SetC). Implements the "exclusive resource assignment"
# candidate set F from the reduced-joint-action formulation:
#
#   F := { x ∈ {0,1}^{n×r} : for every runway v, exactly one queue occupies it }
#
# For this game m = 2^r (each queue picks Occupy/Yield per runway independently),
# so |X| = m^n = 2^(rn), while |F| = n^r.
#
# Deviation costs for the reduced CC-CE LP are evaluated analytically via the
# cost-matrix blocks C[Block(i,j)] (J_def, already defined in 0_GameSetup.jl),
# so a deviation a_i' need NOT itself be an element of F -- exactly matching
# the AS[c,x] definition in the reduction note (a_i' ranges over the full
# individual action set Xi regardless of the recommendation set S).
# =============================================================================

# -----------------------------------------------------------------------------
# 1) Action-set construction (self-contained; does not depend on SetC globals)
# -----------------------------------------------------------------------------

"""
Reconstruct the per-player action ordering used internally by SetC, i.e. the
m = 2^r tuples of 'G'/'S' (Occupy/Stop) across r runways, in the same order
SetC uses to index its cost-matrix blocks.
"""
function BuildRunwayActionSet(r::Int)
    primeAction = "['G','S']"
    stringUnit = (primeAction * ",")^(r - 1) * primeAction
    stringSum = "vec(collect(Iterators.product(" * stringUnit * ")))"
    return eval(Meta.parse(stringSum))
end

"""
Index (1-based) of an occupancy tuple `tup::NTuple{r,Char}` within the action
ordering `actionSet` returned by BuildRunwayActionSet.
"""
function ActionIndex(actionSet, tup::Tuple)
    idx = findfirst(==(tup), actionSet)
    isnothing(idx) && error("Occupancy tuple $tup not found in action set.")
    return idx
end

# -----------------------------------------------------------------------------
# 2) Exclusive-occupancy reduced set F
# -----------------------------------------------------------------------------

"""
Build the exclusive-occupancy candidate set F for n queues and r runways.

Returns a Vector of NTuple{n,Int}, each entry giving, for every queue i, its
action index (1..m, m=2^r) in the SAME indexing SetC uses for its cost blocks.
|F| = n^r.
"""
function ExclusiveJointSet(r::Int, n::Int)
    actionSet = BuildRunwayActionSet(r)
    m = length(actionSet)
    @assert m == 2^r

    Sset = Vector{NTuple{n,Int}}(undef, n^r)
    k = 0
    for y in Iterators.product(ntuple(_ -> 1:n, r)...)
        # y[v] = index of the queue occupying runway v; all others yield on v
        k += 1
        occ = [ntuple(v -> (y[v] == i ? 'G' : 'S'), r) for i in 1:n]
        Sset[k] = ntuple(i -> ActionIndex(actionSet, occ[i]), n)
    end
    return Sset, m
end

# -----------------------------------------------------------------------------
# 3) Single-profile CC-PNE membership test (Definition of CC-PNE: doc "F* ordering")
# -----------------------------------------------------------------------------

"""
Check whether pure joint action `a::NTuple{n,Int}` is a chance-constrained
pure Nash equilibrium (CC-PNE) at confidence level implied by `zalpha*sigma`,
i.e. for every player i and every deviation a_i' != a_i:
    J_i(a_i', a_-i) - J_i(a) >= zalpha * sigma_i        (margin condition)
"""
function IsCCPNEProfile(a::NTuple{N,Int}, C, m::Int, n::Int; zalpha::Real, sigma) where {N}
    σi(i) = isa(sigma, Number) ? sigma : sigma[i]
    for i in 1:n
        c_base = J_def(i, collect(a), C)
        κ = zalpha * σi(i)
        for aibar in 1:m
            aibar == a[i] && continue
            a_dev = collect(a)
            a_dev[i] = aibar
            c_dev = J_def(i, a_dev, C)
            if c_dev < c_base + κ
                return false
            end
        end
    end
    return true
end

"""
Fraction (and list) of Sset that are CC-PNE at the given (zalpha, sigma).
Useful to check the Proposition-1-style uniform certificate empirically:
if all of Sset are CC-PNE, the convex hull of Sset equals the reduced CC-CE
feasible set (F*_{PNE,|F|} = F*_red).
"""
function CheckExclusiveSetCCPNE(Sset, C, m, n; zalpha, sigma)
    flags = [IsCCPNEProfile(a, C, m, n; zalpha = zalpha, sigma = sigma) for a in Sset]
    return (; all_pne = all(flags), frac_pne = mean(flags), flags = flags,
              pne_subset = Sset[flags])
end

# -----------------------------------------------------------------------------
# 4) Reduced-domain cost aggregation (mirrors CalcJ / CalcIndividualJ over Sset)
# -----------------------------------------------------------------------------

function CalcIndividualJReduced(z, Sset, i::Int, C)
    T = eltype(z)
    s = zero(T)
    @inbounds for k in eachindex(Sset)
        s += z[k] * J_def(i, collect(Sset[k]), C)
    end
    return s
end

function CalcJReduced(z, Sset, C, n::Int)
    T = eltype(z)
    s = zero(T)
    @inbounds for k in eachindex(Sset)
        a = collect(Sset[k])
        c = zero(T)
        for i in 1:n
            c += J_def(i, a, C)
        end
        s += z[k] * c
    end
    return s
end

function EvalFairnessReduced(z, Sset, C, n, Δ)
    c = [CalcIndividualJReduced(z, Sset, i, C) for i in 1:n]
    return abs(maximum(c) - minimum(c)) / Δ
end

function EvalGiniReduced(z, Sset, C, n)
    c = [CalcIndividualJReduced(z, Sset, i, C) for i in 1:n]
    us = 0.0
    for i in 1:n-1, j in i+1:n
        us += abs(c[i] - c[j])
    end
    return us / (2 * mean(c) * n^2)
end

# -----------------------------------------------------------------------------
# 5) Reduced CC-CE deviation constraints (Eq. AS[c,x] restricted to x ∈ Sset,
#    a_i' ranging over the FULL individual action set 1:m)
# -----------------------------------------------------------------------------

function CalcHReducedSet(z, Sset, C, m::Int, n::Int; zalpha::Real, sigma)
    T = eltype(z)
    σi(i) = isa(sigma, Number) ? sigma : sigma[i]

    # precompute, for each (i, ai), the set of column indices k with Sset[k][i]==ai
    idx_by_i_ai = [[Int[] for _ in 1:m] for _ in 1:n]
    for k in eachindex(Sset)
        a = Sset[k]
        for i in 1:n
            push!(idx_by_i_ai[i][a[i]], k)
        end
    end

    out = Vector{T}(undef, n * m * (m - 1))
    c = 0
    for i in 1:n
        for ai in 1:m
            ks = idx_by_i_ai[i][ai]
            p_ai = zero(T)
            for k in ks
                p_ai += z[k]
            end
            margin = T(zalpha) * T(σi(i)) * p_ai

            for aibar in 1:m
                ai == aibar && continue
                c += 1

                dev_sum = zero(T)  # Σ_k z_k * ( J_dev - J_follow )
                for k in ks
                    a = Sset[k]
                    c_follow = J_def(i, collect(a), C)
                    a_dev = collect(a)
                    a_dev[i] = aibar
                    c_dev = J_def(i, a_dev, C)
                    dev_sum += z[k] * (c_dev - c_follow)
                end

                out[c] = dev_sum - margin   # feasible iff >= 0
            end
        end
    end
    return out
end

# -----------------------------------------------------------------------------
# 6) Fairness-threshold epigraph packer over the reduced domain (mirrors
#    T1Const/T2Const/T3Const + CalcEFJ from 0_GameSetup.jl, reused as-is)
# -----------------------------------------------------------------------------

function T1ConstReduced(xi, Sset, C, n::Int, d::Int)
    z = xi[1:d]
    c = [CalcIndividualJReduced(z, Sset, i, C) for i in 1:n]
    w = xi[end]
    return w .- c
end

function T2ConstReduced(xi, Sset, C, n::Int, d::Int, Δ)
    z = xi[1:d]
    c = [CalcIndividualJReduced(z, Sset, i, C) for i in 1:n]
    v = xi[d+1:end-1]
    return v - c .- Δ
end

function ReducedPacker(x, Sset, C, m::Int, n::Int, d::Int, Δ; zalpha::Real, sigma)
    z = x[1:d]
    return [CalcHReducedSet(z, Sset, C, m, n; zalpha = zalpha, sigma = sigma);
            T1ConstReduced(x, Sset, C, n, d);
            T2ConstReduced(x, Sset, C, n, d, Δ);
            T3Const(x, d)]
end

# -----------------------------------------------------------------------------
# 7) Reduced CC-CE solver. Same interface style as SearchCorr, but restricted
#    to the candidate set Sset (default: exclusive-occupancy set). Objective
#    defaults to plain aggregate cost (CalcJReduced), matching SearchCorr, so
#    F*_full vs F*_red is directly comparable; pass `objective=:fair` to use
#    the RRCE fairness-threshold objective (CalcEFJ) instead.
# -----------------------------------------------------------------------------

function SearchCorrReduced(r::Int, n::Int, λ, Δ, Sset = nothing;
                            zalpha::Real, sigma, mult = 2.0,
                            objective::Symbol = :aggregate, verbose = false)
    println("Begin reduced Corr Search for m=$(2^r) and n=$n case (|F|=$(isnothing(Sset) ? n^r : length(Sset))).")
    C = SetC(r, n, λ; mult = mult)
    n_ = blocksize(C)[1]
    m = size(C[Block(1)])[1]

    if isnothing(Sset)
        Sset, _ = ExclusiveJointSet(r, n)
    end
    d = length(Sset)

    f = objective == :aggregate ?
        (x, θ) -> CalcJReduced(x[1:d], Sset, C, n) :
        (x, θ) -> CalcEFJ(x, d, n, Δ)

    g(x, θ) = [sum(x[1:d]) - 1]
    h(x, θ) = ReducedPacker(x, Sset, C, m, n, d, Δ; zalpha = zalpha, sigma = sigma)

    problem = ParametricOptimizationProblem(;
        objective = f,
        equality_constraint = g,
        inequality_constraint = h,
        parameter_dimension = 1,
        primal_dimension = d + n + 1,
        equality_dimension = 1,
        inequality_dimension = n * m * (m - 1) + 3 * n,
    )

    solverTime = @elapsed (; primals, variables, status, info) =
        solve(problem, [0]; verbose = verbose)

    z = primals[1:d]
    score = CalcJReduced(z, Sset, C, n)
    avgDelayScore = score / n
    fairScore = EvalFairnessReduced(z, Sset, C, n, Δ)
    giniScore = EvalGiniReduced(z, Sset, C, n)

    (; primals, z, Sset, score, avgDelayScore, fairScore, giniScore,
       d, varsize = length(primals), solverTime, status, info, m, n, C)
end
