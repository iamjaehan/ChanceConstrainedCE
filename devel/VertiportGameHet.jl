using correlated
using JuMP
import Ipopt

# =============================================================================
# VertiportGameHet.jl
#
# The vertiport occupancy game of VertiportGame.jl with a HETEROGENEOUS yield
# cost. The nominal cost of Eq. 22 becomes
#
#   J̄_i(x) = unit·γ·Σ_v x_iv (N_v(x) − 1) + Σ_v w_iv (1 − x_iv)
#
# i.e. the flat per-resource yield cost `unit` is replaced by an agent-and-
# resource specific weight w_iv: how much agent i minds yielding resource v.
# Setting w_iv ≡ unit recovers VertiportGame.jl exactly.
#
# Why this matters. On the exclusive set F every resource has exactly one
# occupant, so for x ∈ F a unilateral deviation of agent i changes its cost by
#
#   v moved occupy → yield :  + w_iv          (i was the sole occupant)
#   v moved yield → occupy :  + (unit·γ − w_iv)   (v already had exactly one)
#
# Both are independent of WHO the other agents are, so the CC-PNE test for
# x ∈ F reduces to, for every agent i,
#
#   min( min_{v ∈ R_i(x)} w_iv , min_{v ∉ R_i(x)} (unit·γ − w_iv) ) ≥ q_α σ_i   (*)
#
# where R_i(x) is the set of resources i holds. With w_iv ≡ unit this collapses
# to unit·min{1, γ−1} ≥ q_α σ_i -- Proposition 1's uniform certificate, which is
# profile-independent and therefore all-or-nothing. With heterogeneous w_iv the
# left side depends on WHICH resources i was assigned, so some x ∈ F pass and
# others fail: partial certification.
#
# With the weights built by `HetYieldWeights` below (one "disliked" resource per
# agent, cycling), (*) says simply:
#
#   x ∈ F is a CC-PNE  ⟺  no agent holds its disliked resource.
#
# NOTE on the reduced formulation. Because the margin is independent of x_-i on
# F, each CC-CE row degenerates to a per-(i, a_i) pass/fail test, and therefore
# Z_red = Δ(P_d) exactly: on F the reduced CC-CE problem IS the CC-PNE hull over
# the certified subset, certificate or no certificate. Correlation only gains
# leverage on candidate sets that admit N_v ≥ 2.
# =============================================================================

"""
Yield weights w_iv for n agents and r resources: each agent dislikes exactly one
resource (weight `wlo`), cycling through the resources; all other resources carry
`wmid`.

With unit·γ = 10 and q_α σ = 4.5, holding needs w ≥ 4.5 and not-holding needs
w ≤ 5.5, so wlo = 4.0 fails only the holding test and wmid = 5.0 passes both.
"""
function HetYieldWeights(n::Int, r::Int; wlo::Real = 4.0, wmid::Real = 5.0)
    return [(((i - 1) % r) + 1 == v ? Float64(wlo) : Float64(wmid)) for i in 1:n, v in 1:r]
end

"""
Nominal cost with heterogeneous yield weights `W` (n x r).
"""
function VertiportCostHet(i::Int, a, actionSet, r::Int, γ::Real, W::AbstractMatrix; unit::Real = 5.0)
    occ_i = OccupancyVec(actionSet, a[i], r)
    Nv = NvVector(actionSet, a, r)
    cost = 0.0
    for v in 1:r
        cost += unit * γ * occ_i[v] * (Nv[v] - 1) + W[i, v] * (1 - occ_i[v])
    end
    return cost
end

# -----------------------------------------------------------------------------
# CC-PNE membership and search
# -----------------------------------------------------------------------------

function IsCCPNEProfileHet(a, actionSet, r::Int, γ::Real, n::Int, W;
                            zalpha::Real, sigma, unit::Real = 5.0)
    m = length(actionSet)
    σi(i) = isa(sigma, Number) ? sigma : sigma[i]
    for i in 1:n
        c_base = VertiportCostHet(i, a, actionSet, r, γ, W; unit = unit)
        κ = zalpha * σi(i)
        for aibar in 1:m
            aibar == a[i] && continue
            a_dev = collect(a); a_dev[i] = aibar
            VertiportCostHet(i, a_dev, actionSet, r, γ, W; unit = unit) < c_base + κ && return false
        end
    end
    return true
end

"""
Brute-force CC-PNE enumeration over the whole joint action space X.
"""
function FindAllCCPNEHet(r::Int, n::Int, γ::Real, W; zalpha::Real, sigma, unit::Real = 5.0)
    actionSet = BuildRunwayActionSet(r)
    m = length(actionSet)
    pne = NTuple{n,Int}[]
    for a in Iterators.product(ntuple(_ -> 1:m, n)...)
        IsCCPNEProfileHet(a, actionSet, r, γ, n, W; zalpha = zalpha, sigma = sigma, unit = unit) &&
            push!(pne, a)
    end
    return pne, actionSet, m
end

"""
Which members of a candidate set are CC-PNE (the partial-certification report).
"""
function CheckSetCCPNEHet(Sset, actionSet, r::Int, γ::Real, n::Int, W;
                           zalpha::Real, sigma, unit::Real = 5.0)
    flags = [IsCCPNEProfileHet(a, actionSet, r, γ, n, W; zalpha = zalpha, sigma = sigma, unit = unit)
             for a in Sset]
    return (; all_pne = all(flags), frac_pne = mean(flags), flags, pne_subset = Sset[flags])
end

# -----------------------------------------------------------------------------
# Shared LP body: CC-CE over an arbitrary support set S, deviations over full 1:m
# -----------------------------------------------------------------------------
function _ccce_lp_het(S, r::Int, n::Int, γ::Real, Δ, W;
                       zalpha::Real, sigma, unit::Real, objective::Symbol,
                       optimizer, verbose::Bool, label::AbstractString)
    actionSet = BuildRunwayActionSet(r)
    m = length(actionSet)
    d = length(S)
    σi(i) = isa(sigma, Number) ? sigma : sigma[i]

    model = Model(optimizer)
    verbose || set_silent(model)

    local Ci_of_k
    buildTime = @elapsed begin
        @variable(model, z[1:d] >= 0)
        @constraint(model, sum(z) == 1)

        idx_by_i_ai = [[Int[] for _ in 1:m] for _ in 1:n]
        for k in 1:d, i in 1:n
            push!(idx_by_i_ai[i][S[k][i]], k)
        end

        for i in 1:n
            for ai in 1:m
                ks = idx_by_i_ai[i][ai]
                isempty(ks) && continue
                p_ai = sum(z[k] for k in ks)
                for aibar in 1:m
                    ai == aibar && continue
                    dev_sum = zero(AffExpr)
                    for k in ks
                        a = collect(S[k])
                        c_follow = VertiportCostHet(i, a, actionSet, r, γ, W; unit = unit)
                        a_dev = copy(a); a_dev[i] = aibar
                        c_dev = VertiportCostHet(i, a_dev, actionSet, r, γ, W; unit = unit)
                        add_to_expression!(dev_sum, c_dev - c_follow, z[k])
                    end
                    @constraint(model, dev_sum - zalpha * σi(i) * p_ai >= 0)
                end
            end
        end

        Ci_of_k = Matrix{Float64}(undef, n, d)
        for i in 1:n, k in 1:d
            Ci_of_k[i, k] = VertiportCostHet(i, collect(S[k]), actionSet, r, γ, W; unit = unit)
        end

        if objective == :aggregate
            total_of_k = [sum(Ci_of_k[i, k] for i in 1:n) for k in 1:d]
            @objective(model, Min, sum(total_of_k[k] * z[k] for k in 1:d))
        elseif objective == :fair
            c_exprs = [@expression(model, sum(Ci_of_k[i, k] * z[k] for k in 1:d)) for i in 1:n]
            @objective(model, Min, _fairness_objective!(model, c_exprs, n, Δ))
        else
            error("objective must be :aggregate or :fair")
        end
    end

    solverTime = @elapsed optimize!(model)
    status = termination_status(model)
    zval = value.(z)
    score = sum(sum(Ci_of_k[i, k] for i in 1:n) * zval[k] for k in 1:d)
    cvals = [sum(Ci_of_k[i, k] * zval[k] for k in 1:d) for i in 1:n]
    fairScore = abs(maximum(cvals) - minimum(cvals)) / Δ
    giniScore = sum(abs(cvals[i] - cvals[j]) for i in 1:n-1 for j in i+1:n) / (2 * mean(cvals) * n^2)

    (; primals = zval, z = zval, S, score, avgDelayScore = score / n, fairScore, giniScore,
       d, varsize = d, buildTime, solverTime, status, m, n, actionSet, cvals, label, model)
end

"""
Full CC-CE over all of X, heterogeneous yield cost.
"""
function SearchCorrLPHet(r::Int, n::Int, γ::Real, Δ, W;
                          zalpha::Real, sigma, unit::Real = 5.0,
                          objective::Symbol = :aggregate,
                          optimizer = Ipopt.Optimizer, verbose = false)
    m = 2^r
    Xall = [Tuple(c) for c in CartesianIndices(ntuple(_ -> m, n))]
    return _ccce_lp_het(Xall, r, n, γ, Δ, W; zalpha, sigma, unit, objective,
                        optimizer, verbose, label = "Full")
end

"""
Reduced CC-CE over the exclusive-occupancy set F (N_v = 1), heterogeneous yield cost.
"""
function SearchCorrReducedLPHet(r::Int, n::Int, γ::Real, Δ, W, Sset = nothing;
                                 zalpha::Real, sigma, unit::Real = 5.0,
                                 objective::Symbol = :aggregate,
                                 optimizer = Ipopt.Optimizer, verbose = false)
    if isnothing(Sset)
        Sset, _ = ExclusiveJointSet(r, n)
    end
    return _ccce_lp_het(Sset, r, n, γ, Δ, W; zalpha, sigma, unit, objective,
                        optimizer, verbose, label = "Reduced")
end

# -----------------------------------------------------------------------------
# CC-PNE hull LPs (no deviation constraints -- every vertex is an equilibrium)
# -----------------------------------------------------------------------------
function _hull_lp_het(pneList, r::Int, n::Int, γ::Real, Δ, W, actionSet;
                       unit::Real, objective::Symbol, optimizer, verbose::Bool,
                       searchTime::Real, label::AbstractString)
    d = length(pneList)
    d == 0 && error("No CC-PNE available; cannot build the hull LP.")

    model = Model(optimizer)
    verbose || set_silent(model)

    local scoreSet
    buildTime = @elapsed begin
        scoreSet = Matrix{Float64}(undef, d, n)
        for k in 1:d, i in 1:n
            scoreSet[k, i] = VertiportCostHet(i, collect(pneList[k]), actionSet, r, γ, W; unit = unit)
        end
        @variable(model, λv[1:d] >= 0)
        @constraint(model, sum(λv) == 1)
        if objective == :aggregate
            tot = [sum(scoreSet[k, i] for i in 1:n) for k in 1:d]
            @objective(model, Min, sum(tot[k] * λv[k] for k in 1:d))
        else
            c_exprs = [@expression(model, sum(scoreSet[k, i] * λv[k] for k in 1:d)) for i in 1:n]
            @objective(model, Min, _fairness_objective!(model, c_exprs, n, Δ))
        end
    end

    solverTime = @elapsed optimize!(model)
    status = termination_status(model)
    λval = value.(λv)
    cvals = [sum(scoreSet[k, i] * λval[k] for k in 1:d) for i in 1:n]
    score = sum(cvals)
    fairScore = abs(maximum(cvals) - minimum(cvals)) / Δ
    giniScore = sum(abs(cvals[i] - cvals[j]) for i in 1:n-1 for j in i+1:n) / (2 * mean(cvals) * n^2)

    (; λ = λval, pneList, scoreSet, score, avgDelayScore = score / n, fairScore, giniScore,
       d, varsize = d, buildTime, solverTime, searchTime, status, n, cvals, label, model)
end

"""
CC-PNE hull with the equilibrium search run over the FULL joint action space X.
"""
function HullLPHetOverX(r::Int, n::Int, γ::Real, Δ, W;
                         zalpha::Real, sigma, unit::Real = 5.0,
                         objective::Symbol = :aggregate,
                         optimizer = Ipopt.Optimizer, verbose = false)
    t = @elapsed ((pneList, actionSet, _) =
        FindAllCCPNEHet(r, n, γ, W; zalpha = zalpha, sigma = sigma, unit = unit))
    return _hull_lp_het(pneList, r, n, γ, Δ, W, actionSet; unit, objective,
                        optimizer, verbose, searchTime = t, label = "Hull / X")
end

"""
CC-PNE hull with the search scoped to the exclusive set F. Valid here because the
CC-PNE of this game all lie in F, but note this is an assumption the uniform
certificate is what normally licenses -- under partial certification it must be
checked, which is what this does (over |F| = n^r candidates rather than m^n).
"""
function HullLPHetOverF(r::Int, n::Int, γ::Real, Δ, W, Sset = nothing;
                         zalpha::Real, sigma, unit::Real = 5.0,
                         objective::Symbol = :aggregate,
                         optimizer = Ipopt.Optimizer, verbose = false)
    actionSet = BuildRunwayActionSet(r)
    if isnothing(Sset)
        Sset, _ = ExclusiveJointSet(r, n)
    end
    t = @elapsed begin
        chk = CheckSetCCPNEHet(Sset, actionSet, r, γ, n, W;
                               zalpha = zalpha, sigma = sigma, unit = unit)
        pneList = Sset[chk.flags]
    end
    return _hull_lp_het(pneList, r, n, γ, Δ, W, actionSet; unit, objective,
                        optimizer, verbose, searchTime = t, label = "Hull / F")
end
