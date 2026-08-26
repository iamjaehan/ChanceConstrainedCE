using correlated
using JuMP
import Ipopt

# =============================================================================
# VertiportGame.jl
#
# The exact vertiport occupancy game from the CC-CE paper (doc2), as opposed
# to devel/0_GameSetup.jl's SetC, which is the RRCE paper's pairwise-summed
# runway cost. The two coincide qualitatively but NOT exactly: SetC's yield
# cost sums a fixed per-opponent contribution, so it scales with (n-1); the
# CC-CE paper's yield cost is a flat constant per resource, independent of n.
# That's why the n-independent uniform CC-PNE certificate (reduction note,
# Proposition 1: q_α*σ_i <= 5*min{1,γ-1}) holds exactly here but only held
# for SetC up to n < mult+1.
#
# Action space reuses BuildRunwayActionSet/ExclusiveJointSet from
# ReducedActionSet.jl unchanged (same {G,S}^r per-player action encoding,
# m = 2^r) -- only the cost function differs.
# =============================================================================

"""
Occupancy indicator vector (length r, 0/1) for action index `ai` under `actionSet`.
"""
function OccupancyVec(actionSet, ai::Int, r::Int)
    tup = actionSet[ai]
    return [tup[v] == 'G' ? 1 : 0 for v in 1:r]
end

"""
Nv(x) for every resource v, given a full joint action `a::Vector{Int}` (or Tuple).
"""
function NvVector(actionSet, a, r::Int)
    Nv = zeros(Int, r)
    for ai in a
        Nv .+= OccupancyVec(actionSet, ai, r)
    end
    return Nv
end

"""
CC-CE paper Eq. 22 nominal cost: J̄_i(x) = unit*γ*Σ_v x_iv*(Nv(x)-1) + unit*Σ_v(1-x_iv).
`a` is the full joint action (Vector/Tuple of per-player action indices).
"""
function VertiportCost(i::Int, a, actionSet, r::Int, γ::Real; unit::Real = 5.0)
    occ_i = OccupancyVec(actionSet, a[i], r)
    Nv = NvVector(actionSet, a, r)
    cost = 0.0
    for v in 1:r
        cost += unit * γ * occ_i[v] * (Nv[v] - 1) + unit * (1 - occ_i[v])
    end
    return cost
end

# -----------------------------------------------------------------------------
# CC-PNE membership / brute-force search (full enumeration -- fine at the
# problem sizes used here; see RRCE paper's "brute-force method").
# -----------------------------------------------------------------------------

function IsCCPNEProfileVertiport(a, actionSet, r::Int, γ::Real, n::Int; zalpha::Real, sigma, unit::Real = 5.0)
    m = length(actionSet)
    σi(i) = isa(sigma, Number) ? sigma : sigma[i]
    for i in 1:n
        c_base = VertiportCost(i, a, actionSet, r, γ; unit = unit)
        κ = zalpha * σi(i)
        for aibar in 1:m
            aibar == a[i] && continue
            a_dev = collect(a); a_dev[i] = aibar
            c_dev = VertiportCost(i, a_dev, actionSet, r, γ; unit = unit)
            if c_dev < c_base + κ
                return false
            end
        end
    end
    return true
end

function FindAllCCPNEVertiport(r::Int, n::Int, γ::Real; zalpha::Real, sigma, unit::Real = 5.0)
    actionSet = BuildRunwayActionSet(r)
    m = length(actionSet)
    pne = NTuple{n,Int}[]
    for a in Iterators.product(ntuple(_ -> 1:m, n)...)
        IsCCPNEProfileVertiport(a, actionSet, r, γ, n; zalpha = zalpha, sigma = sigma, unit = unit) && push!(pne, a)
    end
    return pne, actionSet, m
end

function CheckExclusiveSetCCPNEVertiport(Sset, actionSet, r::Int, γ::Real, n::Int; zalpha::Real, sigma, unit::Real = 5.0)
    flags = [IsCCPNEProfileVertiport(a, actionSet, r, γ, n; zalpha = zalpha, sigma = sigma, unit = unit) for a in Sset]
    return (; all_pne = all(flags), frac_pne = mean(flags), flags)
end

# -----------------------------------------------------------------------------
# Full CC-CE as an LP (same structure as SearchCorrLP in CorrLP.jl).
# -----------------------------------------------------------------------------
function SearchCorrLPVertiport(r::Int, n::Int, γ::Real, Δ;
                                zalpha::Real, sigma, unit::Real = 5.0,
                                objective::Symbol = :aggregate,
                                optimizer = Ipopt.Optimizer, verbose = false)
    actionSet = BuildRunwayActionSet(r)
    m = length(actionSet)
    l = m^n
    CI = CartesianIndices(ntuple(_ -> m, n))
    LI = LinearIndices(CI)
    σi(i) = isa(sigma, Number) ? sigma : sigma[i]

    model = Model(optimizer)
    verbose || set_silent(model)

    Ci_of_k = Matrix{Float64}(undef, n, l)
    buildTime = @elapsed begin
        @variable(model, z[1:l] >= 0)
        @constraint(model, sum(z) == 1)

        for i in 1:n
            for ai in 1:m
                idxs = [LI[ci] for ci in CI if ci[i] == ai]
                isempty(idxs) && continue
                p_ai = sum(z[k] for k in idxs)
                for aibar in 1:m
                    ai == aibar && continue
                    dev_sum = zero(AffExpr)
                    for k in idxs
                        a = collect(Tuple(CI[k]))
                        c_follow = VertiportCost(i, a, actionSet, r, γ; unit = unit)
                        a_dev = copy(a); a_dev[i] = aibar
                        c_dev = VertiportCost(i, a_dev, actionSet, r, γ; unit = unit)
                        add_to_expression!(dev_sum, c_dev - c_follow, z[k])
                    end
                    @constraint(model, dev_sum - zalpha * σi(i) * p_ai >= 0)
                end
            end
        end

        for i in 1:n, k in 1:l
            Ci_of_k[i, k] = VertiportCost(i, collect(Tuple(CI[k])), actionSet, r, γ; unit = unit)
        end

        if objective == :aggregate
            total_of_k = [sum(Ci_of_k[i, k] for i in 1:n) for k in 1:l]
            @objective(model, Min, sum(total_of_k[k] * z[k] for k in 1:l))
        elseif objective == :fair
            c_exprs = [@expression(model, sum(Ci_of_k[i, k] * z[k] for k in 1:l)) for i in 1:n]
            @objective(model, Min, _fairness_objective!(model, c_exprs, n, Δ))
        else
            error("objective must be :aggregate or :fair")
        end
    end

    solverTime = @elapsed optimize!(model)
    status = termination_status(model)
    zval = value.(z)
    score = sum(sum(Ci_of_k[i, k] for i in 1:n) * zval[k] for k in 1:l)
    avgDelayScore = score / n
    cvals = [sum(Ci_of_k[i, k] * zval[k] for k in 1:l) for i in 1:n]
    fairScore = abs(maximum(cvals) - minimum(cvals)) / Δ
    giniScore = sum(abs(cvals[i] - cvals[j]) for i in 1:n-1 for j in i+1:n) / (2 * mean(cvals) * n^2)

    (; primals = zval, z = zval, score, avgDelayScore, fairScore, giniScore,
       varsize = l, buildTime, solverTime, status, m, n, l, actionSet, model)
end

# -----------------------------------------------------------------------------
# Reduced CC-CE as an LP, restricted to Sset (default: exclusive-occupancy set).
# -----------------------------------------------------------------------------
function SearchCorrReducedLPVertiport(r::Int, n::Int, γ::Real, Δ, Sset = nothing;
                                       zalpha::Real, sigma, unit::Real = 5.0,
                                       objective::Symbol = :aggregate,
                                       optimizer = Ipopt.Optimizer, verbose = false)
    actionSet = BuildRunwayActionSet(r)
    m = length(actionSet)
    if isnothing(Sset)
        Sset, _ = ExclusiveJointSet(r, n)
    end
    d = length(Sset)
    σi(i) = isa(sigma, Number) ? sigma : sigma[i]

    model = Model(optimizer)
    verbose || set_silent(model)

    Ci_of_k = Matrix{Float64}(undef, n, d)
    buildTime = @elapsed begin
        @variable(model, z[1:d] >= 0)
        @constraint(model, sum(z) == 1)

        idx_by_i_ai = [[Int[] for _ in 1:m] for _ in 1:n]
        for k in 1:d
            a = Sset[k]
            for i in 1:n
                push!(idx_by_i_ai[i][a[i]], k)
            end
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
                        a = collect(Sset[k])
                        c_follow = VertiportCost(i, a, actionSet, r, γ; unit = unit)
                        a_dev = copy(a); a_dev[i] = aibar
                        c_dev = VertiportCost(i, a_dev, actionSet, r, γ; unit = unit)
                        add_to_expression!(dev_sum, c_dev - c_follow, z[k])
                    end
                    @constraint(model, dev_sum - zalpha * σi(i) * p_ai >= 0)
                end
            end
        end

        for i in 1:n, k in 1:d
            Ci_of_k[i, k] = VertiportCost(i, collect(Sset[k]), actionSet, r, γ; unit = unit)
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
    avgDelayScore = score / n
    cvals = [sum(Ci_of_k[i, k] * zval[k] for k in 1:d) for i in 1:n]
    fairScore = abs(maximum(cvals) - minimum(cvals)) / Δ
    giniScore = sum(abs(cvals[i] - cvals[j]) for i in 1:n-1 for j in i+1:n) / (2 * mean(cvals) * n^2)

    (; primals = zval, z = zval, Sset, score, avgDelayScore, fairScore, giniScore,
       d, varsize = d, buildTime, solverTime, status, m, n, actionSet, model)
end

# -----------------------------------------------------------------------------
# CC-PNE convex-hull LP.
# -----------------------------------------------------------------------------
function BruteNashBasedOptimizerLPVertiport(r::Int, n::Int, γ::Real, Δ;
                                             zalpha::Real, sigma, unit::Real = 5.0,
                                             optimizer = Ipopt.Optimizer, verbose = false)
    t_search = @elapsed (pneList, actionSet, m) = FindAllCCPNEVertiport(r, n, γ; zalpha = zalpha, sigma = sigma, unit = unit)
    d = length(pneList)
    d == 0 && error("No CC-PNE found at this (zalpha, sigma, γ); cannot build convex-hull LP.")
    println("$d CC-PNE found in $(round(t_search, digits=3))s. Solving hull LP.")

    model = Model(optimizer)
    verbose || set_silent(model)

    scoreSet = Matrix{Float64}(undef, d, n)
    buildTime = @elapsed begin
        for k in 1:d, i in 1:n
            scoreSet[k, i] = VertiportCost(i, collect(pneList[k]), actionSet, r, γ; unit = unit)
        end

        @variable(model, λv[1:d] >= 0)
        @constraint(model, sum(λv) == 1)
        c_exprs = [@expression(model, sum(scoreSet[k, i] * λv[k] for k in 1:d)) for i in 1:n]
        @objective(model, Min, _fairness_objective!(model, c_exprs, n, Δ))
    end

    solverTime = @elapsed optimize!(model)
    status = termination_status(model)
    λval = value.(λv)
    cvals = [sum(scoreSet[k, i] * λval[k] for k in 1:d) for i in 1:n]
    score = sum(cvals)
    avgDelayScore = score / n
    fairScore = abs(maximum(cvals) - minimum(cvals)) / Δ
    giniScore = sum(abs(cvals[i] - cvals[j]) for i in 1:n-1 for j in i+1:n) / (2 * mean(cvals) * n^2)

    (; λ = λval, pneList, scoreSet, score, avgDelayScore, fairScore, giniScore,
       d, varsize = d + n + 1, searchTime = t_search, buildTime, solverTime, status, m, n, actionSet, model)
end

# -----------------------------------------------------------------------------
# CC-PNE hull with NO equilibrium search at all.
#
# Proposition 1 of the reduced-joint-action note gives a closed-form certificate
#   q_α σ_i <= unit·min{1, γ−1}   for every i
# under which EVERY x ∈ F is a CC-PNE. When it holds there is nothing to search
# and nothing to verify: take P_d = F directly and solve the hull LP. The whole
# equilibrium-generation step collapses to n scalar comparisons, independent of
# both |X| = m^n and |F| = n^r.
#
# This is available only in the certified regime. Under partial certification the
# bound fails and only a subset of F is CC-PNE, so the per-profile check of
# HullLPOverCandidateSet is required instead -- hence the explicit error rather
# than a silent fallback.
# -----------------------------------------------------------------------------
function CertificateBound(γ::Real; unit::Real = 5.0)
    return unit * min(1.0, γ - 1.0)
end

function HullLPDirect(r::Int, n::Int, γ::Real, Δ;
                      zalpha::Real, sigma, unit::Real = 5.0,
                      optimizer = Ipopt.Optimizer, verbose = false)
    σi(i) = isa(sigma, Number) ? sigma : sigma[i]

    # The entire "search": n scalar comparisons against the Proposition 1 bound.
    t_search = @elapsed begin
        bound = CertificateBound(γ; unit = unit)
        certified = all(zalpha * σi(i) <= bound for i in 1:n)
    end
    certified || error("Proposition 1 certificate fails (q_α σ > $(CertificateBound(γ; unit=unit))); " *
                       "F is not uniformly CC-PNE, so the search cannot be skipped. " *
                       "Use HullLPOverCandidateSet, which verifies each candidate.")

    actionSet = BuildRunwayActionSet(r)
    m = length(actionSet)
    pneList, _ = ExclusiveJointSet(r, n)     # P_d = F, no membership check
    d = length(pneList)
    println("Certificate holds (q_α σ <= $(CertificateBound(γ; unit=unit))); taking P_d = F with $d profiles, no search.")

    model = Model(optimizer)
    verbose || set_silent(model)

    scoreSet = Matrix{Float64}(undef, d, n)
    buildTime = @elapsed begin
        for k in 1:d, i in 1:n
            scoreSet[k, i] = VertiportCost(i, collect(pneList[k]), actionSet, r, γ; unit = unit)
        end
        @variable(model, λv[1:d] >= 0)
        @constraint(model, sum(λv) == 1)
        c_exprs = [@expression(model, sum(scoreSet[k, i] * λv[k] for k in 1:d)) for i in 1:n]
        @objective(model, Min, _fairness_objective!(model, c_exprs, n, Δ))
    end

    solverTime = @elapsed optimize!(model)
    status = termination_status(model)
    λval = value.(λv)
    cvals = [sum(scoreSet[k, i] * λval[k] for k in 1:d) for i in 1:n]
    score = sum(cvals)
    fairScore = abs(maximum(cvals) - minimum(cvals)) / Δ
    giniScore = sum(abs(cvals[i] - cvals[j]) for i in 1:n-1 for j in i+1:n) / (2 * mean(cvals) * n^2)

    (; λ = λval, pneList, scoreSet, score, avgDelayScore = score / n, fairScore, giniScore,
       d, varsize = d + n + 1, searchTime = t_search, buildTime, solverTime, status, m, n, actionSet, model)
end

# -----------------------------------------------------------------------------
# CC-PNE hull, restricted to a candidate universe Sset (default: the
# exclusive-occupancy set F) instead of the full |X| = m^n joint action space.
# Still VERIFIES each candidate's CC-PNE membership (never skipped -- that
# check is what makes the no-deviation-constraint hull LP valid); it just
# checks membership over the much smaller |Sset| candidates rather than |X|.
# Falls back to whatever subset of Sset actually is CC-PNE (may be empty).
# -----------------------------------------------------------------------------
function HullLPOverCandidateSet(r::Int, n::Int, γ::Real, Δ, Sset = nothing;
                                 zalpha::Real, sigma, unit::Real = 5.0,
                                 optimizer = Ipopt.Optimizer, verbose = false)
    actionSet = BuildRunwayActionSet(r)
    m = length(actionSet)
    if isnothing(Sset)
        Sset, _ = ExclusiveJointSet(r, n)
    end

    t_search = @elapsed begin
        chk = CheckExclusiveSetCCPNEVertiport(Sset, actionSet, r, γ, n; zalpha = zalpha, sigma = sigma, unit = unit)
        pneList = Sset[chk.flags]
    end
    d = length(pneList)
    d == 0 && error("No CC-PNE found within the candidate set at this (zalpha, sigma, γ); cannot build hull LP.")
    println("$d / $(length(Sset)) candidates are CC-PNE, found in $(round(t_search, digits=3))s. Solving hull LP.")

    model = Model(optimizer)
    verbose || set_silent(model)

    scoreSet = Matrix{Float64}(undef, d, n)
    buildTime = @elapsed begin
        for k in 1:d, i in 1:n
            scoreSet[k, i] = VertiportCost(i, collect(pneList[k]), actionSet, r, γ; unit = unit)
        end

        @variable(model, λv[1:d] >= 0)
        @constraint(model, sum(λv) == 1)
        c_exprs = [@expression(model, sum(scoreSet[k, i] * λv[k] for k in 1:d)) for i in 1:n]
        @objective(model, Min, _fairness_objective!(model, c_exprs, n, Δ))
    end

    solverTime = @elapsed optimize!(model)
    status = termination_status(model)
    λval = value.(λv)
    cvals = [sum(scoreSet[k, i] * λval[k] for k in 1:d) for i in 1:n]
    score = sum(cvals)
    avgDelayScore = score / n
    fairScore = abs(maximum(cvals) - minimum(cvals)) / Δ
    giniScore = sum(abs(cvals[i] - cvals[j]) for i in 1:n-1 for j in i+1:n) / (2 * mean(cvals) * n^2)

    (; λ = λval, pneList, scoreSet, score, avgDelayScore, fairScore, giniScore,
       d, varsize = d + n + 1, searchTime = t_search, buildTime, solverTime, status, m, n, actionSet, model)
end
