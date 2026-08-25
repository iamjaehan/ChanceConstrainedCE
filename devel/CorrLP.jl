using correlated
using BlockArrays: Block
using JuMP
import Ipopt

# =============================================================================
# CorrLP.jl
#
# CC-CE, reduced CC-CE and (later) the CC-PNE convex hull are all *linear*
# programs in the decision variable (z or λ): both the CE/CC-CE deviation
# constraints and the aggregate/fairness-threshold objectives are linear (the
# fairness-threshold objective is the standard epigraph LP reformulation of a
# max of affine functions -- see T1/T2/T3 in 0_GameSetup.jl). The existing
# solve path routes everything through ParametricOptimizationProblem, which
# builds a full symbolic KKT/complementarity system via Symbolics and compiles
# it to native code -- overhead that is unnecessary for a plain LP and that
# scales badly (slow symbolic construction, and PATH's cold-start sensitivity
# shows up as MCP_NoProgress on some reduced-set instances).
#
# This file solves the SAME full CC-CE problem as SearchCorr (CorrBasedOptimizer.jl)
# but directly as a JuMP LP, so it can be checked against SearchCorr's output on
# the existing scenario as a correctness cross-check, and timed against it.
# =============================================================================

"""
Solve the full CC-CE problem (same feasible set as SearchCorr / CorrPacker) as
a plain LP via JuMP.

`objective`:
- `:aggregate` -- minimize expected total cost Σ_i c_i(z)  (matches SearchCorr's
  default objective, i.e. the original CC-CE paper's J_sys, Eq. 23)
- `:fair`      -- minimize the RRCE fairness-threshold objective (Eq. 20),
  via the same w/v epigraph auxiliary variables as CorrPacker's T1/T2/T3.
"""
function SearchCorrLP(r::Int, n::Int, λ, Δ;
                       zalpha::Real, sigma, mult = 2.0,
                       objective::Symbol = :aggregate,
                       optimizer = Ipopt.Optimizer,
                       verbose = false)
    println("Begin LP Corr Search for m=$(2^r) and n=$n case.")
    C = SetC(r, n, λ; mult = mult)
    m = size(C[Block(1)])[1]
    l = m^n

    CI = CartesianIndices(ntuple(_ -> m, n))
    LI = LinearIndices(CI)
    σi(i) = isa(sigma, Number) ? sigma : sigma[i]

    model = Model(optimizer)
    verbose || set_silent(model)

    @variable(model, z[1:l] >= 0)
    @constraint(model, sum(z) == 1)

    # ---- CE / CC-CE deviation constraints (same math as CalcH) ----
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
                    c_follow = J_def(i, a, C)
                    a_dev = copy(a); a_dev[i] = aibar
                    c_dev = J_def(i, a_dev, C)
                    add_to_expression!(dev_sum, c_dev - c_follow, z[k])
                end
                @constraint(model, dev_sum - zalpha * σi(i) * p_ai >= 0)
            end
        end
    end

    # ---- per-player expected cost c_i(z), precomputed cost lookup ----
    Ci_of_k = [J_def(i, collect(Tuple(CI[k])), C) for i in 1:n, k in 1:l]  # (n x l)
    c_expr(i) = @expression(model, sum(Ci_of_k[i, k] * z[k] for k in 1:l))

    if objective == :aggregate
        total_of_k = [sum(Ci_of_k[i, k] for i in 1:n) for k in 1:l]
        @objective(model, Min, sum(total_of_k[k] * z[k] for k in 1:l))
    elseif objective == :fair
        @variable(model, v[1:n])
        @variable(model, w)
        for i in 1:n
            ci = c_expr(i)
            @constraint(model, w >= ci)
            @constraint(model, v[i] >= ci + Δ)
            @constraint(model, v[i] >= w)
        end
        @objective(model, Min, sum(v) - n * Δ)
    else
        error("objective must be :aggregate or :fair")
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
       varsize = l, solverTime, status, m, n, l, C, model)
end

# -----------------------------------------------------------------------------
# Fairness-threshold epigraph (RRCE Eq. 20), shared by the reduced-domain and
# hull LPs below. Adds v[1:n], w and returns Σv - nΔ (the objective expression);
# caller does @objective(model, Min, <returned expr>).
# -----------------------------------------------------------------------------
function _fairness_objective!(model, c_exprs, n::Int, Δ)
    v = @variable(model, [1:n])
    w = @variable(model)
    for i in 1:n
        @constraint(model, w >= c_exprs[i])
        @constraint(model, v[i] >= c_exprs[i] + Δ)
        @constraint(model, v[i] >= w)
    end
    return sum(v) - n * Δ
end

# -----------------------------------------------------------------------------
# Reduced CC-CE as a plain LP (same feasible set as SearchCorrReduced /
# ReducedPacker in ReducedActionSet.jl, restricted to a candidate set Sset --
# default: the exclusive-occupancy set). Deviations a_i' still range over the
# full 1:m regardless of Sset, matching CalcHReducedSet.
# -----------------------------------------------------------------------------
function SearchCorrReducedLP(r::Int, n::Int, λ, Δ, Sset = nothing;
                              zalpha::Real, sigma, mult = 2.0,
                              objective::Symbol = :aggregate,
                              optimizer = Ipopt.Optimizer,
                              verbose = false)
    C = SetC(r, n, λ; mult = mult)
    m = size(C[Block(1)])[1]

    if isnothing(Sset)
        Sset, _ = ExclusiveJointSet(r, n)
    end
    d = length(Sset)
    println("Begin LP reduced Corr Search for m=$m and n=$n case (|F|=$d).")

    σi(i) = isa(sigma, Number) ? sigma : sigma[i]

    model = Model(optimizer)
    verbose || set_silent(model)

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
                    c_follow = J_def(i, a, C)
                    a_dev = copy(a); a_dev[i] = aibar
                    c_dev = J_def(i, a_dev, C)
                    add_to_expression!(dev_sum, c_dev - c_follow, z[k])
                end
                @constraint(model, dev_sum - zalpha * σi(i) * p_ai >= 0)
            end
        end
    end

    Ci_of_k = [J_def(i, collect(Sset[k]), C) for i in 1:n, k in 1:d]

    if objective == :aggregate
        total_of_k = [sum(Ci_of_k[i, k] for i in 1:n) for k in 1:d]
        @objective(model, Min, sum(total_of_k[k] * z[k] for k in 1:d))
    elseif objective == :fair
        c_exprs = [@expression(model, sum(Ci_of_k[i, k] * z[k] for k in 1:d)) for i in 1:n]
        @objective(model, Min, _fairness_objective!(model, c_exprs, n, Δ))
    else
        error("objective must be :aggregate or :fair")
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
       d, varsize = d, solverTime, status, m, n, C, model)
end

# -----------------------------------------------------------------------------
# CC-PNE convex-hull LP: same CC-PNE set as BruteNashBasedOptimizer
# (via SolveNashBrute), but the mixture-over-PNEs LP is solved directly with
# JuMP instead of routing through ParametricOptimizationProblem/Symbolics --
# that symbolic construction+compile step was the actual bottleneck diagnosed
# earlier (the CC-PNE search itself is fast; d ~100 PNEs made the old
# symbolic-KKT build take minutes). Never materializes the full m^n joint
# distribution (the old code did, via nashSet*λ), so this also scales better
# in memory for large n.
# -----------------------------------------------------------------------------
function BruteNashBasedOptimizerLP(r::Int, n::Int, λ, Δ;
                                    zalpha::Real, sigma, mult = 2.0,
                                    optimizer = Ipopt.Optimizer,
                                    verbose = false)
    C = SetC(r, n, λ; mult = mult)
    m = size(C[Block(1)])[1]

    t_search = @elapsed (nashList, nashIdxList) = SolveNashBrute(C, m, n; zalpha = zalpha, sigma = sigma)
    d = length(nashList)
    d == 0 && error("No CC-PNE found at this (zalpha, sigma, mult); cannot build convex-hull LP.")
    println("$d CC-PNE found in $(round(t_search, digits=3))s. Solving hull LP.")

    scoreSet = [J_def(i, collect(nashIdxList[k]), C) for k in 1:d, i in 1:n]  # (d x n)

    model = Model(optimizer)
    verbose || set_silent(model)

    @variable(model, λv[1:d] >= 0)
    @constraint(model, sum(λv) == 1)

    c_exprs = [@expression(model, sum(scoreSet[k, i] * λv[k] for k in 1:d)) for i in 1:n]
    @objective(model, Min, _fairness_objective!(model, c_exprs, n, Δ))

    solverTime = @elapsed optimize!(model)
    status = termination_status(model)

    λval = value.(λv)
    cvals = [sum(scoreSet[k, i] * λval[k] for k in 1:d) for i in 1:n]
    score = sum(cvals)
    avgDelayScore = score / n
    fairScore = abs(maximum(cvals) - minimum(cvals)) / Δ
    giniScore = sum(abs(cvals[i] - cvals[j]) for i in 1:n-1 for j in i+1:n) / (2 * mean(cvals) * n^2)

    (; λ = λval, nashList, nashIdxList, scoreSet, score, avgDelayScore, fairScore, giniScore,
       d, varsize = d + n + 1, solverTime, searchTime = t_search, status, m, n, C, model)
end
