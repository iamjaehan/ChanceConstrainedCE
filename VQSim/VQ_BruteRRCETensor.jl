using LinearAlgebra
using Statistics

function is_pne(
    k::Int,
    C_air::AbstractMatrix,
    joint_choice::Vector{Vector{Int}},
    choice_to_k::Dict{Tuple{Vararg{Int}},Int},
    action_sizes::Vector{Int};
    tol::Float64 = 0.0,
    zalpha::Real = 0.0,
    sigma = 0.0
)
    nP, K = size(C_air)
    base = joint_choice[k]

    use_uapne = (zalpha != 0.0) && (sigma != 0.0)
    sigma_i(i) = (sigma isa Number) ? sigma : sigma[i]
    kappa_i(i) = zalpha * sigma_i(i)

    for i in 1:nP
        c_base = C_air[i, k]

        # UA-PNE: if any deviation is within kappa of base (or better), fail.
        # Standard PNE: if any deviation is strictly better (by tol), fail.
        κ = use_uapne ? kappa_i(i) : 0.0

        for ai in 1:action_sizes[i]
            ai == base[i] && continue

            tmp = collect(Tuple(base))
            tmp[i] = ai
            k_dev = choice_to_k[Tuple(tmp)]
            c_dev = C_air[i, k_dev]

            # Standard PNE violation (better deviation exists)
            if c_dev + tol < c_base
                return false
            end

            # UA-PNE violation (gap not at least κ)
            if use_uapne && (c_dev < c_base + κ - tol)
                return false
            end
        end
    end
    return true
end

function find_pne_set(
    C_air::AbstractMatrix,
    joint_choice::Vector{Vector{Int}},
    choice_to_k::Dict{Tuple{Vararg{Int}},Int},
    action_sizes::Vector{Int};
    tol::Float64 = 0.0,
    zalpha::Real = 0.0,
    sigma = 0.0
)
    _, K = size(C_air)
    pne_ks = Int[]
    for k in 1:K
        is_pne(k, C_air, joint_choice, choice_to_k, action_sizes; tol=tol, zalpha = zalpha, sigma = sigma) && push!(pne_ks, k)
    end
    return pne_ks
end

"""
Objective identical to CalcEFJ:  1'v - n*Δ
Decision vector layout: xi = [λ(1:d); v(1:n); w]
"""
function CalcEFJ_PNE(xi, d::Int, n::Int, Δ::Real)
    v = xi[d+1 : d+n]
    return sum(v) - n*Δ
end

"""
PNE epigraph constraints (same pattern as old T1/T2/T3 + simplex nonnegativity).

Let c = C_pne * λ  (expected costs per player under mixture over PNEs)

Return inequalities intended as >= 0 (same sign convention as your older packers).
- simplex nonneg: λ >= 0
- T1: w - c >= 0
- T2: v - c - Δ >= 0
- T3: v - w >= 0
"""
function PNEPacker(xi, C_pne::AbstractMatrix, d::Int, n::Int, Δ::Real)
    λ = xi[1:d]
    v = xi[d+1 : d+n]
    w = xi[d+n+1]

    c = C_pne * λ  # length n

    return [
        λ;                # λ >= 0
        (w .- c);         # w >= c
        (v .- c .- Δ);    # v >= c + Δ
        (v .- w)          # v >= w
    ]
end

"""
Solve the same subproblem structure as old SolveBruteSubProblem, but restricted to PNE columns.

Inputs:
- C_air: (n x K) cost tensor flattened over joint actions
- pne_ks: indices of PNE joint actions (length d)
Returns:
- λ, z_use (length K; mass only on pne_ks), and primals for debugging.
"""
function solve_rrce_over_pne(
    C_air::AbstractMatrix,
    pne_ks::Vector{Int},
    K_total::Int;
    Δ::Real = 0.0
)
    n, _ = size(C_air)
    d = length(pne_ks)
    @assert d > 0 "No PNE found; cannot solve RRCE subproblem."

    C_pne = C_air[:, pne_ks]  # (n x d)

    f(x, θ) = CalcEFJ_PNE(x, d, n, Δ)
    g(x, θ) = [sum(x[1:d]) - 1.0]  # simplex sum-to-1
    h(x, θ) = PNEPacker(x, C_pne, d, n, Δ)

    problem = ParametricOptimizationProblem(;
        objective = f,
        equality_constraint = g,
        inequality_constraint = h,
        parameter_dimension = 1,
        primal_dimension = d + n + 1,
        equality_dimension = 1,
        inequality_dimension = d + 3n,
    )

    solverTime = @elapsed (; primals, variables, status, info) = solve(problem, [0])

    λ = primals[1:d]

    # build distribution over all K joint actions (mass only on pne_ks)
    z = zeros(Float64, K_total)
    for (j, k) in enumerate(pne_ks)
        z[k] = λ[j]
    end
    s = sum(z)
    z_use = s > 0 ? z ./ s : z

    return (; λ, z_use, primals, status, solverTime, pne_ks, C_pne)
end

function pne_to_distribution(pne_ks::Vector{Int}, λ::Vector{Float64}, K::Int)
    z = zeros(Float64, K)
    for (j, k) in enumerate(pne_ks)
        z[k] += λ[j]
    end
    s = sum(z)
    return s > 0 ? (z ./ s) : z
end

function sanity_check_mapping(joint_choice, choice_to_k, action_sizes)
    K = length(joint_choice)
    nP = length(action_sizes)

    @assert length(joint_choice[1]) == nP "joint_choice player dim mismatch"

    for k in 1:K
        base = joint_choice[k]
        @assert length(base) == nP
        @assert all(1 .<= base .<= action_sizes) "base choice out of range at k=$k: $base"

        # check all unilateral deviations are mapped
        for i in 1:nP
            for ai in 1:action_sizes[i]
                ai == base[i] && continue
                tmp = collect(base)
                tmp[i] = ai
                key = Tuple(tmp)
                if !haskey(choice_to_k, key)
                    error("Missing mapping for deviation. k=$k, i=$i, base=$base, dev=$key")
                end
            end
        end
    end

    println("sanity_check_mapping: OK (all unilateral deviations mapped).")
    return true
end

function max_regret_per_k(
    C_air::AbstractMatrix,
    joint_choice::Vector{Vector{Int}},
    choice_to_k::Dict{Tuple{Vararg{Int}},Int},
    action_sizes::Vector{Int};
    tol::Float64=0.0
)
    nP, K = size(C_air)
    max_regret = fill(-Inf, K)       # positive means profitable deviation exists
    arg_player = fill(0, K)
    best_dev_ai = fill(0, K)
    best_dev_k  = fill(0, K)

    for k in 1:K
        base = joint_choice[k]
        best_gain = -Inf
        best_i = 0
        best_ai = 0
        best_kdev = 0

        for i in 1:nP
            c_base = C_air[i, k]
            # find best deviation for player i
            c_best = c_base
            ai_best = base[i]
            kdev_best = k

            for ai in 1:action_sizes[i]
                ai == base[i] && continue
                tmp = collect(base); tmp[i] = ai
                k_dev = choice_to_k[Tuple(tmp)]
                c_dev = C_air[i, k_dev]
                if c_dev < c_best
                    c_best = c_dev
                    ai_best = ai
                    kdev_best = k_dev
                end
            end

            gain = c_base - c_best  # >0 means improvement
            if gain > best_gain
                best_gain = gain
                best_i = i
                best_ai = ai_best
                best_kdev = kdev_best
            end
        end

        max_regret[k] = best_gain
        arg_player[k] = best_i
        best_dev_ai[k] = best_ai
        best_dev_k[k]  = best_kdev
    end

    return (max_regret=max_regret, arg_player=arg_player, best_dev_ai=best_dev_ai, best_dev_k=best_dev_k)
end

function summarize_regrets(diag; tol=1e-9, top=10)
    mr = diag.max_regret
    K = length(mr)

    pne = findall(k -> mr[k] <= tol, 1:K)
    println("K = $K")
    println("#PNE (max_regret <= $tol): ", length(pne))

    # show smallest regrets (closest to equilibrium)
    ord = sortperm(mr)
    println("\nTop-$top closest-to-PNE actions (k, max_regret, worst_player):")
    for j in 1:min(top, K)
        k = ord[j]
        println("  k=$k  max_regret=$(mr[k])  worst_i=$(diag.arg_player[k])  best_dev_ai=$(diag.best_dev_ai[k])  best_dev_k=$(diag.best_dev_k[k])")
    end
    return pne
end
