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
RRCE over PNE set with:
- objective: minimize expected coordinator cost (c_coord ⋅ z)
- fairness-threshold constraint (gap): max_i E[u_i] - min_i E[u_i] <= Δ
  where u_i are computed from C_air_fair (recommended: unweighted avg airline delay).

Decision vector:
xi = [ λ(1:d); v(1:nP); w; m ]
  λ : mixture over PNEs (simplex)
  v : expected fairness-cost per airline (v = C_fair_pne * λ)
  w : max(v)
  m : min(v)

Equality constraints:
1) sum(λ) = 1
2) v - C_fair_pne * λ = 0   (nP eqs)

Inequality constraints (>= 0 convention):
- λ >= 0
- w - v_i >= 0          ∀i
- v_i - m >= 0          ∀i
- Δ - (w - m) >= 0
"""
function solve_rrce_over_pne(
    C_air_ic::AbstractMatrix,        # kept for backward compatibility / sanity (not used here)
    pne_ks::Vector{Int},
    K_total::Int;
    Δ::Real = 0.0,
    C_air_fair = nothing,            # (nP x K_total), REQUIRED
    c_coord = nothing                # (K_total,), REQUIRED
)
    d = length(pne_ks)
    @assert d > 0 "No PNE found; cannot solve RRCE subproblem."

    @assert C_air_fair !== nothing "solve_rrce_over_pne requires keyword C_air_fair (nP x K)."
    @assert c_coord !== nothing     "solve_rrce_over_pne requires keyword c_coord (length K)."

    nP, Kcheck = size(C_air_fair)
    @assert Kcheck == K_total "C_air_fair second dim must equal K_total."
    @assert length(c_coord) == K_total "c_coord length must equal K_total."

    # Restrict to PNE columns
    C_fair_pne = C_air_fair[:, pne_ks]     # (nP x d)
    c_pne      = c_coord[pne_ks]           # (d,)

    # ----- variable layout -----
    # xi = [λ(d); v(nP); w; m]
    primal_dim = d + nP + 2
    eq_dim     = 1 + nP

    # inequality: λ (d) + (w-v) (nP) + (v-m) (nP) + gap (1)
    ineq_dim   = d + 2nP + 1

    # ----- objective -----
    f(x, θ) = dot(c_pne, x[1:d])

    # ----- equalities -----
    function g(x, θ)
        λ = x[1:d]
        v = x[d+1 : d+nP]
        return vcat([sum(λ) - 1.0], v .- (C_fair_pne * λ))
    end

    # ----- inequalities (>= 0) -----
    function h(x, θ)
        λ = x[1:d]
        v = x[d+1 : d+nP]
        w = x[d+nP+1]
        m = x[d+nP+2]

        return vcat(
            λ,                 # λ >= 0
            (w .- v),          # w >= v
            (v .- m),          # v >= m
            [Δ - (w - m)]      # w - m <= Δ
        )
    end

    problem = ParametricOptimizationProblem(;
        objective = f,
        equality_constraint = g,
        inequality_constraint = h,
        parameter_dimension = 1,
        primal_dimension = primal_dim,
        equality_dimension = eq_dim,
        inequality_dimension = ineq_dim,
    )

    # ----- initializer -----
    # λ0 = fill(1.0 / d, d)
    # v0 = C_fair_pne * λ0
    # w0 = maximum(v0)
    # m0 = minimum(v0)
    # x0 = vcat(λ0, v0, [w0, m0])

    solverTime = @elapsed (; primals, variables, status, info) = solve(problem, [0.0])

    λ = primals[1:d]

    # build distribution over all K joint actions (mass only on pne_ks)
    z = zeros(Float64, K_total)
    for (j, k) in enumerate(pne_ks)
        z[k] = λ[j]
    end
    s = sum(z)
    z_use = s > 0 ? z ./ s : z

    return (; λ, z_use, primals, status, solverTime, pne_ks, C_fair_pne, c_pne, info)
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
