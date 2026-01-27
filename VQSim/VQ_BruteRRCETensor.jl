using LinearAlgebra
using Statistics

# --- (기존) PNE 찾는 부분은 그대로 두고 is_pne / find_pne_set는 재사용 ---

function is_pne(
    k::Int,
    C_air::AbstractMatrix,
    joint_choice::Vector{Vector{Int}},
    choice_to_k::Dict{Tuple{Vararg{Int}},Int},
    action_sizes::Vector{Int};
    tol::Float64 = 0.0
)
    nP, K = size(C_air)
    base = joint_choice[k]
    for i in 1:nP
        c_base = C_air[i, k]
        for ai in 1:action_sizes[i]
            ai == base[i] && continue
            tmp = collect(Tuple(base))
            tmp[i] = ai
            k_dev = choice_to_k[Tuple(tmp)]
            c_dev = C_air[i, k_dev]
            if c_dev + tol < c_base
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
    tol::Float64 = 0.0
)
    _, K = size(C_air)
    pne_ks = Int[]
    for k in 1:K
        is_pne(k, C_air, joint_choice, choice_to_k, action_sizes; tol=tol) && push!(pne_ks, k)
    end
    return pne_ks
end

"""
Solve:  min_{λ∈Δ}  max_i (C_pne * λ)_i

C_pne: (nP x P) where P = #PNEs
Returns λ (length P) that typically mixes across PNEs.
"""
function solve_rrce_over_pne(
    C_air::AbstractMatrix,
    pne_ks::Vector{Int};
    max_iters::Int = 2000,
    step0::Float64 = 1.0,
    tol::Float64 = 1e-6
)
    nP, _ = size(C_air)
    P = length(pne_ks)
    @assert P > 0 "No PNE found; cannot solve RRCE over PNE hull."

    C_pne = C_air[:, pne_ks]  # nP x P

    # Initialize λ uniformly on simplex
    λ = fill(1.0 / P, P)

    # helper: projection onto simplex
    function proj_simplex(v::Vector{Float64})
        # Euclidean projection onto {x>=0, sum x = 1}
        u = sort(v, rev=true)
        cssv = cumsum(u)
        ρ = findlast(j -> u[j] + (1 - cssv[j]) / j > 0, 1:length(u))
        θ = (1 - cssv[ρ]) / ρ
        w = max.(v .+ θ, 0.0)
        s = sum(w)
        return s > 0 ? (w ./ s) : fill(1.0 / length(v), length(v))
    end

    # Track objective
    prev = Inf
    for it in 1:max_iters
        y = C_pne * λ                 # expected costs per player (nP)
        i_star = argmax(y)            # active constraint / worst-off player
        obj = y[i_star]

        # stopping
        if abs(prev - obj) < tol
            break
        end
        prev = obj

        # subgradient of max_i y_i at λ is C_pne[i_star, :]'
        g = vec(C_pne[i_star, :])

        # diminishing step (safe)
        α = step0 / sqrt(it)

        # projected subgradient step
        λ = proj_simplex(λ .- α .* g)
    end

    return (; λ=λ, pne_ks=pne_ks, C_pne=C_pne, status=:proj_subgrad)
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
