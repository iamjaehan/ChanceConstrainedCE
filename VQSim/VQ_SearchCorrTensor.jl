using StatsBase

"""
Build per-epoch game tensors for CE/NE solvers.

Inputs
- state: must have fields `t::Int`, `Q::Vector{Int}`, `released`
- flights: indexable, each flight has fields `airline_id::Int`, `runway::Int`, `sched_t::Int`, `pax`
- active_airlines: Vector{Int}, defines player ordering for this epoch
- active_by_airline: Dict{Int,Vector{Int}}, eligible flights per airline for this epoch
- actions_by_player: Vector of length nP; actions_by_player[p] is Vector{Vector{Int}} of flight-index subsets
- joint_pushed: Vector{Vector{Int}} length K; each is concatenated pushed flight indices
- joint_choice: Vector{Vector{Int}} length K; each is length nP, local action indices per player

Keyword params
- n_runways, beta_queue, rho_release passed into compute_costs

Returns NamedTuple with:
- C_air::Matrix{Float64} (nP x K)
- c_coord::Vector{Float64} (K)
- action_sizes::Vector{Int} (nP)
- choice_to_k::Dict{NTuple{nP,Int},Int}  (tuple(joint_choice[k]) -> k)
- plus passthrough fields for convenience
"""

function build_epoch_game_tensors(
  state,
  flights,
  active_airlines::Vector{Int},
  active_by_airline::Dict{Int,Vector{Int}},
  actions_by_player::Vector{Vector{Vector{Int}}},
  joint_pushed::Vector{Vector{Int}},
  joint_choice::Vector{Vector{Int}},
  params;
  n_runways::Int,
  beta_queue::Float64 = 1.0,
  rho_release::Float64 = 0.0
)
  nP = length(active_airlines)
  @assert length(actions_by_player) == nP "actions_by_player length mismatch (must equal #active players)"
  K = length(joint_pushed)
  @assert length(joint_choice) == K "joint_choice length mismatch with joint_pushed"

  # local action counts per player (for CE constraint enumeration)
  action_sizes = [length(actions_by_player[p]) for p in 1:nP]
  for p in 1:nP
      @assert action_sizes[p] >= 1 "Player $p has empty action set; cannot build CE game"
  end

  # Build mapping from joint_choice tuple -> joint action index k
  # Use NTuple for speed and correctness (no aliasing).
  # nP is small, so this is fine for now.
  choice_to_k = Dict{NTuple{0,Int},Int}()  # placeholder, will re-init below
  # Julia needs concrete NTuple length; handle nP via generated conversion:
  # We'll store as Tuple{Vararg{Int}} instead if you want simplicity.
  # But you requested NTuple{nP,Int}; we can create it as Dict{Tuple,Int} safely.

  choice_to_k_any = Dict{Tuple{Vararg{Int}}, Int}()

  for k in 1:K
      jc = joint_choice[k]
      @assert length(jc) == nP "joint_choice[$k] has wrong length"
      key = Tuple(jc)
      if haskey(choice_to_k_any, key)
          error("Duplicate joint_choice tuple at k=$k. key=$key already mapped to k=$(choice_to_k_any[key]).")
      end
      choice_to_k_any[key] = k
  end

  # Cost tensors
  C_air = zeros(Float64, nP, K)
  c_coord = zeros(Float64, K)

  # Fill costs by evaluating each joint action once
  for k in 1:K
      pushed = joint_pushed[k]

      out = VQCosts.compute_costs(
          state.Q, pushed, flights,
          active_airlines, active_by_airline, state.t, params;
          n_runways = n_runways,
          beta_queue = beta_queue,
          rho_release = rho_release
      )

      c_coord[k] = out.J_coord

      # airline costs in player order
      for p in 1:nP
          aid = active_airlines[p]
          # ensure key exists; if not, it's a bug in compute_costs initialization
          @assert haskey(out.J_air, aid) "compute_costs did not return J_air for active airline aid=$aid"
          C_air[p, k] = out.J_air[aid]
      end
  end

  # Quick sanity checks on mapping correctness (spot-check a few)
  if K > 0
      ktest = min(K, 3)
      for k in 1:ktest
          key = Tuple(joint_choice[k])
          @assert choice_to_k_any[key] == k "choice_to_k mismatch at k=$k"
      end
  end

  return (
      nP = nP,
      K = K,
      C_air = C_air,
      c_coord = c_coord,
      action_sizes = action_sizes,
      choice_to_k = choice_to_k_any,  # Tuple{Vararg{Int}} -> k
      actions_by_player = actions_by_player,
      joint_pushed = joint_pushed,
      joint_choice = joint_choice,
      active_airlines = active_airlines
  )
end

"""
Standard correlated equilibrium (CE) incentive constraints for cost-minimization games.

Inputs:
- z::Vector{Float64}: length K, distribution over joint actions (must be >=0, sum=1 handled elsewhere)
- C_air::Matrix{Float64}: size (nP, K), player costs at each joint action
- joint_choice::Vector{Vector{Int}}: length K, each is length nP, local action index per player
- choice_to_k::Dict{Tuple{Vararg{Int}},Int}: maps Tuple(joint_choice[k]) -> k
- action_sizes::Vector{Int}: length nP, number of local actions per player

Output:
- h::Vector{Float64}: CE inequalities in the form h >= 0
  For each player i, each recommended action a, each deviation a' != a:
    sum_{k: choice_i(k)=a} z_k * ( C_i(k_deviate) - C_i(k_follow) ) >= 0
"""

"""
Type-stable CE constraints: works for Numeric and Symbolics.Num.

Return h as Vector{T} where T = promote_type(eltype(z), eltype(C_air)).
No Float64 casting.
"""
function CalcH_Tensor(
    z::AbstractVector,
    C_air::AbstractMatrix,
    joint_choice::Vector{Vector{Int}},
    choice_to_k::Dict{Tuple{Vararg{Int}},Int},
    action_sizes::Vector{Int}
)
    nP, K = size(C_air)
    @assert length(z) == K
    @assert length(joint_choice) == K
    @assert length(action_sizes) == nP

    # Precompute ks by (i,a)
    ks_by_i_a = [ [Int[] for _ in 1:action_sizes[i]] for i in 1:nP ]
    for k in 1:K
        jc = joint_choice[k]
        @assert length(jc) == nP
        for i in 1:nP
            push!(ks_by_i_a[i][jc[i]], k)
        end
    end

    # #constraints
    nCE = 0
    for i in 1:nP
        Ai = action_sizes[i]
        nCE += Ai*(Ai-1)
    end

    # IMPORTANT: allocate with correct element type
    T = promote_type(eltype(z), eltype(C_air))
    h = Vector{T}(undef, nCE)

    idx = 1
    for i in 1:nP
        Ai = action_sizes[i]
        for a in 1:Ai
            ks = ks_by_i_a[i][a]
            for ap in 1:Ai
                ap == a && continue

                if isempty(ks)
                    h[idx] = zero(T)
                    idx += 1
                    continue
                end

                acc = zero(T)
                for k in ks
                    c_follow = C_air[i, k]

                    tmp = collect(Tuple(joint_choice[k]))
                    tmp[i] = ap
                    k_dev = choice_to_k[Tuple(tmp)]
                    c_dev = C_air[i, k_dev]

                    acc += z[k] * (c_dev - c_follow)
                end
                h[idx] = acc
                idx += 1
            end
        end
    end

    return h
end

"""
Epigraph constraints:
- T2: v - E[c] - Δ >= 0
- T3: w - v >= 0   (note: this is the "w >= v" direction, consistent with epigraph max)

Inputs:
- xi: primal vector [z; v; w]
- C_air: (nP x K)
- K, nP
- Δ: scalar or Vector{Float64} length nP
"""
function T2ConstTensor(xi, C_air, K::Int, nP::Int, Δ)
  z = xi[1:K]
  v = xi[K+1:K+nP]
  c = C_air * z
  if isa(Δ, Number)
      return v .- c .- Δ
  else
      @assert length(Δ) == nP
      return v .- c .- Δ
  end
end

function T3ConstTensor(xi, K::Int, nP::Int)
    v = xi[K+1:K+nP]
    w = xi[end]
    return w .- v
end

"""
Nonnegativity constraints for z: z >= 0  <=>  z >= 0
Return vector length K.
"""
function ZNonnegConst(xi, K::Int)
    z = xi[1:K]
    return z
end

"""
Objective: minimize sum(v) - sum(Δ) (optional constant shift)
"""
function CalcEFJTensor(xi, K::Int, nP::Int, Δ)
  v = xi[K+1:K+nP]
  if isa(Δ, Number)
      return sum(v) - nP * Δ
  else
      return sum(v) - sum(Δ)
  end
end

"""
CorrPackerTensor: returns inequality vector h(x) >= 0

Stack order:
1) CE constraints: CalcH_Tensor(z, ...)
2) T2: v - E[c] - Δ >= 0
3) T3: w - v >= 0
4) z >= 0
"""
function CorrPackerTensor(
    xi,
    C_air,
    joint_choice,
    choice_to_k,
    action_sizes;
    Δ = 0.0
)
    nP, K = size(C_air)
    z = xi[1:K]

    h_ce = CalcH_Tensor(z, C_air, joint_choice, choice_to_k, action_sizes)
    h_t2 = T2ConstTensor(xi, C_air, K, nP, Δ)
    h_t3 = T3ConstTensor(xi, K, nP)
    h_z  = ZNonnegConst(xi, K)

    return vcat(h_ce, h_t2, h_t3, h_z)
end

"""
Solve standard CE + epigraph for a per-epoch tensor game.

Inputs:
- C_air: (nP x K) airline cost matrix
- joint_choice, choice_to_k, action_sizes: indexing artifacts from Step A

Kwargs:
- Δ: scalar or length-nP vector, default 0.0
- z_init: optional initial distribution (length K). If not given, uniform.
- v_init: optional (length nP). If not given, uses expected costs under z_init.
- w_init: optional scalar. If not given, max(v_init).

Returns NamedTuple:
- primals, z, v, w, status, solverTime
"""
function SearchCorrTensor(
    C_air,
    joint_choice,
    choice_to_k,
    action_sizes;
    Δ = 0.0,
    z_init = nothing,
    v_init = nothing,
    w_init = nothing
)
    nP, K = size(C_air)

    # --- dimensions ---
    primal_dim = K + nP + 1
    eq_dim = 1

    # CE constraints count
    nCE = 0
    for i in 1:nP
        Ai = action_sizes[i]
        nCE += Ai * (Ai - 1)
    end
    ineq_dim = nCE + nP + nP + K

    # --- objective/equality/inequality in ParametricOptimizationProblem style ---
    f(x, θ) = CalcEFJTensor(x, K, nP, Δ)
    g(x, θ) = [sum(x[1:K]) - 1.0]  # simplex equality
    h(x, θ) = CorrPackerTensor(x, C_air, joint_choice, choice_to_k, action_sizes; Δ = Δ)

    problem = ParametricOptimizationProblem(;
        objective = f,
        equality_constraint = g,
        inequality_constraint = h,
        parameter_dimension = 1,
        primal_dimension = primal_dim,
        equality_dimension = eq_dim,
        inequality_dimension = ineq_dim,
    )

    # --- initializer (important for stability) ---
    if z_init === nothing
        z0 = fill(1.0 / K, K)
    else
        @assert length(z_init) == K
        z0 = collect(float.(z_init))
        # (optional) normalize defensively
        z0 ./= sum(z0)
    end

    if v_init === nothing
        c0 = C_air * z0
        if isa(Δ, Number)
            v0 = c0 .+ float(Δ)
        else
            v0 = c0 .+ Δ
        end
    else
        @assert length(v_init) == nP
        v0 = collect(float.(v_init))
    end

    if w_init === nothing
        w0 = maximum(v0)
    else
        w0 = float(w_init)
    end

    x0 = vcat(z0, v0, [w0])

    solverTime = @elapsed (; primals, variables, status, info) = solve(problem, [0.0])

    z = primals[1:K]
    v = primals[K+1:K+nP]
    w = primals[end]

    return (; primals, z, v, w, status, solverTime, info)
end

function sample_k(z::AbstractVector)
  w = Weights(z ./ sum(z))
  return sample(1:length(z), w)
end

"""
Given recommended joint_choice (per-player local action indices),
compute realized joint_choice by letting each player best-respond
under noisy airline costs, holding others fixed at recommendation.

Inputs:
- C_air_noisy: (nP x K) costs with noise added per-player per-joint-action
- joint_choice::Vector{Vector{Int}} length K
- choice_to_k::Dict maps joint choice tuple -> k
- action_sizes::Vector{Int}
- k_rec::Int recommended joint action index

Return:
- k_real::Int realized joint action index
"""
function realized_choice_conditional_BR(
    C_air_noisy,
    joint_choice,
    choice_to_k,
    action_sizes,
    k_rec::Int
)
    nP, K = size(C_air_noisy)
    rec = joint_choice[k_rec]
    choice = copy(rec)

    # sequential conditional BR (you can also do simultaneous argmin with tie-break)
    for i in 1:nP
        best_ai = choice[i]
        best_val = Inf

        for ai in 1:action_sizes[i]
            tmp = copy(choice)
            tmp[i] = ai
            k = choice_to_k[Tuple(tmp)]
            val = C_air_noisy[i, k]
            if val < best_val
                best_val = val
                best_ai = ai
            end
        end

        choice[i] = best_ai
    end

    return choice_to_k[Tuple(choice)]
end