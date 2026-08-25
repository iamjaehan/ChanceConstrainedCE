using correlated
using BlockArrays: Block

function PrepNashBrute(r,n,λ; mult = mult)
    C = SetC(r,n,λ; mult = mult)

    n = blocksize(C)[1] # Number of vehicles
    m = size(C[Block(1)])[1] # Number of actions

    return (; C, m, n)
end

function ExtractNash(aSeq,i,C,m,n)
    # aSeq in vector{Tuple{}} format
    NashIdxList = Vector{Any}(undef,0)
    for j in 1:length(aSeq) # Get BR for specific actionSet
        C_local = zeros(m)
        action = CartesianIndex(aSeq[j]) # jth aSeq
        for k in 1:n # k is opponent player's index
            if k != i
                C_local += C[Block(i,k)][:,action[k]]
            end
        end
        br = argmin(C_local)
        if action[i] == br || action[i] == 0
            action_v = collect(Iterators.flatten(aSeq[j]))
            action_v[i] = br
            localNash = Tuple(Int64(x) for x in action_v)
            NashIdxList = vcat(NashIdxList,[localNash])
        end
    end
    return NashIdxList
end

function ExtractUAPNE(aSeq, i, C, m, n, kappa; tol=1e-12)
    NashIdxList = Vector{Any}(undef,0)

    for j in 1:length(aSeq)
        C_local = zeros(m)
        action = CartesianIndex(aSeq[j])

        # build cost vector over ai
        for k in 1:n
            if k != i
                C_local .+= C[Block(i,k)][:, action[k]]
            end
        end

        cur = action[i]
        br  = argmin(C_local)

        # if action[i]==0 means "wildcard" in your pipeline
        if cur == 0
            # for wildcard, pick a BR, but ALSO require margin for that BR
            cur = br
        end

        # 1) must be (one of) best responses
        if C_local[cur] > C_local[br] + tol
            continue
        end

        # 2) margin condition: best deviation excluding cur
        second_best = Inf
        for a in 1:m
            a == cur && continue
            second_best = min(second_best, C_local[a])
        end

        if second_best < C_local[cur] + kappa - tol
            continue
        end

        # keep candidate
        action_v = collect(Iterators.flatten(aSeq[j]))
        action_v[i] = cur
        localNash = Tuple(Int64(x) for x in action_v)
        NashIdxList = vcat(NashIdxList,[localNash])
    end

    return NashIdxList
end


function SolveNashBrute(C, m, n; zalpha = 0.0, sigma = 0.0)
    NashIdxList = Vector{Any}(undef, 0)

    for i in 1:n
        if isempty(NashIdxList)
            aSeq = generateAseq(i, 0, m, n)
        else
            aSeq = NashIdxList
        end

        sigma_i = isa(sigma, Number) ? sigma : sigma[i]
        kappa = zalpha * sigma_i
        NashIdxList = ExtractUAPNE(aSeq, i, C, m, n, kappa)
    end

    NashList = Vector{Vector{Float64}}()
    for idx in NashIdxList
        jointPrimal = zeros(m^n)
        jointPrimal = reshape(jointPrimal, ntuple(_ -> m, n))
        jointPrimal[CartesianIndex(idx)] = 1.0
        push!(NashList, reshape(jointPrimal, m^n))
    end

    return NashList, NashIdxList
end

using Random

function SearchNashBrute(r, n, λ, Δ, sigma; seed = 1, pick_mode = :random, mult = mult)
    println("Begin brute Nash search for m=$(2^r) and n=$n case.")
    (; C, m, n) = PrepNashBrute(r, n, λ; mult = mult)

    # nominal NE: sigma = 0
    nash_list, nash_idx_list = SolveNashBrute(C, m, n; zalpha = 0.0, sigma = 0.0)

    if isempty(nash_list)
        return (;
            success = false,
            message = "No PNE found",
            primals = nothing,
            joint_index = nothing,
            score = NaN,
            avgDelayScore = NaN,
            fairScore = NaN,
            giniScore = NaN
        )
    end

    chosen_idx = if pick_mode == :first
        1
    elseif pick_mode == :random
        rng = MersenneTwister(seed)
        rand(rng, 1:length(nash_list))
    else
        error("pick_mode must be :first or :random")
    end

    jointPrimal = nash_list[chosen_idx]

    score = CalcJ(jointPrimal, C, m, n)
    avgDelayScore = EvalAverageDelay(jointPrimal, C, m, n)
    fairScore = EvalFairness(jointPrimal, C, m, n, Δ)
    # giniScore = EvalGini(jointPrimal, C, m, n)

    return (;
        success = true,
        primals = jointPrimal,
        joint_index = nash_idx_list[chosen_idx],
        score = score,
        avgDelayScore = avgDelayScore,
        fairScore = fairScore,
        num_pne = length(nash_list)
    )
end