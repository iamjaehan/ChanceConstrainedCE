using correlated
using BlockArrays: Block

function PrepNashBrute(r,n,λ)
    C = SetC(r,n,λ)

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


function SolveNashBrute(C,m,n; zalpha, sigma)
    NashIdxList = Vector{Any}(undef,0)
    NashList = Vector{Any}(undef,0)
    for i in 1:n # Filter BR for i
        if isempty(NashIdxList)
            aSeq = generateAseq(i,0,m,n)
        else
            aSeq = NashIdxList
        end
        # NashIdxList = ExtractNash(aSeq, i, C, m, n)
        kappa = zalpha * sigma
        NashIdxList = ExtractUAPNE(aSeq, i, C, m, n, kappa)
    end
    
    for i in 1:size(NashIdxList)[1]
        jointPrimal = zeros(m^n,1)
        jointPrimal = reshape(jointPrimal,ntuple(i->m,n))
        jointPrimal[CartesianIndex(NashIdxList[i])] = 1
        jointPrimal = reshape(jointPrimal,m^n,1)
        NashList = vcat(NashList,[jointPrimal])
    end
    return NashList
end

function SearchNashBrute(r,n,λ)
    println("Begin Nash Search for m=$(2^r) and n=$n case.")
    (; C, m, n) = PrepNash(r,n,λ)
    (; primals, score, varsize, solverTime) = SolveNashBrute(C,m,n)

    jointPrimal = SwitchIndtoJoint(primals,m,n)
    fairScore = EvalFairness(jointPrimal,C,m,n,Δ)
    giniScore = EvalGini(jointPrimal,C,m,n,Δ)
    avgDelayScore = EvalAverageDelay(jointPrimal,C,m,n)
    
    (; primals, fairScore, giniScore, avgDelayScore, varsize, solverTime)
end 