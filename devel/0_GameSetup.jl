using correlated
using BlockArrays
using Combinatorics
using Statistics
using Random

function SetC(r,n,λ)
    # Parameter setting
    # r Number of runways
    # l Number of sequencing legs
    # d Number of departure terminals
    dF = 300 # Delay factor
    eF = dF*2 # Hazard factor
    sF = dF # All stop factor

    # Number of actions and players
    n = n # number of players = l+d
    m = 2^r # number of actions

    # Action set - DON'T TOUCH this
    primeAction = "['G','S']"
    stringUnit = (primeAction*",")^(r-1)*primeAction
    stringSum = "vec(collect(Iterators.product("*stringUnit*")))"
    global actionSet = eval(Meta.parse(stringSum))

    primeNum = "[1,0]"
    numUnit = (primeNum*",")^(r-1)*primeNum
    numSum = "vec(collect(Iterators.product("*numUnit*")))"
    global numActionSet = collect.(eval(Meta.parse(numSum)))

    # Check validity
    if (length(actionSet)!=m)
        println("WARNING::ActionSet Fault")
    end

    # Construct cost matrix C
    C = BlockArray(zeros(n*m,n*m),vec(m*ones(Int8,(n,1))),vec(m*ones(Int8,(n,1))))
    for i in 1:n
        for j in 1:n
            # Construct local C
            C_local = zeros(m,m)
            if i != j
                for ii in 1:m
                    for jj in 1:m
                        # Compare action by actions
                        cost = 0
                        for rr in 1:r
                            myAction = actionSet[ii][rr]
                            opAction = actionSet[jj][rr]
                            if (myAction == 'G')
                                if (myAction == opAction)
                                    cost += eF * λ[i]
                                else
                                    cost += 0
                                end
                            elseif (myAction == 'S')
                                if (myAction == opAction)
                                    cost += sF * λ[i]
                                else
                                    cost += dF * λ[i]
                                end
                            end
                        end
                        C_local[ii,jj] = cost
                    end
                end
            end
            C[Block(i,j)] = C_local
        end
    end
    return C
end

function generateAseq(i, ai, m, n)
    ranges = ntuple(k -> (i != 0 && k == i) ? (ai:ai) : (1:m), n)
    return collect(Iterators.product(ranges...))  # iterator
end

function CalcPhi(i,ai,_ai,x,m,n,C)
    aSeq = generateAseq(i,ai,m,n) # (i,ai)
    _aSeq = generateAseq(i,_ai,m,n) # (i,_ai)
    l = length(aSeq)
    phi = 0
    for j in 1:l
        # zValue = x[CartesianIndex(aSeq[j])]
        # aSeqV = [k for k in _aSeq[j]]
        # for ii in 1:n
        #     aSeqV[ii] = aSeqV[ii] + m*(ii-1)
        # end
        # cValue = sum(C[m*(i-1)+_ai,aSeqV])
        # phi += zValue * cValue
        cost = 0.0
        a = _aSeq[j]
        for j in 1:n
            cost += C[Block(i,j)][_ai, a[j]]
        end
        phi += x[CartesianIndex(aSeq[j])] * cost
    end
    return phi
end

function CalcJ(x, C, m, n)
    x = reshape(x,ntuple(i->m,n))
    aSeq = generateAseq(0,0,m,n) #Generates all the possible actions
    l = length(aSeq)
    J = 0
    for i in 1:l
        zValue = x[CartesianIndex(aSeq[i])]
        aSeqV = [k for k in aSeq[i]]
        for ii in 1:n
            aSeqV[ii] = aSeqV[ii] + m*(ii-1)
        end
        cValue = sum(C[aSeqV,aSeqV])
        J += zValue * cValue
    end
    return J
end

function CalcIndividualJ(x,idx,C,m,n)
    x = reshape(x,ntuple(i->m,n))
    aSeq = generateAseq(0,0,m,n) #Generates all the possible actions
    l = length(aSeq)
    J = 0
    for i in 1:l
        zValue = x[CartesianIndex(aSeq[i])]
        aSeqV = [k for k in aSeq[i]]
        for ii in 1:n
            aSeqV[ii] = aSeqV[ii] + m*(ii-1)
        end
        cValue = sum(C[aSeqV[idx],aSeqV])
        J += zValue * cValue
    end
    return J
end

function CalcMarginalP(i, ai, x_f, m, n)
    aSeq = generateAseq(i, ai, m, n)
    p = zero(eltype(x_f))
    for j in eachindex(aSeq)
        p += x_f[CartesianIndex(aSeq[j])]
    end
    return p
end

function CalcH(x, m, n, C; zalpha, sigma, zero_sigma_ce_keys = Set{Tuple{Int,Int,Int}}())
    x_f = reshape(x, ntuple(i -> m, n))

    T = eltype(x)
    out = Vector{T}(undef, m^n + m^2*n - n*m)

    out[1:length(x)] = x

    c = length(x)
    for i in 1:n
        for ai in 1:m
            p_ai = CalcMarginalP(i, ai, x_f, m, n)
            base = CalcPhi(i, ai, ai, x_f, m, n, C)

            for _ai in 1:m
                if ai == _ai
                    continue
                end
                c += 1

                mean_diff = base - CalcPhi(i, ai, _ai, x_f, m, n, C)

                σi = isa(sigma, Number) ? sigma : sigma[i]

                sigma_eff = ((i, ai, _ai) in zero_sigma_ce_keys) ? zero(σi) : σi

                margin = zalpha * sigma_eff * p_ai

                out[c] = -(mean_diff + margin)
            end
        end
    end

    return out
end

function SwitchIndtoJoint(primals,m,n)
    aSeq = generateAseq(0,0,m,n)
    l = length(aSeq)
    jointPrimal = zeros(m^n,1)
    jointPrimal = reshape(jointPrimal,ntuple(i->m,n))
    for i in 1:l
        localIndex = CartesianIndex(aSeq[i])
        jointProb = 1
        for j in 1:length(localIndex)
            jointProb *= primals[j][localIndex[j]]
        end
        jointPrimal[localIndex] = jointProb
    end
    return reshape(jointPrimal,m^n)
end

function CalcJNash(C,primals,m,n)
    jointPrimal = SwitchIndtoJoint(primals,m,n)
    J = CalcJ(jointPrimal,C,m,n)
    return J
end

function CalcJNashSet(λ,nashSet,C, m, n)
    # jointProb = sum(nashSet.*λ,dims=1)[1]
    jointProb = nashSet*λ
    return CalcJ(jointProb, C, m ,n)
end

function SmoothMax(x)
    N = 10
    E = exp.(x*N)
    return ( x'*E ) / sum(E)
end

function CalcEFJ(xi, l, n, Δ)
    v = xi[l+1:end-1]
    J = -n*Δ + ones(n)'*v
    return J
end

function T1Const(xi,C,m,n,l)
    x = xi[1:l]
    c = Vector{Any}(undef,n)
    for i in 1:n
        c[i] = CalcIndividualJ(x, i, C, m ,n)
    end
    w = xi[end]
    return w .- c
end

function T2Const(xi,C,m,n,l,Δ)
    x = xi[1:l]
    c = Vector{Any}(undef,n)
    for i in 1:n
        c[i] = CalcIndividualJ(x, i, C, m ,n)
    end
    v = xi[l+1:end-1]
    return v - c .- Δ
end

function T3Const(xi,l)
    v = xi[l+1:end-1]
    w = xi[end]
    return v .- w
end

function CorrPacker(x,C,m,n,l,Δ; zalpha, sigma, zero_sigma_ce_keys = Set{Tuple{Int,Int,Int}}())
    out = [CalcH(x[1:l], m, n, C;
                 zalpha = zalpha,
                 sigma = sigma,
                 zero_sigma_ce_keys = zero_sigma_ce_keys);
           T1Const(x,C,m,n,l);
           T2Const(x,C,m,n,l,Δ);
           T3Const(x,l)]
    return out
end

function T1ConstN(xi,jointScore,C,m,n)
    c = jointScore
    w = xi[end]
    return w .- c
end

function T2ConstN(xi,jointScore,C,m,n,l,Δ)
    c = jointScore
    v = xi[l+1:end-1]
    return v - c .- Δ
end

function NashPacker(x,scoreSet,C,m,n,l,Δ)
    λ = x[1:l]
    jointScore = scoreSet'*λ
    out = [x[1:l];
    T1ConstN(x,jointScore,C,m,n);
    T2ConstN(x,jointScore,C,m,n,l,Δ);
    T3Const(x,l)]
    return out
end

function EvalFairness(primals,C,m,n,Δ)
    c = Vector{Float64}(undef,n)
    for i in 1:n
        c[i] = CalcIndividualJ(primals, i, C, m, n)
    end
    return abs(maximum(c)-minimum(c))/Δ
end

function EvalGini(primals,C,m,n,Δ)
    c = Vector{Float64}(undef,n)
    for i in 1:n
        c[i] = CalcIndividualJ(primals, i, C, m, n)
    end
    us = 0
    for i in 1:n-1
        for j in i+1:n
            us += abs(c[i]-c[j])
        end
    end
    return us/(2*mean(c)*n^2)
end

# function EvalMaxCostDiff(primals, C, m, n)
#     c = Vector{Any}(undef,n)
#     for i in 1:n
#         c[i] = CalcIndividualJ(primals, i, C, m, n)
#     end
#     return abs(maximum(c)-minimum(c))/Δ
# end

function EvalAverageDelay(primals, C, m, n)
    return CalcJ(primals, C, m, n)/n
end

function J_def(i, a, C)
    # a is a Vector{Int} of length n
    n = length(a)
    s = 0.0
    for j in 1:n
        s += C[Block(i,j)][a[i], a[j]]
    end
    return s
end


function max_CE_violation_2p(z, C, m)
    z_f = reshape(z, m, m)
    maxv = -Inf
    # player 1 constraints: for each recommended a1 and deviation a1'
    for a1 in 1:m, a1p in 1:m
        a1 == a1p && continue
        s = 0.0
        for a2 in 1:m
            # Δ = J(rec) - J(dev)
            Jrec = J_def(1, [a1, a2], C)
            Jdev = J_def(1, [a1p, a2], C)
            s += z_f[a1, a2] * (Jrec - Jdev)
        end
        maxv = max(maxv, s)
    end
    # player 2 constraints
    for a2 in 1:m, a2p in 1:m
        a2 == a2p && continue
        s = 0.0
        for a1 in 1:m
            Jrec = J_def(2, [a1, a2], C)
            Jdev = J_def(2, [a1, a2p], C)
            s += z_f[a1, a2] * (Jrec - Jdev)
        end
        maxv = max(maxv, s)
    end
    return maxv
end

function NumCorrPrimalVars(m, n)
    l = m^n
    return l + n + 1
end

function NumCorrEqConstraints()
    return 1
end

function NumCorrIneqConstraints(m, n)
    l = m^n
    return l + n*m*(m-1) + 3*n
end

function SplitCorrSolution(variables, m, n)
    l_pr = NumCorrPrimalVars(m, n)
    l_eq = NumCorrEqConstraints()
    l_ineq = NumCorrIneqConstraints(m, n)

    x_sol = variables[1:l_pr]
    lambda_eq = variables[l_pr + 1 : l_pr + l_eq]
    mu_ineq = variables[l_pr + l_eq + 1 : l_pr + l_eq + l_ineq]

    return x_sol, lambda_eq, mu_ineq
end

function BuildCorrConstraintMap(m, n)
    l = m^n
    meta = Vector{NamedTuple}()

    # 1) x >= 0 constraints from CalcH
    for joint_idx in 1:l
        push!(meta, (
            idx = length(meta) + 1,
            block = :prob_nonneg,
            joint_idx = joint_idx
        ))
    end

    # 2) CE constraints from CalcH
    for i in 1:n
        for ai in 1:m
            for aibar in 1:m
                if ai == aibar
                    continue
                end
                push!(meta, (
                    idx = length(meta) + 1,
                    block = :ce,
                    player = i,
                    rec = ai,
                    dev = aibar
                ))
            end
        end
    end

    # 3) T1 constraints
    for i in 1:n
        push!(meta, (
            idx = length(meta) + 1,
            block = :t1,
            player = i
        ))
    end

    # 4) T2 constraints
    for i in 1:n
        push!(meta, (
            idx = length(meta) + 1,
            block = :t2,
            player = i
        ))
    end

    # 5) T3 constraints
    for i in 1:n
        push!(meta, (
            idx = length(meta) + 1,
            block = :t3,
            player = i
        ))
    end

    return meta
end

function GetCEConstraintIndices(m, n)
    cmap = BuildCorrConstraintMap(m, n)
    return [c.idx for c in cmap if c.block == :ce]
end

function GetActiveConstraints(ineq_vals, mu_ineq; tol_val = 1e-7, tol_mu = 1e-7)
    active = BitVector(undef, length(ineq_vals))
    for k in eachindex(ineq_vals)
        active[k] = (abs(ineq_vals[k]) <= tol_val) && (mu_ineq[k] >= tol_mu)
    end
    return active
end

function GetActiveCEConstraints(res; tol_val = 1e-7, tol_mu = 1e-7)
    cmap = BuildCorrConstraintMap(res.m, res.n)
    active = GetActiveConstraints(res.ineq_vals, res.mu_ineq; tol_val = tol_val, tol_mu = tol_mu)

    ce_list = NamedTuple[]
    for k in eachindex(cmap)
        if cmap[k].block == :ce && active[k]
            push!(ce_list, (
                idx = k,
                player = cmap[k].player,
                rec = cmap[k].rec,
                dev = cmap[k].dev,
                mu = res.mu_ineq[k],
                val = res.ineq_vals[k]
            ))
        end
    end
    return ce_list
end

function CEKeyList(active_ce_list)
    return Set((c.player, c.rec, c.dev) for c in active_ce_list)
end

function TopKActiveCEByMu(res, k; tol_val = 1e-7, tol_mu = 1e-7)
    ce_list = GetActiveCEConstraints(res; tol_val = tol_val, tol_mu = tol_mu)
    sort!(ce_list, by = x -> -x.mu)
    return ce_list[1:min(k, length(ce_list))]
end

function GetCEConstraintInfo(res; sigma = 1.0)
    x = res.primals[1:res.l]
    x_f = reshape(x, ntuple(_ -> res.m, res.n))
    cmap = BuildCorrConstraintMap(res.m, res.n)

    ce_list = NamedTuple[]

    for k in eachindex(cmap)
        if cmap[k].block == :ce

            i   = cmap[k].player
            ai  = cmap[k].rec
            dev = cmap[k].dev

            p_ai = CalcMarginalP(i, ai, x_f, res.m, res.n)

            σi = isa(sigma, Number) ? sigma : sigma[i]

            μ = res.mu_ineq[k]

            push!(ce_list, (
                idx = k,
                player = i,
                rec = ai,
                dev = dev,
                mu = μ,
                val = res.ineq_vals[k],
                p_ai = p_ai,
                sigma = abs(σi),
                mu_sigma = abs(μ * σi),
                mup_sigma = abs(μ * p_ai * σi)
            ))
        end
    end

    return ce_list
end

function TopKCEByMuPSigma(res, k; sigma=1.0)
    ce_list = GetCEConstraintInfo(res; sigma=sigma)
    sort!(ce_list, by = x -> -x.mup_sigma)
    return ce_list[1:min(k, length(ce_list))]
end

function TopKCEByMuSigma(res, k; sigma=1.0)
    ce_list = GetCEConstraintInfo(res; sigma=sigma)
    sort!(ce_list, by = x -> -x.mu_sigma)
    return ce_list[1:min(k, length(ce_list))]
end

function TopKCEByMu(res, k)
    ce_list = GetCEConstraintInfo(res)
    sort!(ce_list, by = x -> -x.mu)
    return ce_list[1:min(k, length(ce_list))]
end

function TopKCEBySigma(res, k; sigma=1.0)
    ce_list = GetCEConstraintInfo(res; sigma=sigma)
    sort!(ce_list, by = x -> -x.sigma)
    return ce_list[1:min(k, length(ce_list))]
end

function RandomKCE(res, k; sigma=1.0, rng = Random.GLOBAL_RNG)
    ce_list = GetCEConstraintInfo(res; sigma=sigma)

    idx = randperm(rng, length(ce_list))[1:min(k,length(ce_list))]

    return ce_list[idx]
end

function PrintCEList(ce_list)
    # for c in ce_list
    #     println(c)
    # end
    println([c.idx for c in ce_list])
end