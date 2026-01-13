using correlated
using BlockArrays
using Combinatorics
using Statistics

function SetC(r,n,λ)
    # Parameter setting
    # r Number of runways
    # l Number of sequencing legs
    # d Number of departure terminals
    dF = 5 # Delay factor
    eF = dF*1000 # Hazard factor
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

# Construct cost matrix _ Sample case
# C12 = -[4 1; 5 0]
# C12 = [10 0; 5 1]
# C21 = C12'
# C = BlockArray([C12 C21],[2],[2,2])
# C11 = zeros(2,2)
# C22 = zeros(2,2)
# C12 = -[3 2; 5 0]
# C21 = C12
# C = BlockArray([C11 C12; C21 C22],[2,2],[2,2])

# Define functions
# function generateAseq(i,ai,m,n)
#     primeChoice = "1:$m"
#     stringUnit = ""
#     for j in 1:n
#         if j == i
#             stringUnit = stringUnit*"$ai"*","
#         else
#             stringUnit = stringUnit*primeChoice*","
#         end
#     end
#     stringUnit = chop(stringUnit)
#     stringSum = "vec(collect(Iterators.product("*stringUnit*")))"
#     aSet = eval(Meta.parse(stringSum))
#     return aSet
# end

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
    aSeq = generateAseq(i,ai,m,n)
    l = length(aSeq)
    p = 0
    for j in 1:l
        p += x_f[CartesianIndex(aSeq[j])]
    end
    return p
end

function CalcH(x,m,n,C; zalpha, sigma)
    x_f = reshape(x,ntuple(i->m,n))
    out = Vector{Any}(undef,m^n + m^2*n - n*m)
    out[1:length(x)] = x
    c = length(x)
    for i in 1:n
        for ai in 1:m
            p_ai = CalcMarginalP(i,ai,x_f,m,n)
            base = CalcPhi(i,ai,ai,x_f,m,n,C)
            for _ai in 1:m
                if ai == _ai
                    continue
                end
                c += 1
                mean_diff = base - CalcPhi(i,ai,_ai,x_f,m,n,C)
                margin = zalpha * sigma * p_ai
                out[c] = mean_diff + margin
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

function CorrPacker(x,C,m,n,l,Δ; zalpha, sigma)
    out = [CalcH(x[1:l], m, n, C; zalpha = zalpha, sigma = sigma);
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
