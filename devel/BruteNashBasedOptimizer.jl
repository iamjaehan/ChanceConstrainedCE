using correlated
using BlockArrays: Block
using PATHSolver: PATHSolver

function SolveBruteSubProblem(nashSet,scoreSet, C, m, n, Δ)
    d = size(scoreSet,1)
    # define problem
    f(x, θ) = CalcEFJ(x, d, n, Δ)
    g(x, θ) = [sum(x[1:d])-1]
    h(x, θ) = NashPacker(x,scoreSet,C,m,n,d,Δ)
    problem = ParametricOptimizationProblem(;
        objective = f,
        equality_constraint = g,
        inequality_constraint = h,
        parameter_dimension = 1,
        primal_dimension = d + n + 1,
        equality_dimension = 1,
        inequality_dimension = d + 3*n,
    )

    solverTime = @elapsed (; primals, variables, status, info) = solve(problem, [0])

    # score = CalcJNashSet(primals[1:d], nashSet, C, m, n)
    primalJoint = nashSet*primals[1:d]
    score = CalcJ(primalJoint,C,m,n)
    varsize = length(primals)

    (; primalJoint, primals, score, varsize, solverTime)
end

function BruteNashBasedOptimizer(r,n,λ,Δ)
    (; C, m, n) = PrepNashBrute(r,n,λ)
    accuSolveTime = @elapsed nashSet = SolveNashBrute(C, m, n)
    # (; problem, C, m, n) = PrepNash(r,n,λ)
    # out = SolveNash(problem, C, m, n, false)
    # if out.status == PATHSolver.MCP_Solved
    #     solCandidate = SwitchIndtoJoint(out.primals,m,n)
    #     nashSet = vcat(nashSet,[solCandidate])
    # end
    
    d = length(nashSet)
    # if d>250
    #     v = rand(d)
    #     v_r = rand(d).<200/d
    #     nashSet = nashSet[v_r]
    #     print("Too many NEs: ",d," NEs -- ")
    # end
    # d = length(nashSet)
    print(d," Nash equilibriums explored. Solving Subproblem")
    scoreSet = Matrix{Any}(undef,d,n)
    global nashSet = nashSet
    for i in 1:d
        for j in 1:n
            scoreSet[i,j] = CalcIndividualJ(nashSet[i],j,C,m,n)
        end
    end
    nashSet = reshape(stack(nashSet'),m^n,d)
    
    (; primalJoint, primals, score, varsize, solverTime) = SolveBruteSubProblem(nashSet,scoreSet, C, m, n, Δ)
    println("--done--")
    
    fairScore = EvalFairness(primalJoint, C, m, n, Δ)
    giniScore = EvalGini(primalJoint, C, m, n, Δ)
    avgDelayScore = EvalAverageDelay(primalJoint, C, m, n)

    solverTime += accuSolveTime

    (; primalJoint, primals, fairScore, giniScore, avgDelayScore, varsize, solverTime, nashSet)
end