using correlated
using BlockArrays: Block
using PATHSolver: PATHSolver

function SolveSubProblem(nashSet,scoreSet, C, m, n, Δ)
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
    primalJoint = nashSet'*primals[1:d]
    score = CalcJ(primalJoint,C,m,n)
    varsize = length(primals)

    (; primalJoint, primals, score, varsize, solverTime)
end

function NashBasedOptimizer(r,n,λ,Δ,iters)
    nashSet = Vector{Any}(undef,0)
    pseudoNashSet = Vector{Any}(undef,0)
    (; problem, C, m, n) = PrepNash(r,n,λ)
    accuSolverTime = 0
    for i = 1:iters
        if i <= 1
            isRndGuess = false
        else
            isRndGuess = true
        end
        out = SolveNash(problem, C, m, n, isRndGuess)
        accuSolverTime += out.solverTime
        if out.status == PATHSolver.MCP_Solved
            solCandidate = SwitchIndtoJoint(out.primals,m,n)
            RoundSolCandidate = map(x->abs.(round.(x,digits=2)),solCandidate)
            if RoundSolCandidate ∉ pseudoNashSet
                pseudoNashSet = vcat(pseudoNashSet, [RoundSolCandidate])
                nashSet = vcat(nashSet, [solCandidate])
            end
        end
    end
    d = length(nashSet)
    print(d," Nash equilibriums explored. Solving Subproblem")
    scoreSet = Matrix{Any}(undef,d,n)
    global nashSet = nashSet
    for i in 1:d
        for j in 1:n
            scoreSet[i,j] = CalcIndividualJ(nashSet[i],j,C,m,n)
        end
    end
    nashSet = stack(nashSet)'
    
    (; primalJoint, primals, score, varsize, solverTime) = SolveSubProblem(nashSet,scoreSet, C, m, n, Δ)
    println("--done--")
    
    fairScore = EvalFairness(primalJoint, C, m, n, Δ)
    giniScore = EvalGini(primalJoint, C, m, n, Δ)
    avgDelayScore = EvalAverageDelay(primalJoint, C, m, n)
    solverTime += accuSolverTime

    (; primalJoint, primals, fairScore, giniScore, avgDelayScore, varsize, solverTime, nashSet)
end