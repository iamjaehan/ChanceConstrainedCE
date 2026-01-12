using correlated
using BlockArrays: Block

function SearchCorr(r,n,λ,Δ)
    println("Begin Corr Search for m=$(2^r) and n=$n case.")
    C = SetC(r,n,λ)

    n = blocksize(C)[1] # Number of vehicles
    m = size(C[Block(1)])[1] # Number of actions

    # f(x, θ) = CalcJ(x,C,m,n) # Utilitarian cost fcn
    f(x, θ) = CalcEFJ(x,m^n,n,Δ) # EF cost fcn
    g(x, θ) = [sum(x[1:m^n]) - 1]
    h(x, θ) = CorrPacker(x,C,m,n,m^n,Δ)
    problem = ParametricOptimizationProblem(;
        objective = f,
        equality_constraint = g,
        inequality_constraint = h,
        parameter_dimension = 1,
        primal_dimension = m^n + n + 1,
        equality_dimension = 1,
        inequality_dimension = m^n + n*m*m + 3*n,
    )

    solverTime = @elapsed (; primals, variables, status, info) = solve(problem, [0])
    score = CalcEFJ(primals,m^n,n,Δ)
    # score = CalcJ(primals,C,m,n)

    fairScore = EvalFairness(primals[1:m^n], C, m, n, Δ)
    giniScore = EvalGini(primals[1:m^n], C, m, n, Δ)
    avgDelayScore = EvalAverageDelay(primals[1:m^n], C, m, n)
    
    (; primals, fairScore, giniScore, avgDelayScore, varsize = size(primals)[1], solverTime)
end