using correlated
using BlockArrays: Block

function SearchCorr(r, n, λ, Δ;
                    zalpha,
                    sigma,
                    zero_sigma_ce_keys = Set{Tuple{Int,Int,Int}}(),
                    mult = 2.0, 
                    verbose = false)

    println("Begin Corr Search for m=$(2^r) and n=$n case.")
    C = SetC(r, n, λ; mult=mult)

    n = blocksize(C)[1]              # Number of players
    m = size(C[Block(1)])[1]         # Number of actions
    l = m^n                          # joint action dimension

    # Expected total system cost
    # f(x, θ) = CalcJ(x[1:l], C, m, n)
    f(x, θ) = CalcWeightedJ(x[1:l], C, m, n, 1 ./ sigma)

    # Probability simplex
    g(x, θ) = [sum(x[1:l]) - 1]

    # Keep current CC-CE implementation with p_ai and EF auxiliary constraints
    h(x, θ) = CorrPacker(x, C, m, n, l, Δ;
                         zalpha = zalpha,
                         sigma = sigma,
                         zero_sigma_ce_keys = zero_sigma_ce_keys)

    problem = ParametricOptimizationProblem(;
        objective = f,
        equality_constraint = g,
        inequality_constraint = h,
        parameter_dimension = 1,
        primal_dimension = l + n + 1,
        equality_dimension = 1,
        inequality_dimension = l + n*m*(m-1) + 3*n,
    )

    solverTime = @elapsed (; primals, variables, status, info) =
        solve(problem, [0]; verbose = verbose)

    # Report expected system cost as the main score
    # score = CalcJ(primals[1:l], C, m, n)
    score = CalcWeightedJ(primals[1:l], C, m, n, 1 ./ sigma)
    avgDelayScore = score / n

    fairScore = EvalFairness(primals[1:l], C, m, n, Δ)
    giniScore = EvalGini(primals[1:l], C, m, n, Δ)

    eq_vals = g(primals, [0])
    ineq_vals = h(primals, [0])

    x_sol, lambda_eq, mu_ineq = SplitCorrSolution(variables, m, n)

    (; primals,
       variables,
       x_sol,
       lambda_eq,
       mu_ineq,
       eq_vals,
       ineq_vals,
       score,
       avgDelayScore,
       fairScore,
       giniScore,
       varsize = length(primals),
       solverTime,
       status,
       info,
       m,
       n,
       l,
       C)
end