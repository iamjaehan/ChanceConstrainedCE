using correlated
using BlockArrays: Block

function PrepNash(r,n,λ)
    C = SetC(r,n,λ)

    n = blocksize(C)[1] # Number of vehicles
    m = size(C[Block(1)])[1] # Number of actions

    # Construct variables
    fs = [(x, θ) -> x[Block(ii)]'*C[m*(ii-1)+1:m*ii,:]*x for ii in 1:n]
    gs = [(x, θ) -> [sum(x[Block(ii)]) - 1] for ii in 1:n]
    hs = [(x, θ) -> x[Block(ii)] for ii in 1:n]
    g̃ = (x, θ) -> [0]
    h̃ = (x, θ) -> [0]
    problem = ParametricGame(
        objectives = fs,
        equality_constraints = gs,
        inequality_constraints = hs,
        shared_equality_constraint = g̃,
        shared_inequality_constraint = h̃,
        parameter_dimension = 1,
        primal_dimensions = fill(m, n),
        equality_dimensions = fill(1, n),
        inequality_dimensions = fill(m, n),
        shared_equality_dimension = 1,
        shared_inequality_dimension = 1,
    )

    return (; problem, C, m, n)
end

function SolveNash(problem,C,m,n,isRndGuess)
    # Randomize seed?
    if isRndGuess
        # initial_guess = rand(total_dim(problem))
        # PNE guess
        # initial_guess = zeros(total_dim(problem))
        # initial_guess[n*m+n+1:end-2] .= rand(1)*50
        # rndIdx = Vector{Int64}(undef,n)
        # for i in 1:n
        #     rndIdx[i] = argmax(rand(m))
        # end
        # for i in 1:n
        #     initial_guess[rndIdx[i]+(i-1)*m] = 1
        #     initial_guess[rndIdx[i]+(i-1)*m + n*m+n] = 0
        # end
        # PNE guess - Smart duals
        initial_guess = zeros(total_dim(problem))
        rndIdx = Vector{Int64}(undef,n)
        equalityMultiplier = zeros(n)
        for i in 1:n
            rndIdx[i] = argmax(rand(m))
        end
        for i in 1:n
            initial_guess[rndIdx[i]+(i-1)*m] = 1
            initial_guess[rndIdx[i]+(i-1)*m + n*m+n] = 0
        end
        for i in 1:n
            primalSum = zeros(m)
            for j in 1:n
                primalSum .+= C[Block(i,j)] * initial_guess[(j-1)*m+1:j*m]
            end
            equalityMultiplier[i] = primalSum[rndIdx[i]]
            initial_guess[n*m+n + (i-1)*m+1 : n*m+n + i*m] = primalSum - ones(m)*equalityMultiplier[i]
        end
        initial_guess[n*m+1:n*m+n] = equalityMultiplier
        # MNE guess
        # initial_guess = zeros(total_dim(problem))
        # for i = 1:n
        #     localGuess = rand(m)
        #     localGuess = localGuess/sum(localGuess)
        #     initial_guess[1+(i-1)*m:i*m] = localGuess
        # end
    else
        initial_guess = zeros(total_dim(problem))
    end

    # Solve!
    solverTime = @elapsed (; primals, variables, status, info) = solve(problem, [0], initial_guess = initial_guess)

    (; primals, score = CalcJNash(C,primals,m,n), varsize = size(primals)[1]*m, solverTime, status)
end

function SearchNash(r,n,λ,isRndGuess,Δ)
    println("Begin Nash Search for m=$(2^r) and n=$n case.")
    (; problem, C, m, n) = PrepNash(r,n,λ)
    (; primals, score, varsize, solverTime, status) = SolveNash(problem, C, m, n, isRndGuess)

    jointPrimal = SwitchIndtoJoint(primals,m,n)
    fairScore = EvalFairness(jointPrimal,C,m,n,Δ)
    giniScore = EvalGini(jointPrimal,C,m,n,Δ)
    avgDelayScore = EvalAverageDelay(jointPrimal,C,m,n)
    
    (; primals, fairScore, giniScore, avgDelayScore, varsize, solverTime)
end 