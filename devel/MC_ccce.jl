using Random, Statistics, MAT

############################################################
# Sigma sampling
############################################################

function sample_sigma(rng, n; sigma_max = 100.0, eps_sigma = 1e-3)
    sigma = rand(rng, n) .* sigma_max
    sigma = max.(sigma, eps_sigma)
    return sigma
end

############################################################
# Deviation check under sampled eta
############################################################

function CEViolationWithNoise(x, C, m, n, eta)
    x_f = reshape(x, ntuple(_ -> m, n))
    max_margin = -Inf
    any_violation = false

    for i in 1:n
        for ai in 1:m
            p_ai = CalcMarginalP(i, ai, x_f, m, n)
            base = CalcPhi(i, ai, ai, x_f, m, n, C)

            for aibar in 1:m
                ai == aibar && continue

                mean_diff = base - CalcPhi(i, ai, aibar, x_f, m, n, C)
                realized_margin = mean_diff + eta[i] * p_ai

                max_margin = max(max_margin, realized_margin)

                if realized_margin > 1e-8
                    any_violation = true
                end
            end
        end
    end

    return any_violation, max_margin
end

############################################################
# Brute-force NE wrapper
############################################################

function SearchNashBruteWrapper(r, n, λ, Δ, sigma; seed = 1, pick_mode = :random, mult = mult)
    res_ne = SearchNashBrute(r, n, λ, Δ, sigma; seed = seed, pick_mode = pick_mode, mult = mult)

    if !res_ne.success
        return (;
            success = false,
            score = NaN,
            avgDelayScore = NaN,
            status = "FAIL"
        )
    end

    return (;
        success = true,
        score = res_ne.score,
        avgDelayScore = res_ne.avgDelayScore,
        status = "OK"
    )
end

############################################################
# Main Monte Carlo experiment
############################################################

function run_mc_ccce_experiment(; 
    r = 2,
    n = 3,
    λ = ones(3),
    Δ = 100.0,
    zalpha = 2.0,
    sigma_max = 100.0,
    k = 5,
    mc_runs = 100,
    noise_runs = 100,
    seed = 42,
    include_ne = true,
    mult = mult
)

    rng = MersenneTwister(seed)
    rows = NamedTuple[]

    for t in 1:mc_runs

        ################################################
        # sample sigma for this outer MC trial
        ################################################
        sigma = sample_sigma(rng, n; sigma_max = sigma_max)

        println("================================================")
        println("MC iteration = ", t)
        println("sigma = ", sigma)

        ################################################
        # baseline CC-CE solve
        ################################################
        res0 = SearchCorr(r, n, λ, Δ;
            zalpha = zalpha,
            sigma = sigma,
            mult = mult)

        ################################################
        # selection rules
        ################################################
        top_mu   = TopKCEByMu(res0, k)
        top_mus  = TopKCEByMuSigma(res0, k; sigma = sigma)
        top_rand = RandomKCE(res0, k; sigma = sigma, rng = rng)

        ################################################
        # re-solve after removing sigma on selected 5 constraints
        ################################################
        res_mu = SearchCorr(r, n, λ, Δ;
            zalpha = zalpha,
            sigma = sigma,
            mult = mult,
            zero_sigma_ce_keys = CEKeyList(top_mu))

        res_mus = SearchCorr(r, n, λ, Δ;
            zalpha = zalpha,
            sigma = sigma,
            mult = mult,
            zero_sigma_ce_keys = CEKeyList(top_mus))

        res_rand = SearchCorr(r, n, λ, Δ;
            zalpha = zalpha,
            sigma = sigma,
            mult = mult,
            zero_sigma_ce_keys = CEKeyList(top_rand))

        ################################################
        # collect CC-CE family
        ################################################
        solns = Dict(
            "baseline" => res0,
            "mu"       => res_mu,
            "mu_sigma" => res_mus,
            "random"   => res_rand,
        )

        ################################################
        # deviation Monte Carlo for each CC-CE method
        ################################################
        for (name, res) in solns
            dev_count = 0
            max_margins = Float64[]

            for _ in 1:noise_runs
                eta = randn(rng, n) .* sigma

                violated, max_margin = CEViolationWithNoise(
                    res.primals[1:res.l],
                    res.C,
                    res.m,
                    res.n,
                    eta
                )

                dev_count += violated ? 1 : 0
                push!(max_margins, max_margin)
            end

            push!(rows, (
                mc_iter = t,
                method = name,
                sigma = copy(sigma),
                score = res.score,
                avgDelay = res.avgDelayScore,
                dev_rate = dev_count / noise_runs,
                mean_max_margin = mean(max_margins),
                status = string(res.status)
            ))
        end

        ################################################
        # NE benchmark
        ################################################
        if include_ne
            res_ne = SearchNashBruteWrapper(r, n, λ, Δ, sigma; seed = t, pick_mode = :random)

            push!(rows, (
                mc_iter = t,
                method = "ne",
                sigma = copy(sigma),
                score = res_ne.score,
                avgDelay = res_ne.avgDelayScore,
                dev_rate = NaN,
                mean_max_margin = NaN,
                status = res_ne.status
            ))
        end
    end

    return rows
end

############################################################
# Save rows to MAT
############################################################

function save_mc_rows_mat(rows, filename = "mc_results.mat")
    mc_iter = Int[]
    method = String[]
    score = Float64[]
    avgDelay = Float64[]
    dev_rate = Float64[]
    mean_max_margin = Float64[]
    status = String[]

    sigma_list = Vector{Vector{Float64}}()

    for r in rows
        push!(mc_iter, r.mc_iter)
        push!(method, r.method)
        push!(score, r.score)
        push!(avgDelay, r.avgDelay)
        push!(dev_rate, r.dev_rate)
        push!(mean_max_margin, r.mean_max_margin)
        push!(status, string(r.status))
        push!(sigma_list, collect(r.sigma))
    end

    # also save sigma as matrix: N x n
    n = length(sigma_list[1])
    sigma_mat = zeros(length(sigma_list), n)
    for i in 1:length(sigma_list)
        sigma_mat[i, :] .= sigma_list[i]
    end

    file = matopen(filename, "w")
    write(file, "mc_iter", mc_iter)
    write(file, "method", method)
    write(file, "score", score)
    write(file, "avgDelay", avgDelay)
    write(file, "dev_rate", dev_rate)
    write(file, "mean_max_margin", mean_max_margin)
    write(file, "status", status)
    write(file, "sigma", sigma_list)       # cell-like
    write(file, "sigma_mat", sigma_mat)    # numeric matrix
    close(file)

    println("Saved Monte Carlo results to ", filename)
end