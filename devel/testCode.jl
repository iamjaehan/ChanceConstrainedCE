############################################################
# Parameter setting
############################################################

r = 2
n = 3
λ = ones(n)
Δ = 100.0

zalpha = 2

# scalar or vector both allowed
# sigma = 100
# sigma = [0.1,0.15,0.05]
# sigma = [60,59.9,30]
# sigma = [20, 100, 70]
sigma = [20, 10, 5]*5

k = 5

############################################################
# Baseline solve
############################################################

res0 = SearchCorr(r, n, λ, Δ;
    zalpha = zalpha,
    sigma = sigma)

println("================================================")
println("BASELINE")
println("score      = ", res0.score)
println("avgDelay   = ", res0.avgDelayScore)
println("status     = ", res0.status)
println("================================================")

############################################################
# Select constraints
############################################################

top_mu      = TopKCEByMu(res0, k)
top_mus     = TopKCEByMuSigma(res0, k; sigma = sigma)
# top_mups    = TopKCEByMuPSigma(res0, k; sigma = sigma)
top_mups    = TopKCEBySigma(res0, k; sigma = sigma)
top_random  = RandomKCE(res0, k)

println("=== TOP μ ===")
PrintCEList(top_mu)

println("=== TOP μσ ===")
PrintCEList(top_mus)

println("=== TOP σ ===")
PrintCEList(top_mups)

println("=== RANDOM ===")
PrintCEList(top_random)

############################################################
# Convert to sigma-removal key set
############################################################

zero_mu    = CEKeyList(top_mu)
zero_mus   = CEKeyList(top_mus)
zero_mups  = CEKeyList(top_mups)
zero_rand  = CEKeyList(top_random)

############################################################
# Re-solve with modified sigma
############################################################

res_mu = SearchCorr(r, n, λ, Δ;
    zalpha = zalpha,
    sigma = sigma,
    zero_sigma_ce_keys = zero_mu)

res_mus = SearchCorr(r, n, λ, Δ;
    zalpha = zalpha,
    sigma = sigma,
    zero_sigma_ce_keys = zero_mus)

res_mups = SearchCorr(r, n, λ, Δ;
    zalpha = zalpha,
    sigma = sigma,
    zero_sigma_ce_keys = zero_mups)

res_rand = SearchCorr(r, n, λ, Δ;
    zalpha = zalpha,
    sigma = sigma,
    zero_sigma_ce_keys = zero_rand)

############################################################
# Result summary
############################################################

println("================================================")
println("RESULT SUMMARY")
println("baseline avgDelay      = ", res0.avgDelayScore)
println("μ avgDelay             = ", res_mu.avgDelayScore)
println("μσ avgDelay            = ", res_mus.avgDelayScore)
println("σ avgDelay           = ", res_mups.avgDelayScore)
println("random avgDelay        = ", res_rand.avgDelayScore)
println("================================================")

println("STATUS CHECK")
println("baseline = ", res0.status)
println("μ        = ", res_mu.status)
println("μσ       = ", res_mus.status)
println("σ      = ", res_mups.status)
println("random   = ", res_rand.status)

println("TESTING DONE")