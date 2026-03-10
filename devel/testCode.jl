r = 2
n = 3
λ = ones(n)
Δ = 100.0
# zalpha = 1.96
zalpha = 2.5
sigma = 0.1

res0 = SearchCorr(r, n, λ, Δ; zalpha = zalpha, sigma = sigma)

println("baseline score = ", res0.score)
println("baseline avgDelay = ", res0.avgDelayScore)
println("status = ", res0.status)

top5_mup = TopKCEByMuPAI(res0, 5)
top5_mu  = TopKCEByMu(res0, 5)

println("=== top 5 by mu * p_ai ===")
PrintCEList(top5_mup)

println("=== top 5 by mu only ===")
PrintCEList(top5_mu)

zero_keys_mup = CEKeyList(top5_mup)
zero_keys_mu  = CEKeyList(top5_mu)

res_mup = SearchCorr(r, n, λ, Δ;
    zalpha = zalpha,
    sigma = sigma,
    zero_sigma_ce_keys = zero_keys_mup)

res_mu = SearchCorr(r, n, λ, Δ;
    zalpha = zalpha,
    sigma = sigma,
    zero_sigma_ce_keys = zero_keys_mu)

println("baseline score = ", res0.score)
println("mu*p_ai score  = ", res_mup.score)
println("mu-only score  = ", res_mu.score)

println("baseline avgDelay = ", res0.avgDelayScore)
println("mu*p_ai avgDelay  = ", res_mup.avgDelayScore)
println("mu-only avgDelay  = ", res_mu.avgDelayScore)
println("TESTING DONE")