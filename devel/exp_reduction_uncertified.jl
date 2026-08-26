# =============================================================================
# exp_reduction_uncertified.jl
#
# The uncertified counterpart to exp_reduction.jl.
#
# Cost model: SetC (the RRCE paper's pairwise-summed runway cost, devel/0_GameSetup.jl),
# NOT the vertiport Eq. 22 cost. The difference matters exactly here: SetC's yield
# cost scales with the number of opponents, so the uniform CC-PNE certificate of the
# reduced-joint-action note does NOT hold -- none of the exclusive-occupancy set F is
# a CC-PNE at these parameters (0% certified at every n), while CC-PNE themselves do
# exist, sitting outside F.
#
# That is the regime this file measures:
#
#   Full CC-CE  SearchCorrLP               mixes over all of X, |X| = m^n
#   Reduced     SearchCorrReducedLP        mixes over F only -- expected to go
#                                          INFEASIBLE, since F carries no equilibria
#   Hull / X    BruteNashBasedOptimizerLP  brute-force CC-PNE search over all of X,
#                                          then the hull LP -- still valid
#
# Hull / F has no entry here by construction: scoping the equilibrium search to F is
# what the certificate licenses, and without it there is nothing in F to find. Since
# F cannot be certified, the search has to go over the full joint action space.
#
# Writes the same CSV columns as exp_reduction.jl so both cases share the MAT
# converter and the MATLAB plotting script.
#
# Usage (from the repo root):
#   julia --project=. devel/exp_reduction_uncertified.jl
# =============================================================================

using correlated
using BlockArrays: Block
using Printf
using DataFrames
using CSV

# -----------------------------------------------------------------------------
# Parameters -- n sweep and (z_α, σ) matched to the certified case so the two
# tables are read side by side.
# -----------------------------------------------------------------------------
const R          = 3        # resources -> m = 2^r = 8 actions per player
const ZALPHA     = 1.5
const SIGMA      = 3.0
const DELTA      = 100.0
const MULT       = 2.0      # SetC's yield-cost multiplier
const NS         = 3:8
const FULL_MAX_N = 6        # full CC-CE only up to here (|X| = m^n variables)
const HULL_MAX_N = 8        # brute CC-PNE search cap
const WARMUP_N   = 3
const OUTFILE    = joinpath(@__DIR__, "..", "exp_reduction_uncertified_results.csv")
const CE_OBJECTIVE = :aggregate

# -----------------------------------------------------------------------------
# Runners. Each returns a row; a solver that reports infeasibility is recorded
# with a NaN score rather than dropped -- "the reduction is not available here"
# is the measurement, not a missing data point.
# -----------------------------------------------------------------------------
feasible(status) = !occursin("INFEASIBLE", uppercase(string(status)))

score_or_nan(res) = feasible(res.status) ? res.score : NaN
row_of(res, method, n, search, d) = (;
    method, n, search,
    build = res.buildTime, solve = res.solverTime, total = NaN,
    score = score_or_nan(res),
    avgDelay = feasible(res.status) ? res.avgDelayScore : NaN,
    fair     = feasible(res.status) ? res.fairScore : NaN,
    gini     = feasible(res.status) ? res.giniScore : NaN,
    varsize = res.varsize, d, status = string(res.status))

function run_full(n)
    λ = ones(n)
    t = @elapsed res = SearchCorrLP(R, n, λ, DELTA;
            zalpha = ZALPHA, sigma = SIGMA, mult = MULT, objective = CE_OBJECTIVE)
    merge(row_of(res, "Full", n, 0.0, res.varsize), (; total = t))
end

function run_reduced(n)
    λ = ones(n)
    t = @elapsed res = SearchCorrReducedLP(R, n, λ, DELTA;
            zalpha = ZALPHA, sigma = SIGMA, mult = MULT, objective = CE_OBJECTIVE)
    merge(row_of(res, "Reduced", n, 0.0, res.d), (; total = t))
end

function run_hull_x(n)
    λ = ones(n)
    t = @elapsed res = BruteNashBasedOptimizerLP(R, n, λ, DELTA;
            zalpha = ZALPHA, sigma = SIGMA, mult = MULT)
    merge(row_of(res, "Hull / X", n, res.searchTime, res.d), (; total = t))
end

const METHODS = [
    ("Full",     run_full,    FULL_MAX_N),
    ("Reduced",  run_reduced, last(NS)),
    ("Hull / X", run_hull_x,  HULL_MAX_N),
]

# -----------------------------------------------------------------------------
# Certification report: how much of F is CC-PNE, and how many CC-PNE lie outside
# F. This is the evidence that the case really is uncertified.
# -----------------------------------------------------------------------------
function certification_report(ns)
    println("CERTIFICATION OF F  (SetC cost, mult = $MULT)")
    println("-"^72)
    @printf("%-4s %8s %10s %10s %8s %10s\n", "n", "|F|", "|CC-PNE|", "in F", "out F", "F cert")
    rows = NamedTuple[]
    for n in ns
        n > HULL_MAX_N && continue
        λ = ones(n)
        C = SetC(R, n, λ; mult = MULT)
        m = size(C[Block(1)])[1]
        Sset, _ = ExclusiveJointSet(R, n)
        Fset = Set(Sset)
        (_, nashIdxList) = SolveNashBrute(C, m, n; zalpha = ZALPHA, sigma = SIGMA)
        pne = [Tuple(x) for x in nashIdxList]
        inF = count(p -> p in Fset, pne)
        chk = CheckExclusiveSetCCPNE(Sset, C, m, n; zalpha = ZALPHA, sigma = SIGMA)
        @printf("%-4d %8d %10d %10d %8d %9.1f%%\n",
                n, length(Sset), length(pne), inF, length(pne) - inF, chk.frac_pne * 100)
        push!(rows, (; n, F = length(Sset), pne = length(pne), inF,
                       outF = length(pne) - inF, frac_cert = chk.frac_pne))
    end
    println("-"^72)
    return DataFrame(rows)
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
function run_benchmark(; ns = NS, warmup = true, outfile = OUTFILE)
    m = 2^R
    println("="^78)
    println("Joint-action reduction benchmark -- UNCERTIFIED case (SetC pairwise cost)")
    println("  r = $R  (m = $m),  z_α = $ZALPHA,  σ = $SIGMA,  Δ = $DELTA,  mult = $MULT")
    println("  F is NOT certified here, so Hull / F is not run at all --")
    println("  the equilibrium search has to cover the full joint action space.")
    println("  n sweep: $(collect(ns)),  full CC-CE capped at n <= $FULL_MAX_N")
    println("  solver: JuMP + Ipopt")
    println("="^78)

    if warmup
        print("Warm-up at n = $WARMUP_N (discarded) ... ")
        for (_, f, _) in METHODS
            try; f(WARMUP_N); catch; end
        end
        println("done.\n")
    end

    rows = NamedTuple[]
    for n in ns
        for (name, f, nmax) in METHODS
            if n > nmax
                @printf("  [skip] %-8s n=%d  (out of reach)\n", name, n)
                continue
            end
            row = try
                f(n)
            catch e
                msg = sprint(showerror, e)
                @printf("  %-8s n=%d  ERROR: %s\n", name, n, first(msg, 70))
                (; method = name, n, search = NaN, build = NaN, solve = NaN,
                   total = NaN, score = NaN, avgDelay = NaN, fair = NaN, gini = NaN,
                   varsize = 0, d = 0, status = "ERROR")
            end
            push!(rows, row)
            sc = isnan(row.score) ? "        n/a" : @sprintf("%11.4f", row.score)
            @printf("  %-8s n=%d  score=%s  search=%8.4f  build=%8.4f  solve=%8.4f  total=%9.4f  [%s]\n",
                    name, n, sc, row.search, row.build, row.solve, row.total, row.status)
        end
        println()
    end

    df = DataFrame(rows)
    df.X = [big(m)^n for n in df.n]
    df.F = [big(n)^R for n in df.n]
    CSV.write(outfile, df)
    println("Wrote $(nrow(df)) rows to $outfile")
    return df
end

function print_summary(df)
    m = 2^R
    println()
    println("="^92)
    println("UNCERTIFIED CASE -- TOTAL SOLVE TIME BY METHOD")
    println("="^92)
    @printf("%-4s %12s %6s | %22s | %22s | %22s\n",
            "n", "|X|=m^n", "|F|", "Full CC-CE", "Reduced", "Hull / X")
    @printf("%-4s %12s %6s | %10s %11s | %10s %11s | %10s %11s\n",
            "", "", "", "score", "total(s)", "score", "total(s)", "score", "total(s)")
    println("-"^92)
    for n in sort(unique(df.n))
        @printf("%-4d %12s %6d ", n, string(big(m)^n), n^R)
        for name in ("Full", "Reduced", "Hull / X")
            sub = df[(df.n .== n) .& (df.method .== name), :]
            if nrow(sub) == 0
                @printf("| %10s %11s ", "n/a", "n/a")
            elseif isnan(sub.score[1])
                @printf("| %10s %11.3f ", "infeas", sub.total[1])
            else
                @printf("| %10.4f %11.3f ", sub.score[1], sub.total[1])
            end
        end
        println()
    end
    println("="^92)
    println("Hull / F: not applicable -- F carries no CC-PNE at these parameters.")

    println()
    println("TIME BREAKDOWN (s)")
    println("-"^68)
    @printf("%-9s %4s %10s %10s %10s %11s\n", "method", "n", "search", "build", "solve", "total")
    println("-"^68)
    for name in ("Full", "Reduced", "Hull / X")
        for r in eachrow(df[df.method .== name, :])
            @printf("%-9s %4d %10.4f %10.4f %10.4f %11.4f\n",
                    name, r.n, r.search, r.build, r.solve, r.total)
        end
    end
    println("-"^68)

    println()
    println("COST OF RESTRICTING TO THE CC-PNE HULL (Full vs Hull / X)")
    for n in sort(unique(df.n))
        fu = df[(df.n .== n) .& (df.method .== "Full"), :]
        hu = df[(df.n .== n) .& (df.method .== "Hull / X"), :]
        (nrow(fu) == 0 || nrow(hu) == 0) && continue
        (isnan(fu.score[1]) || isnan(hu.score[1])) && continue
        gap = (hu.score[1] - fu.score[1]) / fu.score[1] * 100
        @printf("  n=%d  full=%.2f  hull=%.2f  hull is %+.2f%% more costly\n",
                n, fu.score[1], hu.score[1], gap)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    cert = certification_report(NS)
    println()
    df = run_benchmark()
    print_summary(df)
end
