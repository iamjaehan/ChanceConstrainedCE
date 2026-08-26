# =============================================================================
# exp_reduction_partial.jl
#
# PARTIAL-CERTIFICATION counterpart to exp_reduction.jl.
#
# Same vertiport game and same n sweep, but the flat yield cost is replaced by an
# agent-and-resource weight w_iv (VertiportGameHet.jl). Each agent dislikes
# exactly one resource, cycling through the r resources. With unit·γ = 10 and
# q_α σ = 4.5 the per-profile CC-PNE test says
#
#     x ∈ F is a CC-PNE  ⟺  no agent holds its disliked resource,
#
# so roughly 30% of F certifies: |P_d| < |F| < |X|, which is the size ordering
# the reduction is supposed to exploit. Contrast with exp_reduction.jl, where the
# uniform certificate makes every element of F a CC-PNE (|P_d| = |F|).
#
# Objective is aggregate expected cost. Note the three methods are expected to
# report the SAME optimum here: on F the deviation margin does not depend on
# x_-i, so Z_red = Δ(P_d) exactly and the reduced LP coincides with the hull.
# The comparison being measured is therefore cost of computation, not quality.
#
# Usage (from the repo root):
#   julia --project=. devel/exp_reduction_partial.jl
# =============================================================================

using correlated
using Printf
using DataFrames
using CSV

const R          = 3        # resources -> m = 2^r = 8 actions per agent
const GAMMA      = 2.0
const ZALPHA     = 1.5
const SIGMA      = 3.0      # uniform across agents
const DELTA      = 100.0
const UNIT       = 5.0      # congestion scale; unit·γ = 10
const WLO        = 4.0      # disliked resource   (fails the holding test: 4.0 < q_α σ = 4.5)
const WMID       = 5.0      # every other resource (passes both: 4.5 <= 5.0 <= 5.5)
const NS         = 3:8
const FULL_MAX_N = 5        # full CC-CE capped here on time
const WARMUP_N   = 3
const OUTFILE    = joinpath(@__DIR__, "..", "exp_reduction_partial_results.csv")
const CE_OBJECTIVE = :aggregate

W_of(n) = HetYieldWeights(n, R; wlo = WLO, wmid = WMID)

# -----------------------------------------------------------------------------
# Runners
# -----------------------------------------------------------------------------
common(res, method, n, search, d) = (;
    method, n, search, build = res.buildTime, solve = res.solverTime, total = NaN,
    score = res.score, avgDelay = res.avgDelayScore, fair = res.fairScore,
    gini = res.giniScore, varsize = res.varsize, d, status = string(res.status))

function run_full(n)
    t = @elapsed res = SearchCorrLPHet(R, n, GAMMA, DELTA, W_of(n);
            zalpha = ZALPHA, sigma = SIGMA, unit = UNIT, objective = CE_OBJECTIVE)
    merge(common(res, "Full", n, 0.0, res.d), (; total = t))
end

function run_reduced(n)
    t = @elapsed res = SearchCorrReducedLPHet(R, n, GAMMA, DELTA, W_of(n);
            zalpha = ZALPHA, sigma = SIGMA, unit = UNIT, objective = CE_OBJECTIVE)
    merge(common(res, "Reduced", n, 0.0, res.d), (; total = t))
end

function run_hull_x(n)
    t = @elapsed res = HullLPHetOverX(R, n, GAMMA, DELTA, W_of(n);
            zalpha = ZALPHA, sigma = SIGMA, unit = UNIT, objective = CE_OBJECTIVE)
    merge(common(res, "Hull / X", n, res.searchTime, res.d), (; total = t))
end

function run_hull_f(n)
    t = @elapsed res = HullLPHetOverF(R, n, GAMMA, DELTA, W_of(n);
            zalpha = ZALPHA, sigma = SIGMA, unit = UNIT, objective = CE_OBJECTIVE)
    merge(common(res, "Hull / F", n, res.searchTime, res.d), (; total = t))
end

const METHODS = [
    ("Full",     run_full,     FULL_MAX_N),
    ("Reduced",  run_reduced,  last(NS)),
    ("Hull / X", run_hull_x,   last(NS)),
    ("Hull / F", run_hull_f,   last(NS)),
]

# -----------------------------------------------------------------------------
# Certification report -- the evidence that this really is partial certification
# -----------------------------------------------------------------------------
function certification_report(ns)
    actionSet = BuildRunwayActionSet(R)
    println("PARTIAL CERTIFICATION OF F   (w_lo = $WLO on each agent's disliked resource, w_mid = $WMID)")
    println("  holding needs w >= q_α σ = $(ZALPHA*SIGMA);  not holding needs w <= unit·γ − q_α σ = $(UNIT*GAMMA - ZALPHA*SIGMA)")
    println("-"^76)
    @printf("%-4s %10s %10s %12s %10s\n", "n", "|F|=n^r", "|P_d|", "certified %", "|X|=m^n")
    rows = NamedTuple[]
    for n in ns
        W = W_of(n)
        Sset, _ = ExclusiveJointSet(R, n)
        chk = CheckSetCCPNEHet(Sset, actionSet, R, GAMMA, n, W;
                               zalpha = ZALPHA, sigma = SIGMA, unit = UNIT)
        d = count(chk.flags)
        @printf("%-4d %10d %10d %11.1f%% %10s\n",
                n, length(Sset), d, chk.frac_pne * 100, string(big(2^R)^n))
        push!(rows, (; n, F = length(Sset), Pd = d, frac_cert = chk.frac_pne))
    end
    println("-"^76)
    return DataFrame(rows)
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
function run_benchmark(; ns = NS, warmup = true, outfile = OUTFILE,
                        methods = METHODS, merge_existing = false, cap = true)
    m = 2^R
    println("="^78)
    println("Joint-action reduction benchmark -- PARTIAL certification (heterogeneous yield)")
    println("  r = $R (m = $m),  γ = $GAMMA,  z_α = $ZALPHA,  σ = $SIGMA (uniform),  Δ = $DELTA,  unit = $UNIT")
    println("  yield weights: w_iv = $WLO if v = ((i-1) mod $R)+1 else $WMID")
    println("  n sweep: $(collect(ns)),  full CC-CE capped at n <= $FULL_MAX_N")
    println("  solver: JuMP + Ipopt,  objective: $CE_OBJECTIVE")
    println("="^78)

    if warmup
        print("Warm-up at n = $WARMUP_N (discarded) ... ")
        for (_, f, _) in methods
            f(WARMUP_N)
        end
        println("done.\n")
    end

    rows = NamedTuple[]
    for n in ns
        for (name, f, nmax) in methods
            if cap && n > nmax
                @printf("  [skip] %-8s n=%d  (|X| = %s variables, capped)\n", name, n, string(big(m)^n))
                continue
            end
            row = f(n)
            push!(rows, row)
            @printf("  %-8s n=%d  score=%10.4f  d=%-8d search=%8.4f  build=%8.4f  solve=%8.4f  total=%9.4f  [%s]\n",
                    name, n, row.score, row.d, row.search, row.build, row.solve, row.total, row.status)
        end
        println()
    end

    df = DataFrame(rows)
    df.X = [big(m)^n for n in df.n]
    df.F = [big(n)^R for n in df.n]

    # Re-measuring a subset: keep the rows this run did not produce.
    if merge_existing && isfile(outfile)
        old = CSV.read(outfile, DataFrame)
        produced = Set(zip(df.method, df.n))
        keep = [!((old.method[i], old.n[i]) in produced) for i in 1:nrow(old)]
        df = vcat(old[keep, :], df; cols = :union)
        sort!(df, [:n, :method])
        println("Merged with $(count(keep)) existing rows.")
    end

    CSV.write(outfile, df)
    println("Wrote $(nrow(df)) rows to $outfile")
    return df
end

function print_summary(df)
    m = 2^R
    println()
    println("="^108)
    println("PARTIAL CERTIFICATION -- TOTAL SOLVE TIME BY METHOD  (score = aggregate expected cost)")
    println("="^108)
    @printf("%-4s %12s %6s | %22s | %22s | %22s | %22s\n",
            "n", "|X|=m^n", "|F|", "Full CC-CE", "Reduced", "Hull / X", "Hull / F")
    @printf("%-4s %12s %6s | %10s %11s | %10s %11s | %10s %11s | %10s %11s\n",
            "", "", "", "score", "total(s)", "score", "total(s)", "score", "total(s)", "score", "total(s)")
    println("-"^108)
    for n in sort(unique(df.n))
        @printf("%-4d %12s %6d ", n, string(big(m)^n), n^R)
        for name in ("Full", "Reduced", "Hull / X", "Hull / F")
            sub = df[(df.n .== n) .& (df.method .== name), :]
            if nrow(sub) == 0
                @printf("| %10s %11s ", "n/a", "n/a")
            else
                @printf("| %10.4f %11.3f ", sub.score[1], sub.total[1])
            end
        end
        println()
    end
    println("="^108)

    println()
    println("PROBLEM SIZE ACTUALLY SOLVED (probability variables)")
    println("-"^60)
    @printf("%-4s %12s %10s %10s %10s\n", "n", "Full |X|", "Red |F|", "Hull |P_d|", "P_d/|F|")
    for n in sort(unique(df.n))
        hu = df[(df.n .== n) .& (df.method .== "Hull / F"), :]
        rd = df[(df.n .== n) .& (df.method .== "Reduced"), :]
        nrow(hu) == 0 && continue
        @printf("%-4d %12s %10d %10d %9.1f%%\n",
                n, string(big(m)^n), rd.d[1], hu.d[1], hu.d[1] / rd.d[1] * 100)
    end
    println("-"^60)

    println()
    println("TIME BREAKDOWN (s)")
    println("-"^68)
    @printf("%-9s %4s %10s %10s %10s %11s\n", "method", "n", "search", "build", "solve", "total")
    println("-"^68)
    for name in ("Full", "Reduced", "Hull / X", "Hull / F")
        for r in eachrow(df[df.method .== name, :])
            @printf("%-9s %4d %10.4f %10.4f %10.4f %11.4f\n",
                    name, r.n, r.search, r.build, r.solve, r.total)
        end
    end
    println("-"^68)

    println()
    println("AGREEMENT CHECK (max |score - score_reduced| per n)")
    for n in sort(unique(df.n))
        sub = df[df.n .== n, :]
        base = sub[sub.method .== "Reduced", :score]
        isempty(base) && continue
        @printf("  n=%d  max gap = %.3e\n", n, maximum(abs.(sub.score .- base[1])))
    end
    return nothing
end

# Command line:
#   (no args)                      run every method over NS, rewrite the CSV
#   <methods> [n-range]            run only those, merge into the existing CSV,
#                                  ignoring each method's default n cap, e.g.
#     julia --project=. devel/exp_reduction_partial.jl Full 6
if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS)
        certification_report(NS)
        println()
        df = run_benchmark()
    else
        wanted = Set(split(ARGS[1], ","))
        sel = [t for t in METHODS if t[1] in wanted]
        isempty(sel) && error("No method matched $(ARGS[1]). Known: $(join([t[1] for t in METHODS], ", "))")
        ns = length(ARGS) >= 2 ? eval(Meta.parse(ARGS[2])) : NS
        ns = ns isa Integer ? (ns:ns) : ns
        println("Running only: ", join([t[1] for t in sel], ", "), " over n = ", collect(ns),
                "  (merging into existing CSV)\n")
        df = run_benchmark(; ns = ns, methods = sel, merge_existing = true, cap = false)
    end
    print_summary(df)
end
