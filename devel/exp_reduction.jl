# =============================================================================
# exp_reduction.jl
#
# Joint-action reduction benchmark for the vertiport-occupancy CC-CE game
# (VertiportGame.jl cost model, i.e. the CC-CE paper's Eq. 22 -- NOT SetC).
#
# Compares five solve strategies as the number of queues n grows:
#
#   Full CC-CE  SearchCorrLPVertiport          mixes over all of X, |X| = m^n
#   Reduced     SearchCorrReducedLPVertiport   mixes over F only,   |F| = n^r,
#                                              deviations still over full 1:m
#   Hull / X    BruteNashBasedOptimizerLPVertiport
#                                              CC-PNE membership verified over
#                                              all of X, then hull LP
#   Hull / F    HullLPOverCandidateSet         same hull LP, membership check
#                                              scoped to F's n^r candidates
#   Hull direct HullLPDirect                   no search and no per-profile check
#                                              at all: Proposition 1's closed-form
#                                              certificate is tested once (n scalar
#                                              comparisons) and P_d = F is taken
#                                              directly. Only valid in this
#                                              certified regime -- the partial
#                                              certification run has no such row.
#
# Timing is reported as search / build / solve / total:
#   search -- CC-PNE enumeration (0 for full and reduced: F is combinatorial)
#   build  -- JuMP model construction (variables, constraints, objective)
#   solve  -- Ipopt's optimize! call
#   total  -- wall clock around the whole call (search+build+solve+overhead)
#
# A warm-up round at n = WARMUP_N is run first and DISCARDED, so the reported
# n = 3 row is steady-state rather than carrying Julia/Ipopt JIT compilation.
#
# Usage (from the repo root):
#   julia --project=. devel/exp_reduction.jl
# or inside a session that already has `using correlated`:
#   include("devel/exp_reduction.jl")
# =============================================================================

using correlated
using Printf
using DataFrames
using CSV
using Dates

# -----------------------------------------------------------------------------
# Parameters (match the benchmark artifact)
# -----------------------------------------------------------------------------
const R        = 3        # resources (runways/pads) -> m = 2^r = 8 actions/player
const GAMMA    = 2.0      # congestion multiplier γ (default case; override per run)
const ZALPHA   = 1.5      # z_α (confidence quantile)
const SIGMA    = 3.0      # σ_i, uniform across players
const DELTA    = 100.0    # Δ, fairness-threshold scale (hull LPs' objective)
const UNIT     = 5.0      # unit cost in Eq. 22
const NS       = 3:8      # queue counts to sweep
const FULL_MAX_N = 6      # full CC-CE (|X| = m^n variables) only up to here
const WARMUP_N = 3        # discarded warm-up round
const OUTFILE  = joinpath(@__DIR__, "..", "exp_reduction_results.csv")

# γ is the one parameter that moves between cases (certified vs. tight vs.
# uncertified), so it is threaded through every runner rather than read from the
# constant above. The certificate is z_α σ <= 5·min(1, γ-1):
#   γ = 2.0  -> bound 5.0, slack 0.5   certified
#   γ = 1.9  -> bound 4.5, slack 0.0   certified exactly at the boundary
#   γ < 1.9  -> bound < 4.5            uncertified: no profile is CC-PNE at all,
#                                      and the CC-CE LPs themselves go infeasible
const CASE_LABEL = Dict(2.0 => "certified", 1.9 => "tight", 1.5 => "uncertified")

# NOTE on objectives: the CC-CE LPs are called with objective = :aggregate
# (expected total cost, the paper's J_sys), while the two hull LPs are
# hardwired to the fairness-threshold epigraph objective in VertiportGame.jl.
# `score` below is the aggregate expected cost in every case, so the columns
# are comparable; for this cost model every exclusive assignment carries the
# same aggregate cost, which is why the two objectives land on the same score.
const CE_OBJECTIVE = :aggregate

# -----------------------------------------------------------------------------
# Certificate check (reduction note, Proposition 1): if z_α σ_i <= 5 min{1, γ-1}
# then every element of the exclusive-occupancy set F is a CC-PNE, uniformly in
# n -- which is what makes Hull / F valid without scanning X.
# -----------------------------------------------------------------------------
function certificate_bound(γ, unit)
    return unit * min(1.0, γ - 1.0)
end

# -----------------------------------------------------------------------------
# One (method, n) measurement -> NamedTuple row
# -----------------------------------------------------------------------------
function run_full(n)
    t = @elapsed res = SearchCorrLPVertiport(R, n, GAMMA, DELTA;
            zalpha = ZALPHA, sigma = SIGMA, unit = UNIT, objective = CE_OBJECTIVE)
    (; method = "Full", n, search = 0.0, build = res.buildTime,
       solve = res.solverTime, total = t, score = res.score,
       avgDelay = res.avgDelayScore, fair = res.fairScore, gini = res.giniScore,
       varsize = res.varsize, d = res.varsize, status = string(res.status))
end

function run_reduced(n)
    t = @elapsed res = SearchCorrReducedLPVertiport(R, n, GAMMA, DELTA;
            zalpha = ZALPHA, sigma = SIGMA, unit = UNIT, objective = CE_OBJECTIVE)
    (; method = "Reduced", n, search = 0.0, build = res.buildTime,
       solve = res.solverTime, total = t, score = res.score,
       avgDelay = res.avgDelayScore, fair = res.fairScore, gini = res.giniScore,
       varsize = res.varsize, d = res.d, status = string(res.status))
end

function run_hull_x(n)
    t = @elapsed res = BruteNashBasedOptimizerLPVertiport(R, n, GAMMA, DELTA;
            zalpha = ZALPHA, sigma = SIGMA, unit = UNIT)
    (; method = "Hull / X", n, search = res.searchTime, build = res.buildTime,
       solve = res.solverTime, total = t, score = res.score,
       avgDelay = res.avgDelayScore, fair = res.fairScore, gini = res.giniScore,
       varsize = res.varsize, d = res.d, status = string(res.status))
end

function run_hull_f(n)
    t = @elapsed res = HullLPOverCandidateSet(R, n, GAMMA, DELTA;
            zalpha = ZALPHA, sigma = SIGMA, unit = UNIT)
    (; method = "Hull / F", n, search = res.searchTime, build = res.buildTime,
       solve = res.solverTime, total = t, score = res.score,
       avgDelay = res.avgDelayScore, fair = res.fairScore, gini = res.giniScore,
       varsize = res.varsize, d = res.d, status = string(res.status))
end

function run_hull_direct(n)
    t = @elapsed res = HullLPDirect(R, n, GAMMA, DELTA;
            zalpha = ZALPHA, sigma = SIGMA, unit = UNIT)
    (; method = "Hull direct", n, search = res.searchTime, build = res.buildTime,
       solve = res.solverTime, total = t, score = res.score,
       avgDelay = res.avgDelayScore, fair = res.fairScore, gini = res.giniScore,
       varsize = res.varsize, d = res.d, status = string(res.status))
end

const METHODS = [
    ("Full",        run_full,        FULL_MAX_N),
    ("Reduced",     run_reduced,     last(NS)),
    ("Hull / X",    run_hull_x,      last(NS)),
    ("Hull / F",    run_hull_f,      last(NS)),
    ("Hull direct", run_hull_direct, last(NS)),
]

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
function run_benchmark(; ns = NS, warmup = true, outfile = OUTFILE,
                        methods = METHODS, merge_existing = false)
    m = 2^R
    bound = certificate_bound(GAMMA, UNIT)
    ok = ZALPHA * SIGMA <= bound

    println("="^78)
    println("Joint-action reduction benchmark -- vertiport CC-CE (Eq. 22 cost)")
    println("  r = $R  (m = 2^r = $m),  γ = $GAMMA,  z_α = $ZALPHA,  σ = $SIGMA,  Δ = $DELTA,  unit = $UNIT")
    @printf("  certificate: z_α σ = %.2f %s 5·min(1, γ-1) = %.2f  -> %s\n",
            ZALPHA * SIGMA, ok ? "<=" : ">", bound,
            ok ? "F is uniformly CC-PNE (Hull / F valid for all n)" : "NOT certified")
    println("  n sweep: $(collect(ns)),  full CC-CE capped at n <= $FULL_MAX_N")
    println("  solver: JuMP + Ipopt")
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
            if n > nmax
                @printf("  [skip] %-8s n=%d  (|X| = %s variables, out of reach)\n", name, n, string(big(m)^n))
                continue
            end
            row = f(n)
            push!(rows, row)
            @printf("  %-8s n=%d  score=%10.4f  search=%8.4f  build=%8.4f  solve=%8.4f  total=%9.4f  [%s]\n",
                    name, n, row.score, row.search, row.build, row.solve, row.total, row.status)
        end
        println()
    end

    df = DataFrame(rows)
    df.X = [big(m)^n for n in df.n]
    df.F = [big(n)^R for n in df.n]

    # Re-measuring only some methods: keep the other methods' existing rows and
    # replace just the ones this run produced.
    if merge_existing && isfile(outfile)
        old = CSV.read(outfile, DataFrame)
        names_run = unique(df.method)
        kept = old[.!in.(old.method, Ref(names_run)), :]
        df = vcat(kept, df; cols = :union)
        sort!(df, [:n, :method])
        println("Merged with $(nrow(kept)) existing rows from other methods.")
    end

    CSV.write(outfile, df)
    println("Wrote $(nrow(df)) rows to $outfile")
    return df
end

# -----------------------------------------------------------------------------
# Summary table in the artifact's layout: one row per n, score+total per method
# -----------------------------------------------------------------------------
function print_summary(df)
    m = 2^R
    println()
    println("="^131)
    println("TOTAL SOLVE TIME BY METHOD  (score = aggregate expected cost)")
    println("="^131)
    @printf("%-4s %12s %6s | %22s | %22s | %22s | %22s | %22s\n",
            "n", "|X|=m^n", "|F|", "Full CC-CE", "Reduced", "Hull / X", "Hull / F", "Hull direct")
    @printf("%-4s %12s %6s | %10s %11s | %10s %11s | %10s %11s | %10s %11s | %10s %11s\n",
            "", "", "", "score", "total(s)", "score", "total(s)", "score", "total(s)", "score", "total(s)", "score", "total(s)")
    println("-"^131)
    for n in sort(unique(df.n))
        @printf("%-4d %12s %6d ", n, string(big(m)^n), n^R)
        for name in ("Full", "Reduced", "Hull / X", "Hull / F", "Hull direct")
            sub = df[(df.n .== n) .& (df.method .== name), :]
            if nrow(sub) == 0
                @printf("| %10s %11s ", "n/a", "n/a")
            else
                @printf("| %10.4f %11.3f ", sub.score[1], sub.total[1])
            end
        end
        println()
    end
    println("="^131)

    println()
    println("TIME BREAKDOWN (s)")
    println("-"^68)
    @printf("%-9s %4s %10s %10s %10s %11s\n", "method", "n", "search", "build", "solve", "total")
    println("-"^68)
    for name in ("Full", "Reduced", "Hull / X", "Hull / F", "Hull direct")
        for r in eachrow(df[df.method .== name, :])
            @printf("%-9s %4d %10.4f %10.4f %10.4f %11.4f\n",
                    name, r.n, r.search, r.build, r.solve, r.total)
        end
    end
    println("-"^68)

    # cross-method agreement check
    println()
    println("AGREEMENT CHECK (max |score - score_reduced| per n)")
    for n in sort(unique(df.n))
        sub = df[df.n .== n, :]
        base = sub[sub.method .== "Reduced", :score]
        isempty(base) && continue
        gap = maximum(abs.(sub.score .- base[1]))
        @printf("  n=%d  max gap = %.3e  (methods: %s)\n", n, gap, join(sub.method, ", "))
    end
    return nothing
end

# Command line: no arguments runs every method and rewrites the CSV; naming
# methods runs only those and merges them into the existing CSV, e.g.
#   julia --project=. devel/exp_reduction.jl "Hull direct"
if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS)
        df = run_benchmark()
    else
        wanted = Set(ARGS)
        sel = [t for t in METHODS if t[1] in wanted]
        isempty(sel) && error("No method matched $(ARGS). Known: $(join([t[1] for t in METHODS], ", "))")
        println("Running only: ", join([t[1] for t in sel], ", "), "  (merging into existing CSV)\n")
        df = run_benchmark(; methods = sel, merge_existing = true)
    end
    print_summary(df)
end
