# =============================================================================
# exp_reduction_to_mat.jl
#
# Converts the joint-action reduction benchmark results (exp_reduction.jl's CSV)
# into a MATLAB .mat file laid out for plotting.
#
#   julia --project=. devel/exp_reduction_to_mat.jl
#   julia --project=. devel/exp_reduction_to_mat.jl in.csv out.mat
#
# Layout of the .mat (all time fields in seconds, NaN where a method was skipped):
#
#   n, X, F          6x1 doubles -- queue count, |X| = m^n, |F| = n^r
#   methods          1x4 cell    -- display names, in plot order
#   full, reduced, hullX, hullF
#                    structs with fields score, total, search, build, solve (6x1)
#   params           struct: r, m, gamma, zalpha, sigma, delta, unit
#   tbl              long-form struct: method (cell), n, search, build, solve,
#                    total, score, fair, gini, varsize, d, status -- one entry
#                    per run, for anything the wide layout doesn't cover
# =============================================================================

using CSV
using DataFrames
using MAT

const METHOD_KEYS = [("Full", "full"), ("Reduced", "reduced"),
                     ("Hull / X", "hullX"), ("Hull / F", "hullF"),
                     ("Hull direct", "hullDirect")]

"""
Pull one method's per-n series out of the long-form frame, padding with NaN for
the n values that method never ran (full CC-CE above n = 6).
"""
function method_series(df::DataFrame, name::AbstractString, ns::Vector{Int})
    sub = df[df.method .== name, :]
    grab = function (col)
        out = fill(NaN, length(ns))
        for (i, n) in enumerate(ns)
            row = sub[sub.n .== n, :]
            nrow(row) == 1 && (out[i] = Float64(row[1, col]))
        end
        out
    end
    return Dict{String,Any}(
        "score"  => grab(:score),
        "total"  => grab(:total),
        "search" => grab(:search),
        "build"  => grab(:build),
        "solve"  => grab(:solve),
    )
end

function build_mat_dict(df::DataFrame; r = 3, gamma = 2.0, zalpha = 1.5,
                        sigma = 3.0, delta = 100.0, unit = 5.0)
    ns = sort(unique(df.n))
    m = 2^r

    out = Dict{String,Any}(
        "n"       => Float64.(ns),
        "X"       => Float64.(m .^ ns),
        "F"       => Float64.(ns .^ r),
        "methods" => [name for (name, _) in METHOD_KEYS],
        "params"  => Dict{String,Any}(
            "r" => Float64(r), "m" => Float64(m), "gamma" => gamma,
            "zalpha" => zalpha, "sigma" => sigma, "delta" => delta,
            "unit" => unit,
        ),
    )

    for (name, key) in METHOD_KEYS
        out[key] = method_series(df, name, ns)
    end

    # long-form table: everything, one entry per run
    out["tbl"] = Dict{String,Any}(
        "method"  => String.(df.method),
        "n"       => Float64.(df.n),
        "search"  => Float64.(df.search),
        "build"   => Float64.(df.build),
        "solve"   => Float64.(df.solve),
        "total"   => Float64.(df.total),
        "score"   => Float64.(df.score),
        "fair"    => Float64.(df.fair),
        "gini"    => Float64.(df.gini),
        "varsize" => Float64.(df.varsize),
        "d"       => Float64.(df.d),
        "status"  => String.(df.status),
    )

    return out
end

function convert_results(incsv, outmat)
    df = CSV.read(incsv, DataFrame)
    data = build_mat_dict(df)
    matwrite(outmat, data)

    println("Read  $(nrow(df)) rows from $incsv")
    println("Wrote $outmat")
    println("  n       = ", Int.(data["n"]))
    for (name, key) in METHOD_KEYS
        tot = data[key]["total"]
        shown = join([isnan(t) ? "  n/a" : string(round(t, digits = 3)) for t in tot], ", ")
        println("  $(rpad(name, 8)) total(s) = [", shown, "]")
    end
    return data
end

const DEFAULT_CSV = joinpath(@__DIR__, "..", "exp_reduction_results.csv")
const DEFAULT_MAT = joinpath(@__DIR__, "..", "exp_reduction_results.mat")

if abspath(PROGRAM_FILE) == @__FILE__
    incsv  = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_CSV
    outmat = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_MAT
    convert_results(incsv, outmat)
end
