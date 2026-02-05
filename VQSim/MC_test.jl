# =========================
# MC_test.jl
# Monte Carlo driver for simulator.jl
# =========================

using Random
using Statistics
using Dates
using Printf

# If you want CSV output:
using CSV
using DataFrames

include("simulator.jl")  # assumes SimConfig, SolverMode, SigmaMode, run_simulation are defined

# -------------------------
# Experiment sweep spec
# -------------------------
Base.@kwdef mutable struct MCSweep
    seeds::Vector{Int}
    n_reps::Int = 1
    solver_modes::Vector{SolverMode}

    # uncertainty sweep (scalar sigma)
    coord_sigmas::Vector{Float64}
    real_sigmas::Vector{Float64}

    # flags
    enable_deviation::Bool = true

    # sim config shared
    csv_path::String = "schedule/flight_schedule_1h_5b.csv"
    T_sim::Int = 16
    max_subset_size::Int = 1024
    lambda_fair::Float64 = 1.0
    rho_release::Float64 = 0.1
    Δ::Float64 = 0.0
    α::Float64 = 0.9
end

function make_cfg(base::MCSweep; seed::Int, solver::SolverMode, coord_sigma::Float64, real_sigma::Float64)
    zalpha = quantile(Normal(), base.α)

    return SimConfig(
        csv_path = base.csv_path,
        T_sim = base.T_sim,
        max_subset_size = base.max_subset_size,
        lambda_fair = base.lambda_fair,
        rho_release = base.rho_release,
        Δ = base.Δ,
        α = base.α,
        zalpha = zalpha,
        solver_mode = solver,
        seed = seed,

        # uncertainty assumed by coordinator
        coord_sigma_mode = (coord_sigma == 0.0 ? SIGMA_ZERO : SIGMA_SCALAR),
        coord_sigma_scalar = coord_sigma,

        # uncertainty realized at execution / deviation
        real_sigma_mode = (real_sigma == 0.0 ? SIGMA_ZERO : SIGMA_SCALAR),
        real_sigma_scalar = real_sigma,

        enable_deviation = base.enable_deviation
    )
end

# -------------------------
# Helper: Extract time-series metrics from logs
# -------------------------
function summarize_timeseries(res::SimResult)
    logs = res.logs
    T = length(logs)

    t = [logs[i].t for i in 1:T]
    Qsum = [sum(logs[i].Q) for i in 1:T]
    n_backlog = [logs[i].n_backlog_total for i in 1:T]
    n_pushed  = [logs[i].n_pushed for i in 1:T]
    Jcoord    = [logs[i].J_coord for i in 1:T]
    solve_t   = [logs[i].solver_time_sec for i in 1:T]

    # per-epoch fairness based on airline_cost_avg (max-min across airlines that epoch)
    fairness_ep = Float64[]
    for lg in logs
        vals = Float64[]
        for (_, v) in lg.airline_cost_avg
            isfinite(v) && push!(vals, v)
        end
        push!(fairness_ep, isempty(vals) ? 0.0 : (maximum(vals) - minimum(vals)))
    end

    # You asked: "시점별 항공사별 delay, coordinator cost"
    # We'll store airline_cost_avg dictionaries per epoch in a separate table later if needed.
    return (
        t=t,
        Qsum=Qsum,
        n_backlog=n_backlog,
        n_pushed=n_pushed,
        Jcoord=Jcoord,
        solve_t=solve_t,
        fairness_ep=fairness_ep
    )
end

# -------------------------
# Core MC runner
# -------------------------
function run_mc(sweep::MCSweep; outdir::String="mc_out")
    mkpath(outdir)

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_csv = joinpath(outdir, "mc_summary_$timestamp.csv")
    out_csv_epoch = joinpath(outdir, "mc_epoch_$timestamp.csv")

    summary_rows = DataFrame()
    epoch_rows = DataFrame()

    total_jobs = length(sweep.seeds) * length(sweep.solver_modes) * length(sweep.coord_sigmas) * length(sweep.real_sigmas) * sweep.n_reps
    job_idx = 0
    for solver in sweep.solver_modes
        for coord_sigma in sweep.coord_sigmas
            for real_sigma in sweep.real_sigmas
                for seed in sweep.seeds
                    for rep in 1:sweep.n_reps
                        job_idx += 1
    
                        run_seed = 10_000 * seed + rep
    
                        @printf("\n[%d/%d] solver=%s coordσ=%.3g realσ=%.3g seed=%d rep=%d run_seed=%d\n",
                                job_idx, total_jobs, string(solver), coord_sigma, real_sigma, seed, rep, run_seed)
    
                        cfg = make_cfg(sweep; seed=run_seed, solver=solver, coord_sigma=coord_sigma, real_sigma=real_sigma)
    
                        res = run_simulation(cfg)
                        ts = summarize_timeseries(res)
    
                        push!(summary_rows, (
                            timestamp = timestamp,
                            solver = string(solver),
                            seed = seed,
                            rep = rep,
                            run_seed = run_seed,
                            coord_sigma = coord_sigma,
                            real_sigma = real_sigma,
                            enable_deviation = cfg.enable_deviation,
                            total_time_sec = res.total_time_sec,
                            sum_solver_time_sec = sum(ts.solve_t),
                            total_obj = res.total_coord_obj,
                            avg_Qsum = mean(ts.Qsum),
                            avg_backlog = mean(ts.n_backlog),
                            avg_pushed = mean(ts.n_pushed),
                            fairness_max_gap = res.fairness_max_gap
                        ))
    
                        for k in eachindex(ts.t)
                            push!(epoch_rows, (
                                timestamp = timestamp,
                                solver = string(solver),
                                seed = seed,
                                rep = rep,               # ✅ 저장
                                run_seed = run_seed,     # ✅ 저장
                                coord_sigma = coord_sigma,
                                real_sigma = real_sigma,
                                epoch = k,
                                t = ts.t[k],
                                Qsum = ts.Qsum[k],
                                n_backlog = ts.n_backlog[k],
                                n_pushed = ts.n_pushed[k],
                                J_coord = ts.Jcoord[k],
                                solver_time_sec = ts.solve_t[k],
                                fairness_ep = ts.fairness_ep[k],
                            ))
                        end
                    end
                end
            end
        end
    end    

    CSV.write(out_csv, summary_rows)
    CSV.write(out_csv_epoch, epoch_rows)

    println("\nSaved:")
    println("  summary: $out_csv")
    println("  epoch  : $out_csv_epoch")

    return summary_rows, epoch_rows
end

# -------------------------
# Default sweep (edit as needed)
# -------------------------
sweep = MCSweep(
    seeds = collect(1:10),
    n_reps = 1,

    solver_modes = [
        GREEDY_CENTRALIZED,
        AGG_ORACLE_FCFS,
        CE_NAIVE,
        CE_FULL,
        RRCE_PNE
    ],

    # single sigma case (same for coordinator + realization)
    coord_sigmas = [200.0],
    real_sigmas  = [200.0],

    enable_deviation = true,   

    csv_path = "schedule/flight_schedule_1h_5b.csv",
    T_sim = 16,
    max_subset_size = 1024,
    lambda_fair = 1.0,
    rho_release = 0.0,
    Δ = 100.0,
    α = 0.95
)

sweep = MCSweep(
    seeds = collect(1:2),
    n_reps = 1,

    # solver_modes = [
    #     GREEDY_CENTRALIZED,
    #     AGG_ORACLE_FCFS,
    #     CE_NAIVE,
    #     CE_FULL,
    #     RRCE_PNE
    # ],
    solver_modes = [
        GREEDY_CENTRALIZED,
        AGG_ORACLE_FCFS,
        CE_NAIVE,
        RRCE_PNE
    ],

    # single sigma case (same for coordinator + realization)
    coord_sigmas = [0.0],
    real_sigmas  = [0.0],

    enable_deviation = true,
    # csv_path = "schedule/flight_schedule_1h.csv",
    csv_path = "schedule/flight_schedule_1h_5b.csv",
    T_sim = 16,
    max_subset_size = 1024,
    lambda_fair = 1.0,
    rho_release = 0.0,
    Δ = 1e12,
    α = 0.95
)

summary_df, epoch_df = run_mc(sweep; outdir="mc_out")

println("\nMC done. Summary rows = ", nrow(summary_df), ", Epoch rows = ", nrow(epoch_df))
