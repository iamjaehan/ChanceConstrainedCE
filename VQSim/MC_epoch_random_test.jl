# experiments/MC_epoch_test.jl
using Random
using Distributions
using Statistics
using CSV, DataFrames

# ----------------------------
# Epoch generator
# ----------------------------
"""
"그럴듯한 epoch"를 하나 만들어서 반환.
목표: 기존 cost/정의 그대로 유지하면서, B0/Q0/runways/mu/late 분포 등을 파라미터로 제어.

반환:
    flights, state, active_by_airline, active_airlines
"""
function build_fixed_epoch!(
    rng::AbstractRNG,
    flights::Vector{Flight},
    state;
    params::VQState.SimParams,
    B0_total::Int,
    Q0_runway::Vector{Int},
    t_epoch::Int,
    lateness_mean::Float64 = 0.0,
    lateness_std::Float64  = 8.0,
    min_airlines::Int = 2,
)
    # (1) epoch state 세팅
    state.t = t_epoch
    state.Q .= Q0_runway

    N = length(flights)
    if N == 0
        error("No flights loaded from schedule.")
    end

    if B0_total > N
        @warn "B0_total($B0_total) > N($N). Clamping B0_total to $N."
        B0_total = N
    end
    if B0_total < 1
        error("B0_total must be >= 1 after clamping. Current B0_total=$B0_total, N=$N")
    end

    # epoch 전용 flights 복사본 (Flight는 immutable이라 원소 교체로 바꿈)
    flights_epoch = copy(flights)

    function sample_active_by_airline!()
        idx = randperm(rng, N)[1:B0_total]
        active_by_airline = Dict{Int,Vector{Int}}()

        for i in idx
            f = flights_epoch[i]

            aid = f.airline_id
            active_by_airline[aid] = get(active_by_airline, aid, Int[])
            push!(active_by_airline[aid], i)

            # lateness를 유도할 sched_t 생성
            L = max(0.0, lateness_mean + lateness_std * randn(rng))
            new_sched_t = max(0, Int(round(t_epoch - L)))

            # ✅ 여기서 "필드 수정"이 아니라 "새 Flight로 교체"
            flights_epoch[i] = Flight(
                f.flight_id,
                f.airline_id,
                new_sched_t,
                f.ready_t,
                f.runway,
                f.pax
            )
        end

        return active_by_airline
    end

    active_by_airline = sample_active_by_airline!()
    active_airlines = sort(collect(keys(active_by_airline)))

    # airline 다양성 확보 시도
    if length(active_airlines) < min_airlines
        got = false
        for _ in 1:50
            flights_epoch = copy(flights)  # 재샘플할 때는 epoch flights도 새로
            active_by_airline = sample_active_by_airline!()
            active_airlines = sort(collect(keys(active_by_airline)))
            if length(active_airlines) >= min_airlines
                got = true
                break
            end
        end
        if !got
            @warn "Could not meet min_airlines=$min_airlines with B0_total=$B0_total (N=$N). Proceeding with n_airlines=$(length(active_airlines))."
        end
    end

    return flights_epoch, state, active_by_airline, active_airlines
end

# ----------------------------
# MC runner (1 epoch 고정, MC에서 rng만 변경)
# ----------------------------
Base.@kwdef struct MCEpochConfig
    csv_path::String
    params::VQState.SimParams
    max_subset_size::Int

    # epoch shape knobs
    B0_total::Int
    Q0_runway::Vector{Int}
    t_epoch::Int
    lateness_mean::Float64 = 0.0
    lateness_std::Float64  = 8.0

    # solver knobs
    Δ::Float64 = 0.0
    lambda_fair::Float64 = 1.0
    rho_release::Float64 = 0.0
    enable_deviation::Bool = true

    # uncertainty knobs
    coord_sigma_mode::SigmaMode = SIGMA_SCALAR
    coord_sigma_scalar::Float64 = 0.0
    coord_sigma_vec::Vector{Float64} = Float64[]
    real_sigma_mode::SigmaMode = SIGMA_SCALAR
    real_sigma_scalar::Float64 = 0.0
    real_sigma_vec::Vector{Float64} = Float64[]

    # experiment knobs
    N_mc::Int = 50
    base_seed::Int = 1
    solver_modes::Vector{SolverMode} = SolverMode[]
end


function run_mc_epoch_test(cfgE::MCEpochConfig; out_csv::String)
    # 0) load flight pool once
    flights_pool = VQSchedule.load_schedule(cfgE.csv_path)

    # 1) base SimConfig (solver_mode만 루프에서 교체)
    base_cfg = SimConfig(
        csv_path = cfgE.csv_path,
        params = cfgE.params,
        T_sim = 1,
        max_subset_size = cfgE.max_subset_size,
        lambda_fair = cfgE.lambda_fair,
        rho_release = cfgE.rho_release,
        Δ = cfgE.Δ,
        zalpha = quantile(Normal(), 0.9),
        seed = cfgE.base_seed,
        coord_sigma_mode = cfgE.coord_sigma_mode,
        coord_sigma_scalar = cfgE.coord_sigma_scalar,
        coord_sigma_vec = cfgE.coord_sigma_vec,
        real_sigma_mode = cfgE.real_sigma_mode,
        real_sigma_scalar = cfgE.real_sigma_scalar,
        real_sigma_vec = cfgE.real_sigma_vec,
        enable_deviation = cfgE.enable_deviation
    )

    rows = NamedTuple[]

    for mc in 1:cfgE.N_mc
        # --- (A) epoch RNG: mc마다 epoch 새로 만들되 재현 가능하게 ---
        rng_epoch = MersenneTwister(cfgE.base_seed + 1_000_000 + mc)

        # state도 epoch마다 새로 (solve_epoch 내부가 state를 건드릴 수도 있으니 안전하게)
        state = init_state(cfgE.params, length(flights_pool))

        # epoch build (flights도 epoch 버전으로 생성)
        flights, state, active_by_airline, active_airlines = build_fixed_epoch!(
            rng_epoch, flights_pool, state;
            params=cfgE.params,
            B0_total=cfgE.B0_total,
            Q0_runway=cfgE.Q0_runway,
            t_epoch=cfgE.t_epoch,
            lateness_mean=cfgE.lateness_mean,
            lateness_std=cfgE.lateness_std
        )

        n_backlog_total = sum(length(active_by_airline[aid]) for aid in active_airlines)

        # --- (B) action tensors: epoch 바뀌니 매 mc마다 다시 생성 ---
        actions_by_player = Vector{Vector{Vector{Int}}}()
        for aid in active_airlines
            elig = active_by_airline[aid]
            acts = enumerate_actions_subset(elig; max_subset_size=cfgE.max_subset_size)
            push!(actions_by_player, acts)
        end
        joint_pushed, joint_choice = enumerate_joint_actions(actions_by_player)

        # --- (C) solver RNG: 같은 epoch에서 solver별 결과 비교하려면, solver마다 seed 고정 ---
        # (mc, solver)마다 RNG를 분리해주면, solver간 비교가 fair해짐
        for smode in cfgE.solver_modes
            rng_solver = MersenneTwister(cfgE.base_seed + 10_000 + 100*mc + Int(smode))

            cfgS = deepcopy(base_cfg)
            cfgS.solver_mode = smode

            t0 = time()
            status = "OK"
            pushed = Int[]
            solver_time = NaN
            solver_detail = Dict{String,Any}()

            try
                pushed, solver_time, solver_detail = solve_epoch(cfgS, rng_solver,
                    state, flights, active_airlines, active_by_airline,
                    actions_by_player, joint_pushed, joint_choice, cfgE.params)
            catch err
                status = string(typeof(err))
            end
            wall_ms = (time() - t0) * 1e3

            obj = missing
            fair_gap = missing
            num_pne = get(solver_detail, "num_pne", missing)
            dev_rate = get(solver_detail, "dev_rate", missing)
            n_dev = get(solver_detail, "n_dev", missing)

            if status == "OK"
                J_coord, _, _, _, air_avg_unw, _ = compute_epoch_metrics(
                    state, pushed, flights, active_airlines, active_by_airline, state.t, cfgE.params;
                    lambda_fair=cfgE.lambda_fair,
                    rho_release=cfgE.rho_release
                )
                vals = collect(values(air_avg_unw))
                fair_gap = isempty(vals) ? 0.0 : (maximum(vals) - minimum(vals))
                obj = J_coord
            end

            push!(rows, (
                mc = mc,
                solver = string(smode),
                status = status,
                n_backlog_total = n_backlog_total,
                n_airlines = length(active_airlines),
                Q0_runway = join(cfgE.Q0_runway, ";"),
                B0_total = cfgE.B0_total,
                t_epoch = cfgE.t_epoch,
                lateness_mean = cfgE.lateness_mean,
                lateness_std = cfgE.lateness_std,
                coord_sigma = cfgE.coord_sigma_scalar,
                real_sigma  = cfgE.real_sigma_scalar,
                obj = obj,
                fairness_gap = fair_gap,
                n_pushed = length(pushed),
                n_dev = n_dev,
                dev_rate = dev_rate,
                solver_time_sec = solver_time,
                wall_ms = wall_ms,
                num_pne = num_pne
            ))

            println(
                "mc=", mc, "/", cfgE.N_mc, " | ",
                "epoch_seed=", (cfgE.base_seed + 1_000_000 + mc), " | ",
                "solver=", smode, " | ",
                "status=", status, " | ",
                "obj=", obj === missing ? "NA" : round(obj, digits=2), " | ",
                "pushed=", length(pushed), " | ",
                "wall_ms=", round(wall_ms, digits=1), " | ",
                "backlog=", n_backlog_total, " | ",
                "airlines=", length(active_airlines)
            )
        end
    end

    df = DataFrame(rows)
    CSV.write(out_csv, df)
    return df
end

cfgE = MCEpochConfig(
    csv_path = "schedule/flight_schedule_1h_5b.csv",
    params = VQState.SimParams(2, [2,2]),
    max_subset_size = 1024,
    B0_total = 6,
    Q0_runway = [3, 4],
    t_epoch = 0,
    lateness_mean = 10.0,
    lateness_std = 10.0,
    Δ = 1e12,
    lambda_fair = 1.0,
    rho_release = 0.0,
    enable_deviation = true,
    coord_sigma_mode = SIGMA_SCALAR,
    coord_sigma_scalar = 0,
    coord_sigma_vec = Float64[],
    real_sigma_mode = SIGMA_SCALAR,
    real_sigma_scalar = 0,
    real_sigma_vec = Float64[],
    N_mc = 30,
    base_seed = rand(1:10000,1)[1],
    solver_modes = [GREEDY_CENTRALIZED, AGG_ORACLE_FCFS, CE_FULL, CE_NAIVE, RRCE_PNE]
)

df = run_mc_epoch_test(cfgE; out_csv="mc_epoch_results_6a_0s_90c.csv")
