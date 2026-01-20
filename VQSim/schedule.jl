module VQSchedule

using CSV, DataFrames

export Flight, load_schedule

struct Flight
    flight_id::String
    airline_id::Int
    sched_t::Int
    ready_t::Int
    runway::Int
    pax::Int
end

function load_schedule(csv_path::AbstractString)::Vector{Flight}
    df = CSV.read(csv_path, DataFrame)

    # ready_t optional
    if !haskey(df, :ready_t)
        df.ready_t = df.sched_t
    end

    flights = Vector{Flight}(undef, nrow(df))
    for (k, row) in enumerate(eachrow(df))
        flights[k] = Flight(
            String(row.flight_id),
            Int(row.airline_id),
            Int(row.sched_t),
            Int(row.ready_t),
            Int(row.runway),
            Int(row.pax),
        )
    end
    return flights
end

end # module
