# Minimal CSV reading and writing.
#
# CSV.jl/DataFrames build their schema at runtime, which is structurally
# incompatible with trimming. The input schema here is fixed and known, so a
# hand-written parser is both simpler and trim-clean.
#
# Expected input columns (header required, order fixed):
#   time, id, captured, test1, test2, test3, test4, test5, test6
# matching the `test_mat` layout the R package passes to run_hmm_inference.

"Parse a Float64 without pulling in the generic `parse` machinery's error paths."
function parse_f64(s::AbstractString)
    v = tryparse(Float64, s)
    v === nothing && return NaN
    return v
end

"Split one CSV line on commas. Quoted fields are not supported: the input is numeric."
function split_line(line::AbstractString)
    out = String[]
    start = 1
    n = ncodeunits(line)
    i = 1
    while i <= n
        if codeunit(line, i) == UInt8(',')
            push!(out, String(SubString(line, start, i - 1)))
            start = i + 1
        end
        i += 1
    end
    push!(out, String(SubString(line, start, n)))
    return out
end

"""
    read_matrix(path) -> Matrix{Float64}

Read a numeric CSV with a header row into a dense matrix. Empty fields and
unparseable values become NaN, which is how a missing test result is encoded.
"""
function read_matrix(path::String)
    rows = Vector{Vector{Float64}}()
    ncol = 0
    io = open(path, "r")
    try
        first = true
        while !eof(io)
            line = readline(io)
            isempty(strip(line)) && continue
            if first
                first = false
                ncol = length(split_line(line))
                continue
            end
            parts = split_line(line)
            vals = Vector{Float64}(undef, length(parts))
            for k in eachindex(parts)
                vals[k] = parse_f64(strip(parts[k]))
            end
            push!(rows, vals)
        end
    finally
        close(io)
    end
    isempty(rows) && error("no data rows in $path")
    m = length(rows)
    n = length(rows[1])
    out = Matrix{Float64}(undef, m, n)
    for i in 1:m
        length(rows[i]) == n || error("ragged row $i in $path")
        for j in 1:n
            out[i, j] = rows[i][j]
        end
    end
    return out
end

"""
    build_data(mat, test_mask, repeat_captures) -> HMMData

Turn the raw (time, id, captured, test1..6) matrix into the flattened CSR form
the likelihood consumes.

Each badger is expanded onto the full timestep grid from its first to its last
capture, matching the package's `build_gridded_sequences`. Timesteps with no
capture contribute no emissions but still advance the hazard, so supplying only
capture rows and supplying the full grid give identical results. NaN test values
and masked-out assays are skipped.

`repeat_captures`:
  :stack -- every capture in a timestep contributes its own emission terms
  :pool  -- captures merged per timestep, positive if any was positive
  :last  -- keep only the final capture in each timestep
"""
function build_data(mat::Matrix{Float64}, test_mask::Vector{Bool},
                    repeat_captures::Symbol, S::Int)
    nrow = size(mat, 1)
    size(mat, 2) >= 3 + N_TESTS || error("expected at least $(3 + N_TESTS) columns")

    ids_all = Int[]
    for i in 1:nrow
        push!(ids_all, Int(mat[i, 2]))
    end
    uids = sort(unique(ids_all))
    id_pos = Dict{Int,Int}()
    for (k, u) in enumerate(uids)
        id_pos[u] = k
    end
    n_ind = length(uids)

    # Group row indices by badger, preserving time order.
    per_ind = [Int[] for _ in 1:n_ind]
    for i in 1:nrow
        push!(per_ind[id_pos[ids_all[i]]], i)
    end
    for v in per_ind
        sort!(v, by = i -> mat[i, 1])
    end

    times = Int[]
    ptr = Int[1]
    vals = Int[]
    tidx = Int[]
    seq_ends = Int[]
    entry_year = Int[]
    ids = Int[]
    captured = Bool[]

    for k in 1:n_ind
        rowsk = per_ind[k]
        isempty(rowsk) && continue

        # Expand to the FULL grid from first to last capture, exactly as the
        # package's build_gridded_sequences does. This is not cosmetic: an
        # uncaptured timestep carries no emission but still advances the
        # hazard by one lam, so a badger seen at t=10 and t=20 must contribute
        # ten transitions, not one. Stepping capture-to-capture instead is a
        # DIFFERENT MODEL -- on the real cohort it changes the number of
        # transitions from 29,737 to 14,986 and the log density by ~174.
        # The input may or may not already contain captured==0 filler rows;
        # building the grid here makes the result identical either way.
        first_t = typemax(Int)
        last_t = typemin(Int)
        for i in rowsk
            t = Int(mat[i, 1])
            t < first_t && (first_t = t)
            t > last_t && (last_t = t)
        end
        tsteps = Int[]
        for t in first_t:last_t
            push!(tsteps, t)
        end

        push!(entry_year, year_of(first_t, S))
        push!(ids, uids[k])

        for t in tsteps
            # Which raw rows fall in this timestep?
            inrows = Int[]
            for i in rowsk
                Int(mat[i, 1]) == t && push!(inrows, i)
            end
            if repeat_captures === :last && length(inrows) > 1
                inrows = [inrows[end]]
            end

            push!(times, t)
            # Was the badger actually caught here, as opposed to this being a
            # filler step on the grid? Prevalence is reported over CAPTURED
            # badgers (matching the package's prevalence_capture), so the two
            # must be distinguishable -- the grid expansion above makes most
            # timesteps non-captures.
            was_caught = false
            for i in inrows
                mat[i, 3] != 0.0 && (was_caught = true)
            end
            push!(captured, was_caught)
            if repeat_captures === :pool && length(inrows) > 1
                # One emission per assay: positive if any capture was positive.
                for c in 1:N_TESTS
                    test_mask[c] || continue
                    any_obs = false
                    any_pos = false
                    for i in inrows
                        v = mat[i, 3 + c]
                        isnan(v) && continue
                        mat[i, 3] == 0.0 && continue
                        any_obs = true
                        v == 1.0 && (any_pos = true)
                    end
                    if any_obs
                        push!(vals, any_pos ? 1 : 0)
                        push!(tidx, c)
                    end
                end
            else
                for i in inrows
                    mat[i, 3] == 0.0 && continue   # not captured: no emissions
                    for c in 1:N_TESTS
                        test_mask[c] || continue
                        v = mat[i, 3 + c]
                        isnan(v) && continue
                        push!(vals, v == 1.0 ? 1 : 0)
                        push!(tidx, c)
                    end
                end
            end
            push!(ptr, length(vals) + 1)
        end
        push!(seq_ends, length(times))
    end

    n_years = isempty(times) ? 1 : maximum(year_of(t, S) for t in times)
    return HMMData(length(seq_ends), S, n_years, seq_ends, entry_year,
                   times, ptr, vals, tidx, ids, captured)
end

"""
    read_metric(path) -> Matrix{Float64}

Read an inverse mass matrix from a headerless CSV: one row per line, `n` comma
separated values each, `n` lines.

Kept separate from `read_matrix` because that one skips a header row, which a
metric file does not have -- and silently dropping the first row of a mass
matrix would produce a non-square matrix or, worse, a plausible-looking wrong
one. A diagonal metric may also be given as a single row or a single column of
`n` values, which is expanded here.
"""
function read_metric(path::String)
    rows = Vector{Vector{Float64}}()
    io = open(path, "r")
    try
        while !eof(io)
            line = readline(io)
            isempty(strip(line)) && continue
            startswith(strip(line), "#") && continue
            parts = split_line(line)
            vals = Vector{Float64}(undef, length(parts))
            for k in eachindex(parts)
                vals[k] = parse_f64(strip(parts[k]))
            end
            push!(rows, vals)
        end
    finally
        close(io)
    end
    isempty(rows) && error("metric file $path is empty")

    nr = length(rows)
    nc = length(rows[1])
    for r in rows
        length(r) == nc || error("metric file $path has ragged rows")
    end

    # A single row or single column of n values is a diagonal metric.
    if nr == 1 && nc > 1
        m = zeros(nc, nc)
        for i in 1:nc
            m[i, i] = rows[1][i]
        end
        return m
    elseif nc == 1 && nr > 1
        m = zeros(nr, nr)
        for i in 1:nr
            m[i, i] = rows[i][1]
        end
        return m
    end

    nr == nc || error("metric file $path is $(nr)x$(nc); expected a square matrix")
    m = Matrix{Float64}(undef, nr, nc)
    for i in 1:nr, j in 1:nc
        m[i, j] = rows[i][j]
    end
    for i in 1:nr, j in 1:nc
        isfinite(m[i, j]) || error("metric file $path contains a non-finite value")
    end
    return m
end

"""
    write_metric(path, m)

Write an inverse mass matrix as a headerless CSV, so a tuned metric can be
exported from one run and fed to another with `--metric`.
"""
function write_metric(path::String, m::Matrix{Float64})
    io = open(path, "w")
    try
        n = size(m, 1)
        for i in 1:n
            for j in 1:size(m, 2)
                j > 1 && write(io, ",")
                # `string(::Float64)` round-trips exactly; `fmt` rounds to 8
                # digits, which is fine for reported quantities but would lose
                # real precision in a mass matrix whose entries can be small.
                write(io, string(m[i, j]))
            end
            write(io, "
")
        end
    finally
        close(io)
    end
    return nothing
end

# --- output ----------------------------------------------------------------
"Format a Float64 for CSV output without Printf (which is a stdlib load)."
function fmt(x::Float64)
    isnan(x) && return "NA"
    isinf(x) && return x > 0 ? "Inf" : "-Inf"
    return string(round(x, digits = 8))
end

function write_csv(path::String, header::Vector{String}, cols::Vector{Vector{Float64}})
    io = open(path, "w")
    try
        for j in eachindex(header)
            j > 1 && write(io, ",")
            write(io, header[j])
        end
        write(io, "\n")
        n = isempty(cols) ? 0 : length(cols[1])
        for i in 1:n
            for j in eachindex(cols)
                j > 1 && write(io, ",")
                write(io, fmt(cols[j][i]))
            end
            write(io, "\n")
        end
    finally
        close(io)
    end
    return path
end
