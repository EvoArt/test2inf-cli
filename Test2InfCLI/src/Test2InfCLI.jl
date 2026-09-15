"""
    Test2InfCLI

Standalone, trimmable CLI for the badger diagnostic HMM.

This is a dependency-free reimplementation of the inference kernel in
`test2infeR/inst/julia/Test2InfEngine`. Turing, ForwardDiff, Optimization,
HiddenMarkovModels and Distributions are all replaced by hand-written code,
because none of them survive `juliac --trim=safe`:

  * Turing/Optimization reach SciMLBase -> FunctionWrappers, which is
    documented as untrimmable.
  * ForwardDiff builds a binary that passes the trim verifier and then throws
    a MethodError on the Dual-argument objective at runtime.

Deliberately restricted in scope: two year processes (rw1, the default, and
iid), one optimiser (L-BFGS) and two samplers (NUTS, and fixed-length HMC with
a pre-tuned metric). Gradients come from the exact analytic reverse sweep in
grad_analytic.jl, which covers every model variant the CLI exposes; the
hand-rolled forward-mode dual in dual.jl remains as the independent reference
the test suite differentiates against.
"""
module Test2InfCLI

include("dual.jl")
include("model.jl")
include("grad_analytic.jl")
include("optimize.jl")
include("rng.jl")
include("nuts.jl")
include("hmc.jl")
include("pretuned.jl")
include("io.jl")

# Forward-mode chunk width. No longer on the inference path -- the analytic
# gradient handles every variant -- but kept because the forward mode is the
# independent check the tests rely on, and because it must stay compiled to
# stay correct. Must be a compile-time constant so `ntuple` unrolls; note
# `ntuple` must be called as `ntuple(f, Val(N))`, since the `ntuple(f, N)` form
# stops unrolling above 8 and costs ~400x.
const CHUNK = 13

const USAGE = """
test2inf - badger diagnostic HMM inference

USAGE:
  test2inf --data <csv> [options]

REQUIRED:
  --data <path>        Input CSV: time,id,captured,test1..test6 (header row)

OPTIONS:
  --method <m>         map | nutsthenhmc | nuts | hmc      (default: map)
                       map         point estimate by L-BFGS, maximising the
                                   CONSTRAINED-space density, as Turing's
                                   maximum_a_posteriori does.
                       nutsthenhmc the default for full posterior sampling:
                                   NUTS adapts the step size and metric during
                                   warmup, then fixed-L HMC draws with them,
                                   skipping the per-draw tree building. L comes
                                   from NUTS's mean tree depth unless --hmc-L
                                   overrides it. Needs no pre-tuned constants.
                       nuts        pure NUTS throughout; the safest and the one
                                   to compare against when a sampler looks off.
                       hmc         fixed-L HMC with a compiled-in DENSE metric;
                                   fastest by far, but needs --model and is
                                   only valid for the data it was tuned on.
  --out <dir>          Output directory                    (default: .)
  --seasons <int>      Timesteps per year                  (default: 4)
  --tests <list>       Comma-separated assay indices 1-6   (default: all)
  --year-process <p>   rw1c | rw1 | iid | rw2 | none | shrunk (default: rw1c)
                       Different models, not tuning options; each carries its
                       own sigma_g prior (rw1/rw1c 0.05, rw2 0.01, shrunk 0.10,
                       iid/none 0.30). rw1c is rw1 constrained to sum to zero,
                       which identifies the LEVEL of the year series -- plain
                       rw1 leaves it free, so years with no test data drift.
                       ar1 is not supported here: it adds a rho parameter that
                       changes the vector layout.
  --no-year-weight     Drop the data-adaptive increment prior (on by default).
                       That prior shrinks each year-to-year step in proportion
                       to how little test data informs it, so an untested year
                       is pinned to its neighbour instead of drifting, while
                       well-tested years are left to the likelihood.
  --infer-sesp         Estimate Se/Sp instead of fixing them at Table 1
                       (not identifiable with --tests 3 alone)
  --accuracy-prior <f> logitnormal | beta                (default: logitnormal)
                       Only used with --infer-sesp. Beta's shape floor at 0.5
                       distorts the requested moments (culture Sp: asked for
                       mean 0.999, get Beta(3.26,0.5) with mean 0.867) and makes
                       the density diverge at the boundary, so MAP has no
                       interior maximum and is refused. logitnormal takes each
                       literature value as the prior MEDIAN and vanishes at both
                       boundaries.
  --pin-sp <k>=<v>     Pin assay k's specificity to v. Repeatable. `--pin-sp
                       3=1.0` states that a culture positive is never a false
                       positive -- the carry-forward assumption expressed inside
                       the model rather than imported as a data transformation.
                       With --infer-sesp the pinned assay stops being a
                       parameter: it loses its slot and its prior, so the
                       parameter count drops. With Se/Sp fixed it overrides that
                       assay's Table 1 value (culture 0.999 -> 1.0), which is a
                       different model, not a no-op.
  --no-penalty         Drop the Se+Sp>1 identifiability penalty
  --repeat <mode>      stack | pool | last                 (default: stack)
  --draws <int>        Post-warmup draws                   (default: 1000)
  --warmup <int>       Warmup iterations                   (default: 1000)
                       Under nutsthenhmc this is the NUTS adaptation phase.
  --accept <float>     Target acceptance for adaptation    (default: 0.8)
  --seed <int>         RNG seed                            (default: 1)
  --maxiter <int>      L-BFGS iteration cap                (default: 1000)
  --model <name>       Pre-tuned HMC step size, trajectory length and dense
                       mass matrix. One of: all_fixed, all_inferred,
                       culture_only_fixed, no_dpp_fixed, no_dpp_inferred.
                       These were tuned on rw1 with Beta priors and NO increment
                       prior, so they do not match the current defaults: use
                       them only with --year-process rw1 --no-year-weight
                       (and --accuracy-prior beta when inferring). The binary
                       refuses the mismatched combinations rather than sampling
                       badly in silence. Retune with tune/tune_hmc.jl, or use
                       --metric.
  --hmc-eps <float>    Override the pre-tuned step size
  --hmc-L <int>        Override the trajectory length. Under nutsthenhmc this
                       replaces the length inferred from NUTS's tree depth.
  --metric <path>      Inverse mass matrix from a headerless CSV, overriding
                       --model. n rows of n values, or one row/column of n
                       values for a diagonal metric.
  --write-metric <path>
                       Write the metric used, for replay with --metric. Works
                       under --method nuts, which is how to tune for own data.
  --traj-draws <int>   Per-draw prevalence and infection times, from an evenly
                       spaced subsample of this many draws (default: 0, off).
                       nuts/hmc only. Costs one forward-backward pass per
                       badger per draw kept.
  --help               Show this message

OUTPUTS (in --out):
  prevalence.csv       Prevalence by timestep
  parameters.csv       Point estimates (or posterior means) per parameter
  p_infected.csv       P(infected) per badger at each capture
  year_effects.csv     Year effect and implied annual hazard
  draws.csv            Posterior draws, one column per parameter (nuts/hmc)
  prevalence_draws.csv Prevalence per timestep per draw   (--traj-draws)
  trajectories.csv     Infection time per badger per draw (--traj-draws)
"""

# --- argument parsing ------------------------------------------------------
struct Options
    data::String
    method::String
    out::String
    seasons::Int
    tests::Vector{Bool}
    infer_sesp::Bool
    penalty::Bool
    repeat_captures::Symbol
    draws::Int
    warmup::Int
    accept::Float64
    seed::Int
    maxiter::Int
    year_process::Int
    traj_draws::Int      # 0 = do not write per-draw trajectory outputs
    metric::String       # path to an inverse mass matrix CSV, "" if unset
    write_metric::String # path to write the metric used, "" if unset
    model::String        # pre-tuned HMC parameter set, "" if unset
    hmc_eps::Float64     # <=0 means "use the pre-tuned value"
    hmc_L::Int           # <=0 means "use the pre-tuned value"
    accuracy_prior::Int  # ACC_BETA or ACC_LOGITNORMAL
    year_weight::Bool    # data-adaptive year-increment prior
    sp_pin::NTuple{N_TESTS,Float64}   # NaN = infer, otherwise pin to this value
end

function parse_int(s::AbstractString, name::AbstractString)
    v = tryparse(Int, s)
    v === nothing && error("--$name expects an integer, got \"$s\"")
    return v
end

function parse_float(s::AbstractString, name::AbstractString)
    v = tryparse(Float64, s)
    v === nothing && error("--$name expects a number, got \"$s\"")
    return v
end

function parse_args(args::Vector{String})
    data = ""
    method = "map"
    out = "."
    seasons = 4
    tests = fill(true, N_TESTS)
    # Fixed Se/Sp is the default model: it is the main comparator in the
    # current analysis, and inferring accuracy is the assumption check rather
    # than the headline fit.
    infer_sesp = false
    # rw1c + the data-adaptive increment prior is the current year model. Plain
    # rw1 left the level unidentified, which put 1976-77 -- 54 tested captures
    # and zero positives -- at the two highest hazards in the series.
    year_process = PROC_RW1C
    year_weight = true
    # Logit-normal accuracy priors. Beta's shape floor at 0.5 distorts the
    # requested moments and makes MAP unbounded at the boundary.
    accuracy_prior = ACC_LOGITNORMAL
    sp_pin = NO_SP_PIN
    penalty = true
    repeat_captures = :stack
    draws = 1000
    warmup = 1000
    accept = 0.8
    seed = 1
    maxiter = 1000
    traj_draws = 0
    metric = ""
    write_metric_path = ""
    model = ""
    hmc_eps = -1.0
    hmc_L = -1

    i = 1
    n = length(args)
    while i <= n
        a = args[i]
        if a == "--help" || a == "-h"
            return nothing
        elseif a == "--data"
            i += 1; i <= n || error("--data needs a value"); data = args[i]
        elseif a == "--method"
            i += 1; i <= n || error("--method needs a value"); method = args[i]
        elseif a == "--out"
            i += 1; i <= n || error("--out needs a value"); out = args[i]
        elseif a == "--seasons"
            i += 1; i <= n || error("--seasons needs a value")
            seasons = parse_int(args[i], "seasons")
        elseif a == "--tests"
            i += 1; i <= n || error("--tests needs a value")
            fill!(tests, false)
            for part in split_line(args[i])
                k = parse_int(strip(part), "tests")
                (1 <= k <= N_TESTS) || error("--tests index $k out of range 1..$N_TESTS")
                tests[k] = true
            end
        elseif a == "--year-process"
            i += 1; i <= n || error("--year-process needs a value")
            yp = args[i]
            if yp == "rw1"
                year_process = PROC_RW1
            elseif yp == "rw1c"
                year_process = PROC_RW1C
            elseif yp == "iid"
                year_process = PROC_IID
            elseif yp == "rw2"
                year_process = PROC_RW2
            elseif yp == "none"
                year_process = PROC_NONE
            elseif yp == "shrunk"
                year_process = PROC_SHRUNK
            elseif yp == "ar1"
                error("--year-process ar1 is not supported: it adds a rho " *
                      "parameter, which changes the parameter vector layout. " *
                      "Use test2infeR's JIT engine for ar1.")
            else
                error("--year-process must be one of: rw1c, rw1, iid, rw2, none, shrunk")
            end
        elseif a == "--infer-sesp"
            infer_sesp = true
        elseif a == "--accuracy-prior"
            i += 1; i <= n || error("--accuracy-prior needs a value")
            ap = args[i]
            if ap == "logitnormal"
                accuracy_prior = ACC_LOGITNORMAL
            elseif ap == "beta"
                accuracy_prior = ACC_BETA
            else
                error("--accuracy-prior must be logitnormal or beta")
            end
        elseif a == "--no-year-weight"
            year_weight = false
        elseif a == "--pin-sp"
            # "<index>=<value>", repeatable. Pins one test's specificity to a
            # constant: culture Sp=1 is the carry-forward assumption expressed
            # inside the model rather than imported as a data transformation.
            i += 1; i <= n || error("--pin-sp needs a value")
            spec = args[i]
            eq = findfirst('=', spec)
            eq === nothing && error("--pin-sp expects <index>=<value>, got \"$spec\"")
            pk = parse_int(spec[1:(eq - 1)], "pin-sp index")
            (1 <= pk <= N_TESTS) || error("--pin-sp index $pk out of range 1..$N_TESTS")
            pv = parse_float(spec[(eq + 1):end], "pin-sp value")
            (0.0 < pv <= 1.0) || error("--pin-sp value must be in (0, 1], got $pv")
            # Built by explicit assignment, not `ntuple` with a closure: a
            # closure capturing `pk`/`pv` boxes them (Core.Box), which erases
            # the type and breaks trimming. It also boxed the `--tests` loop's
            # own `k` by association, three verifier errors away from here.
            sp_new = zeros(Float64, N_TESTS)
            for j in 1:N_TESTS
                sp_new[j] = j == pk ? pv : sp_pin[j]
            end
            sp_pin = NTuple{N_TESTS,Float64}(sp_new)
        elseif a == "--no-penalty"
            penalty = false
        elseif a == "--repeat"
            i += 1; i <= n || error("--repeat needs a value")
            r = args[i]
            r in ("stack", "pool", "last") || error("--repeat must be stack|pool|last")
            repeat_captures = Symbol(r)
        elseif a == "--draws"
            i += 1; i <= n || error("--draws needs a value")
            draws = parse_int(args[i], "draws")
        elseif a == "--warmup"
            i += 1; i <= n || error("--warmup needs a value")
            warmup = parse_int(args[i], "warmup")
        elseif a == "--accept"
            i += 1; i <= n || error("--accept needs a value")
            accept = parse_float(args[i], "accept")
        elseif a == "--seed"
            i += 1; i <= n || error("--seed needs a value")
            seed = parse_int(args[i], "seed")
        elseif a == "--maxiter"
            i += 1; i <= n || error("--maxiter needs a value")
            maxiter = parse_int(args[i], "maxiter")
        elseif a == "--traj-draws"
            i += 1; i <= n || error("--traj-draws needs a value")
            traj_draws = parse_int(args[i], "traj-draws")
        elseif a == "--metric"
            i += 1; i <= n || error("--metric needs a value"); metric = args[i]
        elseif a == "--write-metric"
            i += 1; i <= n || error("--write-metric needs a value")
            write_metric_path = args[i]
        elseif a == "--model"
            i += 1; i <= n || error("--model needs a value"); model = args[i]
        elseif a == "--hmc-eps"
            i += 1; i <= n || error("--hmc-eps needs a value")
            hmc_eps = parse_float(args[i], "hmc-eps")
        elseif a == "--hmc-L"
            i += 1; i <= n || error("--hmc-L needs a value")
            hmc_L = parse_int(args[i], "hmc-L")
        else
            error("unknown argument: $a")
        end
        i += 1
    end

    isempty(data) && error("--data is required")
    method in ("map", "nuts", "hmc", "nutsthenhmc") ||
        error("--method must be map, nutsthenhmc, nuts or hmc")
    any(tests) || error("--tests selected no assays")
    if method == "hmc" && isempty(model) && (hmc_eps <= 0.0 || hmc_L <= 0)
        error("--method hmc needs --model (or both --hmc-eps and --hmc-L)")
    end
    return Options(data, method, out, seasons, tests, infer_sesp, penalty,
                   repeat_captures, draws, warmup, accept, seed, maxiter,
                   year_process, traj_draws, metric, write_metric_path,
                   model, hmc_eps, hmc_L, accuracy_prior, year_weight, sp_pin)
end

"Name of a year process, for reporting."
function year_process_name(proc::Int)
    proc == PROC_RW1    && return "rw1"
    proc == PROC_RW1C   && return "rw1c"
    proc == PROC_RW2    && return "rw2"
    proc == PROC_NONE   && return "none"
    proc == PROC_SHRUNK && return "shrunk"
    return "iid"
end

# --- initial values --------------------------------------------------------
function initial_theta(layout::ParamLayout)
    theta = zeros(layout.n)
    for s in 1:layout.S
        theta[layout.i_alpha + s - 1] = HAZARD_PRIOR_MEAN
    end
    theta[layout.i_logsigma] = log(0.1)
    theta[layout.i_pi1_0] = -1.7
    theta[layout.i_pi1_mult] = 1.0
    if layout.infer_sesp
        for k in 1:N_TESTS
            # Start at the Table 1 values on the logit scale.
            theta[layout.i_se + k - 1] = log(SE_FIXED_DEFAULT[k] / (1 - SE_FIXED_DEFAULT[k]))
            # Specificities are addressed through the slot map, not a
            # contiguous block: a pinned test has no slot, and every later test
            # shifts down by one.
            slot = layout.sp_slot[k]
            slot == 0 && continue
            theta[slot] = log(SP_FIXED_DEFAULT[k] / (1 - SP_FIXED_DEFAULT[k]))
        end
    end
    return theta
end

function param_names(layout::ParamLayout)
    names = String[]
    for s in 1:layout.S
        push!(names, "alpha[$s]")
    end
    push!(names, "sigma_g")
    for y in 1:layout.n_years
        push!(names, "gamma_raw[$y]")
    end
    push!(names, "pi1_0")
    push!(names, "pi1_mult")
    if layout.infer_sesp
        for k in 1:N_TESTS
            push!(names, "Se[$k]")
        end
        # Only the UNPINNED specificities occupy slots. Emitting all six here
        # would shift every name past the pinned one onto the wrong value.
        for k in 1:N_TESTS
            layout.sp_slot[k] == 0 && continue
            push!(names, "Sp[$k]")
        end
    end
    return names
end

# --- prevalence ------------------------------------------------------------
"""
P(infected) per badger per capture, and the prevalence trajectory over the
grid of observed timesteps.
"""
function decode(data::HMMData, P::Params)
    n = length(data.times)
    p_inf = Vector{Float64}(undef, n)
    start = 1
    for i in 1:data.n_ind
        e = data.seq_ends[i]
        gam = forward_backward_seq(data, start, e,
                                   P.pi1_vec[data.entry_year[i]], P, data.S)
        for (j, gj) in enumerate(gam)
            p_inf[start + j - 1] = gj
        end
        start = e + 1
    end
    return p_inf
end

"""
    draw_outputs(opts, data, layout, draws)

Per-draw prevalence and infection times, written only when `--traj-draws N > 0`.

`03_trajectories.R` in the sql-e2e bundles needs two things the posterior-mean
decode cannot give: prevalence CIs (which need the prevalence of EACH draw) and
per-badger infection times per draw. Both come from a forward-backward pass per
badger per draw, which is why the R pipeline's own comment says trajectory
rebuilding, not sampling, dominates its runtime. Doing it here instead is the
same arithmetic in compiled code over a thinned subset.

Thinning is evenly spaced rather than random: it keeps the posterior spread with
a deterministic, reproducible subsample.

Writes:
  prevalence_draws.csv  time, then one column per draw (capture-weighted)
  trajectories.csv      id, draw, infection_time  (0 = never infected in record)

Infection time is the first timestep where P(infected) >= 0.5, matching the
package's `infection_times_from_draws`. The state is absorbing, so a trajectory
is fully described by when it crossed.
"""
function draw_outputs(opts::Options, data::HMMData, layout::ParamLayout,
                      draws::Matrix{Float64}, n_traj::Int)
    ndraw = size(draws, 2)
    n_traj <= 0 && return nothing
    keep = min(n_traj, ndraw)
    # Evenly spaced indices, first and last included.
    idx = keep == 1 ? [ndraw] :
          [1 + round(Int, (k - 1) * (ndraw - 1) / (keep - 1)) for k in 1:keep]

    tset = sort(unique(data.times))
    tpos = Dict{Int,Int}()
    for (i, t) in enumerate(tset)
        tpos[t] = i
    end

    prev_cols = Vector{Vector{Float64}}(undef, keep)
    traj_id = Float64[]
    traj_draw = Float64[]
    traj_time = Float64[]

    d = size(draws, 1)
    theta_k = Vector{Float64}(undef, d)
    for (kk, k) in enumerate(idx)
        @inbounds for i in 1:d
            theta_k[i] = draws[i, k]
        end
        Pk = extract_params(theta_k, layout)
        pk = decode(data, Pk)

        # prevalence for this draw, over CAPTURED badgers (see prevalence_by_time)
        sums = zeros(length(tset))
        cnts = zeros(Int, length(tset))
        @inbounds for j in eachindex(data.times)
            data.captured[j] || continue
            ti = tpos[data.times[j]]
            sums[ti] += pk[j]
            cnts[ti] += 1
        end
        col = Vector{Float64}(undef, length(tset))
        @inbounds for i in eachindex(tset)
            col[i] = cnts[i] > 0 ? sums[i] / cnts[i] : NaN
        end
        prev_cols[kk] = col

        # infection time per badger for this draw
        start = 1
        @inbounds for i in 1:data.n_ind
            e = data.seq_ends[i]
            t_inf = 0
            for j in start:e
                if pk[j] >= 0.5
                    t_inf = data.times[j]
                    break
                end
            end
            push!(traj_id, Float64(data.ids[i]))
            push!(traj_draw, Float64(kk))
            push!(traj_time, Float64(t_inf))
            start = e + 1
        end
    end

    hdr = Vector{String}(undef, keep + 1)
    hdr[1] = "time"
    cols = Vector{Vector{Float64}}(undef, keep + 1)
    cols[1] = Float64.(tset)
    for kk in 1:keep
        hdr[kk + 1] = "draw" * string(kk)
        cols[kk + 1] = prev_cols[kk]
    end
    write_csv(joinpath(opts.out, "prevalence_draws.csv"), hdr, cols)
    write_csv(joinpath(opts.out, "trajectories.csv"),
              ["id", "draw", "infection_time"], [traj_id, traj_draw, traj_time])
    return nothing
end

"""
Mean P(infected) among badgers actually CAPTURED at each timestep, matching the
package's `prevalence_capture`.

The `captured` filter is load-bearing. Each badger is expanded onto the full
grid between its first and last capture, so most timesteps are filler where the
badger was present but not caught. Averaging over those too raises prevalence by
~0.026 on the real cohort -- filler steps sit later in a record, where infection
probability is higher -- and the effect is worst in low-data years (+0.055 where
n<100 against +0.005 where n>=300).
"""
function prevalence_by_time(data::HMMData, p_inf::Vector{Float64})
    tset = sort(unique(data.times))
    sums = Dict{Int,Float64}()
    counts = Dict{Int,Int}()
    for t in tset
        sums[t] = 0.0
        counts[t] = 0
    end
    for j in eachindex(data.times)
        data.captured[j] || continue
        t = data.times[j]
        sums[t] += p_inf[j]
        counts[t] += 1
    end
    prev = Float64[]
    tot = Float64[]
    for t in tset
        c = counts[t]
        push!(prev, c > 0 ? sums[t] / c : NaN)
        push!(tot, Float64(c))
    end
    return tset, prev, tot
end

# --- output ----------------------------------------------------------------
function write_outputs(opts::Options, data::HMMData, layout::ParamLayout,
                       P::Params, theta::Vector{Float64}, p_inf::Vector{Float64},
                       post_sd::Union{Nothing,Vector{Float64}})
    isdir(opts.out) || mkpath(opts.out)

    tset, prev, tot = prevalence_by_time(data, p_inf)
    write_csv(joinpath(opts.out, "prevalence.csv"),
              ["time", "prevalence", "n_captures"],
              [Float64.(tset), prev, tot])

    # Parameters on the natural scale.
    pnames = String[]
    pvals = Float64[]
    for s in 1:layout.S
        push!(pnames, "alpha[$s]"); push!(pvals, P.alpha[s])
    end
    push!(pnames, "sigma_g"); push!(pvals, P.sigma_g)
    for y in 1:layout.n_years
        push!(pnames, "gamma[$y]"); push!(pvals, P.gamma[y])
    end
    push!(pnames, "pi1_0"); push!(pvals, P.pi1_0)
    push!(pnames, "pi1_mult"); push!(pvals, P.pi1_mult)
    for k in 1:N_TESTS
        push!(pnames, "Se[$k]"); push!(pvals, P.Se[k])
    end
    for k in 1:N_TESTS
        push!(pnames, "Sp[$k]"); push!(pvals, P.Sp[k])
    end

    io = open(joinpath(opts.out, "parameters.csv"), "w")
    try
        write(io, post_sd === nothing ? "parameter,estimate\n" : "parameter,mean,sd\n")
        for i in eachindex(pnames)
            write(io, pnames[i]); write(io, ",")
            write(io, fmt(pvals[i]))
            if post_sd !== nothing
                write(io, ",")
                # sd is only available for the sampled (unconstrained) entries
                write(io, i <= length(post_sd) ? fmt(post_sd[i]) : "NA")
            end
            write(io, "\n")
        end
    finally
        close(io)
    end

    # P(infected) per badger per capture.
    idcol = Float64[]
    tcol = Float64[]
    start = 1
    for i in 1:data.n_ind
        e = data.seq_ends[i]
        for j in start:e
            push!(idcol, Float64(data.ids[i]))
            push!(tcol, Float64(data.times[j]))
        end
        start = e + 1
    end
    write_csv(joinpath(opts.out, "p_infected.csv"),
              ["id", "time", "p_infected"], [idcol, tcol, p_inf])

    yidx = Float64[]
    yeff = Float64[]
    yhaz = Float64[]
    for y in 1:layout.n_years
        push!(yidx, Float64(y))
        push!(yeff, P.gamma[y])
        push!(yhaz, annual_hazard(P.alpha, P.gamma, y, layout.S))
    end
    write_csv(joinpath(opts.out, "year_effects.csv"),
              ["year_index", "gamma", "annual_hazard"], [yidx, yeff, yhaz])
    return nothing
end

"""
    year_weights(mat, n_years, S) -> Vector{Float64}

Number of TESTED captures in each year, for the data-adaptive increment prior.

Counted over the RAW INPUT ROWS, not over gridded captures. The distinction is
not cosmetic: `build_data` expands each badger onto the full timestep grid and
stacks repeat captures within a timestep, so a year's gridded-capture count and
its row count differ wherever a badger was caught twice in one quarter. On the
real cohort that is 15,863 tested rows against 14,571 tested gridded captures.
Rows are what the reference engine counts, and the prior's whole purpose is to
express how much TEST EVIDENCE informs each year -- which is a property of the
results recorded, not of how they were later packed onto a grid.

Years are indexed exactly as the model indexes them, `year_of(t) = (t-1) div S
+ 1`. `ceil(t/S)` disagrees on every multiple of `S` and would misalign the
prior by one year -- silently, since both produce a plausible-looking vector.
"""
function year_weights(mat::Matrix{Float64}, test_mask::Vector{Bool},
                      n_years::Int, S::Int, enabled::Bool = true)
    enabled || return EMPTY_YW
    yw = zeros(Float64, n_years)
    test_cols = (size(mat, 2) - 5):size(mat, 2)
    @inbounds for r in axes(mat, 1)
        # A capture with no test results tells the year effect nothing, so it
        # must not count toward how well that year is informed. Masked-out
        # assays are not results either: under `--tests 3` a row carrying only
        # an ELISA is untested as far as this fit is concerned.
        tested = false
        for (j, c) in enumerate(test_cols)
            test_mask[j] || continue
            isnan(mat[r, c]) || (tested = true)
        end
        tested || continue
        y = year_of(Int(mat[r, 1]), S)
        y <= n_years && (yw[y] += 1.0)
    end
    return yw
end

"""
    grad!(g, logp, x, data, layout, se_pr, sp_pr, penalty, year_weight)

Gradient of the log posterior, by the exact reverse sweep in grad_analytic.jl.
One pass gets every partial at ~1.5-1.8x the cost of a single logp, against
forward mode's ~4.2x and a general reverse-mode tape's ~2.8x at this parameter
count -- and unlike forward mode that ratio does not grow with the parameter
count.

Covers every variant the CLI exposes: any test mask, Se/Sp fixed or inferred,
either accuracy prior family, any year process, with or without the increment
prior, and with any subset of specificities pinned. `logp` is retained only as
the fallback the tests differentiate against; it is not on the sampling path.
"""
@inline function grad!(g::Vector{Float64}, logp::L, x::Vector{Float64},
                       data::HMMData, layout::ParamLayout,
                       se_pr::AccuracyPriors, sp_pr::AccuracyPriors,
                       penalty::Bool,
                       year_weight::Vector{Float64} = EMPTY_YW,
                       jacobian::Bool = true) where {L}
    return logposterior_grad!(g, x, data, layout, se_pr, sp_pr, penalty,
                              year_weight, jacobian)
end

# --- method drivers --------------------------------------------------------
# Split into separate functions deliberately: assigning the result in both arms
# of an if/else boxes it (Core.Box), which erases the type and produces dozens
# of unresolved-call trim errors.

function run_map(opts::Options, data::HMMData, layout::ParamLayout, negf!::F) where {F}
    theta0 = initial_theta(layout)
    res = lbfgs(negf!, theta0; maxiter = opts.maxiter)
    println(Core.stdout, "log posterior: ", res.logp,
            "  iterations: ", res.iterations,
            "  converged: ", res.converged,
            "  |grad|: ", res.grad_norm)
    P = extract_params(res.theta, layout)
    p_inf = decode(data, P)
    write_outputs(opts, data, layout, P, res.theta, p_inf, nothing)
    return nothing
end

function run_nuts(opts::Options, data::HMMData, layout::ParamLayout,
                  logp::L, negf!::F,
                  se_pr::AccuracyPriors, sp_pr::AccuracyPriors,
                  pen::Bool,
                  yw::Vector{Float64} = EMPTY_YW) where {L,F}
    # WITH the Jacobian: this is the unconstrained density the sampler must
    # explore. `negf!` below deliberately omits it, but that is only used to
    # find a starting point, never to sample.
    gradf! = let logp = logp, data = data, layout = layout,
                 se_pr = se_pr, sp_pr = sp_pr, pen = pen, yw = yw
        (g, x) -> grad!(g, logp, x, data, layout, se_pr, sp_pr, pen, yw, true)
    end

    # Start from the MAP: a short optimise first makes warmup far cheaper.
    theta0 = initial_theta(layout)
    opt = lbfgs(negf!, theta0; maxiter = 200)
    println(Core.stdout, "MAP init: log posterior ", opt.logp)

    res = nuts(gradf!, opt.theta, opts.draws, opts.warmup, opts.accept, opts.seed)
    # Warmup and sampling reported separately: warmup divergences are expected
    # while dual averaging walks the step size down, and only the post-warmup
    # count bears on whether the draws are trustworthy.
    println(Core.stdout, "step size: ", res.step_size,
            "  mean tree depth: ", res.mean_depth)
    println(Core.stdout, "warmup:   divergences: ", res.n_divergent_warmup,
            "  mean accept: ", res.accept_rate_warmup)
    println(Core.stdout, "sampling: divergences: ", res.n_divergent,
            "  mean accept: ", res.accept_rate)
    if res.n_divergent > 0
        println(Core.stdout,
                "WARNING: ", res.n_divergent,
                " post-warmup divergences -- draws may be biased.")
    end

    # NUTS adapts a diagonal metric; exporting it is how a user tunes for their
    # OWN data with no Julia and no rebuild: run nuts once with --write-metric,
    # then pass that file to --method hmc --metric on subsequent runs.
    if !isempty(opts.write_metric)
        d = length(res.inv_mass)
        m = zeros(d, d)
        for i in 1:d
            m[i, i] = res.inv_mass[i]
        end
        write_metric(opts.write_metric, m)
        println(Core.stdout, "wrote metric to ", opts.write_metric,
                "  (step size ", res.step_size, ", mean depth ", res.mean_depth, ")")
    end

    finish_draws(opts, data, layout, res.draws)
    return nothing
end

function run_nutsthenhmc(opts::Options, data::HMMData, layout::ParamLayout,
                         logp::L, negf!::F,
                         se_pr::AccuracyPriors, sp_pr::AccuracyPriors,
                         pen::Bool,
                         yw::Vector{Float64} = EMPTY_YW) where {L,F}
    # WITH the Jacobian: both phases sample the unconstrained density.
    gradf! = let logp = logp, data = data, layout = layout,
                 se_pr = se_pr, sp_pr = sp_pr, pen = pen, yw = yw
        (g, x) -> grad!(g, logp, x, data, layout, se_pr, sp_pr, pen, yw, true)
    end

    theta0 = initial_theta(layout)
    opt = lbfgs(negf!, theta0; maxiter = 200)
    println(Core.stdout, "MAP init: log posterior ", opt.logp)

    nth = nuts_then_hmc(gradf!, opt.theta, opts.draws, opts.warmup,
                        opts.accept, opts.seed;
                        n_leapfrog = opts.hmc_L > 0 ? opts.hmc_L : 0)
    warm = nth.warmup
    res = nth.result

    println(Core.stdout, "warmup (NUTS): step size ", warm.step_size,
            "  mean tree depth ", warm.mean_depth,
            "  divergences ", warm.n_divergent_warmup,
            "  mean accept ", warm.accept_rate_warmup)
    println(Core.stdout, "sampling (HMC): L ", nth.L,
            "  divergences ", res.n_divergent,
            "  mean accept ", res.accept_rate)
    if res.n_divergent > 0
        println(Core.stdout,
                "WARNING: ", res.n_divergent,
                " post-warmup divergences -- draws may be biased.")
    end
    # An acceptance rate far from the target means the inherited step size does
    # not suit a FIXED trajectory: NUTS can compensate by varying its tree
    # depth, and fixed-L HMC cannot. Worth saying, because the draws are still
    # produced either way.
    if res.accept_rate < 0.6 || res.accept_rate > 0.99
        println(Core.stdout,
                "WARNING: HMC acceptance ", res.accept_rate,
                " is far from the target ", opts.accept,
                " -- the step size NUTS adapted may not suit a fixed L. ",
                "Compare against --method nuts before trusting these draws.")
    end

    # The adapted metric is worth keeping: it is what turns a later run into the
    # cheap pre-tuned path without any rebuild.
    if !isempty(opts.write_metric)
        d = length(warm.inv_mass)
        m = zeros(d, d)
        for i in 1:d
            m[i, i] = warm.inv_mass[i]
        end
        write_metric(opts.write_metric, m)
        println(Core.stdout, "wrote metric to ", opts.write_metric,
                "  (step size ", warm.step_size, ", L ", nth.L, ")")
    end

    finish_draws(opts, data, layout, res.draws)
    return nothing
end

"""
Posterior summaries from a matrix of draws, shared by the NUTS and HMC paths:
per-parameter mean and sd, the posterior mean of P(infected), and draws.csv.
"""
function finish_draws(opts::Options, data::HMMData, layout::ParamLayout,
                      draws::Matrix{Float64})
    d = size(draws, 1)
    ndraw = size(draws, 2)
    means = Vector{Float64}(undef, d)
    sds = Vector{Float64}(undef, d)
    for i in 1:d
        s = 0.0
        for k in 1:ndraw
            s += draws[i, k]
        end
        m = s / ndraw
        means[i] = m
        v = 0.0
        for k in 1:ndraw
            v += (draws[i, k] - m)^2
        end
        sds[i] = ndraw > 1 ? sqrt(v / (ndraw - 1)) : 0.0
    end

    # Posterior mean of P(infected), averaged over draws.
    n_obs = length(data.times)
    p_acc = zeros(n_obs)
    theta_k = Vector{Float64}(undef, d)
    for k in 1:ndraw
        @inbounds for i in 1:d
            theta_k[i] = draws[i, k]
        end
        Pk = extract_params(theta_k, layout)
        pk = decode(data, Pk)
        @inbounds for j in 1:n_obs
            p_acc[j] += pk[j]
        end
    end
    @inbounds for j in 1:n_obs
        p_acc[j] /= ndraw
    end

    P = extract_params(means, layout)
    write_outputs(opts, data, layout, P, means, p_acc, sds)
    write_draws(opts, layout, draws)
    draw_outputs(opts, data, layout, draws, opts.traj_draws)
    return nothing
end

"""
Look up a pre-tuned parameter set by name. Written as an explicit loop over a
tuple rather than a Dict so it stays trim-clean.
"""
function find_pretuned(name::AbstractString)
    for pt in PRETUNED_ALL
        pt.name == name && return pt
    end
    error("unknown --model '" * String(name) * "'; expected one of " *
          "all_fixed, all_inferred, culture_only_fixed, no_dpp_fixed, no_dpp_inferred")
end

function run_hmc(opts::Options, data::HMMData, layout::ParamLayout,
                 logp::L, negf!::F,
                 se_pr::AccuracyPriors, sp_pr::AccuracyPriors,
                 pen::Bool,
                 yw::Vector{Float64} = EMPTY_YW) where {L,F}
    # WITH the Jacobian, as for NUTS: the sampler explores the unconstrained
    # density, and omitting it would target a different posterior.
    gradf! = let logp = logp, data = data, layout = layout,
                 se_pr = se_pr, sp_pr = sp_pr, pen = pen, yw = yw
        (g, x) -> grad!(g, logp, x, data, layout, se_pr, sp_pr, pen, yw, true)
    end

    # Metric precedence, highest first:
    #   --metric <path>   an explicit matrix, for data this binary was not
    #                     tuned on -- the escape hatch from compiled-in values
    #   --model <name>    the compiled-in tuned metric, if its size matches
    #   otherwise         identity
    #
    # The compiled-in sets exist so the common case needs no extra files, NOT
    # to make them the only option: a metric tuned for other data must not
    # require rebuilding and redistributing a 23 MB bundle.
    inv_mass = zeros(layout.n, layout.n)
    for i in 1:layout.n
        inv_mass[i, i] = 1.0
    end
    eps = opts.hmc_eps
    L_traj = opts.hmc_L
    if !isempty(opts.metric)
        m = read_metric(opts.metric)
        size(m, 1) == layout.n ||
            error("--metric " * opts.metric * " is " * string(size(m, 1)) *
                  "x" * string(size(m, 2)) * " but this run has " *
                  string(layout.n) * " parameters")
        inv_mass = m
        println(Core.stdout, "metric: ", opts.metric)
        # An explicit metric says nothing about step size or trajectory length,
        # so those still have to come from somewhere.
        if eps <= 0.0 || L_traj <= 0
            if !isempty(opts.model)
                pt = find_pretuned(opts.model)
                eps <= 0.0 && (eps = pt.eps)
                L_traj <= 0 && (L_traj = pt.L)
            end
        end
    elseif !isempty(opts.model)
        pt = find_pretuned(opts.model)
        # A pre-tuned set is valid only for the process it was tuned under:
        # rw1's cumsum funnel needs a ~6x smaller step and a ~6x longer
        # trajectory than iid (0.048/L=45 vs 0.28/L=7 on the real cohort), so
        # applying the wrong set samples badly rather than failing.
        pt.year_process == layout.year_process ||
            error("--model " * pt.name * " was tuned for year process " *
                  year_process_name(pt.year_process) *
                  " but this run uses " *
                  year_process_name(layout.year_process) *
                  "; retune with tune/tune_hmc.jl or pass --year-process to match")
        # The stored sets predate the increment prior and the logit-normal
        # accuracy priors, and both reshape the posterior the metric describes:
        # the increment prior tightens exactly the year-to-year directions the
        # dense metric was built to absorb. A mis-matched metric does not fail,
        # it samples badly and quietly, so refuse instead of warning.
        opts.year_weight &&
            error("--model " * pt.name * " was tuned WITHOUT the data-adaptive " *
                  "increment prior, which changes the very year-to-year " *
                  "correlations its dense metric encodes. Pass --no-year-weight " *
                  "to match the tuning, or retune with tune/tune_hmc.jl.")
        opts.infer_sesp && opts.accuracy_prior != ACC_BETA &&
            error("--model " * pt.name * " was tuned with Beta accuracy priors " *
                  "but this run uses logit-normal ones, which changes the " *
                  "geometry in the Se/Sp directions. Pass --accuracy-prior beta " *
                  "to match the tuning, or retune with tune/tune_hmc.jl.")
        any(!isnan, opts.sp_pin) &&
            error("--model " * pt.name * " was tuned with every specificity " *
                  "free; pinning one removes a parameter and invalidates the " *
                  "stored metric. Retune with tune/tune_hmc.jl.")
        if pt.n == layout.n
            inv_mass = pt.inv_mass
            opts.hmc_eps > 0.0 || (eps = pt.eps)
            opts.hmc_L > 0 || (L_traj = pt.L)
            println(Core.stdout, "pre-tuned model: ", pt.name)
        elseif opts.hmc_eps > 0.0 && opts.hmc_L > 0
            # Explicit eps/L given, so the run is still well defined; warn that
            # the tuned metric is unusable rather than silently applying it.
            println(Core.stdout, "WARNING: --model ", pt.name, " is tuned for ",
                    pt.n, " parameters but this run has ", layout.n,
                    "; using an identity metric with the supplied --hmc-eps/--hmc-L.")
        else
            error("--model " * pt.name * " was tuned for " * string(pt.n) *
                  " parameters but this run has " * string(layout.n) *
                  "; check --tests and --infer-sesp match the named model, " *
                  "or pass --hmc-eps and --hmc-L explicitly")
        end
    end
    eps > 0.0 || error("--method hmc needs a step size (--model or --hmc-eps)")
    L_traj > 0 || error("--method hmc needs a trajectory length (--model or --hmc-L)")

    theta0 = initial_theta(layout)
    opt = lbfgs(negf!, theta0; maxiter = 200)
    println(Core.stdout, "MAP init: log posterior ", opt.logp)
    println(Core.stdout, "step size: ", eps, "  L: ", L_traj)

    if !isempty(opts.write_metric)
        write_metric(opts.write_metric, inv_mass)
        println(Core.stdout, "wrote metric to ", opts.write_metric)
    end

    res = hmc(gradf!, opt.theta, inv_mass, eps, L_traj,
              opts.draws, opts.warmup, opts.seed)
    println(Core.stdout, "sampling: divergences: ", res.n_divergent,
            "  mean accept: ", res.accept_rate)
    if res.n_divergent > 0
        println(Core.stdout, "WARNING: ", res.n_divergent,
                " divergences -- draws may be biased.")
    end

    finish_draws(opts, data, layout, res.draws)
    return nothing
end

function write_draws(opts::Options, layout::ParamLayout, draws::Matrix{Float64})
    isdir(opts.out) || mkpath(opts.out)
    names = param_names(layout)
    d = size(draws, 1)
    ndraw = size(draws, 2)
    io = open(joinpath(opts.out, "draws.csv"), "w")
    try
        for j in eachindex(names)
            j > 1 && write(io, ",")
            write(io, names[j])
        end
        write(io, "\n")
        for k in 1:ndraw
            for i in 1:d
                i > 1 && write(io, ",")
                write(io, fmt(draws[i, k]))
            end
            write(io, "\n")
        end
    finally
        close(io)
    end
    return nothing
end

# --- entrypoint ------------------------------------------------------------
function run(args::Vector{String})
    opts = parse_args(args)
    if opts === nothing
        print(Core.stdout, USAGE)
        return 0
    end

    mat = read_matrix(opts.data)
    data = build_data(mat, opts.tests, opts.repeat_captures, opts.seasons)
    layout = ParamLayout(data.S, data.n_years, opts.infer_sesp, opts.year_process,
                         opts.sp_pin)
    se_pr, sp_pr = default_accuracy_priors(opts.accuracy_prior)
    # Concretely typed either way: `year_weights` returns an empty vector when
    # the prior is off, so `yw` is always a Vector{Float64} and every call that
    # takes it stays statically resolvable.
    yw = year_weights(mat, opts.tests, data.n_years, data.S, opts.year_weight)

    # MAP on a prior whose density diverges at a probability boundary has no
    # interior maximum, so whatever the optimiser stops at is not a MAP
    # estimate. Refuse rather than report it. Pinned tests are excluded: they
    # are constants and carry no prior.
    if opts.method == "map" && opts.infer_sesp
        bad = false
        for k in 1:N_TESTS
            is_singular_at_boundary(se_pr[k]) && (bad = true)
            layout.sp_slot[k] != 0 && is_singular_at_boundary(sp_pr[k]) && (bad = true)
        end
        bad && error("MAP refused: a free Beta prior has a singular boundary " *
                     "density, so there is no interior maximum and an optimiser " *
                     "result is not a certified MAP. Use --accuracy-prior " *
                     "logitnormal, posterior sampling, or fixed Se/Sp.")
    end

    println(Core.stdout, "badgers: ", data.n_ind,
            "  captures: ", length(data.times),
            "  years: ", data.n_years,
            "  parameters: ", layout.n,
            "  year process: ",
            year_process_name(layout.year_process),
            opts.year_weight ? "+yw" : "")

    # MAP and sampling maximise DIFFERENT functions, and deliberately so.
    #
    # MAP asks "where is the density highest?", which has no
    # parameterisation-free answer: the mode in sigma is not the image of the
    # mode in log sigma, because the change of variables reweights by the
    # Jacobian. Turing's `maximum_a_posteriori` answers it on the CONSTRAINED
    # scale, so this does too -- otherwise the same data would give a different
    # sigma here than in the reference engine.
    #
    # Sampling asks "draw from the posterior", which IS
    # parameterisation-free. The sampler works unconstrained, and the Jacobian
    # is exactly what makes the unconstrained density represent the same
    # distribution. Dropping it there would sample a different posterior, so it
    # is not optional.
    #
    # Hence one objective, one flag, and both the primal and its gradient must
    # carry the same value of it -- a mismatch would have L-BFGS descending a
    # function it is not evaluating.
    #
    # Closures over the fixed data; concrete-typed so trim can trace them.
    logp = let data = data, layout = layout, se_pr = se_pr, sp_pr = sp_pr,
               pen = opts.penalty, yw = yw
        theta -> logposterior(theta, data, layout, se_pr, sp_pr, pen, yw, true)
    end

    # Negative log posterior + gradient, for the minimiser. No Jacobian: this
    # is only ever handed to L-BFGS, for MAP and for the sampler's MAP init.
    negf! = let logp = logp, data = data, layout = layout,
                se_pr = se_pr, sp_pr = sp_pr, pen = opts.penalty, yw = yw
        (g, x) -> begin
            fx = grad!(g, logp, x, data, layout, se_pr, sp_pr, pen, yw, false)
            @inbounds for i in eachindex(g)
                g[i] = -g[i]
            end
            return -fx
        end
    end

    theta0 = initial_theta(layout)

    # Separate calls, not a variable holding the chosen function: assigning in
    # multiple branches boxes the value (Core.Box) and breaks trimming.
    if opts.method == "map"
        run_map(opts, data, layout, negf!)
    elseif opts.method == "hmc"
        run_hmc(opts, data, layout, logp, negf!, se_pr, sp_pr, opts.penalty, yw)
    elseif opts.method == "nutsthenhmc"
        run_nutsthenhmc(opts, data, layout, logp, negf!, se_pr, sp_pr,
                        opts.penalty, yw)
    else
        run_nuts(opts, data, layout, logp, negf!, se_pr, sp_pr, opts.penalty, yw)
    end

    println(Core.stdout, "wrote outputs to ", opts.out)
    return 0
end

function @main(args::Vector{String})::Cint
    # No try/catch: `sprint(showerror, e)` is not statically resolvable under
    # trimming. Argument and data errors still abort with a Julia-level message.
    return Cint(run(args))
end

end # module
