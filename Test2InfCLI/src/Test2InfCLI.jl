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
  --method <map|nuts|hmc>
                       Inference method                    (default: map)
                       hmc = fixed-L HMC with a dense mass matrix; requires
                       --model to select the pre-tuned parameters
  --out <dir>          Output directory                    (default: .)
  --seasons <int>      Timesteps per year                  (default: 4)
  --tests <list>       Comma-separated assay indices 1-6   (default: all)
  --year-process <p>   rw1 | iid | rw2 | none | shrunk      (default: rw1)
                       Different models, not tuning options; each carries its
                       own sigma_g prior (rw1 0.05, rw2 0.01, shrunk 0.10,
                       iid/none 0.30). ar1 is not supported here -- it adds a
                       rho parameter that changes the vector layout.
  --infer-sesp         Estimate Se/Sp instead of fixing them at Table 1
                       (not identifiable with --tests 3 alone)
  --no-penalty         Drop the Se+Sp>1 identifiability penalty
  --repeat <mode>      stack | pool | last                 (default: stack)
  --draws <int>        NUTS post-warmup draws              (default: 1000)
  --warmup <int>       NUTS warmup iterations              (default: 1000)
  --accept <float>     NUTS target acceptance              (default: 0.8)
  --seed <int>         RNG seed                            (default: 1)
  --maxiter <int>      L-BFGS iteration cap                (default: 1000)
  --model <name>       Pre-tuned HMC step size, trajectory length and dense
                       mass matrix. One of: all_fixed, all_inferred,
                       culture_only_fixed, no_dpp_fixed, no_dpp_inferred.
                       Tuned for the badger cohort; only the parameter count is
                       checked, so other data samples badly rather than
                       failing. Retune with tune/tune_hmc.jl, or use --metric.
  --hmc-eps <float>    Override the pre-tuned step size
  --hmc-L <int>        Override the pre-tuned trajectory length
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
    infer_sesp = false
    year_process = PROC_RW1
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
                error("--year-process must be one of: rw1, iid, rw2, none, shrunk")
            end
        elseif a == "--infer-sesp"
            infer_sesp = true
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
    method in ("map", "nuts", "hmc") || error("--method must be map, nuts or hmc")
    any(tests) || error("--tests selected no assays")
    if method == "hmc" && isempty(model) && (hmc_eps <= 0.0 || hmc_L <= 0)
        error("--method hmc needs --model (or both --hmc-eps and --hmc-L)")
    end
    return Options(data, method, out, seasons, tests, infer_sesp, penalty,
                   repeat_captures, draws, warmup, accept, seed, maxiter,
                   year_process, traj_draws, metric, write_metric_path,
                   model, hmc_eps, hmc_L)
end

"Name of a year process, for reporting."
function year_process_name(proc::Int)
    proc == PROC_RW1    && return "rw1"
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
            theta[layout.i_sp + k - 1] = log(SP_FIXED_DEFAULT[k] / (1 - SP_FIXED_DEFAULT[k]))
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
        for k in 1:N_TESTS
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
    grad!(g, logp, x, data, layout, se_ab, sp_ab, penalty)

Gradient of the log posterior, by the exact reverse sweep in grad_analytic.jl.
One pass gets every partial at ~2x the cost of a single logp, against forward
mode's ~4.1x at 13 parameters and ~7.9x at 25 -- and unlike forward mode that
ratio does not grow with the parameter count.

All five `sql e2e` variants are covered: any test mask, Se/Sp fixed or
inferred. `logp` is retained only as the fallback the tests differentiate
against; it is no longer on the sampling path.
"""
@inline function grad!(g::Vector{Float64}, logp::L, x::Vector{Float64},
                       data::HMMData, layout::ParamLayout,
                       se_ab::NTuple{N_TESTS,Tuple{Float64,Float64}},
                       sp_ab::NTuple{N_TESTS,Tuple{Float64,Float64}},
                       penalty::Bool) where {L}
    return logposterior_grad!(g, x, data, layout, se_ab, sp_ab, penalty)
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
                  se_ab::NTuple{N_TESTS,Tuple{Float64,Float64}},
                  sp_ab::NTuple{N_TESTS,Tuple{Float64,Float64}},
                  pen::Bool) where {L,F}
    gradf! = let logp = logp, data = data, layout = layout,
                 se_ab = se_ab, sp_ab = sp_ab, pen = pen
        (g, x) -> grad!(g, logp, x, data, layout, se_ab, sp_ab, pen)
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
                 se_ab::NTuple{N_TESTS,Tuple{Float64,Float64}},
                 sp_ab::NTuple{N_TESTS,Tuple{Float64,Float64}},
                 pen::Bool) where {L,F}
    gradf! = let logp = logp, data = data, layout = layout,
                 se_ab = se_ab, sp_ab = sp_ab, pen = pen
        (g, x) -> grad!(g, logp, x, data, layout, se_ab, sp_ab, pen)
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
                  (pt.year_process == PROC_RW1 ? "rw1" : "iid") *
                  " but this run uses " *
                  (layout.year_process == PROC_RW1 ? "rw1" : "iid") *
                  "; retune with tune/tune_hmc.jl or pass --year-process to match")
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
    layout = ParamLayout(data.S, data.n_years, opts.infer_sesp, opts.year_process)
    se_ab, sp_ab = default_sesp_priors()

    println(Core.stdout, "badgers: ", data.n_ind,
            "  captures: ", length(data.times),
            "  years: ", data.n_years,
            "  parameters: ", layout.n,
            "  year process: ",
            year_process_name(layout.year_process))

    # Closures over the fixed data; concrete-typed so trim can trace them.
    logp = let data = data, layout = layout, se_ab = se_ab, sp_ab = sp_ab,
               pen = opts.penalty
        theta -> logposterior(theta, data, layout, se_ab, sp_ab, pen)
    end

    # Negative log posterior + gradient, for the minimiser.
    negf! = let logp = logp, data = data, layout = layout,
                se_ab = se_ab, sp_ab = sp_ab, pen = opts.penalty
        (g, x) -> begin
            fx = grad!(g, logp, x, data, layout, se_ab, sp_ab, pen)
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
        run_hmc(opts, data, layout, logp, negf!, se_ab, sp_ab, opts.penalty)
    else
        run_nuts(opts, data, layout, logp, negf!, se_ab, sp_ab, opts.penalty)
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
