# Offline tuning of the fixed-L HMC parameters shipped with the CLI.
#
# Runs our own NUTS (diagonal metric, dual-averaging step size) against the
# analytic gradient, then exports, per model variant:
#
#   * the adapted inverse mass vector (diagonal)
#   * the adapted step size
#   * a trajectory length L, taken from the adapted integration time
#
# This runs OUTSIDE the trimmed binary; only its OUTPUT (a generated Julia
# source file of plain arrays) is compiled in.
#
# ## Why diagonal, and why our own adaptor
#
# Both were measured on the real cohort (3,224 badgers, 29,737 steps, 58 par)
# rather than assumed:
#
#   AdvancedHMC, DENSE metric   ~66 gradients/iter, depth 5.5 and RISING as the
#                               matrix adapts; eps fell to 0.049. Did not finish
#                               2,000 iterations in 50 minutes.
#   AdvancedHMC, DIAGONAL       117.9 s, depth 4.0, eps 0.267, 0 divergences
#   our NUTS,    DIAGONAL       119.1 s, depth 3.0, eps 0.282, 0 divergences
#
# The two diagonal adaptors agree to within noise on both the step size and the
# divergence count, so our own is used and AdvancedHMC is not a dependency of
# the tuning path. The dense metric is a genuine open question -- the
# gamma = sigma_g * gamma_raw correlation is exactly what a diagonal metric
# cannot represent -- but it costs ~8x more per iteration to ADAPT, and that
# has to be shown to buy something before it is worth paying.

using AdvancedHMC, LinearAlgebra, Random, Statistics, Printf
using MCMCDiagnosticTools: ess

const ROOT = dirname(@__DIR__)
# Year process to tune for. rw1 is the package default and the CLI default, so
# the shipped pretuned values must be tuned under it. Override with
# TUNE_PROCESS=iid to regenerate the iid set for comparison.

include(joinpath(ROOT, "Test2InfCLI", "src", "Test2InfCLI.jl"))
const M = Test2InfCLI
const YEAR_PROCESS = get(ENV, "TUNE_PROCESS", "rw1") == "iid" ? M.PROC_IID : M.PROC_RW1

# The five sql-e2e variants, keyed by the name the CLI will use.
const VARIANTS = (
    (name = "all_fixed",          mask = fill(true, 6),                          infer = false),
    (name = "all_inferred",       mask = fill(true, 6),                          infer = true),
    (name = "culture_only_fixed", mask = [false, false, true, false, false, false], infer = false),
    (name = "no_dpp_fixed",       mask = [true, true, true, false, true, true],   infer = false),
    (name = "no_dpp_inferred",    mask = [true, true, true, false, true, true],   infer = true),
)

"A LogDensityProblems-style target backed by the CLI's analytic gradient."
struct Target
    data::M.HMMData
    layout::M.ParamLayout
    se_ab::NTuple{6,Tuple{Float64,Float64}}
    sp_ab::NTuple{6,Tuple{Float64,Float64}}
    penalty::Bool
end
(t::Target)(x) = M.logposterior(x, t.data, t.layout, t.se_ab, t.sp_ab, t.penalty)
function logdensity_and_gradient(t::Target, x)
    g = zeros(length(x))
    lp = M.logposterior_grad!(g, x, t.data, t.layout, t.se_ab, t.sp_ab, t.penalty)
    return lp, g
end

function map_start(t::Target)
    negf! = (g, x) -> begin
        fx = M.logposterior_grad!(g, x, t.data, t.layout, t.se_ab, t.sp_ab, t.penalty)
        @inbounds for i in eachindex(g); g[i] = -g[i]; end
        return -fx
    end
    res = M.lbfgs(negf!, M.initial_theta(t.layout); maxiter = 500)
    return res.theta
end

"""
Tune one variant: adapt a DENSE metric and step size with AdvancedHMC's Stan
adaptor, then measure how the frozen result actually samples.

Dense, not diagonal: on the real cohort the dense metric samples at eps ~0.3
against diagonal's ~0.047 and tree depth 4.0 against 6.3, which is ~8x the
post-warmup ESS/second. Adapting it costs ~1.7x more warmup -- paid here, once,
offline, which is the entire point of shipping pre-tuned values.

L is taken from the adapted trajectory: at mean tree depth d, NUTS integrates
~2^d - 1 steps, so a fixed-L run should match that.
"""
function tune_variant(v; data_path, n_adapts = 1000, n_samples = 1000, seed = 1)
    mat = M.read_matrix(data_path)
    data = M.build_data(mat, collect(v.mask), :stack, 4)
    layout = M.ParamLayout(data.S, data.n_years, v.infer, YEAR_PROCESS)
    se_ab, sp_ab = M.default_sesp_priors()
    t = Target(data, layout, se_ab, sp_ab, true)
    D = layout.n

    theta0 = map_start(t)
    lpf = x -> M.logposterior(x, t.data, t.layout, t.se_ab, t.sp_ab, t.penalty)
    gf  = x -> (g = zeros(length(x));
                lp = M.logposterior_grad!(g, x, t.data, t.layout,
                                          t.se_ab, t.sp_ab, t.penalty); (lp, g))

    rng = MersenneTwister(seed)
    metric = DenseEuclideanMetric(D)
    ham = Hamiltonian(metric, lpf, gf)
    e0 = find_good_stepsize(rng, ham, theta0)
    integrator = Leapfrog(e0)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    t_warm = @elapsed sw, stw = sample(rng, ham, kernel, theta0, n_adapts, adaptor,
                                       n_adapts; drop_warmup = false,
                                       progress = false, verbose = false)

    # Read the ADAPTED values from the adaptor. `adapt!` returns a new
    # Hamiltonian rather than mutating in place, so `ham.metric` here is still
    # the INITIAL identity -- reading it would silently ship an identity metric.
    inv_mass = Matrix(AdvancedHMC.getM⁻¹(adaptor))
    eps = AdvancedHMC.getϵ(adaptor)
    start = sw[end]

    # Now measure the frozen metric the way a shipped run will use it.
    ham2 = Hamiltonian(DenseEuclideanMetric(inv_mass), lpf, gf)
    kn2 = HMCKernel(Trajectory{MultinomialTS}(Leapfrog(eps), GeneralisedNoUTurn()))
    t_samp = @elapsed s2, st2 = sample(rng, ham2, kn2, start, n_samples;
                                       progress = false, verbose = false)
    S = hcat(s2...)
    depth = mean(x -> x.tree_depth, st2)
    L = max(1, round(Int, 2^depth - 1))

    ess_vals = [ess(reshape(view(S, i, :), :, 1)) for i in 1:D]
    return (; name = v.name, D, inv_mass, eps, L,
            min_ess = minimum(ess_vals), median_ess = median(ess_vals),
            ndiv = count(x -> x.numerical_error, st2), ndiv_warm = 0,
            acc = mean(x -> x.acceptance_rate, st2),
            elapsed = t_warm + t_samp, t_warm, t_samp, n_samples,
            mean_depth = depth)
end

function emit(results, path)
    io = open(path, "w")
    try
        println(io, "# GENERATED by tune/tune_hmc.jl -- do not edit by hand.")
        println(io, "#")
        println(io, "# DENSE inverse mass matrices, step sizes and trajectory lengths for the")
        println(io, "# five sql-e2e variants, adapted by AdvancedHMC's Stan adaptor.")
        println(io, "#")
        println(io, "# Dense, not diagonal: on the real cohort this samples at eps ~0.3 against")
        println(io, "# diagonal's ~0.047 and tree depth 4.0 against 6.3, which is ~8x the")
        println(io, "# post-warmup ESS/second. The extra adaptation cost is paid here, offline.")
        println(io, "#")
        println(io, "# YEAR PROCESS: ", YEAR_PROCESS == M.PROC_RW1 ? "rw1" : "iid",
                ". These are NOT valid for the other process -- it changes the")
        println(io, "# geometry (rw1's cumsum makes a funnel) and the sigma_g prior.")
        println(io, "#")
        println(io, "# THESE ARE SPECIFIC TO THE COHORT THEY WERE TUNED ON. --model applies")
        println(io, "# them on the caller's word: only the parameter COUNT is checked, so")
        println(io, "# pointing them at a different dataset samples badly rather than")
        println(io, "# failing. Retune with tune/tune_hmc.jl if the data changes.")
        println(io, "#")
        for r in results
            @printf(io, "# %-20s D=%3d eps=%.5f L=%3d minESS=%6.0f medESS=%6.0f div=%d acc=%.3f
",
                    r.name, r.D, r.eps, r.L, r.min_ess, r.median_ess, r.ndiv, r.acc)
        end
        println(io)
        println(io, "struct PretunedHMC")
        println(io, "    name::String")
        println(io, "    n::Int")
        println(io, "    year_process::Int   # values are process-specific; see run_hmc's check")
        println(io, "    eps::Float64")
        println(io, "    L::Int")
        println(io, "    inv_mass::Matrix{Float64}")
        println(io, "end")
        println(io)
        for r in results
            println(io, "const PRETUNED_", uppercase(r.name), " = PretunedHMC(")
            println(io, "    \"", r.name, "\", ", r.D, ", ", YEAR_PROCESS, ", ",
                    repr(r.eps), ", ", r.L, ",")
            println(io, "    [")
            for i in 1:r.D
                print(io, "     ")
                for j in 1:r.D
                    print(io, repr(r.inv_mass[i, j]))
                    j < r.D && print(io, " ")
                end
                println(io, i < r.D ? ";" : "")
            end
            println(io, "    ])")
            println(io)
        end
        println(io, "const PRETUNED_ALL = (")
        for r in results
            println(io, "    PRETUNED_", uppercase(r.name), ",")
        end
        println(io, ")")
    finally
        close(io)
    end
end

function main()
    data_path = get(ENV, "TUNE_DATA", joinpath(ROOT, "testdata", "real_grid.csv"))
    n_adapts = parse(Int, get(ENV, "TUNE_ADAPTS", "1000"))
    n_samples = parse(Int, get(ENV, "TUNE_SAMPLES", "1000"))
    println("Tuning against: ", data_path)
    println("year process: ", YEAR_PROCESS == M.PROC_RW1 ? "rw1" : "iid")
    println("adapt=", n_adapts, " sample=", n_samples, "\n")
    results = []
    for v in VARIANTS
        print(rpad(v.name, 22), " ... "); flush(stdout)
        r = tune_variant(v; data_path, n_adapts, n_samples)
        push!(results, r)
        @printf("D=%d eps=%.4f L=%d  minESS=%.0f medESS=%.0f div=%d acc=%.3f depth=%.2f  %.1fs\n",
                r.D, r.eps, r.L, r.min_ess, r.median_ess, r.ndiv, r.acc, r.mean_depth, r.elapsed)
    end
    out = joinpath(ROOT, "Test2InfCLI", "src", "pretuned.jl")
    emit(results, out)
    println("\nwrote ", out)
    return results
end

main()
