# The badger diagnostic HMM log-posterior, ported from
# test2infeR/inst/julia/Test2InfEngine/src/hmm_inference.jl without Turing.
#
# Scope, fixed deliberately (see the CLI's --help):
#   * year process: rw1 (the package default) or iid, chosen with --year-process
#   * Se/Sp: fixed at Table 1 values (model 3 of the handoff), or inferred
#   * sampler: MAP via LBFGS, or NUTS
#   * AD: the hand-rolled forward-mode dual in dual.jl

const N_TESTS = 6
const SE_FIXED_DEFAULT = (0.407, 0.407, 0.100, 0.650, 0.809, 0.492)
const SP_FIXED_DEFAULT = (0.943, 0.943, 0.999, 0.943, 0.936, 0.931)

const HAZARD_PRIOR_MEAN = -3.0
const HAZARD_PRIOR_SD   = 1.5
const PENALTY_WEIGHT    = 50.0
const PENALTY_SCALE     = 0.02
const LOG2              = 0.6931471805599453

# Data-adaptive year-increment prior (`year_weight`). The rw1 increments are
# unidentified wherever there is no test data, so the walk drifts freely there --
# which is what put 1976-77 (54 tested captures, zero positives) at the two
# highest hazards in the series. Tightening the prior everywhere would fight
# 16,255 likelihood terms in the years that DO have data. Instead tighten each
# link in proportion to how little data informs it:
#
#     gamma[y] - gamma[y-1] ~ Normal(0, sd_y)
#     sd_y = TAU * n_eff/(1 + n_eff) + FLOOR,  n_eff = min(w[y], w[y-1])
#
# n_eff is the SMALLER of the two adjacent counts: an increment is identified
# only if both endpoints have data. sd_y GROWS with n_eff, so a link between two
# well-tested years is essentially free and a link into an untested year is
# pinned to its neighbour. TAU is deliberately loose so the prior does not fight
# the data in the 45 well-tested years.
const YEAR_LINK_TAU   = 1.0
const YEAR_LINK_FLOOR = 0.05

"SD of the year-to-year increment prior, given the effective count for that link."
@inline year_link_sd(n_eff::Float64) =
    YEAR_LINK_TAU * (n_eff / (1.0 + n_eff)) + YEAR_LINK_FLOOR
# Year-effect process. These MUST track test2infeR's build_gamma/sigma_prior:
#   iid  gamma = sigma_g .* raw           sigma_g ~ Normal+(0, 0.30)
#   rw1  gamma = sigma_g .* cumsum(raw)   sigma_g ~ Normal+(0, 0.05)
# rw1 is the package default (`year_process = "rw1"` in hmm_inference), and
# neither sql-e2e script overrides it, so it is the default here too. The
# tighter sigma prior is part of the process, not an independent knob: rw1
# accumulates, so the same sigma would give far larger year effects.
# The five processes test2infeR supports that have a closed-form adjoint here.
# `ar1` is deliberately absent: it introduces a parameter (rho ~ Beta(5,2)),
# which changes the parameter-vector length and layout and would invalidate
# every pre-tuned metric. It errors clearly rather than being approximated.
const PROC_IID    = 0
const PROC_RW1    = 1
const PROC_RW2    = 2
const PROC_NONE   = 3
const PROC_SHRUNK = 4
const PROC_RW1C   = 5

const SIGMA_G_PRIOR_SD_IID    = 0.30
const SIGMA_G_PRIOR_SD_RW1    = 0.05
const SIGMA_G_PRIOR_SD_RW2    = 0.01
const SIGMA_G_PRIOR_SD_SHRUNK = 0.10
const SIGMA_G_PRIOR_SD_RW1C   = 0.05

# The sigma prior is part of the process, not an independent knob: a process
# that accumulates would give far larger year effects at the same sigma. These
# match test2infeR's `sigma_prior`, including that :none falls through to the
# 0.30 default there.
@inline function sigma_prior_sd(proc::Int)
    proc == PROC_RW1    && return SIGMA_G_PRIOR_SD_RW1
    proc == PROC_RW1C   && return SIGMA_G_PRIOR_SD_RW1C
    proc == PROC_RW2    && return SIGMA_G_PRIOR_SD_RW2
    proc == PROC_SHRUNK && return SIGMA_G_PRIOR_SD_SHRUNK
    return SIGMA_G_PRIOR_SD_IID          # iid and none
end

"""
    build_gamma!(gamma, raw_at, sigma_g, proc, n_years)

Year effects from the unconstrained `raw` vector, matching test2infeR's
`build_gamma`:

    none    gamma_y = 0
    iid     gamma_y = sigma_g * raw_y
    shrunk  gamma_y = sigma_g * raw_y            (differs from iid only in the
                                                  sigma prior, 0.10 vs 0.30)
    rw1     gamma_y = sigma_g * sum_{k<=y} raw_k
    rw1c    gamma_y = rw1_y - mean(rw1)          (sum-to-zero constrained)
    rw2     gamma_y = sigma_g * cumsum(cumsum(raw))_y

`rw1c` centres the walk so it is a pure DEVIATION series. Plain `rw1` leaves the
LEVEL unidentified -- `gamma[1]` is a free parameter and every later year is an
increment from it, so nothing anchors where the walk sits, and years with no
test data drift. Centring makes the level the job of `alpha`, which its own
prior and every capture in the study do anchor. Standard identifiability
constraint for RW/ICAR temporal effects.

`raw_at(y)` reads the y-th raw value, so this works for both `Float64` and the
dual numbers the forward-mode reference uses.
"""
@inline function build_gamma!(gamma, raw_at::F, sigma_g, proc::Int,
                              n_years::Int) where {F}
    if proc == PROC_NONE
        @inbounds for y in 1:n_years
            gamma[y] = zero(sigma_g)
        end
    elseif proc == PROC_RW1
        acc = zero(sigma_g)
        @inbounds for y in 1:n_years
            acc = acc + raw_at(y)
            gamma[y] = sigma_g * acc
        end
    elseif proc == PROC_RW1C
        # Two passes: accumulate the walk, then subtract its own mean. The mean
        # is over the same vector being written, so it cannot be folded into the
        # first pass.
        acc = zero(sigma_g)
        @inbounds for y in 1:n_years
            acc = acc + raw_at(y)
            gamma[y] = sigma_g * acc
        end
        tot = zero(sigma_g)
        @inbounds for y in 1:n_years
            tot = tot + gamma[y]
        end
        mu = tot / n_years
        @inbounds for y in 1:n_years
            gamma[y] = gamma[y] - mu
        end
    elseif proc == PROC_RW2
        # cumsum of the cumsum: `acc` is the running sum of raw, `acc2` the
        # running sum of THAT.
        acc = zero(sigma_g)
        acc2 = zero(sigma_g)
        @inbounds for y in 1:n_years
            acc = acc + raw_at(y)
            acc2 = acc2 + acc
            gamma[y] = sigma_g * acc2
        end
    else                                  # iid and shrunk
        @inbounds for y in 1:n_years
            gamma[y] = sigma_g * raw_at(y)
        end
    end
    return gamma
end

@inline season_of(t::Int, S::Int) = (t - 1) % S + 1
@inline year_of(t::Int, S::Int) = (t - 1) ÷ S + 1

"""
Flattened, immutable description of the data. Every field is concretely typed:
the whole point is that the log-posterior has no abstract fields to dispatch on.

Observations are stored CSR-style: badger `i`'s capture `j` occupies
`ptr[k] : ptr[k+1]-1` in `vals`/`tidx`, where `k` indexes captures globally.
"""
struct HMMData
    n_ind::Int
    S::Int
    n_years::Int
    seq_ends::Vector{Int}     # last capture index (into times) per badger
    entry_year::Vector{Int}   # year index of first capture, per badger
    times::Vector{Int}        # timestep of each capture, concatenated
    ptr::Vector{Int}          # CSR offsets into vals/tidx, length = length(times)+1
    vals::Vector{Int}         # 0/1 test result
    tidx::Vector{Int}         # which assay (1..N_TESTS) each val belongs to
    ids::Vector{Int}          # badger id per sequence
    captured::Vector{Bool}    # was the badger actually caught at this timestep?
end

"""
Which parameters are free, and where they sit in the flat vector θ.

Layout:  alpha[1:S], log_sigma_g, gamma_raw[1:n_years], pi1_0, pi1_mult,
         then (if infer_sesp) logit_Se[1:6], then a slot for each UNPINNED
         logit_Sp.

sigma_g and Se/Sp are stored unconstrained (log / logit) so the optimiser and
the sampler both work on an unbounded space -- this replaces Bijectors.

A pinned specificity is a constant, not a parameter: it gets no slot, no prior
term and no gradient entry. `sp_slot[k]` is that test's index in θ, or 0 when it
is pinned, so `n` shrinks by one per pinned test. Pinning culture (test 3) with
Se/Sp otherwise inferred gives n = 69 rather than 70 -- which is why this cannot
be done by masking a gradient after the fact.
"""
struct ParamLayout
    S::Int
    n_years::Int
    infer_sesp::Bool
    year_process::Int
    n::Int
    i_alpha::Int
    i_logsigma::Int
    i_gamma::Int
    i_pi1_0::Int
    i_pi1_mult::Int
    i_se::Int
    i_sp::Int
    sp_pin::NTuple{N_TESTS,Float64}   # NaN = free, otherwise the pinned value
    sp_slot::NTuple{N_TESTS,Int}      # index in θ, or 0 when pinned
end

const NO_SP_PIN = ntuple(_ -> NaN, N_TESTS)

@inline is_pinned(layout::ParamLayout, k::Int) = layout.sp_slot[k] == 0

function ParamLayout(S::Int, n_years::Int, infer_sesp::Bool,
                     year_process::Int = PROC_RW1,
                     sp_pin::NTuple{N_TESTS,Float64} = NO_SP_PIN)
    i_alpha = 1
    i_logsigma = i_alpha + S
    i_gamma = i_logsigma + 1
    i_pi1_0 = i_gamma + n_years
    i_pi1_mult = i_pi1_0 + 1
    i_se = i_pi1_mult + 1
    i_sp = infer_sesp ? i_se + N_TESTS : i_se
    # Pinning only means anything when Se/Sp are inferred: with them fixed
    # every Sp is already a constant.
    if infer_sesp
        # Built as a plain loop, not `ntuple` with a closure: a closure
        # capturing the running counter boxes it (Core.Box), which erases its
        # type and breaks trimming. Same reason `seed_partials` is a separate
        # function in dual.jl.
        slot_v = zeros(Int, N_TESTS)
        next = i_sp
        for k in 1:N_TESTS
            if isnan(sp_pin[k])
                slot_v[k] = next
                next += 1
            end
        end
        slots = NTuple{N_TESTS,Int}(slot_v)
        n = next - 1
        pins = sp_pin
    else
        slots = ntuple(_ -> 0, N_TESTS)
        n = i_pi1_mult
        pins = NO_SP_PIN
    end
    return ParamLayout(S, n_years, infer_sesp, year_process, n, i_alpha,
                       i_logsigma, i_gamma, i_pi1_0, i_pi1_mult, i_se, i_sp,
                       pins, slots)
end

# --- likelihood ------------------------------------------------------------
# Direct port of seq_loglik. Scaled forward pass over the 2-state chain,
# renormalising at each observed capture so the recursion stays in [0,1].
@inline function seq_loglik(data::HMMData, lo_seq::Int, hi_seq::Int,
                            pi1::T, alpha::AbstractVector{T},
                            gamma::AbstractVector{T},
                            Se::AbstractVector{T}, Sp::AbstractVector{T},
                            S::Int) where {T}
    aU = one(T) - pi1
    aI = pi1
    ll = zero(T)
    @inbounds for j in lo_seq:hi_seq
        if j > lo_seq
            t = data.times[j]
            lam = clamp_prob(logistic(alpha[season_of(t, S)] + gamma[year_of(t, S)]))
            aI = aI + aU * lam
            aU = aU * (one(T) - lam)
        end
        lo = data.ptr[j]
        hi = data.ptr[j + 1] - 1
        if hi >= lo
            bU = one(T)
            bI = one(T)
            for p in lo:hi
                k = data.tidx[p]
                if data.vals[p] == 1
                    bU = bU * (one(T) - Sp[k])
                    bI = bI * Se[k]
                else
                    bU = bU * Sp[k]
                    bI = bI * (one(T) - Se[k])
                end
            end
            aU = aU * bU
            aI = aI * bI
            c = aU + aI
            ll = ll + log(c)
            aU = aU / c
            aI = aI / c
        end
    end
    return ll
end

# --- priors ----------------------------------------------------------------
@inline normal_logpdf(x::T, mu::Float64, sd::Float64) where {T} =
    -0.5 * ((x - mu) / sd)^2 - log(sd) - 0.9189385332046727  # log sqrt(2pi)

# Beta log-density up to a constant in x; the normalising constant is dropped
# because it does not depend on the parameters being optimised.
@inline beta_logpdf_kernel(x::T, a::Float64, b::Float64) where {T} =
    (a - 1.0) * log(x) + (b - 1.0) * log(one(T) - x)

clamp01(x::Float64) = clamp(x, 1e-4, 1.0 - 1e-4)
sd_from_ci(lo::Float64, hi::Float64) = max((clamp01(hi) - clamp01(lo)) / 3.92, 1e-3)

function beta_from_moments(mu::Float64, sd::Float64)
    c = mu * (1.0 - mu) / sd^2 - 1.0
    return (max(mu * c, 0.5), max((1.0 - mu) * c, 0.5))
end

# --- accuracy priors -------------------------------------------------------
# Two families, chosen by `accuracy_prior`:
#
#   ACC_BETA          Beta(a,b) on the probability, the historical default.
#   ACC_LOGITNORMAL   LogitNormal(mu,sd), which the case uses for its inferred
#                     variants.
#
# The Beta family has a defect this exists to escape: `beta_from_moments` floors
# both shapes at 0.5, so a requested mean of 0.999 for culture specificity comes
# back as Beta(3.26, 0.5) with mean 0.867 -- the moments are not what was asked
# for -- and a shape below one makes the density diverge at the boundary, so MAP
# has no interior maximum. `map_refuses_singular_beta` is the guard.
#
# LogitNormal has neither problem: the density vanishes at both boundaries, and
# the literature point value is taken as the MEDIAN (which it is, by
# construction) rather than the mean.
const ACC_BETA        = 0
const ACC_LOGITNORMAL = 1

"""
    logitnormal_from_interval(center, lo, hi) -> (mu, sd)

Logit-scale mean and SD for a LogitNormal whose median is `center` and whose
central 95% interval contains both `lo` and `hi`.

The supplied intervals are asymmetric on the logit scale (Sp[4] is 0.881-0.999
around 0.943, i.e. 1.6 logits below and 4.1 above), so no single SD reproduces
both endpoints. Taking the LARGER half-width covers both rather than splitting
the difference; the interval is then wider than asked for on the tighter side,
which is the conservative direction.
"""
function logitnormal_from_interval(center::Float64, lo::Float64, hi::Float64)
    mu = logit_f(center)
    sd = max(mu - logit_f(lo), logit_f(hi) - mu) / 1.959963984540054
    return (mu, sd)
end

@inline logit_f(x::Float64) = log(x) - log1p(-x)

# In the UNCONSTRAINED space the CLI works in (z = logit(p)), a LogitNormal on p
# is exactly a Normal on z: the logit Jacobian log(p) + log(1-p) cancels the
# LogitNormal's own 1/(p(1-p)) factor. Verified against Distributions.jl to
# 1.1e-13 over a grid of (mu, sd, z). So the prior term is normal_logpdf(z,...)
# and its derivative is -(z-mu)/sd^2 -- no Jacobian term, unlike the Beta path.
@inline logitnormal_logpdf_z(z::T, mu::Float64, sd::Float64) where {T} =
    normal_logpdf(z, mu, sd)

const SE_CI_DEFAULT = ((0.370, 0.530), (0.370, 0.530), (0.025, 0.373),
                       (0.502, 0.798), (0.640, 0.901), (0.441, 0.544))
const SP_CI_DEFAULT = ((0.890, 0.980), (0.890, 0.980), (0.939, 0.999),
                       (0.881, 0.999), (0.621, 0.987), (0.622, 0.986))

"""
Prior parameters for one test's accuracy, in whichever family is in use.

`family` selects the meaning of `(p, q)`: `(a, b)` for `ACC_BETA`, `(mu, sd)`
on the logit scale for `ACC_LOGITNORMAL`. One concrete struct rather than a
union keeps the log-posterior free of abstract fields to dispatch on, which is
what makes it trimmable.
"""
struct AccuracyPrior
    family::Int
    p::Float64
    q::Float64
end

"Beta (a,b) pairs for the Se and Sp priors, from the Table 1 means and CIs."
function default_sesp_priors()
    se_ab = ntuple(k -> beta_from_moments(clamp01(SE_FIXED_DEFAULT[k]),
                                          sd_from_ci(SE_CI_DEFAULT[k][1], SE_CI_DEFAULT[k][2])),
                   N_TESTS)
    sp_ab = ntuple(k -> beta_from_moments(clamp01(SP_FIXED_DEFAULT[k]),
                                          sd_from_ci(SP_CI_DEFAULT[k][1], SP_CI_DEFAULT[k][2])),
                   N_TESTS)
    return se_ab, sp_ab
end

"Accuracy priors in `family`, from the Table 1 point values and intervals."
function default_accuracy_priors(family::Int)
    if family == ACC_BETA
        se_ab, sp_ab = default_sesp_priors()
        se = ntuple(k -> AccuracyPrior(ACC_BETA, se_ab[k][1], se_ab[k][2]), N_TESTS)
        sp = ntuple(k -> AccuracyPrior(ACC_BETA, sp_ab[k][1], sp_ab[k][2]), N_TESTS)
        return se, sp
    end
    se = ntuple(N_TESTS) do k
        mu, sd = logitnormal_from_interval(SE_FIXED_DEFAULT[k],
                                           SE_CI_DEFAULT[k][1], SE_CI_DEFAULT[k][2])
        AccuracyPrior(ACC_LOGITNORMAL, mu, sd)
    end
    sp = ntuple(N_TESTS) do k
        mu, sd = logitnormal_from_interval(SP_FIXED_DEFAULT[k],
                                           SP_CI_DEFAULT[k][1], SP_CI_DEFAULT[k][2])
        AccuracyPrior(ACC_LOGITNORMAL, mu, sd)
    end
    return se, sp
end

const AccuracyPriors = NTuple{N_TESTS,AccuracyPrior}

"""
Log density of one accuracy prior at `p = logistic(z)`.

With `jacobian = true` this is the density in the UNCONSTRAINED space (what a
sampler needs); with `jacobian = false` it is the density on the probability
scale (what Turing's MAP maximises).

  Beta:        beta_logpdf_kernel(p,a,b)  [+ log(p) + log(1-p)]
  LogitNormal: logpdf(LogitNormal, p)     [+ log(p) + log(1-p)]
               and those two together are exactly normal_logpdf(z, mu, sd),
               since the Jacobian cancels the LogitNormal's own 1/(p(1-p))
               factor -- verified against Distributions.jl to 1.1e-13. So the
               unconstrained form is the cheap one and the CONSTRAINED form is
               the one that has to subtract the Jacobian back off.
"""
@inline function accuracy_logpdf_z(pr::AccuracyPrior, z::T, p::T,
                                   jacobian::Bool = true) where {T}
    logjac = log(p) + log(one(T) - p)
    if pr.family == ACC_LOGITNORMAL
        lp = normal_logpdf(z, pr.p, pr.q)
        return jacobian ? lp : lp - logjac
    end
    lp = beta_logpdf_kernel(p, pr.p, pr.q)
    return jacobian ? lp + logjac : lp
end

"""
d(log prior)/dz for one accuracy parameter, matching `accuracy_logpdf_z`.

  LogitNormal: -(z - mu)/sd^2                       [- (1 - 2p) without J]
  Beta:        ((a-1)/p - (b-1)/(1-p)) * p(1-p)     [+ (1 - 2p) with J]

where p(1-p) is dp/dz and (1-2p) is d/dz of the Jacobian term log(p)+log(1-p).
"""
@inline function accuracy_dlogpdf_dz(pr::AccuracyPrior, z::Float64, p::Float64,
                                     jacobian::Bool = true)
    dlogjac = 1.0 - 2.0 * p
    if pr.family == ACC_LOGITNORMAL
        d = -(z - pr.p) / (pr.q * pr.q)
        return jacobian ? d : d - dlogjac
    end
    dp = p * (1.0 - p)
    d = ((pr.p - 1.0) / p - (pr.q - 1.0) / (1.0 - p)) * dp
    return jacobian ? d + dlogjac : d
end

"""
Does `pr` have a density that diverges at a probability boundary?

Only Beta can: a shape below one sends the density to infinity at 0 or 1, so
there is no interior maximum and a MAP estimate is not well defined. The engine
refuses MAP in that case rather than reporting whatever the optimiser stopped
at. LogitNormal vanishes at both boundaries, so it is always safe.
"""
@inline is_singular_at_boundary(pr::AccuracyPrior) =
    pr.family == ACC_BETA && (pr.p < 1.0 || pr.q < 1.0)

# --- log posterior ---------------------------------------------------------
"""
    logposterior(theta, data, layout, se_ab, sp_ab, penalty)

Unnormalised log posterior in the unconstrained space, including the Jacobian
terms for the log/logit transforms. Generic in the element type of `theta` so
the dual-number gradient runs through the identical code path.
"""
function logposterior(theta::AbstractVector{T}, data::HMMData, layout::ParamLayout,
                      se_pr::AccuracyPriors, sp_pr::AccuracyPriors,
                      penalty::Bool,
                      year_weight::Union{Nothing,Vector{Float64}} = nothing,
                      jacobian::Bool = true) where {T}
    S = layout.S
    n_years = layout.n_years
    lp = zero(T)

    # alpha ~ Normal(-3, 1.5), seasonal log-hazard intercepts
    alpha = Vector{T}(undef, S)
    @inbounds for s in 1:S
        alpha[s] = theta[layout.i_alpha + s - 1]
        lp = lp + normal_logpdf(alpha[s], HAZARD_PRIOR_MEAN, HAZARD_PRIOR_SD)
    end

    # sigma_g ~ truncated(Normal(0, sd); lower=0), stored as log_sigma_g.
    #
    # LOG2 because the prior is a HALF normal: truncating to the positive half
    # line doubles the density there, so its log density is the full normal's
    # plus log(2). Dropping it would be harmless for optimisation but makes log
    # densities incomparable with the reference engine's.
    log_sigma = theta[layout.i_logsigma]
    sigma_g = exp(log_sigma)
    lp = lp + normal_logpdf(sigma_g, 0.0, sigma_prior_sd(layout.year_process)) + LOG2
    # + log|J| for the log transform. See `jacobian` in the docstring: sampling
    # needs it, Turing's MAP does not include it.
    jacobian && (lp = lp + log_sigma)

    # gamma_raw ~ Normal(0,1), then the process maps raw to the year effects.
    gamma = Vector{T}(undef, n_years)
    @inbounds for y in 1:n_years
        lp = lp + normal_logpdf(theta[layout.i_gamma + y - 1], 0.0, 1.0)
    end
    build_gamma!(gamma, y -> theta[layout.i_gamma + y - 1], sigma_g,
                 layout.year_process, n_years)

    # Data-adaptive increment prior. `sd_y` depends only on the data, so this
    # contributes through gamma alone.
    if year_weight !== nothing
        yw = year_weight
        @inbounds for y in 2:n_years
            n_eff = min(yw[y], yw[y - 1])
            lp = lp + normal_logpdf(gamma[y] - gamma[y - 1], 0.0, year_link_sd(n_eff))
        end
    end

    pi1_0 = theta[layout.i_pi1_0]
    pi1_mult = theta[layout.i_pi1_mult]
    lp = lp + normal_logpdf(pi1_0, -1.7, 1.0) + normal_logpdf(pi1_mult, 1.0, 1.0)

    # Se/Sp: fixed constants, or inferred on the logit scale. A pinned
    # specificity is a constant with no slot and no prior term.
    Se = Vector{T}(undef, N_TESTS)
    Sp = Vector{T}(undef, N_TESTS)
    if layout.infer_sesp
        @inbounds for k in 1:N_TESTS
            zse = theta[layout.i_se + k - 1]
            se = logistic(zse)
            Se[k] = se
            lp = lp + accuracy_logpdf_z(se_pr[k], zse, se, jacobian)

            slot = layout.sp_slot[k]
            if slot == 0
                Sp[k] = T(layout.sp_pin[k])
            else
                zsp = theta[slot]
                sp = logistic(zsp)
                Sp[k] = sp
                lp = lp + accuracy_logpdf_z(sp_pr[k], zsp, sp, jacobian)
            end
        end
    else
        @inbounds for k in 1:N_TESTS
            Se[k] = T(SE_FIXED_DEFAULT[k])
            Sp[k] = T(SP_FIXED_DEFAULT[k])
        end
    end

    # The Se+Sp>1 penalty applies in BOTH cases, matching the engine, which puts
    # it outside its own `se_fixed === nothing` branch. With Se/Sp fixed it is a
    # constant (-0.3529 at the Table 1 values, essentially all of it from
    # Culture, where Se+Sp-1 = 0.099) so it shifts no gradient and moves no
    # optimum -- but it does shift the log density, and log densities have to be
    # comparable with the engine's for model comparison to mean anything.
    if penalty
        @inbounds for k in 1:N_TESTS
            lp = lp - PENALTY_WEIGHT *
                 log1pexp(-(Se[k] + Sp[k] - 1.0) / PENALTY_SCALE)
        end
    end

    # pi1 per year, then the HMM likelihood badger by badger.
    pi1_vec = Vector{T}(undef, n_years)
    @inbounds for y in 1:n_years
        pi1_vec[y] = clamp_prob(logistic(pi1_0 + pi1_mult * gamma[y]))
    end

    start = 1
    @inbounds for i in 1:data.n_ind
        e = data.seq_ends[i]
        lp = lp + seq_loglik(data, start, e, pi1_vec[data.entry_year[i]],
                             alpha, gamma, Se, Sp, S)
        start = e + 1
    end
    return lp
end

# Back-compatible entry point for callers holding raw Beta (a,b) pairs.
function logposterior(theta::AbstractVector{T}, data::HMMData, layout::ParamLayout,
                      se_ab::NTuple{N_TESTS,Tuple{Float64,Float64}},
                      sp_ab::NTuple{N_TESTS,Tuple{Float64,Float64}},
                      penalty::Bool) where {T}
    se_pr = ntuple(k -> AccuracyPrior(ACC_BETA, se_ab[k][1], se_ab[k][2]), N_TESTS)
    sp_pr = ntuple(k -> AccuracyPrior(ACC_BETA, sp_ab[k][1], sp_ab[k][2]), N_TESTS)
    return logposterior(theta, data, layout, se_pr, sp_pr, penalty, nothing, true)
end

# --- parameter extraction --------------------------------------------------
"Natural-scale parameters from an unconstrained vector."
struct Params
    alpha::Vector{Float64}
    sigma_g::Float64
    gamma::Vector{Float64}
    pi1_0::Float64
    pi1_mult::Float64
    pi1_vec::Vector{Float64}
    Se::Vector{Float64}
    Sp::Vector{Float64}
end

function extract_params(theta::Vector{Float64}, layout::ParamLayout)
    S = layout.S
    n_years = layout.n_years
    alpha = [theta[layout.i_alpha + s - 1] for s in 1:S]
    sigma_g = exp(theta[layout.i_logsigma])
    gamma = Vector{Float64}(undef, n_years)
    build_gamma!(gamma, y -> theta[layout.i_gamma + y - 1], sigma_g,
                 layout.year_process, n_years)
    pi1_0 = theta[layout.i_pi1_0]
    pi1_mult = theta[layout.i_pi1_mult]
    pi1_vec = [clamp_prob(logistic(pi1_0 + pi1_mult * gamma[y])) for y in 1:n_years]
    if layout.infer_sesp
        Se = [logistic(theta[layout.i_se + k - 1]) for k in 1:N_TESTS]
        # Must mirror the model exactly: a pinned Sp[k] is what the fit used, so
        # the smoother has to see the pinned value. Reading a slot for a pinned
        # test would read some other parameter entirely, and reconstructing it
        # from a prior the fit never sampled is the `pi1_free` bug in another
        # costume -- the posterior would be decoded under parameters the
        # likelihood never saw.
        Sp = [layout.sp_slot[k] == 0 ? layout.sp_pin[k] :
              logistic(theta[layout.sp_slot[k]]) for k in 1:N_TESTS]
    else
        Se = [SE_FIXED_DEFAULT[k] for k in 1:N_TESTS]
        Sp = [SP_FIXED_DEFAULT[k] for k in 1:N_TESTS]
    end
    return Params(alpha, sigma_g, gamma, pi1_0, pi1_mult, pi1_vec, Se, Sp)
end

"Annual hazard implied by the seasonal intercepts and that year's effect."
function annual_hazard(alpha::Vector{Float64}, gamma::Vector{Float64}, y::Int, S::Int)
    surv = 1.0
    for s in 1:S
        surv *= 1.0 - clamp_prob(logistic(alpha[s] + gamma[y]))
    end
    return 1.0 - surv
end

# --- posterior decoding ----------------------------------------------------
"""
Forward-backward over the 2-state chain for one badger, returning
P(infected at capture j) for each capture in the sequence.

Written out rather than taken from HiddenMarkovModels.jl, which is not needed
once the likelihood is hand-rolled.
"""
function forward_backward_seq(data::HMMData, lo_seq::Int, hi_seq::Int,
                              pi1::Float64, P::Params, S::Int)
    n = hi_seq - lo_seq + 1
    aU = Vector{Float64}(undef, n)
    aI = Vector{Float64}(undef, n)
    lam_step = Vector{Float64}(undef, n)   # hazard entering step j

    # forward
    u = 1.0 - pi1
    v = pi1
    @inbounds for j in 1:n
        gj = lo_seq + j - 1
        if j > 1
            t = data.times[gj]
            lam = clamp_prob(logistic(P.alpha[season_of(t, S)] + P.gamma[year_of(t, S)]))
            lam_step[j] = lam
            v = v + u * lam
            u = u * (1.0 - lam)
        else
            lam_step[j] = 0.0
        end
        lo = data.ptr[gj]
        hi = data.ptr[gj + 1] - 1
        if hi >= lo
            bU = 1.0
            bI = 1.0
            for p in lo:hi
                k = data.tidx[p]
                if data.vals[p] == 1
                    bU *= 1.0 - P.Sp[k]
                    bI *= P.Se[k]
                else
                    bU *= P.Sp[k]
                    bI *= 1.0 - P.Se[k]
                end
            end
            u *= bU
            v *= bI
            c = u + v
            u /= c
            v /= c
        end
        aU[j] = u
        aI[j] = v
    end

    # backward
    bU = Vector{Float64}(undef, n)
    bI = Vector{Float64}(undef, n)
    bU[n] = 1.0
    bI[n] = 1.0
    @inbounds for j in (n - 1):-1:1
        gj = lo_seq + j
        lam = lam_step[j + 1]
        eU = 1.0
        eI = 1.0
        lo = data.ptr[gj]
        hi = data.ptr[gj + 1] - 1
        if hi >= lo
            for p in lo:hi
                k = data.tidx[p]
                if data.vals[p] == 1
                    eU *= 1.0 - P.Sp[k]
                    eI *= P.Se[k]
                else
                    eU *= P.Sp[k]
                    eI *= 1.0 - P.Se[k]
                end
            end
        end
        # Uninfected can stay uninfected or become infected; infected absorbs.
        nu = (1.0 - lam) * eU * bU[j + 1] + lam * eI * bI[j + 1]
        ni = eI * bI[j + 1]
        c = max(nu + ni, 1e-300)
        bU[j] = nu / c
        bI[j] = ni / c
    end

    gam = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        num = aI[j] * bI[j]
        den = aU[j] * bU[j] + num
        gam[j] = den > 0.0 ? num / den : 0.0
    end
    return gam
end
