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
# Year-effect process. These MUST track test2infeR's build_gamma/sigma_prior:
#   iid  gamma = sigma_g .* raw           sigma_g ~ Normal+(0, 0.30)
#   rw1  gamma = sigma_g .* cumsum(raw)   sigma_g ~ Normal+(0, 0.05)
# rw1 is the package default (`year_process = "rw1"` in hmm_inference), and
# neither sql-e2e script overrides it, so it is the default here too. The
# tighter sigma prior is part of the process, not an independent knob: rw1
# accumulates, so the same sigma would give far larger year effects.
const PROC_IID = 0
const PROC_RW1 = 1
const SIGMA_G_PRIOR_SD_IID = 0.30
const SIGMA_G_PRIOR_SD_RW1 = 0.05
@inline sigma_prior_sd(proc::Int) = proc == PROC_RW1 ? SIGMA_G_PRIOR_SD_RW1 :
                                                       SIGMA_G_PRIOR_SD_IID

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
         then (if infer_sesp) logit_Se[1:6], logit_Sp[1:6].

sigma_g and Se/Sp are stored unconstrained (log / logit) so the optimiser and
the sampler both work on an unbounded space -- this replaces Bijectors.
"""
struct ParamLayout
    S::Int
    n_years::Int
    infer_sesp::Bool
    year_process::Int          # PROC_IID or PROC_RW1
    n::Int
    i_alpha::Int
    i_logsigma::Int
    i_gamma::Int
    i_pi1_0::Int
    i_pi1_mult::Int
    i_se::Int
    i_sp::Int
end

function ParamLayout(S::Int, n_years::Int, infer_sesp::Bool,
                     year_process::Int = PROC_RW1)
    i_alpha = 1
    i_logsigma = i_alpha + S
    i_gamma = i_logsigma + 1
    i_pi1_0 = i_gamma + n_years
    i_pi1_mult = i_pi1_0 + 1
    i_se = i_pi1_mult + 1
    i_sp = infer_sesp ? i_se + N_TESTS : i_se
    n = infer_sesp ? i_sp + N_TESTS - 1 : i_pi1_mult
    return ParamLayout(S, n_years, infer_sesp, year_process, n, i_alpha,
                       i_logsigma, i_gamma, i_pi1_0, i_pi1_mult, i_se, i_sp)
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

const SE_CI_DEFAULT = ((0.370, 0.530), (0.370, 0.530), (0.025, 0.373),
                       (0.502, 0.798), (0.640, 0.901), (0.441, 0.544))
const SP_CI_DEFAULT = ((0.890, 0.980), (0.890, 0.980), (0.939, 0.999),
                       (0.881, 0.999), (0.621, 0.987), (0.622, 0.986))

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

# --- log posterior ---------------------------------------------------------
"""
    logposterior(theta, data, layout, se_ab, sp_ab, penalty)

Unnormalised log posterior in the unconstrained space, including the Jacobian
terms for the log/logit transforms. Generic in the element type of `theta` so
the dual-number gradient runs through the identical code path.
"""
function logposterior(theta::AbstractVector{T}, data::HMMData, layout::ParamLayout,
                      se_ab::NTuple{N_TESTS,Tuple{Float64,Float64}},
                      sp_ab::NTuple{N_TESTS,Tuple{Float64,Float64}},
                      penalty::Bool) where {T}
    S = layout.S
    n_years = layout.n_years
    lp = zero(T)

    # alpha ~ Normal(-3, 1.5), seasonal log-hazard intercepts
    alpha = Vector{T}(undef, S)
    @inbounds for s in 1:S
        alpha[s] = theta[layout.i_alpha + s - 1]
        lp = lp + normal_logpdf(alpha[s], HAZARD_PRIOR_MEAN, HAZARD_PRIOR_SD)
    end

    # sigma_g ~ Normal+(0, 0.3), stored as log_sigma_g; + log|J| = log_sigma_g
    log_sigma = theta[layout.i_logsigma]
    sigma_g = exp(log_sigma)
    lp = lp + normal_logpdf(sigma_g, 0.0, sigma_prior_sd(layout.year_process)) + log_sigma

    # gamma_raw ~ Normal(0,1), then the process maps raw to the year effects:
    #   iid  gamma_y = sigma_g * raw_y
    #   rw1  gamma_y = sigma_g * sum(raw_1..raw_y)     (a random walk)
    gamma = Vector{T}(undef, n_years)
    if layout.year_process == PROC_RW1
        acc = zero(T)
        @inbounds for y in 1:n_years
            raw = theta[layout.i_gamma + y - 1]
            lp = lp + normal_logpdf(raw, 0.0, 1.0)
            acc = acc + raw
            gamma[y] = sigma_g * acc
        end
    else
        @inbounds for y in 1:n_years
            raw = theta[layout.i_gamma + y - 1]
            lp = lp + normal_logpdf(raw, 0.0, 1.0)
            gamma[y] = sigma_g * raw
        end
    end

    pi1_0 = theta[layout.i_pi1_0]
    pi1_mult = theta[layout.i_pi1_mult]
    lp = lp + normal_logpdf(pi1_0, -1.7, 1.0) + normal_logpdf(pi1_mult, 1.0, 1.0)

    # Se/Sp: either fixed constants or inferred on the logit scale.
    Se = Vector{T}(undef, N_TESTS)
    Sp = Vector{T}(undef, N_TESTS)
    if layout.infer_sesp
        @inbounds for k in 1:N_TESTS
            zse = theta[layout.i_se + k - 1]
            zsp = theta[layout.i_sp + k - 1]
            se = logistic(zse)
            sp = logistic(zsp)
            Se[k] = se
            Sp[k] = sp
            a, b = se_ab[k]
            c, d = sp_ab[k]
            # + log|J| for the logit transform: log(p) + log(1-p)
            lp = lp + beta_logpdf_kernel(se, a, b) + log(se) + log(one(T) - se)
            lp = lp + beta_logpdf_kernel(sp, c, d) + log(sp) + log(one(T) - sp)
        end
    else
        @inbounds for k in 1:N_TESTS
            Se[k] = T(SE_FIXED_DEFAULT[k])
            Sp[k] = T(SP_FIXED_DEFAULT[k])
        end
    end

    # The Se+Sp>1 penalty applies in BOTH cases, matching the package, which
    # puts it outside its own `se_fixed === nothing` branch. With Se/Sp fixed it
    # is a constant (-0.3529 at the Table 1 values, essentially all of it from
    # Culture, where Se+Sp-1 = 0.099) so it shifts no gradient and moves no
    # optimum -- but it does shift the log density, and log densities have to be
    # comparable with the package's for model comparison to mean anything.
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
    if layout.year_process == PROC_RW1
        acc = 0.0
        for y in 1:n_years
            acc += theta[layout.i_gamma + y - 1]
            gamma[y] = sigma_g * acc
        end
    else
        for y in 1:n_years
            gamma[y] = sigma_g * theta[layout.i_gamma + y - 1]
        end
    end
    pi1_0 = theta[layout.i_pi1_0]
    pi1_mult = theta[layout.i_pi1_mult]
    pi1_vec = [clamp_prob(logistic(pi1_0 + pi1_mult * gamma[y])) for y in 1:n_years]
    if layout.infer_sesp
        Se = [logistic(theta[layout.i_se + k - 1]) for k in 1:N_TESTS]
        Sp = [logistic(theta[layout.i_sp + k - 1]) for k in 1:N_TESTS]
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
