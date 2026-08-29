# Fixed-trajectory-length Hamiltonian Monte Carlo with a diagonal mass matrix.
#
# The complement to nuts.jl. NUTS chooses its trajectory length adaptively and
# adapts the metric as it goes; this runs a FIXED L with a metric and step size
# supplied up front, tuned offline (tune/tune_hmc.jl). That removes the tree
# building and the adaptation, leaving a fixed number of leapfrog steps per
# draw -- cheaper and fully deterministic given the seed.
#
# ## Dense metric: measured, and it wins
#
# rw1 parameterises gamma_y = sigma_g * cumsum(raw), so every year effect shares
# a factor with log_sigma_g -- a funnel, i.e. exactly the off-diagonal structure
# a diagonal metric cannot represent (it can only rescale axes, not rotate).
#
# Measured on the real cohort (58 par, 1000 kept draws), timing SAMPLING ONLY,
# which is what the pre-tuned path costs at run time:
#
#   metric     sample time   min ESS   min ESS/s
#   diagonal    244-317 s    336-407    1.21-1.67
#   DENSE        47- 53 s    532-659   11.2 -12.3     <- ~8x
#
# Dense samples at eps ~0.29-0.36 against diagonal's ~0.047, with tree depth 4.0
# against 6.3: about 1/6 the gradient evaluations per draw. Adapting a dense
# metric costs ~1.7x more warmup, which is why an earlier comparison that
# divided by TOTAL wall time called it a wash -- but warmup is paid once,
# offline, and compiled in. That is the whole point of `--model`.
#
# The linear algebra is hand-rolled: the shipped bundle has BLAS/LAPACK removed
# (see build.sh), so `cholesky` and matrix `*` are unavailable. At n = 58-70 the
# cost is trivial and the code is a few dozen lines. `chol_lower!` was checked
# to be exactly equal to LAPACK's factor, and `sample_momentum!` to produce the
# right covariance over 400k draws.

"Result of a fixed-L HMC run."
struct HMCResult
    draws::Matrix{Float64}      # n_params x n_draws
    logp::Vector{Float64}
    step_size::Float64
    L::Int
    inv_mass::Union{Vector{Float64},Matrix{Float64}}
    n_divergent::Int            # post-warmup; the correctness signal
    accept_rate::Float64
end

"""
    chol_lower!(L, A)

In-place Cholesky, `A = L*L'`, lower triangular. Hand-rolled because the trimmed
bundle ships without LAPACK. Returns false if `A` is not positive definite, so
the caller can fail loudly rather than propagate NaNs.
"""
function chol_lower!(L::Matrix{Float64}, A::Matrix{Float64})
    n = size(A, 1)
    @inbounds for i in 1:n
        for j in 1:i
            t = A[i, j]
            for k in 1:(j - 1)
                t -= L[i, k] * L[j, k]
            end
            if i == j
                t <= 0.0 && return false
                L[i, i] = sqrt(t)
            else
                L[i, j] = t / L[j, j]
            end
        end
        for j in (i + 1):n
            L[i, j] = 0.0
        end
    end
    return true
end

"y = M * x for a symmetric dense M, without BLAS."
@inline function symv!(y::Vector{Float64}, Mm::Matrix{Float64}, x::Vector{Float64})
    n = length(x)
    @inbounds for i in 1:n
        acc = 0.0
        for j in 1:n
            acc += Mm[i, j] * x[j]
        end
        y[i] = acc
    end
    return y
end

"""
Momentum p ~ N(0, M) where M = inv(inv_mass). Given the Cholesky factor `Lm` of
`inv_mass` (inv_mass = Lm*Lm'), solving Lm' p = z for standard normal z gives a
draw with covariance inv(inv_mass) -- note that is M, NOT inv_mass. Getting this
backwards still runs and samples the wrong distribution, so it is checked in the
test suite against an empirical covariance.
"""
function sample_momentum!(p::Vector{Float64}, Lm::Matrix{Float64},
                          z::Vector{Float64}, rng::Rng)
    n = length(p)
    @inbounds for i in 1:n
        z[i] = rand_normal(rng)
    end
    @inbounds for i in n:-1:1     # back substitution: Lm' is upper triangular
        acc = z[i]
        for k in (i + 1):n
            acc -= Lm[k, i] * p[k]
        end
        p[i] = acc / Lm[i, i]
    end
    return p
end

"Kinetic energy 0.5 * p' * inv_mass * p for a dense metric."
@inline function kinetic_dense(p::Vector{Float64}, inv_mass::Matrix{Float64},
                               tmp::Vector{Float64})
    symv!(tmp, inv_mass, p)
    acc = 0.0
    @inbounds for i in eachindex(p)
        acc += p[i] * tmp[i]
    end
    return 0.5 * acc
end

"One leapfrog step under a dense metric: x += eps * inv_mass * p."
@inline function leapfrog_dense!(x::Vector{Float64}, p::Vector{Float64},
                                 g::Vector{Float64}, eps::Float64,
                                 inv_mass::Matrix{Float64}, tmp::Vector{Float64},
                                 gradf!::F) where {F}
    @inbounds for i in eachindex(p)
        p[i] += 0.5 * eps * g[i]
    end
    symv!(tmp, inv_mass, p)
    @inbounds for i in eachindex(x)
        x[i] += eps * tmp[i]
    end
    lp = gradf!(g, x)
    @inbounds for i in eachindex(p)
        p[i] += 0.5 * eps * g[i]
    end
    return lp
end

"""
    hmc(gradf!, x0, inv_mass, eps, L, n_draws, n_warmup, seed)

Fixed-length HMC with a diagonal inverse mass vector. `n_warmup` draws are
discarded but NOT used to adapt anything: `eps` and `L` are taken as given,
which is the point of the pre-tuned path. Warmup here only lets the chain
reach the typical set from the MAP start.

Jitters the trajectory length uniformly in [1, L] each iteration. Fixed L can
resonate with a periodic posterior and return to near the start point; Neal
(2011) sec. 4.2 recommends jittering, and it costs nothing on average.
"""
function hmc(gradf!::F, x0::Vector{Float64}, inv_mass::Vector{Float64},
             eps::Float64, L::Int, n_draws::Int, n_warmup::Int,
             seed::Int) where {F}
    d = length(x0)
    length(inv_mass) == d ||
        error("inverse mass vector length does not match the parameter count")
    @inbounds for i in 1:d
        inv_mass[i] > 0.0 || error("inverse mass entries must be positive")
    end
    rng = Rng(seed)

    x = copy(x0)
    g = Vector{Float64}(undef, d)
    lp = gradf!(g, x)

    p = Vector{Float64}(undef, d)
    x_save = Vector{Float64}(undef, d)
    g_save = Vector{Float64}(undef, d)

    draws = Matrix{Float64}(undef, d, n_draws)
    lps = Vector{Float64}(undef, n_draws)

    n_div = 0
    acc_sum = 0.0
    acc_n = 0

    total = n_warmup + n_draws
    for it in 1:total
        warm = it <= n_warmup

        # p ~ N(0, M) with M = inv(inv_mass), i.e. sd = 1/sqrt(inv_mass).
        @inbounds for i in 1:d
            p[i] = rand_normal(rng) / sqrt(inv_mass[i])
        end
        h0 = -lp + kinetic(p, inv_mass)

        copyto!(x_save, x)
        copyto!(g_save, g)
        lp_save = lp

        steps = 1 + Int(floor(rand_uniform(rng) * L))
        steps > L && (steps = L)

        diverged = false
        for _ in 1:steps
            lp = leapfrog!(x, p, g, eps, inv_mass, gradf!)
            if !isfinite(lp)
                diverged = true
                break
            end
        end

        h1 = diverged ? Inf : -lp + kinetic(p, inv_mass)
        dh = h0 - h1
        (!isfinite(dh) || dh < -DELTA_MAX) && (diverged = true)

        a = diverged ? 0.0 : (dh > 0.0 ? 1.0 : exp(dh))
        if !diverged && rand_uniform(rng) < a
            # accepted: x, g, lp already hold the new state
        else
            copyto!(x, x_save)
            copyto!(g, g_save)
            lp = lp_save
        end

        if !warm
            diverged && (n_div += 1)
            acc_sum += a
            acc_n += 1
            k = it - n_warmup
            @inbounds for i in 1:d
                draws[i, k] = x[i]
            end
            lps[k] = lp
        end
    end

    return HMCResult(draws, lps, eps, L, inv_mass, n_div,
                     acc_n > 0 ? acc_sum / acc_n : 0.0)
end

"""
    hmc(gradf!, x0, inv_mass::Matrix, eps, L, n_draws, n_warmup, seed)

Dense-metric variant. Identical in structure to the diagonal method; the only
differences are that the momentum draw needs a Cholesky factor and each leapfrog
step needs an n x n matvec. At n = 58-70 that matvec is ~3-5 us against a
~1.5 ms gradient, i.e. under 0.5% -- which is why the dense metric wins so
clearly once its adaptation is paid offline.
"""
function hmc(gradf!::F, x0::Vector{Float64}, inv_mass::Matrix{Float64},
             eps::Float64, L::Int, n_draws::Int, n_warmup::Int,
             seed::Int) where {F}
    d = length(x0)
    size(inv_mass, 1) == d && size(inv_mass, 2) == d ||
        error("inverse mass matrix must be $(d)x$(d)")
    rng = Rng(seed)

    Lm = Matrix{Float64}(undef, d, d)
    chol_lower!(Lm, inv_mass) ||
        error("the pre-tuned inverse mass matrix is not positive definite")

    x = copy(x0)
    g = Vector{Float64}(undef, d)
    lp = gradf!(g, x)

    p = Vector{Float64}(undef, d)
    z = Vector{Float64}(undef, d)
    tmp = Vector{Float64}(undef, d)
    x_save = Vector{Float64}(undef, d)
    g_save = Vector{Float64}(undef, d)

    draws = Matrix{Float64}(undef, d, n_draws)
    lps = Vector{Float64}(undef, n_draws)

    n_div = 0
    acc_sum = 0.0
    acc_n = 0

    total = n_warmup + n_draws
    for it in 1:total
        warm = it <= n_warmup

        sample_momentum!(p, Lm, z, rng)
        h0 = -lp + kinetic_dense(p, inv_mass, tmp)

        copyto!(x_save, x)
        copyto!(g_save, g)
        lp_save = lp

        steps = 1 + Int(floor(rand_uniform(rng) * L))
        steps > L && (steps = L)

        diverged = false
        for _ in 1:steps
            lp = leapfrog_dense!(x, p, g, eps, inv_mass, tmp, gradf!)
            if !isfinite(lp)
                diverged = true
                break
            end
        end

        h1 = diverged ? Inf : -lp + kinetic_dense(p, inv_mass, tmp)
        dh = h0 - h1
        (!isfinite(dh) || dh < -DELTA_MAX) && (diverged = true)

        a = diverged ? 0.0 : (dh > 0.0 ? 1.0 : exp(dh))
        if !diverged && rand_uniform(rng) < a
            # accepted: x, g, lp already hold the new state
        else
            copyto!(x, x_save)
            copyto!(g, g_save)
            lp = lp_save
        end

        if !warm
            diverged && (n_div += 1)
            acc_sum += a
            acc_n += 1
            k = it - n_warmup
            @inbounds for i in 1:d
                draws[i, k] = x[i]
            end
            lps[k] = lp
        end
    end

    return HMCResult(draws, lps, eps, L, inv_mass, n_div,
                     acc_n > 0 ? acc_sum / acc_n : 0.0)
end
