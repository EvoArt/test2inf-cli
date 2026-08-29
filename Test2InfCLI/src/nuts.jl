# No-U-Turn Sampler: multinomial NUTS with a diagonal mass matrix, Nesterov
# dual-averaging step size adaptation and Welford variance estimation, i.e. the
# same algorithm Stan and AdvancedHMC implement.
#
# AdvancedHMC itself is not the blocker -- Turing's DynamicPPL/SciMLBase
# plumbing around it is -- but AdvancedHMC still pulls in a large dependency
# stack for what is ~250 lines against a `logp`/`grad` pair. Hand-rolled here
# so the trimmed binary needs nothing beyond Base.
#
# References: Hoffman & Gelman (2014) Alg. 6; Betancourt (2017) for the
# multinomial sampling and the generalised U-turn criterion.

const MAX_TREE_DEPTH = 10
const DELTA_MAX = 1000.0   # divergence threshold on the energy error

"""
Result of a NUTS run: post-warmup draws plus the adapted tuning parameters.

Divergences and acceptance are reported **separately for warmup and sampling**.
Warmup divergences are expected -- the step size starts far too large and dual
averaging walks it down -- and say nothing about the validity of the draws.
Only `n_divergent` (post-warmup) is a correctness signal. Reporting a combined
figure makes a clean run look broken: the real cohort produces 30 warmup
divergences and zero afterwards.
"""
struct NUTSResult
    draws::Matrix{Float64}      # n_params x n_draws (post-warmup)
    logp::Vector{Float64}
    step_size::Float64
    inv_mass::Vector{Float64}
    n_divergent::Int            # POST-WARMUP only; the number that matters
    accept_rate::Float64        # post-warmup mean Metropolis acceptance
    n_divergent_warmup::Int
    accept_rate_warmup::Float64
    mean_depth::Float64         # post-warmup mean tree depth
end

mutable struct DualAverage
    mu::Float64
    log_eps_bar::Float64
    h_bar::Float64
    counter::Int
    gamma::Float64
    t0::Float64
    kappa::Float64
    delta::Float64
end

DualAverage(eps0::Float64, delta::Float64) =
    DualAverage(log(10.0 * eps0), 0.0, 0.0, 0, 0.05, 10.0, 0.75, delta)

function adapt!(da::DualAverage, accept_prob::Float64)
    da.counter += 1
    m = Float64(da.counter)
    eta = 1.0 / (m + da.t0)
    da.h_bar = (1.0 - eta) * da.h_bar + eta * (da.delta - accept_prob)
    log_eps = da.mu - sqrt(m) / da.gamma * da.h_bar
    mk = m^(-da.kappa)
    da.log_eps_bar = mk * log_eps + (1.0 - mk) * da.log_eps_bar
    return exp(log_eps)
end

final_eps(da::DualAverage) = exp(da.log_eps_bar)

"Welford accumulator for the diagonal mass matrix."
mutable struct WelfordVar
    n::Int
    mean::Vector{Float64}
    m2::Vector{Float64}
end

WelfordVar(d::Int) = WelfordVar(0, zeros(d), zeros(d))

function accumulate!(w::WelfordVar, x::Vector{Float64})
    w.n += 1
    @inbounds for i in eachindex(x)
        delta = x[i] - w.mean[i]
        w.mean[i] += delta / w.n
        w.m2[i] += delta * (x[i] - w.mean[i])
    end
    return w
end

function variance(w::WelfordVar)
    d = length(w.mean)
    out = ones(d)
    if w.n > 2
        # Stan's regularisation toward unit variance for small samples.
        n = Float64(w.n)
        @inbounds for i in 1:d
            v = w.m2[i] / (n - 1.0)
            out[i] = (n / (n + 5.0)) * v + 1e-3 * (5.0 / (n + 5.0))
        end
    end
    return out
end

reset!(w::WelfordVar) = (w.n = 0; fill!(w.mean, 0.0); fill!(w.m2, 0.0); w)

# --- leapfrog --------------------------------------------------------------
@inline function kinetic(p::Vector{Float64}, inv_mass::Vector{Float64})
    s = 0.0
    @inbounds for i in eachindex(p)
        s += p[i] * p[i] * inv_mass[i]
    end
    return 0.5 * s
end

"""
One leapfrog step in place. `gradf!(g, x)` fills `g` with ∇logp(x) and returns
logp(x). Momentum is updated by half steps either side of the position update.
"""
@inline function leapfrog!(x::Vector{Float64}, p::Vector{Float64}, g::Vector{Float64},
                           eps::Float64, inv_mass::Vector{Float64}, gradf!::F) where {F}
    @inbounds for i in eachindex(p)
        p[i] += 0.5 * eps * g[i]
    end
    @inbounds for i in eachindex(x)
        x[i] += eps * inv_mass[i] * p[i]
    end
    lp = gradf!(g, x)
    @inbounds for i in eachindex(p)
        p[i] += 0.5 * eps * g[i]
    end
    return lp
end

# --- tree state ------------------------------------------------------------
# The subtree's edge states, its sampled proposal, and the running totals
# needed for multinomial weighting and the U-turn checks.
mutable struct Tree
    x_minus::Vector{Float64}
    p_minus::Vector{Float64}
    g_minus::Vector{Float64}
    x_plus::Vector{Float64}
    p_plus::Vector{Float64}
    g_plus::Vector{Float64}
    x_prop::Vector{Float64}
    lp_prop::Float64
    log_weight::Float64     # log sum exp of -H over the subtree
    p_sum::Vector{Float64}  # summed momentum, for the U-turn criterion
    n_alpha::Int
    sum_alpha::Float64
    diverged::Bool
    stopped::Bool   # a U-turn was detected inside this subtree
end

"No-U-turn check, generalised form: momentum sum vs both edge momenta."
@inline function no_u_turn(p_sum::Vector{Float64}, p_minus::Vector{Float64},
                           p_plus::Vector{Float64}, inv_mass::Vector{Float64})
    a = 0.0
    b = 0.0
    @inbounds for i in eachindex(p_sum)
        v = inv_mass[i] * p_sum[i]
        a += v * p_minus[i]
        b += v * p_plus[i]
    end
    return a > 0.0 && b > 0.0
end

function build_tree(x::Vector{Float64}, p::Vector{Float64}, g::Vector{Float64},
                    log_u::Float64, depth::Int, direction::Int, eps::Float64,
                    inv_mass::Vector{Float64}, h0::Float64,
                    gradf!::F, rng::Rng) where {F}
    if depth == 0
        xn = copy(x)
        pn = copy(p)
        gn = copy(g)
        lp = leapfrog!(xn, pn, gn, direction * eps, inv_mass, gradf!)
        h = -lp + kinetic(pn, inv_mass)
        if !isfinite(h)
            h = Inf
        end
        diverged = (h - h0) > DELTA_MAX
        log_w = -h
        a = min(1.0, exp(h0 - h))
        return Tree(xn, pn, gn, copy(xn), copy(pn), copy(gn), copy(xn), lp,
                    log_w, copy(pn), 1, a, diverged, false)
    end

    left = build_tree(x, p, g, log_u, depth - 1, direction, eps, inv_mass, h0, gradf!, rng)
    if left.diverged || left.stopped
        return left
    end

    # Extend from whichever edge we are growing toward.
    right = if direction == -1
        build_tree(left.x_minus, left.p_minus, left.g_minus, log_u, depth - 1,
                   direction, eps, inv_mass, h0, gradf!, rng)
    else
        build_tree(left.x_plus, left.p_plus, left.g_plus, log_u, depth - 1,
                   direction, eps, inv_mass, h0, gradf!, rng)
    end
    if right.diverged
        return right
    end

    # Multinomial (progressive) sampling between the two subtrees.
    total = logaddexp(left.log_weight, right.log_weight)
    if log(rand_uniform(rng)) < right.log_weight - total
        left.x_prop = right.x_prop
        left.lp_prop = right.lp_prop
    end
    left.log_weight = total
    left.n_alpha += right.n_alpha
    left.sum_alpha += right.sum_alpha

    if direction == -1
        left.x_minus = right.x_minus
        left.p_minus = right.p_minus
        left.g_minus = right.g_minus
    else
        left.x_plus = right.x_plus
        left.p_plus = right.p_plus
        left.g_plus = right.g_plus
    end
    @inbounds for i in eachindex(left.p_sum)
        left.p_sum[i] += right.p_sum[i]
    end

    # A U-turn inside the subtree is not a divergence: the draw stays valid,
    # we simply stop growing the trajectory.
    if right.stopped || !no_u_turn(left.p_sum, left.p_minus, left.p_plus, inv_mass)
        left.stopped = true
    end
    return left
end

@inline logaddexp(a::Float64, b::Float64) =
    a > b ? a + log1p(exp(b - a)) : b + log1p(exp(a - b))

"Heuristic initial step size: double/halve until the acceptance crosses 0.5."
function find_initial_eps(x0::Vector{Float64}, gradf!::F, inv_mass::Vector{Float64},
                          rng::Rng) where {F}
    d = length(x0)
    eps = 1.0
    g = zeros(d)
    lp = gradf!(g, x0)
    p = rand_normal!(Vector{Float64}(undef, d), rng)
    h0 = -lp + kinetic(p, inv_mass)

    xn = copy(x0)
    pn = copy(p)
    gn = copy(g)
    lpn = leapfrog!(xn, pn, gn, eps, inv_mass, gradf!)
    h1 = -lpn + kinetic(pn, inv_mass)
    a = isfinite(h1) ? exp(h0 - h1) : 0.0
    direction = a > 0.5 ? 1 : -1

    for _ in 1:50
        copyto!(xn, x0)
        copyto!(pn, p)
        copyto!(gn, g)
        lpn = leapfrog!(xn, pn, gn, eps, inv_mass, gradf!)
        h1 = -lpn + kinetic(pn, inv_mass)
        a = isfinite(h1) ? exp(h0 - h1) : 0.0
        if direction == 1
            a > 0.5 || break
            eps *= 2.0
        else
            a < 0.5 || break
            eps *= 0.5
        end
        (eps < 1e-10 || eps > 1e7) && break
    end
    return eps
end

"""
    nuts(logp_grad!, x0, n_draws, n_warmup, delta, seed)

`logp_grad!(g, x)` fills `g` with ∇logp and returns logp. Runs `n_warmup`
adaptation iterations (step size + diagonal mass) followed by `n_draws` kept
draws.
"""
function nuts(gradf!::F, x0::Vector{Float64}, n_draws::Int, n_warmup::Int,
              delta::Float64, seed::Int) where {F}
    d = length(x0)
    rng = Rng(seed)
    inv_mass = ones(d)

    x = copy(x0)
    g = zeros(d)
    lp = gradf!(g, x)

    eps = find_initial_eps(x, gradf!, inv_mass, rng)
    da = DualAverage(eps, delta)
    wel = WelfordVar(d)

    draws = Matrix{Float64}(undef, d, n_draws)
    lps = Vector{Float64}(undef, n_draws)
    n_div = 0
    acc_sum = 0.0
    acc_n = 0
    n_div_warm = 0
    acc_sum_warm = 0.0
    acc_n_warm = 0
    depth_sum = 0

    # Stan-style windowed adaptation: an initial fast interval, a set of
    # doubling slow windows where the metric is learned, and a final fast one.
    win_start = 75
    win_end = max(n_warmup - 50, win_start + 1)
    next_window = win_start + 25

    total = n_warmup + n_draws
    for it in 1:total
        warm = it <= n_warmup

        p = rand_normal!(Vector{Float64}(undef, d), rng)
        @inbounds for i in 1:d
            p[i] /= sqrt(inv_mass[i])   # sample from N(0, M)
        end
        h0 = -lp + kinetic(p, inv_mass)
        log_u = log(rand_uniform(rng)) - h0

        x_minus = copy(x); p_minus = copy(p); g_minus = copy(g)
        x_plus = copy(x);  p_plus = copy(p);  g_plus = copy(g)
        p_sum = copy(p)
        x_prop = copy(x)
        lp_prop = lp
        log_weight = -h0
        depth = 0
        diverged = false
        sum_alpha = 0.0
        n_alpha = 0

        while depth < MAX_TREE_DEPTH
            direction = rand_bool(rng) ? 1 : -1
            sub = if direction == -1
                build_tree(x_minus, p_minus, g_minus, log_u, depth, -1, eps,
                           inv_mass, h0, gradf!, rng)
            else
                build_tree(x_plus, p_plus, g_plus, log_u, depth, 1, eps,
                           inv_mass, h0, gradf!, rng)
            end
            if sub.diverged
                diverged = true
                sum_alpha += sub.sum_alpha
                n_alpha += sub.n_alpha
                break
            end

            # Multinomial acceptance between the existing tree and the new subtree.
            if log(rand_uniform(rng)) < sub.log_weight - log_weight
                copyto!(x_prop, sub.x_prop)
                lp_prop = sub.lp_prop
            end
            log_weight = logaddexp(log_weight, sub.log_weight)
            sum_alpha += sub.sum_alpha
            n_alpha += sub.n_alpha

            if direction == -1
                x_minus = sub.x_minus; p_minus = sub.p_minus; g_minus = sub.g_minus
            else
                x_plus = sub.x_plus; p_plus = sub.p_plus; g_plus = sub.g_plus
            end
            @inbounds for i in 1:d
                p_sum[i] += sub.p_sum[i]
            end

            sub.stopped && break
            no_u_turn(p_sum, p_minus, p_plus, inv_mass) || break
            depth += 1
        end

        # Accept the multinomially chosen proposal.
        copyto!(x, x_prop)
        lp = gradf!(g, x)

        a_bar = n_alpha > 0 ? sum_alpha / n_alpha : 0.0
        if warm
            diverged && (n_div_warm += 1)
            acc_sum_warm += a_bar
            acc_n_warm += 1
        else
            diverged && (n_div += 1)
            acc_sum += a_bar
            acc_n += 1
            depth_sum += depth
        end

        if warm
            eps = adapt!(da, a_bar)
            if it > win_start && it <= win_end
                accumulate!(wel, x)
                if it == next_window
                    inv_mass = variance(wel)
                    reset!(wel)
                    eps = find_initial_eps(x, gradf!, inv_mass, rng)
                    da = DualAverage(eps, delta)
                    next_window = min(it + 2 * (it - win_start), win_end)
                end
            end
            if it == n_warmup
                eps = final_eps(da)
            end
        else
            k = it - n_warmup
            @inbounds for i in 1:d
                draws[i, k] = x[i]
            end
            lps[k] = lp
        end
    end

    return NUTSResult(draws, lps, eps, inv_mass, n_div,
                      acc_n > 0 ? acc_sum / acc_n : 0.0,
                      n_div_warm,
                      acc_n_warm > 0 ? acc_sum_warm / acc_n_warm : 0.0,
                      acc_n > 0 ? depth_sum / acc_n : 0.0)
end
