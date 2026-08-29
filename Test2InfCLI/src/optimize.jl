# L-BFGS with a backtracking Armijo line search.
#
# Replaces Optim.jl/Optimization.jl, which reach SciMLBase and therefore
# FunctionWrappers -- untrimmable. This is the standard two-loop recursion
# (Nocedal & Wright, Alg. 7.4/7.5); nothing exotic, ~100 lines, no deps.

struct OptResult
    theta::Vector{Float64}
    logp::Float64
    iterations::Int
    converged::Bool
    grad_norm::Float64
end

"""
    lbfgs(negf!, x0; m, maxiter, gtol)

Minimise via `negf!(g, x) -> value`, which must fill `g` with the gradient.
Returns the minimiser. Used here on the *negative* log posterior.
"""
function lbfgs(negf!::F, x0::Vector{Float64};
               m::Int=10, maxiter::Int=1000, gtol::Float64=1e-6,
               ftol::Float64=1e-12) where {F}
    n = length(x0)
    x = copy(x0)
    g = zeros(n)
    fx = negf!(g, x)

    S = [zeros(n) for _ in 1:m]   # s_k = x_{k+1} - x_k
    Y = [zeros(n) for _ in 1:m]   # y_k = g_{k+1} - g_k
    rho = zeros(m)
    hist = 0        # how many pairs are populated
    head = 0        # index of the most recent pair

    q = zeros(n)
    d = zeros(n)
    alpha_i = zeros(m)
    x_new = zeros(n)
    g_new = zeros(n)

    iter = 0
    gnorm = norm2(g)
    converged = gnorm <= gtol

    while iter < maxiter && !converged
        iter += 1

        # --- two-loop recursion for the search direction -------------------
        copyto!(q, g)
        for j in 1:hist
            idx = mod1(head - j + 1, m)
            alpha_i[idx] = rho[idx] * dot2(S[idx], q)
            axpy2!(q, -alpha_i[idx], Y[idx])
        end
        # Initial Hessian scaling gamma_k = s'y / y'y
        if hist > 0
            sy = dot2(S[head], Y[head])
            yy = dot2(Y[head], Y[head])
            scale = yy > 0.0 ? sy / yy : 1.0
            @inbounds for i in 1:n
                q[i] *= scale
            end
        end
        for j in hist:-1:1
            idx = mod1(head - j + 1, m)
            beta = rho[idx] * dot2(Y[idx], q)
            axpy2!(q, alpha_i[idx] - beta, S[idx])
        end
        @inbounds for i in 1:n
            d[i] = -q[i]
        end

        # Guard against a non-descent direction from a stale//bad curvature pair.
        dg = dot2(d, g)
        if dg >= 0.0
            @inbounds for i in 1:n
                d[i] = -g[i]
            end
            dg = -dot2(g, g)
            hist = 0
            head = 0
        end

        # --- backtracking Armijo line search -------------------------------
        step = 1.0
        c1 = 1e-4
        ok = false
        fnew = fx
        for _ in 1:60
            @inbounds for i in 1:n
                x_new[i] = x[i] + step * d[i]
            end
            fnew = negf!(g_new, x_new)
            if isfinite(fnew) && fnew <= fx + c1 * step * dg
                ok = true
                break
            end
            step *= 0.5
        end
        if !ok
            break   # line search failed; stop at the current point
        end

        # --- curvature pair ------------------------------------------------
        head = mod1(head + 1, m)
        s_k = S[head]
        y_k = Y[head]
        @inbounds for i in 1:n
            s_k[i] = x_new[i] - x[i]
            y_k[i] = g_new[i] - g[i]
        end
        sy = dot2(s_k, y_k)
        if sy > 1e-12
            rho[head] = 1.0 / sy
            hist = min(hist + 1, m)
        else
            # Skip the update rather than store a non-positive-curvature pair.
            head = mod1(head - 1, m)
        end

        # Relative improvement in the objective, before overwriting fx.
        rel_improve = abs(fx - fnew) / max(1.0, abs(fx))

        copyto!(x, x_new)
        copyto!(g, g_new)
        fx = fnew
        gnorm = norm2(g)

        # Stop on either a small gradient or a stalled objective. The gradient
        # test alone is an absolute threshold that does not scale with the
        # problem: near a flat optimum |g| can sit just above gtol while the
        # objective has stopped moving, burning iterations for nothing.
        converged = gnorm <= gtol || rel_improve <= ftol
    end

    return OptResult(x, -fx, iter, converged, gnorm)
end

# Small BLAS-free helpers: keeping these local avoids pulling in LinearAlgebra,
# which would drag OpenBLAS into the bundle for three trivial loops.
@inline function dot2(a::Vector{Float64}, b::Vector{Float64})
    s = 0.0
    @inbounds for i in eachindex(a)
        s += a[i] * b[i]
    end
    return s
end

@inline function norm2(a::Vector{Float64})
    s = 0.0
    @inbounds for i in eachindex(a)
        s += a[i] * a[i]
    end
    return sqrt(s)
end

@inline function axpy2!(y::Vector{Float64}, a::Float64, x::Vector{Float64})
    @inbounds for i in eachindex(y)
        y[i] += a * x[i]
    end
    return y
end
