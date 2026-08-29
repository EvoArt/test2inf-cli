# Minimal forward-mode dual numbers.
#
# ForwardDiff does not survive `--trim`: the build passes the verifier with zero
# errors and then dies at runtime with a MethodError on the Dual-argument
# instance of the objective (see references/trim-debugging.md). Everything here
# is concrete-typed and dependency-free, which does trim.
#
# N is the chunk width and is a compile-time constant, so `ntuple` unrolls and
# the partials live in registers rather than on the heap.

struct Dual{N} <: Real
    v::Float64
    p::NTuple{N,Float64}
end

Dual{N}(v::Float64) where {N} = Dual{N}(v, ntuple(_ -> 0.0, Val(N)))
Dual{N}(v::Int) where {N} = Dual{N}(Float64(v))

value(x::Dual) = x.v
value(x::Float64) = x
partials(x::Dual{N}) where {N} = x.p

# --- conversion / promotion ------------------------------------------------
# Needed so literals mix with Duals inside generic code without falling back to
# a dynamic promote path.
Base.convert(::Type{Dual{N}}, x::Float64) where {N} = Dual{N}(x)
Base.convert(::Type{Dual{N}}, x::Int) where {N} = Dual{N}(Float64(x))
Base.convert(::Type{Dual{N}}, x::Dual{N}) where {N} = x
Base.promote_rule(::Type{Dual{N}}, ::Type{Float64}) where {N} = Dual{N}
Base.promote_rule(::Type{Dual{N}}, ::Type{Int}) where {N} = Dual{N}

Base.zero(::Type{Dual{N}}) where {N} = Dual{N}(0.0)
Base.one(::Type{Dual{N}}) where {N} = Dual{N}(1.0)
Base.zero(::Dual{N}) where {N} = Dual{N}(0.0)
Base.one(::Dual{N}) where {N} = Dual{N}(1.0)

# --- arithmetic ------------------------------------------------------------
@inline Base.:+(a::Dual{N}, b::Dual{N}) where {N} =
    Dual{N}(a.v + b.v, ntuple(i -> a.p[i] + b.p[i], Val(N)))
@inline Base.:-(a::Dual{N}, b::Dual{N}) where {N} =
    Dual{N}(a.v - b.v, ntuple(i -> a.p[i] - b.p[i], Val(N)))
@inline Base.:*(a::Dual{N}, b::Dual{N}) where {N} =
    Dual{N}(a.v * b.v, ntuple(i -> a.p[i] * b.v + a.v * b.p[i], Val(N)))
@inline function Base.:/(a::Dual{N}, b::Dual{N}) where {N}
    inv_b = 1.0 / b.v
    q = a.v * inv_b
    return Dual{N}(q, ntuple(i -> (a.p[i] - q * b.p[i]) * inv_b, Val(N)))
end
@inline Base.:-(a::Dual{N}) where {N} = Dual{N}(-a.v, ntuple(i -> -a.p[i], Val(N)))

# Mixed Dual/Float64 forms, written out rather than left to promotion: these are
# the hot paths and we want them inlined with no conversion.
@inline Base.:+(a::Dual{N}, b::Float64) where {N} = Dual{N}(a.v + b, a.p)
@inline Base.:+(a::Float64, b::Dual{N}) where {N} = Dual{N}(a + b.v, b.p)
@inline Base.:-(a::Dual{N}, b::Float64) where {N} = Dual{N}(a.v - b, a.p)
@inline Base.:-(a::Float64, b::Dual{N}) where {N} =
    Dual{N}(a - b.v, ntuple(i -> -b.p[i], Val(N)))
@inline Base.:*(a::Dual{N}, b::Float64) where {N} =
    Dual{N}(a.v * b, ntuple(i -> a.p[i] * b, Val(N)))
@inline Base.:*(a::Float64, b::Dual{N}) where {N} =
    Dual{N}(a * b.v, ntuple(i -> a * b.p[i], Val(N)))
@inline Base.:/(a::Dual{N}, b::Float64) where {N} =
    Dual{N}(a.v / b, ntuple(i -> a.p[i] / b, Val(N)))
@inline function Base.:/(a::Float64, b::Dual{N}) where {N}
    inv_b = 1.0 / b.v
    q = a * inv_b
    return Dual{N}(q, ntuple(i -> (-q * b.p[i]) * inv_b, Val(N)))
end

# --- powers ----------------------------------------------------------------
@inline Base.literal_pow(::typeof(^), a::Dual{N}, ::Val{2}) where {N} = a * a
@inline Base.literal_pow(::typeof(^), a::Dual{N}, ::Val{3}) where {N} = a * a * a
@inline function Base.:^(a::Dual{N}, n::Int) where {N}
    c = Float64(n) * a.v^(n - 1)
    return Dual{N}(a.v^n, ntuple(i -> c * a.p[i], Val(N)))
end

# --- elementary functions --------------------------------------------------
@inline function Base.exp(a::Dual{N}) where {N}
    e = exp(a.v)
    return Dual{N}(e, ntuple(i -> e * a.p[i], Val(N)))
end
@inline function Base.log(a::Dual{N}) where {N}
    inv_a = 1.0 / a.v
    return Dual{N}(log(a.v), ntuple(i -> a.p[i] * inv_a, Val(N)))
end
@inline function Base.sqrt(a::Dual{N}) where {N}
    s = sqrt(a.v)
    c = 0.5 / s
    return Dual{N}(s, ntuple(i -> c * a.p[i], Val(N)))
end
@inline function Base.abs(a::Dual{N}) where {N}
    return a.v < 0.0 ? -a : a
end

# --- comparison ------------------------------------------------------------
# Compare on the value only; the derivative plays no part in ordering.
@inline Base.:<(a::Dual, b::Dual) = a.v < b.v
@inline Base.:<(a::Dual, b::Float64) = a.v < b
@inline Base.:<(a::Float64, b::Dual) = a < b.v
@inline Base.:<=(a::Dual, b::Dual) = a.v <= b.v
@inline Base.:<=(a::Dual, b::Float64) = a.v <= b
@inline Base.:<=(a::Float64, b::Dual) = a <= b.v
@inline Base.:(==)(a::Dual, b::Dual) = a.v == b.v
@inline Base.:(==)(a::Dual, b::Float64) = a.v == b
@inline Base.:(==)(a::Float64, b::Dual) = a == b.v
@inline Base.isless(a::Dual, b::Dual) = a.v < b.v

# --- helpers used by the model --------------------------------------------
@inline logistic(x::Float64) = 1.0 / (1.0 + exp(-x))
@inline function logistic(a::Dual{N}) where {N}
    s = logistic(a.v)
    c = s * (1.0 - s)
    return Dual{N}(s, ntuple(i -> c * a.p[i], Val(N)))
end

# log(1 + exp(x)), computed in the numerically stable branch-wise form.
@inline function log1pexp(x::Float64)
    x > 33.3 && return x
    x > -37.0 && return log1p(exp(x))
    return exp(x)
end
@inline function log1pexp(a::Dual{N}) where {N}
    c = logistic(a.v)
    return Dual{N}(log1pexp(a.v), ntuple(i -> c * a.p[i], Val(N)))
end

@inline clamp_prob(x::Float64) = clamp(x, 1e-9, 1.0 - 1e-9)
@inline function clamp_prob(a::Dual{N}) where {N}
    # Clamping is flat outside the bounds, so the derivative is zeroed there.
    if a.v < 1e-9
        return Dual{N}(1e-9)
    elseif a.v > 1.0 - 1e-9
        return Dual{N}(1.0 - 1e-9)
    else
        return a
    end
end

"""
    gradient!(g, f, x, ::Val{N})

Fill `g` with ∇f(x) using chunked forward mode, N partials at a time.
Returns f(x). `f` must accept a `Vector` of `Dual{N}` as well as `Vector{Float64}`.
"""
function gradient!(g::Vector{Float64}, f::F, x::Vector{Float64}, ::Val{N}) where {F,N}
    n = length(x)
    fx = 0.0
    xd = Vector{Dual{N}}(undef, n)
    nchunk = cld(n, N)
    for c in 1:nchunk
        offset = (c - 1) * N
        width = min(N, n - offset)
        # `seed_partials` is a separate function, not a closure: capturing
        # `offset` in an inner closure boxes it (Core.Box), which erases its
        # type and breaks trimming.
        @inbounds for i in 1:n
            xd[i] = Dual{N}(x[i], seed_partials(Val(N), i, offset, width))
        end
        r = f(xd)
        fx = r.v
        @inbounds for k in 1:width
            g[offset + k] = r.p[k]
        end
    end
    return fx
end

"Partial seeds for co-ordinate `i` within the chunk starting at `offset`."
@inline function seed_partials(::Val{N}, i::Int, offset::Int, width::Int) where {N}
    return ntuple(k -> (k <= width && offset + k == i) ? 1.0 : 0.0, Val(N))
end
