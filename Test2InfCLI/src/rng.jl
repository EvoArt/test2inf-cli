# Self-contained xoshiro256++ with Box-Muller normals.
#
# `Random.Xoshiro` would do, but Random's `randn` uses ziggurat lookup tables
# whose initialisation is exactly the kind of thing trimming struggles with.
# This is ~40 lines and removes the stdlib dependency entirely.
#
# xoshiro256++ (Blackman & Vigna), seeded through SplitMix64 as the authors
# recommend. Statistically ample for MCMC; not cryptographic.

mutable struct Rng
    s0::UInt64
    s1::UInt64
    s2::UInt64
    s3::UInt64
    have_spare::Bool
    spare::Float64
end

@inline function splitmix64(state::UInt64)
    state += 0x9E3779B97F4A7C15
    z = state
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    return (z ⊻ (z >> 31)), state
end

function Rng(seed::Int)
    st = UInt64(seed % typemax(Int64)) + 0x2545F4914F6CDD1D
    a, st = splitmix64(st)
    b, st = splitmix64(st)
    c, st = splitmix64(st)
    d, _ = splitmix64(st)
    return Rng(a, b, c, d, false, 0.0)
end

@inline rotl(x::UInt64, k::Int) = (x << k) | (x >> (64 - k))

@inline function next_u64(r::Rng)
    result = rotl(r.s0 + r.s3, 23) + r.s0
    t = r.s1 << 17
    r.s2 ⊻= r.s0
    r.s3 ⊻= r.s1
    r.s1 ⊻= r.s2
    r.s0 ⊻= r.s3
    r.s2 ⊻= t
    r.s3 = rotl(r.s3, 45)
    return result
end

"Uniform on [0,1): take the top 53 bits, the standard double conversion."
@inline function rand_uniform(r::Rng)
    return Float64(next_u64(r) >> 11) * (1.0 / 9007199254740992.0)
end

@inline rand_bool(r::Rng) = (next_u64(r) & 0x1) == 0x1

"Standard normal via polar Box-Muller; the second variate is cached."
function rand_normal(r::Rng)
    if r.have_spare
        r.have_spare = false
        return r.spare
    end
    u = 0.0
    v = 0.0
    s = 0.0
    while true
        u = 2.0 * rand_uniform(r) - 1.0
        v = 2.0 * rand_uniform(r) - 1.0
        s = u * u + v * v
        (s > 0.0 && s < 1.0) && break
    end
    f = sqrt(-2.0 * log(s) / s)
    r.spare = v * f
    r.have_spare = true
    return u * f
end

function rand_normal!(out::Vector{Float64}, r::Rng)
    @inbounds for i in eachindex(out)
        out[i] = rand_normal(r)
    end
    return out
end
