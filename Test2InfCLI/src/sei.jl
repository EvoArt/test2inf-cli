# Three-state model: Uninfected -> Exposed -> Infectious (`--states 3`).
#
# Priors, year process, pi1 and the Se/Sp handling are shared with the two-state
# model (model.jl, grad_analytic.jl). This file has the three-state likelihood,
# its reverse sweep, the smoother and the path sampler.
#
# Transitions per timestep (lam = infection hazard, tau = progression hazard):
#
#     U -> U  1 - lam        E -> E  1 - tau        I -> I  1
#     U -> E  lam            E -> I  tau
#
# No U -> I in one step. Entry distribution is (1 - pi1, pi1 (1 - rho), pi1 rho).
#
# Emissions: U and I as in the two-state model. For E each assay's sensitivity
# is scaled by a multiplier, P(+ | E) = m_k Se_k.
#
# With all m_k = 1 the likelihood is the two-state one for any tau and rho.

# Priors. tau and rho are stored as logits with LogitNormal priors, so they go
# through the same `jacobian` switch as Se/Sp.
#   tau  same as the infection hazard prior (median 0.047 per timestep)
#   rho  centred on 0.5, 95% within 0.05-0.95
#   m_k  Uniform(0, 1)
const SEI_TAU_PRIOR = AccuracyPrior(ACC_LOGITNORMAL, HAZARD_PRIOR_MEAN, HAZARD_PRIOR_SD)
const SEI_RHO_PRIOR = AccuracyPrior(ACC_LOGITNORMAL, 0.0, 1.5)
const SEI_MULT_PRIOR = AccuracyPrior(ACC_BETA, 1.0, 1.0)

"E's factor for one result: m*Se at a positive, 1 - m*Se at a negative."
@inline e_factor(pos::Bool, m, se) = pos ? m * se : one(se) - m * se

# --- likelihood ------------------------------------------------------------
# Generic in T so the forward-mode reference differentiates the same code.
@inline function seq_loglik_sei(data::HMMData, lo_seq::Int, hi_seq::Int,
                                pi1::T, rho::T, tau::T,
                                alpha::AbstractVector{T},
                                gamma::AbstractVector{T},
                                Se::AbstractVector{T}, Sp::AbstractVector{T},
                                em::AbstractVector{T},
                                S::Int) where {T}
    aU = one(T) - pi1
    aE = pi1 * (one(T) - rho)
    aI = pi1 * rho
    ll = zero(T)
    @inbounds for j in lo_seq:hi_seq
        if j > lo_seq
            t = data.times[j]
            lam = clamp_prob(logistic(alpha[season_of(t, S)] + gamma[year_of(t, S)]))
            aI = aI + aE * tau
            aE = aE * (one(T) - tau) + aU * lam
            aU = aU * (one(T) - lam)
        end
        lo = data.ptr[j]
        hi = data.ptr[j + 1] - 1
        if hi >= lo
            bU = one(T)
            bE = one(T)
            bI = one(T)
            for p in lo:hi
                k = data.tidx[p]
                if data.vals[p] == 1
                    fU = one(T) - Sp[k]
                    fI = Se[k]
                else
                    fU = Sp[k]
                    fI = one(T) - Se[k]
                end
                bU = bU * fU
                bI = bI * fI
                bE = bE * e_factor(data.vals[p] == 1, em[k], Se[k])
            end
            aU = aU * bU
            aE = aE * bE
            aI = aI * bI
            c = aU + aE + aI
            ll = ll + log(c)
            aU = aU / c
            aE = aE / c
            aI = aI / c
        end
    end
    return ll
end

# --- reverse sweep ---------------------------------------------------------
"Scratch space for one badger's three-state sweep, sized to the longest record."
struct SEIWork
    u_pre::Vector{Float64}    # state BEFORE the transition entering step j
    e_pre::Vector{Float64}
    v_pre::Vector{Float64}
    lam::Vector{Float64}
    bU::Vector{Float64}
    bE::Vector{Float64}
    bI::Vector{Float64}
    c::Vector{Float64}
    has_obs::Vector{Bool}
    yr::Vector{Int}
    sn::Vector{Int}
end

SEIWork(n::Int) = SEIWork(Vector{Float64}(undef, n), Vector{Float64}(undef, n),
                          Vector{Float64}(undef, n), Vector{Float64}(undef, n),
                          Vector{Float64}(undef, n), Vector{Float64}(undef, n),
                          Vector{Float64}(undef, n), Vector{Float64}(undef, n),
                          Vector{Bool}(undef, n), Vector{Int}(undef, n),
                          Vector{Int}(undef, n))

"""
    sei_seq_sweep!(w, data, lo_seq, hi_seq, pi1, rho, tau, alpha, gamma, Se, Sp,
                   em, S, gbar_alpha, gbar_gamma, sebar, spbar, embar)
        -> (ll, pibar, rhobar, taubar)

Log likelihood and adjoints for one badger. Three-state version of the inline
sweep in `logposterior_grad!`.

Forward step:

    transition   v <- v + e*tau ;  e <- e*(1 - tau) + u*lam ;  u <- u*(1 - lam)
    emission     u <- u*bU ;  e <- e*bE ;  v <- v*bI
    rescale      c = u + e + v ;  ll += log(c) ;  (u, e, v) /= c

The reverse pass undoes these in the opposite order. Hazard adjoints go into
`gbar_alpha`/`gbar_gamma`, emission adjoints into `sebar`/`spbar`/`embar` on the
natural scale. Returns d(ll)/d(pi1), d(ll)/d(rho), d(ll)/d(tau) for the caller
to transform.

`sebar`/`spbar` are filled even when Se/Sp are fixed (the multipliers need the
same quantities); the caller ignores them in that case.
"""
function sei_seq_sweep!(w::SEIWork, data::HMMData, lo_seq::Int, hi_seq::Int,
                        pi1::Float64, rho::Float64, tau::Float64,
                        alpha::Vector{Float64}, gamma::Vector{Float64},
                        Se::Vector{Float64}, Sp::Vector{Float64},
                        em::Vector{Float64}, S::Int,
                        gbar_alpha::Vector{Float64}, gbar_gamma::Vector{Float64},
                        sebar::Vector{Float64}, spbar::Vector{Float64},
                        embar::Vector{Float64})
    n = hi_seq - lo_seq + 1
    ll = 0.0
    u = 1.0 - pi1
    e = pi1 * (1.0 - rho)
    v = pi1 * rho
    @inbounds for j in 1:n
        gj = lo_seq + j - 1
        w.u_pre[j] = u
        w.e_pre[j] = e
        w.v_pre[j] = v
        if j > 1
            t = data.times[gj]
            sn = season_of(t, S)
            yr = year_of(t, S)
            w.sn[j] = sn
            w.yr[j] = yr
            s = 1.0 / (1.0 + exp(-(alpha[sn] + gamma[yr])))
            lam = s < 1e-9 ? 1e-9 : (s > 1.0 - 1e-9 ? 1.0 - 1e-9 : s)
            w.lam[j] = lam
            v = v + e * tau
            e = e * (1.0 - tau) + u * lam
            u = u * (1.0 - lam)
        else
            w.lam[j] = 0.0
            w.sn[j] = 0
            w.yr[j] = 0
        end

        plo = data.ptr[gj]
        phi = data.ptr[gj + 1] - 1
        if phi >= plo
            bU = 1.0
            bE = 1.0
            bI = 1.0
            for p in plo:phi
                k = data.tidx[p]
                if data.vals[p] == 1
                    fU = 1.0 - Sp[k]
                    fI = Se[k]
                else
                    fU = Sp[k]
                    fI = 1.0 - Se[k]
                end
                bU *= fU
                bI *= fI
                bE *= e_factor(data.vals[p] == 1, em[k], Se[k])
            end
            w.bU[j] = bU
            w.bE[j] = bE
            w.bI[j] = bI
            u *= bU
            e *= bE
            v *= bI
            c = u + e + v
            w.c[j] = c
            ll += log(c)
            u /= c
            e /= c
            v /= c
            w.has_obs[j] = true
        else
            w.bU[j] = 1.0
            w.bE[j] = 1.0
            w.bI[j] = 1.0
            w.c[j] = 1.0
            w.has_obs[j] = false
        end
    end

    ubar = 0.0
    ebar = 0.0
    vbar = 0.0
    taubar = 0.0
    @inbounds for j in n:-1:1
        up = w.u_pre[j]
        ep = w.e_pre[j]
        vp = w.v_pre[j]
        if w.has_obs[j]
            c = w.c[j]
            if j > 1
                lam = w.lam[j]
                vt = vp + ep * tau
                et = ep * (1.0 - tau) + up * lam
                ut = up * (1.0 - lam)
            else
                ut = up
                et = ep
                vt = vp
            end
            uo = ut * w.bU[j] / c
            eo = et * w.bE[j] / c
            vo = vt * w.bI[j] / c
            # 1/c from log(c), plus the normalisation
            cbar = 1.0 / c - (ubar * uo + ebar * eo + vbar * vo) / c
            ubar_em = ubar / c + cbar
            ebar_em = ebar / c + cbar
            vbar_em = vbar / c + cbar
            # wX = d(ll)/d(bX) * bX; dividing by one assay's factor gives its
            # share. A zero factor only happens with a pinned Sp = 1 or a
            # pinned m = 0, where the share isn't needed, so skip it.
            wU = ubar_em * ut * w.bU[j]
            wE = ebar_em * et * w.bE[j]
            wI = vbar_em * vt * w.bI[j]
            gj = lo_seq + j - 1
            for pp in data.ptr[gj]:(data.ptr[gj + 1] - 1)
                k = data.tidx[pp]
                pos = data.vals[pp] == 1
                m = em[k]
                fU = pos ? 1.0 - Sp[k] : Sp[k]
                fI = pos ? Se[k] : 1.0 - Se[k]
                fE = e_factor(pos, m, Se[k])
                gU = fU > 0.0 ? wU / fU : 0.0
                gE = fE > 0.0 ? wE / fE : 0.0
                gI = wI / fI
                # fU = 1 - Sp, fI = Se, fE = m Se at a positive;
                # Sp, 1 - Se, 1 - m Se at a negative
                if pos
                    spbar[k] -= gU
                    sebar[k] += gI + m * gE
                    embar[k] += Se[k] * gE
                else
                    spbar[k] += gU
                    sebar[k] -= gI + m * gE
                    embar[k] -= Se[k] * gE
                end
            end
            ubar = ubar_em * w.bU[j]
            ebar = ebar_em * w.bE[j]
            vbar = vbar_em * w.bI[j]
        end

        if j > 1
            lam = w.lam[j]
            # lam moves mass u_pre from U to E; tau moves e_pre from E to I.
            if lam > 1e-9 && lam < 1.0 - 1e-9
                zbar = (ebar - ubar) * up * lam * (1.0 - lam)
                gbar_alpha[w.sn[j]] += zbar
                gbar_gamma[w.yr[j]] += zbar
            end
            taubar += (vbar - ebar) * ep
            new_ubar = ubar * (1.0 - lam) + ebar * lam
            new_ebar = ebar * (1.0 - tau) + vbar * tau
            ubar = new_ubar
            ebar = new_ebar
            # vbar passes through unchanged
        end
    end

    # Seed: u = 1 - pi1, e = pi1 (1 - rho), v = pi1 rho.
    pibar = -ubar + ebar * (1.0 - rho) + vbar * rho
    rhobar = pi1 * (vbar - ebar)
    return (ll, pibar, rhobar, taubar)
end

# --- decoding --------------------------------------------------------------
"Emission factors (bU, bE, bI) at global capture index `gj`; all one if untested."
@inline function emission3(data::HMMData, gj::Int, P::Params)
    bU = 1.0
    bE = 1.0
    bI = 1.0
    @inbounds for p in data.ptr[gj]:(data.ptr[gj + 1] - 1)
        k = data.tidx[p]
        if data.vals[p] == 1
            fU = 1.0 - P.Sp[k]
            fI = P.Se[k]
        else
            fU = P.Sp[k]
            fI = 1.0 - P.Se[k]
        end
        bU *= fU
        bI *= fI
        # two-state Params have no multipliers, and bE is unused there
        P.sei && (bE *= e_factor(data.vals[p] == 1, P.em[k], P.Se[k]))
    end
    return (bU, bE, bI)
end

"""
Filtered state probabilities for one badger, written into `fU`/`fE`/`fI`, with
the hazard entering each step in `lam_step`. Handles both models: with two
states `fE` stays zero and the hazard moves U straight to I.
"""
function forward_filter3!(fU::Vector{Float64}, fE::Vector{Float64},
                          fI::Vector{Float64}, lam_step::Vector{Float64},
                          data::HMMData, lo_seq::Int, hi_seq::Int,
                          pi1::Float64, P::Params, S::Int)
    n = hi_seq - lo_seq + 1
    u = 1.0 - pi1
    e = P.sei ? pi1 * (1.0 - P.rho) : 0.0
    v = P.sei ? pi1 * P.rho : pi1
    @inbounds for j in 1:n
        gj = lo_seq + j - 1
        if j > 1
            t = data.times[gj]
            lam = clamp_prob(logistic(P.alpha[season_of(t, S)] + P.gamma[year_of(t, S)]))
            lam_step[j] = lam
            if P.sei
                v = v + e * P.tau
                e = e * (1.0 - P.tau) + u * lam
            else
                v = v + u * lam
            end
            u = u * (1.0 - lam)
        else
            lam_step[j] = 0.0
        end
        if data.ptr[gj + 1] > data.ptr[gj]
            bU, bE, bI = emission3(data, gj, P)
            u *= bU
            e *= bE
            v *= bI
            c = u + e + v
            u /= c
            e /= c
            v /= c
        end
        fU[j] = u
        fE[j] = e
        fI[j] = v
    end
    return n
end

"""
Forward-backward over the three-state chain for one badger. Writes
P(exposed) and P(infectious) at each capture into `p_exp`/`p_inf` starting at
`lo_seq`; P(uninfected) is the remainder.
"""
function forward_backward_sei!(p_exp::Vector{Float64}, p_inf::Vector{Float64},
                               data::HMMData, lo_seq::Int, hi_seq::Int,
                               pi1::Float64, P::Params, S::Int)
    n = hi_seq - lo_seq + 1
    fU = Vector{Float64}(undef, n)
    fE = Vector{Float64}(undef, n)
    fI = Vector{Float64}(undef, n)
    lam_step = Vector{Float64}(undef, n)
    forward_filter3!(fU, fE, fI, lam_step, data, lo_seq, hi_seq, pi1, P, S)

    tau = P.tau
    bU = 1.0
    bE = 1.0
    bI = 1.0
    @inbounds for j in n:-1:1
        den = fU[j] * bU + fE[j] * bE + fI[j] * bI
        gj = lo_seq + j - 1
        p_exp[gj] = den > 0.0 ? fE[j] * bE / den : 0.0
        p_inf[gj] = den > 0.0 ? fI[j] * bI / den : 0.0
        j == 1 && break
        # Step the backward message from j to j-1 through step j's emission
        # and the transition that entered it.
        eU, eE, eI = emission3(data, gj, P)
        lam = lam_step[j]
        nu = (1.0 - lam) * eU * bU + lam * eE * bE
        ne = (1.0 - tau) * eE * bE + tau * eI * bI
        ni = eI * bI
        c = max(nu + ne + ni, 1e-300)
        bU = nu / c
        bE = ne / c
        bI = ni / c
    end
    return nothing
end

"P(exposed) and P(infectious) per capture, for every badger."
function decode_sei(data::HMMData, P::Params)
    n = length(data.times)
    p_exp = Vector{Float64}(undef, n)
    p_inf = Vector{Float64}(undef, n)
    start = 1
    for i in 1:data.n_ind
        e = data.seq_ends[i]
        forward_backward_sei!(p_exp, p_inf, data, start, e,
                              entry_prob(data, P.pi1_vec, i), P, data.S)
        start = e + 1
    end
    return p_exp, p_inf
end

# --- path sampling ---------------------------------------------------------
"""
    sample_path!(states, rng, fU, fE, fI, lam_step, data, lo_seq, hi_seq,
                 pi1, P, S) -> n

Sample one state path for a badger given the parameters and its test results
(forward filtering, backward sampling). `states[j]` is 0 uninfected, 1 exposed,
2 infectious. Under the two-state model state 1 is never used and 2 means
infected.
"""
function sample_path!(states::Vector{Int}, rng::Rng,
                      fU::Vector{Float64}, fE::Vector{Float64},
                      fI::Vector{Float64}, lam_step::Vector{Float64},
                      data::HMMData, lo_seq::Int, hi_seq::Int,
                      pi1::Float64, P::Params, S::Int)
    n = forward_filter3!(fU, fE, fI, lam_step, data, lo_seq, hi_seq, pi1, P, S)
    @inbounds begin
        r = rand_uniform(rng)
        s = r < fU[n] ? 0 : (r < fU[n] + fE[n] ? 1 : 2)
        states[n] = s
        for j in (n - 1):-1:1
            # weight each state by filtered_j * P(state -> state_{j+1})
            lam = lam_step[j + 1]
            if s == 0
                # only U can precede U
                wU = 1.0; wE = 0.0; wI = 0.0
            elseif P.sei
                if s == 1
                    wU = fU[j] * lam
                    wE = fE[j] * (1.0 - P.tau)
                    wI = 0.0
                else
                    wU = 0.0
                    wE = fE[j] * P.tau
                    wI = fI[j]
                end
            else
                wU = fU[j] * lam
                wE = 0.0
                wI = fI[j]
            end
            tot = wU + wE + wI
            if tot > 0.0
                r = rand_uniform(rng) * tot
                s = r < wU ? 0 : (r < wU + wE ? 1 : 2)
            end
            states[j] = s
        end
    end
    return n
end
