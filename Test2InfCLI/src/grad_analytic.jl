# Analytic gradient of the log posterior, covering all five `sql e2e` model
# variants: any test mask (all / no-DPP / culture-only), Se/Sp either fixed at
# Table 1 or inferred, and both year processes (rw1, the default, and iid --
# see the reverse-cumsum adjoint at the end of this file).
#
# Why this exists: forward-mode AD costs one sweep per chunk of parameters, so
# its gradient runs ~4.2x a single logp at 13 parameters and ~7.9x at 25. A
# reverse sweep gets every partial in ONE pass, at a cost independent of the
# parameter count -- and that advantage grows with the number of year effects
# (the real cohort has ~51, i.e. ~58 parameters, where forward mode is badly
# beaten).
#
# The forward recursion per badger is, with u = P(uninfected, obs so far) and
# v = P(infected, obs so far):
#
#     transition   v <- v + u*lam_j ;  u <- u*(1 - lam_j)
#     emission     u <- u*bU_j      ;  v <- v*bI_j
#     rescale      c_j = u + v      ;  ll += log(c_j) ;  u /= c_j ; v /= c_j
#
# Reverse mode walks these three steps backwards carrying adjoints (ubar, vbar).
#
# When Se/Sp are FIXED the emission factors bU/bI are constants contributing no
# parameter partials, so only lam_j and the initial pi1 do. When Se/Sp are
# INFERRED the emission factors carry partials too, and this is where the
# derivation has to work slightly harder:
#
#   bU_j = prod over assays k observed at j of  (1 - Sp_k) if the result was
#          positive, else Sp_k
#   bI_j = the same product with  Se_k  (positive) and  1 - Se_k  (negative)
#
# Each Se_k / Sp_k appears MULTIPLICATIVELY in that product, so
# d bU_j / d Sp_k = bU_j / f, where f is that assay's own factor -- the
# log-derivative trick. It is safe here because every factor is a logistic (or
# one minus a logistic) of a finite parameter and so is bounded away from zero;
# the fixed-Se/Sp path never needs it. From
#
#     ubar_em = d(lp)/d(u * bU_j)   and   u_t = u after the transition,
#
# the emission-factor adjoints are simply  bUbar_j = ubar_em * u_t  and
# bIbar_j = vbar_em * v_t, and each assay's share is  bUbar_j * bU_j / f
# with the sign of df/dSp_k (-1 for a positive result, +1 for a negative one).
# These accumulate per assay across every capture of every badger, then pass
# through the logit Jacobian, the Beta priors and the Se+Sp>1 penalty.
#
# Parameters: alpha[1:S], log_sigma_g, gamma_raw[1:n_years], pi1_0, pi1_mult,
# and -- when inferred -- logit_Se[1:6], logit_Sp[1:6]. That is 13 for a
# six-year fixed-Se/Sp study and 25 with Se/Sp inferred. Note
# gamma = sigma_g * gamma_raw, so gamma_raw and log_sigma_g share the chain
# rule through every lam and pi1.

"""
    logposterior_grad!(g, theta, data, layout, se_ab, sp_ab, penalty)

Log posterior and its exact gradient in one reverse sweep. Covers every model
variant the CLI exposes: any test mask, Se/Sp fixed or inferred. Returns the log
posterior and fills `g`.

`se_ab`/`sp_ab` are the Beta prior parameters and `penalty` toggles the
Se+Sp>1 soft constraint; both are ignored when `layout.infer_sesp` is false, so
the fixed-Se/Sp call path is unchanged.

Verified against both central finite differences and the forward-mode AD.
"""
function logposterior_grad!(g::Vector{Float64}, theta::Vector{Float64},
                            data::HMMData, layout::ParamLayout,
                            se_pr::AccuracyPriors, sp_pr::AccuracyPriors,
                            penalty::Bool,
                            year_weight::Vector{Float64} = EMPTY_YW,
                            jacobian::Bool = true)
    S = layout.S
    n_years = layout.n_years
    infer = layout.infer_sesp
    fill!(g, 0.0)

    # --- unpack ------------------------------------------------------------
    alpha = Vector{Float64}(undef, S)
    @inbounds for s in 1:S
        alpha[s] = theta[layout.i_alpha + s - 1]
    end
    log_sigma = theta[layout.i_logsigma]
    sigma_g = exp(log_sigma)
    proc = layout.year_process
    raw = Vector{Float64}(undef, n_years)
    gamma = Vector{Float64}(undef, n_years)
    @inbounds for y in 1:n_years
        raw[y] = theta[layout.i_gamma + y - 1]
    end
    build_gamma!(gamma, y -> raw[y], sigma_g, proc, n_years)
    pi1_0 = theta[layout.i_pi1_0]
    pi1_mult = theta[layout.i_pi1_mult]

    Se = Vector{Float64}(undef, N_TESTS)
    Sp = Vector{Float64}(undef, N_TESTS)
    if infer
        @inbounds for k in 1:N_TESTS
            Se[k] = logistic(theta[layout.i_se + k - 1])
            # A pinned specificity is a constant: it has no slot to read and
            # gets no gradient entry, but it still enters every emission and
            # the Se+Sp>1 penalty.
            Sp[k] = layout.sp_slot[k] == 0 ? layout.sp_pin[k] :
                    logistic(theta[layout.sp_slot[k]])
        end
    else
        @inbounds for k in 1:N_TESTS
            Se[k] = SE_FIXED_DEFAULT[k]
            # A pin overrides the tabulated value. Both are constants, so this
            # moves the density but contributes no gradient either way.
            Sp[k] = isnan(layout.sp_pin[k]) ? SP_FIXED_DEFAULT[k] :
                                              layout.sp_pin[k]
        end
    end

    # Adjoints of Se[k]/Sp[k] on the NATURAL scale, accumulated over every
    # emission; pushed through the logit Jacobian once at the end.
    sebar = zeros(N_TESTS)
    spbar = zeros(N_TESTS)

    lp = 0.0

    # --- priors (closed form, and their partials) --------------------------
    # alpha ~ Normal(mu, sd)
    @inbounds for s in 1:S
        z = (alpha[s] - HAZARD_PRIOR_MEAN) / HAZARD_PRIOR_SD
        lp += -0.5 * z^2 - log(HAZARD_PRIOR_SD) - 0.9189385332046727
        g[layout.i_alpha + s - 1] += -z / HAZARD_PRIOR_SD
    end

    # sigma_g ~ truncated(Normal(0, sd); lower=0), parameterised as log_sigma_g.
    # The sd depends on the year process (0.05 for rw1/rw1c, 0.30 for iid).
    # LOG2 is the half-normal normalisation; it is constant, so it enters lp but
    # not the gradient.
    #
    #   d/dlog_sigma [ -sigma^2/(2 sd^2) ]           = -sigma^2/sd^2
    #   d/dlog_sigma [ log_sigma ]                   = 1        (the Jacobian)
    #
    # so the Jacobian contributes exactly the +1. Both lp and its derivative
    # have to honour `jacobian` together, or the optimiser would descend a
    # function it is not evaluating.
    sig_sd = sigma_prior_sd(layout.year_process)
    lp += -0.5 * (sigma_g / sig_sd)^2 - log(sig_sd) -
          0.9189385332046727 + LOG2
    dlogsigma = -(sigma_g^2) / sig_sd^2
    if jacobian
        lp += log_sigma
        dlogsigma += 1.0
    end

    # gamma_raw ~ Normal(0,1)
    @inbounds for y in 1:n_years
        lp += -0.5 * raw[y]^2 - 0.9189385332046727
        g[layout.i_gamma + y - 1] += -raw[y]
    end

    lp += -0.5 * ((pi1_0 + 1.7) / 1.0)^2 - 0.9189385332046727
    g[layout.i_pi1_0] += -(pi1_0 + 1.7)
    lp += -0.5 * ((pi1_mult - 1.0) / 1.0)^2 - 0.9189385332046727
    g[layout.i_pi1_mult] += -(pi1_mult - 1.0)

    # --- Se/Sp priors and the identifiability penalty ----------------------
    # Two prior families, and they differ in WHICH SCALE their derivative is
    # natural on, so they are handled differently here:
    #
    #   Beta. The kernel plus the logit log|J| is
    #     (a-1)log(p) + (b-1)log(1-p) + log(p) + log(1-p) = a*log(p) + b*log(1-p),
    #   whose derivative wrt p is a/p - b/(1-p). That is a NATURAL-scale
    #   derivative, so it joins sebar/spbar and picks up the logit Jacobian with
    #   everything else at the end.
    #
    #   LogitNormal. In this unconstrained space it is exactly Normal(z; mu,sd)
    #   -- the logit Jacobian cancels the LogitNormal's own 1/(p(1-p)) factor
    #   (verified against Distributions.jl to 1.1e-13). Its derivative
    #   -(z-mu)/sd^2 is therefore already on the LOGIT scale and must go
    #   straight into `g`, NOT through sebar/spbar, or it would be multiplied by
    #   the Jacobian a second time.
    if infer
        @inbounds for k in 1:N_TESTS
            se = Se[k]
            zse = theta[layout.i_se + k - 1]
            lp += accuracy_logpdf_z(se_pr[k], zse, se, jacobian)
            if se_pr[k].family == ACC_LOGITNORMAL
                g[layout.i_se + k - 1] +=
                    accuracy_dlogpdf_dz(se_pr[k], zse, se, jacobian)
            else
                # WITH the Jacobian the Beta kernel and log(p)+log(1-p) combine
                # to a*log(p) + b*log(1-p), whose p-derivative is the folded
                # a/p - b/(1-p). WITHOUT it only the kernel remains, so the
                # shapes lose their +1. Both are natural-scale derivatives and
                # pick up dp/dz with everything else at the end.
                a, b = se_pr[k].p, se_pr[k].q
                sebar[k] += jacobian ? a / se - b / (1.0 - se) :
                                       (a - 1.0) / se - (b - 1.0) / (1.0 - se)
            end

            # A pinned specificity contributes no prior term at all: the engine
            # does not sample it, so there is nothing to score.
            slot = layout.sp_slot[k]
            if slot != 0
                sp = Sp[k]
                zsp = theta[slot]
                lp += accuracy_logpdf_z(sp_pr[k], zsp, sp, jacobian)
                if sp_pr[k].family == ACC_LOGITNORMAL
                    g[slot] += accuracy_dlogpdf_dz(sp_pr[k], zsp, sp, jacobian)
                else
                    c, d = sp_pr[k].p, sp_pr[k].q
                    spbar[k] += jacobian ? c / sp - d / (1.0 - sp) :
                                           (c - 1.0) / sp - (d - 1.0) / (1.0 - sp)
                end
            end
        end
    end

    # The penalty applies whether or not Se/Sp are inferred (matching the
    # package). When they are FIXED it is a constant: it enters lp but its
    # derivative goes nowhere, since Se/Sp are not parameters.
    if penalty
        # -w * log1pexp(-(Se+Sp-1)/s); d/dx log1pexp(x) = logistic(x), so
        # with x = -(Se+Sp-1)/s the chain rule gives +w/s * logistic(x)
        # on each of Se and Sp.
        @inbounds for k in 1:N_TESTS
            x = -(Se[k] + Sp[k] - 1.0) / PENALTY_SCALE
            lp -= PENALTY_WEIGHT * log1pexp(x)
            if infer
                dpen = PENALTY_WEIGHT * logistic(x) / PENALTY_SCALE
                sebar[k] += dpen
                # A pinned Sp is a constant, so the penalty still shifts lp
                # through it but carries no derivative. Its Se partner does
                # still get one.
                layout.sp_slot[k] == 0 || (spbar[k] += dpen)
            end
        end
    end

    # --- pi1 per year, with its partials -----------------------------------
    # pi1_y = clamp(logistic(pi1_0 + pi1_mult*gamma_y)); clamping zeroes the
    # derivative outside the bounds, matching clamp_prob in the primal.
    pi1_vec = Vector{Float64}(undef, n_years)
    dpi1_d0 = Vector{Float64}(undef, n_years)      # d pi1_y / d pi1_0
    dpi1_dmult = Vector{Float64}(undef, n_years)   # d pi1_y / d pi1_mult
    dpi1_dgamma = Vector{Float64}(undef, n_years)  # d pi1_y / d gamma_y
    @inbounds for y in 1:n_years
        z = pi1_0 + pi1_mult * gamma[y]
        s = 1.0 / (1.0 + exp(-z))
        if s < 1e-9
            pi1_vec[y] = 1e-9; dpi1_d0[y] = 0.0
            dpi1_dmult[y] = 0.0; dpi1_dgamma[y] = 0.0
        elseif s > 1.0 - 1e-9
            pi1_vec[y] = 1.0 - 1e-9; dpi1_d0[y] = 0.0
            dpi1_dmult[y] = 0.0; dpi1_dgamma[y] = 0.0
        else
            pi1_vec[y] = s
            ds = s * (1.0 - s)
            dpi1_d0[y] = ds
            dpi1_dmult[y] = ds * gamma[y]
            dpi1_dgamma[y] = ds * pi1_mult
        end
    end

    # Accumulators in gamma-space; converted to (raw, log_sigma) at the end.
    gbar_gamma = zeros(n_years)
    gbar_alpha = zeros(S)
    gbar_pi1_0 = 0.0
    gbar_pi1_mult = 0.0

    # --- per-badger forward sweep, storing what the reverse pass needs ------
    nmax = 0
    @inbounds for i in 1:data.n_ind
        lo = i == 1 ? 1 : data.seq_ends[i - 1] + 1
        nmax = max(nmax, data.seq_ends[i] - lo + 1)
    end
    # --- three-state model: tau, rho, multipliers ---------------------------
    # tau and rho are clamped like the other probabilities (zero derivative
    # when clamped). LogitNormal prior derivatives are on the logit scale so
    # they go straight into `g`.
    sei = layout.sei
    tau = 0.0
    rho = 0.0
    dtau_dz = 0.0
    drho_dz = 0.0
    taubar = 0.0
    rhobar = 0.0
    if sei
        ztau = theta[layout.i_tau]
        zrho = theta[layout.i_rho]
        ptau = logistic(ztau)
        prho = logistic(zrho)
        lp += accuracy_logpdf_z(SEI_TAU_PRIOR, ztau, ptau, jacobian) +
              accuracy_logpdf_z(SEI_RHO_PRIOR, zrho, prho, jacobian)
        g[layout.i_tau] += accuracy_dlogpdf_dz(SEI_TAU_PRIOR, ztau, ptau, jacobian)
        g[layout.i_rho] += accuracy_dlogpdf_dz(SEI_RHO_PRIOR, zrho, prho, jacobian)
        tau = clamp_prob(ptau)
        rho = clamp_prob(prho)
        dtau_dz = tau == ptau ? ptau * (1.0 - ptau) : 0.0
        drho_dz = rho == prho ? prho * (1.0 - prho) : 0.0
    end
    # multipliers on the natural scale and their adjoints (pinned ones are
    # constants: no slot, no prior)
    em = Vector{Float64}(undef, sei ? N_TESTS : 0)
    embar = zeros(sei ? N_TESTS : 0)
    if sei
        @inbounds for k in 1:N_TESTS
            slot = layout.e_slot[k]
            if slot == 0
                em[k] = layout.e_pin[k]
            else
                zm = theta[slot]
                m = logistic(zm)
                em[k] = m
                lp += accuracy_logpdf_z(SEI_MULT_PRIOR, zm, m, jacobian)
                g[slot] += accuracy_dlogpdf_dz(SEI_MULT_PRIOR, zm, m, jacobian)
                # penalty on m*Se + Sp > 1 (see logposterior):
                # -w * log1pexp(-(m Se + Sp - 1)/s), derivative w/s * logistic(x)
                # times Se for m, m for Se, 1 for Sp
                if penalty
                    x = -(m * Se[k] + Sp[k] - 1.0) / PENALTY_SCALE
                    lp -= PENALTY_WEIGHT * log1pexp(x)
                    dpen = PENALTY_WEIGHT * logistic(x) / PENALTY_SCALE
                    embar[k] += dpen * Se[k]
                    sebar[k] += dpen * m
                    spbar[k] += dpen
                end
            end
        end
    end
    work = SEIWork(sei ? nmax : 0)

    u_pre = Vector{Float64}(undef, nmax)   # u before the transition at step j
    v_pre = Vector{Float64}(undef, nmax)
    lam_j = Vector{Float64}(undef, nmax)   # hazard applied entering step j
    bU_j = Vector{Float64}(undef, nmax)
    bI_j = Vector{Float64}(undef, nmax)
    c_j = Vector{Float64}(undef, nmax)
    has_obs = Vector{Bool}(undef, nmax)
    yr_j = Vector{Int}(undef, nmax)
    sn_j = Vector{Int}(undef, nmax)

    start = 1
    @inbounds for i in 1:data.n_ind
        e = data.seq_ends[i]
        n = e - start + 1
        ey = data.entry_year[i]
        # born badgers start susceptible, so no pi1 gradient from them
        born = data.born[i]
        pi1 = born ? 0.0 : pi1_vec[ey]

        if sei
            # three-state sweep is in sei.jl
            ll_i, pibar_i, rhobar_i, taubar_i =
                sei_seq_sweep!(work, data, start, e, pi1, rho, tau, alpha, gamma,
                               Se, Sp, em, S,
                               gbar_alpha, gbar_gamma, sebar, spbar, embar)
            lp += ll_i
            rhobar += rhobar_i
            taubar += taubar_i
            if !born
                gbar_pi1_0 += pibar_i * dpi1_d0[ey]
                gbar_pi1_mult += pibar_i * dpi1_dmult[ey]
                gbar_gamma[ey] += pibar_i * dpi1_dgamma[ey]
            end
            start = e + 1
            continue
        end

        u = 1.0 - pi1
        v = pi1
        for j in 1:n
            gj = start + j - 1
            u_pre[j] = u
            v_pre[j] = v
            if j > 1
                t = data.times[gj]
                sn = season_of(t, S)
                yr = year_of(t, S)
                sn_j[j] = sn
                yr_j[j] = yr
                z = alpha[sn] + gamma[yr]
                s = 1.0 / (1.0 + exp(-z))
                lam = s < 1e-9 ? 1e-9 : (s > 1.0 - 1e-9 ? 1.0 - 1e-9 : s)
                lam_j[j] = lam
                v = v + u * lam
                u = u * (1.0 - lam)
            else
                lam_j[j] = 0.0
                sn_j[j] = 0
                yr_j[j] = 0
            end

            plo = data.ptr[gj]
            phi = data.ptr[gj + 1] - 1
            if phi >= plo
                bU = 1.0
                bI = 1.0
                for p in plo:phi
                    k = data.tidx[p]
                    if data.vals[p] == 1
                        bU *= 1.0 - Sp[k]
                        bI *= Se[k]
                    else
                        bU *= Sp[k]
                        bI *= 1.0 - Se[k]
                    end
                end
                bU_j[j] = bU
                bI_j[j] = bI
                u *= bU
                v *= bI
                c = u + v
                c_j[j] = c
                lp += log(c)
                u /= c
                v /= c
                has_obs[j] = true
            else
                bU_j[j] = 1.0
                bI_j[j] = 1.0
                c_j[j] = 1.0
                has_obs[j] = false
            end
        end

        # --- reverse sweep for this badger ---------------------------------
        # Adjoints of (u, v) as they stand AFTER step j's rescale.
        ubar = 0.0
        vbar = 0.0
        for j in n:-1:1
            # undo: u <- u_em/c, v <- v_em/c, ll += log(c), with c = u_em+v_em
            if has_obs[j]
                c = c_j[j]
                # u_out = u_em/c and v_out = v_em/c, so with cbar from both the
                # normalisation and the log(c) term:
                #   d/du_em = ubar/c - (ubar*u_out + vbar*v_out)/c + 1/c
                uem = u_pre[j]  # placeholder, recomputed below
                # Recompute the pre-rescale values from the stored state.
                # u_em = u_after_transition * bU, v_em = v_after_transition * bI
                # We stored u_pre/v_pre (before transition), so redo it:
                if j > 1
                    lam = lam_j[j]
                    vt = v_pre[j] + u_pre[j] * lam
                    ut = u_pre[j] * (1.0 - lam)
                else
                    ut = u_pre[j]
                    vt = v_pre[j]
                end
                uem = ut * bU_j[j]
                vem = vt * bI_j[j]
                uo = uem / c
                vo = vem / c
                # cbar collects the 1/c from log(c) and the -x/c^2 from dividing
                cbar = 1.0 / c - (ubar * uo + vbar * vo) / c
                ubar_em = ubar / c + cbar
                vbar_em = vbar / c + cbar
                # Se/Sp enter only here, multiplicatively inside bU_j/bI_j.
                # bUbar = d(lp)/d(bU_j) = ubar_em * ut, and each assay's own
                # factor f divides out of the product: d bU_j / d f = bU_j / f.
                if infer
                    bUbar = ubar_em * ut
                    bIbar = vbar_em * vt
                    plo = data.ptr[start + j - 1]
                    phi = data.ptr[start + j] - 1
                    for pp in plo:phi
                        k = data.tidx[pp]
                        if data.vals[pp] == 1
                            # bU factor (1 - Sp[k]): d/dSp = -1
                            spbar[k] -= bUbar * bU_j[j] / (1.0 - Sp[k])
                            # bI factor Se[k]: d/dSe = +1
                            sebar[k] += bIbar * bI_j[j] / Se[k]
                        else
                            # bU factor Sp[k]
                            spbar[k] += bUbar * bU_j[j] / Sp[k]
                            # bI factor (1 - Se[k])
                            sebar[k] -= bIbar * bI_j[j] / (1.0 - Se[k])
                        end
                    end
                end
                ubar = ubar_em * bU_j[j]
                vbar = vbar_em * bI_j[j]
            end

            # undo the transition: v = v + u*lam ; u = u*(1-lam)
            if j > 1
                lam = lam_j[j]
                up = u_pre[j]
                # d/dlam of (v + u*lam, u*(1-lam)) contracted with (vbar, ubar)
                lam_bar = vbar * up - ubar * up
                # only accumulate if the hazard was not clamped flat
                if lam > 1e-9 && lam < 1.0 - 1e-9
                    dlam_dz = lam * (1.0 - lam)
                    zbar = lam_bar * dlam_dz
                    gbar_alpha[sn_j[j]] += zbar
                    gbar_gamma[yr_j[j]] += zbar
                end
                new_ubar = ubar * (1.0 - lam) + vbar * lam
                ubar = new_ubar
                # vbar passes through unchanged (v enters v' = v + u*lam)
            end
        end

        # Seed at j=1: u = 1-pi1, v = pi1  =>  d/dpi1 = vbar - ubar
        pibar = vbar - ubar
        if !born
            gbar_pi1_0 += pibar * dpi1_d0[ey]
            gbar_pi1_mult += pibar * dpi1_dmult[ey]
            gbar_gamma[ey] += pibar * dpi1_dgamma[ey]
        end

        start = e + 1
    end

    # --- data-adaptive year-increment prior --------------------------------
    # logpdf(Normal(0, sd_y), gamma[y] - gamma[y-1]) for each adjacent pair.
    # `sd_y` depends only on the tested-capture counts, so it is a CONSTANT
    # here: the whole contribution reaches the parameters through gamma, which
    # is why this has to land in gbar_gamma BEFORE the year-process unwind
    # below rather than going into `g` directly.
    #
    #   d/d(gamma[y])   = -(d/sd^2)   where d = gamma[y] - gamma[y-1]
    #   d/d(gamma[y-1]) = +(d/sd^2)
    if !isempty(year_weight)
        yw = year_weight
        @inbounds for y in 2:n_years
            n_eff = min(yw[y], yw[y - 1])
            sd_y = year_link_sd(n_eff)
            d = gamma[y] - gamma[y - 1]
            lp += -0.5 * (d / sd_y)^2 - log(sd_y) - 0.9189385332046727
            w = d / (sd_y * sd_y)
            gbar_gamma[y]     -= w
            gbar_gamma[y - 1] += w
        end
    end

    # --- fold the likelihood adjoints into g -------------------------------
    @inbounds for s in 1:S
        g[layout.i_alpha + s - 1] += gbar_alpha[s]
    end
    g[layout.i_pi1_0] += gbar_pi1_0
    g[layout.i_pi1_mult] += gbar_pi1_mult
    if sei
        g[layout.i_tau] += taubar * dtau_dz
        g[layout.i_rho] += rhobar * drho_dz
        @inbounds for k in 1:N_TESTS
            slot = layout.e_slot[k]
            slot == 0 && continue
            g[slot] += embar[k] * em[k] * (1.0 - em[k])
        end
    end

    # Fold the gamma adjoints back onto (raw, log_sigma).
    #
    #   iid  gamma_y = sigma_g * raw_y
    #          d/draw_y = sigma_g * gbar_y
    #          d/dlog_sigma += gbar_y * raw_y * sigma_g
    #
    #   rw1  gamma_y = sigma_g * sum_{k<=y} raw_k, so raw_k feeds EVERY gamma
    #        from y=k onward:
    #          d/draw_k = sigma_g * sum_{y>=k} gbar_y   -- a reverse cumsum
    #          d/dlog_sigma += sum_y gbar_y * gamma_y / sigma_g
    #        (gamma_y / sigma_g is that year's accumulated raw, so this is the
    #         same "gbar . dgamma/dlog_sigma" contraction as the iid case.)
    #   none    gamma_y = 0                       -> no raw or sigma partials
    #   iid      gamma_y = sigma_g * raw_y
    #   shrunk   same as iid (only the sigma prior differs)
    #   rw1      gamma_y = sigma_g * sum_{k<=y} raw_k
    #              raw_k feeds every gamma from y=k on, so its adjoint is a
    #              REVERSE cumulative sum of the gamma adjoints.
    #   rw2      gamma_y = sigma_g * sum_{j<=y} sum_{k<=j} raw_k, so raw_k
    #              appears in gamma_y with weight (y - k + 1). Its adjoint is
    #              therefore the reverse cumsum OF THE REVERSE CUMSUM: running
    #              `suffix` backwards accumulates sum_{y>=k} gbar_y, and running
    #              `suffix2` over that accumulates the weighted version.
    #
    # In every case dlogsigma contracts gbar against d(gamma)/d(log sigma),
    # which is just gamma itself (gamma is linear in sigma_g), so the one
    # expression covers all of them.
    if proc == PROC_NONE
        # gamma is identically zero: nothing flows back to raw or sigma_g.
    elseif proc == PROC_RW1
        suffix = 0.0
        @inbounds for y in n_years:-1:1
            suffix += gbar_gamma[y]
            g[layout.i_gamma + y - 1] += suffix * sigma_g
        end
    elseif proc == PROC_RW1C
        # gamma = C * w, where w is the rw1 walk and C = I - (1/n)11' centres
        # it. C is SYMMETRIC, so the adjoint of the centring step is the same
        # operation on the adjoint vector: wbar = gbar - mean(gbar). Centring is
        # applied LAST in the forward direction, so it is undone FIRST here,
        # and only then does the reverse-cumsum of the rw1 walk run.
        #
        # Consequence worth noting: a constant shift in gbar contributes
        # nothing, which is the gradient-side statement of the level being
        # unidentified -- exactly what the constraint exists to fix.
        mean_gbar = 0.0
        @inbounds for y in 1:n_years
            mean_gbar += gbar_gamma[y]
        end
        mean_gbar /= n_years
        suffix = 0.0
        @inbounds for y in n_years:-1:1
            suffix += gbar_gamma[y] - mean_gbar
            g[layout.i_gamma + y - 1] += suffix * sigma_g
        end
    elseif proc == PROC_RW2
        suffix = 0.0
        suffix2 = 0.0
        @inbounds for y in n_years:-1:1
            suffix += gbar_gamma[y]
            suffix2 += suffix
            g[layout.i_gamma + y - 1] += suffix2 * sigma_g
        end
    else                                   # iid and shrunk
        @inbounds for y in 1:n_years
            g[layout.i_gamma + y - 1] += gbar_gamma[y] * sigma_g
        end
    end
    # gamma is LINEAR in sigma_g under every process (rw1c included -- centring
    # is a linear map, so it commutes with the scalar), hence
    # d(gamma)/d(log sigma) = gamma and the contraction is the same expression
    # for all of them.
    if proc != PROC_NONE
        @inbounds for y in 1:n_years
            dlogsigma += gbar_gamma[y] * gamma[y]
        end
    end
    g[layout.i_logsigma] += dlogsigma

    # Se_k = logistic(z_k): dSe/dz = Se*(1-Se). The natural-scale adjoints
    # (likelihood, penalty, and the Beta priors) are converted here in one
    # place. LogitNormal prior terms are NOT in sebar/spbar -- they were added
    # to `g` directly above, already on the logit scale.
    if infer
        @inbounds for k in 1:N_TESTS
            g[layout.i_se + k - 1] += sebar[k] * Se[k] * (1.0 - Se[k])
            slot = layout.sp_slot[k]
            slot == 0 && continue          # pinned: constant, no gradient slot
            g[slot] += spbar[k] * Sp[k] * (1.0 - Sp[k])
        end
    end

    return lp
end

"""
Convenience form for the fixed-Se/Sp model. The Beta priors are unused there,
but `penalty` defaults to `true` to match both `logposterior`'s callers and the
package: with Se/Sp fixed the penalty is a constant, so it changes the returned
log density but not the gradient.
"""
function logposterior_grad!(g::Vector{Float64}, theta::Vector{Float64},
                            data::HMMData, layout::ParamLayout,
                            penalty::Bool = true)
    layout.infer_sesp &&
        error("the short form is for the fixed-Se/Sp model; pass se_pr, sp_pr, penalty")
    se_pr, sp_pr = default_accuracy_priors(ACC_BETA)
    return logposterior_grad!(g, theta, data, layout, se_pr, sp_pr, penalty,
                              EMPTY_YW, true)
end

# Back-compatible entry point for callers holding raw Beta (a,b) pairs.
function logposterior_grad!(g::Vector{Float64}, theta::Vector{Float64},
                            data::HMMData, layout::ParamLayout,
                            se_ab::NTuple{N_TESTS,Tuple{Float64,Float64}},
                            sp_ab::NTuple{N_TESTS,Tuple{Float64,Float64}},
                            penalty::Bool)
    se_pr = ntuple(k -> AccuracyPrior(ACC_BETA, se_ab[k][1], se_ab[k][2]), N_TESTS)
    sp_pr = ntuple(k -> AccuracyPrior(ACC_BETA, sp_ab[k][1], sp_ab[k][2]), N_TESTS)
    return logposterior_grad!(g, theta, data, layout, se_pr, sp_pr, penalty,
                              EMPTY_YW, true)
end
