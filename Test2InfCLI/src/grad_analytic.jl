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
                            se_ab::NTuple{N_TESTS,Tuple{Float64,Float64}},
                            sp_ab::NTuple{N_TESTS,Tuple{Float64,Float64}},
                            penalty::Bool)
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
            Sp[k] = logistic(theta[layout.i_sp + k - 1])
        end
    else
        @inbounds for k in 1:N_TESTS
            Se[k] = SE_FIXED_DEFAULT[k]
            Sp[k] = SP_FIXED_DEFAULT[k]
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

    # sigma_g ~ Normal+(0, sd), parameterised as log_sigma_g. The sd depends on
    # the year process (0.05 for rw1, 0.30 for iid), matching test2infeR.
    # d/dlog_sigma [ -sigma^2/(2 sd^2) + log_sigma ] = -sigma^2/sd^2 + 1
    sig_sd = sigma_prior_sd(layout.year_process)
    lp += -0.5 * (sigma_g / sig_sd)^2 - log(sig_sd) -
          0.9189385332046727 + log_sigma
    dlogsigma = -(sigma_g^2) / sig_sd^2 + 1.0

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
    # Se_k = logistic(z), so the Beta kernel plus the logit log|J| is
    #   (a-1)log(p) + (b-1)log(1-p) + log(p) + log(1-p)
    #     = a*log(p) + b*log(1-p),
    # whose derivative wrt p is a/p - b/(1-p). Accumulated on the natural
    # scale here, exactly like the likelihood's contribution, and converted to
    # the logit scale in one place at the end.
    if infer
        @inbounds for k in 1:N_TESTS
            se = Se[k]
            sp = Sp[k]
            a, b = se_ab[k]
            c, d = sp_ab[k]
            lp += (a - 1.0) * log(se) + (b - 1.0) * log(1.0 - se) +
                  log(se) + log(1.0 - se)
            lp += (c - 1.0) * log(sp) + (d - 1.0) * log(1.0 - sp) +
                  log(sp) + log(1.0 - sp)
            sebar[k] += a / se - b / (1.0 - se)
            spbar[k] += c / sp - d / (1.0 - sp)
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
                spbar[k] += dpen
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
        pi1 = pi1_vec[ey]

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
        gbar_pi1_0 += pibar * dpi1_d0[ey]
        gbar_pi1_mult += pibar * dpi1_dmult[ey]
        gbar_gamma[ey] += pibar * dpi1_dgamma[ey]

        start = e + 1
    end

    # --- fold the likelihood adjoints into g -------------------------------
    @inbounds for s in 1:S
        g[layout.i_alpha + s - 1] += gbar_alpha[s]
    end
    g[layout.i_pi1_0] += gbar_pi1_0
    g[layout.i_pi1_mult] += gbar_pi1_mult

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
    if proc != PROC_NONE
        @inbounds for y in 1:n_years
            dlogsigma += gbar_gamma[y] * gamma[y]
        end
    end
    g[layout.i_logsigma] += dlogsigma

    # Se_k = logistic(z_k): dSe/dz = Se*(1-Se). Everything above accumulated on
    # the natural scale, so one Jacobian factor converts the lot.
    if infer
        @inbounds for k in 1:N_TESTS
            g[layout.i_se + k - 1] += sebar[k] * Se[k] * (1.0 - Se[k])
            g[layout.i_sp + k - 1] += spbar[k] * Sp[k] * (1.0 - Sp[k])
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
        error("the short form is for the fixed-Se/Sp model; pass se_ab, sp_ab, penalty")
    se_ab, sp_ab = default_sesp_priors()
    return logposterior_grad!(g, theta, data, layout, se_ab, sp_ab, penalty)
end
