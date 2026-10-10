# Correctness checks for the standalone CLI kernel.
#
# Run: julia --project=. test/runtests.jl   (needs Julia 1.12+ only for the
# trim build; the tests themselves run on any recent Julia)

using Test
include(joinpath(@__DIR__, "..", "Test2InfCLI", "src", "Test2InfCLI.jl"))
const M = Test2InfCLI

const DATA = joinpath(@__DIR__, "..", "testdata", "sim.csv")
# A second fixture matching the real cohort's sparsity (78% NaN cells, 51% of
# timepoints with no test at all). The dense fixture alone would not exercise
# the empty-emission branch of the recursion at a realistic rate.
const SPARSE = joinpath(@__DIR__, "..", "testdata", "sparse.csv")

@testset "Test2InfCLI" begin
    mat = M.read_matrix(DATA)
    data = M.build_data(mat, fill(true, 6), :stack, 4)
    se_ab, sp_ab = M.default_sesp_priors()
    # The same Beta priors in the AccuracyPrior form the newer entry points
    # take, so tests can exercise the full signature without restating them.
    se_pr_b, sp_pr_b = M.default_accuracy_priors(M.ACC_BETA)

    @testset "data shape" begin
        @test data.n_ind > 0
        @test length(data.ptr) == length(data.times) + 1
        @test data.ptr[end] == length(data.vals) + 1
        @test length(data.vals) == length(data.tidx)
        @test all(1 .<= data.tidx .<= 6)
        @test all(v -> v == 0 || v == 1, data.vals)
    end

    for infer in (false, true)
        layout = M.ParamLayout(data.S, data.n_years, infer)
        logp = th -> M.logposterior(th, data, layout, se_ab, sp_ab, true)

        @testset "gradient vs finite differences (infer_sesp=$infer)" begin
            for trial in 1:3
                th = M.initial_theta(layout) .+ 0.13 * (trial - 1) .+ 0.05
                g = zeros(layout.n)
                M.gradient!(g, logp, th, Val(13))
                maxrel = 0.0
                for i in 1:layout.n
                    h = 1e-6 * max(1.0, abs(th[i]))
                    tp = copy(th); tp[i] += h
                    tm = copy(th); tm[i] -= h
                    fd = (logp(tp) - logp(tm)) / (2h)
                    maxrel = max(maxrel, abs(fd - g[i]) / max(1.0, abs(fd)))
                end
                @test maxrel < 1e-5
            end
        end
    end

    @testset "analytic gradient matches AD (sparse data)" begin
        smat = M.read_matrix(SPARSE)
        sdata = M.build_data(smat, fill(true, 6), :stack, 4)
        # Confirm the fixture really is sparse, so this test keeps its meaning
        # if the generator is ever regenerated.
        n_empty = count(j -> sdata.ptr[j+1] == sdata.ptr[j], 1:length(sdata.times))
        @test n_empty / length(sdata.times) > 0.4
        slayout = M.ParamLayout(sdata.S, sdata.n_years, false)
        slogp = th -> M.logposterior(th, sdata, slayout, se_ab, sp_ab, true)
        for trial in 1:4
            th = M.initial_theta(slayout) .+ 0.13 * (trial - 1) .+ 0.05
            ga = zeros(slayout.n)
            la = M.logposterior_grad!(ga, th, sdata, slayout)
            gf = zeros(slayout.n)
            lf = M.gradient!(gf, slogp, th, Val(13))
            @test isapprox(la, lf; atol = 1e-8)
            @test maximum(abs.(ga .- gf)) < 1e-10
        end
    end

    @testset "analytic gradient matches AD" begin
        layout = M.ParamLayout(data.S, data.n_years, false)
        logp = th -> M.logposterior(th, data, layout, se_ab, sp_ab, true)
        for trial in 1:5
            th = M.initial_theta(layout) .+ 0.11 * (trial - 1) .+ 0.05
            ga = zeros(layout.n)
            la = M.logposterior_grad!(ga, th, data, layout)
            gf = zeros(layout.n)
            lf = M.gradient!(gf, logp, th, Val(13))
            # Same quantity computed two independent ways: agreement should be
            # at machine precision, not merely close.
            @test isapprox(la, lf; atol = 1e-8)
            @test maximum(abs.(ga .- gf)) < 1e-10
        end
    end

    @testset "analytic gradient across the sql-e2e fixed panels" begin
        # The three fixed-Se/Sp variants in sql e2e/scripts/02_fit_and_plot.R
        # differ only by test mask. A mask changes which emission terms enter
        # the product, not the derivative structure, so one derivation covers
        # all three -- this asserts that rather than assuming it.
        smat = M.read_matrix(SPARSE)
        panels = ("full" => fill(true, 6),
                  "no_dpp" => [true, true, false, true, true, true],
                  "culture_only" => [false, false, false, true, false, false])
        for (nm, mask) in panels
            pdata = M.build_data(smat, collect(mask), :stack, 4)
            play = M.ParamLayout(pdata.S, pdata.n_years, false)
            plogp = th -> M.logposterior(th, pdata, play, se_ab, sp_ab, true)
            for trial in 1:3
                th = M.initial_theta(play) .+ 0.13 * (trial - 1) .+ 0.05
                ga = zeros(play.n)
                la = M.logposterior_grad!(ga, th, pdata, play)
                gf = zeros(play.n)
                lf = M.gradient!(gf, plogp, th, Val(13))
                @test isapprox(la, lf; atol = 1e-8)
                @test maximum(abs.(ga .- gf)) < 1e-10
            end
        end
    end

    @testset "analytic gradient across the sql-e2e inferred panels" begin
        # The two inferred-Se/Sp variants (all tests, no DPP). Se[k]/Sp[k] enter
        # every emission factor multiplicatively, so their adjoints accumulate
        # per assay across every capture -- a genuinely different derivation
        # from the fixed case, checked on both fixtures and both penalty
        # settings. 25 parameters, so the AD reference needs Val(25).
        for fixture in (DATA, SPARSE)
            fmat = M.read_matrix(fixture)
            panels = ("full" => fill(true, 6),
                      "no_dpp" => [true, true, false, true, true, true])
            for (nm, mask) in panels, pen in (true, false)
                pdata = M.build_data(fmat, collect(mask), :stack, 4)
                play = M.ParamLayout(pdata.S, pdata.n_years, true)
                @test play.n == play.i_sp + M.N_TESTS - 1
                plogp = th -> M.logposterior(th, pdata, play, se_ab, sp_ab, pen)
                for trial in 1:4
                    th = M.initial_theta(play) .+ 0.13 * (trial - 1) .+ 0.05
                    ga = zeros(play.n)
                    la = M.logposterior_grad!(ga, th, pdata, play, se_ab, sp_ab, pen)
                    gf = zeros(play.n)
                    lf = M.gradient!(gf, plogp, th, Val(25))
                    @test isapprox(la, lf; atol = 1e-8)
                    @test maximum(abs.(ga .- gf)) < 1e-10
                end
            end
        end
    end

    @testset "inferred gradient where the Se+Sp>1 penalty actually bites" begin
        # At the initial values Se+Sp ~ 1.35, so the soft constraint is nearly
        # flat and contributes about -0.35 -- it would pass the tests above even
        # if its derivative were wrong. Push Se+Sp below 1, where the penalty is
        # worth thousands of log units, and check the gradient there too.
        layout = M.ParamLayout(data.S, data.n_years, true)
        th = M.initial_theta(layout)
        for k in 1:M.N_TESTS
            th[layout.i_se + k - 1] = -3.0
            th[layout.i_sp + k - 1] = -1.0
        end
        @test all(k -> M.logistic(th[layout.i_se + k - 1]) +
                       M.logistic(th[layout.i_sp + k - 1]) < 1.0, 1:M.N_TESTS)
        lp_pen = M.logposterior(th, data, layout, se_ab, sp_ab, true)
        lp_off = M.logposterior(th, data, layout, se_ab, sp_ab, false)
        @test lp_off - lp_pen > 100.0   # the penalty is genuinely active here
        for pen in (true, false)
            plogp = t -> M.logposterior(t, data, layout, se_ab, sp_ab, pen)
            ga = zeros(layout.n)
            la = M.logposterior_grad!(ga, th, data, layout, se_ab, sp_ab, pen)
            gf = zeros(layout.n)
            lf = M.gradient!(gf, plogp, th, Val(25))
            @test isapprox(la, lf; rtol = 1e-9)
            # Absolute scale here is ~1e4, so compare relatively.
            @test maximum(abs.(ga .- gf)) < 1e-8 * max(1.0, maximum(abs.(gf)))
        end
    end

    @testset "inferred MAP converges to the same optimum from both gradients" begin
        layout = M.ParamLayout(data.S, data.n_years, true)
        logp = th -> M.logposterior(th, data, layout, se_ab, sp_ab, true)
        th0 = M.initial_theta(layout)
        mk = gf -> (g, x) -> begin
            fx = gf(g, x)
            @inbounds for i in eachindex(g); g[i] = -g[i]; end
            return -fx
        end
        r_ad = M.lbfgs(mk((g, x) -> M.gradient!(g, logp, x, Val(25))), copy(th0))
        r_an = M.lbfgs(mk((g, x) -> M.logposterior_grad!(g, x, data, layout,
                                                        se_ab, sp_ab, true)), copy(th0))
        @test r_ad.converged
        @test r_an.converged
        @test isapprox(r_ad.logp, r_an.logp; atol = 1e-6)
        @test maximum(abs.(r_ad.theta .- r_an.theta)) < 1e-4
    end

    @testset "the 3-argument gradient form still rejects inferred Se/Sp" begin
        layout = M.ParamLayout(data.S, data.n_years, true)
        th = M.initial_theta(layout)
        @test_throws ErrorException M.logposterior_grad!(zeros(layout.n), th, data, layout)
    end

    @testset "MAP converges to the same optimum from both gradients" begin
        layout = M.ParamLayout(data.S, data.n_years, false)
        logp = th -> M.logposterior(th, data, layout, se_ab, sp_ab, true)
        th0 = M.initial_theta(layout)

        mk = gf -> (g, x) -> begin
            fx = gf(g, x)
            @inbounds for i in eachindex(g); g[i] = -g[i]; end
            return -fx
        end
        r_ad = M.lbfgs(mk((g, x) -> M.gradient!(g, logp, x, Val(13))), copy(th0))
        r_an = M.lbfgs(mk((g, x) -> M.logposterior_grad!(g, x, data, layout)), copy(th0))

        @test r_ad.converged
        @test r_an.converged
        @test isapprox(r_ad.logp, r_an.logp; atol = 1e-6)
        @test maximum(abs.(r_ad.theta .- r_an.theta)) < 1e-4
    end

    @testset "each badger is expanded onto the full timestep grid" begin
        # An uncaptured timestep carries no emission but still advances the
        # hazard by one lam, so a badger seen at t=1 and t=5 must contribute
        # four transitions, not one. Stepping capture-to-capture is a different
        # model: on the real cohort it gives 14,986 transitions instead of
        # 29,737 and shifts the log density by ~174.
        #
        # Two inputs describing the SAME badgers -- one with only capture rows,
        # one pre-expanded with captured==0 filler -- must therefore agree
        # exactly. Before this was fixed the whole suite passed while the CLI
        # silently computed the capture-only model on capture-only input.
        caps = Float64[
            1 1 1 1 0 0 0 0 0
            5 1 1 0 1 0 0 0 0
            2 2 1 1 1 0 0 0 0
            4 2 1 0 0 1 0 0 0
        ]
        nan6 = fill(NaN, 6)
        grid = vcat(
            reshape([1.0, 1, 1, 1, 0, 0, 0, 0, 0], 1, 9),
            reshape(vcat([2.0, 1, 0], nan6), 1, 9),
            reshape(vcat([3.0, 1, 0], nan6), 1, 9),
            reshape(vcat([4.0, 1, 0], nan6), 1, 9),
            reshape([5.0, 1, 1, 0, 1, 0, 0, 0, 0], 1, 9),
            reshape([2.0, 2, 1, 1, 1, 0, 0, 0, 0], 1, 9),
            reshape(vcat([3.0, 2, 0], nan6), 1, 9),
            reshape([4.0, 2, 1, 0, 0, 1, 0, 0, 0], 1, 9),
        )
        dc = M.build_data(caps, fill(true, 6), :stack, 4)
        dg = M.build_data(grid, fill(true, 6), :stack, 4)

        # Badger 1 spans t=1..5 (5 steps), badger 2 spans t=2..4 (3 steps).
        @test length(dc.times) == 8
        @test dc.times == dg.times
        @test dc.ptr == dg.ptr
        @test dc.vals == dg.vals
        @test dc.tidx == dg.tidx
        @test dc.seq_ends == dg.seq_ends
        @test dc.entry_year == dg.entry_year
        # Filler steps carry no emissions.
        @test length(dc.vals) == length(dg.vals)

        lay = M.ParamLayout(dc.S, dc.n_years, false)
        th = M.initial_theta(lay) .+ 0.07
        @test isapprox(M.logposterior(th, dc, lay, se_ab, sp_ab, true),
                       M.logposterior(th, dg, lay, se_ab, sp_ab, true); atol = 1e-12)
        gc = zeros(lay.n); M.logposterior_grad!(gc, th, dc, lay, se_ab, sp_ab, true)
        gg = zeros(lay.n); M.logposterior_grad!(gg, th, dg, lay, se_ab, sp_ab, true)
        @test maximum(abs.(gc .- gg)) < 1e-12

        # And the gap really does matter: a capture-only recursion over the same
        # badgers would use fewer transitions and give a different answer.
        @test length(dc.times) > 4   # 4 capture rows, 8 grid steps
    end

    @testset "posterior decoding is a probability" begin
        layout = M.ParamLayout(data.S, data.n_years, false)
        th = M.initial_theta(layout)
        P = M.extract_params(th, layout)
        p_inf = M.decode(data, P)
        @test length(p_inf) == length(data.times)
        @test all(0.0 .<= p_inf .<= 1.0)
    end

    @testset "dense-metric linear algebra (hand-rolled, no BLAS)" begin
        # The shipped bundle has LAPACK removed, so Cholesky, the momentum draw
        # and the dense kinetic energy are all hand-written. Each is checked
        # against an independent computation rather than assumed.
        A = [4.0 2.0 0.6; 2.0 5.0 1.0; 0.6 1.0 3.0]
        Lc = zeros(3, 3)
        @test M.chol_lower!(Lc, A)
        rec = [sum(Lc[i, k] * Lc[j, k] for k in 1:3) for i in 1:3, j in 1:3]
        @test maximum(abs.(rec .- A)) < 1e-12
        @test Lc[1, 2] == 0.0 && Lc[1, 3] == 0.0 && Lc[2, 3] == 0.0
        @test !M.chol_lower!(zeros(2, 2), [1.0 2.0; 2.0 1.0])   # not PD

        # Momentum p ~ N(0, M) with M = inv(inv_mass): the empirical covariance
        # must approach inv(A), NOT A. Getting this backwards still runs and
        # silently samples the wrong distribution, so it is checked directly.
        rng = M.Rng(7)
        z = zeros(3); pv = zeros(3); S3 = zeros(3, 3)
        N = 200_000
        for _ in 1:N
            M.sample_momentum!(pv, Lc, z, rng)
            for i in 1:3, j in 1:3
                S3[i, j] += pv[i] * pv[j]
            end
        end
        S3 ./= N
        det = A[1,1]*(A[2,2]*A[3,3]-A[2,3]*A[3,2]) -
              A[1,2]*(A[2,1]*A[3,3]-A[2,3]*A[3,1]) +
              A[1,3]*(A[2,1]*A[3,2]-A[2,2]*A[3,1])
        @test abs(S3[1,1] - (A[2,2]*A[3,3]-A[2,3]*A[3,2]) / det) < 0.01
        @test abs(S3[2,2] - (A[1,1]*A[3,3]-A[1,3]*A[3,1]) / det) < 0.01
        @test abs(S3[3,3] - (A[1,1]*A[2,2]-A[1,2]*A[2,1]) / det) < 0.01

        tmp = zeros(3); q = [0.3, -1.2, 0.7]
        quad = 0.0
        for i in 1:3, j in 1:3
            quad += q[i] * A[i, j] * q[j]
        end
        @test isapprox(M.kinetic_dense(q, A, tmp), 0.5 * quad; atol = 1e-12)
    end

    @testset "dense HMC agrees with diagonal HMC on the posterior" begin
        # A dense metric with the same variances on its diagonal and zeros
        # off-diagonal must sample the SAME distribution as the diagonal one.
        # This is what catches a transposed or inverted metric.
        layout = M.ParamLayout(data.S, data.n_years, false, M.PROC_RW1)
        gradf! = (g, x) -> M.logposterior_grad!(g, x, data, layout, se_ab, sp_ab, true)
        negf! = (g, x) -> begin
            fx = gradf!(g, x)
            @inbounds for i in eachindex(g); g[i] = -g[i]; end
            return -fx
        end
        th0 = M.lbfgs(negf!, M.initial_theta(layout); maxiter = 200).theta
        rn = M.nuts(gradf!, th0, 400, 400, 0.8, 5)
        vdiag = rn.inv_mass
        vdense = zeros(layout.n, layout.n)
        for k in 1:layout.n
            vdense[k, k] = vdiag[k]
        end
        L_traj = max(1, round(Int, 2^rn.mean_depth - 1))
        r1 = M.hmc(gradf!, th0, vdiag,  rn.step_size, L_traj, 1500, 300, 9)
        r2 = M.hmc(gradf!, th0, vdense, rn.step_size, L_traj, 1500, 300, 9)
        @test r1.n_divergent == 0
        @test r2.n_divergent == 0
        nd = size(r1.draws, 2)
        for i in 1:layout.n
            m1 = sum(view(r1.draws, i, :)) / nd
            m2 = sum(view(r2.draws, i, :)) / nd
            sd = sqrt(sum((r1.draws[i, k] - m1)^2 for k in 1:nd) / (nd - 1))
            @test abs(m1 - m2) < 0.35 * max(sd, 1e-6)
        end
    end

    @testset "metric files round-trip and are validated" begin
        # The compiled-in metrics are a convenience for the badger cohort, not
        # the only way to supply one: --metric must accept a matrix tuned for
        # any data, without rebuilding the binary.
        dir = mktempdir()
        A = [2.0 0.3 0.1; 0.3 1.5 -0.2; 0.1 -0.2 0.8]
        f = joinpath(dir, "m.csv")
        M.write_metric(f, A)
        B = M.read_metric(f)
        # Exact, not approximate: write_metric uses string(::Float64), which
        # round-trips, rather than the 8-digit fmt used for reported values.
        @test B == A

        # A diagonal metric may be given as a single row or a single column.
        f2 = joinpath(dir, "diag_row.csv")
        open(f2, "w") do io; write(io, "2.0,4.0,8.0
"); end
        D1 = M.read_metric(f2)
        @test size(D1) == (3, 3)
        @test [D1[k, k] for k in 1:3] == [2.0, 4.0, 8.0]
        @test D1[1, 2] == 0.0
        f3 = joinpath(dir, "diag_col.csv")
        open(f3, "w") do io; write(io, "2.0
4.0
8.0
"); end
        @test M.read_metric(f3) == D1

        # Comments and blank lines are skipped, so a file can be annotated.
        f4 = joinpath(dir, "commented.csv")
        open(f4, "w") do io
            write(io, "# tuned for cohort X

1.0,0.0
0.0,1.0
")
        end
        @test M.read_metric(f4) == [1.0 0.0; 0.0 1.0]

        # Malformed input must fail loudly rather than produce a wrong metric.
        f5 = joinpath(dir, "ragged.csv")
        open(f5, "w") do io; write(io, "1.0,2.0
3.0
"); end
        @test_throws ErrorException M.read_metric(f5)
        f6 = joinpath(dir, "nonsquare.csv")
        open(f6, "w") do io; write(io, "1.0,2.0,3.0
4.0,5.0,6.0
"); end
        @test_throws ErrorException M.read_metric(f6)
        f7 = joinpath(dir, "empty.csv")
        open(f7, "w") do io; write(io, "
"); end
        @test_throws ErrorException M.read_metric(f7)

        # A metric read from file must drive the sampler identically to the
        # same matrix passed in memory.
        layout = M.ParamLayout(data.S, data.n_years, false, M.PROC_RW1)
        gradf! = (g, x) -> M.logposterior_grad!(g, x, data, layout, se_ab, sp_ab, true)
        negf! = (g, x) -> begin
            fx = gradf!(g, x)
            @inbounds for i in eachindex(g); g[i] = -g[i]; end
            return -fx
        end
        th0 = M.lbfgs(negf!, M.initial_theta(layout); maxiter = 150).theta
        mm = zeros(layout.n, layout.n)
        for k in 1:layout.n; mm[k, k] = 0.5 + 0.01k; end
        fm = joinpath(dir, "run.csv")
        M.write_metric(fm, mm)
        r1 = M.hmc(gradf!, th0, mm, 0.05, 8, 200, 100, 4)
        r2 = M.hmc(gradf!, th0, M.read_metric(fm), 0.05, 8, 200, 100, 4)
        @test r1.draws == r2.draws
    end

    @testset "pre-tuned HMC parameter sets" begin
        # These are tuned on the REAL cohort (58 parameters fixed / 70 inferred),
        # not on the small test fixture, so only their internal consistency can
        # be checked here -- not that they match any layout built from testdata.
        for nm in ("all_fixed", "all_inferred", "culture_only_fixed",
                   "no_dpp_fixed", "no_dpp_inferred")
            pt = M.find_pretuned(nm)
            @test pt.name == nm
            @test size(pt.inv_mass) == (pt.n, pt.n)
            @test all(isfinite, pt.inv_mass)
            @test all(k -> pt.inv_mass[k, k] > 0.0, 1:pt.n)   # variances
            # symmetric, as a covariance must be
            @test maximum(abs.(pt.inv_mass .- transpose(pt.inv_mass))) < 1e-10
            # and usable as a metric at all
            @test M.chol_lower!(zeros(pt.n, pt.n), pt.inv_mass)
            @test pt.eps > 0.0
            @test pt.L >= 1
        end
        # The shipped sets are tuned under rw1 and are NOT valid for iid: the
        # geometries need step sizes and trajectory lengths that differ ~6x
        # (rw1 0.048/L=45 vs iid 0.28/L=7 on the real cohort). Applying the
        # wrong one samples badly rather than failing, so the tag must be there.
        for nm in ("all_fixed", "all_inferred", "culture_only_fixed",
                   "no_dpp_fixed", "no_dpp_inferred")
            @test M.find_pretuned(nm).year_process == M.PROC_RW1
        end
        @test_throws ErrorException M.find_pretuned("no_such_model")
        # The inferred variants carry the 12 extra Se/Sp parameters.
        @test M.find_pretuned("all_inferred").n ==
              M.find_pretuned("all_fixed").n + 2 * M.N_TESTS
    end

    @testset "pre-tuned sets are rejected on a mismatched layout" begin
        # The shipped sets are for the real cohort; using one against the small
        # fixture must fail loudly rather than sample the wrong distribution.
        # This is the ONLY safety check --model makes, so it has to work.
        layout = M.ParamLayout(data.S, data.n_years, false)
        pt = M.find_pretuned("all_fixed")
        @test pt.n != layout.n
        gradf! = (g, x) -> M.logposterior_grad!(g, x, data, layout, se_ab, sp_ab, true)
        th = M.initial_theta(layout)
        @test_throws ErrorException M.hmc(gradf!, th, pt.inv_mass, pt.eps, pt.L, 10, 10, 1)
    end

    @testset "every year process agrees with forward-mode AD" begin
        # The CLI supports five of test2infeR's six year processes; ar1 is
        # excluded because it adds a rho parameter that would change the
        # vector layout. Each process is a different gamma map and so a
        # different adjoint, and rw2's in particular (a reverse cumsum of a
        # reverse cumsum) is easy to get subtly wrong in a way that only shows
        # up as a slightly wrong posterior.
        for proc in (M.PROC_IID, M.PROC_RW1, M.PROC_RW2, M.PROC_NONE, M.PROC_SHRUNK)
            for infer in (false, true), pen in (true, false)
                lay = M.ParamLayout(data.S, data.n_years, infer, proc)
                logp = th -> M.logposterior(th, data, lay, se_ab, sp_ab, pen)
                for trial in 1:3
                    th = M.initial_theta(lay) .+ 0.13 * (trial - 1) .+ 0.05
                    ga = zeros(lay.n)
                    la = M.logposterior_grad!(ga, th, data, lay, se_ab, sp_ab, pen)
                    gf = zeros(lay.n)
                    lf = infer ? M.gradient!(gf, logp, th, Val(25)) :
                                 M.gradient!(gf, logp, th, Val(13))
                    @test isapprox(la, lf; atol = 1e-8)
                    @test maximum(abs.(ga .- gf)) < 1e-10
                end
            end
        end
    end

    @testset "rw1c centres the walk and identifies the level" begin
        # Plain rw1 leaves the LEVEL of the year series free: adding a constant
        # to every gamma is absorbed by nothing, so untested years drift. rw1c
        # subtracts the mean, which is what makes the level alpha's job.
        lay_rw1  = M.ParamLayout(data.S, data.n_years, false, M.PROC_RW1)
        lay_rw1c = M.ParamLayout(data.S, data.n_years, false, M.PROC_RW1C)
        th = M.initial_theta(lay_rw1) .+ 0.11

        P1  = M.extract_params(th, lay_rw1)
        P1c = M.extract_params(th, lay_rw1c)
        @test abs(sum(P1c.gamma)) < 1e-10          # sums to zero by construction
        @test abs(sum(P1.gamma)) > 1e-6            # plain rw1 does not
        # rw1c is rw1 shifted by a constant, and only by a constant.
        shift = P1.gamma .- P1c.gamma
        @test maximum(shift) - minimum(shift) < 1e-10
        @test !isapprox(M.logposterior(th, data, lay_rw1,  se_ab, sp_ab, true),
                        M.logposterior(th, data, lay_rw1c, se_ab, sp_ab, true))
    end

    @testset "the data-adaptive increment prior" begin
        # sd grows with how well the link is informed: pinned at the floor when
        # neither year was tested, approaching TAU once both are well tested.
        @test M.year_link_sd(0.0) ≈ M.YEAR_LINK_FLOOR
        @test M.year_link_sd(1.0) < M.year_link_sd(10.0) < M.year_link_sd(1000.0)
        @test M.year_link_sd(1e6) < M.YEAR_LINK_TAU + M.YEAR_LINK_FLOOR

        # Counted over raw INPUT ROWS, not gridded captures: build_data stacks
        # repeat captures within a timestep, so the two differ wherever a badger
        # was caught twice in one quarter. Rows are the evidence count.
        yw = M.year_weights(mat, fill(true, 6), data.n_years, data.S)
        @test length(yw) == data.n_years
        @test sum(yw) == count(r -> any(!isnan, @view mat[r, 4:9]), axes(mat, 1))

        # A masked-out assay is not evidence either.
        yw_cult = M.year_weights(mat, [false,false,true,false,false,false],
                                 data.n_years, data.S)
        @test sum(yw_cult) <= sum(yw)

        # The prior must actually change the density, and its adjoint must be
        # right in both spaces.
        lay = M.ParamLayout(data.S, data.n_years, false, M.PROC_RW1C)
        th = M.initial_theta(lay) .+ 0.09
        @test !isapprox(M.logposterior(th, data, lay, se_pr_b, sp_pr_b, true, M.EMPTY_YW),
                        M.logposterior(th, data, lay, se_pr_b, sp_pr_b, true, yw))
        for jac in (true, false)
            logp = t -> M.logposterior(t, data, lay, se_pr_b, sp_pr_b, true, yw, jac)
            ga = zeros(lay.n)
            la = M.logposterior_grad!(ga, th, data, lay, se_pr_b, sp_pr_b, true, yw, jac)
            gf = zeros(lay.n)
            lf = M.gradient!(gf, logp, th, Val(13))
            @test isapprox(la, lf; atol = 1e-8)
            @test maximum(abs.(ga .- gf)) < 1e-10
        end
    end

    @testset "logit-normal accuracy priors" begin
        # Each literature point value is the prior MEDIAN, and the central 95%
        # must cover both supplied interval endpoints -- the intervals are
        # asymmetric on the logit scale, so the wider half sets the SD.
        for k in 1:6
            mu, sd = M.logitnormal_from_interval(M.SE_FIXED_DEFAULT[k],
                        M.SE_CI_DEFAULT[k][1], M.SE_CI_DEFAULT[k][2])
            @test M.logistic(mu) ≈ M.SE_FIXED_DEFAULT[k]
            @test M.logistic(mu - 1.959963984540054sd) <= M.SE_CI_DEFAULT[k][1] + 1e-12
            @test M.logistic(mu + 1.959963984540054sd) >= M.SE_CI_DEFAULT[k][2] - 1e-12
        end

        # Unlike Beta, a logit-normal never diverges at a boundary, which is
        # what makes MAP well posed.
        se_ln, sp_ln = M.default_accuracy_priors(M.ACC_LOGITNORMAL)
        @test !any(M.is_singular_at_boundary, se_ln)
        @test !any(M.is_singular_at_boundary, sp_ln)
        # The Beta defaults DO, which is why MAP refuses them.
        se_b, sp_b = M.default_accuracy_priors(M.ACC_BETA)
        @test any(M.is_singular_at_boundary, (se_b..., sp_b...))

        lay = M.ParamLayout(data.S, data.n_years, true, M.PROC_RW1C)
        th = M.initial_theta(lay) .+ 0.05
        for jac in (true, false)
            logp = t -> M.logposterior(t, data, lay, se_ln, sp_ln, true, M.EMPTY_YW, jac)
            ga = zeros(lay.n)
            la = M.logposterior_grad!(ga, th, data, lay, se_ln, sp_ln, true, M.EMPTY_YW, jac)
            gf = zeros(lay.n)
            lf = M.gradient!(gf, logp, th, Val(25))
            @test isapprox(la, lf; atol = 1e-8)
            @test maximum(abs.(ga .- gf)) < 1e-10
        end
    end

    @testset "pinned specificities" begin
        pin = (NaN, NaN, 1.0, NaN, NaN, NaN)
        lay_free = M.ParamLayout(data.S, data.n_years, true, M.PROC_RW1C)
        lay_pin  = M.ParamLayout(data.S, data.n_years, true, M.PROC_RW1C, pin)

        # A pin REMOVES a parameter; it is not a masked gradient entry.
        @test lay_pin.n == lay_free.n - 1
        @test lay_pin.sp_slot[3] == 0
        @test all(lay_pin.sp_slot[k] != 0 for k in (1, 2, 4, 5, 6))
        # Later tests shift down to fill the gap.
        @test lay_pin.sp_slot[4] == lay_free.sp_slot[4] - 1
        # Names must follow the slots, or they label the wrong values.
        @test length(M.param_names(lay_pin)) == lay_pin.n
        @test !("Sp[3]" in M.param_names(lay_pin))
        @test "Sp[4]" in M.param_names(lay_pin)

        # The pinned value must reach the likelihood and the decoder.
        th = M.initial_theta(lay_pin) .+ 0.05
        @test M.extract_params(th, lay_pin).Sp[3] == 1.0

        se_ln, sp_ln = M.default_accuracy_priors(M.ACC_LOGITNORMAL)
        for jac in (true, false)
            logp = t -> M.logposterior(t, data, lay_pin, se_ln, sp_ln, true, M.EMPTY_YW, jac)
            ga = zeros(lay_pin.n)
            la = M.logposterior_grad!(ga, th, data, lay_pin, se_ln, sp_ln, true,
                                      M.EMPTY_YW, jac)
            gf = zeros(lay_pin.n)
            lf = M.gradient!(gf, logp, th, Val(25))
            @test isapprox(la, lf; atol = 1e-8)
            @test maximum(abs.(ga .- gf)) < 1e-10
        end

        # With Se/Sp FIXED a pin overrides the Table 1 value rather than
        # removing a parameter -- still a different model, not a no-op.
        lay_fx     = M.ParamLayout(data.S, data.n_years, false, M.PROC_RW1C)
        lay_fx_pin = M.ParamLayout(data.S, data.n_years, false, M.PROC_RW1C, pin)
        @test lay_fx_pin.n == lay_fx.n
        @test M.extract_params(M.initial_theta(lay_fx_pin), lay_fx_pin).Sp[3] == 1.0
        thf = M.initial_theta(lay_fx) .+ 0.05
        @test !isapprox(M.logposterior(thf, data, lay_fx,     se_ab, sp_ab, true),
                        M.logposterior(thf, data, lay_fx_pin, se_ab, sp_ab, true))
    end

    @testset "MAP and sampling target different spaces" begin
        # Turing's maximum_a_posteriori maximises the CONSTRAINED density, so
        # the two differ by exactly the transform Jacobian. Both must be
        # available, and each must match the primal it is paired with.
        lay = M.ParamLayout(data.S, data.n_years, false, M.PROC_RW1C)
        th = M.initial_theta(lay) .+ 0.05
        lp_j  = M.logposterior(th, data, lay, se_pr_b, sp_pr_b, true, M.EMPTY_YW, true)
        lp_nj = M.logposterior(th, data, lay, se_pr_b, sp_pr_b, true, M.EMPTY_YW, false)
        # For fixed Se/Sp the only transformed parameter is sigma_g.
        @test lp_j - lp_nj ≈ th[lay.i_logsigma]
    end

    @testset "the year processes are genuinely different models" begin
        # A wrong gamma map would still sample; it would just sample the wrong
        # posterior. Pin the actual arithmetic rather than only the gradient.
        lay_iid = M.ParamLayout(data.S, data.n_years, false, M.PROC_IID)
        lay_rw1 = M.ParamLayout(data.S, data.n_years, false, M.PROC_RW1)
        lay_rw2 = M.ParamLayout(data.S, data.n_years, false, M.PROC_RW2)
        lay_non = M.ParamLayout(data.S, data.n_years, false, M.PROC_NONE)
        lay_shr = M.ParamLayout(data.S, data.n_years, false, M.PROC_SHRUNK)
        th = M.initial_theta(lay_rw1) .+ 0.07

        P_iid = M.extract_params(th, lay_iid)
        P_rw1 = M.extract_params(th, lay_rw1)
        P_rw2 = M.extract_params(th, lay_rw2)
        P_non = M.extract_params(th, lay_non)
        P_shr = M.extract_params(th, lay_shr)

        acc = 0.0; acc2 = 0.0
        for y in 1:data.n_years
            r = th[lay_rw1.i_gamma + y - 1]
            acc += r; acc2 += acc
            @test isapprox(P_iid.gamma[y], P_iid.sigma_g * r;    atol = 1e-12)
            @test isapprox(P_rw1.gamma[y], P_rw1.sigma_g * acc;  atol = 1e-12)
            @test isapprox(P_rw2.gamma[y], P_rw2.sigma_g * acc2; atol = 1e-12)
            @test P_non.gamma[y] == 0.0
        end
        # shrunk differs from iid ONLY in the sigma prior, so the gamma map is
        # identical while the log density is not.
        @test P_shr.gamma == P_iid.gamma
        @test M.sigma_prior_sd(M.PROC_SHRUNK) == 0.10
        @test M.sigma_prior_sd(M.PROC_RW2) == 0.01
        @test M.logposterior(th, data, lay_shr, se_ab, sp_ab, true) !=
              M.logposterior(th, data, lay_iid, se_ab, sp_ab, true)
    end

    @testset "rw1 and iid are different models" begin
        # Guards the class of bug that made every earlier benchmark invalid:
        # the CLI silently fitting iid while the package fitted rw1.
        d = data
        liid = M.ParamLayout(d.S, d.n_years, false, M.PROC_IID)
        lrw1 = M.ParamLayout(d.S, d.n_years, false, M.PROC_RW1)
        @test liid.n == lrw1.n              # same parameter COUNT ...
        th = M.initial_theta(lrw1) .+ 0.07
        lp_iid = M.logposterior(th, d, liid, se_ab, sp_ab, true)
        lp_rw1 = M.logposterior(th, d, lrw1, se_ab, sp_ab, true)
        @test !isapprox(lp_iid, lp_rw1; atol = 1e-6)   # ... different density
        # rw1 accumulates: gamma_y = sigma_g * sum(raw_1..raw_y)
        Piid = M.extract_params(th, liid)
        Prw1 = M.extract_params(th, lrw1)
        acc = 0.0
        for y in 1:d.n_years
            acc += th[lrw1.i_gamma + y - 1]
            @test isapprox(Prw1.gamma[y], Prw1.sigma_g * acc; atol = 1e-12)
            @test isapprox(Piid.gamma[y],
                           Piid.sigma_g * th[liid.i_gamma + y - 1]; atol = 1e-12)
        end
        # and the sigma_g prior sd differs with the process
        @test M.sigma_prior_sd(M.PROC_RW1) == 0.05
        @test M.sigma_prior_sd(M.PROC_IID) == 0.30
    end

    @testset "HMC rejects a malformed metric" begin
        layout = M.ParamLayout(data.S, data.n_years, false)
        gradf! = (g, x) -> M.logposterior_grad!(g, x, data, layout, se_ab, sp_ab, true)
        th = M.initial_theta(layout)
        # Wrong length, and a non-positive entry: both must fail loudly rather
        # than silently sampling the wrong distribution.
        @test_throws ErrorException M.hmc(gradf!, th, ones(layout.n - 1), 0.1, 5, 10, 10, 1)
        bad = ones(layout.n); bad[3] = -1.0
        @test_throws ErrorException M.hmc(gradf!, th, bad, 0.1, 5, 10, 10, 1)
        # dense: wrong size, and not positive definite
        @test_throws ErrorException M.hmc(gradf!, th, zeros(3, 3), 0.1, 5, 10, 10, 1)
        npd = Matrix{Float64}(undef, layout.n, layout.n)
        fill!(npd, 0.0); for k in 1:layout.n; npd[k, k] = -1.0; end
        @test_throws ErrorException M.hmc(gradf!, th, npd, 0.1, 5, 10, 10, 1)
    end

    @testset "HMC and NUTS agree on the posterior" begin
        # Two different samplers on the same target must give the same answer.
        # This is the check that would catch a wrong metric, a bad momentum
        # draw or a sign error in the leapfrog -- none of which stop the
        # sampler running, they just move it to the wrong distribution.
        layout = M.ParamLayout(data.S, data.n_years, false)
        logp = th -> M.logposterior(th, data, layout, se_ab, sp_ab, true)
        gradf! = (g, x) -> M.logposterior_grad!(g, x, data, layout, se_ab, sp_ab, true)
        negf! = (g, x) -> begin
            fx = gradf!(g, x)
            @inbounds for i in eachindex(g); g[i] = -g[i]; end
            return -fx
        end
        th0 = M.lbfgs(negf!, M.initial_theta(layout); maxiter = 300).theta

        # Tune on THIS fixture rather than borrowing a shipped set: the
        # pre-tuned values are for the real cohort (58/70 parameters) and the
        # test fixture has 13, so they are not interchangeable. Taking the
        # metric from a NUTS run is exactly what tune/tune_hmc.jl does, so this
        # also exercises that path end to end.
        rn = M.nuts(gradf!, th0, 3000, 800, 0.8, 11)
        L_traj = max(1, round(Int, 2^rn.mean_depth - 1))
        rh = M.hmc(gradf!, th0, rn.inv_mass, rn.step_size, L_traj, 3000, 800, 11)
        @test rn.n_divergent == 0
        @test rh.n_divergent == 0
        @test rh.accept_rate > 0.6

        nd = size(rn.draws, 2)
        for i in 1:layout.n
            mn = sum(view(rn.draws, i, :)) / nd
            mh = sum(view(rh.draws, i, :)) / nd
            sn = sqrt(sum((rn.draws[i, k] - mn)^2 for k in 1:nd) / (nd - 1))
            # Allow a generous multiple of the MC error: ESS is well below the
            # draw count, so the tolerance is deliberately loose. A wrong
            # sampler fails this by many sds, not by a fraction of one.
            @test abs(mn - mh) < 0.35 * max(sn, 1e-6)
        end
    end

    @testset "nuts_then_hmc agrees with pure NUTS" begin
        # The default full-Bayes path: NUTS adapts, fixed-L HMC draws with what
        # it learned. It must land on the same posterior as NUTS alone -- a
        # wrong hand-off (stale metric, wrong step size, a start point outside
        # the typical set) would still produce draws, just of the wrong thing.
        layout = M.ParamLayout(data.S, data.n_years, false, M.PROC_RW1C)
        logp = th -> M.logposterior(th, data, layout, se_pr_b, sp_pr_b, true)
        gradf! = (g, x) -> M.logposterior_grad!(g, x, data, layout,
                                                se_pr_b, sp_pr_b, true)
        negf! = (g, x) -> begin
            fx = gradf!(g, x)
            @inbounds for i in eachindex(g); g[i] = -g[i]; end
            return -fx
        end
        th0 = M.lbfgs(negf!, M.initial_theta(layout); maxiter = 300).theta

        rn  = M.nuts(gradf!, th0, 3000, 800, 0.8, 21)
        nth = M.nuts_then_hmc(gradf!, th0, 3000, 800, 0.8, 21)

        @test nth.L >= 1
        @test nth.result.n_divergent == 0
        @test nth.result.accept_rate > 0.6
        # L is inferred from the adapted tree depth, not fixed in advance.
        @test nth.L == max(1, min(round(Int, 2.0^nth.warmup.mean_depth) - 1, 1024))
        # It inherits the adapted values rather than re-deriving them.
        @test nth.result.step_size == nth.warmup.step_size
        @test nth.result.inv_mass == nth.warmup.inv_mass

        nd = size(rn.draws, 2)
        for i in 1:layout.n
            mn = sum(view(rn.draws, i, :)) / nd
            mh = sum(view(nth.result.draws, i, :)) / size(nth.result.draws, 2)
            sn = sqrt(sum((rn.draws[i, k] - mn)^2 for k in 1:nd) / (nd - 1))
            @test abs(mn - mh) < 0.35 * max(sn, 1e-6)
        end

        # An explicit L overrides the inferred one.
        nth2 = M.nuts_then_hmc(gradf!, th0, 200, 400, 0.8, 22; n_leapfrog = 7)
        @test nth2.L == 7
    end

    @testset "per-draw trajectory outputs" begin
        # These feed 03_trajectories.R in the sql-e2e bundles: prevalence CIs
        # need each draw's prevalence, and the trajectory outputs need each
        # draw's infection time per badger. Both come from a forward-backward
        # pass per badger per draw.
        layout = M.ParamLayout(data.S, data.n_years, false, M.PROC_RW1)
        gradf! = (g, x) -> M.logposterior_grad!(g, x, data, layout, se_ab, sp_ab, true)
        negf! = (g, x) -> begin
            fx = gradf!(g, x)
            @inbounds for i in eachindex(g); g[i] = -g[i]; end
            return -fx
        end
        th0 = M.lbfgs(negf!, M.initial_theta(layout); maxiter = 200).theta
        res = M.nuts(gradf!, th0, 60, 60, 0.8, 3)

        out = mktempdir()
        opts = M.Options(DATA, "nuts", out, 4, fill(true, 6), false, true,
                         :stack, 60, 60, 0.8, 3, 1000, M.PROC_RW1, 10,
                         "", "", "", -1.0, -1,
                         M.ACC_BETA, false, M.NO_SP_PIN,
                         false, M.NO_E_PIN, false, 0.0, "", 1)
        M.draw_outputs(opts, data, layout, res.draws, 10)

        pd = M.read_matrix(joinpath(out, "prevalence_draws.csv"))
        tj = M.read_matrix(joinpath(out, "trajectories.csv"))

        tset = sort(unique(data.times))
        @test size(pd, 1) == length(tset)          # one row per timestep
        @test size(pd, 2) == 11                    # time + 10 draws
        @test pd[:, 1] == Float64.(tset)
        # Prevalences are probabilities (NaN where a timestep has no capture).
        @test all(x -> isnan(x) || (0.0 <= x <= 1.0), pd[:, 2:end])
        # At least one real value per draw column, or the decode did nothing.
        @test all(k -> any(!isnan, pd[:, k]), 2:11)

        @test size(tj, 1) == data.n_ind * 10       # badgers x draws
        @test size(tj, 2) == 3
        @test sort(unique(tj[:, 2])) == collect(1.0:10.0)
        # infection_time is 0 (never) or an actual timestep in that badger's record
        valid = Set(Float64.(data.times))
        @test all(t -> t == 0.0 || t in valid, tj[:, 3])
        # Each badger appears exactly once per draw.
        @test length(unique(tj[:, 1])) == data.n_ind

        # traj_draws = 0 must write nothing at all.
        out2 = mktempdir()
        opts2 = M.Options(DATA, "nuts", out2, 4, fill(true, 6), false, true,
                          :stack, 60, 60, 0.8, 3, 1000, M.PROC_RW1, 0,
                          "", "", "", -1.0, -1,
                          M.ACC_BETA, false, M.NO_SP_PIN,
                          false, M.NO_E_PIN, false, 0.0, "", 1)
        M.draw_outputs(opts2, data, layout, res.draws, 0)
        @test !isfile(joinpath(out2, "prevalence_draws.csv"))
        @test !isfile(joinpath(out2, "trajectories.csv"))
    end

    @testset "three-state gradient agrees with forward-mode AD" begin
        # Same check as the two-state gradient, over accuracy mode, Jacobian,
        # Sp pinning, prior family, year process and multiplier pattern.
        # Sp[3] = 1 with a multiplier pinned to 0 gives a zero emission factor,
        # which is the case the guards in the sweep are for.
        all_e = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        mix_e = (NaN, 0.3, 0.0, NaN, 1.0, NaN)
        for fixture in (DATA, SPARSE)
            fmat = M.read_matrix(fixture)
            fdata = M.build_data(fmat, fill(true, 6), :stack, 4)
            yw = M.year_weights(fmat, fill(true, 6), fdata.n_years, fdata.S, true)
            for infer in (false, true), jac in (true, false), pin in (false, true),
                fam in (M.ACC_LOGITNORMAL, M.ACC_BETA),
                proc in (M.PROC_RW1C, M.PROC_IID),
                edet in (M.NO_E_PIN, M.E_PIN_CULTURE_SPLIT, mix_e, all_e)

                sp_pin = pin ? (NaN, NaN, 1.0, NaN, NaN, NaN) : M.NO_SP_PIN
                lay = M.ParamLayout(fdata.S, fdata.n_years, infer, proc, sp_pin,
                                    true, edet)
                se_pr, sp_pr = M.default_accuracy_priors(fam)
                f = th -> M.logposterior(th, fdata, lay, se_pr, sp_pr, true, yw, jac)
                th = M.initial_theta(lay) .+ 0.07 .* sin.(1:lay.n)
                ga = zeros(lay.n)
                la = M.logposterior_grad!(ga, th, fdata, lay, se_pr, sp_pr, true, yw, jac)
                gf = zeros(lay.n)
                lf = M.gradient!(gf, f, th, Val(13))
                @test isapprox(la, lf; atol = 1e-8)
                @test all(isfinite, ga)
                @test maximum(abs.(ga .- gf)) < 1e-9
            end
        end
    end

    @testset "three-state layout extends the two-state one" begin
        for infer in (false, true)
            l2 = M.ParamLayout(data.S, data.n_years, infer, M.PROC_RW1C)
            l3 = M.ParamLayout(data.S, data.n_years, infer, M.PROC_RW1C,
                               M.NO_SP_PIN, true)
            # tau, rho and six multipliers go at the end
            @test l3.n == l2.n + 8
            @test (l3.i_tau, l3.i_rho) == (l2.n + 1, l2.n + 2)
            @test l3.e_slot == ntuple(k -> l2.n + 2 + k, 6)
            @test l3.sp_slot == l2.sp_slot
            @test (l2.i_tau, l2.i_rho) == (0, 0)
            @test l2.e_slot == ntuple(_ -> 0, 6)
            @test M.param_names(l3)[end-7:end] ==
                  vcat(["tau", "rho"], ["e_mult[$k]" for k in 1:6])
            @test length(M.param_names(l3)) == l3.n
            # pinned multipliers have no slot or name
            lp = M.ParamLayout(data.S, data.n_years, infer, M.PROC_RW1C,
                               M.NO_SP_PIN, true, (NaN, 0.5, 0.0, NaN, NaN, 1.0))
            @test lp.n == l2.n + 5
            @test lp.e_slot == (l2.n + 3, 0, 0, l2.n + 4, l2.n + 5, 0)
            @test M.param_names(lp)[end-2:end] ==
                  ["e_mult[1]", "e_mult[4]", "e_mult[5]"]
            th = M.initial_theta(lp); th[lp.e_slot[4]] = 1.3
            P = M.extract_params(th, lp)
            @test P.em[[2, 3, 6]] == [0.5, 0.0, 1.0]
            @test isapprox(P.em[4], M.logistic(1.3))
        end
    end

    @testset "three states collapse to two when E and I emit alike" begin
        # With every multiplier at 1, E and I emit the same, and E+I has the
        # two-state transitions. So the likelihood should equal the two-state
        # one whatever tau and rho are.
        all_e = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        se_pr, sp_pr = M.default_accuracy_priors(M.ACC_LOGITNORMAL)
        for infer in (false, true)
            l2 = M.ParamLayout(data.S, data.n_years, infer, M.PROC_RW1C)
            l3 = M.ParamLayout(data.S, data.n_years, infer, M.PROC_RW1C,
                               M.NO_SP_PIN, true, all_e)
            th2 = M.initial_theta(l2) .+ 0.05 .* cos.(1:l2.n)
            lp2 = M.logposterior(th2, data, l2, se_pr, sp_pr, true, M.EMPTY_YW, true)
            p2 = M.decode(data, M.extract_params(th2, l2))
            for (zt, zr) in ((-3.0, 0.0), (0.7, -1.2), (-6.0, 2.5))
                th3 = vcat(th2, [zt, zr])
                prior = M.accuracy_logpdf_z(M.SEI_TAU_PRIOR, zt, M.logistic(zt), true) +
                        M.accuracy_logpdf_z(M.SEI_RHO_PRIOR, zr, M.logistic(zr), true)
                lp3 = M.logposterior(th3, data, l3, se_pr, sp_pr, true, M.EMPTY_YW, true)
                @test isapprox(lp3 - prior, lp2; atol = 1e-9)
                # and P(E) + P(I) should be the two-state P(infected)
                P3 = M.extract_params(th3, l3)
                p_exp, p_ifs = M.decode_sei(data, P3)
                @test maximum(abs.(p_exp .+ p_ifs .- p2)) < 1e-10
                @test maximum(abs.(M.decode(data, P3) .- p2)) < 1e-10
            end
        end
    end

    @testset "exposed sensitivity is the infectious one times a multiplier" begin
        # P(+ | E) = m Se. Check the factors at the first tested capture.
        lay = M.ParamLayout(data.S, data.n_years, true, M.PROC_RW1C,
                            M.NO_SP_PIN, true)
        th = M.initial_theta(lay) .+ 0.2 .* sin.(1:lay.n)
        P = M.extract_params(th, lay)
        gj = findfirst(j -> data.ptr[j + 1] > data.ptr[j], 1:length(data.times))
        bU, bE, bI = M.emission3(data, gj, P)
        wantU = 1.0; wantE = 1.0; wantI = 1.0
        for p in data.ptr[gj]:(data.ptr[gj + 1] - 1)
            k = data.tidx[p]
            pos = data.vals[p] == 1
            wantU *= pos ? 1 - P.Sp[k] : P.Sp[k]
            wantI *= pos ? P.Se[k] : 1 - P.Se[k]
            wantE *= pos ? P.em[k] * P.Se[k] : 1 - P.em[k] * P.Se[k]
        end
        @test isapprox(bU, wantU; rtol = 1e-12)
        @test isapprox(bI, wantI; rtol = 1e-12)
        @test isapprox(bE, wantE; rtol = 1e-12)
        @test all(m -> 0.0 < m < 1.0, P.em)
        # m = 1 gives I's factor
        l1 = M.ParamLayout(data.S, data.n_years, true, M.PROC_RW1C,
                           M.NO_SP_PIN, true, (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        @test isapprox(M.emission3(data, gj, M.extract_params(th[1:l1.n], l1))[2],
                       bI; rtol = 1e-12)
        # m = 0: factor is 0 if the capture has any positive, else 1
        l0 = M.ParamLayout(data.S, data.n_years, true, M.PROC_RW1C,
                           M.NO_SP_PIN, true, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        P0 = M.extract_params(th[1:l0.n], l0)
        for j in 1:length(data.times)
            data.ptr[j + 1] > data.ptr[j] || continue
            anypos = any(p -> data.vals[p] == 1, data.ptr[j]:(data.ptr[j + 1] - 1))
            @test M.emission3(data, j, P0)[2] == (anypos ? 0.0 : 1.0)
        end
        # flat prior on the multipliers
        @test M.SEI_MULT_PRIOR.family == M.ACC_BETA
        @test (M.SEI_MULT_PRIOR.p, M.SEI_MULT_PRIOR.q) == (1.0, 1.0)
    end

    @testset "the exposed state is ordered above uninfected" begin
        # Penalty is -w * log1pexp(-(m Se + Sp - 1)/s) for each free
        # multiplier, on top of the Se+Sp>1 term. Recover it as penalised
        # minus unpenalised log posterior.
        se_pr, sp_pr = M.default_accuracy_priors(M.ACC_LOGITNORMAL)
        pen_of(lay, th) =
            M.logposterior(th, data, lay, se_pr, sp_pr, true, M.EMPTY_YW, true) -
            M.logposterior(th, data, lay, se_pr, sp_pr, false, M.EMPTY_YW, true)
        term(d) = -M.PENALTY_WEIGHT * M.log1pexp(-d / M.PENALTY_SCALE)
        for infer in (false, true)
            lay = M.ParamLayout(data.S, data.n_years, infer, M.PROC_RW1C,
                                M.NO_SP_PIN, true, (NaN, NaN, NaN, 0.0, 1.0, NaN))
            th = M.initial_theta(lay) .+ 0.2 .* sin.(1:lay.n)
            # push one multiplier low enough for its term to matter
            th[lay.e_slot[3]] = -6.0
            P = M.extract_params(th, lay)
            want = sum(term(P.Se[k] + P.Sp[k] - 1.0) for k in 1:6) +
                   sum(term(P.em[k] * P.Se[k] + P.Sp[k] - 1.0) for k in (1, 2, 3, 6))
            @test isapprox(pen_of(lay, th), want; atol = 1e-9)
            # assays 4 and 5 are pinned so they contribute nothing, even
            # though m = 0 would be heavily penalised
            @test term(0.0 * P.Se[4] + P.Sp[4] - 1.0) < -1.0
            # culture term is large here and goes away with a higher multiplier
            @test term(P.em[3] * P.Se[3] + P.Sp[3] - 1.0) < -1.0
            th2 = copy(th); th2[lay.e_slot[3]] = 2.0
            @test pen_of(lay, th2) > pen_of(lay, th) + 1.0
            # gradient includes it, also with fixed Se/Sp
            f = x -> M.logposterior(x, data, lay, se_pr, sp_pr, true, M.EMPTY_YW, true)
            ga = zeros(lay.n); gf = zeros(lay.n)
            M.logposterior_grad!(ga, th, data, lay, se_pr, sp_pr, true, M.EMPTY_YW, true)
            M.gradient!(gf, f, th, Val(13))
            @test maximum(abs.(ga .- gf)) < 1e-8
            @test abs(ga[lay.e_slot[3]]) > 1.0
        end
    end

    @testset "three-state smoother" begin
        lay = M.ParamLayout(data.S, data.n_years, false, M.PROC_RW1C,
                            M.NO_SP_PIN, true)
        th = M.initial_theta(lay) .+ 0.05 .* sin.(1:lay.n)
        th[lay.i_tau] = -1.0
        P = M.extract_params(th, lay)
        @test P.sei && isapprox(P.tau, M.logistic(-1.0))
        p_exp, p_ifs = M.decode_sei(data, P)
        @test all(x -> 0.0 <= x <= 1.0, p_exp)
        @test all(x -> 0.0 <= x <= 1.0, p_ifs)
        @test all(p_exp .+ p_ifs .<= 1.0 + 1e-12)
        # P(infectious) and P(exposed or infectious) are non-decreasing within
        # a badger. P(exposed) on its own can fall.
        start = 1
        for i in 1:data.n_ind
            e = data.seq_ends[i]
            for j in (start + 1):e
                @test p_ifs[j] >= p_ifs[j - 1] - 1e-12
                @test p_exp[j] + p_ifs[j] >= p_exp[j - 1] + p_ifs[j - 1] - 1e-12
            end
            start = e + 1
        end
        # brute-force the first badger's likelihood over all monotone paths
        e1 = data.seq_ends[1]
        n1 = e1
        if n1 <= 12
            tot = 0.0
            pi1 = P.pi1_vec[data.entry_year[1]]
            for a in 1:(n1 + 1), b in a:(n1 + 1)
                # U before step a, E from a to b-1, I from b on
                pr = 1.0
                for j in 1:n1
                    st = j < a ? 0 : (j < b ? 1 : 2)
                    prev = j == 1 ? -1 : ((j - 1) < a ? 0 : ((j - 1) < b ? 1 : 2))
                    if j == 1
                        pr *= st == 0 ? 1 - pi1 : (st == 1 ? pi1 * (1 - P.rho) : pi1 * P.rho)
                    else
                        t = data.times[j]
                        lam = M.clamp_prob(M.logistic(P.alpha[M.season_of(t, data.S)] +
                                                      P.gamma[M.year_of(t, data.S)]))
                        if prev == 0
                            pr *= st == 0 ? 1 - lam : (st == 1 ? lam : 0.0)
                        elseif prev == 1
                            pr *= st == 1 ? 1 - P.tau : (st == 2 ? P.tau : 0.0)
                        end
                    end
                    bU, bE, bI = M.emission3(data, j, P)
                    pr *= st == 0 ? bU : (st == 1 ? bE : bI)
                end
                tot += pr
            end
            ll = M.seq_loglik_sei(data, 1, e1, pi1, P.rho, P.tau, P.alpha, P.gamma,
                                  P.Se, P.Sp, P.em, data.S)
            @test isapprox(log(tot), ll; atol = 1e-10)
        end
    end

    @testset "sampled state paths follow the smoothed marginals" begin
        # Share of sampled paths in each state at each step should match the
        # forward-backward marginals, for both models.
        for sei in (false, true)
            lay = M.ParamLayout(data.S, data.n_years, false, M.PROC_RW1C,
                                M.NO_SP_PIN, sei)
            th = M.initial_theta(lay) .+ 0.05 .* sin.(1:lay.n)
            sei && (th[lay.i_tau] = -1.0)
            P = M.extract_params(th, lay)
            p_inf, p_exp, p_ifs = M.decode_all(data, P)
            @test isempty(p_exp) == !sei
            nmax = length(data.times)
            states = zeros(Int, nmax)
            fU = zeros(nmax); fE = zeros(nmax); fI = zeros(nmax); lam = zeros(nmax)
            rng = M.Rng(11)
            nrep = 4000
            worst = 0.0
            start = 1
            for i in 1:min(data.n_ind, 12)
                e = data.seq_ends[i]
                n = e - start + 1
                c_any = zeros(n); c_ifs = zeros(n)
                for _ in 1:nrep
                    M.sample_path!(states, rng, fU, fE, fI, lam, data, start, e,
                                   P.pi1_vec[data.entry_year[i]], P, data.S)
                    # paths never go backwards
                    @test issorted(view(states, 1:n))
                    sei || @test all(s -> s != 1, view(states, 1:n))
                    for j in 1:n
                        c_any[j] += states[j] >= 1
                        c_ifs[j] += states[j] == 2
                    end
                end
                for j in 1:n
                    worst = max(worst, abs(c_any[j] / nrep - p_inf[start + j - 1]))
                    sei && (worst = max(worst, abs(c_ifs[j] / nrep - p_ifs[start + j - 1])))
                end
                start = e + 1
            end
            # 4000 draws, binomial sd <= 0.008, so 0.04 is 5 sd
            @test worst < 0.04
        end
    end

    @testset "three-state trajectory outputs" begin
        lay = M.ParamLayout(data.S, data.n_years, false, M.PROC_RW1C,
                            M.NO_SP_PIN, true)
        se_pr, sp_pr = M.default_accuracy_priors(M.ACC_LOGITNORMAL)
        gradf! = (g, x) -> M.logposterior_grad!(g, x, data, lay, se_pr, sp_pr, true,
                                                M.EMPTY_YW, true)
        res = M.nuts(gradf!, M.initial_theta(lay), 40, 60, 0.8, 5)
        valid = Set(Float64.(data.times))
        for sample in (false, true)
            out = mktempdir()
            opts = M.Options(DATA, "nuts", out, 4, fill(true, 6), false, true,
                             :stack, 40, 60, 0.8, 5, 1000, M.PROC_RW1C, 8,
                             "", "", "", -1.0, -1,
                             M.ACC_LOGITNORMAL, false, M.NO_SP_PIN,
                             true, M.NO_E_PIN, sample, 0.0, "", 1)
            M.finish_draws(opts, data, lay, res.draws)
            tj = M.read_matrix(joinpath(out, "trajectories.csv"))
            @test size(tj) == (data.n_ind * 8, 4)
            @test all(t -> t == 0.0 || t in valid, tj[:, 3])
            @test all(t -> t == 0.0 || t in valid, tj[:, 4])
            # infectious_time needs an infection_time, and can't precede it
            for r in axes(tj, 1)
                tj[r, 4] == 0.0 && continue
                @test tj[r, 3] != 0.0
                @test tj[r, 4] >= tj[r, 3]
            end
            pin = M.read_matrix(joinpath(out, "p_infected.csv"))
            @test size(pin, 2) == 5
            # 1e-6 because the CSV has ~8 significant figures
            @test maximum(abs.(pin[:, 3] .- pin[:, 4] .- pin[:, 5])) < 1e-6
            pv = M.read_matrix(joinpath(out, "prevalence.csv"))
            @test size(pv, 2) == 4
            @test all(r -> isnan(pv[r, 2]) || pv[r, 4] <= pv[r, 2] + 1e-12, axes(pv, 1))
            pd = M.read_matrix(joinpath(out, "prevalence_infectious_draws.csv"))
            @test size(pd, 2) == 9
        end
        # same seed, same trajectories
        outs = [mktempdir(), mktempdir()]
        for o in outs
            opts = M.Options(DATA, "nuts", o, 4, fill(true, 6), false, true,
                             :stack, 40, 60, 0.8, 5, 1000, M.PROC_RW1C, 8,
                             "", "", "", -1.0, -1,
                             M.ACC_LOGITNORMAL, false, M.NO_SP_PIN,
                             true, M.NO_E_PIN, true, 0.0, "", 1)
            M.draw_outputs(opts, data, lay, res.draws, 8)
        end
        @test read(joinpath(outs[1], "trajectories.csv")) ==
              read(joinpath(outs[2], "trajectories.csv"))
    end

    @testset "draws.csv is on the scale its column names claim" begin
        # draws.csv should hold sigma_g, Se, Sp, tau, rho, e_mult on the
        # natural scale, not the log/logit values the sampler uses
        lay = M.ParamLayout(data.S, data.n_years, true, M.PROC_RW1C,
                            (NaN, NaN, 1.0, NaN, NaN, NaN), true)
        draws = reshape(collect(range(-2.0, 2.0; length = 3 * lay.n)), lay.n, 3)
        out = mktempdir()
        opts = M.Options(DATA, "nuts", out, 4, fill(true, 6), true, true,
                         :stack, 3, 3, 0.8, 1, 1000, M.PROC_RW1C, 0,
                         "", "", "", -1.0, -1,
                         M.ACC_LOGITNORMAL, false, (NaN, NaN, 1.0, NaN, NaN, NaN),
                         true, M.NO_E_PIN, false, 0.0, "", 1)
        M.write_draws(opts, lay, draws)
        hdr = split(readline(joinpath(out, "draws.csv")), ",")
        got = M.read_matrix(joinpath(out, "draws.csv"))
        @test hdr == M.param_names(lay)
        @test !("Sp[3]" in hdr)                       # pinned: not a parameter
        for (j, nm) in enumerate(hdr), k in 1:3
            x = draws[j, k]
            want = nm == "sigma_g" ? exp(x) :
                   (startswith(nm, "Se[") || startswith(nm, "Sp[") ||
                    startswith(nm, "e_mult[") ||
                    nm == "tau" || nm == "rho") ? M.logistic(x) : x
            @test isapprox(got[k, j], want; rtol = 1e-6, atol = 1e-8)
        end
    end

    @testset "badgers with a known birth start susceptible there" begin
        # Every third badger gets a birth 4 timesteps before its first row.
        # One gets a birth after its first row and one a birth at 0; both
        # should be ignored.
        ids = sort(unique(Int.(mat[:, 2])))
        firstt = Dict(i => minimum(Int.(mat[mat[:, 2] .== i, 1])) for i in ids)
        births = Dict{Int,Int}()
        for (n, i) in enumerate(ids)
            n % 3 == 0 && firstt[i] > 4 && (births[i] = firstt[i] - 4)
        end
        bad_after = first(i for i in ids if !haskey(births, i)); births[bad_after] = firstt[bad_after] + 1
        bad_before = first(i for i in ids if !haskey(births, i)); births[bad_before] = 0
        db = M.build_data(mat, fill(true, 6), :stack, 4, births)
        @test db.n_ind == data.n_ind
        @test !any(data.born)                       # no births given: nobody
        n_ok = count(i -> get(births, i, 0) >= 1 && births[i] <= firstt[i], ids)
        @test count(db.born) == n_ok
        @test !db.born[findfirst(==(bad_after), db.ids)]
        @test !db.born[findfirst(==(bad_before), db.ids)]
        # four extra uncaptured, untested steps each
        @test length(db.times) == length(data.times) + 4 * n_ok
        @test length(db.vals) == length(data.vals)
        start = 1
        for i in 1:db.n_ind
            e = db.seq_ends[i]
            if db.born[i]
                @test db.times[start] == births[db.ids[i]]
                @test !any(db.captured[start:(start + 3)])
                @test all(j -> db.ptr[j + 1] == db.ptr[j], start:(start + 3))
            end
            start = e + 1
        end

        se_pr, sp_pr = M.default_accuracy_priors(M.ACC_LOGITNORMAL)
        for sei in (false, true), infer in (false, true)
            lay = M.ParamLayout(db.S, db.n_years, infer, M.PROC_RW1C, M.NO_SP_PIN, sei)
            th = M.initial_theta(lay) .+ 0.1 .* sin.(1:lay.n)
            f = x -> M.logposterior(x, db, lay, se_pr, sp_pr, true, M.EMPTY_YW, true)
            ga = zeros(lay.n); gf = zeros(lay.n)
            la = M.logposterior_grad!(ga, th, db, lay, se_pr, sp_pr, true, M.EMPTY_YW, true)
            lf = M.gradient!(gf, f, th, Val(13))
            @test isapprox(la, lf; atol = 1e-8)
            @test maximum(abs.(ga .- gf)) < 1e-9
            # smoother gives 0 at birth whatever pi1_0 is
            for z in (-1.7, 3.0)
                th2 = copy(th); th2[lay.i_pi1_0] = z
                p = M.decode(db, M.extract_params(th2, lay))
                st = 1
                for i in 1:db.n_ind
                    db.born[i] && @test p[st] == 0.0
                    st = db.seq_ends[i] + 1
                end
            end
            # sampled paths start uninfected too
            P = M.extract_params(th, lay)
            nmax = length(db.times); states = zeros(Int, nmax)
            fU = zeros(nmax); fE = zeros(nmax); fI = zeros(nmax); lam = zeros(nmax)
            rng = M.Rng(3); st = 1
            for i in 1:db.n_ind
                e = db.seq_ends[i]
                if db.born[i]
                    for _ in 1:20
                        M.sample_path!(states, rng, fU, fE, fI, lam, db, st, e,
                                       M.entry_prob(db, P.pi1_vec, i), P, db.S)
                        @test states[1] == 0
                    end
                end
                st = e + 1
            end
        end
        # if every badger has a birth, pi1 only appears in its priors
        allb = Dict(i => firstt[i] for i in ids)
        da = M.build_data(mat, fill(true, 6), :stack, 4, allb)
        @test all(da.born)
        lay = M.ParamLayout(da.S, da.n_years, false, M.PROC_RW1C, M.NO_SP_PIN, true)
        th = M.initial_theta(lay); g = zeros(lay.n)
        M.logposterior_grad!(g, th, da, lay, se_pr, sp_pr, true, M.EMPTY_YW, true)
        @test isapprox(g[lay.i_pi1_0], -(th[lay.i_pi1_0] + 1.7); atol = 1e-12)
        @test isapprox(g[lay.i_pi1_mult], -(th[lay.i_pi1_mult] - 1.0); atol = 1e-12)

        # read_births round trip; empty birth = unknown
        path = joinpath(mktempdir(), "birth.csv")
        open(path, "w") do io
            println(io, "id,birth")
            for (n, i) in enumerate(ids)
                println(io, i, ",", haskey(births, i) ? string(births[i]) : "")
            end
        end
        @test M.read_births(path) == births
    end

    @testset "start-point jitter" begin
        lay = M.ParamLayout(data.S, data.n_years, true, M.PROC_RW1C,
                            M.NO_SP_PIN, true)
        mk(seed, sd) = M.Options(DATA, "nuts", ".", 4, fill(true, 6), true, true,
                                 :stack, 3, 3, 0.8, seed, 1000, M.PROC_RW1C, 0,
                                 "", "", "", -1.0, -1,
                                 M.ACC_LOGITNORMAL, false, M.NO_SP_PIN,
                                 true, M.NO_E_PIN, false, sd, "", 1)
        base = M.initial_theta(lay)
        # off by default
        @test M.start_theta(mk(1, 0.0), lay) == base
        a = M.start_theta(mk(1, 1.0), lay)
        # same for a given seed, different between seeds
        @test a == M.start_theta(mk(1, 1.0), lay)
        @test a != M.start_theta(mk(2, 1.0), lay)
        @test all(a .!= base)
        @test isapprox(M.start_theta(mk(1, 0.5), lay) .- base, 0.5 .* (a .- base))
    end

    @testset "multi-start initialisation" begin
        # Toy objective with two optima in the first coordinate: -2 (height 0)
        # and +3 (height 2). The basins meet at 0.1, so the default start at 0
        # finds the lower one.
        lay = M.ParamLayout(data.S, data.n_years, false, M.PROC_RW1C)
        mk(seed, n) = M.Options(DATA, "map", ".", 4, fill(true, 6), false, true,
                                :stack, 3, 3, 0.8, seed, 1000, M.PROC_RW1C, 0,
                                "", "", "", -1.0, -1,
                                M.ACC_LOGITNORMAL, false, M.NO_SP_PIN,
                                false, M.NO_E_PIN, false, 0.0, "", n)
        base = M.initial_theta(lay)
        ca = zeros(lay.n); ca[1] = -2.0
        cb = zeros(lay.n); cb[1] = 3.0
        negf! = (g, x) -> begin
            z = x .- base
            a = -0.5 * sum(abs2, z .- ca); b = 2.0 - 0.5 * sum(abs2, z .- cb)
            # log(exp(a) + exp(b)), and its gradient
            m = max(a, b); lse = m + log(exp(a - m) + exp(b - m))
            wa = exp(a - lse); wb = exp(b - lse)
            g .= wa .* (z .- ca) .+ wb .* (z .- cb)
            return -lse
        end
        one_ = M.best_start(mk(1, 1), lay, negf!, 200)
        many = M.best_start(mk(1, 12), lay, negf!, 200)
        @test one_.theta[1] - base[1] < 0.0              # the lower optimum
        @test many.theta[1] - base[1] > 2.5              # the higher one
        @test many.logp > one_.logp + 1.5
        # doesn't depend on the seed
        @test M.best_start(mk(7, 12), lay, negf!, 200).theta == many.theta
        # one start = previous behaviour
        @test one_.theta == M.lbfgs(negf!, M.initial_theta(lay); maxiter = 200).theta
    end

    @testset "RNG" begin
        r1 = M.Rng(42); r2 = M.Rng(42)
        @test [M.rand_uniform(r1) for _ in 1:5] == [M.rand_uniform(r2) for _ in 1:5]
        r3 = M.Rng(7)
        xs = [M.rand_uniform(r3) for _ in 1:20000]
        @test all(0.0 .<= xs .< 1.0)
        @test abs(sum(xs) / length(xs) - 0.5) < 0.02
        r4 = M.Rng(9)
        ns = [M.rand_normal(r4) for _ in 1:20000]
        @test abs(sum(ns) / length(ns)) < 0.05
        v = sum(x -> x^2, ns) / length(ns)
        @test abs(v - 1.0) < 0.05
    end
end
