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
                         "", "", "", -1.0, -1)
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
                          "", "", "", -1.0, -1)
        M.draw_outputs(opts2, data, layout, res.draws, 0)
        @test !isfile(joinpath(out2, "prevalence_draws.csv"))
        @test !isfile(joinpath(out2, "trajectories.csv"))
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
