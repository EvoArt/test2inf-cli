# test2inf — standalone badger HMM inference

A single portable binary that runs the badger diagnostic HMM. No Julia
installation, no R, no package environment needed on the target machine.

- **Executable:** 3.2 MB
- **Bundle:** 23 MB (ship the whole `dist/` directory)
- **Runtime deps:** none
- **Platforms:** Windows, Linux and macOS, each built by CI on its own runner

## Why this exists rather than trimming `test2infeR`

`juliac --trim=safe` cannot compile the existing engine. Two hard blockers,
both verified rather than assumed:

1. **Turing → Optimization → SciMLBase → FunctionWrappers.** FunctionWrappers is
   documented as untrimmable, and roughly 688 packages depend on it. There is a
   replacement (`TypedCallable`) in an unlanded upstream PR.
2. **ForwardDiff produces a binary that passes the trim verifier with zero
   errors and then crashes at runtime** with a `MethodError` on the
   `Dual`-argument objective. Concrete chunk sizes, generic signatures and an
   explicit `Base.Experimental.entrypoint` all failed to fix it.

So the inference kernel is reimplemented here with no dependencies beyond
`Base`, and the scope is deliberately narrow: the model variants the badger
project actually fits, one optimiser (L-BFGS) and two samplers (NUTS, and
fixed-length HMC with a pre-tuned dense metric).

Gradients are an exact analytic reverse sweep rather than AD. On the real
cohort (3,224 badgers, 29,737 timepoints, 58 parameters) that is 1.63 ms
against 6.02 ms for the best AD backend measured through Turing, and 29.5 ms
for ForwardDiff. End to end, a five-model fit went from about an hour per model
to roughly three minutes for all five.

## What it computes

The same model as `Test2InfEngine`: a two-state (uninfected → infected,
absorbing) hidden Markov model over badger capture histories, with six
diagnostic assays contributing emissions at each capture, a seasonal log-hazard
and an iid year effect.

| Component | Implementation |
|---|---|
| Gradient (default model) | **Exact analytic reverse sweep** ([grad_analytic.jl](Test2InfCLI/src/grad_analytic.jl)) |
| Gradient (`--infer-sesp`) | Hand-rolled forward-mode dual numbers ([dual.jl](Test2InfCLI/src/dual.jl)) |
| Optimiser | L-BFGS + Armijo backtracking ([optimize.jl](Test2InfCLI/src/optimize.jl)) |
| Sampler | Multinomial NUTS, dual averaging, diagonal metric ([nuts.jl](Test2InfCLI/src/nuts.jl)) |
| RNG | xoshiro256++ with Box–Muller normals ([rng.jl](Test2InfCLI/src/rng.jl)) |
| Model | Log-posterior + forward-backward ([model.jl](Test2InfCLI/src/model.jl)) |
| CSV | Hand-written reader/writer ([io.jl](Test2InfCLI/src/io.jl)) |

Year process is **iid** (`gamma_y = sigma_g * z_y`), matching the package
default and the validated reference fit.

## Usage

```
test2inf --data <csv> [options]
```

Input CSV needs a header and these columns in order:
`time, id, captured, test1, test2, test3, test4, test5, test6`.
Use `NA` (or any unparseable value) for a missing test result.

| Option | Default | Meaning |
|---|---|---|
| `--method map\|nuts` | `map` | Point estimate or full posterior |
| `--out <dir>` | `.` | Output directory |
| `--seasons <int>` | 4 | Timesteps per year |
| `--tests <list>` | all | Comma-separated assay indices, e.g. `--tests 4` |
| `--infer-sesp` | off | Estimate Se/Sp instead of fixing at Table 1 |
| `--no-penalty` | off | Drop the Se+Sp>1 identifiability penalty |
| `--repeat stack\|pool\|last` | `stack` | Repeat-capture handling |
| `--draws` / `--warmup` | 1000 / 1000 | NUTS iterations |
| `--accept <float>` | 0.8 | NUTS target acceptance |
| `--seed <int>` | 1 | RNG seed |
| `--maxiter <int>` | 1000 | L-BFGS cap |

### Outputs

`prevalence.csv`, `parameters.csv`, `p_infected.csv`, `year_effects.csv`, and
for NUTS also `draws.csv` (raw posterior draws for downstream diagnostics).

### Examples

```sh
# MAP, all six assays, Se/Sp fixed at Table 1  (model 1)
test2inf --data captures.csv --out results/

# MAP with Se/Sp inferred  (model 2)
test2inf --data captures.csv --out results/ --infer-sesp

# Culture only, Se/Sp fixed  (model 3)
test2inf --data captures.csv --out results/ --tests 4

# Full posterior
test2inf --data captures.csv --out results/ --method nuts --draws 2000 --warmup 1000
```

Note the handoff's warning still applies: culture-only **with** `--infer-sesp`
is not identifiable and the posterior is bimodal. That combination is not
blocked by the CLI, but the result is not a finding.

## Building

```sh
./build.sh
```

Needs Julia 1.12+ (`juliaup add 1.12`) and a MinGW C compiler. Override the
interpreter with `JULIA=/path/to/julia ./build.sh`.

The script builds with `--trim=safe`, strips the unused runtime DLLs
(114 MB → 23 MB), then **runs all three modes with `JULIA_LOAD_CODEGEN_LIB=0`**
before declaring success. That last step is not optional: a trimmed binary can
link cleanly and still die on a method that was trimmed away, and disabling
codegen makes any JIT fallback fail loudly instead of silently masking it.

## Performance

Wall clock for the trimmed binary on the 200-badger / 1,105-capture simulated
set, including process start (there is no JIT warmup):

| Task | Time |
|---|---|
| MAP, default model | **0.45 s** |
| MAP, `--infer-sesp` | **0.33 s** |
| NUTS, 1000 warmup + 1000 draws | **2.9 s** |

### Gradients

The default model (Se/Sp fixed) uses an **exact analytic gradient**: a single
reverse sweep over the same forward recursion, derived by hand. Because Se/Sp
are constants there, the emission factors contribute no partials, and only the
hazard and the initial state do — which is what makes the derivation tractable.

| Params | logp | forward AD | analytic | speedup |
|---|---|---|---|---|
| 13 (default) | 33.6 µs | 138.3 µs (4.12x logp) | **66.8 µs (1.99x logp)** | **2.07x** |

`1.99x logp` is the textbook cost of reverse mode, and unlike forward mode it is
**independent of the parameter count** — so the advantage grows with the number
of year effects. On a study with ~51 years (~58 parameters) forward mode would
be far behind; here it is 2x.

End to end this took **NUTS from 7.6 s to 2.9 s (2.6x)**.

The advantage survives realistic sparsity. Re-measured on a fixture matching the
real cohort's profile (78.2% NaN cells, 51.2% of timepoints with no test):

| Fixture | emissions | logp | forward AD | analytic | speedup |
|---|---|---|---|---|---|
| dense (27.8% NaN) | 4,788 | 32.8 µs | 135.8 µs (4.14x) | 61.4 µs (1.87x) | 2.21x |
| sparse (78.2% NaN) | 1,397 | 26.2 µs | 81.0 µs (3.09x) | 40.6 µs (1.55x) | 2.00x |

Both fixtures are in the test suite, and the analytic gradient agrees with the
AD to 1e-15 on each.

There is also a robustness argument, not just a speed one: a peer session
benchmarking reverse-mode AD backends on the full cohort had **Mooncake crash
with `ReadOnlyMemoryError`** partway through a 4x1000 NUTS run, after two of
three models had completed. An analytic gradient has no AD library to fail.

`--infer-sesp` (25 params) still uses forward-mode AD; no analytic derivation
for the Se/Sp block is implemented. Both paths are compiled into the binary and
selected automatically, and the test suite asserts they agree to **1e-15**.

### Forward-mode AD vs ForwardDiff

Benchmarked on identical model code, since the hand-rolled dual exists only
because ForwardDiff does not trim:

| Params | ours | ForwardDiff (best) | ratio |
|---|---|---|---|
| 13 | 138 µs (chunk 13) | 144 µs (chunk 13) | 0.96x |
| 25 | 275 µs (chunk 13) | 266 µs (chunk 25) | 1.03x |

Parity — the workaround costs nothing.

**Scale caveat, and it matters.** These are 200-badger numbers, and this
project's CSR layout (`ptr`/`vals`/`tidx`, no `isnan` in the hot loop) buys less
here than on the real data: the dense fixture is only 27.8% sparse against the
cohort's 79%, so the compaction step is closer to its floor. On the real
cohort (3,224 badgers, 29,737 timepoints, ~51 year effects) a separate factorial
benchmark in `gridded/HMM_PERFORMANCE.md` finds reverse-mode backends beat
ForwardDiff by ~3x on the same likelihood (19.8 ms → 6.1 ms, Mooncake). Forward
mode's disadvantage grows with both dataset size and parameter count, so:

- for the **default model** the analytic gradient here is reverse mode, and
  should hold up at scale — on that cohort's 58 parameters a ~2x-primal sweep
  would be ~2.5 ms against their best measured backend's 6.1 ms;
- for **`--infer-sesp`** the forward-mode fallback will lose ground on a large
  cohort. Deriving the Se/Sp block analytically is the obvious next step if that
  path matters on real data.

**One trap worth knowing:** `ntuple(f, N)` where `N` is a type parameter stops
unrolling above 8 elements and falls back to an allocating path — chunk 12 cost
**86 ms**, a 400x cliff. `ntuple(f, Val(N))` is the unrolled form. Every `ntuple`
in `dual.jl` uses `Val(N)`.

## Validation

Run `julia test/runtests.jl` (34 assertions).

- **Analytic gradient vs forward-mode AD:** agree to **1e-15** across five
  parameter points — two independent derivations of the same quantity.
- **Gradient vs central finite differences:** max relative error **2.7e-7**
  (13 params) and **9.0e-7** (25 params) — finite-difference precision.
- **Both gradient paths reach the same MAP optimum** (log posterior to 1e-6,
  parameters to 1e-4).
- **Trimmed vs JIT:** all output files byte-identical for MAP and for NUTS
  (same step size, divergence count and acceptance rate).
- **Parameter recovery** on simulated data with known truth, `--infer-sesp`:

  | Assay | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|
  | Se true | 0.407 | 0.407 | 0.100 | 0.650 | 0.809 | 0.492 |
  | Se est. | 0.406 | 0.398 | 0.140 | 0.600 | 0.824 | 0.489 |

## Limitations

- **Year process is iid only.** `rw1`/`rw2`/`ar1`/`shrunk`/`none` are not built
  in. The handoff shows per-badger P(infected) correlates at r ≥ 0.994 between
  every pair of variants, so this is a defensible restriction, not a silent one.
- **No trajectory draws or prevalence CIs on the MAP path.** The R engine's
  `prevalence_trajectory_draws` is not ported; NUTS gives posterior uncertainty
  instead.
- **Single chain.** No `MCMCThreads` equivalent; run separate seeds for R-hat.
- **No chain caching / serialization.**
- **Errors abort with a raw Julia message.** `sprint(showerror, e)` is not
  statically resolvable under trimming, so there is no tidy error handler.
- **`--infer-sesp` has no analytic gradient** and falls back to forward-mode AD,
  which will be the slow path on a large cohort.
- **Bundle slimming is manual** and tuned to this program's actual calls. If you
  add anything touching `LinearAlgebra` factorizations, BLAS must go back in —
  re-verify after any dependency change.
