# test2inf

Standalone badger diagnostic HMM inference. One binary, no Julia or R on the
target machine. Executable 3.2 MB, bundle 23 MB.

Two-state absorbing HMM (uninfected → infected) over capture histories, six
imperfect tests, seasonal hazard with a year effect, sensitivity and specificity
either fixed or inferred.

## Install

Download the bundle for your platform from
[releases](https://github.com/EvoArt/test2inf-cli/releases) and unpack it. Ship
the whole `dist/` directory; the runtime libraries beside the executable are
required.

Windows x86-64, Linux x86-64, macOS arm64 and macOS x86-64 are built by CI. To
build elsewhere you need Julia 1.12+:

```sh
./build.sh
```

There is no cross-compilation: a bundle must be built on the platform it runs
on.

## Usage

```sh
test2inf --data captures.csv --out results --method hmc --model all_fixed
```

Input CSV, header required: `time,id,captured,test1..test6`. Empty or
unparseable test fields mean "not tested". Rows with `captured == 0` are
optional; each badger is expanded onto the full timestep grid between its first
and last capture either way.

## Options

| | | default |
|---|---|---|
| `--data <path>` | input CSV | required |
| `--out <dir>` | output directory | `.` |
| `--method <map\|nuts\|hmc>` | inference method | `map` |
| `--model <name>` | pre-tuned HMC parameters | |
| `--tests <list>` | comma-separated assay indices 1-6 | all |
| `--infer-sesp` | estimate Se/Sp | off |
| `--year-process <rw1\|iid>` | year effect process | `rw1` |
| `--no-penalty` | drop the Se+Sp>1 penalty | on |
| `--seasons <int>` | timesteps per year | 4 |
| `--repeat <stack\|pool\|last>` | multiple captures in one timestep | `stack` |
| `--draws <int>` | post-warmup draws | 1000 |
| `--warmup <int>` | warmup iterations | 1000 |
| `--accept <float>` | NUTS target acceptance | 0.8 |
| `--traj-draws <int>` | draws for per-draw outputs | 0 |
| `--metric <path>` | inverse mass matrix CSV | |
| `--write-metric <path>` | export the metric used | |
| `--hmc-eps <float>` | override tuned step size | |
| `--hmc-L <int>` | override tuned trajectory length | |
| `--seed <int>` | RNG seed | 1 |
| `--maxiter <int>` | L-BFGS iteration cap | 1000 |

`--model` is one of `all_fixed`, `all_inferred`, `culture_only_fixed`,
`no_dpp_fixed`, `no_dpp_inferred`.

## Outputs

| file | |
|---|---|
| `parameters.csv` | point estimates, or posterior mean and sd |
| `prevalence.csv` | prevalence per timestep, over captured badgers |
| `p_infected.csv` | `id, time, p_infected` |
| `year_effects.csv` | `year_index, gamma, annual_hazard` |
| `draws.csv` | posterior draws (`nuts`/`hmc`) |
| `prevalence_draws.csv` | `time` + one column per draw (`--traj-draws`) |
| `trajectories.csv` | `id, draw, infection_time`, 0 = never (`--traj-draws`) |

## Metrics

`--model` metrics are tuned for the badger cohort; only the parameter count is
checked. For other data, tune once and reuse:

```sh
test2inf --data d.csv --out r --method nuts --write-metric m.csv
test2inf --data d.csv --out r --method hmc  --metric m.csv --hmc-eps 0.3 --hmc-L 15
```

Metric files are headerless: `n` rows of `n` values, or one row/column of `n`
values for a diagonal metric.

## Tests

```sh
julia --project=. test/runtests.jl
```

`build.sh` also runs every model variant under both methods with
`JULIA_LOAD_CODEGEN_LIB=0` and fails if any does.

## R interface

[BadgerInfeR](https://github.com/EvoArt/BadgerInfeR).

See [DESIGN.md](DESIGN.md) for why the program is built this way.
