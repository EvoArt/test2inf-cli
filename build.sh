#!/usr/bin/env bash
# Build the trimmed test2inf binary and a slim, portable bundle.
#
# Requires Julia 1.12+ (trim does not exist before it) and a C toolchain. On
# Windows juliaup's build provides MinGW; elsewhere the system compiler is used.
#
# Runs on Windows, Linux and macOS. There is NO cross-compilation: JuliaC builds
# for the host, so each platform's bundle must be built on that platform. That is
# what .github/workflows/build-cli.yml exists for.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) PLATFORM=windows; EXE=test2inf.exe ;;
  Darwin)               PLATFORM=macos;   EXE=test2inf ;;
  *)                    PLATFORM=linux;   EXE=test2inf ;;
esac
echo "Platform: $PLATFORM"

# Find a Julia that is actually 1.12+, rather than the first one on PATH: a
# machine can easily have an older default (this one has 1.10 on PATH and
# 1.12 under juliaup). Candidates in order: $JULIA, PATH, juliaup installs.
julia_ok() {
  [ -x "$1" ] || return 1
  case "$("$1" --version 2>/dev/null)" in
    *" 1.1"[2-9]*|*" 1."[2-9][0-9]*) return 0 ;;
    *) return 1 ;;
  esac
}
if [ -z "${JULIA:-}" ]; then
  for cand in "$(command -v julia || true)" \n              "$HOME"/.julia/juliaup/julia-1.1[2-9]*/bin/julia \n              "$HOME"/.julia/juliaup/julia-1.1[2-9]*/bin/julia.exe; do
    if julia_ok "$cand"; then JULIA="$cand"; break; fi
  done
fi
JULIA="${JULIA:-julia}"
if [ ! -x "$JULIA" ]; then
  echo "Julia 1.12+ not found (tried PATH and the juliaup default)" >&2
  echo "Install with: juliaup add 1.12   (then set JULIA=...)" >&2
  exit 1
fi

VER="$("$JULIA" --version)"
echo "Using $VER"
case "$VER" in
  *" 1.1"[2-9]*|*" 1."[2-9][0-9]*) ;;
  *) echo "Need Julia 1.12 or newer for --trim; got: $VER" >&2; exit 1 ;;
esac

# One-off: an environment holding JuliaC itself.
if [ ! -d juliac-env ]; then
  echo "Creating juliac-env..."
  "$JULIA" -e 'using Pkg; Pkg.activate("juliac-env"); Pkg.add("JuliaC")'
fi

# Build for a portable CPU baseline, not the build machine's own silicon.
#
# Without this, Julia targets the host microarchitecture and bakes it into the
# image. A CI runner on AMD Zen 3 produced a bundle that refused to start on an
# Intel Skylake laptop with "Rejecting this target due to use of
# runtime-disabled features" -- a hard failure, on a machine with no way to
# diagnose it. `generic` is the portable baseline; the extra targets let newer
# CPUs still pick a faster path at load time, and `clone_all` makes each one a
# complete copy rather than a partial specialisation.
: "${JULIA_CPU_TARGET:=generic;sandybridge,clone_all;haswell,clone_all;skylake-avx512,clone_all}"
export JULIA_CPU_TARGET
echo "CPU target: $JULIA_CPU_TARGET"

echo "Building (trim=safe)..."
"$JULIA" --project=juliac-env -e 'using JuliaC; JuliaC.main(ARGS)' -- \
  --output-exe test2inf \
  --bundle build_cli \
  --trim=safe \
  --experimental \
  ./Test2InfCLI

# JuliaC bundles the whole Julia runtime regardless of what the trimmed code
# reaches (JuliaC#129). Nothing here calls BLAS, LAPACK, GMP/MPFR or the C++
# runtime, so those are dropped: 114 MB -> ~23 MB. Verified below.
# NO slimming. The bundle ships whatever JuliaC produced.
#
# Two attempts at trimming it failed, and both failures were invisible on a
# developer machine:
#
#   1. A hand-maintained "these are unused" list deleted libgmp, libmpfr,
#      libblastrampoline, libgfortran and libopenblas64_ -- all of which the
#      executable actually IMPORTS, because Julia links them into every build
#      whether or not the program calls BLAS.
#   2. Keeping exactly the executable's imports deleted their TRANSITIVE
#      dependencies (libgcc_s_seh, libwinpthread, libuv, libz, ...).
#
# Both produced a bundle that ran fine locally -- the loader found the missing
# DLLs on PATH, because a developer machine has Julia installed -- and failed
# on a clean CI runner with exit 127 and no output whatsoever. The 23 MB figure
# quoted in the README was never a self-contained bundle.
#
# A correct slimmer would need the full transitive closure of the import graph.
# That is a real tool, not a shell loop, and until someone writes one a
# ~120 MB bundle that works everywhere beats a 23 MB one that works only where
# Julia is already installed.
echo "Packaging bundle..."
rm -rf dist
cp -r build_cli dist
# Never ship a bundle that has not been run. A trimmed binary can link cleanly
# and still die at runtime on a method that was trimmed away, so exercise every
# code path with codegen disabled -- any JIT fallback then fails loudly.
#
# All five `sql e2e` variants run under both methods, not just three modes: the
# analytic gradient now serves the inferred-Se/Sp models too, so its Se/Sp
# reverse block is on the shipped path and has to be exercised here.
echo "Verifying..."
if [ -f testdata/sim.csv ]; then
  fail=0
  # A bundle that cannot start at all fails every variant identically with no
  # stderr, which is indistinguishable from a model bug in the output above.
  # Check that first, and say what is actually in the bundle if it happens.
  # `$?` inside the `if` body reports the LAST command, not the tested one, so
  # capture the status explicitly. `set -e` is off for this call deliberately:
  # a non-zero exit here is the thing being measured, not an error.
  set +e
  JULIA_LOAD_CODEGEN_LIB=0 "./dist/bin/$EXE" --help >.verify.out 2>.verify.err
  start_status=$?
  set -e
  if [ "$start_status" -ne 0 ]; then
    echo "  FAILED: the binary will not start at all (exit $start_status)." >&2
    echo "  --- stdout ---" >&2
    sed 's/^/    /' .verify.out >&2
    echo "  --- stderr ---" >&2
    sed 's/^/    /' .verify.err >&2
    echo "  --- dist/bin ---" >&2
    ls -la dist/bin | sed 's/^/    /' >&2
    [ -d dist/lib ] && { echo "  --- dist/lib ---" >&2; ls -la dist/lib | sed 's/^/    /' >&2; }
    rm -f .verify.err .verify.out
    exit 1
  fi
  rm -f .verify.out
  run_variant() {  # usage: run_variant <name> <cli args...>
    name="$1"; shift
    if JULIA_LOAD_CODEGEN_LIB=0 "./dist/bin/$EXE" \
         --data testdata/sim.csv --out .verify "$@" >/dev/null 2>.verify.err; then
      echo "  ok: $name"
    else
      echo "  FAILED: $name" >&2
      sed 's/^/    /' .verify.err >&2
      fail=1
    fi
  }
  for method in map nuts; do
    if [ "$method" = nuts ]; then
      set -- --draws 50 --warmup 50
    else
      set --
    fi
    run_variant "$method: all tests (fixed)"    --method $method "$@"
    run_variant "$method: all tests (inferred)" --method $method "$@" --infer-sesp
    run_variant "$method: culture only (fixed)" --method $method "$@" --tests 3
    run_variant "$method: no DPP (fixed)"       --method $method "$@" --tests 1,2,3,5,6
    run_variant "$method: no DPP (inferred)"    --method $method "$@" --tests 1,2,3,5,6 --infer-sesp
  done
  # Both year processes must trim: rw1 is the default and iid is the fallback,
  # and they take different branches in both the likelihood and the gradient.
  for yp in iid rw1 rw2 none shrunk; do
    run_variant "$yp year process" --method map --year-process "$yp"
  done
  run_variant "iid + inferred"   --method map --year-process iid --infer-sesp
  # HMC needs a --model whose parameter count matches the data. The shipped
  # sets are tuned on the real cohort (58/70 par); testdata/sim.csv has 13/25,
  # so the pre-tuned path cannot be exercised on it. Verify the explicit
  # eps/L override instead, which is the same sampler minus the lookup.
  run_variant "nuts + traj-draws"   --method nuts --draws 50 --warmup 50 --traj-draws 10
  # A metric supplied as a FILE, which is how a user tunes for their own data
  # without rebuilding -- the compiled-in metrics are only a default. Round
  # trip it: export from a nuts run, then sample with it under hmc.
  run_variant "nuts: --write-metric" --method nuts --draws 30 --warmup 30               --write-metric .metric_probe.csv
  if [ -s .metric_probe.csv ]; then
    run_variant "hmc: --metric from file" --method hmc --draws 30 --warmup 30                 --hmc-eps 0.2 --hmc-L 5 --metric .metric_probe.csv
  else
    echo "  FAILED: --write-metric produced no file" >&2
    fail=1
  fi
  rm -f .metric_probe.csv
  run_variant "hmc: explicit eps/L" --method hmc --draws 50 --warmup 50               --hmc-eps 0.3 --hmc-L 7 --model all_fixed || true
  rm -rf .verify .verify.err
  if [ "$fail" -ne 0 ]; then
    echo "Bundle verification FAILED; not fit to ship." >&2
    exit 1
  fi
  echo "All five variants ran under both methods."
else
  echo "WARNING: testdata/sim.csv missing; bundle NOT verified." >&2
  exit 1
fi

echo
echo "exe:    $(du -h "dist/bin/$EXE" | cut -f1)"
echo "bundle: $(du -sh dist | cut -f1)"
echo "Distribute the whole dist/ directory."
