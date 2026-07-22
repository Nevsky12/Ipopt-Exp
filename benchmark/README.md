# Ipopt performance benchmarks

These benchmarks are opt-in so normal builds and correctness tests stay fast.
They emit machine-readable JSON and check that the measured callback was
actually executed.

Configure a dedicated release build:

```sh
mkdir build-benchmark
cd build-benchmark
../configure --enable-benchmarks \
  CXXFLAGS="-O3 -DNDEBUG -std=c++23 -march=native"
make -j
```

Run the benchmark directly:

```sh
make benchmark
```

This runs both the fixed-variable Hessian and objective-gradient adapter
microbenchmarks.

For reproducible metadata, CPU pinning, single-threaded math-library settings,
and a regression gate, use the runner from the source tree:

```sh
python3 benchmark/run_benchmarks.py \
  --build-dir build-benchmark \
  --benchmark all \
  --cpu 4 \
  --output benchmark-current.json
```

The runner defaults to the Hessian benchmark for compatibility with existing
baselines. Select `--benchmark gradient` for the gradient path, or
`--benchmark all` to run both into one result document.

Compare a later run with the saved result and fail with exit status 3 if the
median regresses by more than 5%:

```sh
python3 benchmark/run_benchmarks.py \
  --build-dir build-benchmark \
  --benchmark all \
  --cpu 4 \
  --baseline benchmark-baseline.json \
  --output benchmark-current.json \
  --max-regression-percent 5
```

Use the same physical core, compiler, flags, dependencies, solver options, and
machine state for both runs. The current benchmarks isolate the fixed-variable
mapping paths in `TNLPAdapter::Eval_h` and `TNLPAdapter::Eval_grad_f`; they do
not stand in for an end-to-end solve benchmark. End-to-end CUTEst and
structured KKT cases live in `fatrop-research/fatrop_implicit/research` and
should be run as a separate required gate for algorithmic changes.

To compare two ABI-compatible Ipopt shared libraries with that repository's
decoded CUTEst adapter, use the interleaved process runner:

```sh
python3 benchmark/run_cutest_comparison.py \
  --executable /tmp/fatrop-build/research/cutest_ipopt_benchmark \
  --cases-dir ~/projects/fatrop-research/research-deps/problems \
  --baseline-library-dir /usr/local/lib \
  --candidate-library-dir /tmp/ipopt-build/src/.libs \
  --fatrop-executable /tmp/fatrop-build/research/dtoc_fatrop_benchmark \
  --fatrop-source-root ~/projects/fatrop-research/fatrop_implicit \
  --repetitions 7 \
  --warmup 1 \
  --cpu 4 \
  --output cutest-comparison.json
```

To compare two different executables, such as the ordinary CUTEst adapter and
the C++23 DTOC3 stage candidate built from the same stable Ipopt library, use:

```sh
python3 benchmark/run_cutest_comparison.py \
  --baseline-executable /tmp/fatrop-build/research/cutest_ipopt_benchmark \
  --candidate-executable /tmp/fatrop-build/research/cutest_ipopt_stage_benchmark \
  --cases-dir ~/projects/fatrop-research/research-deps/problems \
  --case DTOC3_N50 \
  --case DTOC3_N5000 \
  --case DTOC3_N30000 \
  --repetitions 7 \
  --warmup 1 \
  --cpu 4 \
  --output dtoc3-stage-candidate.json
```

The runner rotates baseline, candidate, and optional FATROP order, fixes common
BLAS thread counts to one, records the resolved `libipopt` and both source
commits, and rejects status, iteration, objective, dimension, or feasibility
differences before reporting median wall time. `candidate_over_fatrop` is the
candidate Ipopt time divided by FATROP time, so values above one mean FATROP
was faster. FATROP has a slightly different equality-row formulation for these
DTOC cases; the gate therefore requires matching primal dimension and solution
quality rather than identical constraint counts.

Schema 3 also records every extra numeric CSV field as per-run telemetry and a
median for fields ending in `_ms`. When candidate-first counters are present,
the runner requires positive requests, request/accept equality, zero fallback,
and zero backend-factory failures in every candidate sample.

The DTOC3 stage executable exercises the real variable-block kernel, equality
full-KKT assembler, prepared workspaces, lazy backend factory, exact callback
cache, and both inner and outer true-residual gates. Its pinned results are in
`graphify-out/dtoc3-stage-candidate-benchmark.json`: the timed Optimize call is
7.86%--10.83% faster for N=50, 5000, and 30000. Blueprint, layout, provider,
and solver-workspace preparation occurs before that timer and is recorded
separately; including it makes a cold single large solve slower, with
break-even at the third repeated solve in the measured N=5000 and N=30000
cases.

The CAR2 stage route is the representative nonlinear inequality gate. It
exercises the full eight-block bordered assembler on 1000 stages with path
inequalities, primal and slack bounds, one global primal, dense non-identity
next-state Jacobians, and adjacent-stage Hessian blocks. In five interleaved
CPU-pinned Release samples, all 171 candidate requests were accepted with zero
fallback and a final maximum constraint violation of `1.63e-13`. Its median
Optimize time was 2124.192 ms versus 1931.988 ms for the coordinate-matched
MUMPS baseline: 9.95% slower despite taking 171 rather than 236 iterations.
The exact record is `graphify-out/car2-bordered-stage-candidate-benchmark.json`.

Both CUTEst stage routes still infer topology only for recognized DTOC3 and
CAR2 names; production integration requires explicit public TNLP stage
metadata. CAR2 also requires `make_constraint`, no NLP scaling, and an exact
Hessian, and the bordered restoration transform is not implemented. The
separate non-bordered `RestoIpoptNLP` topology transform continues to pass its
live forced-restoration correctness gate. Synthetic kernel medians and
limitations are in `graphify-out/block-tridiagonal-benchmark.json`.
