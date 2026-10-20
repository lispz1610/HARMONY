# Non-Invasive PBS Super Job Architecture for Variance Investigation

After earlier multi-stage PBS pipelines and invasive C++ diagnostic modifications triggered regressions that required full reverts (`f147b52..be794f7`), we decided to implement the variance investigation plan strictly as a single-allocation PBS batch script ("Super Job") with external driver and post-processing tools, leaving all HARMONY C++ core code completely intact. This guarantees that no cluster toolchain or MPI lifecycle bugs are re-introduced, while leveraging existing `[DCL][balance-metrics]` runtime logs, external `nvidia-smi` telemetry, and prebuilt frontend binaries to isolate whether the seed-123 slowdown is an algorithmic or environmental artifact.

## Status

accepted

## Considered Options

- **Invasive source-level instrumentation**: Modifying `kneighbor.cpp` and `leanmd.cpp` to emit custom trace CSVs as originally sketched in Task 2. Rejected because modifying C++ source code risks re-introducing cluster compilation bugs and MPI lifecycle defects.
- **Multi-stage PBS dependency chain (`qsub -W depend=afterok`)**: Splitting execution across multiple small PBS jobs. Rejected because queue latency, job scheduling delays, and varying node allocations introduce uncontrolled hardware state divergence between runs.
- **Single-allocation non-invasive Super Job**: Retaining the clean C++ codebase, executing an untimed warm-up, a rotated 5-repetition matrix, an A/B/A test for seed 123, and defensive Python stdlib post-processing inside a single PBS allocation. Accepted.

## Consequences

- All C++ source files (`dcl/`, `benchmarks/*.cpp`, `main*.cpp`) and build procedures (`cluster_build.sh`) remain untouched and protected from regressions.
- Iteration-level phase and partition trajectory data are parsed directly from existing `[DCL][balance-metrics]` log entries rather than new trace hooks.
- The entire evaluation completes deterministically on the exact same physical node and GPU state within a single 2-hour PBS reservation.
