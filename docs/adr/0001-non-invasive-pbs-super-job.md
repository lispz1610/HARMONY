# Non-Invasive PBS Super Job Architecture for Variance Investigation

After earlier multi-stage PBS pipelines and invasive C++ diagnostic modifications triggered regressions that required full reverts (`f147b52..be794f7`), we decided to implement the variance investigation plan strictly as a single-allocation PBS batch script ("Super Job") with external driver and post-processing tools, keeping the core HARMONY C++ runtime in `dcl/` 100% frozen and untouched. This guarantees that no cluster toolchain or MPI lifecycle bugs are re-introduced into the runtime, while leveraging existing `[DCL][balance-metrics]` runtime logs, external `nvidia-smi` telemetry, and prebuilt frontend binaries to isolate whether the seed-123 slowdown is an algorithmic or environmental artifact.

Per PR review, minor non-invasive adjustments were made strictly to benchmark drivers (`benchmarks/kneighbor.cpp` and `benchmarks/leanmd.cpp` per Tickets #23 and #25 to reject non-finite stencil outputs and isolate pure loop execution time `loop_time_s` from post-run validation) and preflight testing (`tests/test_async_halo_opencl.cpp` per Ticket #20 to enforce 1 verified OpenCL GPU per MPI rank). The core DCL library runtime remains completely frozen and unmodified.

## Status

accepted

## Considered Options

- **Invasive runtime-level instrumentation**: Modifying core DCL runtime files (`dcl/runtime_impl.hpp`, `dcl/algorithms.hpp`) to emit custom trace CSVs or alter scheduling internals. Rejected because modifying core C++ source code risks re-introducing cluster compilation bugs and MPI lifecycle defects.
- **Multi-stage PBS dependency chain (`qsub -W depend=afterok`)**: Splitting execution across multiple small PBS jobs. Rejected because queue latency, job scheduling delays, and varying node allocations introduce uncontrolled hardware state divergence between runs.
- **Single-allocation non-invasive Super Job**: Retaining the frozen DCL runtime core, executing an untimed warm-up, a rotated 5-repetition matrix, an A/B/A test for seed 123, and defensive Python stdlib post-processing inside a single PBS allocation. Accepted.

## Consequences

- The core runtime library in `dcl/` remains 100% frozen, untouched, and protected from regressions.
- Benchmark drivers `benchmarks/kneighbor.cpp` and `benchmarks/leanmd.cpp` isolate pure computation `loop_time_s` from verification inflation without altering runtime load-balancing behavior (Tickets #23, #25).
- Preflight validation `tests/test_async_halo_opencl.cpp` strictly verifies 1 OpenCL GPU per MPI rank before benchmark execution begins (Ticket #20).
- Iteration-level phase and partition trajectory data are parsed directly from existing `[DCL][balance-metrics]` log entries rather than invasive runtime trace hooks.
- Build procedures (`cluster_build.sh`) continue compiling frontend binaries cleanly without recompilation on compute nodes.
- The entire evaluation completes deterministically on the exact same physical node and GPU state within a single 2-hour PBS reservation.

## Amendment (2026-10-21)

While the core runtime library under `dcl/` remains 100% frozen and untouched, minimal modifications were authorized and applied strictly to benchmark drivers and preflight tests:
- `tests/test_async_halo_opencl.cpp` (Ticket #20): Enforce a minimum of 1 verified OpenCL GPU per MPI rank in preflight validation.
- `benchmarks/kneighbor.cpp` (Tickets #23, #25): Add finite numerical checks (`!std::isfinite(val)`) to reject non-finite/NaN/Inf stencil outputs (Ticket #23), and isolate pure loop execution time (`loop_time_s`) from verification inflation (Ticket #25).
- `benchmarks/leanmd.cpp` (Tickets #23, #25): Add finite numerical checks to reject non-finite force outputs (Ticket #23), and isolate pure loop execution time (`loop_time_s`) from verification inflation (Ticket #25).

No files under `dcl/` were modified.
