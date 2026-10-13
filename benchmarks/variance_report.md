# Two-GPU variance investigation: pending cluster validation

## Current evidence and correctness

The historical run is archived read-only under `Results/2026-10-13-historical-gpu-comparison/`; file hashes and the recovered PBS ID `183164.frontend` are in `benchmarks/historical_archive_manifest.json`. The archived output and reconstruction are summarized in `benchmarks/historical_gpu_evidence.md`. All 12 historical condition files report `gpu_opencl` and contain 100 ordered iterations. The earlier two-rank asynchronous OpenCL halo reference was reported as passing in the investigation plan, but the original run metadata lacks a verifiable per-rank physical GPU fingerprint, compiler, revision, and complete environment record. Those missing facts remain unknown.

The k-neighbor tolerance is absolute error at most `1e-5`. The LeanMD tolerance is `1e-4 + 1e-4 * abs(reference)`. Both require finite output and reference values. A fresh two-rank halo reference result must be captured with the controlled full run before drawing a performance conclusion. No local CPU fallback validates the OpenCL kernels or physical GPU mapping.

## Historical performance finding

For both workloads, every ERAD run and the HWTOPOLB seed-123 run enters a roughly 20 ms ordinary-step path at iteration 0. The first scheduled rebalance is at iteration 10, so later partition movement cannot explain the **onset**. The archived full-wall time has an unexplained remainder of roughly one second in each slow run after recorded step times are summed. The old logs do not isolate that host interval. OpenCL event times are a separate, potentially overlapping device timeline. These are single historical samples; no useful-gain or causal claim follows from them.

The evaluated HWTOPOLB variant uses throughput and seeded perturbation. It does **not** consume the measured topology JSON. Any topology-aware performance claim requires a separate implementation and experiment.

## Controlled report command

After completing the pilot, two-rank diagnostic trace, unprofiled full matrix, and halo reference under PBS, generate a report from the preserved full-run directory:

```sh
python3 benchmarks/report_variability.py \
  --run-dir "$PWD/Results/full-YYYYMMDD-unique" \
  --output "$PWD/Results/full-YYYYMMDD-unique-report.md"
```

The command reads the `run_manifest.json`, four warm-ups, 60 full samples, and the full-run `halo_reference` record. It validates each sample's CSV, log, inventory, build digest, strict GPU mapping, reference result, order, and unique paths before reporting the 30 adjacent pairs. It calculates each ERAD/HWTOPOLB loop-time ratio within its pair and reports median, min/max, and Tukey-hinge quartiles for each workload and seed. The raw paired table keeps the primary complete-loop time, secondary full-wall time, post-loop validation time, and descriptive `loop minus sum of step maxima` gap separate. MPI maxima may come from different ranks, so that gap is not an additive phase residual and may be negative. The command refuses to overwrite an existing output file. The pilot and traced diagnostic runs never enter the headline matrix.

A useful gain requires every workload/seed median paired ratio to reach `1/0.9`, with stable repetitions. The predeclared stability check requires `(Q3-Q1)/median <= 0.10` and `Q1 >= 1/0.9`; a near-threshold or unstable group calls for more rotated samples or an inconclusive conclusion. A seed-level reversal blocks a general gain claim. If seed 123 remains anomalous, run a separate same-allocation ERAD/HWTOPOLB/ERAD diagnostic sequence; preserve its exact order and raw artifacts outside the paired estimate.

The current report generator does **not** parse rank-local diagnostic traces. It therefore leaves the cause and phase reconciliation inconclusive: loop time and per-step maxima are not necessarily measured on the same rank, and OpenCL event times cannot be added to host phases. Once the trace schema and cluster data exist, the final report must cite exact PBS run, sample, rank, iteration, phase, and old/new partition offsets and counts. If measured rank-local host phases fail to reconcile with that rank's loop, state the missing measurement and do not attribute the remainder to a particular component.

## Outstanding PBS evidence

- Complete rotated pilot before rebalance and its analysis.
- One small two-rank GPU trace smoke, a matching unprofiled invocation, and a validated complete trace from both ranks.
- Full unprofiled matrix on two ranks with one verified physical GPU per rank, all 64 invocations accepted, plus a fresh two-rank halo reference PASS in the same run manifest.
- Hardware, compiler and flags, MPI, node/affinity, OpenCL and driver, clocks/power/contention, input, warm-up, repeated timings, and same-environment baseline; mark any unmeasured optional field unavailable.
- Isolated seed-123 A/B/A if the rotated matrix still shows the anomaly.

These gates remain unvalidated in this workspace because PBS and the two-GPU cluster are unavailable here. Do not replace this pending status with a conclusion until the raw cluster artifacts have been reviewed.
