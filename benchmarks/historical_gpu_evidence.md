# Historical two-GPU comparison: validated evidence

The original 12 CSVs, 12 logs, comparison summary, recovered metadata, topology record, and PBS output were copied on 2026-10-13 to `Results/2026-10-13-historical-gpu-comparison/`. The 28 source files remain unchanged. SHA-256 values for every copy are in `historical_archive_manifest.json`. The original PBS job ID, `183164.frontend`, is recoverable from the `prun` hostfile path in the preserved PBS output and per-condition logs. This is a recovered identifier, not a fresh scheduler query. The `Results/` tree is ignored by Git.

Run the read-only validator from the repository root:

```sh
python3 benchmarks/analyze_variability.py \
  --run-dir Results/2026-10-13-historical-gpu-comparison \
  --output-dir Results/2026-10-13-historical-gpu-analysis \
  --workloads kneighbor leanmd --algorithms erad hwtopolb \
  --seeds 42 123 456 --iterations 100
```

It accepted exactly 12 `gpu_opencl` conditions and 1,200 ordered iterations. The output consists of `per_condition.csv` and `per_iteration.csv`. Use a fresh output directory on each invocation; existing output is never overwritten. The expected matrix and iteration count are explicit so short pilot runs can use the same command. The original files are read, never edited.

| Workload | Algorithm | Seed | Wall (s) | Ordinary steps (s) | Rebalance steps (s) | First step (ms) | Last step (ms) | Unaccounted (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-neighbor | ERAD | 42 | 3.740 | 1.830 | 0.911 | 20.141 | 20.138 | 0.999 |
| k-neighbor | ERAD | 123 | 3.580 | 1.830 | 0.751 | 20.121 | 20.123 | 0.999 |
| k-neighbor | ERAD | 456 | 3.880 | 1.829 | 1.060 | 20.171 | 20.073 | 0.990 |
| k-neighbor | HWTOPOLB | 42 | 0.221 | 0.040 | 0.122 | 1.560 | 0.412 | 0.059 |
| k-neighbor | HWTOPOLB | 123 | 3.820 | 1.830 | 0.971 | 20.121 | 20.080 | 1.019 |
| k-neighbor | HWTOPOLB | 456 | 0.214 | 0.049 | 0.120 | 0.892 | 0.730 | 0.045 |
| LeanMD | ERAD | 42 | 4.300 | 1.830 | 1.431 | 20.195 | 20.130 | 1.040 |
| LeanMD | ERAD | 123 | 4.330 | 1.829 | 1.471 | 20.146 | 20.077 | 1.030 |
| LeanMD | ERAD | 456 | 4.221 | 1.830 | 1.361 | 20.144 | 20.071 | 1.031 |
| LeanMD | HWTOPOLB | 42 | 0.438 | 0.047 | 0.308 | 1.212 | 0.417 | 0.083 |
| LeanMD | HWTOPOLB | 123 | 4.460 | 1.830 | 1.590 | 20.220 | 20.077 | 1.040 |
| LeanMD | HWTOPOLB | 456 | 0.362 | 0.045 | 0.249 | 1.691 | 0.740 | 0.067 |

The HWTOPOLB seed-123 slow path is visible at iteration 0 in both workloads, before the first scheduled rebalance at iteration 10. Ordinary-step sums are approximately 1.83 s in the slow runs and 0.04–0.05 s in the fast HWTOPOLB runs. The excess is therefore not concentrated in rebalance steps. The `unaccounted_s` column is **wall time minus the sum of recorded step times**. It is not assigned to kernel execution, MPI, validation, or any other phase because this historical run has no measurement that separates those costs. The large remainder near 1 s in each slow run remains unexplained.

These are single historical samples with no warm-up or controlled order. They establish the onset and accounting gap, not a repeatable speedup or a cause for the variance. The topology JSON was measured separately and was not consumed by the benchmark algorithms.
