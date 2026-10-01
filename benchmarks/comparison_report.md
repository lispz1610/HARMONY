# ERAD and HWTOPOLB comparison

The earlier timing table in this file came from a local CPU simulation with incomplete wall-time accounting. It is historical and does not establish a GPU, network, halo, or migration speedup. No hardware performance conclusion should be drawn from those numbers.

Run `benchmarks/compare_algos.sh` to produce a new comparison. The script repeats each algorithm with seeds 42, 123, and 456, and writes `comparison_summary.csv`, per-run CSV and logs, and `comparison_metadata.json`. The summary uses the maximum total wall time across MPI ranks for each run. It records the execution mode so CPU simulation and OpenCL runs remain distinguishable. The metadata records the commit, dirty-tree status, host, compiler, flags, launcher, input size, iteration count, and seeds. There are no warm-up iterations.

For an OpenCL performance claim, run the same input and rank configuration on the same hardware, validate the output, and compare repeated timings against a baseline from that environment. Record the GPU model, driver, platform, rank-to-device assignment, and any CPU fallback. The current local environment has no OpenCL platform, so the local smoke tests exercise the CPU simulation only.
