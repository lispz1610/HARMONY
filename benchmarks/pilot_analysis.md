# Pre-rebalance two-GPU pilot: manual cluster validation

The pilot is a diagnostic experiment for issue #8. It uses two rotated passes of eight iterations, so no scheduled rebalance at iteration 10 occurs. Its 24 invocations form 12 adjacent ERAD/HWTOPOLB pairs. Pilot samples must stay out of the full 100-iteration performance comparison.

On the cluster frontend, build the binaries and inspect the planned order before submitting:

```sh
bash cluster_build.sh
python3 benchmarks/run_variability.py --dry-run --mode pilot \
  --run-dir "$PWD/Results/pilot-YYYYMMDD-unique"
```

Choose a **new** run directory for each submission. The preview prints every invocation and expected artifact without launching MPI or creating the directory. Then submit the two-rank PBS wrapper:

```sh
export HARMONY_VARIABILITY_MODE=pilot
export HARMONY_VARIABILITY_RUN_DIR="$PWD/Results/pilot-YYYYMMDD-unique"
export HARMONY_PBS_OUTPUT_PATH="${HARMONY_VARIABILITY_RUN_DIR}.pbs.out"
test ! -e "$HARMONY_VARIABILITY_RUN_DIR"
test ! -e "$HARMONY_PBS_OUTPUT_PATH"
qsub -o "$HARMONY_PBS_OUTPUT_PATH" -j oe -V job_harmony_variability.pbs
```

Keep the PBS output and scheduler exit status with the run directory as described in `docs/controlled-gpu-variance-run.md`. Inspect the job exit status and `run_manifest.json`; every sample must be `accepted`, the run must be `completed`, both ranks must have verified physical GPU identities and stable mapping, and the numerical reference checks must pass. The analyzer also checks that the run baseline mapping matches the sample inventories, including each rank's CPU affinity or its explicit unavailability reason. The runner preserves per-sample CSV, log, inventory, and manifest artifacts. If any sample fails, retain the failed directory and submit a fresh unique directory after fixing the cause. Do not replace partial artifacts.

After a completed pilot, generate its evidence report from the preserved raw files:

```sh
python3 benchmarks/analyze_pilot.py \
  --run-dir "$PWD/Results/pilot-YYYYMMDD-unique" \
  --output "$PWD/Results/pilot-YYYYMMDD-unique-analysis.md"
```

The analyzer independently checks the 24 accepted invocations and their successful launcher exits, order rotation, pre-rebalance iteration sequence, `gpu_opencl` mode, finite host timings, successful validation, physical-device mapping, and manifest/log consistency. It refuses a full or incomplete run and never overwrites an existing report. The report lists iteration-0 latency, earliest slow ordinary iteration, per-invocation order, and group summaries by seed, algorithm, pass, and run position. A 10 ms threshold is descriptive of the historical slow path, not proof of cause. Two short passes can suggest an association, but cannot establish a causal explanation. If the slow path persists, the next diagnostic is rank-local host timing around `runtime.execute()` and the benchmark MPI reductions; device event time remains a separate timeline.

This repository currently has no accessible PBS scheduler or two-GPU allocation in the development workspace. The cluster execution, numerical GPU validation, physical mapping check, and pilot findings remain pending until the user runs the commands and provides the resulting artifacts.
