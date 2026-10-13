# Controlled GPU variance run

Build on the cluster frontend with `bash cluster_build.sh`. This writes
`build/cluster/build_manifest.json` with source revision, source digest, compiler,
flags, exact benchmark commands, OpenCL library, and SHA-256 values for both
benchmarks and the two-rank asynchronous OpenCL halo test.
The source digest covers C++ headers and sources in `dcl/` and `benchmarks/`,
plus OpenCL kernels. The binary hashes are the execution gate.

Preview without PBS, MPI, binaries, or filesystem writes:

```sh
python3 benchmarks/run_variability.py --dry-run --mode pilot --run-dir Results/pilot-2026-10-13
python3 benchmarks/run_variability.py --dry-run --mode full --run-dir Results/full-2026-10-13
```

The pilot has two 8-iteration repetitions of every workload, seed, and algorithm
condition, with rotated seeds and alternating algorithm order. It is diagnostic
and stays outside full-matrix statistics. The full plan has four warm-ups and
60 timed samples (five repetitions per condition). Each ERAD/HWTOPOLB pair is
adjacent. Both modes use `N=1,000,000`, interval 10, perturbation 0.10,
k-neighbor `k=5`, and LeanMD cutoff 4. A different input needs a separate,
clearly labeled protocol and run directory.

Submit from the cluster frontend with a new, unique output path each time:

```sh
export HARMONY_VARIABILITY_MODE=pilot
export HARMONY_VARIABILITY_RUN_DIR="$PWD/Results/pilot-$(date -u +%Y%m%dT%H%M%SZ)"
export HARMONY_PBS_OUTPUT_PATH="${HARMONY_VARIABILITY_RUN_DIR}.pbs.out"
test ! -e "$HARMONY_VARIABILITY_RUN_DIR"
test ! -e "$HARMONY_PBS_OUTPUT_PATH"
qsub -o "$HARMONY_PBS_OUTPUT_PATH" -j oe -V job_harmony_variability.pbs
```

Inspect the pilot run manifest and all 24 sample manifests, CSVs, logs, and
device inventories. Every sample must be accepted with finite timing,
`validation=PASS`, two physical GPUs, and complete GPU iterations. Run the
two-rank diagnostic trace separately and compare it with an uninstrumented
condition before interpreting phase timings. Do not include pilot or traced
runs in the full-matrix statistics.

After the pilot, diagnostic, and numerical checks pass, repeat with
`HARMONY_VARIABILITY_MODE=full` and a fresh `Results/full-...` directory. The
full executor first runs the prebuilt two-rank asynchronous OpenCL halo test
on the same PBS allocation. It requires exit code 0 and the CPU-reference
`PASS` marker; `SKIP`, mismatch, timeout, or missing binary rejects the run
before any warm-up or timed sample. Preserve `halo_reference.log` and the
`halo_reference` result in `run_manifest.json`.

PBS stdout/stderr is written to the unique `HARMONY_PBS_OUTPUT_PATH` sibling
path and recorded in the run manifest. The PBS script writes its own exit code
and job ID to `${HARMONY_VARIABILITY_RUN_DIR}.pbs.status`. Once the scheduler
has finished writing output, copy both sibling artifacts into the run directory
without replacing any existing file:

```sh
cp -n "$HARMONY_PBS_OUTPUT_PATH" "$HARMONY_VARIABILITY_RUN_DIR/pbs.out"
cp -n "${HARMONY_VARIABILITY_RUN_DIR}.pbs.status" "$HARMONY_VARIABILITY_RUN_DIR/pbs.status"
qstat -xf JOBID > "$HARMONY_VARIABILITY_RUN_DIR/pbs.scheduler-status.txt"
```

If `qstat` no longer retains the job, record the exit status from the site
accounting command in `pbs.scheduler-status.txt`. The script status is not a
substitute for the scheduler's final status. Do not infer success from CSV
presence. Retain all raw sample logs, CSVs, inventories, per-sample manifests,
the run manifest, PBS artifacts, and build manifest together in the ignored
dated results archive. Keep the historical archive untouched.
The executor does not resume or overwrite a run directory. It verifies both
binary digests before every invocation. Missing PBS job/nodefile, wrong two-node
slot shape, missing `prun`, or changed binaries refuse execution.

The executor depends on benchmark support for `--require-two-rank-gpu`,
`--device-inventory-json`, `loop_time_s`, `validation_time_s`, and
`validation=PASS` from issue #6. Build the integrated branch before submission.

Every sample has a separate CSV, log, device inventory JSON, and status manifest.
The run manifest records the PBS allocation, build provenance hash, exact
commands, input parameters, module list, MPI version, CPU model, and available
GPU clock/power/contention observations. An unavailable optional observation
is represented by `value: null` with an `unavailable_reason`. A sample is
accepted only with complete GPU CSV iterations, `validation=PASS`, finite
`loop_time_s` and `validation_time_s`, and two stable, distinct physical GPU
identities. A rejected sample and its raw artifacts remain in place. A paired
sample is also rejected when its partner fails or the rank/device mapping
changes. Never use pilot or diagnostic samples as full timed data.

This workspace has no PBS launcher or real GPU. The two-rank cluster execution
and all hardware measurements are pending manual validation.
