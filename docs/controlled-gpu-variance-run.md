# Controlled GPU variance run

From the published integration branch on the cluster frontend, choose one
stable experiment ID and run:

```sh
python3 benchmarks/controlled_pbs.py --experiment-id variance-20261014-a
```

The command builds with `cluster_build.sh` if needed, verifies the C++20/OpenCL
binaries and their source revision, and submits the pilot. It remains active
on the frontend, checks PBS every 30 seconds, and advances automatically after
each completed stage passes its evidence gate. Keep the frontend session alive
until it prints `Workflow complete`. If the session disconnects, rerun the
same command and experiment ID; queued and running PBS jobs are reused.
Set `--poll-interval-seconds N` to change the check interval (1–3600 seconds).
Use `--preview` separately if you want a read-only schedule first.

The stages are ordered because a scheduler dependency alone checks job exit,
while this workflow must also validate the pilot and diagnostic evidence
before submitting the full matrix:

| Stage | PBS work | Evidence required before the next stage |
| --- | --- | --- |
| Pilot | 24 rotated eight-iteration GPU samples | All accepted; pilot analyzer validates order, rank mapping, iteration zero, and reference checks. |
| Diagnostic smoke | Two 100-iteration k-neighbor seed-123 HWTOPOLB samples at `N=100,000`, one without a trace and one with rank-local traces, plus a two-rank halo reference | Both accepted on the same GPU mapping; complete trace from both ranks; trace overhead recorded outside headline timings. |
| Full matrix | Two-rank halo reference, four warm-ups, 60 unprofiled 100-iteration samples at `N=1,000,000`, then one separate traced capture on the same allocation | All samples accepted; halo, numerical reference, and trace checks pass; 30 adjacent pairs appear in the report. |

The default full-matrix diagnostic capture is invocation 8, k-neighbor /
HWTOPOLB / seed 123. Choose another full-matrix ordinal (5–64) with
`--diagnostic-ordinal ORDINAL` and repeat that option on every invocation.
`--diagnostic-ordinal 0` disables the full-matrix capture and leaves phase
attribution provisional. The smoke and capture are diagnostic runs and never
enter the 30 paired performance comparisons.

When the final report shows a persistent seed-123 reversal, request an
isolated A/B/A run for each affected workload. Add the option to the same
experiment ID; that single invocation waits for its PBS job and updates the
report:

```sh
python3 benchmarks/controlled_pbs.py --experiment-id variance-20261014-a \
  --aba-workload kneighbor
```

Use `--aba-workload leanmd` as well if LeanMD reverses. Requests are additive;
already accepted A/B/A runs are not repeated. The coordinator generates a new
report containing the three raw ERAD/HWTOPOLB/ERAD timings, separately from
the original paired estimate.

## What a rerun preserves

All output is under `Results/controlled-EXPERIMENT_ID/`, which is ignored by
Git. `workflow.json` records the pinned build hash, PBS job IDs, stage results,
and report hashes. Each stage has a unique run directory and sibling PBS output
and script-status files. After a terminal scheduler record with exit status
zero, the coordinator copies the PBS output, script status, scheduler record,
and build manifest into that stage directory without overwriting existing
files. It validates the raw run before submitting the next stage. Pilot and
full reports are written once and checked by hash on subsequent invocations.

The coordinator holds a lock while checking and advancing state. It records a
submission intent **before** calling `qsub`, so an interrupted or ambiguous
submission cannot silently create a second job. If it reports an uncertain
submission, inspect `qstat` for the recorded job name and resume with
`--adopt-job-id JOBID`. Adoption verifies the job name. Never start a new ID
merely because a queued job has not produced a run directory yet.

If `qstat -xf JOBID` becomes unavailable after completion, provide a saved
raw terminal `qstat -xf` record with `--scheduler-record MODE=PATH`, where MODE
is `pilot`, `smoke`, `full`, `aba-kneighbor`, or `aba-leanmd`. The record must
contain the matching job ID and exit status. The script does not infer
scheduler success from CSV presence or the PBS script status alone.

A rejected sample, failed PBS job, changed binary/source revision, changed
workflow option, or modified preserved artifact stops this experiment ID.
Keep its raw evidence and use a **new** ID after repairing the cause. The
sample timeout defaults to 1,800 seconds; `--sample-timeout-seconds N` may be
set at the first invocation and must remain the same thereafter.
Use `--once` only when a scheduler check without waiting is desired.

The build manifest records the compiler, exact flags, source revision and
digest, OpenCL library, and binary SHA-256 hashes. Each sample requires exactly
two ranks and one verifiable physical OpenCL GPU per rank, ordered GPU CSV
iterations, `validation=PASS`, a `gpu_opencl` summary, finite positive loop
time, and stable mapping. Unavailable clock, power, or contention observations
are recorded as unavailable. A local CPU simulation or this script's mock PBS
tests do not validate GPU kernels or establish a performance gain.

After the workflow finishes, preserve the entire
`Results/controlled-EXPERIMENT_ID/` tree, including all raw CSVs, logs,
inventories, manifests, rank traces, PBS records, and reports, and provide it
for review before closing the GPU-dependent issues.
