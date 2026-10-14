# Pre-rebalance two-GPU pilot: manual cluster validation

The pilot is a diagnostic experiment for issue #8. It uses two rotated passes of eight iterations, so no scheduled rebalance at iteration 10 occurs. Its 24 invocations form 12 adjacent ERAD/HWTOPOLB pairs. Pilot samples must stay out of the full 100-iteration performance comparison.

On the cluster frontend, run the resumable coordinator
described in `docs/controlled-gpu-variance-run.md`:

```sh
python3 benchmarks/controlled_pbs.py --experiment-id variance-20261014-a
```

The command builds if needed, submits the pilot, waits for PBS, and then
preserves the scheduler record, PBS output and exit status, raw samples, and
pilot report before advancing to the diagnostic smoke. The pilot run is at
`Results/controlled-EXPERIMENT_ID/pilot/`. Every sample must be `accepted`,
the run must be `completed`, both ranks must have verified physical GPU
identities and stable mapping, and the numerical reference checks must pass.
The analyzer checks that the run baseline mapping matches sample inventories,
including CPU affinity or an explicit unavailability reason. A rejected run
stops the workflow without resubmitting or replacing artifacts.

The coordinator writes `pilot/pilot-analysis.md` automatically after a
completed pilot. `analyze_pilot.py` remains available for independent
revalidation against the preserved raw files.

The analyzer independently checks the 24 accepted invocations and their successful launcher exits, order rotation, pre-rebalance iteration sequence, `gpu_opencl` mode, finite host timings, successful validation, physical-device mapping, and manifest/log consistency. It refuses a full or incomplete run and never overwrites an existing report. The report lists iteration-0 latency, earliest slow ordinary iteration, per-invocation order, and group summaries by seed, algorithm, pass, and run position. A 10 ms threshold is descriptive of the historical slow path, not proof of cause. Two short passes can suggest an association, but cannot establish a causal explanation. If the slow path persists, the next diagnostic is rank-local host timing around `runtime.execute()` and the benchmark MPI reductions; device event time remains a separate timeline.

This repository currently has no accessible PBS scheduler or two-GPU allocation in the development workspace. The cluster execution, numerical GPU validation, physical mapping check, and pilot findings remain pending until the user runs the commands and provides the resulting artifacts.
