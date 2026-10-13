#!/usr/bin/env python3
"""Validate a completed two-GPU full matrix and render its paired report."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

from run_variability import ALGORITHMS, SEEDS, WORKLOADS, accepted_output, schedule

FIELDS = ('iteration', 'step_time_s', 'action', 'rebalanced', 'rebalance_gain', 'execution_mode')
THRESHOLD = 1 / 0.9


class InvalidRun(ValueError):
    """The controlled matrix cannot support a performance calculation."""


def json_file(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidRun(f'{path}: invalid or missing JSON: {error}') from error


def finite(value, label, positive=False):
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        raise InvalidRun(f'{label}: invalid number {value!r}') from None
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise InvalidRun(f'{label}: nonfinite, negative, or zero number {value!r}')
    return number


def artifact(root, value, label):
    if not isinstance(value, str):
        raise InvalidRun(f'{label}: missing artifact path')
    path = Path(value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise InvalidRun(f'{label}: missing or external artifact {path}')
    return path


def log_number(text, name, label):
    matches = re.findall(rf'\b{re.escape(name)}=([^\s]+)', text)
    if len(matches) != 1:
        raise InvalidRun(f'{label}: expected one {name}, found {len(matches)}')
    return finite(matches[0], f'{label} {name}')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate(run_dir):
    root = run_dir.resolve()
    run = json_file(root / 'run_manifest.json')
    if run.get('schema_version') != 1 or run.get('mode') != 'full' or \
            run.get('status') != 'completed' or run.get('rejected_samples') != 0:
        raise InvalidRun('Expected a completed, unrejected schema-1 full run')
    environment = run.get('environment')
    pbs = environment.get('pbs') if isinstance(environment, dict) else None
    if not isinstance(pbs, dict) or not pbs.get('job_id') or \
            not isinstance(pbs.get('slots'), list) or len(pbs['slots']) != 2 or \
            len(set(pbs['slots'])) != 2:
        raise InvalidRun('Missing two-node PBS allocation record')
    if not isinstance(run.get('build_manifest_sha256'), str) or \
            not re.fullmatch(r'[0-9a-f]{64}', run['build_manifest_sha256']) or \
            not isinstance(run.get('build'), dict):
        raise InvalidRun('Missing build provenance')
    params = run.get('input_parameters')
    expected_params = dict(n=1000000, full_iterations=100, rebalance_interval=10,
                           perturbation=0.10, kneighbor_k=5, leanmd_cutoff=4)
    if not isinstance(params, dict) or any(params.get(k) != v for k, v in expected_params.items()):
        raise InvalidRun('Wrong full-matrix input parameters')
    expected = schedule(root, Path('/unused/binaries'), 'full', 8)
    actual = run.get('invocations')
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise InvalidRun(f'Expected {len(expected)} invocations: four warm-ups and 60 timed samples')
    used = set()
    baseline = None
    rows = []
    for index, (item, planned) in enumerate(zip(actual, expected), 1):
        label = f'invocation {index}'
        if not isinstance(item, dict) or any(item.get(key) != planned[key] for key in
              ('phase', 'ordinal', 'repetition', 'workload', 'seed', 'algorithm',
               'pair_id', 'iterations', 'csv', 'log', 'inventory', 'manifest')):
            raise InvalidRun(f'{label}: schedule/order mismatch')
        argv = item.get('argv')
        if not isinstance(argv, list) or len(argv) < 2 or \
                not str(argv[0]).endswith('/' + item['workload'] + '.out'):
            raise InvalidRun(f'{label}: missing binary or argument record')
        required = {'--n': '1000000', '--iterations': '100', '--balance-interval': '10',
                    '--perturbation': '0.10', '--seed': str(item['seed']),
                    '--balance-algo': item['algorithm'],
                    '--k': '5' if item['workload'] == 'kneighbor' else '4',
                    '--output-csv': item['csv'], '--device-inventory-json': item['inventory']}
        for flag, value in required.items():
            if argv.count(flag) != 1 or argv[argv.index(flag) + 1:argv.index(flag) + 2] != [value]:
                raise InvalidRun(f'{label}: wrong {flag} argument')
        if any(argv.count(flag) != 1 for flag in ('--require-two-rank-gpu',
                                                   '--quiet-runtime-metrics')) or \
                '--diagnostics-prefix' in argv:
            raise InvalidRun(f'{label}: timed sample requires strict GPU mode without tracing')
        paths = {key: artifact(root, item.get(key), f'{label} {key}')
                 for key in ('csv', 'log', 'inventory', 'manifest')}
        if used.intersection(paths.values()):
            raise InvalidRun(f'{label}: reused artifact path')
        used.update(paths.values())
        sample = json_file(paths['manifest'])
        if sample.get('status') != 'accepted' or sample.get('rejection_reason') is not None or \
                sample.get('build_manifest_sha256') != run['build_manifest_sha256'] or \
                any(sample.get(key) != item[key] for key in
                    ('phase', 'ordinal', 'repetition', 'workload', 'seed', 'algorithm',
                     'pair_id', 'iterations', 'csv', 'log', 'inventory', 'manifest', 'argv')):
            raise InvalidRun(f'{label}: rejected or inconsistent sample manifest')
        try:
            times, mapping = accepted_output(item)
        except (ValueError, IndexError) as error:
            raise InvalidRun(f'{label}: {error}') from error
        if sample.get('mapping') != mapping:
            raise InvalidRun(f'{label}: physical GPU mapping differs from inventory')
        if baseline is None:
            baseline = mapping
        elif mapping != baseline:
            raise InvalidRun(f'{label}: physical GPU mapping changed')
        for key, value in times.items():
            recorded = finite((sample.get('times') or {}).get(key), f'{label} manifest {key}')
            if not math.isclose(recorded, value, rel_tol=1e-6, abs_tol=1e-9):
                raise InvalidRun(f'{label}: manifest {key} disagrees with log')
        try:
            with paths['csv'].open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != list(FIELDS):
                    raise InvalidRun(f'{label}: malformed iteration CSV header')
                records = list(reader)
        except (OSError, UnicodeError) as error:
            raise InvalidRun(f'{label}: cannot read iteration CSV: {error}') from error
        if len(records) != 100:
            raise InvalidRun(f'{label}: missing iteration rows')
        step_sum = 0.0
        for iteration, record in enumerate(records):
            rebalance = iteration != 0 and iteration % 10 == 0
            if None in record or any(v is None for v in record.values()) or \
                    record['iteration'] != str(iteration) or \
                    record['execution_mode'] != 'gpu_opencl' or \
                    record['action'] != ('rebalance' if rebalance else 'compute') or \
                    record['rebalanced'] != ('1' if rebalance else '0'):
                raise InvalidRun(f'{label}: malformed, CPU, or reordered iteration {iteration}')
            step_sum += finite(record['step_time_s'], f'{label} step {iteration}')
            finite(record['rebalance_gain'], f'{label} gain {iteration}')
        try:
            log = paths['log'].read_text(encoding='utf-8')
        except (OSError, UnicodeError) as error:
            raise InvalidRun(f'{label}: cannot read log: {error}') from error
        if re.findall(r'\bvalidation=([^\s]+)', log) != ['PASS'] or \
                re.findall(r'\bmode=([^\s]+)', log) != ['gpu_opencl']:
            raise InvalidRun(f'{label}: GPU mode or numerical reference validation failed')
        total = log_number(log, 'total_time_s', label)
        loop = finite(times['loop_time_s'], f'{label} loop', positive=True)
        validation = times['validation_time_s']
        # Each metric may be a different MPI maximum. Their sums need not fit
        # another maximum when the slow rank changes between phases/steps.
        if total + 1e-9 < max(loop, validation):
            raise InvalidRun(f'{label}: total wall time is below a measured phase')
        rows.append(dict(item, loop=loop, validation=validation, total=total,
                         step_sum=step_sum, mapping=mapping, paths=paths,
                         started_at=sample.get('started_at'),
                         completed_at=sample.get('completed_at')))
    pairs = []
    for offset in range(4, len(rows), 2):
        first, second = rows[offset:offset + 2]
        if first['phase'] != 'full' or second['phase'] != 'full' or \
                first['pair_id'] != second['pair_id'] or \
                {first['algorithm'], second['algorithm']} != set(ALGORITHMS):
            raise InvalidRun(f'invocation {offset + 1}: broken adjacent pair')
        pair = {item['algorithm']: item for item in (first, second)}
        pairs.append(dict(id=first['pair_id'], workload=first['workload'],
                          seed=first['seed'], repetition=first['repetition'],
                          order=f"{first['algorithm'].upper()}/{second['algorithm'].upper()}",
                          erad=pair['erad'], hwtopolb=pair['hwtopolb'],
                          ratio=pair['erad']['loop'] / pair['hwtopolb']['loop']))
    if len(pairs) != 30:
        raise InvalidRun('Expected 30 adjacent pairs')
    return run, rows, pairs


def quartiles(values):
    ordered = sorted(values)
    # Tukey hinges: medians of the lower and upper halves of five repeats.
    half = len(ordered) // 2
    return statistics.median(ordered[:half]), statistics.median(ordered[-half:])


def observation(value):
    if not isinstance(value, dict):
        return 'unavailable: not recorded'
    if value.get('value'):
        return str(value['value']).replace('\n', '; ')
    return f"unavailable: {value.get('unavailable_reason') or 'not recorded'}"


def halo_reference(run, root):
    evidence = run.get('halo_reference')
    if evidence is None:
        return False, 'not recorded in full-run manifest'
    if not isinstance(evidence, dict) or evidence.get('status') != 'passed' or \
            evidence.get('exit_code') != 0 or evidence.get('rejection_reason') is not None:
        return False, 'halo reference was not accepted'
    command = evidence.get('command')
    if not isinstance(command, list) or len(command) != 2 or \
            not str(command[0]).endswith('prun') or \
            not str(command[1]).endswith('test_async_halo_opencl.out'):
        return False, 'halo command or two-rank launcher is missing'
    digest = evidence.get('binary_sha256')
    if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
        return False, 'halo binary digest is missing'
    binaries = run.get('build', {}).get('binaries')
    halo_binary = binaries.get('test_async_halo_opencl') if isinstance(binaries, dict) else None
    if not isinstance(halo_binary, dict):
        return False, 'halo binary is missing from the build manifest'
    expected_digest = halo_binary.get('sha256')
    if digest != expected_digest:
        return False, 'halo binary digest differs from the build manifest'
    if command[1] != halo_binary.get('path'):
        return False, 'halo binary path differs from the build manifest'
    launcher = run['environment']['pbs'].get('launcher')
    if command[0] != launcher:
        return False, 'halo launcher differs from the PBS run launcher'
    try:
        log = artifact(root, evidence.get('log'), 'halo log').read_text(encoding='utf-8')
    except (InvalidRun, OSError, UnicodeError) as error:
        return False, str(error)
    if log.splitlines().count('PASS: asynchronous OpenCL halo matches two-step CPU reference') != 1 or \
            re.search(r'\bSKIP\b|\bFAIL\b', log):
        return False, 'halo log lacks an unambiguous reference PASS'
    return True, 'verified from the full-run halo log and exit status'


def diagnostic_evidence(manifest_path, run, rows):
    manifest_path = manifest_path.resolve()
    evidence = json_file(manifest_path)
    if not isinstance(evidence, dict) or evidence.get('schema_version') != 1 or \
            evidence.get('status') != 'captured' or \
            evidence.get('exit_code') != 0:
        raise InvalidRun('Diagnostic invocation did not complete successfully')
    pbs = evidence.get('pbs')
    if pbs != run['environment']['pbs']:
        raise InvalidRun('Diagnostic PBS allocation differs from the full matrix')
    if evidence.get('build_manifest_sha256') != run['build_manifest_sha256']:
        raise InvalidRun('Diagnostic build manifest differs from the full matrix')
    workload, seed, algorithm = (evidence.get(key) for key in ('workload', 'seed', 'algorithm'))
    if workload not in WORKLOADS or seed not in SEEDS or algorithm not in ALGORITHMS:
        raise InvalidRun('Diagnostic condition is not in the controlled matrix')
    binaries = run['build'].get('binaries')
    binary = binaries.get(workload) if isinstance(binaries, dict) else None
    if not isinstance(binary, dict) or evidence.get('binary_sha256') != binary.get('sha256'):
        raise InvalidRun('Diagnostic executable digest differs from full matrix build')
    matched = Path(str(evidence.get('matched_sample_manifest'))).resolve()
    candidates = [row for row in rows if row['phase'] == 'full' and
                  row['paths']['manifest'] == matched]
    if len(candidates) != 1:
        raise InvalidRun('Diagnostic matched sample is absent from unprofiled matrix')
    sample = candidates[0]
    if (sample['workload'], sample['seed'], sample['algorithm']) != (workload, seed, algorithm):
        raise InvalidRun('Diagnostic and matched sample conditions differ')
    root = manifest_path.parent
    artifacts = evidence.get('artifacts')
    digests = evidence.get('artifact_sha256')
    names = ('csv', 'log', 'inventory', 'rank0', 'rank1')
    if not isinstance(artifacts, dict) or not isinstance(digests, dict):
        raise InvalidRun('Diagnostic artifacts and hashes are missing')
    paths = {key: artifact(root, artifacts.get(key), f'diagnostic {key}') for key in names}
    if len(set(paths.values())) != len(names) or \
            any(path in {p for row in rows for p in row['paths'].values()}
                for path in paths.values()):
        raise InvalidRun('Diagnostic artifacts collide with each other or unprofiled samples')
    for name, path in paths.items():
        if digests.get(name) != sha256(path):
            raise InvalidRun(f'Diagnostic {name} SHA-256 mismatch')
    prefix = Path(str(evidence.get('trace_prefix'))).resolve()
    if paths['rank0'] != Path(f'{prefix}.rank0.csv') or \
            paths['rank1'] != Path(f'{prefix}.rank1.csv'):
        raise InvalidRun('Diagnostic trace prefix does not identify both rank files')
    command = evidence.get('command')
    expected = [pbs['launcher'], *sample['argv']]
    for flag, path in (('--output-csv', paths['csv']),
                       ('--device-inventory-json', paths['inventory'])):
        expected[expected.index(flag) + 1] = str(path)
    expected += ['--diagnostics-prefix', str(prefix)]
    if command != expected:
        raise InvalidRun('Diagnostic command differs from the matched sample or trace paths')
    try:
        times, mapping = accepted_output(dict(csv=str(paths['csv']),
                                              log=str(paths['log']),
                                              inventory=str(paths['inventory']),
                                              iterations=100))
    except (ValueError, IndexError) as error:
        raise InvalidRun(f'Diagnostic sample is not valid: {error}') from error
    if mapping != sample['mapping']:
        raise InvalidRun('Diagnostic physical GPU mapping differs from matched sample')
    log = paths['log'].read_text(encoding='utf-8')
    if re.findall(r'\bmode=([^\s]+)', log) != ['gpu_opencl'] or \
            re.findall(r'\bvalidation=([^\s]+)', log) != ['PASS']:
        raise InvalidRun('Diagnostic GPU mode or numerical validation failed')
    total = log_number(log, 'total_time_s', 'diagnostic')
    if total + 1e-9 < max(times.values()):
        raise InvalidRun('Diagnostic full wall is below a measured phase')
    validator = Path(__file__).with_name('validate_trace.py')
    if not validator.is_file():
        raise InvalidRun('Rank trace validator is unavailable')
    with tempfile.TemporaryDirectory() as temporary:
        summary_path = Path(temporary) / 'summary.json'
        result = subprocess.run([sys.executable, str(validator), str(prefix),
                                 '--ranks', '2', '--iterations', '100',
                                 '--elements', '1000000', '--balance-interval', '10',
                                 '--summary-json', str(summary_path)],
                                capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise InvalidRun(f'Invalid rank traces: {result.stderr.strip()}')
        summary = json_file(summary_path)
    if summary.get('schema_version') != 1 or summary.get('iterations') != 100 or \
            summary.get('elements') != 1000000 or summary.get('seed') != str(seed) or \
            summary.get('algorithm') != algorithm:
        raise InvalidRun('Rank trace summary differs from diagnostic condition')
    rank_data = summary.get('ranks')
    if not isinstance(rank_data, list) or len(rank_data) != 2 or \
            [item.get('rank') for item in rank_data] != [0, 1]:
        raise InvalidRun('Rank trace summary does not contain both ranks')
    phase_names = ('runtime_execute', 'rebalance_plan', 'rebalance_apply',
                   'mpi_step_max', 'mpi_device_times')
    reconciled = []
    for rank, item in enumerate(rank_data):
        if Path(item.get('trace', '')).resolve() != paths[f'rank{rank}']:
            raise InvalidRun(f'Rank {rank} trace path differs from diagnostic manifest')
        totals = item.get('host_phase_totals_s')
        if not isinstance(totals, dict) or any(name not in totals for name in
              ('loop_total', *phase_names, 'gather', 'reference', 'mpi_validation')) or \
                'cpu_simulation' in totals:
            raise InvalidRun(f'Rank {rank} host phases are incomplete or CPU-only')
        values = {key: finite(value, f'rank {rank} {key}') for key, value in totals.items()}
        loop = values['loop_total']
        host = sum(values[name] for name in phase_names)
        device = finite(item.get('device_event_total_s'), f'rank {rank} device events')
        with paths[f'rank{rank}'].open(newline='', encoding='utf-8') as stream:
            raw = list(csv.DictReader(stream))
        inventory = [row for row in raw if row['record_type'] == 'inventory']
        if len(inventory) != 1 or inventory[0]['device_global'] != str(mapping[rank]['global_index']) or \
                inventory[0]['device_identity'] != \
                f"{mapping[rank]['identity_kind']}:{mapping[rank]['identity']}":
            raise InvalidRun(f'Rank {rank} trace GPU identity differs from inventory')
        reconciled.append(dict(rank=rank, loop=loop, phases=values, host=host,
                               unaccounted=loop-host, device=device,
                               trace=paths[f'rank{rank}'], raw=raw,
                               partition_rows=item.get('partition_rows')))
    if not math.isclose(max(item['loop'] for item in reconciled),
                        times['loop_time_s'], rel_tol=1e-4, abs_tol=1e-5):
        raise InvalidRun('Diagnostic global loop differs from rank-local loop totals')
    return dict(manifest=manifest_path, evidence=evidence, matched=sample,
                times=times, mapping=mapping, ranks=reconciled, paths=paths)


def render_diagnostic(diag):
    lines = ['', '## Rank-local diagnostic accounting', '',
             f"PBS job `{diag['evidence']['pbs']['job_id']}`; traced condition "
             f"`{diag['evidence']['workload']}` / `{diag['evidence']['algorithm']}` / "
             f"seed `{diag['evidence']['seed']}`. Diagnostic manifest: "
             f"`{diag['manifest']}`. The matched unprofiled sample is invocation "
             f"{diag['matched']['ordinal']} (`{diag['matched']['paths']['manifest']}`).", '',
             f"Traced loop `{diag['times']['loop_time_s']:.6f}` s; matched unprofiled "
             f"loop `{diag['matched']['loop']:.6f}` s; one-condition trace overhead "
             f"ratio `{diag['times']['loop_time_s']/diag['matched']['loop']:.4f}`. "
             'These diagnostic times do not enter the 30 headline pairs.', '',
             '| Rank | Loop total (s) | Execute | Rebalance plan/apply | MPI step/device reductions | '
             'Measured host phases | Unaccounted host (s) | OpenCL device events (s) | '
             'Post-loop gather/reference/MPI validation (s) | Trace |',
             '| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | --- |']
    for item in diag['ranks']:
        p = item['phases']
        lines.append(f"| {item['rank']} | {item['loop']:.6f} | "
                     f"{p['runtime_execute']:.6f} | "
                     f"{p['rebalance_plan']:.6f} / {p['rebalance_apply']:.6f} | "
                     f"{p['mpi_step_max']:.6f} / {p['mpi_device_times']:.6f} | "
                     f"{item['host']:.6f} | {item['unaccounted']:.6f} | "
                     f"{item['device']:.6f} | "
                     f"{p['gather']:.6f} / {p['reference']:.6f} / "
                     f"{p['mpi_validation']:.6f} | `{item['trace']}` |")
    lines += ['', 'Unaccounted host time is `loop_total` minus nonoverlapping host phases '
              'on the **same rank**. A positive remainder is a missing measurement; a '
              'negative remainder indicates overlap or inconsistent boundaries. OpenCL '
              'event time is a separate, potentially overlapping device timeline and is '
              'not added to host phases. Post-loop validation is outside `loop_total`. '
              'Boundary shifts are element-count estimates; transfer bytes are unknown '
              'unless the trace contains a measured byte count.', '']
    for item in diag['ranks']:
        first = [row for row in item['raw'] if row['iteration'] == '0' and
                 row['record_type'] in ('phase', 'partition', 'device_event')]
        lines += [f"### Rank {item['rank']}, iteration 0 (before rebalance)", '',
                  '| Trace seq | Type/phase | Partition | Device/owner | Offset/count | Host duration (s) | '
                  'Device event (s) |',
                  '| ---: | --- | ---: | --- | --- | ---: | ---: |']
        for row in first:
            lines.append(f"| {row['seq']} | {row['record_type']}/{row['phase']} | "
                         f"{row['part'] or '-'} | {row['device_global'] or '-'} / "
                         f"{row['owner_rank'] or '-'} | {row['offset'] or '-'} / "
                         f"{row['count'] or '-'} | {row['duration_s'] or '-'} | "
                         f"{row['device_event_s'] or '-'} |")
        lines.append('')
    lines += ['The trace establishes timing and partition records for this diagnostic '
              'invocation only. A seed-dependent causal explanation requires matching '
              'fast/control traces and, when the seed-123 reversal persists, the separate '
              'ERAD/HWTOPOLB/ERAD sequence. No causal attribution is made here.', '']
    return '\n'.join(lines)


def render(run, rows, pairs, halo_ok, halo_detail):
    groups = []
    near_or_unstable = False
    reversal = False
    all_gain = True
    for workload in WORKLOADS:
        for seed in SEEDS:
            group = [p for p in pairs if p['workload'] == workload and p['seed'] == seed]
            ratios = [p['ratio'] for p in group]
            median = statistics.median(ratios)
            q1, q3 = quartiles(ratios)
            spread = q3 - q1
            early = statistics.median(p['ratio'] for p in group[:2])
            late = statistics.median(p['ratio'] for p in group[-2:])
            drift = abs(late - early) / early > 0.10
            stable = spread / median <= 0.10 and q1 >= THRESHOLD and not drift
            near = abs(median - THRESHOLD) <= spread
            near_or_unstable |= near or not stable
            reversal |= median <= 1
            all_gain &= median >= THRESHOLD and stable
            erad_loops = [p['erad']['loop'] for p in group]
            hwtopolb_loops = [p['hwtopolb']['loop'] for p in group]
            groups.append((workload, seed, median, min(ratios), max(ratios), q1, q3,
                           early, late, min(erad_loops), max(erad_loops),
                           min(hwtopolb_loops), max(hwtopolb_loops), stable))
    verdict = ('Paired performance supports a useful gain for these workloads, seeds, '
               'and allocation only; variance diagnosis remains provisional pending '
               'rank-local traces.'
               if all_gain and halo_ok else
               'No useful gain across every seed; no general speedup claim.'
               if reversal and halo_ok else
               'Performance conclusion inconclusive pending correctness or stable per-seed evidence.'
               if not halo_ok or near_or_unstable else
               'No useful gain across every seed; no general speedup claim.')
    lines = ['# Controlled two-GPU variance report', '',
             '**Report status:** provisional until rank-local diagnostic traces are '
             'validated and phase attribution is reconciled.', '',
             '## Correctness and provenance', '',
             f"PBS job: `{run['environment']['pbs']['job_id']}`; nodes: "
             f"`{', '.join(run['environment']['pbs']['slots'])}`. ",
             f"Build manifest SHA-256: `{run['build_manifest_sha256']}`. ",
             f"Source revision: `{run['build'].get('source_revision', 'unavailable')}`. ",
             'All 64 invocations are accepted `gpu_opencl` runs with two distinct physically identified GPUs; '
             'each CSV has 100 ordered iterations and numerical `validation=PASS`. '
             'k-neighbor uses finite outputs and absolute error <= 1e-5; LeanMD uses finite '
             'outputs and error <= 1e-4 + 1e-4 * abs(reference).', '',
             f"Two-rank asynchronous halo reference: {'PASS' if halo_ok else 'pending/unavailable'} "
             f"({halo_detail}).",
             'The benchmark validation and halo reference are separate correctness gates.', '',
             '| Rank | Node | GPU | Physical identity | OpenCL platform/version | Device/driver |',
             '| ---: | --- | --- | --- | --- | --- |']
    for device in rows[0]['mapping']:
        lines.append(f"| {device['rank']} | {device['host']} | {device['name']} | "
                     f"{device['identity_kind']}:{device['identity']} | "
                     f"{device['platform_name']} / {device['platform_version']} | "
                     f"{device['device_version']} / {device['driver_version']} |")
    lines += ['', '| Environment | Observation |', '| --- | --- |']
    for key in ('module_list', 'mpi_version', 'cpu_model', 'gpu_clock', 'gpu_power',
                'gpu_contention', 'affinity'):
        lines.append(f"| {key} | {observation(run['environment'].get(key))} |")
    lines += ['', '## Performance', '',
              f'Validated {len(pairs)} adjacent pairs from five rotated repetitions. '
              'The primary metric is the complete 100-iteration loop through its MPI reductions; '
              'final gather/reference validation is excluded. All headline pairs are unprofiled.', '',
              '| Workload | Seed | Median ERAD/HWTOPOLB | Min | Max | Q1 | Q3 | '
              'Early pair median | Late pair median | ERAD loop range (s) | '
              'HWTOPOLB loop range (s) | Stable |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | '
              '--- | --- | --- |']
    for workload, seed, median, minimum, maximum, q1, q3, early, late, \
            erad_min, erad_max, hwt_min, hwt_max, stable in groups:
        lines.append(f'| {workload} | {seed} | {median:.4f} | {minimum:.4f} | '
                     f'{maximum:.4f} | {q1:.4f} | {q3:.4f} | {early:.4f} | '
                     f'{late:.4f} | {erad_min:.6f}–{erad_max:.6f} | '
                     f'{hwt_min:.6f}–{hwt_max:.6f} | {"yes" if stable else "no"} |')
    lines += ['', 'Q1 and Q3 are Tukey hinges (the medians of the lower and upper two '
              'values among five sorted ratios). Stability requires (Q3-Q1)/median <= 0.10 '
              'and Q1 >= 1/0.9, with early-versus-late paired-ratio median drift '
              '<= 10% (first two versus last two pairs in run order). '
              'A useful gain requires each seed median >= 1/0.9 '
              'and stable repeats; pooling seeds is not allowed.', '',
              '| Pair | Run order | ERAD ordinal/loop (s) | HWTOPOLB ordinal/loop (s) | '
              'Paired ratio | ERAD/HWTOPOLB full wall (s) | Validation (s) | '
              'Loop minus sum of step maxima (s) | UTC start ERAD/HWTOPOLB | '
              'UTC end ERAD/HWTOPOLB | Raw CSV/log |',
              '| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |']
    for p in pairs:
        a, b = p['erad'], p['hwtopolb']
        lines.append(f"| {p['id']} | {p['order']} | {a['ordinal']} / {a['loop']:.6f} | "
                     f"{b['ordinal']} / {b['loop']:.6f} | {p['ratio']:.4f} | "
                     f"{a['total']:.6f} / {b['total']:.6f} | "
                     f"{a['validation']:.6f} / {b['validation']:.6f} | "
                     f"{a['loop'] - a['step_sum']:.6f} / "
                     f"{b['loop'] - b['step_sum']:.6f} | "
                     f"{a['started_at'] or 'unavailable'} / "
                     f"{b['started_at'] or 'unavailable'} | "
                     f"{a['completed_at'] or 'unavailable'} / "
                     f"{b['completed_at'] or 'unavailable'} | "
                     f"`{a['csv']}`; `{a['log']}`; `{b['csv']}`; `{b['log']}` |")
    lines += ['', f'**Verdict:** {verdict}', '',
              '## Timing interpretation and cause', '',
              'The legacy full-wall metric includes final validation and is secondary; '
              'it is not comparable with historical full-wall numbers as a loop-time baseline. '
              'Per-step CSV times omit benchmark MPI reductions. The table shows '
              '`loop minus sum of step maxima` for every sample as a descriptive '
              'accounting gap. Loop, validation, full-wall, and individual steps may '
              'each be maxima from different ranks, so this gap is not a rank-local '
              'residual; it can even be negative. It cannot be assigned to MPI, kernel, '
              'migration, or another host phase. Device OpenCL '
              'events may overlap host phases and must be shown on a separate timeline. '
              'Rank-local diagnostic traces and a same-condition unprofiled comparison are '
              'required to reconcile execute, rebalance, benchmark MPI, validation, and '
              'unaccounted time. No causal attribution is made from this matrix alone.', '',
              'Historical archive: `benchmarks/historical_archive_manifest.json` and '
              '`benchmarks/historical_gpu_evidence.md`. The earlier seed-123 slow path '
              'starts at iteration 0, before rebalance, but the physical GPU identity and '
              'host phase cause were not measured then.', '']
    if reversal:
        lines.append('A seed-level reversal persists. Run an isolated same-allocation '
                     'ERAD/HWTOPOLB/ERAD sequence and inspect diagnostic traces before '
                     'attributing the cause. These diagnostic runs stay outside the paired estimate.')
    elif near_or_unstable:
        lines.append('Repeat the controlled rotated matrix because at least one seed is '
                     'near the gain threshold or unstable. Keep each new allocation distinct.')
    else:
        lines.append('A diagnostic A/B/A sequence is needed only if the seed-123 anomaly persists.')
    lines += ['', 'This HWTOPOLB variant uses throughput and seeded perturbation; it does '
              'not consume the measured topology JSON. No topology-aware performance '
              'claim follows from these data.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-manifest', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'output already exists: {args.output}')
    try:
        run, rows, pairs = validate(args.run_dir)
        halo_ok, halo_detail = halo_reference(run, args.run_dir.resolve())
        report = render(run, rows, pairs, halo_ok, halo_detail)
        if args.diagnostic_manifest:
            diag = diagnostic_evidence(args.diagnostic_manifest, run, rows)
            report += render_diagnostic(diag)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(report)
    except (InvalidRun, OSError, ValueError, TypeError, KeyError, IndexError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    sys.exit(main())
