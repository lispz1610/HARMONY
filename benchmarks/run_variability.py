#!/usr/bin/env python3
"""Preview or execute the controlled two-GPU HARMONY comparison."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

WORKLOADS = ('kneighbor', 'leanmd')
ALGORITHMS = ('erad', 'hwtopolb')
SEEDS = (42, 123, 456)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary, path)


def schedule(run_dir, bin_dir, mode, pilot_iterations):
    rows = []
    count = 0

    def add(phase, repetition, workload, seed, algorithm, pair_id=None):
        nonlocal count
        count += 1
        stem = f'{count:03d}_{workload}_{algorithm}_{seed}'
        prefix = run_dir / phase / stem
        iterations = 100 if phase in ('full', 'warmup') else pilot_iterations
        argv = [str(bin_dir / f'{workload}.out'), '--n', '1000000',
                '--iterations', str(iterations), '--balance-interval', '10',
                '--perturbation', '0.10', '--seed', str(seed),
                '--balance-algo', algorithm,
                '--k', '5' if workload == 'kneighbor' else '4',
                '--output-csv', str(prefix) + '.csv',
                '--device-inventory-json', str(prefix) + '.inventory.json',
                '--require-two-rank-gpu', '--quiet-runtime-metrics']
        rows.append(dict(phase=phase, ordinal=count, repetition=repetition,
                         workload=workload, seed=seed, algorithm=algorithm,
                         pair_id=pair_id, iterations=iterations, argv=argv,
                         csv=str(prefix) + '.csv', log=str(prefix) + '.log',
                         inventory=str(prefix) + '.inventory.json',
                         manifest=str(prefix) + '.manifest.json'))

    if mode == 'full':
        for workload in WORKLOADS:
            for algorithm in ALGORITHMS:
                add('warmup', 0, workload, SEEDS[0], algorithm)
        repetitions = range(5)
    else:
        repetitions = range(2)
    for repetition in repetitions:
        seeds = SEEDS[repetition % len(SEEDS):] + SEEDS[:repetition % len(SEEDS)]
        algorithms = ALGORITHMS if repetition % 2 == 0 else ALGORITHMS[::-1]
        for workload in WORKLOADS:
            for seed in seeds:
                pair_id = f'r{repetition + 1}_{workload}_{seed}'
                for algorithm in algorithms:
                    add(mode, repetition + 1, workload, seed, algorithm, pair_id)
    paths = [row[key] for row in rows for key in ('csv', 'log', 'inventory', 'manifest')]
    if len(paths) != len(set(paths)):
        raise ValueError('Schedule contains duplicate artifact paths')
    return rows


def build_verified(bin_dir, manifest_path):
    with open(manifest_path, encoding='utf-8') as stream:
        manifest = json.load(stream)
    for workload in WORKLOADS:
        binary = bin_dir / f'{workload}.out'
        recorded = manifest['binaries'][workload]
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f'Missing executable: {binary}')
        if digest(binary) != recorded['sha256']:
            raise ValueError(f'Binary SHA-256 mismatch: {binary}')
    return manifest


def pbs_environment():
    job_id = os.environ.get('PBS_JOBID')
    nodefile = os.environ.get('PBS_NODEFILE')
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise ValueError('PBS_JOBID and a readable PBS_NODEFILE are required')
    slots = Path(nodefile).read_text(encoding='utf-8').splitlines()
    if len(slots) != 2 or len(set(slots)) != 2:
        raise ValueError('Expected two PBS slots on distinct one-GPU nodes')
    launcher = shutil.which('prun')
    if not launcher:
        raise ValueError('prun launcher is required')
    return dict(job_id=job_id, nodefile=nodefile, slots=slots, launcher=launcher,
                scheduler_output=os.environ.get('HARMONY_PBS_OUTPUT_PATH'))


def observation(command):
    try:
        value = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT,
                                        timeout=10).strip()
        return dict(value=value, unavailable_reason=None)
    except (OSError, subprocess.SubprocessError) as error:
        return dict(value=None, unavailable_reason=str(error))


def environment_manifest(pbs):
    return dict(pbs=pbs, started_at=datetime.now(timezone.utc).isoformat(),
                module_list=observation(['bash', '-lc', 'module list 2>&1']),
                mpi_version=observation(['mpirun', '--version']),
                cpu_model=observation(['bash', '-lc',
                                       "awk -F: '/model name/ {print $2; exit}' /proc/cpuinfo"]),
                gpu_clock=observation(['prun', 'nvidia-smi', '--query-gpu=clocks.current.graphics',
                                       '--format=csv,noheader']),
                gpu_power=observation(['prun', 'nvidia-smi', '--query-gpu=power.draw',
                                       '--format=csv,noheader']),
                gpu_contention=observation(['prun', 'nvidia-smi', '--query-compute-apps=pid',
                                            '--format=csv,noheader']))


def accepted_output(row):
    try:
        with open(row['csv'], newline='', encoding='utf-8') as stream:
            records = list(csv.DictReader(stream))
        if len(records) != row['iterations']:
            raise ValueError(f'Expected {row["iterations"]} iterations, found {len(records)}')
        if [int(record['iteration']) for record in records] != list(range(row['iterations'])):
            raise ValueError('Missing or duplicate iteration')
        if any(record['execution_mode'] != 'gpu_opencl' for record in records):
            raise ValueError('CPU fallback or mixed execution mode')
        if any(not math.isfinite(float(record['step_time_s'])) or
               float(record['step_time_s']) < 0 for record in records):
            raise ValueError('Nonfinite or negative step timing')
        log = Path(row['log']).read_text(encoding='utf-8')
        if 'validation=PASS' not in log:
            raise ValueError('Numerical validation did not pass')
        times = {}
        for name in ('loop_time_s', 'validation_time_s'):
            match = re.search(rf'\b{name}=([^\s]+)', log)
            if not match:
                raise ValueError(f'Missing {name}')
            times[name] = float(match.group(1))
            if not math.isfinite(times[name]) or times[name] < 0:
                raise ValueError(f'Invalid {name}')
        inventory = json.loads(Path(row['inventory']).read_text(encoding='utf-8'))
        ranks = inventory['ranks']
        if inventory.get('schema_version') != 1 or len(ranks) != 2 or \
                {item['rank'] for item in ranks} != {0, 1}:
            raise ValueError('Expected one inventory record for each of two ranks')
        required = ('rank', 'host', 'gpu_count', 'local_index', 'global_index', 'name',
                    'platform_name', 'platform_version', 'device_version',
                    'driver_version', 'identity_kind', 'identity')
        if any(any(item.get(key) is None for key in required) for item in ranks):
            raise ValueError('Incomplete device inventory')
        if any(item['gpu_count'] != 1 for item in ranks):
            raise ValueError('Expected exactly one GPU per rank')
        if any(item['identity_kind'] not in ('uuid', 'pci') or not item['identity']
               for item in ranks):
            raise ValueError('Unverifiable physical GPU identity')
        physical = {(item['host'], item['identity_kind'], item['identity']) for item in ranks}
        if len(physical) != 2:
            raise ValueError('The two ranks resolved to the same physical GPU')
        mapping = [{key: item[key] for key in required} for item in sorted(ranks, key=lambda x: x['rank'])]
        return times, mapping
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f'Incomplete sample artifacts: {error}') from error


def execute(args, rows):
    bin_dir = args.bin_dir.resolve()
    build_manifest = args.build_manifest.resolve()
    build = build_verified(bin_dir, build_manifest)
    pbs = pbs_environment()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    run = dict(schema_version=1, mode=args.mode, status='running',
               build_manifest=str(build_manifest), build_manifest_sha256=digest(build_manifest),
               build=build, environment=environment_manifest(pbs),
               input_parameters=dict(n=1000000, full_iterations=100,
                                     pilot_iterations=args.pilot_iterations,
                                     rebalance_interval=10, perturbation=0.10,
                                     kneighbor_k=5, leanmd_cutoff=4),
               invocations=rows)
    write_json(args.run_dir / 'run_manifest.json', run)
    baseline = None
    pairs = {}
    failures = 0
    failed_pairs = set()
    for row in rows:
        sample = dict(row, status='planned', rejection_reason=None,
                      build_manifest_sha256=run['build_manifest_sha256'],
                      started_at=None, completed_at=None)
        path = Path(row['manifest'])
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, sample)
        try:
            if row['pair_id'] in failed_pairs:
                raise ValueError('Earlier member of pair was rejected')
            build_verified(bin_dir, build_manifest)
            if digest(build_manifest) != run['build_manifest_sha256']:
                raise ValueError('Build manifest changed during the run')
            if any(Path(row[key]).exists() for key in ('csv', 'log', 'inventory')):
                raise ValueError('Sample artifact path already exists')
            sample['status'] = 'running'
            sample['started_at'] = datetime.now(timezone.utc).isoformat()
            write_json(path, sample)
            with open(row['log'], 'x', encoding='utf-8') as log:
                result = subprocess.run([pbs['launcher'], *row['argv']], stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0:
                raise ValueError(f'Benchmark exited {result.returncode}')
            times, mapping = accepted_output(row)
            if row['pair_id']:
                previous = pairs.get(row['pair_id'])
                if previous and previous['mapping'] != mapping:
                    previous_sample = previous['sample']
                    previous_sample['status'] = 'rejected'
                    previous_sample['rejection_reason'] = 'Pair device mapping mismatch'
                    write_json(Path(previous_sample['manifest']), previous_sample)
                    raise ValueError('Pair device mapping mismatch')
                pairs[row['pair_id']] = dict(mapping=mapping, sample=sample)
            if baseline is None:
                baseline = mapping
            elif mapping != baseline:
                raise ValueError('Device mapping differs from run baseline')
            sample.update(status='accepted', times=times, mapping=mapping)
        except (ValueError, OSError) as error:
            sample.update(status='rejected', rejection_reason=str(error))
            if row['pair_id']:
                failed_pairs.add(row['pair_id'])
            previous = pairs.get(row['pair_id']) if row['pair_id'] else None
            if previous and previous['sample'] is not sample:
                prior = previous['sample']
                prior['status'] = 'rejected'
                prior['rejection_reason'] = f'Paired sample rejected: {error}'
                write_json(Path(prior['manifest']), prior)
            failures += 1
        sample['completed_at'] = datetime.now(timezone.utc).isoformat()
        write_json(path, sample)
    run['status'] = 'completed' if failures == 0 else 'rejected'
    run['rejected_samples'] = sum(
        json.loads(Path(row['manifest']).read_text(encoding='utf-8'))['status'] == 'rejected'
        for row in rows)
    run['completed_at'] = datetime.now(timezone.utc).isoformat()
    write_json(args.run_dir / 'run_manifest.json', run)
    return 0 if failures == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--execute', action='store_true')
    parser.add_argument('--mode', choices=('pilot', 'full'), required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--bin-dir', type=Path, default=Path('build/cluster'))
    parser.add_argument('--build-manifest', type=Path, default=Path('build/cluster/build_manifest.json'))
    parser.add_argument('--pilot-iterations', type=int, default=8)
    args = parser.parse_args()
    if args.pilot_iterations < 1 or args.pilot_iterations >= 10:
        parser.error('Pilot iterations must be from 1 to 9')
    rows = schedule(args.run_dir.resolve(), args.bin_dir.resolve(), args.mode,
                    args.pilot_iterations)
    if args.dry_run:
        print(json.dumps(dict(schema_version=1, mode=args.mode,
                              run_dir=str(args.run_dir.resolve()), invocations=rows), indent=2))
        return 0
    try:
        return execute(args, rows)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f'Controlled run refused: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
