"""Command-level checks for the final GPU comparison report."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'benchmarks' / 'report_variability.py'
WORKLOADS = ('kneighbor', 'leanmd')
SEEDS = (42, 123, 456)
ALGORITHMS = ('erad', 'hwtopolb')


class ReportCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'full'
        self.run.mkdir()
        self.output = self.root / 'report.md'
        self.mapping = [dict(rank=rank, host=f'node{rank}', gpu_count=1,
                             local_index=0, global_index=rank, name='GPU A',
                             platform_name='Vendor', platform_version='OpenCL 3.0',
                             device_version='OpenCL 3.0', driver_version='1',
                             identity_kind='uuid', identity=f'gpu-{rank}')
                        for rank in (0, 1)]
        self.rows = []
        for workload in WORKLOADS:
            for algorithm in ALGORITHMS:
                self.add('warmup', 0, workload, 42, algorithm, None)
        for repetition in range(1, 6):
            seeds = SEEDS[(repetition - 1) % 3:] + SEEDS[:(repetition - 1) % 3]
            algorithms = ALGORITHMS if repetition % 2 else ALGORITHMS[::-1]
            for workload in WORKLOADS:
                for seed in seeds:
                    for algorithm in algorithms:
                        self.add('full', repetition, workload, seed, algorithm,
                                 f'r{repetition}_{workload}_{seed}')
        self.run_data = dict(schema_version=1, mode='full', status='completed',
                             rejected_samples=0, invocations=self.rows,
                             build_manifest='/build/manifest.json', build_manifest_sha256='c' * 64,
                             build=dict(source_revision='deadbeef',
                                        binaries={**{name: dict(sha256=name + '-sha') for name in WORKLOADS},
                                                  'test_async_halo_opencl': dict(
                                                      path='/bin/test_async_halo_opencl.out',
                                                      sha256='a' * 64)}),
                             input_parameters=dict(n=1000000, full_iterations=100,
                                                   rebalance_interval=10, perturbation=0.10,
                                                   kneighbor_k=5, leanmd_cutoff=4),
                             environment=dict(pbs=dict(job_id='123.cluster', launcher='/bin/prun',
                                                       slots=['node0', 'node1']),
                                              gpu_clock=dict(value=None, unavailable_reason='no tool')))
        (self.run / 'run_manifest.json').write_text(json.dumps(self.run_data))

    def add(self, phase, repetition, workload, seed, algorithm, pair_id):
        ordinal = len(self.rows) + 1
        prefix = self.run / phase / f'{ordinal:03d}_{workload}_{algorithm}_{seed}'
        prefix.parent.mkdir(exist_ok=True)
        row = dict(phase=phase, repetition=repetition, workload=workload,
                   seed=seed, algorithm=algorithm, pair_id=pair_id, ordinal=ordinal,
                   iterations=100, csv=str(prefix) + '.csv', log=str(prefix) + '.log',
                   inventory=str(prefix) + '.inventory.json',
                   manifest=str(prefix) + '.manifest.json',
                   argv=['/bin/' + workload + '.out', '--n', '1000000',
                         '--iterations', '100', '--balance-interval', '10',
                         '--perturbation', '0.10', '--seed', str(seed),
                         '--balance-algo', algorithm,
                         '--k', '5' if workload == 'kneighbor' else '4',
                         '--require-two-rank-gpu', '--quiet-runtime-metrics',
                         '--output-csv', str(prefix) + '.csv',
                         '--device-inventory-json', str(prefix) + '.inventory.json'])
        self.rows.append(row)
        loop = 1.0 if algorithm == 'erad' else 0.8
        with open(row['csv'], 'w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['iteration', 'step_time_s', 'action', 'rebalanced',
                             'rebalance_gain', 'execution_mode'])
            for iteration in range(100):
                writer.writerow([iteration, '0.005', 'rebalance' if iteration and iteration % 10 == 0 else 'compute',
                                 int(iteration != 0 and iteration % 10 == 0), 0, 'gpu_opencl'])
        Path(row['log']).write_text(f'mode=gpu_opencl loop_time_s={loop} '
                                    f'validation_time_s=0.1 total_time_s={loop + 0.2} '
                                    'validation=PASS\n')
        Path(row['inventory']).write_text(json.dumps(dict(schema_version=1, ranks=self.mapping)))
        Path(row['manifest']).write_text(json.dumps(dict(row, status='accepted',
            rejection_reason=None, build_manifest_sha256='c' * 64, mapping=self.mapping,
            times=dict(loop_time_s=loop, validation_time_s=0.1),
            started_at='2026-10-13T00:00:00Z', completed_at='2026-10-13T00:00:01Z')))

    def invoke(self, *extra):
        return subprocess.run([sys.executable, str(SCRIPT), '--run-dir', str(self.run),
                               '--output', str(self.output), *extra], capture_output=True, text=True)

    def test_complete_matrix_reports_all_adjacent_pairs_without_claiming_cluster_correctness(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.output.read_text()
        self.assertIn('30 adjacent pairs', report)
        self.assertIn('r1_kneighbor_42', report)
        self.assertIn('1.2500', report)
        self.assertIn('1.200000', report)
        self.assertIn('0.800000', report)
        self.assertIn('unavailable: no tool', report)
        self.assertIn('halo', report.lower())
        self.assertIn('inconclusive', report.lower())

    def test_incomplete_cpu_or_changed_mapping_is_rejected_before_report(self):
        row = self.rows[5]
        csv_path = Path(row['csv'])
        inventory_path = Path(row['inventory'])
        original_csv = csv_path.read_text()
        original_inventory = inventory_path.read_text()
        for altered in (''.join(original_csv.splitlines(keepends=True)[:-1]),
                        original_csv.replace('gpu_opencl', 'cpu_simulation', 1)):
            with self.subTest(kind=altered[-30:]):
                csv_path.write_text(altered)
                self.assertNotEqual(self.invoke().returncode, 0)
                self.assertFalse(self.output.exists())
        csv_path.write_text(original_csv)
        inventory = json.loads(original_inventory)
        inventory['ranks'][0]['identity'] = 'different'
        inventory_path.write_text(json.dumps(inventory))
        self.assertNotEqual(self.invoke().returncode, 0)
        self.assertFalse(self.output.exists())

    def test_single_seed_reversal_prevents_general_gain_even_with_halo_pass(self):
        for row in self.rows:
            if row['phase'] == 'full' and row['seed'] == 123 and row['algorithm'] == 'hwtopolb':
                log_path = Path(row['log'])
                log_path.write_text(log_path.read_text().replace('loop_time_s=0.8', 'loop_time_s=1.1')
                                    .replace('total_time_s=1.0', 'total_time_s=1.3'))
                manifest_path = Path(row['manifest'])
                sample = json.loads(manifest_path.read_text())
                sample['times']['loop_time_s'] = 1.1
                manifest_path.write_text(json.dumps(sample))
        halo = self.run / 'halo_reference.log'
        halo.write_text('PASS: asynchronous OpenCL halo matches two-step CPU reference\n')
        self.run_data['halo_reference'] = dict(command=['/bin/prun',
                                        '/bin/test_async_halo_opencl.out'], log=str(halo),
                                        binary_sha256='a' * 64, status='passed', exit_code=0)
        (self.run / 'run_manifest.json').write_text(json.dumps(self.run_data))
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.output.read_text()
        self.assertIn('0.9091', report)
        self.assertIn('No useful gain', report)
        self.assertIn('ERAD/HWTOPOLB/ERAD', report)

    def test_existing_report_is_never_overwritten(self):
        self.output.write_text('previous result')
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.read_text(), 'previous result')

    def test_unverified_halo_record_cannot_unlock_useful_gain(self):
        halo = self.run / 'halo_reference.log'
        halo.write_text('PASS: asynchronous OpenCL halo matches two-step CPU reference\n')
        self.run_data['halo_reference'] = dict(command=['/bin/prun',
            '/bin/test_async_halo_opencl.out'], log=str(halo),
            binary_sha256='b' * 64, status='passed', exit_code=0)
        (self.run / 'run_manifest.json').write_text(json.dumps(self.run_data))
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.output.read_text()
        self.assertIn('digest differs', report)
        self.assertIn('inconclusive', report.lower())
        self.assertNotIn('Useful gain supported', report)

    def test_verified_halo_allows_bounded_performance_signal_but_not_causal_claim(self):
        halo = self.run / 'halo_reference.log'
        halo.write_text('PASS: asynchronous OpenCL halo matches two-step CPU reference\n')
        self.run_data['halo_reference'] = dict(command=['/bin/prun',
            '/bin/test_async_halo_opencl.out'], log=str(halo),
            binary_sha256='a' * 64, status='passed', exit_code=0)
        (self.run / 'run_manifest.json').write_text(json.dumps(self.run_data))
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.output.read_text()
        self.assertIn('Paired performance supports a useful gain', report)
        self.assertIn('Report status:** provisional', report)
        self.assertIn('No causal attribution', report)

    def test_rejected_manifest_and_nonfinite_loop_are_refused(self):
        row = self.rows[4]
        path = Path(row['manifest'])
        original = json.loads(path.read_text())
        for changed in [dict(original, status='rejected'),
                        dict(original, times=dict(loop_time_s='nan', validation_time_s=0.1))]:
            with self.subTest(changed=changed['status'], times=changed.get('times')):
                path.write_text(json.dumps(changed))
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())

    def test_global_step_maxima_need_not_sum_to_max_rank_loop(self):
        row = self.rows[4]
        csv_path = Path(row['csv'])
        csv_path.write_text(csv_path.read_text().replace('0.005', '0.020'))
        log_path = Path(row['log'])
        log_path.write_text(log_path.read_text().replace('total_time_s=1.2',
                                                       'total_time_s=1.0'))
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.output.read_text()
        self.assertIn('-1.000000', report)
        self.assertIn('may each be maxima from different ranks', report)

    def test_early_late_drift_blocks_gain_even_when_iqr_is_small(self):
        ratios = (1.2, 1.2, 1.25, 1.321, 1.321)
        for row in self.rows:
            if row['phase'] != 'full' or row['workload'] != 'kneighbor' or \
                    row['seed'] != 42 or row['algorithm'] != 'hwtopolb':
                continue
            loop = 1 / ratios[row['repetition'] - 1]
            log_path = Path(row['log'])
            log_path.write_text(log_path.read_text().replace('loop_time_s=0.8',
                                                        f'loop_time_s={loop}'))
            manifest_path = Path(row['manifest'])
            sample = json.loads(manifest_path.read_text())
            sample['times']['loop_time_s'] = loop
            manifest_path.write_text(json.dumps(sample))
        halo = self.run / 'halo_reference.log'
        halo.write_text('PASS: asynchronous OpenCL halo matches two-step CPU reference\n')
        self.run_data['halo_reference'] = dict(command=['/bin/prun',
            '/bin/test_async_halo_opencl.out'], log=str(halo),
            binary_sha256='a' * 64, status='passed', exit_code=0)
        (self.run / 'run_manifest.json').write_text(json.dumps(self.run_data))
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.output.read_text()
        group = next(line for line in report.splitlines()
                     if line.startswith('| kneighbor | 42 |'))
        self.assertIn('| 1.2000 | 1.3210 |', group)
        self.assertTrue(group.endswith('| no |'))
        self.assertIn('inconclusive', report.lower())
        self.assertIn('UTC end ERAD/HWTOPOLB', report)


if __name__ == '__main__':
    unittest.main()
