"""Command-boundary execution gate tests with synthetic benchmark artifacts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'benchmarks' / 'run_variability.py'
FAKE_PRUN = '''#!/usr/bin/env python3
import csv, json, os, sys
from pathlib import Path
if '--version' in sys.argv:
    print('fake-prun 1.0')
    sys.exit(0)
def value(name):
    return sys.argv[sys.argv.index(name) + 1]
output = Path(value('--output-csv'))
count = int(value('--iterations'))
mode = 'cpu_simulation' if os.environ.get('FAKE_CPU') else 'gpu_opencl'
with output.open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['iteration', 'step_time_s', 'execution_mode'])
    for step in range(count):
        writer.writerow([step, '0.1', mode])
identity = 'changed' if os.environ.get('FAKE_MAPPING') and 'hwtopolb' in output.name else 'gpu'
ranks = [dict(rank=rank, host=f'node{rank}', gpu_count=1, local_index=0, global_index=rank,
              name='fake GPU', platform_name='fake', platform_version='3.0',
              device_version='3.0', driver_version='1', identity_kind='uuid',
              identity=f'{identity}-{rank}') for rank in (0, 1)]
Path(value('--device-inventory-json')).write_text(json.dumps({'schema_version': 1, 'ranks': ranks}))
print('validation=PASS loop_time_s=1.0 validation_time_s=0.1')
'''


class ExecuteGates(unittest.TestCase):
    def fixture(self, root):
        binaries = root / 'bin'
        binaries.mkdir()
        for workload in ('kneighbor', 'leanmd'):
            binary = binaries / (workload + '.out')
            binary.write_text('#!/bin/sh\nexit 0\n')
            binary.chmod(0o755)
        manifest = binaries / 'build_manifest.json'
        manifest.write_text(json.dumps({'binaries': {
            workload: {'sha256': hashlib.sha256((binaries / (workload + '.out')).read_bytes()).hexdigest()}
            for workload in ('kneighbor', 'leanmd')}}))
        launcher = root / 'prun'
        launcher.write_text(FAKE_PRUN)
        launcher.chmod(0o755)
        nodefile = root / 'nodefile'
        nodefile.write_text('node0\nnode1\n')
        env = dict(os.environ, PBS_JOBID='fake.1', PBS_NODEFILE=str(nodefile),
                   PATH=str(root) + os.pathsep + os.environ['PATH'])
        command = [sys.executable, str(SCRIPT), '--execute', '--mode', 'pilot',
                   '--run-dir', str(root / 'run'), '--bin-dir', str(binaries),
                   '--build-manifest', str(manifest)]
        return command, env, binaries, nodefile

    def test_wrong_allocation_and_binary_digest_refused_before_creating_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, env, binaries, nodefile = self.fixture(root)
            nodefile.write_text('node0\n')
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('two PBS slots', process.stderr)
            self.assertFalse((root / 'run').exists())
            nodefile.write_text('node0\nnode1\n')
            (binaries / 'leanmd.out').write_text('#!/bin/sh\nexit 2\n')
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('SHA-256 mismatch', process.stderr)
            self.assertFalse((root / 'run').exists())

    def test_cpu_fallback_and_pair_mapping_drift_reject_samples(self):
        for flag in ('FAKE_CPU', 'FAKE_MAPPING'):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                command, env, _, _ = self.fixture(root)
                env[flag] = '1'
                process = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertNotEqual(process.returncode, 0)
                samples = [json.loads(path.read_text()) for path in (root / 'run').rglob('*.manifest.json')]
                self.assertEqual(len(samples), 24)
                self.assertTrue(any(row['status'] == 'rejected' for row in samples))
                if flag == 'FAKE_MAPPING':
                    first = [row for row in samples if row['pair_id'] == 'r1_kneighbor_42']
                    self.assertEqual({row['status'] for row in first}, {'rejected'})

    def test_valid_synthetic_pilot_accepts_all_24_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, env, _, _ = self.fixture(root)
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            run = json.loads((root / 'run' / 'run_manifest.json').read_text())
            self.assertEqual(run['status'], 'completed')
            self.assertEqual(run['rejected_samples'], 0)
            self.assertEqual(len(list((root / 'run').rglob('*.manifest.json'))), 24)
            second = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn('File exists', second.stderr)


if __name__ == '__main__':
    unittest.main()
