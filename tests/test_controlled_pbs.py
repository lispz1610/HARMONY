"""Command-level checks for resumable PBS submission and evidence collection."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest


REPO = Path(__file__).resolve().parents[1]


class ControlledPbsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'benchmarks').mkdir()
        for name in ('controlled_pbs.py', 'run_variability.py'):
            shutil.copy2(REPO / 'benchmarks' / name, self.root / 'benchmarks' / name)
        for name in ('analyze_pilot.py', 'report_variability.py'):
            (self.root / 'benchmarks' / name).write_text(
                '#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n'
                "Path(sys.argv[sys.argv.index('--output') + 1]).write_text('validated\\n')\n")
        (self.root / 'cluster_build.sh').write_text('#!/bin/bash\nexit 99\n')
        (self.root / 'job_harmony_variability.pbs').write_text('#!/bin/bash\n')
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        subprocess.run(['git', 'add', 'benchmarks', 'cluster_build.sh',
                        'job_harmony_variability.pbs'], cwd=self.root, check=True)
        subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                        'commit', '-qm', 'fixture'], cwd=self.root, check=True)
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.root,
                                           text=True).strip()
        binary_dir = self.root / 'build' / 'cluster'
        binary_dir.mkdir(parents=True)
        binaries = {}
        for name in ('kneighbor', 'leanmd', 'test_async_halo_opencl'):
            path = binary_dir / f'{name}.out'
            path.write_text('#!/bin/sh\nexit 0\n')
            path.chmod(0o755)
            binaries[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (binary_dir / 'build_manifest.json').write_text(json.dumps(dict(
            source_revision=revision, source_sha256=hashlib.sha256(b'').hexdigest(),
            binaries=binaries)))
        scripts = self.root / 'fake-bin'
        scripts.mkdir()
        qsub = scripts / 'qsub'
        qsub.write_text('''#!/usr/bin/env python3
import os, sys
from pathlib import Path
counter = Path(os.environ['FAKE_QSUB_COUNT'])
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
if os.environ.get('FAKE_QSUB_BAD'):
    print('submission response unavailable')
    sys.exit(1)
print(f'{122 + count}.fake')
''')
        qsub.chmod(0o755)
        qstat = scripts / 'qstat'
        qstat.write_text('''#!/usr/bin/env python3
import os, sys
from pathlib import Path
if os.environ.get('FAKE_QSTAT_MISSING'):
    sys.exit(153)
state = os.environ.get('FAKE_JOB_STATE', 'R')
if os.environ.get('FAKE_AUTO_DIR'):
    state = 'F' if (Path(os.environ['FAKE_AUTO_DIR']) / (sys.argv[-1] + '.done')).exists() else 'Q'
print(f'Job Id: {sys.argv[-1]}')
print(f'    Job_Name = {os.environ.get("FAKE_JOB_NAME", "unknown")}')
print(f'    job_state = {state}')
if state == 'F':
    print(f'    Exit_status = {os.environ.get("FAKE_JOB_EXIT", "0")}')
''')
        qstat.chmod(0o755)
        self.env = dict(os.environ, PATH=str(scripts) + os.pathsep + os.environ['PATH'],
                        FAKE_QSUB_COUNT=str(self.root / 'qsub-count'))
        self.command = [sys.executable, str(self.root / 'benchmarks' / 'controlled_pbs.py'),
                        '--experiment-id', 'sample', '--results-root', str(self.root / 'Results')]
        self.workflow_root = self.root / 'Results' / 'controlled-sample'

    def invoke(self, *extra, env=None):
        return subprocess.run([*self.command, '--once', *extra], cwd=self.root,
                              env=env or self.env, capture_output=True, text=True)

    def workflow(self):
        return json.loads((self.workflow_root / 'workflow.json').read_text())

    def finish_job(self, mode, job_id):
        run_dir = self.workflow_root / mode
        run_dir.mkdir()
        (self.workflow_root / f'{mode}.pbs.out').write_text(f'PBS output for {job_id}\n')
        (self.workflow_root / f'{mode}.pbs.status').write_text(
            f'pbs_job_id={job_id}\nscript_exit_code=0\n')
        (run_dir / 'run_manifest.json').write_text(json.dumps(dict(
            mode='aba' if mode.startswith('aba-') else mode, status='completed',
            rejected_samples=0,
            build_manifest_sha256=self.workflow()['build_manifest_sha256'],
            environment=dict(pbs=dict(job_id=job_id)))))
        if mode == 'smoke':
            paths = [run_dir / 'smoke_trace_summary.json',
                     run_dir / 'smoke.rank0.csv', run_dir / 'smoke.rank1.csv']
            for path in paths:
                path.write_text('validated trace\n')
            run = json.loads((run_dir / 'run_manifest.json').read_text())
            run['smoke_comparison'] = dict(
                condition=dict(workload='kneighbor', seed=123, algorithm='hwtopolb',
                               n=100000, iterations=100),
                unprofiled_loop_s=1.0, profiled_loop_s=1.2,
                trace_overhead_ratio=1.2,
                trace=dict(summary=str(paths[0]),
                           summary_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                           rank_traces=[dict(path=str(path),
                                             sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                        for path in paths[1:]]))
            (run_dir / 'run_manifest.json').write_text(json.dumps(run))
        if mode == 'full':
            diag = run_dir / 'diagnostic-8'
            diag.mkdir()
            (diag / 'diagnostic.manifest.json').write_text('{}')

    def test_one_command_waits_for_pbs_and_completes_all_stages(self):
        done = self.root / 'finished'
        done.mkdir()
        errors = []

        def finish_in_order():
            try:
                for mode, job_id in [('pilot', '123.fake'), ('smoke', '124.fake'),
                                     ('full', '125.fake')]:
                    deadline = time.monotonic() + 12
                    while time.monotonic() < deadline:
                        state = self.workflow() if (self.workflow_root / 'workflow.json').is_file() else {}
                        stage = state.get('stages', {}).get(mode)
                        if stage and stage['status'] == 'submitted':
                            self.finish_job(mode, job_id)
                            (done / f'{job_id}.done').touch()
                            break
                        time.sleep(0.02)
                    else:
                        raise AssertionError(f'{mode} was never submitted')
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=finish_in_order)
        worker.start()
        try:
            result = subprocess.run([*self.command, '--poll-interval-seconds', '1'],
                                    cwd=self.root, env=dict(self.env, FAKE_AUTO_DIR=str(done)),
                                    capture_output=True, text=True, timeout=15)
        finally:
            worker.join(timeout=15)
        self.assertEqual(errors, [])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Workflow complete', result.stdout)
        self.assertEqual((self.root / 'qsub-count').read_text(), '3')
        self.assertEqual({mode: stage['status'] for mode, stage in
                          self.workflow()['stages'].items()},
                         {'pilot': 'complete', 'smoke': 'complete', 'full': 'complete'})
        self.assertTrue((self.workflow_root / 'full-report.md').is_file())
        repeated = subprocess.run([*self.command, '--poll-interval-seconds', '1'],
                                  cwd=self.root, env=dict(self.env, FAKE_AUTO_DIR=str(done)),
                                  capture_output=True, text=True, timeout=5)
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '3')

    def test_interrupted_wait_resumes_without_resubmitting(self):
        process = subprocess.Popen([*self.command, '--poll-interval-seconds', '1'],
                                   cwd=self.root, env=self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = self.workflow() if (self.workflow_root / 'workflow.json').is_file() else {}
                if state.get('stages', {}).get('pilot', {}).get('status') == 'submitted':
                    break
                time.sleep(0.02)
            else:
                self.fail('Pilot was not submitted')
            process.send_signal(signal.SIGINT)
            _, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 130)
            self.assertIn('rerun the same command', stderr)
            resumed = self.invoke()
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual((self.root / 'qsub-count').read_text(), '1')
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_rerun_never_duplicates_jobs_and_advances_after_verified_completion(self):
        first = self.invoke()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '1')
        self.assertEqual(self.workflow()['stages']['pilot']['job_id'], '123.fake')
        second = self.invoke()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn('is R', second.stdout)
        self.assertEqual((self.root / 'qsub-count').read_text(), '1')
        self.finish_job('pilot', '123.fake')
        terminal = dict(self.env, FAKE_JOB_STATE='F')
        third = self.invoke(env=terminal)
        self.assertEqual(third.returncode, 0, third.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '2')
        self.assertTrue((self.workflow_root / 'pilot' / 'pbs.out').is_file())
        self.assertTrue((self.workflow_root / 'pilot' / 'pilot-analysis.md').is_file())
        self.assertEqual(self.workflow()['stages']['smoke']['job_id'], '124.fake')
        self.finish_job('smoke', '124.fake')
        fourth = self.invoke(env=terminal)
        self.assertEqual(fourth.returncode, 0, fourth.stderr)
        self.assertEqual(self.workflow()['stages']['full']['job_id'], '125.fake')
        self.assertEqual((self.root / 'qsub-count').read_text(), '3')
        self.finish_job('full', '125.fake')
        fifth = self.invoke(env=terminal)
        self.assertEqual(fifth.returncode, 0, fifth.stderr)
        self.assertTrue((self.workflow_root / 'full-report.md').is_file())
        self.assertEqual((self.root / 'qsub-count').read_text(), '3')
        sixth = self.invoke(env=terminal)
        self.assertEqual(sixth.returncode, 0, sixth.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '3')

    def test_uncertain_qsub_result_cannot_submit_twice(self):
        bad = dict(self.env, FAKE_QSUB_BAD='1')
        first = self.invoke(env=bad)
        self.assertNotEqual(first.returncode, 0)
        self.assertEqual(self.workflow()['stages']['pilot']['status'], 'submission_pending')
        second = self.invoke()
        self.assertNotEqual(second.returncode, 0)
        self.assertIn('no duplicate submission', second.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '1')
        name = self.workflow()['stages']['pilot']['job_name']
        adopted = self.invoke('--adopt-job-id', '123.fake',
                              env=dict(self.env, FAKE_JOB_NAME=name))
        self.assertEqual(adopted.returncode, 0, adopted.stderr)
        self.assertEqual(self.workflow()['stages']['pilot']['job_id'], '123.fake')
        self.assertEqual((self.root / 'qsub-count').read_text(), '1')

    def test_later_stage_uncertain_submission_can_be_adopted(self):
        self.assertEqual(self.invoke().returncode, 0)
        self.finish_job('pilot', '123.fake')
        uncertain = self.invoke(env=dict(self.env, FAKE_JOB_STATE='F', FAKE_QSUB_BAD='1'))
        self.assertNotEqual(uncertain.returncode, 0)
        self.assertEqual(self.workflow()['stages']['smoke']['status'], 'submission_pending')
        name = self.workflow()['stages']['smoke']['job_name']
        adopted = self.invoke('--adopt-job-id', '124.fake',
                              env=dict(self.env, FAKE_JOB_NAME=name))
        self.assertEqual(adopted.returncode, 0, adopted.stderr)
        self.assertEqual(self.workflow()['stages']['smoke']['job_id'], '124.fake')
        self.assertEqual((self.root / 'qsub-count').read_text(), '2')

    def test_preview_is_read_only_and_needs_no_scheduler(self):
        env = dict(self.env, PATH=os.environ['PATH'])
        result = self.invoke('--preview', env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.workflow_root.exists())

    def test_saved_terminal_scheduler_record_can_advance_after_qstat_expiry(self):
        self.assertEqual(self.invoke().returncode, 0)
        self.finish_job('pilot', '123.fake')
        saved = self.root / 'pilot-qstat.txt'
        saved.write_text('Job Id: 123.fake\n    job_state = F\n    Exit_status = 0\n')
        env = dict(self.env, FAKE_QSTAT_MISSING='1')
        blocked = self.invoke(env=env)
        self.assertNotEqual(blocked.returncode, 0)
        self.assertEqual((self.root / 'qsub-count').read_text(), '1')
        resumed = self.invoke('--scheduler-record', f'pilot={saved}', env=env)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '2')

    def test_terminal_job_waits_for_staged_output_even_if_qstat_expires(self):
        self.assertEqual(self.invoke().returncode, 0)
        terminal = self.invoke(env=dict(self.env, FAKE_JOB_STATE='F'))
        self.assertEqual(terminal.returncode, 0, terminal.stderr)
        self.assertIn('waiting for output staging', terminal.stdout)
        self.assertEqual((self.root / 'qsub-count').read_text(), '1')
        self.finish_job('pilot', '123.fake')
        resumed = self.invoke(env=dict(self.env, FAKE_QSTAT_MISSING='1'))
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '2')
        self.assertTrue((self.workflow_root / 'pilot' / 'pbs.scheduler-status.txt').is_file())

    def test_failed_pbs_exit_preserves_artifacts_without_submitting_next_stage(self):
        self.assertEqual(self.invoke().returncode, 0)
        self.finish_job('pilot', '123.fake')
        env = dict(self.env, FAKE_JOB_STATE='F', FAKE_JOB_EXIT='1')
        failed = self.invoke(env=env)
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual((self.root / 'qsub-count').read_text(), '1')
        self.assertEqual(self.workflow()['stages']['pilot']['status'], 'failed')
        self.assertTrue((self.workflow_root / 'pilot' / 'pbs.out').is_file())

    def test_missing_build_is_created_once_before_first_submission(self):
        manifest = self.root / 'build' / 'cluster' / 'build_manifest.json'
        saved = self.root / 'saved-build-manifest.json'
        saved.write_bytes(manifest.read_bytes())
        manifest.unlink()
        build_counter = self.root / 'build-count'
        (self.root / 'cluster_build.sh').write_text(
            '#!/bin/bash\nset -eu\n'
            f'printf x >> "{build_counter}"\n'
            f'cp "{saved}" "{manifest}"\n')
        first = self.invoke()
        self.assertEqual(first.returncode, 0, first.stderr)
        second = self.invoke()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(build_counter.read_text(), 'x')
        self.assertEqual((self.root / 'qsub-count').read_text(), '1')


if __name__ == '__main__':
    unittest.main()
