#!/usr/bin/env python3
"""
Offline integration test harness for the Super Job matrix runner.
Verifies command-line argument validation, mandatory prebuilt binary enforcement,
dry-run execution mode, mock launcher execution, and directory isolation.
"""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest


class TestSuperMatrixDryRun(unittest.TestCase):
    """Offline test suite for benchmarks/run_super_matrix.sh."""

    @classmethod
    def setUpClass(cls):
        cls.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.script_path = os.path.join(cls.root_dir, "benchmarks", "run_super_matrix.sh")
        cls.mock_launcher_path = os.path.join(cls.root_dir, "tests", "mock_cluster_launcher.py")

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_super_matrix_")
        self.bin_dir = os.path.join(self.temp_dir, "bin")
        self.out_dir = os.path.join(self.temp_dir, "out")
        os.makedirs(self.bin_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

        # Create dummy executable files for required binaries
        self.required_binaries = [
            "kneighbor.out",
            "leanmd.out",
            "topo_probe.out",
            "test_async_halo_opencl.out",
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_dummy_binaries(self, binaries=None):
        """Creates minimal executable placeholder scripts in bin_dir."""
        if binaries is None:
            binaries = self.required_binaries
        for b in binaries:
            path = os.path.join(self.bin_dir, b)
            with open(path, "w", encoding="utf-8") as f:
                f.write("#!/bin/sh\nexit 0\n")
            st = os.stat(path)
            os.chmod(path, st.st_mode | stat.S_IEXEC)

    def _run_script(self, args, env=None, check=False):
        """Helper to invoke benchmarks/run_super_matrix.sh."""
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        cmd = ["bash", self.script_path] + args
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=run_env,
            check=check,
        )

    def test_missing_binary_directory_fails(self):
        """Runner must fail if --bin-dir points to a non-existent directory."""
        non_existent_dir = os.path.join(self.temp_dir, "no_such_dir")
        proc = self._run_script(["--bin-dir", non_existent_dir, "--output-dir", self.out_dir, "--dry-run"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Missing", proc.stderr + proc.stdout)

    def test_missing_single_binary_fails(self):
        """Runner must fail immediately if any required binary is missing from bin-dir."""
        # Only create 3 of the 4 binaries (omit test_async_halo_opencl.out)
        self._create_dummy_binaries(
            ["kneighbor.out", "leanmd.out", "topo_probe.out"]
        )
        proc = self._run_script(["--bin-dir", self.bin_dir, "--output-dir", self.out_dir, "--dry-run"])
        self.assertNotEqual(proc.returncode, 0)
        err = proc.stderr + proc.stdout
        self.assertIn("test_async_halo_opencl.out", err)
        self.assertIn("Missing prebuilt binary", err)

    def test_forbid_compute_node_compilation(self):
        """Runner must strictly forbid compilation and never invoke mpic++ on compute nodes."""
        self._create_dummy_binaries()
        mock_bin_dir = os.path.join(self.temp_dir, "mock_tools")
        os.makedirs(mock_bin_dir, exist_ok=True)
        mock_mpicxx = os.path.join(mock_bin_dir, "mpic++")
        with open(mock_mpicxx, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\necho 'ERROR: mpic++ invoked during run!' >&2\nexit 99\n")
        st = os.stat(mock_mpicxx)
        os.chmod(mock_mpicxx, st.st_mode | stat.S_IEXEC)

        proc = self._run_script(
            ["--bin-dir", self.bin_dir, "--output-dir", self.out_dir, "--dry-run"],
            env={"PATH": f"{mock_bin_dir}:{os.environ.get('PATH', '')}"},
        )
        self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}")
        self.assertNotIn("ERROR: mpic++ invoked", proc.stderr)

    def test_invalid_argument_n(self):
        """Runner must validate --n to be a positive integer."""
        self._create_dummy_binaries()
        for invalid_n in ["0", "-100", "abc"]:
            proc = self._run_script([
                "--bin-dir", self.bin_dir,
                "--output-dir", self.out_dir,
                "--n", invalid_n,
                "--dry-run",
            ])
            self.assertNotEqual(proc.returncode, 0, f"Expected failure for --n {invalid_n}")

    def test_invalid_argument_iterations(self):
        """Runner must validate --iterations to be a positive integer."""
        self._create_dummy_binaries()
        for invalid_iter in ["0", "-5", "xyz"]:
            proc = self._run_script([
                "--bin-dir", self.bin_dir,
                "--output-dir", self.out_dir,
                "--iterations", invalid_iter,
                "--dry-run",
            ])
            self.assertNotEqual(proc.returncode, 0, f"Expected failure for --iterations {invalid_iter}")

    def test_invalid_argument_seeds(self):
        """Runner must validate --seeds to be non-empty valid integers."""
        self._create_dummy_binaries()
        for invalid_seed in ["", "abc", "42,foo"]:
            proc = self._run_script([
                "--bin-dir", self.bin_dir,
                "--output-dir", self.out_dir,
                "--seeds", invalid_seed,
                "--dry-run",
            ])
            self.assertNotEqual(proc.returncode, 0, f"Expected failure for --seeds '{invalid_seed}'")

    def test_help_flag(self):
        """--help or -h should display usage and exit 0."""
        for flag in ["--help", "-h"]:
            proc = self._run_script([flag])
            self.assertEqual(proc.returncode, 0)
            self.assertIn("Usage:", proc.stdout)

    def test_output_dir_auto_creation(self):
        """Runner must create --output-dir if it does not already exist."""
        self._create_dummy_binaries()
        nested_out = os.path.join(self.temp_dir, "nested", "run_out")
        self.assertFalse(os.path.exists(nested_out))
        proc = self._run_script([
            "--bin-dir", self.bin_dir,
            "--output-dir", nested_out,
            "--dry-run",
        ])
        self.assertEqual(proc.returncode, 0, f"Script failed: {proc.stderr}")
        self.assertTrue(os.path.isdir(nested_out))

    def test_dry_run_mode(self):
        """--dry-run should print complete execution plan without executing jobs."""
        self._create_dummy_binaries()
        proc = self._run_script([
            "--bin-dir", self.bin_dir,
            "--output-dir", self.out_dir,
            "--dry-run",
        ])
        self.assertEqual(proc.returncode, 0, f"Script failed: {proc.stderr}")
        out = proc.stdout
        self.assertIn("[DRY-RUN]", out)
        self.assertIn("Preflight Verification", out)
        self.assertIn("topo_probe.out", out)
        self.assertIn("test_async_halo_opencl.out", out)
        self.assertIn("kneighbor", out)
        self.assertIn("leanmd", out)
        self.assertIn("123", out)

    def test_mock_launcher_execution(self):
        """Runner should execute successfully with MPI_LAUNCHER override pointing to mock launcher."""
        self._create_dummy_binaries()
        launcher_cmd = f"python3 {self.mock_launcher_path}"
        proc = self._run_script([
            "--bin-dir", self.bin_dir,
            "--output-dir", self.out_dir,
            "--n", "64",
            "--iterations", "2",
            "--seeds", "42",
            "--launcher", launcher_cmd,
        ])
        self.assertEqual(proc.returncode, 0, f"Script failed with error:\n{proc.stderr}\nOutput:\n{proc.stdout}")

        # Check that preflight topo probe produced valid JSON
        topo_json = os.path.join(self.out_dir, "topo_metrics_cluster.json")
        self.assertTrue(os.path.isfile(topo_json), "topo_metrics_cluster.json was not created")
        with open(topo_json, encoding="utf-8") as f:
            data = json.load(f)
        self.assertFalse(data.get("synthetic", True), "Expected synthetic=False in mock topology")
        self.assertEqual(len(data.get("pcie_latency_ns", [])), 2)

        # Check benchmark CSVs and logs (matrix rep1 and warmup, no unversioned symlinks)
        for bench in ["kneighbor", "leanmd"]:
            for algo in ["erad", "hwtopolb"]:
                rep1_csv = os.path.join(self.out_dir, f"{bench}_{algo}_42_rep1.csv")
                rep1_log = os.path.join(self.out_dir, f"{bench}_{algo}_42_rep1.log")
                warmup_csv = os.path.join(self.out_dir, f"warmup_{bench}_{algo}_42.csv")
                warmup_log = os.path.join(self.out_dir, f"warmup_{bench}_{algo}_42.log")
                unversioned_csv = os.path.join(self.out_dir, f"{bench}_{algo}_42.csv")

                self.assertTrue(os.path.isfile(rep1_csv), f"Missing rep1 CSV: {rep1_csv}")
                self.assertTrue(os.path.isfile(rep1_log), f"Missing rep1 log: {rep1_log}")
                self.assertTrue(os.path.isfile(warmup_csv), f"Missing warmup CSV: {warmup_csv}")
                self.assertTrue(os.path.isfile(warmup_log), f"Missing warmup log: {warmup_log}")
                # Verify unversioned symlink is NOT created (prevent duplicate file ingestion)
                self.assertFalse(os.path.exists(unversioned_csv), f"Unversioned symlink should not exist: {unversioned_csv}")

                with open(rep1_csv, encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                # Header + 2 iteration rows
                self.assertEqual(len(lines), 3)
                self.assertIn("gpu_opencl", lines[1])

                with open(rep1_log, encoding="utf-8") as f:
                    log_text = f.read()
                self.assertIn("NVIDIA A100", log_text)
                self.assertIn("[DCL][balance-metrics]", log_text)
                self.assertIn("total_time_s=", log_text)

    def test_output_isolation_does_not_modify_historical_benchmarks(self):
        """Runner must never write to or modify existing files in benchmarks/."""
        self._create_dummy_binaries()
        benchmarks_dir = os.path.join(self.root_dir, "benchmarks")

        # Record hashes of existing historical CSV files
        historical_files = [
            f for f in os.listdir(benchmarks_dir)
            if f.endswith(".csv") or f.endswith(".log")
        ]
        self.assertGreater(len(historical_files), 0, "Expected existing benchmark files")

        hashes_before = {}
        for fname in historical_files:
            fpath = os.path.join(benchmarks_dir, fname)
            with open(fpath, "rb") as f:
                hashes_before[fname] = hashlib.sha256(f.read()).hexdigest()

        # Run mock execution in isolated directory
        launcher_cmd = f"python3 {self.mock_launcher_path}"
        proc = self._run_script([
            "--bin-dir", self.bin_dir,
            "--output-dir", self.out_dir,
            "--n", "32",
            "--iterations", "2",
            "--seeds", "42",
            "--launcher", launcher_cmd,
        ])
        self.assertEqual(proc.returncode, 0, f"Script failed: {proc.stderr}")

        # Check that no historical file was modified
        for fname in historical_files:
            fpath = os.path.join(benchmarks_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                hashes_before[fname],
                f"Historical benchmark file {fname} was modified!"
            )

    def test_preflight_failure_aborts_execution(self):
        """If preflight halo check fails, runner must abort before running benchmarks."""
        self._create_dummy_binaries()
        launcher_cmd = f"python3 {self.mock_launcher_path}"
        proc = self._run_script(
            [
                "--bin-dir", self.bin_dir,
                "--output-dir", self.out_dir,
                "--n", "32",
                "--iterations", "2",
                "--seeds", "42",
                "--launcher", launcher_cmd,
            ],
            env={"MOCK_FAIL_PREFLIGHT_HALO": "1"},
        )
        self.assertNotEqual(proc.returncode, 0, "Expected non-zero exit code on preflight failure")
        self.assertIn("Preflight", proc.stderr + proc.stdout)

        # Confirm no benchmark CSVs were created
        benchmark_csvs = [f for f in os.listdir(self.out_dir) if f.endswith(".csv")]
        self.assertEqual(len(benchmark_csvs), 0, "CSVs should not be created if preflight fails")

    def test_asymmetric_gpu_allocation_fails_preflight(self):
        """Asymmetric GPU configuration (e.g., 2 GPUs on rank 0, 0 GPUs on rank 1) must fail preflight."""
        self._create_dummy_binaries()
        launcher_cmd = f"python3 {self.mock_launcher_path}"
        proc = self._run_script(
            [
                "--bin-dir", self.bin_dir,
                "--output-dir", self.out_dir,
                "--n", "32",
                "--iterations", "2",
                "--seeds", "42",
                "--launcher", launcher_cmd,
            ],
            env={"MOCK_ASYMMETRIC_GPUS": "1"},
        )
        self.assertNotEqual(proc.returncode, 0, "Expected non-zero exit code on asymmetric GPU allocation")
        output = proc.stderr + proc.stdout
        self.assertIn("ERROR: Asynchronous halo test preflight failed (requires at least 1 OpenCL GPU per rank)", output)

        # Confirm no benchmark CSVs were created
        benchmark_csvs = [f for f in os.listdir(self.out_dir) if f.endswith(".csv")]
        self.assertEqual(len(benchmark_csvs), 0, "CSVs should not be created if preflight fails")

    def test_warmup_matrix_and_aba_execution_schedule(self):
        """Verify warmup (4), 5 repetitions (60), and ABA (6) execute in exact scheduled order."""
        self._create_dummy_binaries()
        launcher_cmd = f"python3 {self.mock_launcher_path}"
        proc = self._run_script([
            "--bin-dir", self.bin_dir,
            "--output-dir", self.out_dir,
            "--n", "64",
            "--iterations", "2",
            "--seeds", "42,123,456",
            "--launcher", launcher_cmd,
        ])
        self.assertEqual(proc.returncode, 0, f"Script failed: {proc.stderr}\n{proc.stdout}")

        manifest_path = os.path.join(self.out_dir, "run_manifest.json")
        self.assertTrue(os.path.isfile(manifest_path), "run_manifest.json was not created")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        runs = manifest_data.get("runs", manifest_data)
        self.assertEqual(len(runs), 70, f"Expected 70 total runs (4 warmup + 60 matrix + 6 ABA), got {len(runs)}")

        # 1. Warmup checks (4 runs: kneighbor erad/hwtopolb, leanmd erad/hwtopolb with default seed 42)
        warmup_runs = [r for r in runs if r.get("stage") == "warmup"]
        self.assertEqual(len(warmup_runs), 4)
        expected_warmup = [
            ("kneighbor", "erad", 42),
            ("kneighbor", "hwtopolb", 42),
            ("leanmd", "erad", 42),
            ("leanmd", "hwtopolb", 42),
        ]
        for r, exp in zip(warmup_runs, expected_warmup):
            self.assertEqual((r["benchmark"], r["algorithm"], r["seed"]), exp)
            self.assertEqual(r["status"], "success")
            self.assertEqual(r["exit_code"], 0)

        # 2. Matrix checks (5 repetitions = 60 runs)
        matrix_runs = [r for r in runs if r.get("stage") == "matrix"]
        self.assertEqual(len(matrix_runs), 60)

        expected_rep_seeds = {
            1: [42, 123, 456],
            2: [123, 456, 42],
            3: [456, 42, 123],
            4: [42, 456, 123],
            5: [123, 42, 456],
        }
        expected_rep_algos = {
            1: ["erad", "hwtopolb"],
            2: ["hwtopolb", "erad"],
            3: ["erad", "hwtopolb"],
            4: ["hwtopolb", "erad"],
            5: ["erad", "hwtopolb"],
        }

        for rep in range(1, 6):
            rep_runs = [r for r in matrix_runs if r.get("repetition") == rep]
            self.assertEqual(len(rep_runs), 12, f"Repetition {rep} should have 12 runs")

            exp_seeds = expected_rep_seeds[rep]
            exp_algos = expected_rep_algos[rep]

            idx = 0
            for bench in ["kneighbor", "leanmd"]:
                for algo in exp_algos:
                    for seed in exp_seeds:
                        r = rep_runs[idx]
                        self.assertEqual(
                            (r["benchmark"], r["algorithm"], r["seed"]),
                            (bench, algo, seed),
                            f"Mismatch in Rep {rep} run {idx}: expected {(bench, algo, seed)}, got {(r['benchmark'], r['algorithm'], r['seed'])}"
                        )
                        self.assertEqual(r["status"], "success")
                        idx += 1

        # 3. Dedicated A/B/A checks for seed 123 (6 runs)
        aba_runs = [r for r in runs if r.get("stage") == "aba"]
        self.assertEqual(len(aba_runs), 6)
        expected_aba = [
            ("kneighbor", "erad", 123),
            ("kneighbor", "hwtopolb", 123),
            ("kneighbor", "erad", 123),
            ("leanmd", "erad", 123),
            ("leanmd", "hwtopolb", 123),
            ("leanmd", "erad", 123),
        ]
        for r, exp in zip(aba_runs, expected_aba):
            self.assertEqual((r["benchmark"], r["algorithm"], r["seed"]), exp)
            self.assertEqual(r["status"], "success")

    def test_hardware_telemetry_created(self):
        """Verify hardware_telemetry.log is created with snapshots before/after blocks."""
        self._create_dummy_binaries()
        launcher_cmd = f"python3 {self.mock_launcher_path}"
        proc = self._run_script([
            "--bin-dir", self.bin_dir,
            "--output-dir", self.out_dir,
            "--n", "64",
            "--iterations", "2",
            "--seeds", "42",
            "--launcher", launcher_cmd,
        ])
        self.assertEqual(proc.returncode, 0, f"Script failed: {proc.stderr}\n{proc.stdout}")

        telem_file = os.path.join(self.out_dir, "hardware_telemetry.log")
        self.assertTrue(os.path.isfile(telem_file), "hardware_telemetry.log was not created")

        with open(telem_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Without nvidia-smi installed, should contain placeholder entries
        self.assertGreater(len(content.strip()), 0)
        self.assertIn("warmup", content)
        self.assertIn("rep", content)
        self.assertIn("aba", content)

    def test_hardware_telemetry_with_mock_nvidia_smi(self):
        """Verify hardware_telemetry queries nvidia-smi with exact query flags when available."""
        self._create_dummy_binaries()
        mock_bin_dir = os.path.join(self.temp_dir, "mock_smi_dir")
        os.makedirs(mock_bin_dir, exist_ok=True)
        mock_smi = os.path.join(mock_bin_dir, "nvidia-smi")
        with open(mock_smi, "w", encoding="utf-8") as f:
            f.write(
                '#!/bin/sh\n'
                'echo "2026/10/20 22:00:00.000, NVIDIA A100-PCIE-40GB, 38, 12, 1410, 1215, 78.42, [N/A]"\n'
            )
        st = os.stat(mock_smi)
        os.chmod(mock_smi, st.st_mode | stat.S_IEXEC)

        launcher_cmd = f"python3 {self.mock_launcher_path}"
        proc = self._run_script(
            [
                "--bin-dir", self.bin_dir,
                "--output-dir", self.out_dir,
                "--n", "64",
                "--iterations", "2",
                "--seeds", "42",
                "--launcher", launcher_cmd,
            ],
            env={"PATH": f"{mock_bin_dir}:{os.environ.get('PATH', '')}"},
        )
        self.assertEqual(proc.returncode, 0, f"Script failed: {proc.stderr}\n{proc.stdout}")

        telem_file = os.path.join(self.out_dir, "hardware_telemetry.log")
        with open(telem_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("NVIDIA A100-PCIE-40GB", content)
        self.assertIn("78.42", content)

    def test_hardware_telemetry_multinode_mock_launcher(self):
        """Mock launcher queries telemetry across both allocated nodes (compute-1-0 and compute-1-1)."""
        self._create_dummy_binaries()
        launcher_cmd = f"python3 {self.mock_launcher_path}"
        proc = self._run_script([
            "--bin-dir", self.bin_dir,
            "--output-dir", self.out_dir,
            "--n", "64",
            "--iterations", "2",
            "--seeds", "42",
            "--launcher", launcher_cmd,
        ])
        self.assertEqual(proc.returncode, 0, f"Script failed: {proc.stderr}\n{proc.stdout}")

        telem_file = os.path.join(self.out_dir, "hardware_telemetry.log")
        with open(telem_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("compute-1-0", content)
        self.assertIn("compute-1-1", content)
        # Verify rank tagging
        self.assertIn("compute-1-0, 0,", content)
        self.assertIn("compute-1-1, 1,", content)

    def test_hardware_telemetry_pbs_nodefile(self):
        """When launcher is not set but PBS_NODEFILE is provided, records telemetry for each node in file."""
        self._create_dummy_binaries()
        nodefile = os.path.join(self.temp_dir, "pbs_nodes")
        with open(nodefile, "w", encoding="utf-8") as f:
            f.write("compute-1-0\ncompute-1-1\n")

        # Pass empty or non-matching launcher flag override
        proc = self._run_script(
            [
                "--bin-dir", self.bin_dir,
                "--output-dir", self.out_dir,
                "--n", "64",
                "--iterations", "2",
                "--seeds", "42",
                "--skip-preflight",
            ],
            env={"PBS_NODEFILE": nodefile, "MPI_LAUNCHER": ""},
        )
        self.assertEqual(proc.returncode, 0, f"Script failed: {proc.stderr}\n{proc.stdout}")

        telem_file = os.path.join(self.out_dir, "hardware_telemetry.log")
        with open(telem_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("compute-1-0", content)
        self.assertIn("compute-1-1", content)
        self.assertIn("compute-1-0, 0,", content)
        self.assertIn("compute-1-1, 1,", content)

    def test_fault_tolerance_sample_failure_continues_remaining_runs(self):
        """When mock launcher fails on one run, runner logs failure and continues remaining runs."""
        self._create_dummy_binaries()
        fail_marker = os.path.join(self.temp_dir, "fail_once.marker")
        launcher_cmd = f"python3 {self.mock_launcher_path}"
        proc = self._run_script(
            [
                "--bin-dir", self.bin_dir,
                "--output-dir", self.out_dir,
                "--n", "64",
                "--iterations", "2",
                "--seeds", "42",
                "--launcher", launcher_cmd,
            ],
            env={"MOCK_FAIL_ONCE_FILE": fail_marker},
        )
        self.assertEqual(proc.returncode, 1, "Runner should exit with code 1 on sample failure")
        self.assertIn("1 sample(s) failed or timed out during matrix execution", proc.stderr)

        manifest_path = os.path.join(self.out_dir, "run_manifest.json")
        self.assertTrue(os.path.isfile(manifest_path))
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        runs = manifest.get("runs", manifest)
        # Should have executed all scheduled runs despite the single failure
        # For 1 seed (42): 4 warmup + 20 matrix (2 bench * 2 algo * 1 seed * 5 rep) + 6 ABA = 30 runs
        self.assertEqual(len(runs), 30)

        failed_runs = [r for r in runs if r.get("status") == "failed"]
        self.assertEqual(len(failed_runs), 1, "Exactly one run should be recorded as failed")
        self.assertNotEqual(failed_runs[0]["exit_code"], 0)

        success_runs = [r for r in runs if r.get("status") == "success"]
        self.assertEqual(len(success_runs), 29)

    def test_fault_tolerance_sample_timeout_continues_remaining_runs(self):
        """When a sample times out, runner logs timeout in manifest and continues remaining runs."""
        self._create_dummy_binaries()
        sleep_marker = os.path.join(self.temp_dir, "sleep_once.marker")
        launcher_cmd = f"python3 {self.mock_launcher_path}"
        proc = self._run_script(
            [
                "--bin-dir", self.bin_dir,
                "--output-dir", self.out_dir,
                "--n", "64",
                "--iterations", "2",
                "--seeds", "42",
                "--launcher", launcher_cmd,
            ],
            env={
                "SAMPLE_TIMEOUT_SECONDS": "2",
                "MOCK_SLEEP_ONCE_FILE": sleep_marker,
                "MOCK_SLEEP_SECONDS": "4",
            },
        )
        self.assertEqual(proc.returncode, 1, "Runner should exit with code 1 on sample timeout")
        self.assertIn("1 sample(s) failed or timed out during matrix execution", proc.stderr)

        manifest_path = os.path.join(self.out_dir, "run_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        runs = manifest.get("runs", manifest)
        self.assertEqual(len(runs), 30)
        timed_out = [r for r in runs if r.get("status") == "timeout"]
        self.assertEqual(len(timed_out), 1)
        self.assertEqual(timed_out[0]["exit_code"], 124)

        success_runs = [r for r in runs if r.get("status") == "success"]
        self.assertEqual(len(success_runs), 29)

    def test_skip_preflight_flag(self):
        """--skip-preflight bypasses preflight checks even when mock failure triggers are active."""
        self._create_dummy_binaries()
        launcher_cmd = f"python3 {self.mock_launcher_path}"
        proc = self._run_script(
            [
                "--bin-dir", self.bin_dir,
                "--output-dir", self.out_dir,
                "--n", "32",
                "--iterations", "2",
                "--seeds", "42",
                "--launcher", launcher_cmd,
                "--skip-preflight",
            ],
            env={"MOCK_FAIL_PREFLIGHT_HALO": "1", "MOCK_FAIL_TOPO_PROBE": "1"},
        )
        self.assertEqual(proc.returncode, 0, f"Script failed with --skip-preflight: {proc.stderr}")
        self.assertIn("SKIPPED via --skip-preflight", proc.stdout)

        # Check benchmark ran successfully
        manifest_path = os.path.join(self.out_dir, "run_manifest.json")
        self.assertTrue(os.path.isfile(manifest_path))

    def test_preflight_wrapped_with_timeout_in_script(self):
        """run_super_matrix.sh must wrap preflight probe and halo calls with 'timeout 60s'."""
        with open(self.script_path, "r", encoding="utf-8") as f:
            script_text = f.read()

        self.assertIn("timeout 60s", script_text)
        self.assertTrue(
            "timeout 60s \"${launcher_cmd[@]}\" \"$bin_dir/topo_probe.out\"" in script_text or
            "timeout 60s" in script_text,
            "Expected preflight topo probe wrapped with timeout 60s"
        )
        self.assertTrue(
            "timeout 60s \"${launcher_cmd[@]}\" \"$bin_dir/test_async_halo_opencl.out\"" in script_text or
            "timeout 60s" in script_text,
            "Expected preflight halo test wrapped with timeout 60s"
        )

    def test_multinode_telemetry_dryrun_e2e_analysis(self):
        """End-to-end integration: runner captures multi-node telemetry and analyzer produces json & report."""
        self._create_dummy_binaries()
        launcher_cmd = f"python3 {self.mock_launcher_path}"
        proc = self._run_script([
            "--bin-dir", self.bin_dir,
            "--output-dir", self.out_dir,
            "--n", "32",
            "--iterations", "2",
            "--seeds", "42",
            "--launcher", launcher_cmd,
            "--skip-preflight",
        ])
        self.assertEqual(proc.returncode, 0, f"Script failed: {proc.stderr}\n{proc.stdout}")

        # Run analyzer
        analyzer_script = os.path.join(self.root_dir, "benchmarks", "analyze_super_matrix.py")
        res = subprocess.run(
            ["python3", analyzer_script, "--output-dir", self.out_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Analyzer failed: {res.stderr}\n{res.stdout}")

        # Check hardware_telemetry.json
        telemetry_json = os.path.join(self.out_dir, "hardware_telemetry.json")
        self.assertTrue(os.path.isfile(telemetry_json), "hardware_telemetry.json not found")
        with open(telemetry_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        hosts = {entry.get("host") for entry in data}
        ranks = {entry.get("rank") for entry in data}
        self.assertIn("compute-1-0", hosts)
        self.assertIn("compute-1-1", hosts)
        self.assertIn(0, ranks)
        self.assertIn(1, ranks)

        # Check SUPER_JOB_REPORT.md
        report_md = os.path.join(self.out_dir, "SUPER_JOB_REPORT.md")
        self.assertTrue(os.path.isfile(report_md), "SUPER_JOB_REPORT.md not found")
        with open(report_md, "r", encoding="utf-8") as f:
            report_text = f.read()

        self.assertIn("compute-1-0", report_text)
        self.assertIn("compute-1-1", report_text)
        self.assertIn("Hardware Telemetry by Node & Rank", report_text)
        self.assertIn("Aggregate Cluster Metrics", report_text)


if __name__ == "__main__":
    unittest.main()
