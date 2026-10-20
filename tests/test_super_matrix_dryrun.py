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

        # Check benchmark CSVs and logs
        for bench in ["kneighbor", "leanmd"]:
            for algo in ["erad", "hwtopolb"]:
                csv_path = os.path.join(self.out_dir, f"{bench}_{algo}_42.csv")
                log_path = os.path.join(self.out_dir, f"{bench}_{algo}_42.log")
                self.assertTrue(os.path.isfile(csv_path), f"Missing CSV: {csv_path}")
                self.assertTrue(os.path.isfile(log_path), f"Missing log: {log_path}")

                with open(csv_path, encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                # Header + 2 iteration rows
                self.assertEqual(len(lines), 3)
                self.assertIn("gpu_opencl", lines[1])

                with open(log_path, encoding="utf-8") as f:
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


if __name__ == "__main__":
    unittest.main()
