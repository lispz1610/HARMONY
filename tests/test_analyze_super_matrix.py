#!/usr/bin/env python3
"""
Test suite for benchmarks/analyze_super_matrix.py.
Verifies log parsing, metric extraction, CSV reconciliation,
paired analysis, summary statistics, artifact generation,
and defensive handling of partial or failed runs.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Add benchmarks/ to sys.path so we can import the module directly
ROOT_DIR = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = ROOT_DIR / "benchmarks"
sys.path.insert(0, str(BENCHMARKS_DIR))

try:
    import analyze_super_matrix as asm
except ImportError:
    asm = None


class TestAnalyzeSuperMatrix(unittest.TestCase):
    """Unit and integration tests for analyze_super_matrix.py."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_asm_")
        self.output_dir = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_module_importable(self):
        """Module analyze_super_matrix must exist and be importable."""
        self.assertIsNotNone(asm, "analyze_super_matrix could not be imported")

    def test_parse_balance_metrics_synthetic(self):
        """Parses [DCL][balance-metrics] block into structured records."""
        sample_log = """
[DCL][balance-metrics] iteration=1
  policy=off
  action=skip
  reason=balance disabled
  T_compute_interval_s=2.7648e-05
  T_comm_interval_s=0
  T_balance_total_s=0
  T_rebalance_apply_s=0
  T_compute_accum_s=2.7648e-05
  T_comm_accum_s=0
  T_balance_accum_s=0
  T_rebalance_apply_accum_s=0
  device_times_s:
    part[0]=2.3552e-05
    part[1]=2.7648e-05
  OLD loads:
    part[0] cum=50% share=50% device_global=0 rank=0 local_index=0 offset=0 count=500000
    part[1] cum=100% share=50% device_global=1 rank=1 local_index=-1 offset=500000 count=500000
    total=100%
  CURRENT loads:
    part[0] cum=40% share=40% device_global=0 rank=0 local_index=0 offset=0 count=400000
    part[1] cum=100% share=60% device_global=1 rank=1 local_index=-1 offset=400000 count=600000
    total=100%
"""
        metrics = asm.parse_balance_metrics(sample_log)
        self.assertEqual(len(metrics), 1)
        m = metrics[0]
        self.assertEqual(m["iteration"], 1)
        self.assertAlmostEqual(m["T_compute_interval_s"], 2.7648e-05)
        self.assertAlmostEqual(m["T_comm_interval_s"], 0.0)
        self.assertAlmostEqual(m["T_balance_total_s"], 0.0)
        self.assertAlmostEqual(m["T_rebalance_apply_s"], 0.0)

        # Check OLD loads
        self.assertEqual(len(m["old_loads"]), 2)
        self.assertEqual(m["old_loads"][0]["rank"], 0)
        self.assertEqual(m["old_loads"][0]["offset"], 0)
        self.assertEqual(m["old_loads"][0]["count"], 500000)
        self.assertEqual(m["old_loads"][1]["rank"], 1)
        self.assertEqual(m["old_loads"][1]["offset"], 500000)
        self.assertEqual(m["old_loads"][1]["count"], 500000)

        # Check CURRENT loads
        self.assertEqual(len(m["current_loads"]), 2)
        self.assertEqual(m["current_loads"][0]["count"], 400000)
        self.assertEqual(m["current_loads"][1]["count"], 600000)

    def test_parse_real_historical_logs(self):
        """Validates log parsing against actual historical logs in benchmarks/."""
        erad_log = BENCHMARKS_DIR / "kneighbor_erad_42.log"
        hw_log = BENCHMARKS_DIR / "kneighbor_hwtopolb_123.log"
        if not erad_log.exists() or not hw_log.exists():
            self.skipTest("Historical logs not found in benchmarks/")

        erad_data = asm.parse_log_file(erad_log)
        self.assertEqual(erad_data["benchmark"], "kneighbor")
        self.assertEqual(erad_data["algo"], "erad")
        self.assertEqual(erad_data["mode"], "gpu_opencl")
        self.assertAlmostEqual(erad_data["total_time_s"], 3.73996)
        self.assertEqual(erad_data["num_rebalances"], 9)
        self.assertAlmostEqual(erad_data["avg_rebalance_gain"], 6.34934e-05)
        self.assertEqual(len(erad_data["balance_metrics"]), 100)
        self.assertEqual(len(erad_data["devices"]), 2)
        self.assertEqual(erad_data["devices"][0]["name"], "NVIDIA A100 80GB PCIe")

        hw_data = asm.parse_log_file(hw_log)
        self.assertEqual(hw_data["benchmark"], "kneighbor")
        self.assertEqual(hw_data["algo"], "hwtopolb")
        self.assertAlmostEqual(hw_data["total_time_s"], 3.81972)

    def test_parse_real_historical_csv(self):
        """Validates CSV parsing and unaccounted time calculation on historical CSV."""
        csv_path = BENCHMARKS_DIR / "kneighbor_erad_42.csv"
        if not csv_path.exists():
            self.skipTest("Historical CSV not found in benchmarks/")

        csv_data = asm.parse_csv_file(csv_path)
        self.assertEqual(csv_data["step_count"], 100)
        self.assertEqual(csv_data["rebalance_count"], 9)
        self.assertEqual(csv_data["execution_modes"], {"gpu_opencl"})
        # 1.830s ordinary steps + 0.911s rebalanced steps ≈ 2.741s
        self.assertAlmostEqual(csv_data["step_sum_s"], 2.741042784, places=4)

        # Unaccounted calculation: 3.73996 - 2.741042784 ≈ 0.9989s
        unaccounted = 3.73996 - csv_data["step_sum_s"]
        self.assertAlmostEqual(unaccounted, 0.9989, places=3)

    def test_parse_hardware_telemetry(self):
        """Parses hardware_telemetry.log into structured telemetry records."""
        telemetry_file = self.output_dir / "hardware_telemetry.log"
        telemetry_file.write_text(
            "timestamp, name, temperature.gpu, utilization.gpu [%], utilization.memory [%], power.draw [W], clocks.current.sm [MHz]\n"
            "2026/10/20 22:00:00.000, NVIDIA A100 80GB PCIe, 38, 0 %, 0 %, 64.20 W, 1410 MHz\n"
            "2026/10/20 22:00:05.000, NVIDIA A100 80GB PCIe, 42, 98 %, 35 %, 210.50 W, 1410 MHz\n",
            encoding="utf-8"
        )
        records = asm.parse_hardware_telemetry(telemetry_file)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["temperature_gpu"], 38)
        self.assertEqual(records[0]["utilization_gpu_pct"], 0)
        self.assertAlmostEqual(records[1]["power_draw_w"], 210.50)
        self.assertEqual(records[1]["utilization_gpu_pct"], 98)

    def test_paired_analysis_and_statistics(self):
        """Computes ERAD ÷ HWTOPOLB speedup ratios, percentages, and repetition statistics."""
        runs = [
            # Repetition 1
            {
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 42,
                "repetition": 1,
                "stage": "matrix",
                "wall_s": 3.74,
                "step_sum_s": 2.74,
                "unaccounted_s": 1.00,
                "rebalance_count": 9,
                "status": "SUCCESS",
            },
            {
                "benchmark": "kneighbor",
                "algo": "hwtopolb",
                "seed": 42,
                "repetition": 1,
                "stage": "matrix",
                "wall_s": 0.22,
                "step_sum_s": 0.16,
                "unaccounted_s": 0.06,
                "rebalance_count": 9,
                "status": "SUCCESS",
            },
            # Repetition 2
            {
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 42,
                "repetition": 2,
                "stage": "matrix",
                "wall_s": 3.76,
                "step_sum_s": 2.75,
                "unaccounted_s": 1.01,
                "rebalance_count": 9,
                "status": "SUCCESS",
            },
            {
                "benchmark": "kneighbor",
                "algo": "hwtopolb",
                "seed": 42,
                "repetition": 2,
                "stage": "matrix",
                "wall_s": 0.24,
                "step_sum_s": 0.17,
                "unaccounted_s": 0.07,
                "rebalance_count": 9,
                "status": "SUCCESS",
            },
        ]

        paired = asm.compute_paired_analysis(runs)
        self.assertEqual(len(paired), 2)
        # Rep 1 ratio: 3.74 / 0.22 = 17.0
        self.assertAlmostEqual(paired[0]["ratio"], 3.74 / 0.22, places=2)
        # Rep 1 speedup %: (3.74 - 0.22) / 3.74 * 100 = 94.117%
        self.assertAlmostEqual(paired[0]["speedup_pct"], ((3.74 - 0.22) / 3.74) * 100, places=2)

        stats = asm.compute_summary_statistics(runs)
        self.assertEqual(len(stats), 2)
        erad_stat = [s for s in stats if s["algo"] == "erad"][0]
        self.assertEqual(erad_stat["repetition_count"], 2)
        self.assertAlmostEqual(erad_stat["median_wall_s"], 3.75)
        self.assertAlmostEqual(erad_stat["mean_wall_s"], 3.75)
        self.assertAlmostEqual(erad_stat["min_wall_s"], 3.74)
        self.assertAlmostEqual(erad_stat["max_wall_s"], 3.76)
        self.assertGreater(erad_stat["stddev_wall_s"], 0.0)

    def test_artifact_generation(self):
        """Generates super_matrix_summary.csv, super_matrix_paired.csv, and SUPER_JOB_REPORT.md."""
        runs = [
            {
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 42,
                "repetition": 1,
                "stage": "matrix",
                "wall_s": 3.73996,
                "step_sum_s": 2.74104,
                "unaccounted_s": 0.99892,
                "rebalance_count": 9,
                "status": "SUCCESS",
            },
            {
                "benchmark": "kneighbor",
                "algo": "hwtopolb",
                "seed": 42,
                "repetition": 1,
                "stage": "matrix",
                "wall_s": 0.220817,
                "step_sum_s": 0.16200,
                "unaccounted_s": 0.058817,
                "rebalance_count": 9,
                "status": "SUCCESS",
            },
            {
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 123,
                "repetition": 1,
                "stage": "matrix",
                "wall_s": 3.57973,
                "step_sum_s": 2.58100,
                "unaccounted_s": 0.99873,
                "rebalance_count": 9,
                "status": "SUCCESS",
            },
            {
                "benchmark": "kneighbor",
                "algo": "hwtopolb",
                "seed": 123,
                "repetition": 1,
                "stage": "matrix",
                "wall_s": 3.81972,
                "step_sum_s": 2.80100,
                "unaccounted_s": 1.01872,
                "rebalance_count": 9,
                "status": "SUCCESS",
            },
        ]

        # Write dummy telemetry log
        telemetry_file = self.output_dir / "hardware_telemetry.log"
        telemetry_file.write_text(
            "timestamp, name, temperature.gpu, utilization.gpu [%], utilization.memory [%], power.draw [W], clocks.current.sm [MHz]\n"
            "2026/10/20 22:00:00.000, NVIDIA A100 80GB PCIe, 38, 0 %, 0 %, 64.20 W, 1410 MHz\n",
            encoding="utf-8"
        )

        asm.generate_artifacts(self.output_dir, runs)

        # Check super_matrix_summary.csv
        summary_csv = self.output_dir / "super_matrix_summary.csv"
        self.assertTrue(summary_csv.exists(), "super_matrix_summary.csv not created")
        with open(summary_csv, newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        self.assertEqual(len(reader), 4)
        self.assertIn("unaccounted_s", reader[0])
        self.assertIn("status", reader[0])

        # Check super_matrix_paired.csv
        paired_csv = self.output_dir / "super_matrix_paired.csv"
        self.assertTrue(paired_csv.exists(), "super_matrix_paired.csv not created")
        with open(paired_csv, newline="", encoding="utf-8") as f:
            paired_reader = list(csv.DictReader(f))
        self.assertEqual(len(paired_reader), 2)
        self.assertIn("ratio", paired_reader[0])
        self.assertIn("speedup_pct", paired_reader[0])

        # Check hardware_telemetry.json
        telemetry_json = self.output_dir / "hardware_telemetry.json"
        self.assertTrue(telemetry_json.exists(), "hardware_telemetry.json not created")
        with open(telemetry_json, encoding="utf-8") as f:
            tel_data = json.load(f)
        self.assertEqual(len(tel_data), 1)

        # Check SUPER_JOB_REPORT.md
        report_md = self.output_dir / "SUPER_JOB_REPORT.md"
        self.assertTrue(report_md.exists(), "SUPER_JOB_REPORT.md not created")
        report_text = report_md.read_text(encoding="utf-8")
        self.assertIn("SUPER JOB REPORT", report_text.upper())
        self.assertIn("Environment and Execution Summary", report_text)
        self.assertIn("Paired Comparison", report_text)
        self.assertIn("Seed 123", report_text)

    def test_defensive_handling_failed_and_partial_runs(self):
        """Gracefully handles truncated CSVs, missing logs, and non-zero exit codes."""
        # Create a partial CSV
        partial_csv = self.output_dir / "kneighbor_erad_42.csv"
        with open(partial_csv, "w", encoding="utf-8") as f:
            f.write("iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n")
            f.write("0,0.02014,compute,0,0,gpu_opencl\n")

        # Create a broken log without total_time_s
        broken_log = self.output_dir / "kneighbor_erad_42.log"
        broken_log.write_text("[kneighbor] crashed before finishing\n", encoding="utf-8")

        run = asm.parse_run(broken_log, partial_csv, expected_iterations=100)
        self.assertEqual(run["status"], "FAILED")
        self.assertIsNone(run["wall_s"])

        # Ensure artifact generation handles FAILED runs without crashing
        asm.generate_artifacts(self.output_dir, [run])
        summary_csv = self.output_dir / "super_matrix_summary.csv"
        self.assertTrue(summary_csv.exists())
        with open(summary_csv, newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        self.assertEqual(reader[0]["status"], "FAILED")

    def test_cli_summarize_only_idempotent(self):
        """CLI with --summarize-only runs cleanly and is idempotent across repeated executions."""
        # Setup small mock output dir
        log_content = (
            "[device] rank=0 global_index=0 name=NVIDIA A100 80GB PCIe\n"
            "[kneighbor] algo=erad mode=gpu_opencl total_time_s=3.74 num_rebalances=9 avg_rebalance_gain=6.34e-05\n"
        )
        csv_content = (
            "iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n"
            "0,0.02,compute,0,0,gpu_opencl\n"
            "1,0.02,compute,0,0,gpu_opencl\n"
        )
        (self.output_dir / "kneighbor_erad_42.log").write_text(log_content, encoding="utf-8")
        (self.output_dir / "kneighbor_erad_42.csv").write_text(csv_content, encoding="utf-8")

        script = BENCHMARKS_DIR / "analyze_super_matrix.py"
        cmd = [sys.executable, str(script), "--output-dir", str(self.output_dir), "--summarize-only"]

        # First execution
        proc1 = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(proc1.returncode, 0, f"Run 1 failed: {proc1.stderr}")

        summary_1 = (self.output_dir / "super_matrix_summary.csv").read_text(encoding="utf-8")

        # Second execution (idempotent test)
        proc2 = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(proc2.returncode, 0, f"Run 2 failed: {proc2.stderr}")

        summary_2 = (self.output_dir / "super_matrix_summary.csv").read_text(encoding="utf-8")
        self.assertEqual(summary_1, summary_2)

    def test_historical_dataset_reconciliation(self):
        """Copies all 12 historical cluster files to a temp dir and checks exact ratio reconciliation."""
        historical_files = list(BENCHMARKS_DIR.glob("kneighbor_*.csv")) + \
                           list(BENCHMARKS_DIR.glob("kneighbor_*.log")) + \
                           list(BENCHMARKS_DIR.glob("leanmd_*.csv")) + \
                           list(BENCHMARKS_DIR.glob("leanmd_*.log"))

        if len(historical_files) < 24:
            self.skipTest("Not all 24 historical run files present in benchmarks/")

        for src in historical_files:
            shutil.copy(src, self.output_dir / src.name)

        script = BENCHMARKS_DIR / "analyze_super_matrix.py"
        cmd = [sys.executable, str(script), "--output-dir", str(self.output_dir), "--summarize-only"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(proc.returncode, 0, f"Script failed: {proc.stderr}")

        # Validate super_matrix_paired.csv against preliminary report values
        paired_csv = self.output_dir / "super_matrix_paired.csv"
        self.assertTrue(paired_csv.exists())
        with open(paired_csv, newline="", encoding="utf-8") as f:
            pairs = list(csv.DictReader(f))

        self.assertEqual(len(pairs), 6)  # 2 workloads * 3 seeds
        for p in pairs:
            bench = p["benchmark"]
            seed = int(p["seed"])
            ratio = float(p["ratio"])
            speedup = float(p["speedup_pct"])

            expected = asm.HISTORICAL_PRELIMINARY[(bench, seed)]
            self.assertAlmostEqual(ratio, expected["ratio"], places=1)
            self.assertAlmostEqual(speedup, expected["speedup_pct"], places=1)

    def test_missing_telemetry_file_handled_gracefully(self):
        """When hardware_telemetry.log is missing, analysis succeeds and hardware_telemetry.json is omitted."""
        run = {
            "benchmark": "kneighbor",
            "algo": "erad",
            "seed": 42,
            "repetition": 1,
            "stage": "matrix",
            "wall_s": 3.74,
            "step_sum_s": 2.74,
            "unaccounted_s": 1.00,
            "rebalance_count": 9,
            "status": "SUCCESS",
        }
        asm.generate_artifacts(self.output_dir, [run])
        self.assertTrue((self.output_dir / "super_matrix_summary.csv").exists())
        self.assertFalse((self.output_dir / "hardware_telemetry.json").exists())

    def test_cpu_fallback_detection(self):
        """Detects CPU fallback when execution_mode is not gpu_opencl."""
        csv_file = self.output_dir / "kneighbor_erad_42.csv"
        log_file = self.output_dir / "kneighbor_erad_42.log"

        csv_file.write_text(
            "iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n"
            "0,0.02,compute,0,0,cpu_reference\n",
            encoding="utf-8",
        )
        log_file.write_text(
            "[kneighbor] algo=erad mode=cpu_reference total_time_s=10.0 num_rebalances=0 avg_rebalance_gain=0\n",
            encoding="utf-8",
        )
        run = asm.parse_run(log_file, csv_file, expected_iterations=1)
        self.assertEqual(run["status"], "CPU_FALLBACK")


if __name__ == "__main__":
    unittest.main()
