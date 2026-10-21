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
            "timestamp,name,temperature.gpu,utilization.gpu,clocks.current.graphics,clocks.current.memory,power.draw,clocks_throttle_reasons.active\n"
            "2026/10/20 22:00:00.000, NVIDIA A100 80GB PCIe, 38, 0, 1410, 1215, 64.20, [N/A]\n"
            "2026/10/20 22:00:05.000, NVIDIA A100 80GB PCIe, 42, 98, 1410, 1215, 210.50, [N/A]\n",
            encoding="utf-8"
        )
        records = asm.parse_hardware_telemetry(telemetry_file)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["temperature_gpu"], 38)
        self.assertEqual(records[0]["utilization_gpu_pct"], 0)
        self.assertEqual(records[0]["clocks_graphics_mhz"], 1410)
        self.assertEqual(records[0]["clocks_memory_mhz"], 1215)
        self.assertAlmostEqual(records[0]["power_draw_w"], 64.20)
        self.assertAlmostEqual(records[1]["power_draw_w"], 210.50)
        self.assertEqual(records[1]["utilization_gpu_pct"], 98)

    def test_parse_multi_node_hardware_telemetry(self):
        """Parses multi-node hardware_telemetry.log tagged with host and rank."""
        telemetry_file = self.output_dir / "hardware_telemetry.log"
        telemetry_file.write_text(
            "host, rank, timestamp, name, temperature.gpu, utilization.gpu, clocks.current.graphics, clocks.current.memory, power.draw, clocks_throttle_reasons.active\n"
            "compute-1-0, 0, 2026/10/20 22:00:00.000, NVIDIA A100 80GB PCIe, 38, 12, 1410, 1215, 64.20, [N/A]\n"
            "compute-1-1, 1, 2026/10/20 22:00:00.000, NVIDIA A100 80GB PCIe, 42, 18, 1410, 1215, 78.50, [N/A]\n",
            encoding="utf-8"
        )
        records = asm.parse_hardware_telemetry(telemetry_file)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["host"], "compute-1-0")
        self.assertEqual(records[0]["rank"], 0)
        self.assertEqual(records[0]["temperature_gpu"], 38)
        self.assertEqual(records[0]["utilization_gpu_pct"], 12)
        self.assertEqual(records[0]["clocks_graphics_mhz"], 1410)
        self.assertEqual(records[0]["clocks_memory_mhz"], 1215)
        self.assertAlmostEqual(records[0]["power_draw_w"], 64.20)

        self.assertEqual(records[1]["host"], "compute-1-1")
        self.assertEqual(records[1]["rank"], 1)
        self.assertEqual(records[1]["temperature_gpu"], 42)
        self.assertEqual(records[1]["utilization_gpu_pct"], 18)
        self.assertEqual(records[1]["clocks_graphics_mhz"], 1410)
        self.assertEqual(records[1]["clocks_memory_mhz"], 1215)
        self.assertAlmostEqual(records[1]["power_draw_w"], 78.50)

    def test_multi_node_telemetry_in_json_and_report(self):
        """Preserves host and rank in hardware_telemetry.json and generates per-node report summaries."""
        telemetry_file = self.output_dir / "hardware_telemetry.log"
        telemetry_file.write_text(
            "compute-1-0, 0, 2026/10/20 22:00:00.000, NVIDIA A100 80GB PCIe, 38, 12, 1410, 1215, 64.20, [N/A]\n"
            "compute-1-1, 1, 2026/10/20 22:00:00.000, NVIDIA A100 80GB PCIe, 42, 18, 1410, 1215, 78.50, [N/A]\n",
            encoding="utf-8"
        )
        runs = [
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
            }
        ]
        asm.generate_artifacts(self.output_dir, runs)

        # 1. Check hardware_telemetry.json
        telemetry_json = self.output_dir / "hardware_telemetry.json"
        self.assertTrue(telemetry_json.exists())
        with open(telemetry_json, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["host"], "compute-1-0")
        self.assertEqual(data[0]["rank"], 0)
        self.assertEqual(data[1]["host"], "compute-1-1")
        self.assertEqual(data[1]["rank"], 1)

        # 2. Check SUPER_JOB_REPORT.md
        report_md = self.output_dir / "SUPER_JOB_REPORT.md"
        self.assertTrue(report_md.exists())
        content = report_md.read_text(encoding="utf-8")
        self.assertIn("compute-1-0", content)
        self.assertIn("compute-1-1", content)
        self.assertIn("Hardware Telemetry by Node & Rank", content)
        self.assertIn("Inter-Node Thermal Stability", content)

    def test_unavailable_telemetry_marker_is_reported_not_dropped(self):
        """An UNAVAILABLE marker from the runner yields an explicit unavailable record and report row."""
        telemetry_file = self.output_dir / "hardware_telemetry.log"
        telemetry_file.write_text(
            "# [telemetry] block=before_warmup timestamp=2026-10-20 22:00:00\n"
            "compute-1-0, 0, 2026/10/20 22:00:00.000, NVIDIA A100 80GB PCIe, 38, 12, 1410, 1215, 64.20, [N/A]\n"
            "# [telemetry-unavailable] host=compute-1-1 rank=1 block=before_warmup reason=no ssh-free path to remote node\n",
            encoding="utf-8",
        )
        records = asm.parse_hardware_telemetry(telemetry_file)
        unavailable = [r for r in records if r.get("host") == "compute-1-1"]
        self.assertEqual(len(unavailable), 1)
        self.assertEqual(unavailable[0]["rank"], 1)
        self.assertFalse(unavailable[0]["available"])
        self.assertIn("no ssh-free path", unavailable[0]["unavailable_reason"])
        self.assertTrue([r for r in records if r["host"] == "compute-1-0"][0]["available"])

        runs = [{
            "benchmark": "kneighbor", "algo": "erad", "seed": 42, "repetition": 1, "stage": "matrix",
            "wall_s": 3.74, "step_sum_s": 2.74, "unaccounted_s": 1.0, "rebalance_count": 9, "status": "SUCCESS",
        }]
        asm.generate_artifacts(self.output_dir, runs)
        with open(self.output_dir / "hardware_telemetry.json", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("compute-1-1", {r["host"] for r in data})
        content = (self.output_dir / "SUPER_JOB_REPORT.md").read_text(encoding="utf-8")
        row = next(l for l in content.splitlines() if l.startswith("| compute-1-1 |"))
        self.assertIn("UNAVAILABLE", row)
        # Unavailable samples must not contaminate aggregate statistics (zero temperature etc.).
        self.assertNotIn("| GPU Temperature (°C) | 0", content)

    def test_telemetry_parsing_creates_hardware_telemetry_json(self):
        """Hardware telemetry parsing creates hardware_telemetry.json with distinct graphics/memory clocks and power."""
        telemetry_file = self.output_dir / "hardware_telemetry.log"
        telemetry_file.write_text(
            "timestamp,name,temperature.gpu,utilization.gpu,clocks.current.graphics,clocks.current.memory,power.draw,clocks_throttle_reasons.active\n"
            "2026/10/20 22:00:00.000, NVIDIA A100-PCIE-40GB, 45, 85, 1410, 1215, 250.5, 0x0000000000000000\n",
            encoding="utf-8"
        )
        runs = [
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
            }
        ]
        asm.generate_artifacts(self.output_dir, runs)
        telemetry_json = self.output_dir / "hardware_telemetry.json"
        self.assertTrue(telemetry_json.exists(), "hardware_telemetry.json was not created")
        with open(telemetry_json, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        entry = data[0]
        self.assertEqual(entry["timestamp"], "2026/10/20 22:00:00.000")
        self.assertEqual(entry["name"], "NVIDIA A100-PCIE-40GB")
        self.assertEqual(entry["temperature_gpu"], 45)
        self.assertEqual(entry["utilization_gpu_pct"], 85)
        self.assertEqual(entry["clocks_graphics_mhz"], 1410)
        self.assertEqual(entry["clocks_memory_mhz"], 1215)
        self.assertAlmostEqual(entry["power_draw_w"], 250.5, places=4)
        self.assertEqual(entry["throttle_reasons"], "0x0000000000000000")

        report_md = (self.output_dir / "SUPER_JOB_REPORT.md").read_text(encoding="utf-8")
        self.assertIn("Graphics Clock (MHz)", report_md)
        self.assertIn("Memory Clock (MHz)", report_md)
        self.assertIn("Power Draw (W)", report_md)

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
            "timestamp,name,temperature.gpu,utilization.gpu,clocks.current.graphics,clocks.current.memory,power.draw,clocks_throttle_reasons.active\n"
            "2026/10/20 22:00:00.000, NVIDIA A100 80GB PCIe, 38, 0, 1410, 1215, 64.20, [N/A]\n",
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
        self.assertIn("T_compute_s", reader[0])
        self.assertIn("T_comm_s", reader[0])
        self.assertIn("T_balance_s", reader[0])
        self.assertIn("boundary_shifts", reader[0])

        # Check super_matrix_paired.csv
        paired_csv = self.output_dir / "super_matrix_paired.csv"
        self.assertTrue(paired_csv.exists(), "super_matrix_paired.csv not created")
        with open(paired_csv, newline="", encoding="utf-8") as f:
            paired_reader = list(csv.DictReader(f))
        self.assertEqual(len(paired_reader), 2)
        self.assertIn("ratio", paired_reader[0])
        self.assertIn("speedup_pct", paired_reader[0])
        self.assertIn("step", paired_reader[0])

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
        self.assertIn("### Observed", report_text)
        self.assertIn("### Inferred", report_text)
        self.assertIn("### Unverified", report_text)

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

    def test_manifest_naming_and_algo_key_alignment(self):
        """Discovers run_manifest.json as well as manifest.json, supporting 'algo' and 'algorithm'."""
        # Create run_manifest.json
        manifest_data = {
            "parameters": {"n": 64, "iterations": 2, "balance_interval": 1, "seeds": [42]},
            "runs": [
                {
                    "run_id": 1,
                    "stage": "matrix",
                    "repetition": 1,
                    "benchmark": "kneighbor",
                    "algorithm": "erad",
                    "seed": 42,
                    "step": None,
                    "csv_file": "run1.csv",
                    "log_file": "run1.log",
                    "exit_code": 0,
                    "elapsed_seconds": 1.23,
                    "status": "success",
                },
                {
                    "run_id": 2,
                    "stage": "matrix",
                    "repetition": 1,
                    "benchmark": "kneighbor",
                    "algo": "hwtopolb",
                    "seed": 42,
                    "step": None,
                    "csv_file": "run2.csv",
                    "log_file": "run2.log",
                    "exit_code": 0,
                    "elapsed_seconds": 0.23,
                    "status": "success",
                },
            ],
        }
        (self.output_dir / "run_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

        # Create mock run files
        for i, algo in enumerate(["erad", "hwtopolb"], 1):
            (self.output_dir / f"run{i}.log").write_text(
                f"[kneighbor] algo={algo} mode=gpu_opencl total_time_s=1.{i} num_rebalances=0 avg_rebalance_gain=0\n",
                encoding="utf-8",
            )
            (self.output_dir / f"run{i}.csv").write_text(
                "iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n0,0.01,compute,0,0,gpu_opencl\n1,0.01,compute,0,0,gpu_opencl\n",
                encoding="utf-8",
            )

        runs = asm.discover_runs(self.output_dir, expected_iterations=2)
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]["algo"], "erad")
        self.assertEqual(runs[1]["algo"], "hwtopolb")

    def test_statistics_isolation_matrix_only(self):
        """5-repetition statistics calculated strictly from stage == 'matrix' runs."""
        runs = []
        # Warmup runs (e.g. wall time 100.0)
        for i in range(4):
            runs.append({
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 42,
                "repetition": None,
                "stage": "warmup",
                "wall_s": 100.0,
                "step_sum_s": 10.0,
                "unaccounted_s": 90.0,
                "rebalance_count": 0,
                "status": "SUCCESS",
            })

        # Matrix runs (5 repetitions: wall times 3.50, 3.60, 3.70, 3.80, 3.90 -> median 3.70)
        for rep, wall in enumerate([3.50, 3.60, 3.70, 3.80, 3.90], 1):
            runs.append({
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 42,
                "repetition": rep,
                "stage": "matrix",
                "wall_s": wall,
                "step_sum_s": wall - 0.5,
                "unaccounted_s": 0.5,
                "rebalance_count": 9,
                "status": "SUCCESS",
            })

        # Dedicated ABA runs (e.g. wall time 999.0)
        for step in [1, 3]:
            runs.append({
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 42,
                "repetition": step,
                "step": step,
                "stage": "aba",
                "wall_s": 999.0,
                "step_sum_s": 900.0,
                "unaccounted_s": 99.0,
                "rebalance_count": 9,
                "status": "SUCCESS",
            })

        # By default compute_summary_statistics should strictly isolate stage == 'matrix'
        stats = asm.compute_summary_statistics(runs)
        self.assertEqual(len(stats), 1)
        stat_entry = stats[0]
        self.assertEqual(stat_entry["repetition_count"], 5)
        self.assertAlmostEqual(stat_entry["median_wall_s"], 3.70)
        self.assertAlmostEqual(stat_entry["mean_wall_s"], 3.70)
        self.assertAlmostEqual(stat_entry["min_wall_s"], 3.50)
        self.assertAlmostEqual(stat_entry["max_wall_s"], 3.90)

        # Breakdowns provide separate tables for warmup and aba
        breakdowns = asm.compute_stage_breakdowns(runs)
        self.assertEqual(len(breakdowns["matrix"]), 1)
        self.assertEqual(breakdowns["matrix"][0]["repetition_count"], 5)
        self.assertEqual(len(breakdowns["warmup"]), 1)
        self.assertEqual(breakdowns["warmup"][0]["repetition_count"], 4)
        self.assertAlmostEqual(breakdowns["warmup"][0]["median_wall_s"], 100.0)
        self.assertEqual(len(breakdowns["aba"]), 1)
        self.assertEqual(breakdowns["aba"][0]["repetition_count"], 2)
        self.assertAlmostEqual(breakdowns["aba"][0]["median_wall_s"], 999.0)

    def test_aba_thermal_drift_and_pairing_no_overwrite(self):
        """ERAD step 1 is not overwritten by ERAD step 3, and dedicated drift metrics are computed."""
        runs = [
            # Step 1: ERAD
            {
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 123,
                "repetition": 1,
                "step": 1,
                "stage": "aba",
                "wall_s": 3.50,
                "step_sum_s": 2.50,
                "unaccounted_s": 1.00,
                "rebalance_count": 9,
                "status": "SUCCESS",
            },
            # Step 2: HWTOPOLB
            {
                "benchmark": "kneighbor",
                "algo": "hwtopolb",
                "seed": 123,
                "repetition": 2,
                "step": 2,
                "stage": "aba",
                "wall_s": 0.25,
                "step_sum_s": 0.18,
                "unaccounted_s": 0.07,
                "rebalance_count": 9,
                "status": "SUCCESS",
            },
            # Step 3: ERAD
            {
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 123,
                "repetition": 3,
                "step": 3,
                "stage": "aba",
                "wall_s": 3.60,
                "step_sum_s": 2.55,
                "unaccounted_s": 1.05,
                "rebalance_count": 9,
                "status": "SUCCESS",
            },
        ]

        pairs = asm.compute_paired_analysis(runs)
        self.assertEqual(len(pairs), 2, "Expected 2 pairs for ABA: step 1 vs 2 and step 3 vs 2")

        p_step1 = [p for p in pairs if p.get("step") == 1][0]
        p_step3 = [p for p in pairs if p.get("step") == 3][0]

        # Step 1 must have wall_s 3.50, not overwritten
        self.assertAlmostEqual(p_step1["erad_wall_s"], 3.50)
        self.assertAlmostEqual(p_step1["ratio"], 3.50 / 0.25)
        # Step 3 must have wall_s 3.60
        self.assertAlmostEqual(p_step3["erad_wall_s"], 3.60)
        self.assertAlmostEqual(p_step3["ratio"], 3.60 / 0.25)

        # Drift metrics
        drift = asm.compute_aba_drift_analysis(runs)
        self.assertEqual(len(drift), 1)
        d = drift[0]
        self.assertEqual(d["benchmark"], "kneighbor")
        self.assertEqual(d["seed"], 123)
        self.assertAlmostEqual(d["drift_delta_s"], 0.10)  # 3.60 - 3.50
        self.assertAlmostEqual(d["drift_pct"], (0.10 / 3.50) * 100.0, places=2)
        self.assertAlmostEqual(d["bracketed_erad_mean_s"], 3.55)  # (3.50 + 3.60) / 2
        self.assertAlmostEqual(d["bracketed_ratio"], 3.55 / 0.25)  # 14.2x
        self.assertAlmostEqual(d["bracketed_speedup_pct"], ((3.55 - 0.25) / 3.55) * 100.0, places=2)

    def test_balance_metrics_and_boundary_shifts_extraction(self):
        """Extracts T_compute_s, T_comm_s, T_balance_s and boundary_shifts from log files."""
        log_content = """
[DCL][balance-metrics] iteration=1
  policy=off
  action=skip
  reason=balance disabled
  T_compute_interval_s=0.01
  T_comm_interval_s=0.002
  T_balance_total_s=0.001
  T_rebalance_apply_s=0
  T_compute_accum_s=0.01
  T_comm_accum_s=0.002
  T_balance_accum_s=0.001
  T_rebalance_apply_accum_s=0
  OLD loads:
    part[0] cum=50% share=50% device_global=0 rank=0 local_index=0 offset=0 count=500000
    part[1] cum=100% share=50% device_global=1 rank=1 local_index=-1 offset=500000 count=500000
    total=100%
  CURRENT loads:
    part[0] cum=40% share=40% device_global=0 rank=0 local_index=0 offset=0 count=400000
    part[1] cum=100% share=60% device_global=1 rank=1 local_index=-1 offset=400000 count=600000
    total=100%
[kneighbor] algo=erad mode=gpu_opencl total_time_s=1.0 num_rebalances=1 avg_rebalance_gain=0.001
"""
        csv_content = "iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n0,0.01,compute,0,0,gpu_opencl\n"

        log_path = self.output_dir / "kneighbor_erad_42.log"
        csv_path = self.output_dir / "kneighbor_erad_42.csv"
        log_path.write_text(log_content, encoding="utf-8")
        csv_path.write_text(csv_content, encoding="utf-8")

        run = asm.parse_run(log_path, csv_path, expected_iterations=1)
        self.assertAlmostEqual(run["T_compute_s"], 0.01)
        self.assertAlmostEqual(run["T_comm_s"], 0.002)
        self.assertAlmostEqual(run["T_balance_s"], 0.001)
        self.assertEqual(run["boundary_shifts"], 100000)

    def test_manifest_failure_overrides_csv_success_and_excludes_from_paired(self):
        """
        Manifest failure or timeout status overrides CSV/log success and excludes run from pairing.
        Verifies Ticket #22 behavior.
        """
        erad_log = self.output_dir / "kneighbor_erad_42.log"
        erad_csv = self.output_dir / "kneighbor_erad_42.csv"
        hw_log = self.output_dir / "kneighbor_hwtopolb_42.log"
        hw_csv = self.output_dir / "kneighbor_hwtopolb_42.csv"

        erad_log.write_text(
            "[kneighbor] algo=erad mode=gpu_opencl total_time_s=1.5 num_rebalances=0 avg_rebalance_gain=0\n",
            encoding="utf-8",
        )
        erad_csv.write_text(
            "iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n0,0.01,compute,0,0,gpu_opencl\n",
            encoding="utf-8",
        )
        hw_log.write_text(
            "[kneighbor] algo=hwtopolb mode=gpu_opencl total_time_s=0.5 num_rebalances=0 avg_rebalance_gain=0\n",
            encoding="utf-8",
        )
        hw_csv.write_text(
            "iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n0,0.01,compute,0,0,gpu_opencl\n",
            encoding="utf-8",
        )

        # Baseline: with valid files and without manifest failure, status is SUCCESS
        run_hw = asm.parse_run(hw_log, hw_csv, expected_iterations=1)
        self.assertEqual(run_hw["status"], "SUCCESS")

        # Scenario 1: manifest_run specifies status: "failed", exit_code: 1
        manifest_failed = {"status": "failed", "exit_code": 1}
        run_erad_failed = asm.parse_run(
            erad_log, erad_csv, expected_iterations=1, manifest_run=manifest_failed
        )
        self.assertEqual(run_erad_failed["status"], "FAILED")

        # Scenario 2: manifest_run specifies status: "timeout", exit_code: 124
        manifest_timeout = {"status": "timeout", "exit_code": 124}
        run_erad_timeout = asm.parse_run(
            erad_log, erad_csv, expected_iterations=1, manifest_run=manifest_timeout
        )
        self.assertEqual(run_erad_timeout["status"], "TIMEOUT")

        # Scenario 3: manifest_run specifies exit_code != 0 (e.g. exit_code=2)
        manifest_exit2 = {"exit_code": 2, "status": "success"}
        run_erad_exit2 = asm.parse_run(
            erad_log, erad_csv, expected_iterations=1, manifest_run=manifest_exit2
        )
        self.assertEqual(run_erad_exit2["status"], "FAILED")

        # Scenario 4: passed via metadata parameter (backward compatibility)
        run_erad_meta = asm.parse_run(
            erad_log, erad_csv, expected_iterations=1, metadata=manifest_failed
        )
        self.assertEqual(run_erad_meta["status"], "FAILED")

        # Verify paired analysis excludes failed or timed-out runs
        pairs_failed = asm.compute_paired_analysis([run_erad_failed, run_hw])
        self.assertEqual(len(pairs_failed), 0, "Failed run must not be paired")

        pairs_timeout = asm.compute_paired_analysis([run_erad_timeout, run_hw])
        self.assertEqual(len(pairs_timeout), 0, "Timed-out run must not be paired")

        # Verify that normal SUCCESS runs pair as expected
        run_erad_success = asm.parse_run(
            erad_log, erad_csv, expected_iterations=1, manifest_run={"status": "success", "exit_code": 0}
        )
        self.assertEqual(run_erad_success["status"], "SUCCESS")
        pairs_success = asm.compute_paired_analysis([run_erad_success, run_hw])
        self.assertEqual(len(pairs_success), 1, "Success runs should be paired")

    def test_seed_123_verdict_filters_matrix_stage_exclusively(self):
        """
        Diagnostic verdict for Seed 123 in Section 6 must strictly evaluate pairs from
        the rotated matrix stage ('matrix'), excluding warmup and dedicated ABA stages.
        Verifies Ticket #24 behavior.
        """
        runs = [
            # Warmup runs for seed 123 (ratio: 0.5x, ERAD faster)
            {
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 123,
                "repetition": 1,
                "stage": "warmup",
                "step": None,
                "wall_s": 0.5,
                "step_sum_s": 0.4,
                "unaccounted_s": 0.1,
                "rebalance_count": 0,
                "status": "SUCCESS",
            },
            {
                "benchmark": "kneighbor",
                "algo": "hwtopolb",
                "seed": 123,
                "repetition": 1,
                "stage": "warmup",
                "step": None,
                "wall_s": 1.0,
                "step_sum_s": 0.8,
                "unaccounted_s": 0.2,
                "rebalance_count": 0,
                "status": "SUCCESS",
            },
            # Dedicated A/B/A step for seed 123 (ratio: 0.7x, ERAD faster)
            # Step 1: ERAD
            {
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 123,
                "repetition": 1,
                "stage": "aba",
                "step": 1,
                "wall_s": 0.7,
                "step_sum_s": 0.5,
                "unaccounted_s": 0.2,
                "rebalance_count": 0,
                "status": "SUCCESS",
            },
            # Step 2: HWTOPOLB
            {
                "benchmark": "kneighbor",
                "algo": "hwtopolb",
                "seed": 123,
                "repetition": 2,
                "stage": "aba",
                "step": 2,
                "wall_s": 1.0,
                "step_sum_s": 0.8,
                "unaccounted_s": 0.2,
                "rebalance_count": 0,
                "status": "SUCCESS",
            },
        ]

        # 5 Rotated matrix repetitions for seed 123 (ratios >= 10.0x, HWTOPOLB faster)
        for rep in range(1, 6):
            runs.append({
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 123,
                "repetition": rep,
                "stage": "matrix",
                "step": None,
                "wall_s": 10.0,
                "step_sum_s": 8.0,
                "unaccounted_s": 2.0,
                "rebalance_count": 5,
                "status": "SUCCESS",
            })
            runs.append({
                "benchmark": "kneighbor",
                "algo": "hwtopolb",
                "seed": 123,
                "repetition": rep,
                "stage": "matrix",
                "step": None,
                "wall_s": 1.0,
                "step_sum_s": 0.8,
                "unaccounted_s": 0.2,
                "rebalance_count": 5,
                "status": "SUCCESS",
            })

        asm.generate_artifacts(self.output_dir, runs)
        report_md = self.output_dir / "SUPER_JOB_REPORT.md"
        self.assertTrue(report_md.exists(), "SUPER_JOB_REPORT.md should be created")
        report_text = report_md.read_text(encoding="utf-8")

        # Verify that only the 5 rotated matrix pairs are evaluated for seed 123 verdict
        self.assertIn("- **Evaluated Seed 123 Rotated Matrix Pairs:** 5", report_text)
        self.assertNotIn("- **Evaluated Seed 123 Paired Runs:**", report_text)
        self.assertIn("Verdict: Observed HWTOPOLB Advantage Across Rotated Matrix Repetitions", report_text)
        self.assertIn("rotated matrix repetitions", report_text)
        self.assertIn("Descriptive Scope:", report_text)

    def test_no_fallback_to_other_stages_when_matrix_runs_fail(self):
        """
        When all matrix runs fail and only warmup or aba runs exist,
        compute_matrix_statistics(runs, stage='matrix') must NOT fall back to including
        warmup or aba runs in the statistics.
        Verifies Ticket #26 acceptance criteria.
        """
        runs = [
            # Failed/timed-out matrix runs
            {
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 42,
                "repetition": 1,
                "stage": "matrix",
                "wall_s": None,
                "step_sum_s": None,
                "unaccounted_s": None,
                "rebalance_count": 0,
                "status": "FAILED",
            },
            {
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 42,
                "repetition": 2,
                "stage": "matrix",
                "wall_s": 3.5,
                "step_sum_s": 2.5,
                "unaccounted_s": 1.0,
                "rebalance_count": 0,
                "status": "TIMEOUT",
            },
            # Successful warmup runs
            {
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 42,
                "repetition": None,
                "stage": "warmup",
                "wall_s": 100.0,
                "step_sum_s": 10.0,
                "unaccounted_s": 90.0,
                "rebalance_count": 0,
                "status": "SUCCESS",
            },
            # Successful ABA runs
            {
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 42,
                "repetition": 1,
                "step": 1,
                "stage": "aba",
                "wall_s": 999.0,
                "step_sum_s": 900.0,
                "unaccounted_s": 99.0,
                "rebalance_count": 9,
                "status": "SUCCESS",
            },
        ]

        # compute_matrix_statistics must not fall back to other stages
        stats = asm.compute_matrix_statistics(runs, stage="matrix")
        self.assertEqual(len(stats), 0, "compute_matrix_statistics must not include non-matrix stages when matrix runs fail")

        # Also test with compute_summary_statistics
        summary_stats = asm.compute_summary_statistics(runs, stage="matrix")
        self.assertEqual(len(summary_stats), 0, "compute_summary_statistics must not include non-matrix stages when matrix runs fail")

        # Paired analysis for matrix stage must not include non-matrix stages
        matrix_pairs = asm.compute_paired_analysis(runs, stage="matrix")
        self.assertEqual(len(matrix_pairs), 0, "Paired analysis for matrix stage must not include non-matrix runs")

        # Verify report indicates no successful matrix samples rather than substituting warmup/aba data
        asm.generate_artifacts(self.output_dir, runs)
        report_md = self.output_dir / "SUPER_JOB_REPORT.md"
        self.assertTrue(report_md.exists())
        report_text = report_md.read_text(encoding="utf-8")

        self.assertIn("no successful matrix samples", report_text.lower())
        # The 5-repetition rotated matrix summary table (before sub-breakdowns) must not contain substituted warmup/aba wall times
        matrix_summary_table = report_text.split("## 2.")[1].split("###")[0]
        self.assertNotIn("100.000000", matrix_summary_table)
        self.assertNotIn("999.000000", matrix_summary_table)
        self.assertIn("*(no successful matrix samples)*", matrix_summary_table)

        # Also test with completely empty runs
        empty_dir = Path(tempfile.mkdtemp(prefix="test_empty_"))
        try:
            asm.generate_artifacts(empty_dir, [])
            empty_report = (empty_dir / "SUPER_JOB_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("no successful matrix samples", empty_report.lower())
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)

    def test_parse_log_with_loop_time_s(self):
        """Verifies parsing of loop_time_s from summary line and standalone line (#25)."""
        # Case A: Summary line with both total_time_s and loop_time_s (without duplicate standalone line)
        log_a = (
            "[device] rank=0 global_index=0 name=NVIDIA A100 80GB PCIe\n"
            "[kneighbor] algo=erad mode=gpu_opencl total_time_s=4.120000 loop_time_s=3.250000 num_rebalances=9 avg_rebalance_gain=6.34934e-05\n"
        )
        log_path_a = self.output_dir / "test_a.log"
        log_path_a.write_text(log_a, encoding="utf-8")
        parsed_a = asm.parse_log_file(log_path_a)
        self.assertIsNotNone(parsed_a)
        self.assertAlmostEqual(parsed_a["total_time_s"], 4.12)
        self.assertAlmostEqual(parsed_a["loop_time_s"], 3.25)
        self.assertEqual(parsed_a["algo"], "erad")
        self.assertEqual(parsed_a["benchmark"], "kneighbor")

        # Case B: Standalone loop_time_s line only, backward-compatible summary line
        log_b = (
            "loop_time_s: 0.185000\n"
            "[leanmd] algo=hwtopolb mode=gpu_opencl total_time_s=0.450000 num_rebalances=5 avg_rebalance_gain=0.0001\n"
        )
        log_path_b = self.output_dir / "test_b.log"
        log_path_b.write_text(log_b, encoding="utf-8")
        parsed_b = asm.parse_log_file(log_path_b)
        self.assertIsNotNone(parsed_b)
        self.assertAlmostEqual(parsed_b["total_time_s"], 0.45)
        self.assertAlmostEqual(parsed_b["loop_time_s"], 0.185)

        # Case C: Historical log without loop_time_s
        log_c = (
            "[kneighbor] algo=erad mode=gpu_opencl total_time_s=3.74 num_rebalances=9 avg_rebalance_gain=6.34e-05\n"
        )
        log_path_c = self.output_dir / "test_c.log"
        log_path_c.write_text(log_c, encoding="utf-8")
        parsed_c = asm.parse_log_file(log_path_c)
        self.assertIsNotNone(parsed_c)
        self.assertAlmostEqual(parsed_c["total_time_s"], 3.74)
        self.assertIsNone(parsed_c.get("loop_time_s"))

    def test_paired_analysis_prioritizes_loop_time_s_avoiding_inflation(self):
        """Paired analysis uses loop_time_s instead of total_time_s when present, avoiding verification inflation (#25)."""
        # ERAD: loop = 2.0s, total = 12.0s (10s verification inflation)
        # HWTOPOLB: loop = 0.2s, total = 10.2s (10s verification inflation)
        # With total_time_s, ratio = 12.0 / 10.2 = ~1.176 (distorted)
        # With loop_time_s, ratio = 2.0 / 0.2 = 10.0 (true speedup)
        runs = [
            {
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 42,
                "repetition": 1,
                "stage": "matrix",
                "wall_s": 12.0,
                "loop_s": 2.0,
                "step_sum_s": 1.9,
                "unaccounted_s": 10.1,
                "rebalance_count": 9,
                "status": "SUCCESS",
            },
            {
                "benchmark": "kneighbor",
                "algo": "hwtopolb",
                "seed": 42,
                "repetition": 1,
                "stage": "matrix",
                "wall_s": 10.2,
                "loop_s": 0.2,
                "step_sum_s": 0.18,
                "unaccounted_s": 10.02,
                "rebalance_count": 9,
                "status": "SUCCESS",
            },
        ]
        paired = asm.compute_paired_analysis(runs)
        self.assertEqual(len(paired), 1)
        p = paired[0]
        self.assertAlmostEqual(p["ratio"], 10.0, places=2)
        self.assertAlmostEqual(p["speedup_pct"], 90.0, places=2)
        self.assertAlmostEqual(p["erad_wall_s"], 12.0)
        self.assertAlmostEqual(p["hwtopolb_wall_s"], 10.2)
        self.assertAlmostEqual(p["erad_loop_s"], 2.0)
        self.assertAlmostEqual(p["hwtopolb_loop_s"], 0.2)

    def test_paired_analysis_backward_compatibility_missing_loop_s(self):
        """Paired analysis cleanly falls back to wall_s when loop_s is missing (#25)."""
        runs = [
            {
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 42,
                "repetition": 1,
                "stage": "matrix",
                "wall_s": 3.74,
                "step_sum_s": 2.74,
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
                "status": "SUCCESS",
            },
        ]
        paired = asm.compute_paired_analysis(runs)
        self.assertEqual(len(paired), 1)
        p = paired[0]
        self.assertAlmostEqual(p["ratio"], 3.74 / 0.22, places=2)
        self.assertAlmostEqual(p["speedup_pct"], ((3.74 - 0.22) / 3.74) * 100.0, places=2)
        self.assertIsNone(p.get("erad_loop_s"))
        self.assertIsNone(p.get("hwtopolb_loop_s"))

    def test_summary_statistics_prioritizes_loop_time_s(self):
        """Summary statistics calculates distribution over loop_time_s when present (#25)."""
        runs = [
            {
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 42,
                "repetition": 1,
                "stage": "matrix",
                "wall_s": 12.0,
                "loop_s": 2.0,
                "status": "SUCCESS",
            },
            {
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 42,
                "repetition": 2,
                "stage": "matrix",
                "wall_s": 14.0,
                "loop_s": 2.2,
                "status": "SUCCESS",
            },
        ]
        stats = asm.compute_matrix_statistics(runs, stage="matrix")
        self.assertEqual(len(stats), 1)
        s = stats[0]
        self.assertAlmostEqual(s["median_s"], 2.1)
        self.assertAlmostEqual(s["median_loop_s"], 2.1)
        self.assertAlmostEqual(s["median_wall_s"], 13.0)

    def test_artifact_generation_both_fields(self):
        """Artifact generation outputs both wall_s and loop_s in CSVs and markdown (#25)."""
        runs = [
            {
                "benchmark": "kneighbor",
                "algo": "erad",
                "seed": 42,
                "repetition": 1,
                "stage": "matrix",
                "wall_s": 4.0,
                "loop_s": 3.0,
                "step_sum_s": 2.8,
                "unaccounted_s": 1.2,
                "rebalance_count": 9,
                "status": "SUCCESS",
            },
            {
                "benchmark": "kneighbor",
                "algo": "hwtopolb",
                "seed": 42,
                "repetition": 1,
                "stage": "matrix",
                "wall_s": 0.5,
                "loop_s": 0.2,
                "step_sum_s": 0.18,
                "unaccounted_s": 0.32,
                "rebalance_count": 9,
                "status": "SUCCESS",
            },
        ]
        asm.generate_artifacts(self.output_dir, runs)

        # Check super_matrix_summary.csv has loop_s column
        summary_csv = self.output_dir / "super_matrix_summary.csv"
        self.assertTrue(summary_csv.exists())
        with open(summary_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertIn("wall_s", rows[0])
        self.assertIn("loop_s", rows[0])
        self.assertAlmostEqual(float(rows[0]["wall_s"]), 4.0)
        self.assertAlmostEqual(float(rows[0]["loop_s"]), 3.0)

        # Check super_matrix_paired.csv has both wall and loop columns
        paired_csv = self.output_dir / "super_matrix_paired.csv"
        self.assertTrue(paired_csv.exists())
        with open(paired_csv, newline="", encoding="utf-8") as f:
            prows = list(csv.DictReader(f))
        self.assertIn("erad_wall_s", prows[0])
        self.assertIn("hwtopolb_wall_s", prows[0])
        self.assertIn("erad_loop_s", prows[0])
        self.assertIn("hwtopolb_loop_s", prows[0])
        self.assertAlmostEqual(float(prows[0]["erad_wall_s"]), 4.0)
        self.assertAlmostEqual(float(prows[0]["erad_loop_s"]), 3.0)
        self.assertAlmostEqual(float(prows[0]["ratio"]), 3.0 / 0.2)  # Prioritizes loop_s!

    def test_configurable_iterations_from_manifest_and_metadata(self):
        """Manifest parameters.iterations=5 prevents runs with 5 steps from being marked PARTIAL even when expected_iterations=100."""
        manifest_data = {
            "parameters": {
                "n": 1000,
                "iterations": 5,
                "balance_interval": 2,
                "seeds": [42],
            },
            "runs": [
                {
                    "run_id": 1,
                    "stage": "matrix",
                    "repetition": 1,
                    "benchmark": "kneighbor",
                    "algorithm": "erad",
                    "seed": 42,
                    "step": None,
                    "csv_file": "run_erad.csv",
                    "log_file": "run_erad.log",
                    "exit_code": 0,
                    "elapsed_seconds": 0.5,
                    "status": "success",
                },
                {
                    "run_id": 2,
                    "stage": "matrix",
                    "repetition": 1,
                    "benchmark": "kneighbor",
                    "algorithm": "hwtopolb",
                    "seed": 42,
                    "step": None,
                    "csv_file": "run_hwtopolb.csv",
                    "log_file": "run_hwtopolb.log",
                    "exit_code": 0,
                    "elapsed_seconds": 0.4,
                    "status": "success",
                },
            ],
        }
        (self.output_dir / "run_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

        csv_header = "iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n"
        csv_rows = "".join(f"{i},0.01,compute,0,0,gpu_opencl\n" for i in range(5))
        (self.output_dir / "run_erad.csv").write_text(csv_header + csv_rows, encoding="utf-8")
        (self.output_dir / "run_hwtopolb.csv").write_text(csv_header + csv_rows, encoding="utf-8")

        (self.output_dir / "run_erad.log").write_text(
            "[kneighbor] algo=erad mode=gpu_opencl total_time_s=0.5 loop_time_s=0.45 num_rebalances=0 avg_rebalance_gain=0\n",
            encoding="utf-8",
        )
        (self.output_dir / "run_hwtopolb.log").write_text(
            "[kneighbor] algo=hwtopolb mode=gpu_opencl total_time_s=0.4 loop_time_s=0.35 num_rebalances=0 avg_rebalance_gain=0\n",
            encoding="utf-8",
        )

        run_direct = asm.parse_run(
            self.output_dir / "run_erad.log",
            self.output_dir / "run_erad.csv",
            expected_iterations=100,
            metadata={"parameters": {"iterations": 5}, "benchmark": "kneighbor", "algo": "erad"},
        )
        self.assertEqual(run_direct["status"], "SUCCESS", "parse_run should use iterations from metadata/parameters")

        runs = asm.discover_runs(self.output_dir, expected_iterations=100)
        self.assertEqual(len(runs), 2)
        for r in runs:
            self.assertEqual(
                r["status"],
                "SUCCESS",
                f"Run {r['algo']} with 5 steps should be SUCCESS when manifest has iterations=5, got: {r['status']}",
            )

        pairs = asm.compute_paired_analysis(runs)
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0]["ratio"], 0.45 / 0.35)



if __name__ == "__main__":
    unittest.main()
