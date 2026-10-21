#!/usr/bin/env python3
"""
Mock MPI Launcher for HARMONY Super Job offline integration tests.
Simulates two-rank OpenCL benchmark outputs, device discovery, topology metrics,
and balance-metrics logs without requiring live GPUs or PBS.
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime


def parse_args(argv):
    """Finds the target binary and splits launcher flags from target arguments."""
    binary_index = -1
    for i, arg in enumerate(argv):
        if arg.endswith(".out") or any(
            name in arg
            for name in (
                "topo_probe",
                "test_async_halo_opencl",
                "kneighbor",
                "leanmd",
                "telemetry",
                "nvidia-smi",
            )
        ):
            binary_index = i
            break

    if binary_index == -1:
        # Default: first argument is binary if any
        if len(argv) > 0:
            return argv[0], argv[1:]
        return None, []

    binary = argv[binary_index]
    target_args = argv[binary_index + 1 :]
    return binary, target_args


def extract_named_arg(args, name, default=None):
    """Extracts a named argument like --n 1000 from args."""
    for i in range(len(args) - 1):
        if args[i] == name:
            return args[i + 1]
    return default


def simulate_topo_probe(target_args):
    """Simulates topo_probe.out execution."""
    if os.environ.get("MOCK_FAIL_TOPO_PROBE") == "1":
        sys.stderr.write("ERROR: Simulated topo_probe failure\n")
        sys.exit(1)

    out_json = extract_named_arg(target_args, "--output-json")
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        topo_data = {
            "pcie_latency_ns": [42780.25, 31074.1],
            "pcie_bandwidth_gbps": [10.79253818, 10.37190217],
            "mpi_latency_ns": [0.0, 4998767.433, 4999223.567, 0.0],
            "mpi_bandwidth_gbps": [0.0, 0.02786097291, 0.03073437697, 0.0],
            "numa_distance": [],
            "device_numa_node": [],
            "memory_contention_factor": [],
            "thermal_tdp_watts": [],
            "current_power_watts": [],
            "synthetic": False,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(topo_data, f, indent=2)

    sys.stdout.write("[topo_probe] Probed 2 ranks, 2 devices\n")
    sys.exit(0)


def simulate_test_async_halo(target_args):
    """Simulates test_async_halo_opencl.out execution."""
    if os.environ.get("MOCK_FAIL_PREFLIGHT_HALO") == "1":
        sys.stderr.write("Preflight failed: Simulated halo communication failure\n")
        sys.exit(1)

    if os.environ.get("MOCK_ASYMMETRIC_GPUS") == "1":
        sys.stdout.write("SKIP: asynchronous halo test needs at least one OpenCL GPU per MPI rank (found min=0, total=2)\n")
        sys.exit(77)

    sys.stdout.write("[device] rank=0 global_index=0 name=NVIDIA A100 80GB PCIe\n")
    sys.stdout.write("[device] rank=1 global_index=1 name=NVIDIA A100 80GB PCIe\n")
    sys.stdout.write("[DCL][balance-metrics] iteration=1\n")
    sys.stdout.write("  policy=off\n")
    sys.stdout.write("  action=skip\n")
    sys.stdout.write("  reason=balance disabled\n")
    sys.stdout.write("PASS: asynchronous OpenCL halo matches two-step CPU reference\n")
    sys.exit(0)


def simulate_benchmark(binary_name, target_args):
    """Simulates kneighbor.out and leanmd.out benchmark execution."""
    if os.environ.get("MOCK_FAIL_BENCHMARK") == "1":
        sys.stderr.write(f"ERROR: Simulated benchmark failure in {binary_name}\n")
        sys.exit(1)

    fail_pattern = os.environ.get("MOCK_FAIL_IF_CONTAINS")
    if fail_pattern and fail_pattern in " ".join(target_args):
        sys.stderr.write(f"ERROR: Simulated failure for {binary_name} with pattern: {fail_pattern}\n")
        sys.exit(1)

    fail_once_file = os.environ.get("MOCK_FAIL_ONCE_FILE")
    if fail_once_file and not os.path.exists(fail_once_file):
        with open(fail_once_file, "w", encoding="utf-8") as f:
            f.write("failed once\n")
        sys.stderr.write(f"ERROR: One-time simulated failure in {binary_name}\n")
        sys.exit(1)

    sleep_pattern = os.environ.get("MOCK_SLEEP_IF_CONTAINS")
    if sleep_pattern and sleep_pattern in " ".join(target_args):
        import time
        sleep_sec = float(os.environ.get("MOCK_SLEEP_SECONDS", "5"))
        time.sleep(sleep_sec)

    sleep_once_file = os.environ.get("MOCK_SLEEP_ONCE_FILE")
    if sleep_once_file and not os.path.exists(sleep_once_file):
        with open(sleep_once_file, "w", encoding="utf-8") as f:
            f.write("slept once\n")
        import time
        sleep_sec = float(os.environ.get("MOCK_SLEEP_SECONDS", "5"))
        time.sleep(sleep_sec)

    n = int(extract_named_arg(target_args, "--n", "1000000"))
    iterations = int(extract_named_arg(target_args, "--iterations", "100"))
    interval = int(extract_named_arg(target_args, "--balance-interval", "10"))
    algo = extract_named_arg(target_args, "--balance-algo", "erad")
    seed = extract_named_arg(target_args, "--seed", "42")
    output_csv = extract_named_arg(target_args, "--output-csv")

    is_kneighbor = "kneighbor" in binary_name
    bench_name = "kneighbor" if is_kneighbor else "leanmd"

    # Produce CSV if requested
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        with open(output_csv, "w", encoding="utf-8") as f:
            f.write("iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n")
            for i in range(iterations):
                is_rebal = (i > 0 and i % interval == 0)
                step_time = 0.050129352 if is_rebal else 0.020121021
                action = "rebalance" if is_rebal else "compute"
                rebalanced = 1 if is_rebal else 0
                gain = 0.000063493 if is_rebal else 0.0
                f.write(f"{i},{step_time:.9f},{action},{rebalanced},{gain:.9f},gpu_opencl\n")

    # Write log content to stdout
    sys.stdout.write("[prun] Master compute host = compute-1-0\n")
    sys.stdout.write("[prun] Resource manager = openpbs\n")
    sys.stdout.write("[device] rank=0 global_index=0 name=NVIDIA A100 80GB PCIe\n")
    sys.stdout.write("[device] rank=1 global_index=1 name=NVIDIA A100 80GB PCIe\n")
    sys.stdout.write("[DCL][balance-metrics] iteration=1\n")
    sys.stdout.write("  policy=off\n")
    sys.stdout.write("  action=skip\n")
    sys.stdout.write("  reason=balance disabled\n")
    sys.stdout.write("  T_compute_interval_s=2.7648e-05\n")
    sys.stdout.write("  T_comm_interval_s=0\n")
    sys.stdout.write("  T_balance_total_s=0\n")
    sys.stdout.write("  T_rebalance_apply_s=0\n")
    sys.stdout.write("  T_compute_accum_s=2.7648e-05\n")
    sys.stdout.write("  T_comm_accum_s=0\n")
    sys.stdout.write("  T_balance_accum_s=0\n")
    sys.stdout.write("  T_rebalance_apply_accum_s=0\n")
    sys.stdout.write("  device_times_s:\n")
    sys.stdout.write("    part[0]=2.3552e-05\n")
    sys.stdout.write("    part[1]=2.7648e-05\n")
    sys.stdout.write("  OLD loads:\n")
    sys.stdout.write("    part[0] cum=50% share=50% device_global=0 rank=0 local_index=0 offset=0 count=500000\n")
    sys.stdout.write("    part[1] cum=100% share=50% device_global=1 rank=1 local_index=-1 offset=500000 count=500000\n")
    sys.stdout.write("    total=100%\n")
    sys.stdout.write("  CURRENT loads:\n")
    sys.stdout.write("    part[0] cum=50% share=50% device_global=0 rank=0 local_index=0 offset=0 count=500000\n")
    sys.stdout.write("    part[1] cum=100% share=50% device_global=1 rank=1 local_index=-1 offset=500000 count=500000\n")
    sys.stdout.write("    total=100%\n")

    # Total wall time simulation
    wall_time = 3.73996 if is_kneighbor else 0.438039
    loop_time = 3.50000 if is_kneighbor else 0.400000
    rebalances = (iterations - 1) // interval if interval > 0 else 0
    sys.stdout.write(f"loop_time_s: {loop_time:.6f}\n")
    sys.stdout.write(
        f"[{bench_name}] algo={algo} mode=gpu_opencl total_time_s={wall_time} "
        f"loop_time_s={loop_time:.6f} num_rebalances={rebalances} avg_rebalance_gain=6.34934e-05\n"
    )
    sys.exit(0)


def simulate_telemetry(target_args):
    """Simulates multi-rank hardware telemetry output across 2 ranks/hosts."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # If mock or real nvidia-smi is in PATH, query it to preserve exact query expectations
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            res = subprocess.run(
                [
                    smi,
                    "--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,clocks.current.graphics,clocks.current.memory,power.draw,clocks_throttle_reasons.active",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
            if lines:
                sys.stdout.write(f"compute-1-0, 0, {lines[0]}\n")
                line1 = lines[1] if len(lines) > 1 else lines[0]
                sys.stdout.write(f"compute-1-1, 1, {line1}\n")
                sys.exit(0)
        except Exception:
            pass

    # Default mock telemetry across 2 ranks/hosts
    sys.stdout.write(f"compute-1-0, 0, {now}, NVIDIA A100-PCIE-40GB, 38, 12, 1410, 1215, 78.42, [N/A]\n")
    sys.stdout.write(f"compute-1-1, 1, {now}, NVIDIA A100-PCIE-40GB, 42, 18, 1410, 1215, 82.10, [N/A]\n")
    sys.exit(0)


def main():
    binary, target_args = parse_args(sys.argv[1:])
    if not binary:
        # Check if telemetry requested via flags
        if any("telemetry" in a or "nvidia-smi" in a for a in sys.argv[1:]):
            simulate_telemetry([])
        sys.stderr.write("mock_cluster_launcher: No executable binary specified\n")
        sys.exit(1)

    binary_name = os.path.basename(binary)
    if "topo_probe" in binary_name:
        simulate_topo_probe(target_args)
    elif "test_async_halo" in binary_name:
        simulate_test_async_halo(target_args)
    elif "kneighbor" in binary_name or "leanmd" in binary_name:
        simulate_benchmark(binary_name, target_args)
    elif "telemetry" in binary_name or "nvidia-smi" in binary_name or any("telemetry" in a for a in sys.argv[1:]):
        simulate_telemetry(target_args)
    else:
        # Generic successful exit for other test executables
        sys.stdout.write(f"Mock execution for {binary_name} succeeded.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
