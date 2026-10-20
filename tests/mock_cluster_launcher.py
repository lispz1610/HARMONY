#!/usr/bin/env python3
"""
Mock MPI Launcher for HARMONY Super Job offline integration tests.
Simulates two-rank OpenCL benchmark outputs, device discovery, topology metrics,
and balance-metrics logs without requiring live GPUs or PBS.
"""

import json
import os
import sys


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
    rebalances = (iterations - 1) // interval if interval > 0 else 0
    sys.stdout.write(
        f"[{bench_name}] algo={algo} mode=gpu_opencl total_time_s={wall_time} "
        f"num_rebalances={rebalances} avg_rebalance_gain=6.34934e-05\n"
    )
    sys.exit(0)


def main():
    binary, target_args = parse_args(sys.argv[1:])
    if not binary:
        sys.stderr.write("mock_cluster_launcher: No executable binary specified\n")
        sys.exit(1)

    binary_name = os.path.basename(binary)
    if "topo_probe" in binary_name:
        simulate_topo_probe(target_args)
    elif "test_async_halo" in binary_name:
        simulate_test_async_halo(target_args)
    elif "kneighbor" in binary_name or "leanmd" in binary_name:
        simulate_benchmark(binary_name, target_args)
    else:
        # Generic successful exit for other test executables
        sys.stdout.write(f"Mock execution for {binary_name} succeeded.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
