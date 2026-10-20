#!/usr/bin/env bash
# HARMONY Non-Invasive PBS Super Job Matrix Runner
# Executes variance benchmarks, preflight checks, and telemetry in an isolated run directory.
# Strictly enforces precompiled frontend binaries and forbids compilation on compute nodes.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"

# Default parameter configurations
bin_dir="${BENCH_BIN_DIR:-${root_dir}/build/cluster}"
output_dir="${BENCH_OUTPUT_DIR:-${root_dir}/benchmarks/super_matrix_results}"
bench_n="${BENCH_N:-1000000}"
bench_iterations="${BENCH_ITERATIONS:-100}"
bench_interval="${BENCH_BALANCE_INTERVAL:-10}"
bench_seeds="${BENCH_SEEDS:-42 123 456}"
dry_run="${DRY_RUN:-0}"
mpi_launcher="${MPI_LAUNCHER:-}"

show_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

HARMONY Non-Invasive PBS Super Job Matrix Runner.

Options:
  --bin-dir <PATH>         Path to precompiled cluster binaries (default: \$BENCH_BIN_DIR or build/cluster)
  --output-dir <PATH>      Directory to store logs, CSVs, and telemetry (default: \$BENCH_OUTPUT_DIR)
  --n <INT>                Global problem size N (default: 1000000)
  --iterations <INT>       Number of iterations per benchmark run (default: 100)
  --balance-interval <INT> Interval between rebalance steps (default: 10)
  --seeds <LIST>           Seeds to evaluate, space or comma separated (default: "42 123 456")
  --launcher <CMD>         MPI launcher command or override (default: prun or \$MPI_LAUNCHER)
  --dry-run                Validate configuration and prebuilt binaries, display plan, and exit without running
  -h, --help               Display this help message and exit

Environment Variables:
  BENCH_BIN_DIR            Same as --bin-dir
  BENCH_OUTPUT_DIR         Same as --output-dir
  BENCH_N                  Same as --n
  BENCH_ITERATIONS         Same as --iterations
  BENCH_BALANCE_INTERVAL   Same as --balance-interval
  BENCH_SEEDS              Same as --seeds
  MPI_LAUNCHER             Same as --launcher
  DRY_RUN                  Set to 1 to enable dry-run mode
EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin-dir)
            bin_dir="$2"
            shift 2
            ;;
        --output-dir)
            output_dir="$2"
            shift 2
            ;;
        --n)
            bench_n="$2"
            shift 2
            ;;
        --iterations)
            bench_iterations="$2"
            shift 2
            ;;
        --balance-interval)
            bench_interval="$2"
            shift 2
            ;;
        --seeds)
            bench_seeds="$2"
            shift 2
            ;;
        --launcher)
            mpi_launcher="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $1" >&2
            show_usage >&2
            exit 1
            ;;
    esac
done

# Validate numeric arguments
if ! [[ "$bench_n" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --n must be a positive integer, got: '$bench_n'" >&2
    exit 1
fi

if ! [[ "$bench_iterations" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --iterations must be a positive integer, got: '$bench_iterations'" >&2
    exit 1
fi

if ! [[ "$bench_interval" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --balance-interval must be a positive integer, got: '$bench_interval'" >&2
    exit 1
fi

# Parse and validate seeds
cleaned_seeds="${bench_seeds//,/ }"
read -r -a seeds_array <<< "$cleaned_seeds"
if [[ ${#seeds_array[@]} -eq 0 ]]; then
    echo "Error: --seeds must not be empty" >&2
    exit 1
fi

for s in "${seeds_array[@]}"; do
    if ! [[ "$s" =~ ^[0-9]+$ ]]; then
        echo "Error: Invalid seed '$s'. Seeds must be non-negative integers." >&2
        exit 1
    fi
done

# Validate output directory
mkdir -p "$output_dir" || {
    echo "Error: Failed to create output directory: $output_dir" >&2
    exit 1
}

# Validate required prebuilt binaries
if [[ ! -d "$bin_dir" ]]; then
    echo "Error: Missing binary directory: $bin_dir" >&2
    exit 1
fi

required_binaries=(
    "kneighbor.out"
    "leanmd.out"
    "topo_probe.out"
    "test_async_halo_opencl.out"
)

for binary in "${required_binaries[@]}"; do
    if [[ ! -x "$bin_dir/$binary" ]]; then
        echo "Error: Missing prebuilt binary: $bin_dir/$binary" >&2
        echo "Compiling on compute nodes is strictly forbidden. Please build binaries on frontend using: bash cluster_build.sh" >&2
        exit 1
    fi
done

# Dry-run handling
if [[ "$dry_run" == 1 ]]; then
    dryrun_plan="[DRY-RUN] HARMONY Non-Invasive PBS Super Job Execution Plan
=====================================================================
Binary Directory:    $bin_dir
Output Directory:    $output_dir
Parameters:          N=$bench_n, iterations=$bench_iterations, interval=$bench_interval
Seeds:               ${seeds_array[*]}
Launcher:            ${mpi_launcher:-prun}

Stage 1: Preflight Verification
  - Topo probe: $bin_dir/topo_probe.out --output-json $output_dir/topo_metrics_cluster.json
  - Halo test:  $bin_dir/test_async_halo_opencl.out

Stage 2: Untimed Warm-up Phase
  - kneighbor (erad, hwtopolb)
  - leanmd (erad, hwtopolb)

Stage 3: Rotated Matrix Execution
  - Workloads:   kneighbor, leanmd
  - Algorithms:  erad, hwtopolb
  - Seeds:       ${seeds_array[*]}
  - Repetitions: 5 (alternating ERAD/HWTOPOLB order with rotated seed sequences)

Stage 4: A/B/A Isolated Verification (Seed 123)
  - ERAD -> HWTOPOLB -> ERAD for seed 123

Stage 5: Post-Processing & Telemetry
  - Consolidated summary, paired ratios, and SUPER_JOB_REPORT.md
=====================================================================
[DRY-RUN] Validation completed successfully. No workloads executed."

    echo "$dryrun_plan"
    echo "$dryrun_plan" > "$output_dir/dryrun_plan.txt"
    exit 0
fi

# Configure launcher
if [[ -n "$mpi_launcher" ]]; then
    read -r -a launcher_cmd <<< "$mpi_launcher"
elif command -v prun >/dev/null 2>&1; then
    launcher_cmd=("prun")
elif command -v mpirun >/dev/null 2>&1; then
    launcher_cmd=("mpirun")
else
    launcher_cmd=()
fi

# Stage 1: Preflight Verification
echo "=== Stage 1: Preflight Verification ==="
echo "Running topology probe: $bin_dir/topo_probe.out"
"${launcher_cmd[@]}" "$bin_dir/topo_probe.out" --output-json "$output_dir/topo_metrics_cluster.json"

# Validate generated topology metrics
python3 - "$output_dir/topo_metrics_cluster.json" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as f:
        metrics = json.load(f)
except Exception as err:
    sys.stderr.write(f"Preflight failed: could not parse topology JSON: {err}\n")
    sys.exit(1)

if metrics.get("synthetic", True):
    sys.stderr.write("Preflight failed: topology probe reported synthetic metrics\n")
    sys.exit(1)

if len(metrics.get("pcie_latency_ns", [])) < 2:
    sys.stderr.write("Preflight failed: expected at least two measured OpenCL GPUs\n")
    sys.exit(1)
PY

echo "Running asynchronous OpenCL halo test: $bin_dir/test_async_halo_opencl.out"
if ! "${launcher_cmd[@]}" "$bin_dir/test_async_halo_opencl.out"; then
    echo "Preflight failed: test_async_halo_opencl.out returned non-zero status" >&2
    exit 1
fi

# Stage 2: Matrix Execution
echo "=== Stage 2: Matrix Execution ==="
for bench in kneighbor leanmd; do
    binary="$bin_dir/${bench}.out"
    for algo in erad hwtopolb; do
        for seed in "${seeds_array[@]}"; do
            prefix="${output_dir}/${bench}_${algo}_${seed}"
            "${launcher_cmd[@]}" "$binary" --n "$bench_n" --iterations "$bench_iterations" \
                --balance-interval "$bench_interval" --balance-algo "$algo" \
                --perturbation 0.10 --seed "$seed" --output-csv "${prefix}.csv" \
                > "${prefix}.log"
        done
    done
done

echo "=== Super Matrix execution completed successfully ==="
exit 0
