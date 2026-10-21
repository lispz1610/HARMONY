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
bench_timeout="${SAMPLE_TIMEOUT_SECONDS:-180}"
dry_run="${DRY_RUN:-0}"
skip_preflight="${BENCH_SKIP_PREFLIGHT:-0}"
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
  --timeout <INT>          Timeout per sample in seconds (default: 180 or \$SAMPLE_TIMEOUT_SECONDS)
  --launcher <CMD>         MPI launcher command or override (default: prun or \$MPI_LAUNCHER)
  --skip-preflight         Bypass Stage 1 preflight checks (topo probe and halo test)
  --dry-run                Validate configuration and prebuilt binaries, display plan, and exit without running
  -h, --help               Display this help message and exit

Environment Variables:
  BENCH_BIN_DIR            Same as --bin-dir
  BENCH_OUTPUT_DIR         Same as --output-dir
  BENCH_N                  Same as --n
  BENCH_ITERATIONS         Same as --iterations
  BENCH_BALANCE_INTERVAL   Same as --balance-interval
  BENCH_SEEDS              Same as --seeds
  BENCH_SKIP_PREFLIGHT     Set to 1 to bypass preflight checks
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
        --timeout)
            bench_timeout="$2"
            shift 2
            ;;
        --launcher)
            mpi_launcher="$2"
            shift 2
            ;;
        --skip-preflight)
            skip_preflight=1
            shift
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

if ! [[ "$bench_timeout" =~ ^[1-9][0-9]*s?$ ]]; then
    echo "Error: --timeout must be a positive integer or duration (e.g. 180 or 180s), got: '$bench_timeout'" >&2
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
if [[ "$mpi_launcher" == "none" ]]; then
    launcher_cmd=()
elif [[ -n "$mpi_launcher" ]]; then
    read -r -a launcher_cmd <<< "$mpi_launcher"
elif command -v prun >/dev/null 2>&1; then
    launcher_cmd=("prun")
elif command -v mpirun >/dev/null 2>&1; then
    launcher_cmd=("mpirun")
else
    launcher_cmd=()
fi

# Configure sample timeout string
if [[ "$bench_timeout" =~ ^[0-9]+$ ]]; then
    sample_timeout_str="${bench_timeout}s"
else
    sample_timeout_str="$bench_timeout"
fi

# Hardware Telemetry helper
record_telemetry() {
    local block_name="$1"
    local telem_file="${output_dir}/hardware_telemetry.log"
    local now
    now="$(date -u +"%Y-%m-%d %H:%M:%S" 2>/dev/null || date +"%Y-%m-%d %H:%M:%S")"
    echo "# [telemetry] block=${block_name} timestamp=${now}" >> "$telem_file"

    local probed=0

    # 1. Query telemetry across all participating ranks/nodes via configured launcher
    if [[ ${#launcher_cmd[@]} -gt 0 ]]; then
        local probe_script="${script_dir}/telemetry_probe.sh"
        local probe_out=""
        if probe_out=$("${launcher_cmd[@]}" "$probe_script" 2>/dev/null) && [[ -n "$probe_out" ]]; then
            if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE:-}" ]]; then
                local all_found=1
                while IFS= read -r node || [[ -n "$node" ]]; do
                    node="$(echo "$node" | tr -d '[:space:]')"
                    [[ -z "$node" ]] && continue
                    local node_clean="${node%\[*\]}"
                    if ! echo "$probe_out" | grep -Fq "$node_clean" && ! echo "$probe_out" | grep -Fq "$node"; then
                        all_found=0
                        break
                    fi
                done < "$PBS_NODEFILE"
                if [[ $all_found -eq 1 ]]; then
                    echo "$probe_out" >> "$telem_file"
                    probed=1
                fi
            else
                echo "$probe_out" >> "$telem_file"
                probed=1
            fi
        fi
    fi

    # 2. Check $PBS_NODEFILE if present and not yet probed.
    # Only nodes this shell can reach without ssh (rank 0 and the local host) are queried;
    # every other node gets an explicit "telemetry-unavailable" marker instead of a fake sample.
    if [[ $probed -eq 0 && -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE:-}" ]]; then
        local rank=0
        local local_host
        local_host="$(hostname -s 2>/dev/null || hostname)"
        while IFS= read -r node || [[ -n "$node" ]]; do
            node="$(echo "$node" | tr -d '[:space:]')"
            [[ -z "$node" ]] && continue
            local node_clean="${node%\[*\]}"
            local node_short="${node_clean%%.*}"
            local node_probed=0
            local unavailable_reason="no ssh-free path to query nvidia-smi on a non-local node (launcher telemetry probe failed or did not cover it)"
            if [[ $rank -eq 0 || "$node_short" == "$local_host" ]]; then
                if command -v nvidia-smi >/dev/null 2>&1; then
                    local smi_out=""
                    if smi_out=$(nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,clocks.current.graphics,clocks.current.memory,power.draw,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>/dev/null) && [[ -n "$smi_out" ]]; then
                        while IFS= read -r line; do
                            [[ -n "$line" ]] && echo "${node_clean}, ${rank}, ${line}" >> "$telem_file"
                        done <<< "$smi_out"
                        local apps_out=""
                        apps_out=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
                        local apps_found=0
                        while IFS= read -r app_line; do
                            if [[ -n "$app_line" ]] && [[ "$app_line" != *"NVIDIA"* ]]; then
                                echo "# [compute-apps] host=${node_clean} rank=${rank} ${app_line}" >> "$telem_file"
                                apps_found=1
                            fi
                        done <<< "$apps_out"
                        if [[ $apps_found -eq 0 ]]; then
                            echo "# [compute-apps] host=${node_clean} rank=${rank} shared_workload=none" >> "$telem_file"
                        fi
                        node_probed=1
                    else
                        unavailable_reason="nvidia-smi query failed on the local node"
                    fi
                else
                    unavailable_reason="nvidia-smi not available on the local node"
                fi
            fi
            if [[ $node_probed -eq 0 ]]; then
                echo "# [telemetry-unavailable] host=${node_clean} rank=${rank} block=${block_name} reason=${unavailable_reason}" >> "$telem_file"
            fi
            probed=1
            rank=$((rank + 1))
        done < "$PBS_NODEFILE"
    fi

    # 3. Graceful fallback to local nvidia-smi or placeholder
    if [[ $probed -eq 0 ]]; then
        local host
        host="$(hostname -s 2>/dev/null || hostname)"
        if command -v nvidia-smi >/dev/null 2>&1; then
            local smi_out=""
            if smi_out=$(nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,clocks.current.graphics,clocks.current.memory,power.draw,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>/dev/null) && [[ -n "$smi_out" ]]; then
                while IFS= read -r line; do
                    [[ -n "$line" ]] && echo "${host}, 0, ${line}" >> "$telem_file"
                done <<< "$smi_out"
                local apps_out=""
                apps_out=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
                if [[ -n "$apps_out" ]]; then
                    local apps_found=0
                    while IFS= read -r app_line; do
                        if [[ -n "$app_line" ]] && [[ "$app_line" != *"NVIDIA"* ]]; then
                            echo "# [compute-apps] host=${host} rank=0 ${app_line}" >> "$telem_file"
                            apps_found=1
                        fi
                    done <<< "$apps_out"
                    if [[ $apps_found -eq 0 ]]; then
                        echo "# [compute-apps] host=${host} rank=0 shared_workload=none" >> "$telem_file"
                    fi
                else
                    echo "# [compute-apps] host=${host} rank=0 shared_workload=none" >> "$telem_file"
                fi
            else
                echo "${host}, 0, ${now}, placeholder, N/A, N/A, N/A, N/A, N/A, [N/A] (nvidia-smi query failed for block ${block_name})" >> "$telem_file"
                echo "# [compute-apps] host=${host} rank=0 shared_workload=none" >> "$telem_file"
            fi
        else
            echo "${host}, 0, ${now}, placeholder, N/A, N/A, N/A, N/A, N/A, [N/A] (nvidia-smi not available for block ${block_name})" >> "$telem_file"
            echo "# [compute-apps] host=${host} rank=0 shared_workload=none" >> "$telem_file"
        fi
    fi
}

# Run Manifest helpers
init_manifest() {
    python3 - "$output_dir/run_manifest.json" "$bench_n" "$bench_iterations" "$bench_interval" "${seeds_array[*]}" <<'PY'
import json
import os
import sys

manifest_path = sys.argv[1]
parameters = {
    "n": int(sys.argv[2]),
    "iterations": int(sys.argv[3]),
    "balance_interval": int(sys.argv[4]),
    "seeds": [int(s) for s in sys.argv[5].split() if s]
}
data = {
    "parameters": parameters,
    "runs": []
}
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
PY
}

record_sample_manifest() {
    local run_id="$1"
    local stage="$2"
    local rep="$3"
    local bench="$4"
    local algo="$5"
    local seed="$6"
    local step="$7"
    local cmd_str="$8"
    local csv_rel="$9"
    local log_rel="${10}"
    local exit_code="${11}"
    local elapsed="${12}"
    local status="${13}"

    python3 - "$output_dir/run_manifest.json" \
        "$run_id" "$stage" "$rep" "$bench" "$algo" "$seed" "$step" \
        "$cmd_str" "$csv_rel" "$log_rel" "$exit_code" "$elapsed" "$status" <<'PY'
import json
import os
import sys

manifest_path = sys.argv[1]
entry = {
    "run_id": int(sys.argv[2]),
    "stage": sys.argv[3],
    "repetition": int(sys.argv[4]) if sys.argv[4] not in ("null", "None", "") else None,
    "benchmark": sys.argv[5],
    "algorithm": sys.argv[6],
    "algo": sys.argv[6],
    "seed": int(sys.argv[7]),
    "step": int(sys.argv[8]) if sys.argv[8] not in ("null", "None", "") else None,
    "command": sys.argv[9],
    "csv_file": sys.argv[10],
    "log_file": sys.argv[11],
    "exit_code": int(sys.argv[12]),
    "elapsed_seconds": float(sys.argv[13]),
    "status": sys.argv[14]
}

data = {"runs": []}
if os.path.exists(manifest_path):
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {"runs": []}

if "runs" not in data:
    data["runs"] = []

data["runs"].append(entry)
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
PY
}

get_rep_seeds() {
    local rep="$1"
    if [[ ${#seeds_array[@]} -eq 3 ]]; then
        local s0="${seeds_array[0]}"
        local s1="${seeds_array[1]}"
        local s2="${seeds_array[2]}"
        local rot=$(( (rep - 1) % 3 ))
        case "$rot" in
            0) echo "$s0 $s1 $s2" ;;
            1) echo "$s1 $s2 $s0" ;;
            2) echo "$s2 $s0 $s1" ;;
        esac
    else
        echo "${seeds_array[*]}"
    fi
}

get_rep_algos() {
    local rep="$1"
    if (( rep % 2 == 1 )); then
        echo "erad hwtopolb"
    else
        echo "hwtopolb erad"
    fi
}

run_counter=0
failed_samples=0

execute_sample() {
    local stage="$1"
    local rep="$2"
    local bench="$3"
    local algo="$4"
    local seed="$5"
    local step="$6"

    run_counter=$((run_counter + 1))
    local run_id="$run_counter"
    local binary="$bin_dir/${bench}.out"

    local csv_basename
    local log_basename
    if [[ "$stage" == "warmup" ]]; then
        csv_basename="warmup_${bench}_${algo}_${seed}.csv"
        log_basename="warmup_${bench}_${algo}_${seed}.log"
    elif [[ "$stage" == "matrix" ]]; then
        csv_basename="${bench}_${algo}_${seed}_rep${rep}.csv"
        log_basename="${bench}_${algo}_${seed}_rep${rep}.log"
    elif [[ "$stage" == "aba" ]]; then
        csv_basename="${bench}_${algo}_${seed}_aba${step}.csv"
        log_basename="${bench}_${algo}_${seed}_aba${step}.log"
    else
        csv_basename="${bench}_${algo}_${seed}.csv"
        log_basename="${bench}_${algo}_${seed}.log"
    fi

    local csv_file="${output_dir}/${csv_basename}"
    local log_file="${output_dir}/${log_basename}"

    local cmd=("${launcher_cmd[@]}" "$binary" --n "$bench_n" --iterations "$bench_iterations" \
        --balance-interval "$bench_interval" --balance-algo "$algo" \
        --perturbation 0.10 --seed "$seed" --output-csv "$csv_file")

    echo "[Sample $run_id] stage=$stage rep=$rep bench=$bench algo=$algo seed=$seed step=$step"

    local start_time
    local end_time
    start_time=$(python3 -c 'import time; print(time.time())')

    set +e
    timeout "$sample_timeout_str" "${cmd[@]}" > "$log_file" 2>&1
    local exit_code=$?
    set -e

    end_time=$(python3 -c 'import time; print(time.time())')
    local elapsed
    elapsed=$(python3 -c "print(round($end_time - $start_time, 4))")

    local status="success"
    if [[ $exit_code -eq 124 ]]; then
        status="timeout"
        failed_samples=$((failed_samples + 1))
        echo "[WARNING] Sample $run_id timed out after $sample_timeout_str: $bench $algo seed $seed" >&2
    elif [[ $exit_code -ne 0 ]]; then
        status="failed"
        failed_samples=$((failed_samples + 1))
        echo "[WARNING] Sample $run_id failed with exit code $exit_code: $bench $algo seed $seed" >&2
    fi

    record_sample_manifest "$run_id" "$stage" "$rep" "$bench" "$algo" "$seed" "$step" \
        "${cmd[*]}" "$csv_basename" "$log_basename" "$exit_code" "$elapsed" "$status"
}

# Stage 1: Preflight Verification
if [[ "$skip_preflight" -eq 1 ]]; then
    echo "=== Stage 1: Preflight Verification (SKIPPED via --skip-preflight) ==="
else
    echo "=== Stage 1: Preflight Verification ==="
    # The spec allows 60s in total for all preflight probes: one shared deadline.
    preflight_budget="${PREFLIGHT_BUDGET_SECONDS:-60}"
    preflight_deadline=$((SECONDS + preflight_budget))
    preflight_remaining() {
        local remaining=$((preflight_deadline - SECONDS))
        if [[ $remaining -lt 1 ]]; then
            echo "Preflight failed: budget of ${preflight_budget}s exhausted before $1" >&2
            exit 1
        fi
        echo "$remaining"
    }

    topo_stdout="$output_dir/topo_probe.stdout"
    remaining=$(preflight_remaining "topology probe")
    echo "Running topology probe (shared preflight budget: ${remaining}s of ${preflight_budget}s left): $bin_dir/topo_probe.out"
    timeout "${remaining}s" "${launcher_cmd[@]}" "$bin_dir/topo_probe.out" --output-json "$output_dir/topo_metrics_cluster.json" | tee "$topo_stdout"

    # Validate generated topology metrics
    python3 - "$output_dir/topo_metrics_cluster.json" "$topo_stdout" <<'PY'
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

# The topology JSON only holds flat per-device arrays, so per-rank GPU counts come from
# the "rank_device_counts=" line that topo_probe prints on rank 0.
counts = None
with open(sys.argv[2], encoding="utf-8") as stdout_log:
    for line in stdout_log:
        if line.startswith("rank_device_counts="):
            try:
                counts = [int(v) for v in line.split("=", 1)[1].strip().split(",")]
            except ValueError:
                pass
if counts is None:
    sys.stderr.write("Preflight failed: topo_probe did not report rank_device_counts; rebuild it with cluster_build.sh\n")
    sys.exit(1)
if len(counts) < 2 or any(c < 1 for c in counts):
    sys.stderr.write(f"Preflight failed: per-rank OpenCL GPU counts must be >= 1 on every rank (at least 2 ranks), got {counts}\n")
    sys.exit(1)
if sum(counts) != len(metrics["pcie_latency_ns"]):
    sys.stderr.write(f"Preflight failed: per-rank GPU counts {counts} disagree with {len(metrics['pcie_latency_ns'])} measured devices\n")
    sys.exit(1)
PY

    remaining=$(preflight_remaining "asynchronous halo test")
    echo "Running asynchronous OpenCL halo test (shared preflight budget: ${remaining}s of ${preflight_budget}s left): $bin_dir/test_async_halo_opencl.out"
    if ! timeout "${remaining}s" "${launcher_cmd[@]}" "$bin_dir/test_async_halo_opencl.out"; then
        echo "ERROR: Asynchronous halo test preflight failed (requires at least 1 OpenCL GPU per rank)" >&2
        exit 1
    fi
fi

# Initialize Manifest
init_manifest

# Phase 1: Untimed Warm-up Phase
echo "=== Phase 1: Untimed Warm-up Phase ==="
record_telemetry "before_warmup"
default_seed="${seeds_array[0]}"
for bench in kneighbor leanmd; do
    for algo in erad hwtopolb; do
        execute_sample "warmup" "null" "$bench" "$algo" "$default_seed" "null"
    done
done
record_telemetry "after_warmup"

# Phase 2: Rotated Matrix Execution (5 repetitions)
echo "=== Phase 2: Rotated Matrix Execution ==="
for rep in 1 2 3 4 5; do
    echo "--- Repetition $rep / 5 ---"
    record_telemetry "before_rep_${rep}"
    read -r -a rep_algos <<< "$(get_rep_algos "$rep")"
    read -r -a rep_seeds <<< "$(get_rep_seeds "$rep")"

    for bench in kneighbor leanmd; do
        for algo in "${rep_algos[@]}"; do
            for seed in "${rep_seeds[@]}"; do
                execute_sample "matrix" "$rep" "$bench" "$algo" "$seed" "null"
            done
        done
    done
    record_telemetry "after_rep_${rep}"
done

# Phase 3: Dedicated A/B/A Test (Seed 123)
echo "=== Phase 3: Dedicated A/B/A Test (Seed 123) ==="
record_telemetry "before_aba"
for bench in kneighbor leanmd; do
    execute_sample "aba" "null" "$bench" "erad" "123" "1"
    execute_sample "aba" "null" "$bench" "hwtopolb" "123" "2"
    execute_sample "aba" "null" "$bench" "erad" "123" "3"
done
record_telemetry "after_aba"

if [[ "$failed_samples" -gt 0 ]]; then
    echo "ERROR: $failed_samples sample(s) failed or timed out during matrix execution" >&2
    exit 1
fi

echo "=== Super Matrix execution completed successfully ==="
exit 0
