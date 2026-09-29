#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

KNEIGHBOR_BIN="${ROOT_DIR}/kneighbor.out"
LEANMD_BIN="${ROOT_DIR}/leanmd.out"

if [[ ! -x "${KNEIGHBOR_BIN}" ]]; then
    echo "Compiling kneighbor.out..."
    mpic++ -std=c++20 -Wall -Wextra -Wpedantic -Wno-unused-parameter -O3 "${ROOT_DIR}/benchmarks/kneighbor.cpp" -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 -o "${KNEIGHBOR_BIN}"
fi

if [[ ! -x "${LEANMD_BIN}" ]]; then
    echo "Compiling leanmd.out..."
    mpic++ -std=c++20 -Wall -Wextra -Wpedantic -Wno-unused-parameter -O3 "${ROOT_DIR}/benchmarks/leanmd.cpp" -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 -o "${LEANMD_BIN}"
fi

MPI_LAUNCHER=${MPI_LAUNCHER:-}

echo "=== Running R4 Benchmarks (ERAD vs HWTOPOLB) with multiple seeds ==="
SEEDS=(42 123 456)

for seed in "${SEEDS[@]}"; do
    echo "Running seed ${seed}..."
    ${MPI_LAUNCHER} "${KNEIGHBOR_BIN}" --n 1000000 --iterations 100 --balance-interval 10 --balance-algo erad --seed ${seed} --output-csv "${SCRIPT_DIR}/kneighbor_erad_${seed}.csv" > /dev/null
    ${MPI_LAUNCHER} "${KNEIGHBOR_BIN}" --n 1000000 --iterations 100 --balance-interval 10 --balance-algo hwtopolb --perturbation 0.10 --seed ${seed} --output-csv "${SCRIPT_DIR}/kneighbor_hwtopolb_${seed}.csv" > /dev/null
    ${MPI_LAUNCHER} "${LEANMD_BIN}" --n 1000000 --iterations 100 --balance-interval 10 --balance-algo erad --seed ${seed} --output-csv "${SCRIPT_DIR}/leanmd_erad_${seed}.csv" > /dev/null
    ${MPI_LAUNCHER} "${LEANMD_BIN}" --n 1000000 --iterations 100 --balance-interval 10 --balance-algo hwtopolb --perturbation 0.10 --seed ${seed} --output-csv "${SCRIPT_DIR}/leanmd_hwtopolb_${seed}.csv" > /dev/null
done

echo ""
python3 - "${ROOT_DIR}" << 'PYEOF'
import csv
import sys
import os
import statistics

root_dir = sys.argv[1]
seeds = [42, 123, 456]
runs = [
    ("kneighbor", "erad"),
    ("kneighbor", "hwtopolb"),
    ("leanmd", "erad"),
    ("leanmd", "hwtopolb"),
]

table_rows = []
for bench, algo in runs:
    times = []
    rebalances = []
    gains = []
    
    for seed in seeds:
        csv_path = os.path.join(root_dir, "benchmarks", f"{bench}_{algo}_{seed}.csv")
        total_time = 0.0
        num_rebalances = 0
        sum_gain = 0.0
        
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                step_t = float(row['step_time_s'])
                total_time += step_t
                rebal = int(row['rebalanced'])
                if rebal == 1:
                    num_rebalances += 1
                    sum_gain += float(row['rebalance_gain'])
        
        times.append(total_time)
        rebalances.append(num_rebalances)
        if num_rebalances > 0:
            gains.append(sum_gain / num_rebalances)
        else:
            gains.append(0.0)
            
    median_time = statistics.median(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    avg_rebalances = sum(rebalances) / len(rebalances)
    avg_gain = sum(gains) / len(gains)
    
    table_rows.append((bench, algo, f"{median_time:.6f}", f"{std_time:.6f}", f"{avg_rebalances:.1f}", f"{avg_gain:.6f}"))

csv_out = os.path.join(root_dir, "benchmarks", "comparison_summary.csv")
with open(csv_out, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["benchmark", "algo", "median_time_s", "stddev_time_s", "avg_rebalances", "avg_rebalance_gain"])
    for r in table_rows:
        writer.writerow(r)

print("benchmark,algo,median_time_s,stddev_time_s,avg_rebalances,avg_rebalance_gain")
for r in table_rows:
    print(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}")


PYEOF

exit 0
