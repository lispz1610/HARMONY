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

CSV_KN_ERAD="${SCRIPT_DIR}/kneighbor_erad.csv"
CSV_KN_HW="${SCRIPT_DIR}/kneighbor_hwtopolb.csv"
CSV_MD_ERAD="${SCRIPT_DIR}/leanmd_erad.csv"
CSV_MD_HW="${SCRIPT_DIR}/leanmd_hwtopolb.csv"

echo "=== Running R4 Benchmarks (ERAD vs HWTOPOLB) ==="
echo "Running kneighbor (erad)..."
"${KNEIGHBOR_BIN}" --n 1000000 --iterations 100 --balance-interval 10 --balance-algo erad --output-csv "${CSV_KN_ERAD}" > /dev/null

echo "Running kneighbor (hwtopolb)..."
"${KNEIGHBOR_BIN}" --n 1000000 --iterations 100 --balance-interval 10 --balance-algo hwtopolb --perturbation 0.10 --output-csv "${CSV_KN_HW}" > /dev/null

echo "Running leanmd (erad)..."
"${LEANMD_BIN}" --n 1000000 --iterations 100 --balance-interval 10 --balance-algo erad --output-csv "${CSV_MD_ERAD}" > /dev/null

echo "Running leanmd (hwtopolb)..."
"${LEANMD_BIN}" --n 1000000 --iterations 100 --balance-interval 10 --balance-algo hwtopolb --perturbation 0.10 --output-csv "${CSV_MD_HW}" > /dev/null

echo ""
python3 - "${ROOT_DIR}" << 'PYEOF'
import csv
import sys
import os

root_dir = sys.argv[1]
runs = [
    ("kneighbor", "erad", os.path.join(root_dir, "benchmarks", "kneighbor_erad.csv")),
    ("kneighbor", "hwtopolb", os.path.join(root_dir, "benchmarks", "kneighbor_hwtopolb.csv")),
    ("leanmd", "erad", os.path.join(root_dir, "benchmarks", "leanmd_erad.csv")),
    ("leanmd", "hwtopolb", os.path.join(root_dir, "benchmarks", "leanmd_hwtopolb.csv")),
]

table_rows = []
for bench, algo, csv_path in runs:
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
    
    avg_gain = (sum_gain / num_rebalances) if num_rebalances > 0 else 0.0
    table_rows.append((bench, algo, f"{total_time:.6f}", str(num_rebalances), f"{avg_gain:.6f}"))

# Print summary table
header = "benchmark | algo | total_time_s | num_rebalances | avg_rebalance_gain"
sep = "-" * len(header)
print(header)
print(sep)
for r in table_rows:
    print(f"{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]}")
PYEOF

exit 0
