#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"
output_dir="${BENCH_OUTPUT_DIR:-${script_dir}}"
mkdir -p "${output_dir}"

cxx_flags="-std=c++20 -Wall -Wextra -Wpedantic -Wno-unused-parameter -O3 -DCL_TARGET_OPENCL_VERSION=300"
bin_dir="${BENCH_BIN_DIR:-${output_dir}}"
kneighbor_bin="${bin_dir}/kneighbor.out"
leanmd_bin="${bin_dir}/leanmd.out"
if [[ "${BENCH_SKIP_BUILD:-0}" == 1 ]]; then
    for binary in "${kneighbor_bin}" "${leanmd_bin}"; do
        if [[ ! -x "${binary}" ]]; then
            printf 'Missing prebuilt benchmark: %s\n' "${binary}" >&2
            exit 1
        fi
    done
else
    mkdir -p "${bin_dir}"
    mpic++ ${cxx_flags} "${root_dir}/benchmarks/kneighbor.cpp" -lOpenCL -o "${kneighbor_bin}"
    mpic++ ${cxx_flags} "${root_dir}/benchmarks/leanmd.cpp" -lOpenCL -o "${leanmd_bin}"
fi

if [[ "${BENCH_BUILD_ONLY:-0}" == 1 ]]; then
    exit 0
fi

read -r -a launcher <<< "${MPI_LAUNCHER:-}"
bench_n="${BENCH_N:-1000000}"
bench_iterations="${BENCH_ITERATIONS:-100}"
bench_interval="${BENCH_BALANCE_INTERVAL:-10}"
seeds=(42 123 456)

for seed in "${seeds[@]}"; do
    for bench in kneighbor leanmd; do
        if [[ "${bench}" == kneighbor ]]; then binary="${kneighbor_bin}"; else binary="${leanmd_bin}"; fi
        for algo in erad hwtopolb; do
            prefix="${output_dir}/${bench}_${algo}_${seed}"
            "${launcher[@]}" "${binary}" --n "${bench_n}" --iterations "${bench_iterations}" \
                --balance-interval "${bench_interval}" --balance-algo "${algo}" \
                --perturbation 0.10 --seed "${seed}" --output-csv "${prefix}.csv" \
                > "${prefix}.log"
        done
    done
done

python3 - "${output_dir}" "${root_dir}" "${bench_n}" "${bench_iterations}" "${bench_interval}" "${MPI_LAUNCHER:-direct}" "${cxx_flags}" <<'PY'
import csv
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys

output_dir, root_dir, n, iterations, interval, launcher, flags = sys.argv[1:]
seeds = (42, 123, 456)
rows = []
for bench in ('kneighbor', 'leanmd'):
    for algo in ('erad', 'hwtopolb'):
        runs = []
        for seed in seeds:
            prefix = os.path.join(output_dir, f'{bench}_{algo}_{seed}')
            with open(prefix + '.csv', newline='') as stream:
                records = list(csv.DictReader(stream))
            if len(records) != int(iterations):
                raise RuntimeError(f'Incomplete benchmark: {prefix}')
            modes = {row['execution_mode'] for row in records}
            if len(modes) != 1:
                raise RuntimeError(f'Mixed execution modes: {prefix}')
            with open(prefix + '.log') as stream:
                log = stream.read()
            match = re.search(r'total_time_s=([0-9.eE+-]+)', log)
            if not match:
                raise RuntimeError(f'Missing global wall time: {prefix}')
            runs.append((float(match.group(1)), modes.pop(),
                         sum(int(row['rebalanced']) for row in records), seed))
        modes = {run[1] for run in runs}
        if len(modes) != 1:
            raise RuntimeError(f'Mixed execution modes across runs: {bench}/{algo}')
        times = [run[0] for run in runs]
        rows.append([bench, algo, modes.pop(), statistics.median(times),
                     statistics.stdev(times), statistics.mean(run[2] for run in runs),
                     ','.join(str(run[3]) for run in runs)])

summary = os.path.join(output_dir, 'comparison_summary.csv')
with open(summary, 'w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['benchmark', 'algo', 'execution_mode', 'median_total_wall_s',
                     'stddev_total_wall_s', 'avg_rebalances', 'seeds'])
    writer.writerows(rows)

metadata = {
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root_dir, text=True).strip(),
    'working_tree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root_dir, text=True).strip()),
    'hostname': platform.node(),
    'platform': platform.platform(),
    'compiler': subprocess.check_output(['mpic++', '--version'], text=True).splitlines()[0],
    'compiler_flags': flags,
    'mpi_launcher': launcher,
    'global_elements': int(n),
    'iterations': int(iterations),
    'rebalance_interval': int(interval),
    'warmup_iterations': 0,
    'seeds': list(seeds),
    'execution_modes': sorted({row[2] for row in rows}),
}
source_digest = hashlib.sha256()
for relative_path in (
        'dcl/algorithms.hpp', 'dcl/runtime_impl.hpp', 'dcl/topo_metrics_io.hpp',
        'dcl/types.hpp', 'benchmarks/kneighbor.cpp', 'benchmarks/kneighbor.cl',
        'benchmarks/leanmd.cpp', 'benchmarks/leanmd.cl'):
    with open(os.path.join(root_dir, relative_path), 'rb') as stream:
        source_digest.update(relative_path.encode() + b'\0' + stream.read())
metadata['source_sha256'] = source_digest.hexdigest()
with open(os.path.join(output_dir, 'comparison_metadata.json'), 'w') as stream:
    json.dump(metadata, stream, indent=2)
print(f'Summary: {summary}')
print(f'Metadata: {os.path.join(output_dir, "comparison_metadata.json")}')
PY
