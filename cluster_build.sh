#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"

module load gnu13/13.2.0
module load openmpi5/5.0.5
module load prun/2.2

echo "=== Frontend build toolchain ==="
module list 2>&1
command -v mpic++
mpic++ --version
printf '#include <span>\nint main() { std::span<int> value; return value.size(); }\n' \
    | mpic++ -std=c++20 -x c++ -fsyntax-only -

bin_dir="$root_dir/build/cluster"
mkdir -p "$bin_dir"
common_flags=(-std=c++20 -Wall -Wextra -O2 -DCL_TARGET_OPENCL_VERSION=300)

echo "=== Build test suites ==="
for name in topo_metrics numa_cost hierarchical algorithms contention_power \
            probe_runtime_integration async_halo_opencl; do
    mpic++ "${common_flags[@]}" "tests/test_${name}.cpp" -lOpenCL \
        -o "$bin_dir/test_${name}.out"
done

echo "=== Build topology probe and main-program smoke executable ==="
mpic++ "${common_flags[@]}" topo_probe.cpp -lOpenCL \
    -o "$bin_dir/topo_probe.out"
mpic++ "${common_flags[@]}" main_high_order_residual.cpp dcl/runtime.cpp \
    -lOpenCL -o "$bin_dir/harmony_smoke.out"

echo "=== Build benchmarks ==="
BENCH_BIN_DIR="$bin_dir" BENCH_BUILD_ONLY=1 \
    bash "$root_dir/benchmarks/compare_algos.sh"

echo "Frontend build complete: $bin_dir"
