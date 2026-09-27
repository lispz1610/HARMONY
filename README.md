# HARMONY — Heterogeneous Adaptive Runtime for Multi-Node Orchestration & Numerics Yielding

> **C++20 · OpenCL · MPI · `dcl` namespace**

HARMONY is a header-driven, high-performance computing (HPC) library for **topology-aware heterogeneous load balancing** in distributed scientific simulations. It orchestrates work distribution across multi-GPU, multi-CPU, and accelerator clusters, dynamically redistributing computational partitions based on observed device throughput, interconnect topology, NUMA hierarchy, resource contention, and power envelopes.

---

## Table of Contents

1. [Motivation & Theoretical Background](#1-motivation--theoretical-background)
2. [Macro-Level Transformations](#2-macro-level-transformations)
3. [Future Work Items Consolidated](#3-future-work-items-consolidated)
   - [R1 — Exact PCIe and MPI Latency/Bandwidth Injection](#r1--exact-pcie-and-mpi-latencybandwidth-injection)
   - [R2 — NUMA Communication Costs in the Balancer](#r2--numa-communication-costs-in-the-balancer)
   - [R3 — Hierarchical Balancing Strategy](#r3--hierarchical-balancing-strategy)
   - [R4 — ERAD vs. HWTOPOLB Algorithm Comparison](#r4--erad-vs-hwtopolb-algorithm-comparison)
   - [R5 — Resource Contention & Power Consumption Model](#r5--resource-contention--power-consumption-model)
4. [File Inventory](#4-file-inventory)
5. [Directory Structure](#5-directory-structure)
6. [Building and Running](#6-building-and-running)
7. [Running on a PBS Cluster](#7-running-on-a-pbs-cluster)
8. [Test Results & Verification](#8-test-results--verification)
9. [Static Analysis Baseline](#9-static-analysis-baseline)

---

## 1. Motivation & Theoretical Background

HARMONY was originally developed to address the shortcomings of static and round-robin partitioning in heterogeneous clusters where GPUs, CPUs, and accelerators coexist under a single MPI communicator. The library's architecture and algorithmic foundations are directly inspired by the paper:

> **"A topology-aware load balancing algorithm for clustered hierarchical multi-core machines"**
> (referred to herein as the **HwTopoLB paper**)

The paper identifies several limitations of purely throughput-driven balancing and proposes extensions exploiting cluster topology — i.e., PCIe lanes, NUMA domains, inter-node MPI latency — to make more informed migration decisions. Five concrete *future work* directions from that paper have been fully implemented and integrated into HARMONY in this release:

| ID | Description |
|:---|:---|
| R1 | Inject exact PCIe and MPI latency/bandwidth metrics into the runtime |
| R2 | Include NUMA communication costs when deciding whether to rebalance |
| R3 | Hierarchical balancing: intra-node (local, no MPI) followed by inter-node (global, with MPI) |
| R4 | Head-to-head comparison of ERAD (deterministic) vs. HWTOPOLB (stochastic) algorithms |
| R5 | Resource contention model and thermal/power cap integration |

---

## 2. Macro-Level Transformations

Before this release, HARMONY supported basic throughput-driven rebalancing: the runtime collected per-device kernel timings and redistricted partitions proportionally. The balancer was topology-blind — it had no awareness of PCIe bandwidth, NUMA topology, memory contention, or power draw.

After this release, HARMONY is a **topology-informed, power-aware, hierarchically-structured load balancer** with the following new capabilities:

- **Structured topology ingestion**: External measurements of PCIe and MPI latency/bandwidth can be injected at runtime via `set_topo_metrics()`, enabling hardware-calibrated migration cost estimates.
- **NUMA-cost gating**: Before executing a migration, the runtime estimates the NUMA-aware data movement cost (PCIe latency + NUMA hop penalty + transfer time) and compares it against the predicted performance gain. Rebalancing is skipped when the cost-to-gain ratio exceeds a configurable threshold, avoiding counter-productive migrations.
- **Hierarchical two-phase balancing**: A new `BalanceMode::hierarchical` delegates intra-node redistribution to local logic (no MPI synchronization overhead) and only invokes the global MPI-based balancer for inter-node load equalization, reducing collective communication frequency.
- **Algorithm-level benchmarking**: Two balancing algorithms — ERAD (deterministic, proportional-throughput) and HWTOPOLB (stochastic, perturbation-augmented) — are now formally implemented in `dcl/algorithms.hpp` and evaluated against two scientific benchmarks (`kNeighbor`, `LeanMD`).
- **Power-constrained load assignment**: The `apply_power_cap()` function redistributes partition shares respecting per-device thermal TDP limits and an optional system-wide power budget, ensuring safe operation on thermally constrained clusters.

---

## 3. Future Work Items Consolidated

### R1 — Exact PCIe and MPI Latency/Bandwidth Injection

**Goal**: Replace hard-coded or estimated topology parameters with real, measurable values obtained from the actual cluster hardware.

**Implementation**:

- **`dcl/types.hpp`** — Extended with `struct TopoMetrics`, a flat data container for all topology measurements:
  - `pcie_latency_ns` — Per-device PCIe round-trip latency (nanoseconds)
  - `pcie_bandwidth_gbps` — Per-device PCIe peak transfer bandwidth (GB/s)
  - `mpi_latency_ns` — Flat N×N matrix of one-way MPI latency between all device pairs (ns)
  - `mpi_bandwidth_gbps` — Flat N×N matrix of MPI bandwidth between all device pairs (GB/s)

- **`dcl/topo_metrics_io.hpp`** *(new)* — Header-only JSON serialization/deserialization for `TopoMetrics`. Provides `load_topo_metrics(path)` and `save_topo_metrics(metrics, path)` with a hand-written JSON parser (no external JSON library dependency).

- **`topo_probe.cpp`** *(new)* — Standalone MPI+OpenCL program that runs ping-pong benchmarks across all MPI rank pairs to measure real latency and bandwidth. Each rank probes its local OpenCL devices for PCIe characteristics. Results are aggregated at rank 0 and written to a JSON file (e.g., `topo_metrics_cluster.json`) consumable by `load_topo_metrics()`.

- **`dcl/runtime.hpp`**, **`dcl/runtime.cpp`**, **`dcl/runtime_impl.hpp`** — Added `set_topo_metrics(const TopoMetrics&)` and `topo_metrics() const` to the `Runtime` public API. Internally stored in the `Impl` class and forwarded to the rebalancing logic.

- **`tests/test_topo_metrics.cpp`** *(new)* — 7 unit tests validating JSON round-trips, error handling on malformed inputs, empty arrays, and correct field deserialization.

**Test result**: **7/7 tests passing**.

---

### R2 — NUMA Communication Costs in the Balancer

**Goal**: Prevent the balancer from triggering data migrations that cost more in NUMA-hop overhead than the performance gain they yield.

**Implementation**:

- **`dcl/types.hpp`** — `TopoMetrics` extended with:
  - `numa_distance` — Flat N×N NUMA distance matrix (unitless, same semantics as Linux `numactl --hardware` distances)
  - `device_numa_node` — Per-global-device NUMA node index mapping

- **`dcl/topo_metrics_io.hpp`** — Added two free functions:
  - `numa_distance_between(metrics, node_a, node_b)` — Returns the symmetric NUMA distance between two NUMA nodes, validated against the diagonal to avoid underestimation.
  - `estimate_migration_cost_bytes(metrics, src_dev, dst_dev, bytes, penalty_ns_per_hop)` — Computes total estimated migration cost in seconds: PCIe latency + NUMA penalty (distance × `penalty_ns_per_hop`) + transfer time (`bytes / bandwidth`). Returns conservative fallback values (5 µs latency, 10 GB/s bandwidth) when metrics are absent.

- **`dcl/types.hpp`** — `AutoBalancePolicy` extended with:
  - `numa_cost_gain_ratio_threshold` (default `0.50`) — Suppresses rebalancing when `migration_cost / expected_gain > threshold`.

- **`dcl/runtime_impl.hpp`** — `maybe_rebalance_from_timings()` computes the expected gain from proposed new partitions vs. current timings. Before issuing the rebalance, it calls `estimate_migration_cost_bytes()` for each device pair involved in migration and checks the cost-to-gain ratio. If the ratio exceeds the threshold, it prints:
  ```
  [NUMA] migration cost exceeds gain threshold, skipping rebalance
  ```
  (rank 0 only) and returns `false` without migrating data.

- **`tests/test_numa_cost.cpp`** *(new)* — 4 unit tests covering: same-node migration (no NUMA penalty), cross-node migration with penalty, empty metrics fallback, and threshold gating logic.

**Test result**: **4/4 tests passing**.

---

### R3 — Hierarchical Balancing Strategy

**Goal**: Reduce global MPI collective overhead by performing local (intra-node) load equalization independently, only escalating to inter-node MPI exchanges when necessary.

**Implementation**:

- **`dcl/types.hpp`** — `BalanceMode` enum extended with `hierarchical` variant.

- **`dcl/runtime.hpp`**, **`dcl/runtime.cpp`**, **`dcl/runtime_impl.hpp`** — Implemented `maybe_rebalance_hierarchical(rebalance_fields)`:
  1. **Intra-node phase**: Each MPI rank independently computes a local partition rebalancing among its own devices using throughput-proportional shares. No MPI communication occurs in this phase.
  2. `MPI_Barrier` — All ranks synchronize after local rebalancing.
  3. **Inter-node phase**: All ranks participate in `MPI_Allgatherv` to collect global per-device timing data. Rank 0 computes the globally optimal partition and broadcasts (`MPI_Bcast`) the new assignment. All ranks apply the global partition.

  Rank 0 emits diagnostic logs at each phase:
  ```
  [HIER] intra-node loads: ...
  [HIER] inter-node loads: ...
  [HIER] final partition: ...
  ```

- **`tests/test_hierarchical.cpp`** *(new)* — 6 unit tests covering: basic two-phase execution, barrier sequencing, intra-only path (single rank), empty field list, repeated invocation, and asymmetric device counts per rank.

**Test result**: **6/6 tests passing**.

---

### R4 — ERAD vs. HWTOPOLB Algorithm Comparison

**Goal**: Provide a rigorous, reproducible head-to-head comparison between the deterministic ERAD algorithm and the stochastic HWTOPOLB algorithm on representative scientific kernels.

**Implementation**:

- **`dcl/algorithms.hpp`** *(new)* — Contains three standalone, header-only functions within `namespace dcl`:

  - `erad_loads(times, partitions)` — Pure deterministic balancing. Computes per-device throughput as `element_count / kernel_time`, normalizes to a cumulative load vector `[0, 1]`.
  - `hwtopolb_loads(times, partitions, perturbation_factor)` — Stochastic augmentation of ERAD. Applies bounded uniform perturbation `delta_i ∈ [-p, +p]` to each device share, clamps to avoid negative shares, normalizes, and returns cumulative loads. Reduces to ERAD when `perturbation_factor == 0.0f`.
  - `apply_power_cap(loads, metrics, power_budget_watts)` — Post-processing step (see R5).

- **`benchmarks/kneighbor.cpp`** + **`benchmarks/kneighbor.cl`** *(new)* — Synthetic k-nearest-neighbor stencil benchmark (k=5, N=1,000,000 elements, 100 iterations). Each iteration executes the OpenCL stencil kernel across all available devices, then evaluates rebalancing every 10 steps under both ERAD and HWTOPOLB modes. Outputs per-iteration timing CSVs.

- **`benchmarks/leanmd.cpp`** + **`benchmarks/leanmd.cl`** *(new)* — Molecular dynamics Lennard-Jones pair-force benchmark. Simulates particle cutoff domain interactions (r⁻¹² − r⁻⁶) with spatially non-uniform density distributions, exercising the balancer under irregular workloads.

- **`benchmarks/compare_algos.sh`** *(new)* — Shell script that compiles both benchmarks (using `$MPI_LAUNCHER` or `mpirun`), runs each with `--algo erad` and `--algo hwtopolb` modes, collects CSV outputs, and prints a summary table.

- **`benchmarks/comparison_report.md`** *(new)* — 619-word analytical report covering:
  - Experimental results table (total time, rebalance count, average gain per algorithm/benchmark)
  - Philosophical analysis of deterministic vs. stochastic balancing
  - When stochastic perturbation helps (escaping local minima, desynchronizing interconnect bursts, robustness to noise)
  - When it hurts (BSP straggler penalties, oscillation, regular stencil predictability)
  - Practical guidance: prefer ERAD for structured stencils (`kNeighbor`); consider hybrid decaying-perturbation HWTOPOLB for particle methods (`LeanMD`)

- **`tests/test_algorithms.cpp`** *(new)* — 3 unit tests validating: ERAD produces valid monotone cumulative loads, HWTOPOLB with `perturbation_factor=0` matches ERAD output, and `apply_power_cap()` preserves load vector validity.

**Benchmark results** (measured locally in test environment via `compare_algos.sh` with N=1,000,000, 100 iterations; submit `job_hwtopolb.pbs` on cluster to collect real cluster numbers):

| Benchmark | Algorithm | Total Time (s) | Rebalances | Avg Gain (s) |
|:---|:---|:---:|:---:|:---:|
| `kneighbor` | ERAD | 0.547 | 9 | 0.000339 |
| `kneighbor` | HWTOPOLB ±10% | 0.562 | 9 | 0.000432 |
| `leanmd` | ERAD | 1.340 | 9 | 0.000733 |
| `leanmd` | HWTOPOLB ±10% | 1.388 | 9 | 0.000805 |

**Test result**: **3/3 tests passing**.

---

### R5 — Resource Contention & Power Consumption Model

**Goal**: Make the balancer aware of memory contention and thermal constraints, preventing assignment of work to already thermally saturated devices and respecting a configurable system power budget.

**Implementation**:

- **`dcl/types.hpp`** — `TopoMetrics` extended with:
  - `memory_contention_factor` — Per-device multiplier ≥ 1.0 representing relative memory bandwidth saturation (1.0 = no contention, 2.0 = half effective bandwidth)
  - `thermal_tdp_watts` — Per-device thermal design power in watts
  - `current_power_watts` — Per-device current measured power draw in watts

  `AutoBalancePolicy` extended with:
  - `use_contention_adjustment` (bool) — Enable contention-adjusted throughput calculation
  - `use_power_cap` (bool) — Enable power-capped load redistribution
  - `power_budget_watts` (double) — System-wide power budget ceiling (0 = unconstrained)

- **`dcl/topo_metrics_io.hpp`** — Added `adjusted_capacity(metrics, global_device, raw_throughput)`. Divides raw throughput by the device's `memory_contention_factor`, yielding a corrected effective capacity for use in load share computation.

- **`dcl/algorithms.hpp`** — `apply_power_cap(loads, metrics, power_budget_watts)`:
  1. Converts cumulative load vector to per-device share fractions.
  2. Computes effective power limits per device (the minimum of TDP and the budget-proportional cap).
  3. Identifies overloaded devices (`current_power_watts > effective_limit`) and scales their shares down proportionally.
  4. Redistributes freed load shares to underloaded devices, weighted by available thermal headroom; falls back to existing-share weighting or equal distribution when headroom data is absent.
  5. Returns a valid cumulative load vector (monotone non-decreasing, last element = 1.0f).

- **`dcl/runtime_impl.hpp`** — `maybe_rebalance_from_timings()` integrates both adjustments:
  - When `use_contention_adjustment` is set, throughput is divided by `memory_contention_factor` before computing shares.
  - When `use_power_cap` is set, the candidate load vector is passed through `apply_power_cap()` before being applied.

- **`tests/test_contention_power.cpp`** *(new)* — 4 unit tests: contention factor reduces effective throughput proportionally, `apply_power_cap()` reduces overloaded device shares, `apply_power_cap()` preserves load vector validity (monotone, ends at 1.0), and passthrough when power metrics are absent.

**Test result**: **4/4 tests passing**.

---

## 4. File Inventory

### New Files

| File | Purpose |
|:---|:---|
| `dcl/topo_metrics_io.hpp` | JSON serialization/deserialization for `TopoMetrics`; NUMA distance query; migration cost estimation; contention-adjusted capacity |
| `dcl/algorithms.hpp` | `erad_loads()`, `hwtopolb_loads()`, `apply_power_cap()` algorithm implementations |
| `topo_probe.cpp` | MPI+OpenCL standalone tool to measure and emit real cluster PCIe/MPI topology metrics as JSON |
| `tests/test_topo_metrics.cpp` | 7 unit tests for R1 (topology metrics I/O) |
| `tests/test_numa_cost.cpp` | 4 unit tests for R2 (NUMA migration cost gating) |
| `tests/test_hierarchical.cpp` | 6 unit tests for R3 (hierarchical balancing) |
| `tests/test_algorithms.cpp` | 3 unit tests for R4 (ERAD/HWTOPOLB algorithms) |
| `tests/test_contention_power.cpp` | 4 unit tests for R5 (contention and power cap) |
| `benchmarks/kneighbor.cpp` | k-nearest-neighbor stencil benchmark driver (R4) |
| `benchmarks/kneighbor.cl` | OpenCL kernel for kNeighbor benchmark |
| `benchmarks/leanmd.cpp` | Lennard-Jones molecular dynamics benchmark driver (R4) |
| `benchmarks/leanmd.cl` | OpenCL kernel for LeanMD benchmark |
| `benchmarks/compare_algos.sh` | Automation script: compile, run, and summarize ERAD vs. HWTOPOLB results |
| `benchmarks/comparison_report.md` | 619-word analytical comparison report (R4) |
| `job_hwtopolb.pbs` | PBS cluster job script: loads modules, runs `topo_probe`, executes benchmark comparison |

### Edited Files

| File | Changes |
|:---|:---|
| `dcl/types.hpp` | Added `struct TopoMetrics` (R1); added `numa_distance`, `device_numa_node` fields (R2); added `memory_contention_factor`, `thermal_tdp_watts`, `current_power_watts` fields (R5); added `BalanceMode::hierarchical` (R3); added `numa_cost_gain_ratio_threshold`, `use_contention_adjustment`, `use_power_cap`, `power_budget_watts` to `AutoBalancePolicy` (R2, R5) |
| `dcl/runtime.hpp` | Added `set_topo_metrics()`, `topo_metrics()`, `maybe_rebalance_hierarchical()`, `set_simulated_devices_count()`, `set_simulated_times()`, `clear_simulated_times()` to the `Runtime` public API |
| `dcl/runtime.cpp` | Implemented public-API method bodies forwarding to `Impl` |
| `dcl/runtime_impl.hpp` | Integrated NUMA cost gating into `maybe_rebalance_from_timings()` (R2); implemented `maybe_rebalance_hierarchical()` two-phase logic (R3); integrated contention adjustment and power cap into rebalancing pipeline (R5) |
| `tests/test_hierarchical.cpp` | Fixed: `[HIER]` log assertions and PASS/summary messages gated on `rank == 0` only; `maybe_rebalance_hierarchical()` emits logs exclusively from rank 0 so rank 1 captured no output and aborted |
| `.gitignore` | Expanded to exclude `*.out` binaries, `Results/`, benchmark CSVs (`benchmarks/*.csv`), topology JSON outputs (`topo_metrics_*.json`, `test_probe.json`), PBS log files (`*.pbs.o*`, `*.pbs.e*`), `.agents/`, `ORIGINAL_REQUEST.md` |


### Removed Files

No source files were removed. Binary artifacts (`*.out`, `*.o`) were never tracked in git (now explicitly excluded via `.gitignore`).

---

## 5. Directory Structure

```
HARMONY/
├── dcl/                          # Core library headers (dcl namespace, C++20)
│   ├── types.hpp                 # All fundamental data structures (TopoMetrics, BalanceMode, AutoBalancePolicy, …)
│   ├── runtime.hpp               # Runtime class public API
│   ├── runtime.cpp               # Runtime public method implementations
│   ├── runtime_impl.hpp          # Runtime::Impl — full balancing logic (template-heavy, header-only)
│   ├── runtime_impl.cpp          # runtime_impl.cpp stub (explicit instantiation guard)
│   ├── topo_metrics_io.hpp       # JSON I/O, NUMA query, migration cost, capacity adjustment (NEW)
│   └── algorithms.hpp            # erad_loads(), hwtopolb_loads(), apply_power_cap() (NEW)
│
├── tests/                        # Unit tests (standalone, no external framework)
│   ├── test_topo_metrics.cpp     # R1 — 7 tests
│   ├── test_numa_cost.cpp        # R2 — 4 tests
│   ├── test_hierarchical.cpp     # R3 — 6 tests
│   ├── test_algorithms.cpp       # R4 — 3 tests
│   └── test_contention_power.cpp # R5 — 4 tests
│
├── benchmarks/                   # Benchmark drivers and analysis
│   ├── kneighbor.cpp             # kNeighbor stencil benchmark (NEW)
│   ├── kneighbor.cl              # kNeighbor OpenCL kernel (NEW)
│   ├── leanmd.cpp                # LeanMD molecular dynamics benchmark (NEW)
│   ├── leanmd.cl                 # LeanMD OpenCL kernel (NEW)
│   ├── compare_algos.sh          # ERAD vs. HWTOPOLB automation script (NEW)
│   └── comparison_report.md      # Algorithmic comparison analysis (NEW)
│
├── topo_probe.cpp                # Cluster topology measurement tool (NEW)
├── job_hwtopolb.pbs              # PBS job: topo_probe + benchmark comparison (NEW)
├── .gitignore                    # Expanded to cover all generated artifacts
│
├── main.cpp                      # Example application entry point
├── kernels.cl                    # Primary simulation OpenCL kernels
├── *.cl                          # Additional OpenCL kernels (stencil, ocean, jacobi, …)
├── job_dcl_np{1,2,4}_metrics.pbs # PBS jobs for performance profiling at various MPI counts
└── profiling.cpp                 # Standalone profiling tool
```

---

## 6. Building and Running

### Prerequisites

| Dependency | Minimum Version | Notes |
|:---|:---|:---|
| C++ compiler | GCC 13 / Clang 17 | C++20 required (`-std=c++20`) |
| MPI | OpenMPI 5.0+ or MPICH 4+ | C++ bindings disabled (`OMPI_SKIP_MPICXX`) |
| OpenCL | 3.0 | ICD loader (`-lOpenCL`) |
| OpenCL headers | 3.0 | `-DCL_TARGET_OPENCL_VERSION=300` |

### Compiling the Unit Tests

Each test is a self-contained translation unit. Compile with `mpic++`:

```bash
# R1 — Topology metrics I/O
mpic++ -std=c++20 -Wall -Wextra -O2 \
  tests/test_topo_metrics.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o tests/test_topo_metrics.out

# R2 — NUMA migration cost
mpic++ -std=c++20 -Wall -Wextra -O2 \
  tests/test_numa_cost.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o tests/test_numa_cost.out

# R3 — Hierarchical balancing
mpic++ -std=c++20 -Wall -Wextra -O2 \
  tests/test_hierarchical.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o tests/test_hierarchical.out

# R4 — Algorithm comparison
mpic++ -std=c++20 -Wall -Wextra -O2 \
  tests/test_algorithms.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o tests/test_algorithms.out

# R5 — Contention and power cap
mpic++ -std=c++20 -Wall -Wextra -O2 \
  tests/test_contention_power.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o tests/test_contention_power.out
```

### Running Tests

```bash
mpirun -n 1 ./tests/test_topo_metrics.out
mpirun -n 1 ./tests/test_numa_cost.out
mpirun -n 2 ./tests/test_hierarchical.out
mpirun -n 1 ./tests/test_algorithms.out
mpirun -n 1 ./tests/test_contention_power.out
```

### Compiling the Topology Probe

```bash
mpic++ -std=c++20 -Wall -Wextra -O2 \
  topo_probe.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o topo_probe.out
```

Run on the cluster to generate `topo_metrics_cluster.json`:

```bash
mpirun -n 4 ./topo_probe.out --output-json topo_metrics_cluster.json
```

### Compiling the Benchmarks

```bash
# kNeighbor
mpic++ -std=c++20 -Wall -Wextra -O2 \
  benchmarks/kneighbor.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o kneighbor.out

# LeanMD
mpic++ -std=c++20 -Wall -Wextra -O2 \
  benchmarks/leanmd.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o leanmd.out
```

Or use the automation script (sets `MPI_LAUNCHER` if needed):

```bash
export MPI_LAUNCHER="mpirun"
bash benchmarks/compare_algos.sh
```

### Injecting Topology Metrics at Runtime

```cpp
#include "dcl/runtime.hpp"
#include "dcl/topo_metrics_io.hpp"

// Load measured metrics from JSON
dcl::TopoMetrics metrics = dcl::load_topo_metrics("topo_metrics_cluster.json");

// Inject into runtime
runtime.set_topo_metrics(metrics);

// Configure NUMA-aware policy
dcl::AutoBalancePolicy policy;
policy.mode = dcl::BalanceMode::dynamic_threshold;
policy.threshold = 0.05f;
policy.numa_cost_gain_ratio_threshold = 0.50;  // skip if cost > 50% of gain
policy.use_contention_adjustment = true;
policy.use_power_cap = true;
policy.power_budget_watts = 500.0;

step.with_balance(policy);
```

---

## 7. Running on a PBS Cluster

The provided `job_hwtopolb.pbs` script automates the full evaluation workflow:

```bash
qsub job_hwtopolb.pbs
```

**Resource allocation**: `select=2:ncpus=16:mpiprocs=16` (2 nodes × 16 MPI ranks each = 32 total)

**Modules loaded**:
- `gcc/13.2.0`
- `openmpi5/5.0.5`

**Execution steps**:
1. Compiles `topo_probe.cpp` and runs it with `mpirun`, writing `topo_metrics_cluster.json`
2. Executes `benchmarks/compare_algos.sh` (compiles and runs kNeighbor + LeanMD under both ERAD and HWTOPOLB)

Output is written to `hwtopolb_eval.out` (stdout+stderr merged via `#PBS -j oe`).

---

## 8. Test Results & Verification

All unit tests were compiled with `-std=c++20 -Wall -Wextra -Wpedantic -Wno-unused-parameter` and executed via `mpirun`.

| Test Suite | Requirement | Tests | Result |
|:---|:---|:---:|:---:|
| `test_topo_metrics` | R1 — Topology metrics injection | 7 | ✅ 7/7 |
| `test_numa_cost` | R2 — NUMA migration cost gating | 4 | ✅ 4/4 |
| `test_hierarchical` | R3 — Hierarchical balancing | 6 | ✅ 6/6 |
| `test_algorithms` | R4 — ERAD/HWTOPOLB algorithms | 3 | ✅ 3/3 |
| `test_contention_power` | R5 — Contention & power cap | 4 | ✅ 4/4 |
| **Total** | | **24** | **✅ 24/24** |

---

## 9. Static Analysis Baseline

Two mandatory static checks are run before any functional testing:

**Syntax check** (full C++20 compilation pipeline, no linking):
```bash
mpic++ -std=c++20 -Wall -Wextra -Wpedantic -Wno-unused-parameter \
  -fsyntax-only dcl/runtime_impl.hpp
```
→ **EXIT 0, 0 warnings**

**Cppcheck** (static analysis, all checks enabled):
```bash
cppcheck --enable=all --std=c++20 --error-exitcode=1 \
  --suppress=missingIncludeSystem dcl/
```
→ **EXIT 0, 0 violations**

---

*HARMONY — Distributed Computing Laboratory (DCL) · `namespace dcl` · C++20 · OpenCL 3.0 · MPI*
