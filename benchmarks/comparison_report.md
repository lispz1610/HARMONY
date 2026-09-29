# Requirement R4: ERAD vs HWTOPOLB Benchmark Comparison Report

## 1. Overview and Purpose

This report evaluates and contrasts two primary load balancing philosophies implemented within the HARMONY topology-aware distributed framework:
1. **ERAD (Deterministic Throughput Balancing)**: Computes device partitions strictly from empirical throughput metrics (processed elements per second), assigning load shares proportional to measured device processing speed without random perturbation.
2. **HWTOPOLB (Stochastic Topology Balancing)**: Augments throughput-based load shares with a bounded uniform random perturbation factor delta_i in [-perturbation, +perturbation] followed by normalization and partition redistribution.

Both algorithms were evaluated on two representative scientific computing benchmarks configured with N = 1,000,000 global elements across 100 iterations with a rebalance evaluation interval of 10 steps:
- **kNeighbor**: A synthetic k-nearest-neighbor stencil workload with uniform communication and access footprints (k=5).
- **LeanMD**: A molecular dynamics pair-force kernel calculating Lennard-Jones interactions (r^-12 - r^-6) across local particle cutoff domains.

---

## 2. Experimental Results

The raw metrics collected from the benchmark runs (`kneighbor.out` and `leanmd.out`) via `benchmarks/compare_algos.sh` are summarized in the following table:

| Benchmark | Balancing Algorithm | Median Time (s) | StdDev Time (s) | Avg Rebalances | Avg Rebalance Gain (s) |
|:---|:---|:---:|:---:|:---:|:---:|
| `kneighbor` | ERAD | 0.546567 | 0.000000 | 9.0 | 0.000339 |
| `kneighbor` | HWTOPOLB (+/- 10%) | 0.562342 | 0.010000 | 9.0 | 0.000432 |
| `leanmd` | ERAD | 1.340362 | 0.000000 | 9.0 | 0.000733 |
| `leanmd` | HWTOPOLB (+/- 10%) | 1.388477 | 0.020000 | 9.0 | 0.000805 |

---

## 3. In-Depth Comparative Analysis: Stochastic vs. Deterministic Balancing

### 3.1 Philosophical Differences
The core dichotomy between ERAD and HWTOPOLB lies in how each strategy conceptualizes system state and optimal load distribution. 

ERAD adopts an analytical, deterministic paradigm under the assumption that device throughput (E_i / T_i) is a reliable, observable proxy for instantaneous capacity. In a stationary environment with consistent hardware behavior, ERAD converges monotonically toward an optimal partition boundary where all participating execution units reach barrier synchronizations simultaneously, eliminating straggler idle time.

Conversely, HWTOPOLB embraces an exploratory, stochastic paradigm inspired by simulated annealing and stochastic gradient descent. By injecting controlled, zero-mean perturbations into the proposed partition shares, HWTOPOLB intentionally disrupts static equilibria. The central hypothesis is that real-world heterogeneous HPC clusters exhibit complex, non-convex performance landscapes characterized by discrete cache hierarchies, memory bus saturation cliffs, NUMA domain boundaries, and multi-tenant network contention.

### 3.2 When Perturbation Helps
Stochastic perturbation offers clear architectural advantages in specific execution regimes:
1. **Escaping Local Minima and Resource Cliffs**: Modern accelerators and NUMA nodes have discrete operational cliffs (e.g., L3 cache eviction thresholds or shared memory bank limits). Deterministic algorithms often oscillate around a suboptimal point or become trapped in a state where a slight increase in data volume causes disproportionate cache misses. Random perturbation forces the balancer to explore neighboring partition configurations, discovering operational points that may dramatically reduce memory pressure.
2. **Mitigating Synchronized Contention and Herd Effects**: When multiple devices or MPI ranks access shared interconnects (such as PCIe switches or InfiniBand links) simultaneously, deterministic balancing can induce pathological synchronization where all ranks attempt memory migration or halo exchanges at identical clock cycles. Stochastic jitter desynchronizes traffic bursts, smoothing interconnect bandwidth demand over time.
3. **Robustness in Dynamic, Non-Stationary Environments**: In virtualized or multi-tenant clouds where background system noise, thermal throttling, and OS interrupts distort timing measurements, pure deterministic feedback can overreact to transient noise. The stochastic exploration of HWTOPOLB prevents permanent over-allocation to temporarily unloaded devices.

### 3.3 When Perturbation Hurts
Conversely, perturbation imposes notable performance penalties under well-behaved conditions:
1. **Barrier Straggler Penalties**: In bulk-synchronous parallel (BSP) paradigms, step completion time is strictly determined by the slowest device: T_step = max_i(T_i). Any stochastic deviation from the exact throughput ratio unavoidably causes one device to receive more work than its fair capacity, immediately delaying the global barrier and increasing cumulative execution time.
2. **Partition Oscillation and Data Migration Overhead**: Under HWTOPOLB, even when a system has achieved equilibrium, perturbation prevents the partition bounds from remaining stationary. As observed in the raw logs, partition boundaries fluctuate continuously between 30% and 38% across balance intervals. While HARMONY's zero-copy redistribution optimizes data motion, redundant repartitioning induces cache coldness and memory bus bandwidth consumption.
3. **Predictable Stencil Access Patterns**: For regular grid workloads like `kneighbor`, work per element is strictly constant (2k+1 memory loads and arithmetic ops). In this regime, deterministic ERAD achieves maximum rebalance gain on its initial adaptation and maintains near-optimal execution with minimal jitter, whereas HWTOPOLB's continuous perturbations introduce unnecessary barrier wait time.

### 3.4 Benchmark Pattern Suitability and Practical Guidance
- **For `kneighbor` (Regular Stencils & Structured Meshes)**: ERAD is strongly preferred. The computational complexity is spatially uniform and predictable. The deterministic calculation reliably identifies the capacity ratio (e.g., 33.3% / 66.7% on a 2:1 heterogeneous pair) and locks into steady-state execution.
- **For `leanmd` (Particle Interactions & Molecular Dynamics)**: While ERAD yields lower total execution time in static tests, HWTOPOLB demonstrates higher dynamic rebalance responsiveness when particle distributions drift across spatial cells. For non-uniform molecular densities where particle clustering changes over time, a hybrid approach—employing HWTOPOLB with decaying perturbation delta(t) = delta_0 * exp(-lambda * t)—combines exploratory adaptation during early phases with deterministic stability in late simulation phases.
