#include "benchmark_common.hpp"
#include "harmony/runtime.hpp"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#ifndef KERNEL_FILE
#define KERNEL_FILE "benchmarks/leanmd.cl"
#endif

namespace {

void run_simulated_partition_work(const std::vector<harmony::DevicePartition>& partitions,
                                  int rank,
                                  int cutoff,
                                  const std::vector<float>& pos_x,
                                  const std::vector<float>& pos_y,
                                  const std::vector<float>& pos_z,
                                  std::vector<float>& force_out,
                                  std::vector<double>& local_device_times) {
    const std::size_t n = pos_x.size();
    for (std::size_t p_idx = 0; p_idx < partitions.size(); ++p_idx) {
        const auto& p = partitions[p_idx];
        if (p.owning_rank != rank || p.element_count == 0) continue;

        const std::size_t offset = p.global_offset;
        const std::size_t count = p.element_count;
        const int work_passes = (p.device_global_index == 0 ? 2 : 1);

        const auto t0 = std::chrono::steady_clock::now();
        for (int pass = 0; pass < work_passes; ++pass) {
            for (std::size_t i = offset; i < offset + count; ++i) {
                const float xi = pos_x[i];
                const float yi = pos_y[i];
                const float zi = pos_z[i];
                float f_total = 0.0f;

                const std::size_t s =
                    (i > static_cast<std::size_t>(cutoff)) ? (i - static_cast<std::size_t>(cutoff)) : 0;
                const std::size_t e = std::min(n - 1, i + static_cast<std::size_t>(cutoff));
                for (std::size_t j = s; j <= e; ++j) {
                    if (i == j) continue;
                    const float dx = xi - pos_x[j];
                    const float dy = yi - pos_y[j];
                    const float dz = zi - pos_z[j];
                    const float r2 = dx * dx + dy * dy + dz * dz + 1.0e-6f;
                    const float r2inv = 1.0f / r2;
                    const float r6inv = r2inv * r2inv * r2inv;
                    f_total += r2inv * r6inv * (2.0f * r6inv - 1.0f);
                }
                force_out[i] = f_total;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        local_device_times[p_idx] = std::chrono::duration<double>(t1 - t0).count();
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        harmony::Runtime runtime = harmony::Runtime::create(argc, argv);
        const int rank = runtime.rank();

        harmony_bench::BenchmarkOptions opt;
        try {
            opt = harmony_bench::parse_options(argc, argv, "leanmd", /*default_k=*/4);
        } catch (const std::exception& e) {
            if (rank == 0) std::cerr << "Argument error: " << e.what() << "\n";
            return 1;
        }

        const bool opencl_available = harmony_bench::init_benchmark_devices(runtime, opt);
        runtime.set_verbose(opt.verbose);
        runtime.set_numa_cost_gain_ratio_threshold(opt.numa_threshold);
        if (!opt.topo_json.empty()) {
            runtime.set_topo_metrics(harmony::load_topo_metrics(opt.topo_json));
        }

        runtime.set_partition({opt.n, 1, sizeof(float), 1});

        std::vector<float> pos_x(opt.n), pos_y(opt.n), pos_z(opt.n), force_out(opt.n, 0.0f);
        for (std::size_t i = 0; i < opt.n; ++i) {
            const float fi = static_cast<float>(i);
            pos_x[i] = fi * 0.1f;
            pos_y[i] = std::sin(fi * 0.05f);
            pos_z[i] = std::cos(fi * 0.05f);
        }

        auto h_pos_x = runtime.create_field({"pos_x", opt.n, 1, sizeof(float), harmony::BufferUsage::read_only,
                                             pos_x.data(), harmony::RedistributionDependency::proportional});
        auto h_pos_y = runtime.create_field({"pos_y", opt.n, 1, sizeof(float), harmony::BufferUsage::read_only,
                                             pos_y.data(), harmony::RedistributionDependency::proportional});
        auto h_pos_z = runtime.create_field({"pos_z", opt.n, 1, sizeof(float), harmony::BufferUsage::read_only,
                                             pos_z.data(), harmony::RedistributionDependency::proportional});
        auto h_force = runtime.create_field({"force_out", opt.n, 1, sizeof(float), harmony::BufferUsage::write_only,
                                             force_out.data(), harmony::RedistributionDependency::proportional});

        const harmony::AutoBalancePolicy policy = harmony_bench::build_balance_policy(opt);
        auto step_builder = runtime.step("leanmd_step");

        harmony::KernelHandle h_kernel;
        harmony::KernelBinding binding;
        if (opencl_available) {
            const std::string kernel_path =
                harmony_bench::resolve_kernel_path(opt.kernel_file, KERNEL_FILE, "leanmd.cl");
            h_kernel = runtime.create_kernel({kernel_path, "leanmd_pair_force", ""});
            binding = runtime.bind(h_kernel)
                .arg(0, h_pos_x, harmony::Access::read)
                .arg(1, h_pos_y, harmony::Access::read)
                .arg(2, h_pos_z, harmony::Access::read)
                .arg(3, h_force, harmony::Access::write)
                .arg(4, harmony::ScalarArg(static_cast<int>(opt.n)))
                .arg(5, harmony::ScalarArg(opt.k))
                .build();
            step_builder.invoke(binding, harmony::LaunchGeometry{0, opt.n, std::nullopt});
            if (opt.k > 0) {
                step_builder.with_halo_exchange({static_cast<std::size_t>(opt.k), {h_pos_x, h_pos_y, h_pos_z}});
            }
        }
        step_builder.with_balance(policy)
            .tag_field(h_force, harmony::StepFieldRole::rebalance_source)
            .synchronize_at_end(true);
        const harmony::ExecutionStep step = step_builder.build();

        std::vector<double> step_times(static_cast<std::size_t>(opt.iterations), 0.0);
        std::vector<double> sim_window_times;
        bool sim_static_attempted = false;

        const auto total_bench_start = std::chrono::steady_clock::now();
        runtime.start_timing();

        for (int it = 0; it < opt.iterations; ++it) {
            const auto step_start = std::chrono::steady_clock::now();

            if (!opencl_available) {
                const auto& partitions = runtime.partitions();
                if (sim_window_times.size() != partitions.size()) {
                    sim_window_times.assign(partitions.size(), 0.0);
                }
                std::vector<double> local_device_times(partitions.size(), 0.0);
                run_simulated_partition_work(
                    partitions, rank, opt.k, pos_x, pos_y, pos_z, force_out, local_device_times);
                std::vector<double> iter_device_times(partitions.size(), 0.0);
                MPI_Allreduce(local_device_times.data(), iter_device_times.data(),
                              static_cast<int>(partitions.size()), MPI_DOUBLE, MPI_MAX, runtime.communicator());
                for (std::size_t i = 0; i < partitions.size(); ++i) {
                    sim_window_times[i] += iter_device_times[i];
                }
                runtime.set_simulated_times(sim_window_times);
            }

            runtime.execute(step);

            if (!opencl_available &&
                harmony_bench::should_reset_sim_window(policy, it + 1, opt.iterations, sim_static_attempted)) {
                std::fill(sim_window_times.begin(), sim_window_times.end(), 0.0);
            }

            const auto step_end = std::chrono::steady_clock::now();
            step_times[static_cast<std::size_t>(it)] =
                std::chrono::duration<double>(step_end - step_start).count();
        }

        const double loop_time_s = runtime.stop_timing();

        // Post-loop host verification, decoupled from loop_time_s.
        if (opencl_available) {
            runtime.gather(h_force, force_out.data(), opt.n * sizeof(float));
        }

        int local_mismatch = 0;
        const auto& partitions = runtime.partitions();
        for (const auto& part : partitions) {
            if (!opencl_available && part.owning_rank != rank) continue;
            if (opencl_available && rank != 0) continue;
            for (std::size_t i = part.global_offset; i < part.global_offset + part.element_count; ++i) {
                const float expected = harmony_bench::leanmd_ref_at(pos_x, pos_y, pos_z, i, opt.k);
                if (!harmony_bench::verify_finite_and_close(force_out[i], expected, 1e-4f, 1e-4f)) {
                    std::cerr << "LeanMD verification failed at index " << i
                              << ": actual=" << force_out[i]
                              << " reference=" << expected << "\n";
                    local_mismatch = 1;
                    break;
                }
            }
            if (local_mismatch != 0) break;
        }

        int any_mismatch = 0;
        MPI_Allreduce(&local_mismatch, &any_mismatch, 1, MPI_INT, MPI_MAX, runtime.communicator());
        if (any_mismatch != 0) {
            return 1;
        }

        const auto total_bench_end = std::chrono::steady_clock::now();
        const double local_duration_s =
            std::chrono::duration<double>(total_bench_end - total_bench_start).count();
        double total_duration_s = 0.0;
        MPI_Allreduce(&local_duration_s, &total_duration_s, 1, MPI_DOUBLE, MPI_MAX, runtime.communicator());

        if (!harmony_bench::write_benchmark_outputs(
                runtime, "leanmd", opt, step_times, total_duration_s, loop_time_s, opencl_available)) {
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[leanmd] error: " << e.what() << "\n";
        int initialized = 0;
        int finalized = 0;
        MPI_Initialized(&initialized);
        MPI_Finalized(&finalized);
        if (initialized && !finalized) MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
