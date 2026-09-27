#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX 1
#endif

#include <mpi.h>
#if __has_include("dcl/runtime.hpp")
#include "dcl/runtime.hpp"
#include "dcl/runtime.cpp"
#include "dcl/algorithms.hpp"
#else
#include "../dcl/runtime.hpp"
#include "../dcl/runtime.cpp"
#include "../dcl/algorithms.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

struct StepRecord {
    int iteration{0};
    double step_time_s{0.0};
    std::string action;
    int rebalanced{0};
    double rebalance_gain{0.0};
};

int main(int argc, char** argv) {
    int k = 5;
    std::size_t n = 1000000;
    int iterations = 100;
    std::string balance_algo = "erad";
    float perturbation = 0.10f;
    int balance_interval = 10;
    std::string output_csv = "kneighbor.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--k" && i + 1 < argc) {
            k = std::atoi(argv[++i]);
        } else if (arg == "--n" && i + 1 < argc) {
            n = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (arg == "--balance-algo" && i + 1 < argc) {
            balance_algo = argv[++i];
        } else if (arg == "--perturbation" && i + 1 < argc) {
            perturbation = std::strtof(argv[++i], nullptr);
        } else if (arg == "--balance-interval" && i + 1 < argc) {
            balance_interval = std::atoi(argv[++i]);
        } else if (arg == "--output-csv" && i + 1 < argc) {
            output_csv = argv[++i];
        }
    }

    dcl::Runtime runtime = dcl::Runtime::create(argc, argv);
    const int rank = runtime.rank();

    // Check OpenCL hardware availability
    bool opencl_available = false;
    try {
        cl_uint n_platforms = 0;
        cl_int cl_res = clGetPlatformIDs(0, nullptr, &n_platforms);
        if (cl_res == CL_SUCCESS && n_platforms > 0) {
            runtime.discover_devices({dcl::DeviceKind::all, 0});
            if (!runtime.devices().empty()) {
                opencl_available = true;
            }
        }
    } catch (...) {
        opencl_available = false;
    }

    if (!opencl_available) {
        // CPU simulation fallback: 2 simulated devices with heterogeneous performance
        runtime.set_simulated_devices_count(2);
    }

    dcl::PartitionSpec ps;
    ps.global_elements = n;
    ps.units_per_element = 1;
    ps.bytes_per_unit = sizeof(float);
    ps.granularity = 1;
    runtime.set_partition(ps);

    // Setup load calculation algorithm callback
    using LoadAlgoFunc = std::function<std::vector<float>(
        const std::vector<double>& times,
        const std::vector<dcl::DevicePartition>& partitions
    )>;

    LoadAlgoFunc algo_fn;
    if (balance_algo == "erad") {
        algo_fn = [](const std::vector<double>& t, const std::vector<dcl::DevicePartition>& p) {
            return dcl::erad_loads(t, p);
        };
    } else if (balance_algo == "hwtopolb") {
        algo_fn = [perturbation](const std::vector<double>& t, const std::vector<dcl::DevicePartition>& p) {
            return dcl::hwtopolb_loads(t, p, perturbation);
        };
    } else {
        if (rank == 0) {
            std::cerr << "Unknown balance-algo: " << balance_algo << "\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Allocate synthetic data
    std::vector<float> in_data(n, 1.0f);
    std::vector<float> out_data(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        in_data[i] = static_cast<float>(i % 100) * 0.01f;
    }

    std::vector<StepRecord> records;
    records.reserve(static_cast<std::size_t>(iterations));

    std::vector<double> prev_device_times;
    const auto total_bench_start = std::chrono::steady_clock::now();

    for (int it = 0; it < iterations; ++it) {
        bool do_rebalance = (balance_interval > 0 && it > 0 && (it % balance_interval == 0) && !prev_device_times.empty());
        double gain = 0.0;

        if (do_rebalance) {
            const auto& old_parts = runtime.partitions();
            double total_cap = 0.0;
            std::vector<double> capacity(old_parts.size(), 0.0);
            double t_max_current = 0.0;

            for (std::size_t i = 0; i < old_parts.size(); ++i) {
                if (prev_device_times[i] > t_max_current) {
                    t_max_current = prev_device_times[i];
                }
                if (old_parts[i].element_count > 0 && prev_device_times[i] > 1e-12) {
                    capacity[i] = static_cast<double>(old_parts[i].element_count) / prev_device_times[i];
                    total_cap += capacity[i];
                }
            }

            std::vector<float> new_loads = algo_fn(prev_device_times, runtime.partitions());
            runtime.rebalance_to(new_loads);

            const auto& new_parts = runtime.partitions();
            double t_max_projected = t_max_current;
            if (total_cap > 0.0) {
                t_max_projected = 0.0;
                for (std::size_t i = 0; i < new_parts.size(); ++i) {
                    if (new_parts[i].element_count == 0) continue;
                    if (i >= capacity.size() || capacity[i] <= 0.0) {
                        t_max_projected = t_max_current;
                        break;
                    }
                    const double proj = static_cast<double>(new_parts[i].element_count) / capacity[i];
                    if (proj > t_max_projected) {
                        t_max_projected = proj;
                    }
                }
            }
            gain = std::max(0.0, t_max_current - t_max_projected);
        }

        const auto& partitions = runtime.partitions();
        std::vector<double> device_times(partitions.size(), 0.0);

        // Execute synthetic k-nearest-neighbor stencil workload per partition
        for (std::size_t p_idx = 0; p_idx < partitions.size(); ++p_idx) {
            const auto& p = partitions[p_idx];
            const std::size_t offset = p.global_offset;
            const std::size_t count = p.element_count;
            const int dev_id = p.device_global_index;

            // Model heterogeneous hardware: dev 0 has 2x compute cost per element
            const int work_passes = (dev_id == 0 ? 2 : 1);

            const auto t0 = std::chrono::steady_clock::now();
            const float inv_k = 1.0f / static_cast<float>(2 * k + 1);

            for (int pass = 0; pass < work_passes; ++pass) {
                for (std::size_t i = offset; i < offset + count; ++i) {
                    float sum = 0.0f;
                    for (int o = -k; o <= k; ++o) {
                        const long long idx = static_cast<long long>(i) + o;
                        if (idx >= 0 && idx < static_cast<long long>(n)) {
                            sum += in_data[static_cast<std::size_t>(idx)];
                        }
                    }
                    out_data[i] = sum * inv_k;
                }
            }
            const auto t1 = std::chrono::steady_clock::now();
            device_times[p_idx] = std::chrono::duration<double>(t1 - t0).count();
        }

        const double t_max_step = *std::max_element(device_times.begin(), device_times.end());

        if (do_rebalance) {
            records.push_back({it, t_max_step, "rebalance", 1, gain});
        } else {
            records.push_back({it, t_max_step, "compute", 0, 0.0});
        }

        prev_device_times = device_times;
        runtime.set_simulated_times(device_times);
    }

    const auto total_bench_end = std::chrono::steady_clock::now();
    const double total_duration_s = std::chrono::duration<double>(total_bench_end - total_bench_start).count();

    if (rank == 0) {
        std::ofstream csv(output_csv);
        if (!csv.is_open()) {
            std::cerr << "Failed to open output CSV: " << output_csv << "\n";
            MPI_Finalize();
            return 1;
        }

        csv << "iteration,step_time_s,action,rebalanced,rebalance_gain\n";
        int num_rebalances = 0;
        double sum_gain = 0.0;

        for (const auto& rec : records) {
            csv << rec.iteration << ","
                << std::fixed << std::setprecision(6) << rec.step_time_s << ","
                << rec.action << ","
                << rec.rebalanced << ","
                << std::fixed << std::setprecision(6) << rec.rebalance_gain << "\n";
            if (rec.rebalanced == 1) {
                num_rebalances++;
                sum_gain += rec.rebalance_gain;
            }
        }
        csv.close();

        const double avg_gain = (num_rebalances > 0) ? (sum_gain / num_rebalances) : 0.0;
        std::cout << "[kneighbor] algo=" << balance_algo
                  << " total_time_s=" << total_duration_s
                  << " num_rebalances=" << num_rebalances
                  << " avg_rebalance_gain=" << avg_gain << "\n";
    }

    MPI_Finalize();
    return 0;
}
