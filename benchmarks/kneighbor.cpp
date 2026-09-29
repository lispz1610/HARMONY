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
#include <optional>
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
    unsigned int seed = 0;

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
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<unsigned int>(std::stoul(argv[++i]));
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
        algo_fn = [perturbation, seed](const std::vector<double>& t, const std::vector<dcl::DevicePartition>& p) {
            return dcl::hwtopolb_loads(t, p, perturbation, seed);
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

    dcl::FieldHandle h_in, h_out;
    dcl::KernelHandle h_kernel;
    dcl::KernelBinding binding;
    if (opencl_available) {
        dcl::FieldSpec f_in{"in_data", n, 1, sizeof(float), dcl::BufferUsage::read_only, in_data.data(), dcl::RedistributionDependency::proportional};
        dcl::FieldSpec f_out{"out_data", n, 1, sizeof(float), dcl::BufferUsage::write_only, out_data.data(), dcl::RedistributionDependency::proportional};
        h_in = runtime.create_field(f_in);
        h_out = runtime.create_field(f_out);

        dcl::KernelSpec k_spec{"benchmarks/kneighbor.cl", "kneighbor_stencil", ""};
        h_kernel = runtime.create_kernel(k_spec);

        binding = runtime.bind(h_kernel)
            .arg(0, h_in)
            .arg(1, h_out)
            .arg(2, dcl::ScalarArg(static_cast<int>(n)))
            .arg(3, dcl::ScalarArg(k))
            .build();
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

            std::vector<float> new_loads;
            if (rank == 0) {
                new_loads = algo_fn(prev_device_times, runtime.partitions());
            }
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
        std::vector<double> local_device_times(partitions.size(), 0.0);

        const auto t0 = std::chrono::steady_clock::now();

        if (opencl_available) {
            auto builder = runtime.step("kneighbor_step");
            for (std::size_t p_idx = 0; p_idx < partitions.size(); ++p_idx) {
                const auto& p = partitions[p_idx];
                if (p.owning_rank == rank && p.element_count > 0) {
                    int work_passes = (p.device_global_index == 0 ? 2 : 1);
                    for(int pass=0; pass < work_passes; ++pass) {
                        builder.invoke(binding, dcl::LaunchGeometry{p.global_offset, p.element_count, std::nullopt});
                    }
                }
            }
            builder.synchronize_at_end(true);
            runtime.execute(builder.build());
            
            for (std::size_t p_idx = 0; p_idx < partitions.size(); ++p_idx) {
                const auto& p = partitions[p_idx];
                if (p.owning_rank == rank && p.local_index >= 0 && p.local_index < static_cast<int>(runtime.device_timings().size())) {
                    local_device_times[p_idx] = runtime.device_timings()[p.local_index].kernel_seconds_last;
                }
            }
        } else {
            // CPU fallback simulation
            for (std::size_t p_idx = 0; p_idx < partitions.size(); ++p_idx) {
                const auto& p = partitions[p_idx];
                if (p.owning_rank != rank) continue;
                
                const std::size_t offset = p.global_offset;
                const std::size_t count = p.element_count;
                const int dev_id = p.device_global_index;
                const int work_passes = (dev_id == 0 ? 2 : 1);
                
                const auto t0_cpu = std::chrono::steady_clock::now();
                for (int pass = 0; pass < work_passes; ++pass) {
                    for (std::size_t i = offset; i < offset + count; ++i) {
                        float sum = 0.0f;
                        int valid_neighbors = 0;
                        for (int o = -k; o <= k; ++o) {
                            const long long idx = static_cast<long long>(i) + o;
                            if (idx >= 0 && idx < static_cast<long long>(n)) {
                                sum += in_data[static_cast<std::size_t>(idx)];
                                valid_neighbors++;
                            }
                        }
                        out_data[i] = sum / static_cast<float>(valid_neighbors);
                    }
                }
                const auto t1_cpu = std::chrono::steady_clock::now();
                local_device_times[p_idx] = std::chrono::duration<double>(t1_cpu - t0_cpu).count();
            }
        }

        const auto t1 = std::chrono::steady_clock::now();
        double local_step_time = std::chrono::duration<double>(t1 - t0).count();

        double t_max_step = 0.0;
        MPI_Allreduce(&local_step_time, &t_max_step, 1, MPI_DOUBLE, MPI_MAX, runtime.communicator());

        std::vector<double> device_times(partitions.size(), 0.0);
        MPI_Allreduce(local_device_times.data(), device_times.data(), partitions.size(), MPI_DOUBLE, MPI_MAX, runtime.communicator());

        // M1 Verification on the last iteration
        if (it == iterations - 1 && opencl_available) {
            runtime.gather(h_out, out_data.data(), n * sizeof(float));
            for (std::size_t p_idx = 0; p_idx < partitions.size(); ++p_idx) {
                const auto& p = partitions[p_idx];
                if (p.owning_rank != rank) continue;
                for (std::size_t i = p.global_offset; i < p.global_offset + p.element_count; ++i) {
                    float sum = 0.0f;
                    int valid_neighbors = 0;
                    for (int o = -k; o <= k; ++o) {
                        const long long idx = static_cast<long long>(i) + o;
                        if (idx >= 0 && idx < static_cast<long long>(n)) {
                            sum += in_data[static_cast<std::size_t>(idx)];
                            valid_neighbors++;
                        }
                    }
                    float cpu_val = sum / static_cast<float>(valid_neighbors);
                    if (std::abs(out_data[i] - cpu_val) > 1e-5f) {
                        std::cerr << "M1 Verification failed at index " << i << ": GPU=" << out_data[i] << " CPU=" << cpu_val << "\n";
                        MPI_Abort(runtime.communicator(), 1);
                    }
                }
            }
        }

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
