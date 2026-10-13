#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>
#include "../dcl/runtime.hpp"
#include "../dcl/runtime.cpp"

#include <cassert>
#include <iostream>
#include <sstream>
#include <vector>

struct Observation {
    std::vector<std::size_t> counts;
    std::string output;
};

Observation run_case(int& argc, char**& argv, bool quiet) {
    auto runtime = dcl::Runtime::create(argc, argv);
    runtime.set_simulated_devices_count(2);
    if (quiet) runtime.set_verbose_metrics(false);
    dcl::PartitionSpec partition;
    partition.global_elements = 100;
    partition.units_per_element = 1;
    partition.bytes_per_unit = sizeof(float);
    partition.granularity = 1;
    runtime.set_partition(partition);
    std::vector<float> data(100, 1.0f);
    dcl::FieldSpec field{"input", 100, 1, sizeof(float), dcl::BufferUsage::read_only,
                         data.data(), dcl::RedistributionDependency::proportional};
    const auto handle = runtime.create_field(field);
    std::ostringstream captured;
    auto* old = std::cout.rdbuf(captured.rdbuf());
    runtime.set_simulated_times({0.020, 0.010});
    runtime.maybe_rebalance_from_timings({handle}, 0.0f);
    Observation result;
    for (const auto& part : runtime.partitions()) result.counts.push_back(part.element_count);
    runtime.set_simulated_times({0.010, 0.020});
    runtime.maybe_rebalance_from_timings({handle}, 0.0f);
    std::cout.rdbuf(old);
    for (const auto& part : runtime.partitions()) result.counts.push_back(part.element_count);
    result.output = captured.str();
    return result;
}

int main(int argc, char** argv) {
    const auto verbose = run_case(argc, argv, false);
    const auto quiet = run_case(argc, argv, true);
    assert(verbose.counts == quiet.counts);
    assert(verbose.counts.size() == 4);
    assert(verbose.counts[0] + verbose.counts[1] == 100);
    assert(verbose.counts[2] + verbose.counts[3] == 100);
    if (quiet.output.find("[DCL][balance-metrics]") != std::string::npos ||
        quiet.output.find("OLD loads") != std::string::npos) return 1;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0 && verbose.output.find("[DCL][balance-metrics]") == std::string::npos) return 1;
    std::cout << "Runtime metric modes: PASS\n";
    MPI_Finalize();
}
