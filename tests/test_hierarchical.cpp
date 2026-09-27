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
#include "dcl/topo_metrics_io.hpp"
#else
#include "../dcl/runtime.hpp"
#include "../dcl/runtime.cpp"
#include "../dcl/topo_metrics_io.hpp"
#endif

#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void test_hierarchical_timing_imbalance(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_hierarchical_timing_imbalance..." << std::endl;

    // 1. Configure simulated 2-device runtime on single rank
    rt.set_simulated_devices_count(2);

    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4;
    ps.granularity = 1;
    rt.set_partition(ps);

    // Initial partitions: [0, 500000) on dev 0, [500000, 1000000) on dev 1
    assert(rt.partitions().size() == 2);
    assert(rt.partitions()[0].element_count == 500000);
    assert(rt.partitions()[1].element_count == 500000);

    // 2. Inject simulated kernel times to simulate imbalance:
    //    Device 0 time = 10.0 ms (0.010 s)
    //    Device 1 time = 20.0 ms (0.020 s)
    rt.set_simulated_times({0.010, 0.020});

    // 3. Capture stdout to verify all [HIER] log lines appear
    std::stringstream captured;
    std::streambuf* old_buf = std::cout.rdbuf(captured.rdbuf());

    // 4. Call hierarchical balancing path
    const bool ok = rt.maybe_rebalance_hierarchical();

    // Restore stdout
    std::cout.rdbuf(old_buf);
    const std::string out_str = captured.str();

    // Echo to stdout so test logs and auditor can see it
    std::cout << out_str;

    assert(ok && "Expected maybe_rebalance_hierarchical to return true");

    // 5. Verify all [HIER] log lines appear
    assert(out_str.find("[HIER] intra-node loads:") != std::string::npos &&
           "Expected log line '[HIER] intra-node loads:' not found");
    assert(out_str.find("[HIER] inter-node loads:") != std::string::npos &&
           "Expected log line '[HIER] inter-node loads:' not found");
    assert(out_str.find("[HIER] final partition:") != std::string::npos &&
           "Expected log line '[HIER] final partition:' not found");

    // 6. Verify resulting partitions reflect per-device timing ratios
    //    Capacities: C0 = 1/10 = 0.1, C1 = 1/20 = 0.05
    //    Proportions: p0 = 0.1 / 0.15 = 2/3 (~66.67%), p1 = 0.05 / 0.15 = 1/3 (~33.33%)
    //    Expected elements: Dev 0 = 666667, Dev 1 = 333333
    const auto& parts = rt.partitions();
    assert(parts.size() == 2);
    assert(parts[0].global_offset == 0);
    assert(parts[0].element_count == 666667);
    assert(parts[1].global_offset == 666667);
    assert(parts[1].element_count == 333333);
    assert(parts[0].element_count + parts[1].element_count == 1000000);

    const double ratio0 = static_cast<double>(parts[0].element_count) / 1000000.0;
    const double ratio1 = static_cast<double>(parts[1].element_count) / 1000000.0;
    assert(std::fabs(ratio0 - (2.0 / 3.0)) < 1e-4);
    assert(std::fabs(ratio1 - (1.0 / 3.0)) < 1e-4);

    const double timing_ratio = static_cast<double>(parts[0].element_count) /
                                static_cast<double>(parts[1].element_count);
    assert(std::fabs(timing_ratio - 2.0) < 1e-3);

    rt.clear_simulated_times();
    std::cout << "  -> PASS" << std::endl;
}

void test_hierarchical_reverse_imbalance(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_hierarchical_reverse_imbalance..." << std::endl;

    rt.set_simulated_devices_count(2);

    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4;
    ps.granularity = 1;
    rt.set_partition(ps);

    // Reversed times: Device 0 = 20.0 ms, Device 1 = 10.0 ms
    // Device 1 is faster, so Device 0 gets 1/3 and Device 1 gets 2/3
    rt.set_simulated_times({0.020, 0.010});

    std::stringstream captured;
    std::streambuf* old_buf = std::cout.rdbuf(captured.rdbuf());

    const bool ok = rt.maybe_rebalance_hierarchical();

    std::cout.rdbuf(old_buf);
    const std::string out_str = captured.str();
    std::cout << out_str;

    assert(ok);
    assert(out_str.find("[HIER] intra-node loads:") != std::string::npos);
    assert(out_str.find("[HIER] inter-node loads:") != std::string::npos);
    assert(out_str.find("[HIER] final partition:") != std::string::npos);

    const auto& parts = rt.partitions();
    assert(parts.size() == 2);
    assert(parts[0].global_offset == 0);
    assert(parts[0].element_count == 333333);
    assert(parts[1].global_offset == 333333);
    assert(parts[1].element_count == 666667);
    assert(parts[0].element_count + parts[1].element_count == 1000000);

    const double ratio0 = static_cast<double>(parts[0].element_count) / 1000000.0;
    const double ratio1 = static_cast<double>(parts[1].element_count) / 1000000.0;
    assert(std::fabs(ratio0 - (1.0 / 3.0)) < 1e-4);
    assert(std::fabs(ratio1 - (2.0 / 3.0)) < 1e-4);

    rt.clear_simulated_times();
    std::cout << "  -> PASS" << std::endl;
}

void test_hierarchical_equal_times(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_hierarchical_equal_times..." << std::endl;

    rt.set_simulated_devices_count(2);

    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4;
    ps.granularity = 1;
    rt.set_partition(ps);

    // Equal times: both devices get 50%
    rt.set_simulated_times({0.015, 0.015});

    std::stringstream captured;
    std::streambuf* old_buf = std::cout.rdbuf(captured.rdbuf());

    const bool ok = rt.maybe_rebalance_hierarchical();

    std::cout.rdbuf(old_buf);
    const std::string out_str = captured.str();
    std::cout << out_str;

    assert(ok);
    assert(out_str.find("[HIER] intra-node loads:") != std::string::npos);
    assert(out_str.find("[HIER] inter-node loads:") != std::string::npos);
    assert(out_str.find("[HIER] final partition:") != std::string::npos);

    const auto& parts = rt.partitions();
    assert(parts.size() == 2);
    assert(parts[0].element_count == 500000);
    assert(parts[1].element_count == 500000);

    rt.clear_simulated_times();
    std::cout << "  -> PASS" << std::endl;
}

void test_hierarchical_with_granularity(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_hierarchical_with_granularity..." << std::endl;

    rt.set_simulated_devices_count(2);

    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4;
    ps.granularity = 1000; // Alignment to 1000
    rt.set_partition(ps);

    rt.set_simulated_times({0.010, 0.020});

    std::stringstream captured;
    std::streambuf* old_buf = std::cout.rdbuf(captured.rdbuf());

    const bool ok = rt.maybe_rebalance_hierarchical();

    std::cout.rdbuf(old_buf);
    const std::string out_str = captured.str();
    std::cout << out_str;

    assert(ok);
    const auto& parts = rt.partitions();
    assert(parts.size() == 2);

    // Cuts must be aligned to granularity 1000
    assert(parts[0].global_offset % 1000 == 0);
    assert(parts[0].element_count % 1000 == 0);
    assert(parts[1].global_offset % 1000 == 0);
    assert(parts[1].element_count % 1000 == 0);
    assert(parts[0].element_count + parts[1].element_count == 1000000);

    // 2/3 of 1000000 rounded to 1000 is 667000 or 666000
    assert(parts[0].element_count == 667000 || parts[0].element_count == 666000);

    rt.clear_simulated_times();
    std::cout << "  -> PASS" << std::endl;
}

void test_hierarchical_with_registered_field(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_hierarchical_with_registered_field..." << std::endl;

    rt.set_simulated_devices_count(2);

    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4;
    ps.granularity = 1;
    rt.set_partition(ps);

    dcl::FieldSpec fs;
    fs.name = "hier_test_field";
    fs.global_elements = 1000000;
    fs.units_per_element = 1;
    fs.bytes_per_unit = 4;
    fs.usage = dcl::BufferUsage::read_write;
    fs.redistribution = dcl::RedistributionDependency::proportional;
    dcl::FieldHandle fh = rt.create_field(fs);

    rt.set_simulated_times({0.010, 0.020});

    std::stringstream captured;
    std::streambuf* old_buf = std::cout.rdbuf(captured.rdbuf());

    const bool ok = rt.maybe_rebalance_hierarchical({fh});

    std::cout.rdbuf(old_buf);
    const std::string out_str = captured.str();
    std::cout << out_str;

    assert(ok);
    assert(out_str.find("[HIER] intra-node loads:") != std::string::npos);
    assert(out_str.find("[HIER] inter-node loads:") != std::string::npos);
    assert(out_str.find("[HIER] final partition:") != std::string::npos);

    const auto& parts = rt.partitions();
    assert(parts.size() == 2);
    assert(parts[0].element_count == 666667);
    assert(parts[1].element_count == 333333);

    rt.clear_simulated_times();
    std::cout << "  -> PASS" << std::endl;
}

void test_balance_mode_enum_and_policy() {
    std::cout << "[TEST] Running test_balance_mode_enum_and_policy..." << std::endl;

    dcl::AutoBalancePolicy pol;
    pol.mode = dcl::BalanceMode::hierarchical;
    pol.interval = 10;
    pol.threshold = 0.05f;

    assert(pol.mode == dcl::BalanceMode::hierarchical);
    assert(pol.interval == 10);

    // Switch statement covering all BalanceMode enumerators without compiler warnings
    int mode_id = -1;
    switch (pol.mode) {
        case dcl::BalanceMode::off:
            mode_id = 0;
            break;
        case dcl::BalanceMode::static_threshold:
            mode_id = 1;
            break;
        case dcl::BalanceMode::dynamic_threshold:
            mode_id = 2;
            break;
        case dcl::BalanceMode::static_profiled:
            mode_id = 3;
            break;
        case dcl::BalanceMode::dynamic_profiled:
            mode_id = 4;
            break;
        case dcl::BalanceMode::hierarchical:
            mode_id = 5;
            break;
    }
    assert(mode_id == 5);

    std::cout << "  -> PASS" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    dcl::Runtime rt = dcl::Runtime::create(argc, argv);

    test_balance_mode_enum_and_policy();
    test_hierarchical_timing_imbalance(rt);
    test_hierarchical_reverse_imbalance(rt);
    test_hierarchical_equal_times(rt);
    test_hierarchical_with_granularity(rt);
    test_hierarchical_with_registered_field(rt);

    std::cout << "\nAll test_hierarchical unit tests PASSED successfully!" << std::endl;
    MPI_Finalize();
    return 0;
}
