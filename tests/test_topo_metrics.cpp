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
#include <cstdio>
#include <fstream>
#include <iostream>

namespace {

void test_accept_correct_dimensions(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_accept_correct_dimensions..." << std::endl;
    rt.set_simulated_devices_count(2);

    dcl::TopoMetrics m;
    m.pcie_latency_ns = {4200.5, 4350.2};
    m.pcie_bandwidth_gbps = {15.8, 15.6};
    m.mpi_latency_ns = {0.0, 1250.0, 1250.0, 0.0};
    m.mpi_bandwidth_gbps = {0.0, 3.8, 3.8, 0.0};
    m.memory_contention_factor = {1.0, 1.25};

    rt.set_topo_metrics(m);

    const auto& stored = rt.topo_metrics();
    assert(stored.has_value());
    assert(stored->pcie_latency_ns.size() == 2);
    assert(std::fabs(stored->pcie_latency_ns[0] - 4200.5) < 1e-6);
    assert(stored->mpi_latency_ns.size() == 4);
    assert(std::fabs(stored->mpi_latency_ns[1] - 1250.0) < 1e-6);
    std::cout << "  -> PASS" << std::endl;
}

void test_reject_wrong_dimensions(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_reject_wrong_dimensions..." << std::endl;
    rt.set_simulated_devices_count(2);

    dcl::TopoMetrics base;
    base.pcie_latency_ns = {4200.0, 4300.0};
    base.pcie_bandwidth_gbps = {15.0, 15.0};
    base.mpi_latency_ns = {0.0, 1000.0, 1000.0, 0.0};
    base.mpi_bandwidth_gbps = {0.0, 3.0, 3.0, 0.0};

    // Case 1: mpi_latency_ns wrong size (3 instead of 4)
    {
        dcl::TopoMetrics bad = base;
        bad.mpi_latency_ns = {0.0, 1000.0, 1000.0};
        bool caught = false;
        try {
            rt.set_topo_metrics(bad);
        } catch (const dcl::Error&) {
            caught = true;
        }
        assert(caught && "Expected dcl::Error for invalid mpi_latency_ns dimension");
    }

    // Case 2: mpi_bandwidth_gbps wrong size (5 instead of 4)
    {
        dcl::TopoMetrics bad = base;
        bad.mpi_bandwidth_gbps = {0.0, 3.0, 3.0, 0.0, 0.0};
        bool caught = false;
        try {
            rt.set_topo_metrics(bad);
        } catch (const dcl::Error&) {
            caught = true;
        }
        assert(caught && "Expected dcl::Error for invalid mpi_bandwidth_gbps dimension");
    }
    std::cout << "  -> PASS" << std::endl;
}

void test_contention_factor_validation(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_contention_factor_validation..." << std::endl;
    rt.set_simulated_devices_count(2);

    dcl::TopoMetrics m;
    m.pcie_latency_ns = {4000.0, 4000.0};
    m.pcie_bandwidth_gbps = {15.0, 15.0};
    m.mpi_latency_ns = {0.0, 1000.0, 1000.0, 0.0};
    m.mpi_bandwidth_gbps = {0.0, 3.0, 3.0, 0.0};
    m.memory_contention_factor = {0.95, 1.0}; // 0.95 < 1.0 must fail

    bool caught = false;
    try {
        rt.set_topo_metrics(m);
    } catch (const dcl::Error&) {
        caught = true;
    }
    assert(caught && "Expected dcl::Error for memory_contention_factor < 1.0");
    std::cout << "  -> PASS" << std::endl;
}

void test_device_numa_node_validation(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_device_numa_node_validation..." << std::endl;
    rt.set_simulated_devices_count(2);

    dcl::TopoMetrics m;
    m.pcie_latency_ns = {4000.0, 4000.0};
    m.pcie_bandwidth_gbps = {15.0, 15.0};
    m.mpi_latency_ns = {0.0, 1000.0, 1000.0, 0.0};
    m.mpi_bandwidth_gbps = {0.0, 3.0, 3.0, 0.0};
    m.numa_distance = {10, 20, 20, 10}; // 2x2 -> num_nodes = 2
    m.device_numa_node = {0, 5}; // node 5 is out of range [0, 2)

    bool caught = false;
    try {
        rt.set_topo_metrics(m);
    } catch (const dcl::Error&) {
        caught = true;
    }
    assert(caught && "Expected dcl::Error for device_numa_node out of range");
    std::cout << "  -> PASS" << std::endl;
}

void test_save_load_round_trip() {
    std::cout << "[TEST] Running test_save_load_round_trip..." << std::endl;
    const std::string test_file = "test_roundtrip.json";

    dcl::TopoMetrics original;
    original.pcie_latency_ns = {3120.25, 4560.75};
    original.pcie_bandwidth_gbps = {15.75, 12.5};
    original.mpi_latency_ns = {0.0, 850.5, 850.5, 0.0};
    original.mpi_bandwidth_gbps = {0.0, 10.25, 10.25, 0.0};
    original.numa_distance = {10, 21, 21, 10};
    original.device_numa_node = {0, 1};
    original.memory_contention_factor = {1.0, 1.5};
    original.thermal_tdp_watts = {250.0, 300.0};
    original.current_power_watts = {115.5, 198.2};

    dcl::save_topo_metrics(original, test_file);
    dcl::TopoMetrics loaded = dcl::load_topo_metrics(test_file);

    assert(loaded.pcie_latency_ns.size() == original.pcie_latency_ns.size());
    for (std::size_t i = 0; i < loaded.pcie_latency_ns.size(); ++i) {
        assert(std::fabs(loaded.pcie_latency_ns[i] - original.pcie_latency_ns[i]) < 1e-6);
        assert(std::fabs(loaded.pcie_bandwidth_gbps[i] - original.pcie_bandwidth_gbps[i]) < 1e-6);
    }

    assert(loaded.mpi_latency_ns.size() == original.mpi_latency_ns.size());
    for (std::size_t i = 0; i < loaded.mpi_latency_ns.size(); ++i) {
        assert(std::fabs(loaded.mpi_latency_ns[i] - original.mpi_latency_ns[i]) < 1e-6);
        assert(std::fabs(loaded.mpi_bandwidth_gbps[i] - original.mpi_bandwidth_gbps[i]) < 1e-6);
    }

    assert(loaded.numa_distance == original.numa_distance);
    assert(loaded.device_numa_node == original.device_numa_node);

    assert(loaded.memory_contention_factor.size() == original.memory_contention_factor.size());
    for (std::size_t i = 0; i < loaded.memory_contention_factor.size(); ++i) {
        assert(std::fabs(loaded.memory_contention_factor[i] - original.memory_contention_factor[i]) < 1e-6);
        assert(std::fabs(loaded.thermal_tdp_watts[i] - original.thermal_tdp_watts[i]) < 1e-6);
        assert(std::fabs(loaded.current_power_watts[i] - original.current_power_watts[i]) < 1e-6);
    }

    std::remove(test_file.c_str());
    std::cout << "  -> PASS" << std::endl;
}

void test_error_handling() {
    std::cout << "[TEST] Running test_error_handling..." << std::endl;
    // Non-existent file
    bool caught = false;
    try {
        (void)dcl::load_topo_metrics("non_existent_file_xyz_123.json");
    } catch (const dcl::Error&) {
        caught = true;
    }
    assert(caught && "Expected dcl::Error on missing file");

    // Malformed JSON
    const std::string bad_file = "test_bad.json";
    {
        std::ofstream ofs(bad_file);
        ofs << "{ \"pcie_latency_ns\": [ 123.4, abc ] }";
    }
    caught = false;
    try {
        (void)dcl::load_topo_metrics(bad_file);
    } catch (const dcl::Error&) {
        caught = true;
    }
    assert(caught && "Expected dcl::Error on malformed JSON");
    std::remove(bad_file.c_str());
    std::cout << "  -> PASS" << std::endl;
}

void test_helpers() {
    std::cout << "[TEST] Running test_helpers..." << std::endl;
    dcl::TopoMetrics m;
    m.pcie_latency_ns = {5000.0, 5000.0};
    m.pcie_bandwidth_gbps = {10.0, 10.0};
    m.numa_distance = {10, 20, 20, 10};
    m.device_numa_node = {0, 1};
    m.memory_contention_factor = {1.0, 2.0};

    assert(dcl::numa_distance_between(m, 0, 0) == 10);
    assert(dcl::numa_distance_between(m, 0, 1) == 20);

    assert(std::fabs(dcl::adjusted_capacity(m, 0, 1000.0) - 1000.0) < 1e-6);
    assert(std::fabs(dcl::adjusted_capacity(m, 1, 1000.0) - 500.0) < 1e-6);

    double same_cost = dcl::estimate_migration_cost_bytes(m, 0, 0, 1024 * 1024);
    double diff_cost = dcl::estimate_migration_cost_bytes(m, 0, 1, 1024 * 1024);
    assert(diff_cost > same_cost);
    std::cout << "  -> PASS" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    dcl::Runtime rt = dcl::Runtime::create(argc, argv);

    test_accept_correct_dimensions(rt);
    test_reject_wrong_dimensions(rt);
    test_contention_factor_validation(rt);
    test_device_numa_node_validation(rt);
    test_save_load_round_trip();
    test_error_handling();
    test_helpers();

    std::cout << "\nAll test_topo_metrics unit tests PASSED successfully!" << std::endl;
    MPI_Finalize();
    return 0;
}
