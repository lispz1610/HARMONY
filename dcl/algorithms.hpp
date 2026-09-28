#ifndef DCL_ALGORITHMS_HPP
#define DCL_ALGORITHMS_HPP

#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

namespace dcl {

/**
 * @brief ERAD deterministic load balancing algorithm (R4.1).
 *
 * Deterministic strategy: computes loads purely from partition throughput
 * (elements per second), without stochastic perturbation.
 *
 * @param times Execution times per partition.
 * @param partitions Partition metadata for all active partitions.
 * @return Cumulative load vector (monotone non-decreasing, ending at 1.0f).
 */
inline std::vector<float> erad_loads(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions
) {
    if (times.empty() || times.size() != partitions.size()) {
        return {};
    }

    std::vector<double> capacity(times.size(), 0.0);
    double total_capacity = 0.0;

    for (std::size_t i = 0; i < times.size(); ++i) {
        if (partitions[i].element_count == 0 || times[i] <= 1.0e-12) {
            continue;
        }

        capacity[i] =
            static_cast<double>(partitions[i].element_count) / times[i];
        total_capacity += capacity[i];
    }

    std::vector<float> loads(times.size(), 0.0f);
    if (total_capacity <= 0.0) {
        const float step = 1.0f / static_cast<float>(times.size());
        float acc = 0.0f;
        for (std::size_t i = 0; i < loads.size(); ++i) {
            acc += step;
            loads[i] = acc;
        }
        loads.back() = 1.0f;
        return loads;
    }

    double cumulative = 0.0;
    for (std::size_t i = 0; i < capacity.size(); ++i) {
        cumulative += capacity[i] / total_capacity;
        loads[i] = static_cast<float>(cumulative);
    }

    loads.back() = 1.0f;
    return loads;
}

/**
 * @brief HWTOPOLB stochastic load balancing algorithm (R4.2).
 *
 * Computes base unperturbed shares from partition throughput, then if
 * perturbation_factor > 0.0f applies a bounded uniform random perturbation
 * delta_i in [-perturbation_factor, +perturbation_factor], clamps shares,
 * normalizes them, and returns cumulative loads.
 * If perturbation_factor == 0.0f, shares remain identical to ERAD.
 *
 * @param times Execution times per partition.
 * @param partitions Partition metadata for all active partitions.
 * @param perturbation_factor Maximum relative perturbation magnitude.
 * @return Cumulative load vector (monotone non-decreasing, ending at 1.0f).
 */
inline std::vector<float> hwtopolb_loads(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions,
    float perturbation_factor,
    unsigned int seed = 0
) {
    if (times.empty() || times.size() != partitions.size()) {
        return {};
    }

    if (perturbation_factor <= 0.0f) {
        return erad_loads(times, partitions);
    }

    std::vector<double> capacity(times.size(), 0.0);
    double total_capacity = 0.0;

    for (std::size_t i = 0; i < times.size(); ++i) {
        if (partitions[i].element_count == 0 || times[i] <= 1.0e-12) {
            continue;
        }

        capacity[i] =
            static_cast<double>(partitions[i].element_count) / times[i];
        total_capacity += capacity[i];
    }

    const std::size_t n = times.size();
    std::vector<float> share(n, 0.0f);
    if (total_capacity <= 0.0) {
        const float equal_share = 1.0f / static_cast<float>(n);
        for (std::size_t i = 0; i < n; ++i) {
            share[i] = equal_share;
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            share[i] = static_cast<float>(capacity[i] / total_capacity);
        }
    }

    std::mt19937 gen;
    if (seed != 0) {
        gen.seed(seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    std::uniform_real_distribution<float> dist(-perturbation_factor, perturbation_factor);

    std::vector<float> perturbed_share(n, 0.0f);
    float sum_perturbed = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float delta_i = dist(gen);
        perturbed_share[i] = std::max(1e-6f, share[i] * (1.0f + delta_i));
        sum_perturbed += perturbed_share[i];
    }

    if (sum_perturbed <= 0.0f) {
        return erad_loads(times, partitions);
    }

    std::vector<float> loads(n, 0.0f);
    double cumulative = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const float final_share = perturbed_share[i] / sum_perturbed;
        cumulative += static_cast<double>(final_share);
        loads[i] = static_cast<float>(cumulative);
    }

    loads.back() = 1.0f;
    return loads;
}

/**
 * @brief Power-aware load cap algorithm (R5.3).
 *
 * Converts cumulative partition loads to individual device shares, identifies
 * devices exceeding thermal TDP or system power budget limits, reduces their load
 * shares proportionally, and redistributes the excess load to underloaded devices
 * according to spare thermal headroom or existing shares.
 *
 * @param loads Cumulative load vector (monotone non-decreasing, ending at 1.0f).
 * @param metrics Topology metrics containing thermal_tdp_watts and current_power_watts.
 * @param power_budget_watts Overall cluster/node power budget cap (0.0 if unconstrained).
 * @return Valid cumulative load vector (monotone non-decreasing, ending at 1.0f).
 */
inline std::vector<float> apply_power_cap(
    const std::vector<float>& loads,
    const TopoMetrics& metrics,
    double power_budget_watts = 0.0
) {
    if (loads.empty()) {
        return {};
    }
    if (loads.size() == 1) {
        return {1.0f};
    }

    const std::size_t n = loads.size();

    // If power metrics are not configured, return input loads as-is
    if (metrics.current_power_watts.empty() || metrics.thermal_tdp_watts.empty()) {
        return loads;
    }

    // 1. Convert cumulative loads to per-device individual load shares:
    //    p[i] = loads[i] - (i > 0 ? loads[i-1] : 0.0f)
    std::vector<double> shares(n, 0.0);
    double prev = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double cur = static_cast<double>(loads[i]);
        shares[i] = std::max(0.0, cur - prev);
        prev = cur;
    }

    const double sum_initial = std::accumulate(shares.begin(), shares.end(), 0.0);
    if (sum_initial <= 0.0) {
        for (std::size_t i = 0; i < n; ++i) {
            shares[i] = 1.0 / static_cast<double>(n);
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            shares[i] /= sum_initial;
        }
    }

    // 2. Compute effective per-device limits taking power_budget_watts into account
    std::vector<double> effective_limit(n, 0.0);
    double total_tdp = 0.0;
    double total_curr_power = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double tdp = (i < metrics.thermal_tdp_watts.size()) ? metrics.thermal_tdp_watts[i] : 0.0;
        const double curr = (i < metrics.current_power_watts.size()) ? metrics.current_power_watts[i] : 0.0;
        total_tdp += tdp;
        total_curr_power += curr;
    }

    for (std::size_t i = 0; i < n; ++i) {
        double limit = (i < metrics.thermal_tdp_watts.size()) ? metrics.thermal_tdp_watts[i] : 0.0;
        if (power_budget_watts > 0.0) {
            if (total_tdp > power_budget_watts && total_tdp > 0.0) {
                limit = std::min(limit, limit * (power_budget_watts / total_tdp));
            }
            if (total_curr_power > power_budget_watts && total_curr_power > 0.0 &&
                i < metrics.current_power_watts.size()) {
                limit = std::min(limit, metrics.current_power_watts[i] * (power_budget_watts / total_curr_power));
            }
        }
        effective_limit[i] = limit;
    }

    // 3. Identify overloaded devices: current_power_watts[i] > effective_limit[i]
    //    and scale down their load shares proportionally
    double freed_load = 0.0;
    std::vector<double> headroom(n, 0.0);
    double total_headroom = 0.0;
    double total_underloaded_shares = 0.0;
    std::vector<std::size_t> underloaded_indices;

    for (std::size_t i = 0; i < n; ++i) {
        const double curr_pwr = (i < metrics.current_power_watts.size()) ? metrics.current_power_watts[i] : 0.0;
        const double limit = effective_limit[i];

        if (limit > 0.0 && curr_pwr > limit) {
            const double scale = limit / curr_pwr;
            const double new_share = shares[i] * scale;
            freed_load += (shares[i] - new_share);
            shares[i] = new_share;
        } else {
            underloaded_indices.push_back(i);
            const double hr = std::max(0.0, limit - curr_pwr);
            headroom[i] = hr;
            total_headroom += hr;
            total_underloaded_shares += shares[i];
        }
    }

    // 4. Redistribute the freed excess load to underloaded devices
    if (freed_load > 0.0 && !underloaded_indices.empty()) {
        if (total_headroom > 1e-12) {
            for (std::size_t idx : underloaded_indices) {
                shares[idx] += freed_load * (headroom[idx] / total_headroom);
            }
        } else if (total_underloaded_shares > 1e-12) {
            for (std::size_t idx : underloaded_indices) {
                shares[idx] += freed_load * (shares[idx] / total_underloaded_shares);
            }
        } else {
            const double equal_share = freed_load / static_cast<double>(underloaded_indices.size());
            for (std::size_t idx : underloaded_indices) {
                shares[idx] += equal_share;
            }
        }
    } else if (freed_load > 0.0 && underloaded_indices.empty()) {
        // All devices were overloaded; their shares were reduced proportionally.
        // Normalize so shares sum to 1.0.
        const double s_sum = std::accumulate(shares.begin(), shares.end(), 0.0);
        if (s_sum > 0.0) {
            for (std::size_t i = 0; i < n; ++i) {
                shares[i] /= s_sum;
            }
        }
    }

    // Ensure sum is exactly 1.0
    const double final_sum = std::accumulate(shares.begin(), shares.end(), 0.0);
    if (final_sum > 0.0) {
        for (std::size_t i = 0; i < n; ++i) {
            shares[i] /= final_sum;
        }
    }

    // 5. Convert back to a valid cumulative load vector (strictly monotone non-decreasing, last element = 1.0f)
    std::vector<float> result(n, 0.0f);
    double cumulative = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        cumulative += shares[i];
        result[i] = static_cast<float>(cumulative);
    }

    for (std::size_t i = 1; i < n; ++i) {
        if (result[i] < result[i - 1]) {
            result[i] = result[i - 1];
        }
    }
    result.back() = 1.0f;

    return result;
}

} // namespace dcl

using dcl::apply_power_cap;
using dcl::erad_loads;
using dcl::hwtopolb_loads;

#endif // DCL_ALGORITHMS_HPP
