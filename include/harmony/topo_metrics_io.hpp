// Zero-dependency JSON serialization and deserialization for harmony::TopoMetrics.

#ifndef HARMONY_TOPO_METRICS_IO_HPP
#define HARMONY_TOPO_METRICS_IO_HPP

#include "harmony/types.hpp"

#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace harmony {

namespace detail {

inline void topo_skip_whitespace(const std::string& str, std::size_t& pos) noexcept {
    while (pos < str.size() && std::isspace(static_cast<unsigned char>(str[pos])) != 0) {
        ++pos;
    }
}

inline bool topo_consume_char(const std::string& str, std::size_t& pos, char c) noexcept {
    topo_skip_whitespace(str, pos);
    if (pos < str.size() && str[pos] == c) {
        ++pos;
        return true;
    }
    return false;
}

inline std::string topo_parse_string(const std::string& str, std::size_t& pos) {
    topo_skip_whitespace(str, pos);
    if (pos >= str.size() || str[pos] != '"') {
        throw Error("JSON parse error: expected double quote at position " + std::to_string(pos));
    }
    ++pos;
    std::string result;
    while (pos < str.size()) {
        char c = str[pos++];
        if (c == '"') {
            return result;
        }
        if (static_cast<unsigned char>(c) < 0x20) {
            throw Error("JSON parse error: unescaped control character");
        }
        if (c == '\\') {
            if (pos >= str.size()) {
                throw Error("JSON parse error: trailing escape backslash");
            }
            c = str[pos++];
            switch (c) {
                case '"':
                case '\\':
                case '/':
                    break;
                case 'b':
                    c = '\b';
                    break;
                case 'f':
                    c = '\f';
                    break;
                case 'n':
                    c = '\n';
                    break;
                case 'r':
                    c = '\r';
                    break;
                case 't':
                    c = '\t';
                    break;
                default:
                    throw Error("JSON parse error: invalid string escape");
            }
        }
        result.push_back(c);
    }
    throw Error("JSON parse error: unterminated string");
}

inline std::size_t topo_scan_number(const std::string& str, std::size_t& pos, bool& has_fraction_or_exp) {
    topo_skip_whitespace(str, pos);
    const std::size_t start = pos;
    if (pos < str.size() && str[pos] == '-') {
        ++pos;
    }
    if (pos >= str.size()) {
        throw Error("JSON parse error: missing number");
    }
    if (str[pos] == '0') {
        ++pos;
        if (pos < str.size() && std::isdigit(static_cast<unsigned char>(str[pos])) != 0) {
            throw Error("JSON parse error: leading zero");
        }
    } else {
        if (str[pos] < '1' || str[pos] > '9') {
            throw Error("JSON parse error: invalid number");
        }
        while (pos < str.size() && std::isdigit(static_cast<unsigned char>(str[pos])) != 0) {
            ++pos;
        }
    }

    has_fraction_or_exp = false;
    if (pos < str.size() && str[pos] == '.') {
        has_fraction_or_exp = true;
        ++pos;
        if (pos >= str.size() || std::isdigit(static_cast<unsigned char>(str[pos])) == 0) {
            throw Error("JSON parse error: invalid fraction");
        }
        while (pos < str.size() && std::isdigit(static_cast<unsigned char>(str[pos])) != 0) {
            ++pos;
        }
    }
    if (pos < str.size() && (str[pos] == 'e' || str[pos] == 'E')) {
        has_fraction_or_exp = true;
        ++pos;
        if (pos < str.size() && (str[pos] == '+' || str[pos] == '-')) {
            ++pos;
        }
        if (pos >= str.size() || std::isdigit(static_cast<unsigned char>(str[pos])) == 0) {
            throw Error("JSON parse error: invalid exponent");
        }
        while (pos < str.size() && std::isdigit(static_cast<unsigned char>(str[pos])) != 0) {
            ++pos;
        }
    }
    return start;
}

template <typename T>
inline std::vector<T> topo_parse_number_array(const std::string& str, std::size_t& pos) {
    if (!topo_consume_char(str, pos, '[')) {
        throw Error("JSON parse error: expected '[' at position " + std::to_string(pos));
    }
    std::vector<T> result;
    topo_skip_whitespace(str, pos);
    if (pos < str.size() && str[pos] == ']') {
        ++pos;
        return result;
    }

    while (pos < str.size()) {
        bool has_fraction_or_exp = false;
        const std::size_t start = topo_scan_number(str, pos, has_fraction_or_exp);
        if constexpr (std::is_integral_v<T>) {
            if (has_fraction_or_exp) {
                throw Error("JSON parse error: expected integer");
            }
        }
        T val{};
        const auto parsed = std::from_chars(str.data() + start, str.data() + pos, val);
        if (parsed.ec != std::errc{} || parsed.ptr != str.data() + pos) {
            throw Error("JSON parse error: number out of range");
        }
        if (!std::isfinite(static_cast<double>(val))) {
            throw Error("JSON parse error: non-finite number");
        }
        result.push_back(val);

        topo_skip_whitespace(str, pos);
        if (pos < str.size() && str[pos] == ',') {
            ++pos;
            topo_skip_whitespace(str, pos);
            if (pos >= str.size() || str[pos] == ']') {
                throw Error("JSON parse error: trailing array comma");
            }
            continue;
        }
        if (pos < str.size() && str[pos] == ']') {
            ++pos;
            return result;
        }
        throw Error("JSON parse error: expected ',' or ']' at position " + std::to_string(pos));
    }
    throw Error("JSON parse error: missing closing ']'");
}

inline void topo_skip_json_value(const std::string& str, std::size_t& pos, int depth = 0) {
    if (depth > 64) {
        throw Error("JSON parse error: excessive nesting");
    }
    topo_skip_whitespace(str, pos);
    if (pos >= str.size()) {
        throw Error("JSON parse error: missing value");
    }
    if (str[pos] == '"') {
        (void)topo_parse_string(str, pos);
        return;
    }
    if (str[pos] == '[' || str[pos] == '{') {
        const bool is_object = str[pos] == '{';
        ++pos;
        topo_skip_whitespace(str, pos);
        const char closing = is_object ? '}' : ']';
        if (pos < str.size() && str[pos] == closing) {
            ++pos;
            return;
        }
        while (true) {
            if (is_object) {
                (void)topo_parse_string(str, pos);
                if (!topo_consume_char(str, pos, ':')) {
                    throw Error("JSON parse error: expected ':'");
                }
            }
            topo_skip_json_value(str, pos, depth + 1);
            if (topo_consume_char(str, pos, closing)) {
                return;
            }
            if (!topo_consume_char(str, pos, ',')) {
                throw Error("JSON parse error: expected comma");
            }
            topo_skip_whitespace(str, pos);
            if (pos >= str.size() || str[pos] == closing) {
                throw Error("JSON parse error: trailing comma");
            }
        }
    }
    for (const char* literal : {"true", "false", "null"}) {
        const std::size_t length = std::char_traits<char>::length(literal);
        if (str.compare(pos, length, literal) == 0) {
            pos += length;
            return;
        }
    }

    bool has_fraction_or_exp = false;
    const std::size_t start = topo_scan_number(str, pos, has_fraction_or_exp);
    double value = 0.0;
    const auto parsed = std::from_chars(str.data() + start, str.data() + pos, value);
    if (parsed.ec != std::errc{} || parsed.ptr != str.data() + pos || !std::isfinite(value)) {
        throw Error("JSON parse error: invalid number");
    }
}

template <typename T>
inline void topo_serialize_array(std::ostream& os, const std::string& key,
                                 const std::vector<T>& vec, bool is_last = false) {
    os << "  \"" << key << "\": [";
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) os << ", ";
        os << std::setprecision(15) << vec[i];
    }
    os << "]";
    if (!is_last) os << ",";
    os << "\n";
}

inline void topo_require_finite_doubles(const std::vector<double>& values, const char* field_name) {
    for (double v : values) {
        if (!std::isfinite(v)) {
            throw Error(std::string("Cannot serialize non-finite value in ") + field_name);
        }
    }
}

} // namespace detail

inline TopoMetrics load_topo_metrics(const std::string& json_path) {
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        throw Error("Cannot open file: " + json_path);
    }
    std::ostringstream buffer;
    buffer << ifs.rdbuf();
    if (ifs.bad()) {
        throw Error("Failed reading file: " + json_path);
    }
    const std::string content = buffer.str();

    TopoMetrics metrics;
    std::size_t pos = 0;
    if (!detail::topo_consume_char(content, pos, '{')) {
        throw Error("JSON parse error: expected '{' at start of file: " + json_path);
    }

    bool closed = false;
    std::unordered_set<std::string> seen_keys;
    while (pos < content.size()) {
        detail::topo_skip_whitespace(content, pos);
        if (pos < content.size() && content[pos] == '}') {
            ++pos;
            closed = true;
            break;
        }
        const std::string key = detail::topo_parse_string(content, pos);
        if (!seen_keys.insert(key).second) {
            throw Error("JSON parse error: duplicate key '" + key + "'");
        }
        if (!detail::topo_consume_char(content, pos, ':')) {
            throw Error("JSON parse error: expected ':' after key '" + key + "'");
        }
        detail::topo_skip_whitespace(content, pos);

        if (key == "pcie_latency_ns") {
            metrics.pcie_latency_ns = detail::topo_parse_number_array<double>(content, pos);
        } else if (key == "pcie_bandwidth_gbps") {
            metrics.pcie_bandwidth_gbps = detail::topo_parse_number_array<double>(content, pos);
        } else if (key == "mpi_latency_ns") {
            metrics.mpi_latency_ns = detail::topo_parse_number_array<double>(content, pos);
        } else if (key == "mpi_bandwidth_gbps") {
            metrics.mpi_bandwidth_gbps = detail::topo_parse_number_array<double>(content, pos);
        } else if (key == "numa_distance") {
            metrics.numa_distance = detail::topo_parse_number_array<int>(content, pos);
        } else if (key == "device_numa_node") {
            metrics.device_numa_node = detail::topo_parse_number_array<int>(content, pos);
        } else if (key == "memory_contention_factor") {
            metrics.memory_contention_factor = detail::topo_parse_number_array<double>(content, pos);
        } else if (key == "thermal_tdp_watts") {
            metrics.thermal_tdp_watts = detail::topo_parse_number_array<double>(content, pos);
        } else if (key == "current_power_watts") {
            metrics.current_power_watts = detail::topo_parse_number_array<double>(content, pos);
        } else if (key == "synthetic") {
            if (content.compare(pos, 4, "true") == 0) {
                metrics.synthetic = true;
                pos += 4;
            } else if (content.compare(pos, 5, "false") == 0) {
                metrics.synthetic = false;
                pos += 5;
            } else {
                throw Error("JSON parse error: synthetic must be boolean");
            }
        } else {
            detail::topo_skip_json_value(content, pos);
        }

        detail::topo_skip_whitespace(content, pos);
        if (pos < content.size() && content[pos] == ',') {
            ++pos;
            detail::topo_skip_whitespace(content, pos);
            if (pos >= content.size() || content[pos] == '}') {
                throw Error("JSON parse error: trailing comma before '}'");
            }
            continue;
        }
        if (pos < content.size() && content[pos] == '}') {
            ++pos;
            closed = true;
            break;
        }
        throw Error("JSON parse error: expected ',' or '}' at position " + std::to_string(pos));
    }

    if (!closed) {
        throw Error("JSON parse error: missing closing '}'");
    }
    detail::topo_skip_whitespace(content, pos);
    if (pos != content.size()) {
        throw Error("JSON parse error: trailing characters after object");
    }

    return metrics;
}

inline void save_topo_metrics(const TopoMetrics& metrics, const std::string& json_path) {
    struct DoubleField {
        const std::vector<double>& values;
        const char* name;
    };
    const std::array<DoubleField, 7> double_fields{{
        {metrics.pcie_latency_ns, "pcie_latency_ns"},
        {metrics.pcie_bandwidth_gbps, "pcie_bandwidth_gbps"},
        {metrics.mpi_latency_ns, "mpi_latency_ns"},
        {metrics.mpi_bandwidth_gbps, "mpi_bandwidth_gbps"},
        {metrics.memory_contention_factor, "memory_contention_factor"},
        {metrics.thermal_tdp_watts, "thermal_tdp_watts"},
        {metrics.current_power_watts, "current_power_watts"}
    }};
    for (const auto& field : double_fields) {
        detail::topo_require_finite_doubles(field.values, field.name);
    }

    std::ofstream ofs(json_path);
    if (!ofs.is_open()) {
        throw Error("Cannot open file for writing: " + json_path);
    }
    ofs << "{\n";
    detail::topo_serialize_array(ofs, "pcie_latency_ns", metrics.pcie_latency_ns);
    detail::topo_serialize_array(ofs, "pcie_bandwidth_gbps", metrics.pcie_bandwidth_gbps);
    detail::topo_serialize_array(ofs, "mpi_latency_ns", metrics.mpi_latency_ns);
    detail::topo_serialize_array(ofs, "mpi_bandwidth_gbps", metrics.mpi_bandwidth_gbps);
    detail::topo_serialize_array(ofs, "numa_distance", metrics.numa_distance);
    detail::topo_serialize_array(ofs, "device_numa_node", metrics.device_numa_node);
    detail::topo_serialize_array(ofs, "memory_contention_factor", metrics.memory_contention_factor);
    detail::topo_serialize_array(ofs, "thermal_tdp_watts", metrics.thermal_tdp_watts);
    detail::topo_serialize_array(ofs, "current_power_watts", metrics.current_power_watts);
    ofs << "  \"synthetic\": " << (metrics.synthetic ? "true" : "false") << "\n";
    ofs << "}\n";
    ofs.flush();
    if (!ofs) {
        throw Error("Failed writing file: " + json_path);
    }
}

} // namespace harmony

#endif // HARMONY_TOPO_METRICS_IO_HPP
