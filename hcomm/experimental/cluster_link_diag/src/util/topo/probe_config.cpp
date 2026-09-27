/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "probe_config.h"

#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
std::string device_id_to_string(const json& value, const std::string& field)
{
    std::string device_id;
    if (value.is_string()) {
        device_id = value.get<std::string>();
    } else if (value.is_number_integer()) {
        const int numeric_id = value.get<int>();
        if (numeric_id < 0) {
            throw std::runtime_error("[probe_config] " + field + " device id must be non-negative");
        }
        device_id = std::to_string(numeric_id);
    } else {
        throw std::runtime_error("[probe_config] " + field + " device id must be string or integer");
    }

    if (device_id.empty() || !std::all_of(device_id.begin(), device_id.end(), [](unsigned char ch) {
            return std::isdigit(ch) != 0;
        })) {
        throw std::runtime_error(
            "[probe_config] " + field + " device id must be a non-negative integer, range syntax is unsupported");
    }
    return device_id;
}

int read_positive_int(const json& object, const std::string& key, int default_value, const std::string& field)
{
    int value = default_value;
    if (object.contains(key)) {
        if (!object.at(key).is_number_integer()) {
            throw std::runtime_error("[probe_config] " + field + "." + key + " must be integer");
        }
        value = object.at(key).get<int>();
    }
    if (value <= 0) {
        throw std::runtime_error("[probe_config] " + field + "." + key + " must be positive");
    }
    return value;
}

int read_int_in_range(
    const json& object, const std::string& key, int default_value, int min_value, int max_value,
    const std::string& field)
{
    const int value = read_positive_int(object, key, default_value, field);
    if (value < min_value || value > max_value) {
        throw std::runtime_error(
            "[probe_config] " + field + "." + key + " must be in [" + std::to_string(min_value) + ", "
            + std::to_string(max_value) + "]");
    }
    return value;
}

std::string read_output_subdir(const json& object, const std::string& field)
{
    if (!object.contains("output_subdir")) {
        return "";
    }
    if (!object.at("output_subdir").is_string()) {
        throw std::runtime_error("[probe_config] " + field + ".output_subdir must be string");
    }

    std::string subdir = object.at("output_subdir").get<std::string>();
    if (subdir.empty()) {
        return subdir;
    }

    std::filesystem::path path(subdir);
    if (path.is_absolute()) {
        throw std::runtime_error("[probe_config] " + field + ".output_subdir must be relative to output/");
    }
    for (const auto& part : path) {
        std::string item = part.string();
        if (item.empty() || item == "." || item == "..") {
            throw std::runtime_error(
                "[probe_config] " + field + ".output_subdir cannot contain '.', '..', or empty path segments");
        }
    }
    return subdir;
}

std::string read_output_filename(
    const json& object, const std::string& key, const std::string& default_value, const std::string& field)
{
    if (!object.contains(key)) {
        return default_value;
    }
    if (!object.at(key).is_string()) {
        throw std::runtime_error("[probe_config] " + field + "." + key + " must be string");
    }

    std::string filename = object.at(key).get<std::string>();
    if (filename.empty()) {
        throw std::runtime_error("[probe_config] " + field + "." + key + " cannot be empty");
    }

    std::filesystem::path path(filename);
    if (path.is_absolute() || path.has_parent_path() || path.filename().string() != filename) {
        throw std::runtime_error("[probe_config] " + field + "." + key + " must be a filename under output_subdir");
    }
    return filename;
}

std::map<std::string, std::vector<std::string>> parse_scope(const json& scope_json, const std::string& field)
{
    if (!scope_json.is_object() || scope_json.empty()) {
        throw std::runtime_error("[probe_config] " + field + " must be a non-empty object");
    }

    std::map<std::string, std::vector<std::string>> scope;
    for (const auto& [host_ip, devices_json] : scope_json.items()) {
        if (host_ip.empty()) {
            throw std::runtime_error("[probe_config] " + field + " contains empty host ip");
        }
        if (!devices_json.is_array() || devices_json.empty()) {
            throw std::runtime_error("[probe_config] " + field + "." + host_ip + " must be a non-empty array");
        }

        std::set<std::string> seen_devices;
        std::vector<std::string> devices;
        for (size_t i = 0; i < devices_json.size(); i++) {
            std::string device_id = device_id_to_string(devices_json.at(i), field + "." + host_ip);
            if (device_id.empty()) {
                throw std::runtime_error("[probe_config] " + field + "." + host_ip + " contains empty device id");
            }
            if (!seen_devices.insert(device_id).second) {
                throw std::runtime_error(
                    "[probe_config] " + field + "." + host_ip + " contains duplicate device id: " + device_id);
            }
            devices.emplace_back(std::move(device_id));
        }
        scope[host_ip] = std::move(devices);
    }
    return scope;
}

std::map<std::string, std::string> parse_v2_host_ip_map(const json& root)
{
    if (!root.contains("hosts") || !root.at("hosts").is_array() || root.at("hosts").empty()) {
        throw std::runtime_error("[probe_config] hosts must be a non-empty array for schema_version=2");
    }

    std::map<std::string, std::string> host_ip_by_id;
    std::set<std::string> seen_ips;
    for (size_t i = 0; i < root.at("hosts").size(); i++) {
        const json& host = root.at("hosts").at(i);
        const std::string field = "hosts[" + std::to_string(i) + "]";
        if (!host.is_object()) {
            throw std::runtime_error("[probe_config] " + field + " must be object");
        }
        if (!host.contains("id") || !host.at("id").is_string() || host.at("id").get<std::string>().empty()) {
            throw std::runtime_error("[probe_config] " + field + ".id must be non-empty string");
        }
        if (!host.contains("ip") || !host.at("ip").is_string() || host.at("ip").get<std::string>().empty()) {
            throw std::runtime_error("[probe_config] " + field + ".ip must be non-empty string");
        }

        const std::string id = host.at("id").get<std::string>();
        const std::string ip = host.at("ip").get<std::string>();
        if (!host_ip_by_id.emplace(id, ip).second) {
            throw std::runtime_error("[probe_config] duplicate host id: " + id);
        }
        if (!seen_ips.insert(ip).second) {
            throw std::runtime_error("[probe_config] duplicate host ip: " + ip);
        }
    }
    return host_ip_by_id;
}

std::vector<std::string> parse_v2_devices(const json& scope_entry, const std::string& field)
{
    json devices_json;
    if (scope_entry.is_array()) {
        devices_json = scope_entry;
    } else if (scope_entry.is_object()) {
        const bool has_devices = scope_entry.contains("devices");
        const bool has_range = scope_entry.contains("device_range");
        if (has_devices == has_range) {
            throw std::runtime_error(
                "[probe_config] " + field + " must contain exactly one of devices or device_range");
        }
        if (has_devices) {
            devices_json = scope_entry.at("devices");
        } else {
            const json& range = scope_entry.at("device_range");
            if (!range.is_array() || range.size() != 2 || !range.at(0).is_number_integer()
                || !range.at(1).is_number_integer()) {
                throw std::runtime_error("[probe_config] " + field + ".device_range must be [begin, end]");
            }
            const int begin = range.at(0).get<int>();
            const int end = range.at(1).get<int>();
            if (begin < 0 || end < begin) {
                throw std::runtime_error(
                    "[probe_config] " + field + ".device_range must be non-negative and ascending");
            }
            devices_json = json::array();
            for (int device = begin; device <= end; device++) {
                devices_json.push_back(device);
            }
        }
    } else {
        throw std::runtime_error("[probe_config] " + field + " must be array or object");
    }

    if (!devices_json.is_array() || devices_json.empty()) {
        throw std::runtime_error("[probe_config] " + field + " must resolve to a non-empty device array");
    }

    std::set<std::string> seen_devices;
    std::vector<std::string> devices;
    for (size_t i = 0; i < devices_json.size(); i++) {
        std::string device_id = device_id_to_string(devices_json.at(i), field);
        if (!seen_devices.insert(device_id).second) {
            throw std::runtime_error("[probe_config] " + field + " contains duplicate device id: " + device_id);
        }
        devices.emplace_back(std::move(device_id));
    }
    return devices;
}

std::map<std::string, std::vector<std::string>>
parse_v2_scope(const json& scope_json, const std::map<std::string, std::string>& host_ip_by_id)
{
    if (!scope_json.is_object() || scope_json.empty()) {
        throw std::runtime_error("[probe_config] probe.scope must be a non-empty object");
    }

    std::map<std::string, std::vector<std::string>> scope;
    for (const auto& [host_id, scope_entry] : scope_json.items()) {
        const auto host_it = host_ip_by_id.find(host_id);
        if (host_it == host_ip_by_id.end()) {
            throw std::runtime_error("[probe_config] probe.scope references unknown host id: " + host_id);
        }
        scope[host_it->second] = parse_v2_devices(scope_entry, "probe.scope." + host_id);
    }
    return scope;
}

int read_schema_version(const json& root)
{
    if (!root.contains("schema_version")) {
        return 1;
    }
    if (!root.at("schema_version").is_number_integer()) {
        throw std::runtime_error("[probe_config] schema_version must be integer");
    }
    return root.at("schema_version").get<int>();
}

} // namespace

std::string format_probe_scope(const std::map<std::string, std::vector<std::string>>& scope)
{
    std::ostringstream oss;
    oss << "{";
    bool first_host = true;
    for (const auto& [host_ip, devices] : scope) {
        if (!first_host) {
            oss << ", ";
        }
        first_host = false;
        oss << host_ip << ":[";
        for (size_t i = 0; i < devices.size(); i++) {
            if (i != 0) {
                oss << ",";
            }
            oss << devices[i];
        }
        oss << "]";
    }
    oss << "}";
    return oss.str();
}

json make_control_topo_json(const std::map<std::string, std::vector<std::string>>& scope)
{
    json root = json::object();
    root["control_topo"] = json::object();
    for (const auto& [host_ip, devices] : scope) {
        root["control_topo"][host_ip] = devices;
    }
    return root;
}

std::string make_probe_topo_json_path(const std::string& project_path, const ProbeRuntimeConfig& config)
{
    std::filesystem::path output_root = std::filesystem::path(project_path) / "output";
    if (config.enabled && !config.probe_topo_tracert.output_subdir.empty()) {
        output_root /= config.probe_topo_tracert.output_subdir;
    }
    return (output_root / "probe_topo.json").string();
}

std::string make_probe_topo_lldp_json_path(const std::string& project_path, const ProbeRuntimeConfig& config)
{
    std::filesystem::path output_root = std::filesystem::path(project_path) / "output";
    if (config.enabled && !config.probe_topo_tracert.output_subdir.empty()) {
        output_root /= config.probe_topo_tracert.output_subdir;
    }
    return (output_root / "probe_topo_lldp.json").string();
}

std::string make_allpath_json_path(const std::string& project_path, const ProbeRuntimeConfig& config)
{
    std::filesystem::path output_root = std::filesystem::path(project_path) / "output";
    if (config.enabled && !config.probe_topo_tracert.output_subdir.empty()) {
        output_root /= config.probe_topo_tracert.output_subdir;
    }
    const std::string filename = config.enabled ? config.probe_topo_tracert.allpath_output : "allpath.json";
    return (output_root / filename).string();
}

std::string make_l2_path_json_path(const std::string& project_path, const ProbeRuntimeConfig& config)
{
    std::filesystem::path output_root = std::filesystem::path(project_path) / "output";
    if (config.enabled && !config.probe_topo_tracert.output_subdir.empty()) {
        output_root /= config.probe_topo_tracert.output_subdir;
    }
    const std::string filename = config.enabled ? config.probe_topo_tracert.l2_path_output : "l2_fullmesh_path.json";
    return (output_root / filename).string();
}

ProbeRuntimeConfig load_probe_runtime_config(const std::string& json_path)
{
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("[probe_config] JSON file unavailable: " + json_path);
    }

    json root;
    try {
        root = json::parse(ifs);
    } catch (const std::exception& e) {
        throw std::runtime_error("[probe_config] JSON parse error: " + std::string(e.what()));
    }

    const int schema_version = read_schema_version(root);
    if (schema_version != 1 && schema_version != 2) {
        throw std::runtime_error("[probe_config] unsupported schema_version: " + std::to_string(schema_version));
    }

    if (schema_version == 2) {
        if (!root.contains("probe")) {
            return ProbeRuntimeConfig{};
        }
        const json& probe = root.at("probe");
        if (!probe.is_object()) {
            throw std::runtime_error("[probe_config] probe must be object");
        }
        if (!probe.contains("scope")) {
            throw std::runtime_error("[probe_config] missing probe.scope");
        }

        const auto host_ip_by_id = parse_v2_host_ip_map(root);
        const json& topology = probe.value("topology", json::object());
        const json& pingpong = probe.value("pingpong", json::object());

        ProbeRuntimeConfig config;
        config.enabled = true;
        config.probe_topo_tracert.scope = parse_v2_scope(probe.at("scope"), host_ip_by_id);
        config.probe_topo_tracert.sport_begin = read_positive_int(topology, "sport_begin", 49152, "probe.topology");
        config.probe_topo_tracert.sport_count = read_positive_int(topology, "sport_count", 1, "probe.topology");
        config.probe_topo_tracert.tree_probe_sport_count
            = read_positive_int(topology, "tree_probe_sport_count", 1, "probe.topology");
        config.probe_topo_tracert.topology_optimized = topology.value("topology_optimized", true);
        config.probe_topo_tracert.l2_path_aware = topology.value("l2_path_aware", true);
        config.probe_topo_tracert.output_subdir = read_output_subdir(topology, "probe.topology");
        if (topology.contains("allpath_output")) {
            config.probe_topo_tracert.allpath_output
                = read_output_filename(topology, "allpath_output", "allpath.json", "probe.topology");
        }
        if (topology.contains("l2_path_output")) {
            config.probe_topo_tracert.l2_path_output
                = read_output_filename(topology, "l2_path_output", "l2_fullmesh_path.json", "probe.topology");
        }

        config.probe_controller_pingpong.scope = config.probe_topo_tracert.scope;
        config.probe_controller_pingpong.times = read_positive_int(pingpong, "times", 50, "probe.pingpong");
        config.probe_controller_pingpong.turns = read_positive_int(pingpong, "turns", 1000, "probe.pingpong");
        config.probe_controller_pingpong.payload_len
            = read_int_in_range(pingpong, "payload_len", 12, 1, 1500, "probe.pingpong");
        config.probe_controller_pingpong.interval_ms = read_positive_int(pingpong, "interval_ms", 1, "probe.pingpong");

        return config;
    }

    const bool has_probe_topo = root.contains("probe_topo") && root.at("probe_topo").contains("tracert");
    const bool has_probe_controller
        = root.contains("probe_controller") && root.at("probe_controller").contains("pingpong");
    if (!has_probe_topo && !has_probe_controller) {
        return ProbeRuntimeConfig{};
    }
    if (has_probe_topo != has_probe_controller) {
        throw std::runtime_error(
            "[probe_config] probe_topo.tracert and probe_controller.pingpong must appear together");
    }

    const json& tracert = root.at("probe_topo").at("tracert");
    const json& pingpong = root.at("probe_controller").at("pingpong");
    if (!root.contains("probe_scope")) {
        throw std::runtime_error("[probe_config] missing probe_scope");
    }

    const json& probe_scope_json = root.at("probe_scope");

    ProbeRuntimeConfig config;
    config.enabled = true;
    config.probe_topo_tracert.scope = parse_scope(probe_scope_json, "probe_scope");
    config.probe_topo_tracert.sport_begin = read_positive_int(tracert, "sport_begin", 49152, "probe_topo.tracert");
    config.probe_topo_tracert.sport_count = read_positive_int(tracert, "sport_count", 1, "probe_topo.tracert");
    config.probe_topo_tracert.tree_probe_sport_count
        = read_positive_int(tracert, "tree_probe_sport_count", 1, "probe_topo.tracert");
    config.probe_topo_tracert.topology_optimized = tracert.value("topology_optimized", true);
    config.probe_topo_tracert.l2_path_aware = tracert.value("l2_path_aware", true);
    config.probe_topo_tracert.output_subdir = read_output_subdir(tracert, "probe_topo.tracert");
    if (tracert.contains("allpath_output")) {
        config.probe_topo_tracert.allpath_output
            = read_output_filename(tracert, "allpath_output", "allpath.json", "probe_topo.tracert");
    }
    if (tracert.contains("l2_path_output")) {
        config.probe_topo_tracert.l2_path_output
            = read_output_filename(tracert, "l2_path_output", "l2_fullmesh_path.json", "probe_topo.tracert");
    }

    config.probe_controller_pingpong.scope = config.probe_topo_tracert.scope;
    config.probe_controller_pingpong.times = read_positive_int(pingpong, "times", 50, "probe_controller.pingpong");
    config.probe_controller_pingpong.turns = read_positive_int(pingpong, "turns", 1000, "probe_controller.pingpong");
    config.probe_controller_pingpong.payload_len
        = read_int_in_range(pingpong, "payload_len", 12, 1, 1500, "probe_controller.pingpong");
    config.probe_controller_pingpong.interval_ms
        = read_positive_int(pingpong, "interval_ms", 1, "probe_controller.pingpong");

    return config;
}
