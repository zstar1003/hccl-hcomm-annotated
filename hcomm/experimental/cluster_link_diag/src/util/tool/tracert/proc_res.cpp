/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <iomanip>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

struct DeviceInfo {
    std::string host_name;
    std::string host_ip;
    std::string dev_name;
    std::string device_ip;
    std::string lldp_mgmt_ip;
    int device_id;
};

struct ProbeRecord {
    std::string src_ip;
    std::string dst_ip;
    int sport = -1;
    std::vector<std::string> hops;
};

struct PathInfo {
    std::vector<std::string> path;
    std::vector<std::string> middle;
    std::set<int> sports;
    int first_sport = -1;
};

struct ProcConfig {
    int multipath_num = 0;
    std::vector<DeviceInfo> device_table;
    bool topology_optimization = false;
    std::vector<std::vector<int>> tor_groups;
};

static const int SPORT_BEGIN = 49152;
static const fs::path DEFAULT_INFO_JSON = fs::path("../../../../control_json") / "910b2_info.json";
static const fs::path OUTPUT_JSON = fs::path("../../../../output") / "topo.json";

std::vector<std::vector<int>> default_tor_groups() { return {{0, 1, 6, 7}, {2, 3, 4, 5}}; }

std::string trim(std::string value)
{
    auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return "";
    auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string run_hccn_tool_command(const std::string& cmd)
{
    std::array<char, 4096> buffer{};
    std::string output;

    auto close_pipe = [](FILE* pipe) {
        if (pipe)
            pclose(pipe);
    };
    std::unique_ptr<FILE, decltype(close_pipe)> pipe(popen((cmd + " 2>&1").c_str(), "r"), close_pipe);
    if (!pipe)
        throw std::runtime_error("[proc_res] popen failed");

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
        output += buffer.data();
    return output;
}

std::string resolve_device_ip(int device_id)
{
    std::ostringstream cmd;
    cmd << "hccn_tool -i " << device_id << " -ip -g";
    std::string output = run_hccn_tool_command(cmd.str());

    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        line = trim(line);
        if (line.rfind("ipaddr:", 0) == 0)
            return trim(line.substr(std::string("ipaddr:").size()));
    }

    throw std::runtime_error("failed to resolve device ip for device_id=" + std::to_string(device_id));
}

std::string resolve_lldp_management_ip(int device_id)
{
    std::ostringstream cmd;
    cmd << "hccn_tool -i " << device_id << " -lldp -g";
    std::string output = run_hccn_tool_command(cmd.str());

    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        line = trim(line);
        if (line.rfind("IPv4:", 0) == 0)
            return trim(line.substr(std::string("IPv4:").size()));
    }

    return "";
}

ProcConfig load_proc_config(const fs::path& json_path)
{
    std::ifstream ifs(json_path);
    if (!ifs.is_open())
        throw std::runtime_error("failed to open json: " + json_path.string());

    json info = json::parse(ifs);
    ProcConfig config;
    config.multipath_num = info.at("multipath_num").get<int>();
    config.topology_optimization = info.value("topology_optimization", false);
    if (config.topology_optimization) {
        config.tor_groups = default_tor_groups();
        if (info.contains("tor_groups")) {
            config.tor_groups.clear();
            for (const auto& group_json : info.at("tor_groups")) {
                std::vector<int> group;
                for (const auto& device_id_json : group_json) {
                    if (device_id_json.is_number_integer())
                        group.push_back(device_id_json.get<int>());
                    else
                        group.push_back(std::stoi(device_id_json.get<std::string>()));
                }
                if (!group.empty())
                    config.tor_groups.push_back(std::move(group));
            }
        }
    }
    if (config.multipath_num <= 0)
        throw std::runtime_error("invalid multipath_num in json: " + std::to_string(config.multipath_num));

    const auto& control_topo_id = info.at("control_topo_id");
    int host_index = 0;
    for (const auto& [host_ip, device_ids] : control_topo_id.items()) {
        ++host_index;
        std::string host_name = "host" + std::to_string(host_index);
        for (const auto& device_id_json : device_ids) {
            int device_id = 0;
            if (device_id_json.is_number_integer())
                device_id = device_id_json.get<int>();
            else
                device_id = std::stoi(device_id_json.get<std::string>());

            config.device_table.push_back(DeviceInfo{
                host_name,
                host_ip,
                "dev" + std::to_string(device_id),
                resolve_device_ip(device_id),
                resolve_lldp_management_ip(device_id),
                device_id,
            });
        }
    }

    return config;
}

std::vector<std::string> split(const std::string& line, char delimiter)
{
    std::vector<std::string> parts;
    std::stringstream ss(line);
    std::string part;
    while (std::getline(ss, part, delimiter))
        parts.emplace_back(part);
    return parts;
}

bool consume_prefix(const std::string& text, const std::string& prefix, std::string& value)
{
    if (text.rfind(prefix, 0) != 0)
        return false;
    value = text.substr(prefix.size());
    return true;
}

std::string join(const std::vector<std::string>& items, const std::string& delimiter)
{
    std::ostringstream oss;
    for (size_t i = 0; i < items.size(); i++) {
        if (i != 0)
            oss << delimiter;
        oss << items[i];
    }
    return oss.str();
}

std::string format_sports(const std::set<int>& sports)
{
    if (sports.empty())
        return "";

    std::ostringstream oss;
    auto it = sports.begin();
    int begin = *it;
    int last = *it;
    ++it;

    auto flush_range = [&]() {
        if (oss.tellp() > 0)
            oss << ",";
        if (begin == last)
            oss << begin;
        else
            oss << begin << "-" << last;
    };

    for (; it != sports.end(); ++it) {
        if (*it == last + 1) {
            last = *it;
            continue;
        }
        flush_range();
        begin = *it;
        last = *it;
    }
    flush_range();
    return oss.str();
}

std::map<std::string, const DeviceInfo*> build_ip_index(const std::vector<DeviceInfo>& device_table)
{
    std::map<std::string, const DeviceInfo*> index;
    for (const auto& device : device_table)
        index[device.device_ip] = &device;
    return index;
}

std::vector<const DeviceInfo*>
devices_on_host(const std::vector<DeviceInfo>& device_table, const std::string& host_name)
{
    std::vector<const DeviceInfo*> devices;
    for (const auto& device : device_table) {
        if (device.host_name == host_name)
            devices.emplace_back(&device);
    }
    return devices;
}

std::vector<std::string> known_hosts(const std::vector<DeviceInfo>& device_table)
{
    std::vector<std::string> hosts;
    for (const auto& device : device_table) {
        if (std::find(hosts.begin(), hosts.end(), device.host_name) == hosts.end())
            hosts.emplace_back(device.host_name);
    }
    return hosts;
}

bool parse_probe_line(const std::string& line, ProbeRecord& record)
{
    if (line.rfind("<srcIp>", 0) != 0)
        return false;

    std::vector<std::string> parts = split(line, ',');
    if (parts.size() < 4)
        return false;

    std::string sport_text;
    if (!consume_prefix(parts[0], "<srcIp>", record.src_ip) || !consume_prefix(parts[1], "<tarIP>", record.dst_ip)
        || !consume_prefix(parts[2], "<sport>", sport_text)) {
        return false;
    }

    record.sport = std::stoi(sport_text);
    record.hops.assign(parts.begin() + 3, parts.end());
    return true;
}

std::vector<std::string> middle_hops(const std::vector<std::string>& hops)
{
    if (hops.empty())
        return {};
    return std::vector<std::string>(hops.begin(), hops.end() - 1);
}

std::vector<std::string> full_path(const ProbeRecord& record)
{
    std::vector<std::string> path;
    path.emplace_back(record.src_ip);
    path.insert(path.end(), record.hops.begin(), record.hops.end());
    return path;
}

using PathList = std::vector<PathInfo>;
using DstDevMap = std::map<std::string, PathList>;
using DstHostMap = std::map<std::string, DstDevMap>;
using TopoMap = std::map<std::string, DstHostMap>;

void insert_record(
    const ProbeRecord& record, const std::map<std::string, const DeviceInfo*>& ip_index, TopoMap& topo_paths,
    std::map<std::string, std::map<std::string, size_t>>& path_index)
{
    const auto src_it = ip_index.find(record.src_ip);
    const auto dst_it = ip_index.find(record.dst_ip);
    if (src_it == ip_index.end() || dst_it == ip_index.end()) {
        std::cerr << "skip unknown device ip: src=" << record.src_ip << " dst=" << record.dst_ip << std::endl;
        return;
    }

    const DeviceInfo& src = *src_it->second;
    const DeviceInfo& dst = *dst_it->second;
    std::vector<std::string> middle = middle_hops(record.hops);
    std::string flow_key = src.device_ip + "->" + dst.device_ip;
    std::string path_key = join(middle, "|");

    PathList& paths = topo_paths[src.dev_name][dst.host_name][dst.dev_name];
    auto& index = path_index[flow_key];
    const auto path_it = index.find(path_key);
    if (path_it == index.end()) {
        PathInfo info;
        info.path = full_path(record);
        info.middle = std::move(middle);
        info.sports.insert(record.sport);
        info.first_sport = record.sport;
        index[path_key] = paths.size();
        paths.emplace_back(std::move(info));
        return;
    }

    paths[path_it->second].sports.insert(record.sport);
}

void read_result_file(
    const fs::path& path, const std::map<std::string, const DeviceInfo*>& ip_index, TopoMap& topo_paths,
    std::map<std::string, std::map<std::string, size_t>>& path_index)
{
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "failed to open result file: " << path << std::endl;
        return;
    }

    std::string line;
    while (std::getline(in, line)) {
        ProbeRecord record;
        if (!parse_probe_line(line, record))
            continue;
        insert_record(record, ip_index, topo_paths, path_index);
    }
}

std::vector<fs::path> existing_result_roots(int multipath_num)
{
    std::vector<fs::path> roots;
    const fs::path range_root = fs::path("testfile") / "res"
                                / (std::to_string(SPORT_BEGIN) + "_" + std::to_string(SPORT_BEGIN + multipath_num - 1));
    if (!fs::exists(range_root) || !fs::is_directory(range_root))
        return roots;

    fs::path latest_run_root;
    std::string latest_name;
    for (const auto& entry : fs::directory_iterator(range_root)) {
        if (!entry.is_directory())
            continue;

        const std::string dir_name = entry.path().filename().string();
        if (dir_name > latest_name) {
            latest_name = dir_name;
            latest_run_root = entry.path();
        }
    }

    if (!latest_run_root.empty())
        roots.emplace_back(latest_run_root);
    return roots;
}

std::vector<const DeviceInfo*>
source_devices_with_results(const std::vector<DeviceInfo>& device_table, const std::vector<fs::path>& result_roots)
{
    std::vector<const DeviceInfo*> sources;
    for (const auto& device : device_table) {
        for (const auto& root : result_roots) {
            fs::path file = root / device.host_name / ("npu" + std::to_string(device.device_id) + ".txt");
            if (fs::exists(file)) {
                sources.emplace_back(&device);
                break;
            }
        }
    }
    return sources;
}

json path_to_json(const PathInfo& path)
{
    json item = json::object();
    item["path"] = path.path;
    item["middle"] = path.middle;
    item["sports"] = format_sports(path.sports);
    item["sport_count"] = path.sports.size();
    return item;
}

std::string device_label(const DeviceInfo& device) { return device.host_name + "/" + device.dev_name; }

std::string switch_group_key(const DeviceInfo& device)
{
    if (!device.lldp_mgmt_ip.empty())
        return device.lldp_mgmt_ip;
    return "unknown:" + device_label(device);
}

std::string tor_group_key(const DeviceInfo& device, const std::vector<std::vector<int>>& tor_groups)
{
    for (const auto& group : tor_groups) {
        if (std::find(group.begin(), group.end(), device.device_id) != group.end()) {
            std::ostringstream oss;
            oss << "tor_group:";
            for (size_t i = 0; i < group.size(); ++i) {
                if (i != 0)
                    oss << "-";
                oss << group[i];
            }
            return oss.str();
        }
    }
    return "";
}

std::string first_hop_node_id(const DeviceInfo& device, const PathInfo& path, const ProcConfig& config)
{
    if (config.topology_optimization) {
        const std::string tor_key = tor_group_key(device, config.tor_groups);
        if (!tor_key.empty())
            return "leaf:" + tor_key;
    }

    if (!device.lldp_mgmt_ip.empty())
        return "leaf:" + device.lldp_mgmt_ip;
    if (!path.middle.empty())
        return "leaf:" + path.middle.front();
    return "leaf:unknown:" + device_label(device);
}

std::string first_hop_label(const DeviceInfo& device, const PathInfo& path, const ProcConfig& config)
{
    if (config.topology_optimization) {
        const std::string tor_key = tor_group_key(device, config.tor_groups);
        if (!tor_key.empty())
            return tor_key;
    }

    if (!device.lldp_mgmt_ip.empty())
        return device.lldp_mgmt_ip;
    if (!path.middle.empty())
        return path.middle.front();
    return "unknown";
}

json build_path_aggregated_topology(
    const ProcConfig& config, const std::vector<const DeviceInfo*>& sources, const TopoMap& topo_paths)
{
    json nodes = json::object();
    json edges = json::object();
    json flows = json::array();

    auto ensure_device_node = [&](const DeviceInfo& device) {
        const std::string node_id = "device:" + device_label(device);
        if (nodes.contains(node_id))
            return node_id;

        nodes[node_id] = {
            {"id", node_id},
            {"type", "device"},
            {"label", device.dev_name},
            {"host_name", device.host_name},
            {"host_ip", device.host_ip},
            {"device_ip", device.device_ip},
            {"device_id", device.device_id},
        };
        return node_id;
    };

    auto ensure_leaf_node = [&](const DeviceInfo& device, const PathInfo& path) {
        const std::string node_id = first_hop_node_id(device, path, config);
        if (nodes.contains(node_id))
            return node_id;

        nodes[node_id] = {
            {"id", node_id},
            {"type", "first_hop_switch"},
            {"label", first_hop_label(device, path, config)},
            {"switch_ip", device.lldp_mgmt_ip},
        };
        return node_id;
    };

    auto ensure_ip_hop_node = [&](const std::string& hop_ip) {
        const std::string node_id = "hop:" + hop_ip;
        if (nodes.contains(node_id))
            return node_id;

        nodes[node_id] = {
            {"id", node_id},
            {"type", "intermediate_ip"},
            {"label", hop_ip},
            {"ip", hop_ip},
        };
        return node_id;
    };

    auto add_edge = [&](const std::string& from, const std::string& to, const std::set<int>& sports) {
        const std::string edge_key = from + "->" + to;
        if (!edges.contains(edge_key)) {
            edges[edge_key] = {
                {"from", from},
                {"to", to},
                {"sports", json::array()},
            };
        }

        auto& sports_json = edges[edge_key]["sports"];
        std::set<int> merged;
        for (const auto& item : sports_json)
            merged.insert(item.get<int>());
        merged.insert(sports.begin(), sports.end());

        json new_sports = json::array();
        for (int sport : merged)
            new_sports.push_back(sport);
        edges[edge_key]["sports"] = std::move(new_sports);
        edges[edge_key]["sport_count"] = merged.size();
    };

    for (const DeviceInfo* src : sources) {
        const auto src_it = topo_paths.find(src->dev_name);
        if (src_it == topo_paths.end())
            continue;

        const std::string src_node = ensure_device_node(*src);
        for (const std::string& host : known_hosts(config.device_table)) {
            for (const DeviceInfo* dst : devices_on_host(config.device_table, host)) {
                if (src->device_ip == dst->device_ip)
                    continue;

                const auto host_it = src_it->second.find(dst->host_name);
                if (host_it == src_it->second.end())
                    continue;

                const auto dst_it = host_it->second.find(dst->dev_name);
                if (dst_it == host_it->second.end())
                    continue;

                const std::string dst_node = ensure_device_node(*dst);
                PathList paths = dst_it->second;
                std::sort(paths.begin(), paths.end(), [](const PathInfo& lhs, const PathInfo& rhs) {
                    return lhs.first_sport < rhs.first_sport;
                });

                for (const auto& path : paths) {
                    std::vector<std::string> normalized_nodes;
                    normalized_nodes.push_back(src_node);
                    normalized_nodes.push_back(ensure_leaf_node(*src, path));

                    for (size_t i = 1; i < path.middle.size(); ++i)
                        normalized_nodes.push_back(ensure_ip_hop_node(path.middle[i]));

                    normalized_nodes.push_back(dst_node);

                    for (size_t i = 0; i + 1 < normalized_nodes.size(); ++i)
                        add_edge(normalized_nodes[i], normalized_nodes[i + 1], path.sports);

                    json flow = json::object();
                    flow["src"] = src_node;
                    flow["dst"] = dst_node;
                    flow["src_device"] = device_label(*src);
                    flow["dst_device"] = device_label(*dst);
                    flow["sports"] = format_sports(path.sports);
                    flow["sport_count"] = path.sports.size();
                    flow["render_path"] = json::array();
                    for (const auto& node_id : normalized_nodes)
                        flow["render_path"].push_back(node_id);
                    flow["raw_path"] = path.path;
                    flow["raw_middle"] = path.middle;
                    flows.push_back(std::move(flow));
                }
            }
        }
    }

    json edge_list = json::array();
    for (const auto& [_, edge] : edges.items())
        edge_list.push_back(edge);

    json node_list = json::array();
    for (const auto& [_, node] : nodes.items())
        node_list.push_back(node);

    json root = json::object();
    root["rule"] = {
        {"first_hop",
         config.topology_optimization ? "grouped_by_tor_group_then_lldp" : "grouped_by_lldp_management_ip"},
        {"later_non_last_hops", "grouped_by_hop_ip"},
        {"last_hop", "kept_as_target_device"},
    };
    root["nodes"] = std::move(node_list);
    root["edges"] = std::move(edge_list);
    root["flows"] = std::move(flows);
    return root;
}

json build_json(
    const std::vector<DeviceInfo>& device_table, const std::vector<const DeviceInfo*>& sources,
    const TopoMap& topo_paths)
{
    json root = json::object();
    for (const DeviceInfo* src : sources) {
        json src_json = json::object();
        const auto src_it = topo_paths.find(src->dev_name);

        for (const std::string& host : known_hosts(device_table)) {
            json host_json = json::object();
            for (const DeviceInfo* dst : devices_on_host(device_table, host)) {
                json dst_json = json::object();
                if (src->device_ip != dst->device_ip && src_it != topo_paths.end()) {
                    const auto host_it = src_it->second.find(dst->host_name);
                    if (host_it != src_it->second.end()) {
                        const auto dst_it = host_it->second.find(dst->dev_name);
                        if (dst_it != host_it->second.end()) {
                            PathList paths = dst_it->second;
                            std::sort(paths.begin(), paths.end(), [](const PathInfo& lhs, const PathInfo& rhs) {
                                return lhs.first_sport < rhs.first_sport;
                            });

                            for (size_t i = 0; i < paths.size(); i++)
                                dst_json["path" + std::to_string(i + 1)] = path_to_json(paths[i]);
                        }
                    }
                }

                host_json[dst->dev_name] = dst_json;
            }
            src_json[host] = host_json;
        }
        root[src->dev_name] = src_json;
    }
    return root;
}

json build_aggregated_switch_topology(
    const std::vector<DeviceInfo>& device_table, const std::vector<const DeviceInfo*>& sources,
    const TopoMap& topo_paths)
{
    json root = json::object();

    for (const DeviceInfo* src : sources) {
        const std::string src_switch = switch_group_key(*src);
        json& src_switch_json = root[src_switch];
        if (src_switch_json.empty()) {
            src_switch_json = json::object();
            src_switch_json["switch_ip"] = src->lldp_mgmt_ip;
            src_switch_json["source_members"] = json::array();
            src_switch_json["hosts"] = json::object();
        }

        auto& source_members = src_switch_json["source_members"];
        const std::string src_member = device_label(*src);
        if (std::find(source_members.begin(), source_members.end(), src_member) == source_members.end())
            source_members.push_back(src_member);

        const auto src_it = topo_paths.find(src->dev_name);
        for (const std::string& host : known_hosts(device_table)) {
            json& host_json = src_switch_json["hosts"][host];
            if (host_json.empty())
                host_json = json::object();

            for (const DeviceInfo* dst : devices_on_host(device_table, host)) {
                const std::string dst_switch = switch_group_key(*dst);
                json& dst_switch_json = host_json[dst_switch];
                if (dst_switch_json.empty()) {
                    dst_switch_json = json::object();
                    dst_switch_json["switch_ip"] = dst->lldp_mgmt_ip;
                    dst_switch_json["target_members"] = json::array();
                    dst_switch_json["flows"] = json::object();
                }

                auto& target_members = dst_switch_json["target_members"];
                const std::string dst_member = device_label(*dst);
                if (std::find(target_members.begin(), target_members.end(), dst_member) == target_members.end())
                    target_members.push_back(dst_member);

                if (src->device_ip == dst->device_ip || src_it == topo_paths.end())
                    continue;

                const auto host_it = src_it->second.find(dst->host_name);
                if (host_it == src_it->second.end())
                    continue;

                const auto dst_it = host_it->second.find(dst->dev_name);
                if (dst_it == host_it->second.end())
                    continue;

                json flow_json = json::object();
                PathList paths = dst_it->second;
                std::sort(paths.begin(), paths.end(), [](const PathInfo& lhs, const PathInfo& rhs) {
                    return lhs.first_sport < rhs.first_sport;
                });
                for (size_t i = 0; i < paths.size(); i++)
                    flow_json["path" + std::to_string(i + 1)] = path_to_json(paths[i]);

                dst_switch_json["flows"][src->dev_name + "->" + dst->dev_name] = flow_json;
            }
        }
    }

    return root;
}

int main()
{
    ProcConfig config;
    try {
        config = load_proc_config(DEFAULT_INFO_JSON);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    const auto result_roots = existing_result_roots(config.multipath_num);
    if (result_roots.empty()) {
        std::cerr << "no result root exists" << std::endl;
        return 1;
    }

    const auto ip_index = build_ip_index(config.device_table);
    const auto sources = source_devices_with_results(config.device_table, result_roots);
    if (sources.empty()) {
        std::cerr << "no npu result file found under configured result roots" << std::endl;
        return 1;
    }

    TopoMap topo_paths;
    std::map<std::string, std::map<std::string, size_t>> path_index;
    for (const auto& root : result_roots) {
        for (const DeviceInfo* src : sources) {
            fs::path file = root / src->host_name / ("npu" + std::to_string(src->device_id) + ".txt");
            if (fs::exists(file))
                read_result_file(file, ip_index, topo_paths, path_index);
        }
    }

    json topo = build_json(config.device_table, sources, topo_paths);
    topo["__aggregated_switch_topo"] = build_aggregated_switch_topology(config.device_table, sources, topo_paths);
    topo["__path_aggregated_topo"] = build_path_aggregated_topology(config, sources, topo_paths);
    fs::create_directories(OUTPUT_JSON.parent_path());
    std::ofstream out(OUTPUT_JSON);
    if (!out.is_open()) {
        std::cerr << "failed to open output json: " << OUTPUT_JSON << std::endl;
        return 1;
    }

    out << topo.dump(4) << '\n';
    std::cout << "wrote " << OUTPUT_JSON << " from";
    for (const auto& root : result_roots)
        std::cout << " " << root;
    std::cout << std::endl;
    return 0;
}
