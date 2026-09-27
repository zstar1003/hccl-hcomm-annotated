/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "probe_plan.h"

#include <algorithm>
#include <functional>
#include <limits>

namespace probe_plan {
namespace {
    int dev_sort_key(const std::string& label)
    {
        if (label.rfind("dev", 0) == 0) {
            try {
                return std::stoi(label.substr(3));
            } catch (...) {
            }
        }
        return std::numeric_limits<int>::max();
    }

    std::vector<std::string> get_domain_devices_by_order(
        const MeshTopo& mesh_topo, const std::string& mesh_name,
        const std::map<std::string, std::string>& device_label_by_ip)
    {
        std::vector<std::string> devices;
        auto mesh_it = mesh_topo.mesh_map.find(mesh_name);
        if (mesh_it == mesh_topo.mesh_map.end()) {
            return devices;
        }
        for (const auto& [device_ip, links] : mesh_it->second.kid_to_link) {
            devices.emplace_back(device_ip);
        }
        std::sort(devices.begin(), devices.end(), [&](const std::string& left, const std::string& right) {
            const auto left_it = device_label_by_ip.find(left);
            const auto right_it = device_label_by_ip.find(right);
            const std::string left_label = left_it == device_label_by_ip.end() ? "" : left_it->second;
            const std::string right_label = right_it == device_label_by_ip.end() ? "" : right_it->second;
            const int left_key = dev_sort_key(left_label);
            const int right_key = dev_sort_key(right_label);
            return left_key != right_key ? left_key < right_key : left < right;
        });
        return devices;
    }

    int mesh_sort_key(const std::string& mesh_name)
    {
        const std::string prefix = "TOPO[0,";
        if (mesh_name.rfind(prefix, 0) != 0 || mesh_name.empty() || mesh_name.back() != ']') {
            return std::numeric_limits<int>::max();
        }
        try {
            return std::stoi(mesh_name.substr(prefix.size(), mesh_name.size() - prefix.size() - 1));
        } catch (...) {
            return std::numeric_limits<int>::max();
        }
    }

    std::vector<std::string> get_l0_mesh_names_by_order(const MeshTopo& mesh_topo)
    {
        std::vector<std::string> names;
        for (const auto& [name, mesh] : mesh_topo.mesh_map) {
            if (mesh.level == 0) {
                names.emplace_back(name);
            }
        }
        std::sort(names.begin(), names.end(), [](const std::string& left, const std::string& right) {
            const int left_key = mesh_sort_key(left);
            const int right_key = mesh_sort_key(right);
            return left_key != right_key ? left_key < right_key : left < right;
        });
        return names;
    }

    void append_l2_fullmesh_tasks(
        std::vector<ProbeHelper::ExplicitPingTask>& tasks, const std::vector<std::string>& from_devices,
        const std::vector<std::string>& to_devices, const std::map<std::string, std::string>& labels,
        const std::string& tag_prefix, int src_port)
    {
        for (const auto& from_ip : from_devices) {
            const auto from_it = labels.find(from_ip);
            const std::string from_label = from_it == labels.end() ? from_ip : from_it->second;
            for (const auto& to_ip : to_devices) {
                const auto to_it = labels.find(to_ip);
                const std::string to_label = to_it == labels.end() ? to_ip : to_it->second;
                const auto duplicate = std::any_of(tasks.begin(), tasks.end(), [&](const auto& task) {
                    return task.from_ip == from_ip && task.to_ip == to_ip && task.src_port == src_port;
                });
                if (!duplicate) {
                    tasks.push_back({from_ip, to_ip, src_port, tag_prefix + ":" + from_label + "->" + to_label});
                }
            }
        }
    }
} // namespace

PingList build_l0_pinglist_from_lldp_topo(MeshTopo mesh_topo)
{
    PingList pinglist;
    for (auto& [mesh_name, mesh] : mesh_topo.mesh_map) {
        const auto kids = mesh.get_kids();
        auto& tasks = pinglist[mesh_name];
        if (kids.size() < 2) {
            continue;
        }
        if (kids.size() == 2) {
            tasks.emplace_back(kids[0], kids[1], 1);
            continue;
        }
        const auto ring = mesh.get_ring();
        const size_t node_count = ring.empty() ? 0 : ring.size() - 1;
        if (node_count % 2 == 1) {
            for (size_t i = 0; i + 1 < ring.size(); ++i) {
                tasks.emplace_back(ring[i], ring[i + 1], 1);
            }
            continue;
        }

        const size_t core_count = node_count - 1;
        for (size_t i = 0; i + 1 < core_count; ++i) {
            tasks.emplace_back(ring[i], ring[i + 1], 1);
        }
        tasks.emplace_back(ring[core_count - 1], ring[0], 1);

        const size_t attach_index = std::hash<std::string>{}(mesh_name) % core_count;
        tasks.emplace_back(ring[node_count - 1], ring[attach_index], 1);
    }
    return pinglist;
}

std::vector<ProbeHelper::ExplicitPingTask> build_l2_fullmesh_ping_tasks(
    const MeshTopo& mesh_topo, const std::map<std::string, std::string>& device_label_by_ip, int src_port)
{
    const auto names = get_l0_mesh_names_by_order(mesh_topo);
    if (names.size() < 2) {
        return {};
    }

    std::vector<ProbeHelper::ExplicitPingTask> tasks;
    for (size_t i = 0; i < names.size(); ++i) {
        const auto& from_mesh = names[i];
        const auto& to_mesh = names[(i + 1) % names.size()];
        const auto from_devices = get_domain_devices_by_order(mesh_topo, from_mesh, device_label_by_ip);
        const auto to_devices = get_domain_devices_by_order(mesh_topo, to_mesh, device_label_by_ip);
        if (from_devices.empty() || to_devices.empty()) {
            continue;
        }
        append_l2_fullmesh_tasks(
            tasks, from_devices, to_devices, device_label_by_ip, "l2-fullmesh:" + from_mesh + "->" + to_mesh, src_port);
    }
    return tasks;
}
} // namespace probe_plan
