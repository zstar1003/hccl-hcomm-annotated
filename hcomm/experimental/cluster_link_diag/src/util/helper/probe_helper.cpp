/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "probe_helper.h"

#include <Eigen/Dense>

#include <limits>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <cmath>

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "tool/tracert/tracert.h"
#include "tool/hccn_pingpong/rpc_func.h"

ProbeHelper::ProbeHelper(
    ControlTopo* ct, HccnDeviceIpListParallel hccn_device_ip_list_parallel,
    TracertPortsMultiParallel tracert_ports_multi_parallel, PinglistInsertMutiParallel pinglist_insert_muti_parallel,
    UdPingpongTxMutiParallel ud_pingpong_tx_muti_parallel)
    : ct(ct)
{
    this->hccn_device_ip_list_parallel = std::move(hccn_device_ip_list_parallel);
    this->tracert_ports_multi_parallel = tracert_ports_multi_parallel;
    this->pinglist_insert_muti_parallel = pinglist_insert_muti_parallel;
    this->ud_pingpong_tx_muti_parallel = ud_pingpong_tx_muti_parallel;

    device_control_list = ct->LeafList();
    int n = device_control_list.size();
    /*通过 hccn_tool -i <devId> -ip -g 得到 HCCN 设备 IP*/
    std::vector<std::string> server_list;
    std::vector<int> server_device_count_list;
    for (const auto& server : ct->ServerTreeList()) {
        server_list.emplace_back(server.ip);
        server_device_count_list.emplace_back(static_cast<int>(server.kids.size()));
    }

    auto device_ip_group_list = this->hccn_device_ip_list_parallel(server_list, server_device_count_list);
    for (auto& device_ip_group : device_ip_group_list) {
        for (auto& device_ip : device_ip_group) {
            device_ip_list.emplace_back(std::move(device_ip));
        }
    }

    const size_t mapped_device_count = std::min(device_ip_list.size(), device_control_list.size());
    for (size_t i = 0; i < mapped_device_count; i++) {
        device_ip_to_control[device_ip_list[i]] = device_control_list[i];
    }
}

int ProbeHelper::get_path_level(const std::vector<std::string>& path)
{
    int len = path.size();
    for (int i = 0; i < level_trace_len.size(); i++) {
        if (len <= level_trace_len[i]) // or == ?
        {
            return i;
        }
    }
    return level_trace_len.size();
};

int ProbeHelper::get_level_trace_port_num(int level) const
{
    if (level_trace_port_num.empty()) {
        return 1;
    }
    if (level < 0) {
        return level_trace_port_num.front();
    }
    if (static_cast<size_t>(level) < level_trace_port_num.size()) {
        return level_trace_port_num[level];
    }
    return level_trace_port_num.back();
}

void ProbeHelper::set_tracert_port_config(int tree_probe_sport_count, int fullmesh_sport_count)
{
    if (tree_probe_sport_count <= 0 || fullmesh_sport_count <= 0) {
        throw std::runtime_error("[probe_topo] tracert sport counts must be positive");
    }
    this->tree_probe_sport_count = tree_probe_sport_count;
    level_trace_port_num = {fullmesh_sport_count};
}

std::string ProbeHelper::get_level_link_ip(const std::vector<std::string>& path, int level, bool near_ip)
{
    int index = !near_ip ? level_ip_index[level] : path.size() - level_ip_index[level] - 1;
    if (index >= path.size()) {
        throw std::runtime_error("[get_level_link]index out of range");
    }
    return path[index];
};

std::vector<int>
ProbeHelper::route_to_ids(const std::vector<std::string>& path, const std::string& from, const std::string& to)
{
    int level_size = get_path_level(path) + 1;
    int len = path.size();
    std::string from_kid_name = from;
    std::string to_kid_name = to;
    std::vector<int> from_ids;
    std::vector<int> to_ids;
    for (int i = 0; i < level_size - (len % 2); i++) {
        auto from_ip = get_level_link_ip(path, i, false);
        auto from_father_name = meshTopo.get_mesh_father(from_kid_name);
        auto from_mesh = meshTopo.get_mesh(from_father_name);
        for (auto& [l_ip, r_ip, index] : from_mesh.kid_to_link[from_kid_name]) {
            if (from_ip.empty() || r_ip.empty() || from_ip == r_ip) {
                from_ids.emplace_back(index);
                break;
            }
        }

        auto to_ip = get_level_link_ip(path, i, true);
        auto to_father_name = meshTopo.get_mesh_father(to_kid_name);
        auto to_mesh = meshTopo.get_mesh(to_father_name);
        for (auto& [l_ip, r_ip, index] : to_mesh.kid_to_link[to_kid_name]) {
            if (to_ip.empty() || l_ip.empty() || to_ip == l_ip) {
                to_ids.emplace_back(index);
                break;
            }
        }

        from_kid_name = from_father_name;
        to_kid_name = to_father_name;
    }

    if (len % 2) {
        auto father_name = meshTopo.get_mesh_father(from_kid_name);
        auto father_mesh = meshTopo.get_mesh(father_name);
        auto ip = path[len / 2];
        for (auto& link : father_mesh.kid_to_kid_link[from_kid_name][to_kid_name]) {
            if (ip == std::get<1>(link)) {
                from_ids.emplace_back(std::get<2>(link));
                break;
            }
        }
    }
    from_ids.insert(from_ids.end(), to_ids.rbegin(), to_ids.rend());
    return from_ids;
}

void ProbeHelper::gen_topo()
{
    int n = device_control_list.size();
    if (n == 0) {
        throw std::runtime_error("[probe_topo] device control list is empty");
    }
    if (device_ip_list.empty()) {
        throw std::runtime_error("[probe_topo] device ip list is empty");
    }
    if (device_ip_list.size() != device_control_list.size()) {
        throw std::runtime_error("[probe_topo] device ip count does not match configured device count");
    }

    /*探测拓扑形态(假设每层是mesh,并且node0与node n-1最远,跨越根节点)*/
    std::vector<std::vector<std::string>> trace_target_list(n);
    for (int i = 0; i < n; i++) {
        trace_target_list[i] = std::vector<std::string>{device_ip_list[(i + 1) % n]};
    }
    auto trace_res_list = tracert_ports_multi_parallel(
        device_control_list, device_ip_list, trace_target_list,
        std::vector<std::vector<int>>(n, std::vector<int>(1, tree_probe_sport_count)));
    if (level_ip_index.empty() && level_trace_len.empty()) {
        int last_index = -1;
        int last_len = 0;
        for (auto& trace_ports_multi_res : trace_res_list) {
            auto& trace_res = trace_ports_multi_res[0][0];
            int len = trace_res.size() + trace_res.size() % 2;
            if (len > last_len) {
                for (int i = last_index + 1; i < len / 2; i++) {
                    auto& this_ip = trace_res[i];
                    if (this_ip != "*") {
                        level_ip_index.emplace_back(i);
                        last_index = i;
                        break;
                    }
                }
                level_trace_len.emplace_back(len);
                last_len = len;
            }
        }
    } else if (level_ip_index.size() != level_trace_len.size()) {
        throw std::runtime_error("[probe_topo]level_ip_index.size() != level_trace_len.size()");
    }

    /*解析topo形状*/
    int max_level = get_path_level(trace_res_list[n - 1][0][0]) + 1; // 注意,这是不可达的!
    std::vector<std::string> mesh_stack(max_level);
    std::vector<int> index_stack(max_level + 1);
    for (int i = 0; i < n; i++) {
        int this_level = i > 0 ? get_path_level(trace_res_list[i - 1][0][0]) : max_level;
        for (int j = this_level; j >= 0; j--) {
            std::string mesh_name = (j == 0) ? device_ip_list[i] : MeshTopo::get_mesh_name(j - 1, index_stack[j]);
            if (j > 0) // 创建新一级Topo
            {
                meshTopo.add_mesh(Mesh(j - 1, mesh_name));
                mesh_stack[j - 1] = mesh_name;
            }
            // 上级topo不超过stack
            if (j < max_level) {
                meshTopo.add_kid(mesh_stack[j], mesh_name);
            }
            index_stack[j]++;
        }
    }
    meshTopo.root_mesh = mesh_stack[max_level - 1];
}

MeshTopo ProbeHelper::build_lldp_topo(const std::map<std::string, std::string>& device_lldp_mgmt_ip)
{
    std::map<std::string, std::vector<std::string>> switch_to_devices;
    for (const auto& device_ip : device_ip_list) {
        auto it = device_lldp_mgmt_ip.find(device_ip);
        std::string switch_ip = it == device_lldp_mgmt_ip.end() ? "" : it->second;
        if (switch_ip.empty()) {
            switch_ip = "unknown:" + device_ip;
        }
        switch_to_devices[switch_ip].emplace_back(device_ip);
    }

    MeshTopo lldp_topo;
    lldp_device_groups.clear();
    int switch_index = 0;
    int link_index = 0;
    for (const auto& [switch_ip, devices] : switch_to_devices) {
        const std::string mesh_name = MeshTopo::get_mesh_name(0, switch_index++);
        Mesh mesh(0, mesh_name);
        lldp_device_groups.emplace_back(devices);
        for (const auto& device_ip : devices) {
            mesh.add_kid(device_ip);
            const std::string link_ip = switch_ip.rfind("unknown:", 0) == 0 ? "" : switch_ip;
            if (!link_ip.empty()) {
                mesh.add_kid_link(device_ip, link_ip, false, link_index);
            }
        }
        lldp_topo.add_mesh(mesh);
        if (lldp_topo.root_mesh.empty()) {
            lldp_topo.root_mesh = mesh_name;
        }
    }
    return lldp_topo;
}

MeshTopo& ProbeHelper::probe_topo_ring()
{
    gen_topo();
    return meshTopo;
}

MeshTopo ProbeHelper::probe_lldp_topo(
    std::function<std::vector<std::string>(const std::vector<std::string>&)> lldp_mgmt_ip_parallel)
{
    std::cout << "[lldp] rpc start, source_count=" << device_control_list.size() << std::endl;
    auto lldp_mgmt_ip_list = lldp_mgmt_ip_parallel(device_control_list);

    std::map<std::string, std::string> device_lldp_mgmt_ip;
    for (size_t i = 0; i < device_ip_list.size() && i < lldp_mgmt_ip_list.size(); i++) {
        const std::string switch_ip = lldp_mgmt_ip_list[i];
        device_lldp_mgmt_ip[device_ip_list[i]] = switch_ip;
        std::cout << "[lldp] control_dev=" << device_control_list[i] << ", device_ip=" << device_ip_list[i]
                  << ", switch_mgmt_ip=" << (switch_ip.empty() ? "<empty>" : switch_ip) << std::endl;
    }
    return build_lldp_topo(device_lldp_mgmt_ip);
}

void ProbeHelper::mesh_full_trace()
{
    // mesh探测
    // 更具层级归类meshTopo的那一层
    /*为每个mesh进行full_mesh的tracert,以便于获取链路ip*/
    for (auto& [mesh_name, mesh] : meshTopo.mesh_map) {
        int link_level = mesh.level;
        auto kids = meshTopo.get_leaf_kids(mesh); // 子节点 下面的随机卡
        auto owner_kids = mesh.get_kids();
        std::vector<std::string> device_list;
        for (auto kid : kids) {
            std::string this_control = device_ip_to_control[kid];
            device_list.emplace_back(this_control);
        }
        int kids_size = kids.size();
        int trace_port_num = get_level_trace_port_num(mesh.level);
        auto trace_ports_multi_parallel_res = tracert_ports_multi_parallel(
            device_list, kids, std::vector<std::vector<std::string>>(kids_size, kids),
            std::vector<std::vector<int>>(kids_size, std::vector<int>(kids_size, trace_port_num)));
        for (int i = 0; i < kids_size; i++) {
            std::string kid = kids[i];
            std::string owner_kid = owner_kids[i];
            auto& trace_ports_multi_res = trace_ports_multi_parallel_res[i];

            for (int j = 0; j < owner_kids.size(); j++) {
                auto& target_kid = owner_kids[j];
                auto& trace_ports_res = trace_ports_multi_res[j];
                if (target_kid != owner_kid) {
                    for (auto& this_trace : trace_ports_res) {
                        int len = this_trace.size();
                        if ((len != level_trace_len[link_level]) && (len + 1 != level_trace_len[link_level])) {
                            std::cout << "[probe_topo][Warning] path length error detected!";
                            this_trace.resize(level_trace_len[link_level]);
                        } else if (len % 2 == 0) {
                            mesh.add_kid_link(
                                owner_kid, get_level_link_ip(this_trace, link_level, false), false, global_id);
                            mesh.add_kid_link(
                                target_kid, get_level_link_ip(this_trace, link_level, true), true, global_id);
                        } else {
                            std::string kid_to_kid_ip = this_trace[len / 2];
                            mesh.add_kid_to_kid_link(owner_kid, target_kid, kid_to_kid_ip, global_id);
                        }
                    }
                }
                tracert_res[kid][kids[j]] = trace_ports_res;
            }
        }
    }
}

// 辅助构造pinglist
static void mesh_pinglist_insert(
    std::vector<std::tuple<std::string, std::string, int>>& meshlist, std::vector<std::string> from,
    std::vector<std::string> to, int port_num)
{
    int from_len = from.size();
    int to_len = to.size();
    int min_len = std::min(from_len, to_len);
    if (min_len == 0 || port_num <= 0) {
        return;
    }
    int port_sum = port_num;
    int each_kid_port_num = (port_sum + min_len - 1) / min_len;
    for (int i = 0; i < min_len; i++) {
        if (port_sum > each_kid_port_num) {
            port_sum -= each_kid_port_num;
        } else {
            each_kid_port_num = port_sum;
        }
        meshlist.emplace_back(std::make_tuple(from[from_len - i - 1], to[to_len - i - 1], each_kid_port_num));
    }
}

void ProbeHelper::gen_pinglist()
{
    for (auto& [mesh_name, mesh] : meshTopo.mesh_map) {
        pinglist[mesh_name] = std::vector<std::tuple<std::string, std::string, int>>();
        auto kids = mesh.get_kids();
        int n = kids.size();
        if (n > 2) {
            auto ring = mesh.get_ring();
            for (int i = 0; i < n; i++) {
                auto& from_ip = ring[i];
                auto& to_ip = ring[i + 1];
                mesh_pinglist_insert(
                    pinglist[mesh_name], meshTopo.get_leaf_kids(from_ip), meshTopo.get_leaf_kids(to_ip),
                    get_level_trace_port_num(mesh.level));
            }
        } else if (n == 2) {
            mesh_pinglist_insert(
                pinglist[mesh_name], meshTopo.get_leaf_kids(kids[0]), meshTopo.get_leaf_kids(kids[1]),
                get_level_trace_port_num(mesh.level));
        }
        for (auto& [kid0, kid_to_links] : mesh.kid_to_kid_link) {
            for (auto& [kid1, unused] : kid_to_links) {
                mesh_pinglist_insert(
                    pinglist[mesh_name], meshTopo.get_leaf_kids(kid0), meshTopo.get_leaf_kids(kid1),
                    get_level_trace_port_num(mesh.level));
            }
        }
    }
}

void ProbeHelper::addition_tracert_for_pinglist()
{
    std::vector<std::string> add_from_device;
    std::vector<std::string> add_control_device;
    std::map<std::string, int> add_from_device_index;
    std::vector<std::vector<std::string>> add_to_device;
    std::vector<std::vector<int>> add_port_num;
    // 从pinglist补充探测
    std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>> tracert_list_for_pinglist;
    for (auto& [mesh_name, mesh_pinglist] : pinglist) // 遍历mesh
    {
        for (auto& mesh_pinglist_it : mesh_pinglist) // 遍历mesh下的每个任务
        {
            auto& from_ip = std::get<0>(mesh_pinglist_it);
            auto& to_ip = std::get<1>(mesh_pinglist_it);
            auto port_num = std::get<2>(mesh_pinglist_it);
            auto add_tracert_list = [&](std::string& from_ip, std::string& to_ip) {
                if (tracert_res.find(from_ip) == tracert_res.end()
                    || tracert_res[from_ip].find(to_ip) == tracert_res[from_ip].end()
                    || port_num > tracert_res[from_ip][to_ip].size()) {
                    tracert_list_for_pinglist[mesh_name].emplace_back(std::make_tuple(from_ip, to_ip, port_num));
                    if (add_from_device_index.find(from_ip) == add_from_device_index.end()) {
                        add_from_device_index[from_ip] = add_from_device.size();
                        add_from_device.emplace_back(from_ip);
                        add_control_device.emplace_back(device_ip_to_control[from_ip]);
                        add_to_device.emplace_back(std::vector<std::string>());
                        add_port_num.emplace_back(std::vector<int>());
                    }
                    int from_index = add_from_device_index[from_ip];
                    add_to_device[from_index].emplace_back(to_ip);
                    add_port_num[from_index].emplace_back(port_num);
                }
            };
            add_tracert_list(from_ip, to_ip);
            add_tracert_list(to_ip, from_ip);
        }
    }
    auto add_tracert_res
        = tracert_ports_multi_parallel(add_control_device, add_from_device, add_to_device, add_port_num);
    std::vector<int> to_indexes(add_from_device.size(), 0);
    for (auto& pinglist_it : tracert_list_for_pinglist) // 遍历mesh
    {
        std::string mesh_name = pinglist_it.first;
        auto& mesh = meshTopo.mesh_map[mesh_name];
        int link_level = mesh.level;
        auto& mesh_pinglist = pinglist_it.second;
        for (auto& mesh_pinglist_it : mesh_pinglist) // 遍历mesh下的每个任务
        {
            auto& from_ip = std::get<0>(mesh_pinglist_it);
            auto& to_ip = std::get<1>(mesh_pinglist_it);
            auto port_num = std::get<2>(mesh_pinglist_it);
            int from_index = add_from_device_index[from_ip];
            int to_index = to_indexes[from_index]++;
            auto& trace_ports_res = add_tracert_res[from_index][to_index];
            std::string owner_kid = from_ip;
            std::string target_kid = to_ip;
            for (int i = 0; i < link_level; i++) {
                owner_kid = meshTopo.mesh_father[owner_kid];
                target_kid = meshTopo.mesh_father[target_kid];
            }
            if (target_kid != owner_kid) {
                for (auto& this_trace : trace_ports_res) {
                    int len = this_trace.size();
                    if ((len != level_trace_len[link_level]) && (len + 1 != level_trace_len[link_level])) {
                        std::cout << "[probe_topo][Warning] path length error detected!";
                        this_trace.resize(level_trace_len[link_level]);
                    } else if (len % 2 == 0) {
                        mesh.add_kid_link(
                            owner_kid, get_level_link_ip(this_trace, link_level, false), false, global_id);
                        mesh.add_kid_link(target_kid, get_level_link_ip(this_trace, link_level, true), true, global_id);
                    } else {
                        std::string kid_to_kid_ip = this_trace[len / 2];
                        mesh.add_kid_to_kid_link(owner_kid, target_kid, kid_to_kid_ip, global_id);
                    }
                }
            }
            tracert_res[from_ip][to_ip] = trace_ports_res;
        }
    }
}

void ProbeHelper::gen_id()
{
    if (Mesh::get_unfinished_link() > 0) {
        // throw std::runtime_error("[probe_topo] there are unfinished link");
        std::cout << "[probe_topo][Warning] there are unfinished links" << std::endl;
    }
    auto& tracert_res_id = meshTopo.tracert_res_id;
    /*生成tracert_res_id(探测结果从ip_list转化为id_list)*/
    for (auto& from : tracert_res) {
        std::string from_ip = from.first;
        auto& to_map = from.second;
        if (tracert_res_id.find(from_ip) == tracert_res_id.end()) {
            tracert_res_id[from_ip] = std::map<std::string, std::vector<std::vector<int>>>();
        }
        auto& to_map_id = tracert_res_id[from_ip];
        for (auto& to : to_map) {
            std::string to_ip = to.first;
            if (to_ip == from_ip) {
                continue;
            }
            auto& trace_ports_res = to.second;
            int port_num = trace_ports_res.size();
            if (to_map_id.find(to_ip) == to_map_id.end()) {
                to_map_id[to_ip] = std::vector<std::vector<int>>(port_num);
            } else {
                to_map_id[to_ip].resize(port_num);
            }
            auto& port_id = to_map_id[to_ip];
            for (int i = 0; i < port_num; i++) {
                port_id[i] = (route_to_ids(trace_ports_res[i], from_ip, to_ip));
            }
        }
    }
    get_link_from_mesh_topo();
}

MeshTopo& ProbeHelper::probe_topo()
{
    gen_topo();
    mesh_full_trace();
    gen_pinglist();
    addition_tracert_for_pinglist();
    gen_id();
    return meshTopo;
}

void ProbeHelper::get_link_from_mesh_topo()
{
    /*填冲link_list*/
    for (auto& [mesh_name, mesh] : meshTopo.mesh_map) {
        for (auto& [kid, links] : mesh.kid_to_link) {
            for (auto& link : links) {
                int index = std::get<2>(link);
                int size = index + 1;
                if (size > link_list.size()) {
                    link_list.resize(size);
                }
                link_list[index] = link;
            }
        }
        for (auto& [kid1, kid_to_links] : mesh.kid_to_kid_link) {
            for (auto& [kid2, links] : kid_to_links) {
                for (auto& link : links) {
                    int index = std::get<2>(link);
                    int size = index + 1;
                    if (size > link_list.size()) {
                        link_list.resize(size);
                    }
                    link_list[index] = link;
                }
            }
        }
    }
    global_id = link_list.size();
}

void ProbeHelper::set_mesh_topo(const MeshTopo& meshTopo)
{
    this->meshTopo = meshTopo;
    get_link_from_mesh_topo();
}
void ProbeHelper::set_mesh_topo(const json& j)
{
    this->meshTopo = MeshTopo(j);
    get_link_from_mesh_topo();
}
void ProbeHelper::set_mesh_topo(const std::string& json_path)
{
    this->meshTopo = MeshTopo(json_path);
    get_link_from_mesh_topo();
}

std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>& ProbeHelper::get_pinglist()
{
    if (pinglist.size() == 0) {
        gen_pinglist();
    }
    return pinglist;
}

// 辅助函数，判断两条路径是否是对方的翻转
static int if_filp_route_id(const std::vector<int>& route_0, const std::vector<int>& route_1)
{
    if (route_0.size() != route_1.size())
        return std::numeric_limits<int>::min();
    return std::inner_product(
               route_0.begin(), route_0.end(), route_1.rbegin(), 0, std::plus<int>(),
               [](int a, int b) {
                   return a == b ? 1 : 0;
               })
           - route_0.size();
}

int ProbeHelper::get_return_port(const std::string& from_ip, const std::string& to_ip, int port)
{
    auto from_it = meshTopo.tracert_res_id.find(from_ip);
    if (from_it == meshTopo.tracert_res_id.end()) {
        return port;
    }
    auto this_to_it = from_it->second.find(to_ip);
    if (this_to_it == from_it->second.end() || port < 0 || port >= static_cast<int>(this_to_it->second.size())) {
        return port;
    }
    auto to_it = meshTopo.tracert_res_id.find(to_ip);
    if (to_it == meshTopo.tracert_res_id.end()) {
        return port;
    }
    auto that_to_it = to_it->second.find(from_ip);
    if (that_to_it == to_it->second.end() || that_to_it->second.empty()) {
        return port;
    }

    int res = std::min(port, static_cast<int>(that_to_it->second.size()) - 1);
    auto& this_route_id = this_to_it->second[port];
    int max_same = std::numeric_limits<int>::min();
    for (int j = 0; j < that_to_it->second.size(); j++) {
        auto& that_route_id = that_to_it->second[j];
        int same = if_filp_route_id(this_route_id, that_route_id);
        if (same == 0) {
            res = j;
            return res;
        } else if (same > max_same) {
            max_same = same;
            res = j;
        }
    }
    return res;
}

std::vector<int> ProbeHelper::get_mesh_link_route_ids(const std::string& from_ip, const std::string& to_ip) const
{
    std::vector<int> route_ids;
    std::set<int> seen;
    auto append_link_ids = [&](const std::vector<std::tuple<std::string, std::string, int>>& links) {
        for (const auto& link : links) {
            const int link_id = std::get<2>(link);
            if (seen.insert(link_id).second) {
                route_ids.emplace_back(link_id);
            }
        }
    };

    for (const auto& [mesh_name, mesh] : meshTopo.mesh_map) {
        auto from_it = mesh.kid_to_link.find(from_ip);
        auto to_it = mesh.kid_to_link.find(to_ip);
        if (from_it != mesh.kid_to_link.end() && to_it != mesh.kid_to_link.end()) {
            append_link_ids(from_it->second);
            append_link_ids(to_it->second);
            return route_ids;
        }
    }
    return route_ids;
}

std::vector<int>
ProbeHelper::get_route_ids_for_ping(const std::string& from_ip, const std::string& to_ip, int port) const
{
    auto from_it = meshTopo.tracert_res_id.find(from_ip);
    if (from_it != meshTopo.tracert_res_id.end()) {
        auto to_it = from_it->second.find(to_ip);
        if (to_it != from_it->second.end() && port >= 0 && port < static_cast<int>(to_it->second.size())) {
            return to_it->second[port];
        }
    }
    return get_mesh_link_route_ids(from_ip, to_ip);
}

std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>
ProbeHelper::limit_pinglist_targets_per_source(
    const std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>& pinglist,
    int max_targets_per_source)
{
    if (max_targets_per_source <= 0) {
        throw std::runtime_error("[probe_helper] max pinglist targets per source must be positive");
    }

    std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>> limited_pinglist;
    std::map<std::string, int> target_count_by_source;
    std::map<std::string, int> dropped_count_by_source;
    for (const auto& [mesh_name, mesh_pinglist] : pinglist) {
        auto& limited_mesh_pinglist = limited_pinglist[mesh_name];
        for (const auto& [from_ip, to_ip, port_num] : mesh_pinglist) {
            if (port_num <= 0) {
                continue;
            }

            const int used = target_count_by_source[from_ip];
            const int remaining = max_targets_per_source - used;
            if (remaining <= 0) {
                dropped_count_by_source[from_ip] += port_num;
                continue;
            }

            const int kept_port_num = std::min(port_num, remaining);
            limited_mesh_pinglist.emplace_back(from_ip, to_ip, kept_port_num);
            target_count_by_source[from_ip] += kept_port_num;
            if (kept_port_num < port_num) {
                dropped_count_by_source[from_ip] += port_num - kept_port_num;
            }
        }
    }

    for (auto it = limited_pinglist.begin(); it != limited_pinglist.end();) {
        if (it->second.empty()) {
            it = limited_pinglist.erase(it);
        } else {
            ++it;
        }
    }
    for (const auto& [from_ip, dropped_count] : dropped_count_by_source) {
        std::cout << "[pingpong-plan][Warning] truncate pinglist targets for source " << from_ip
                  << ": max=" << max_targets_per_source << ", dropped=" << dropped_count << std::endl;
    }
    return limited_pinglist;
}

void ProbeHelper::set_pinglist(
    const std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>& pinglist)
{
    const auto limited_pinglist = limit_pinglist_targets_per_source(pinglist);
    // 从pinglist到pinglist_insert_muti_parallel所需的参数
    int n = device_ip_list.size();
    std::vector<std::vector<std::string>> dst_list(n, std::vector<std::string>());
    std::vector<std::vector<int>> src_ports(n, std::vector<int>());
    std::vector<std::vector<int>> src_ports_r(n, std::vector<int>());
    std::vector<int> rx_nums(n, 0);
    std::map<std::string, int> device_ip_to_index;
    for (int i = 0; i < n; i++) {
        auto& from = device_ip_list[i];
        device_ip_to_index[from] = i;
    }
    for (const auto& [mesh_name, mesh_pinglist] : limited_pinglist) // 遍历mesh
    {
        for (const auto& [from_ip, to_ip, port_num] : mesh_pinglist) // 遍历mesh下的每个任务
        {
            int from_index = device_ip_to_index[from_ip];
            int to_index = device_ip_to_index[to_ip];
            rx_nums[to_index] += port_num;
            std::vector<int> this_src_ports(port_num);
            std::vector<int> this_src_ports_r(port_num);
            for (int i = 0; i < port_num; i++) {
                this_src_ports[i] = DEFAULT_SRC_PORT + i;
                this_src_ports_r[i] = DEFAULT_SRC_PORT + get_return_port(from_ip, to_ip, i);
            }
            std::vector<std::string> this_dst_list(port_num, to_ip);
            dst_list[from_index].insert(dst_list[from_index].end(), this_dst_list.begin(), this_dst_list.end());
            src_ports[from_index].insert(src_ports[from_index].end(), this_src_ports.begin(), this_src_ports.end());
            src_ports_r[from_index].insert(
                src_ports_r[from_index].end(), this_src_ports_r.begin(), this_src_ports_r.end());
        }
    }
    for (const auto& task : explicit_ping_tasks) {
        auto from_it = device_ip_to_index.find(task.from_ip);
        auto to_it = device_ip_to_index.find(task.to_ip);
        if (from_it == device_ip_to_index.end() || to_it == device_ip_to_index.end()) {
            std::cout << "[pingpong-plan][Warning] skip explicit task with unknown device: " << task.from_ip << " -> "
                      << task.to_ip << std::endl;
            continue;
        }
        int from_index = from_it->second;
        int to_index = to_it->second;
        dst_list[from_index].emplace_back(task.to_ip);
        src_ports[from_index].emplace_back(task.src_port);
        rx_nums[to_index]++;
    }
    pinglist_insert_muti_parallel(device_control_list, device_ip_list, dst_list, src_ports, src_ports_r, rx_nums);
}

void ProbeHelper::set_explicit_ping_tasks(const std::vector<ExplicitPingTask>& tasks) { explicit_ping_tasks = tasks; }

ProbeHelper::PingpongRawResult ProbeHelper::get_pingpong_res(int times)
{
    return ud_pingpong_tx_muti_parallel(
        device_control_list, device_ip_list, std::vector<int>(device_control_list.size(), times));
}

ProbeHelper::PingpongMetricMatrix
ProbeHelper::reduce_pingpong_res(const PingpongRawResult& pingpong_res, PingpongMetric metric, int times) const
{
    if (metric == PingpongMetric::LogPassRate && times <= 0) {
        throw std::runtime_error("[probe_helper] pingpong times must be positive for pass-rate reduction");
    }

    PingpongMetricMatrix metric_matrix(pingpong_res.size());
    for (size_t device_index = 0; device_index < pingpong_res.size(); device_index++) {
        metric_matrix[device_index].reserve(pingpong_res[device_index].size());
        for (const auto& task_res : pingpong_res[device_index]) {
            auto read_metric = [&](size_t metric_index) -> float {
                if (metric_index >= task_res.size()) {
                    return std::numeric_limits<float>::quiet_NaN();
                }
                return static_cast<float>(task_res[metric_index]);
            };

            switch (metric) {
                case PingpongMetric::P90Lat:
                    metric_matrix[device_index].emplace_back(read_metric(PingpongResult::P90Lat));
                    break;
                case PingpongMetric::P99Lat:
                    metric_matrix[device_index].emplace_back(read_metric(PingpongResult::P99Lat));
                    break;
                case PingpongMetric::MeanLat:
                    metric_matrix[device_index].emplace_back(read_metric(PingpongResult::Mean));
                    break;
                case PingpongMetric::LogPassRate: {
                    if (PingpongResult::Pass >= task_res.size()) {
                        metric_matrix[device_index].emplace_back(std::numeric_limits<float>::quiet_NaN());
                        break;
                    }
                    const uint64_t pass_count = task_res[PingpongResult::Pass];
                    const float log_pass_rate
                        = pass_count > static_cast<uint64_t>(times) ? std::numeric_limits<float>::quiet_NaN() :
                          pass_count == 0                           ? -1e10f :
                                            std::log2f(static_cast<float>(pass_count) / static_cast<float>(times));
                    metric_matrix[device_index].emplace_back(log_pass_rate);
                    break;
                }
            }
        }
    }
    return metric_matrix;
}

// 矩阵求解辅助函数
static std::vector<float>
solve_equation(const std::vector<std::vector<int>>& route_id_list, const std::vector<float>& res, int n)
{
    const int m = res.size();
    const auto invalid_result = [n]() {
        return std::vector<float>(n, std::numeric_limits<float>::quiet_NaN());
    };
    if (m == 0 || n == 0 || static_cast<int>(route_id_list.size()) != m) {
        return invalid_result();
    }
    // 创建系数矩阵 A (m x n)
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(m, n);
    // 创建右侧向量 b (m x 1)
    Eigen::VectorXf b(m);
    // 填充系数矩阵和右侧向量
    for (int i = 0; i < m; i++) {
        // 每个方程对应一行
        const auto& equation = route_id_list[i];
        for (int var_index : equation) {
            if (var_index < n) {
                A(i, var_index) += 1.0f;
            }
        }
        // 设置右侧常数项
        if (!std::isfinite(res[i])) {
            return invalid_result();
        }
        b(i) = res[i];
    }
    // 求解方程组 Ax = b
    // 使用最小二乘法解决可能超定或欠定的系统
    Eigen::BDCSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.setThreshold(1e-6f);
    if (svd.rank() < n) {
        return invalid_result();
    }
    const auto singular_values = svd.singularValues();
    if (singular_values.size() == 0 || singular_values[singular_values.size() - 1] <= 0.0f
        || singular_values[0] / singular_values[singular_values.size() - 1] > 1e6f) {
        return invalid_result();
    }
    Eigen::VectorXf x = svd.solve(b);
    // 使用列主元QR分解（更快但需要矩阵是满秩的）
    // Eigen::VectorXf x = A.colPivHouseholderQr().solve(b);
    // 将结果转换为 std::vector
    std::vector<float> result(n);
    for (int i = 0; i < n; i++) {
        result[i] = x[i];
    }
    return result;
}

ProbeHelper::LinkMetricVector ProbeHelper::solve_pingpong_res(
    const PingpongMetricMatrix& pingpong_res,
    const std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>& pinglist)
{
    std::vector<float> solve_res(global_id, std::numeric_limits<float>::quiet_NaN());

    std::vector<int> dst_list_to_task_index(device_ip_list.size(), 0);
    std::map<std::string, int> device_ip_to_index;
    for (int i = 0; i < device_ip_list.size(); i++) {
        auto& from = device_ip_list[i];
        device_ip_to_index[from] = i;
    }
    for (auto& [mesh_name, mesh_pinglist] : pinglist) // 遍历mesh
    {
        // 记录mesh下local_id的关系
        std::map<int, int> global_link_id_to_local;
        std::vector<int> local_link_id_to_global;
        // 每个mesh独立求解
        std::vector<std::vector<int>> local_id_matrix;
        std::vector<float> local_res_matrix;
        for (auto& [from_ip, to_ip, port_num] : mesh_pinglist) // 遍历mesh下每个ping任务
        {
            int from_index = device_ip_to_index[from_ip];
            for (int port = 0; port < port_num; port++) // 默认任务ports是从0递增的
            {
                int this_task_index = dst_list_to_task_index[from_index]++; // 和创建pinglist时emplace时的顺序一致
                auto this_route_id = get_route_ids_for_ping(from_ip, to_ip, port);
                // 清洗已经求解的link
                std::vector<int> local_id_list_without_res;
                float this_res = pingpong_res[from_index][this_task_index];
                for (int route_global_id : this_route_id) {
                    if (route_global_id < 0 || route_global_id >= static_cast<int>(solve_res.size())) {
                        continue;
                    }
                    if (!std::isnan(solve_res[route_global_id])) {
                        this_res -= static_cast<float>(solve_res[route_global_id]);
                    } else {
                        if (global_link_id_to_local.find(route_global_id) == global_link_id_to_local.end()) {
                            int local_id = local_link_id_to_global.size();
                            global_link_id_to_local[route_global_id] = local_id;
                            local_link_id_to_global.emplace_back(route_global_id);
                        }
                        int local_id = global_link_id_to_local[route_global_id];
                        local_id_list_without_res.emplace_back(local_id);
                    }
                }
                local_id_matrix.emplace_back(local_id_list_without_res);
                local_res_matrix.emplace_back(this_res);
            }
        } // 完成mesh下的求解矩阵
        int local_id_num = local_link_id_to_global.size();
        if (local_id_num == 0) {
            continue;
        }
        // 求解
        auto local_solve_res = solve_equation(local_id_matrix, local_res_matrix, local_id_num);
        for (int i = 0; i < local_id_num; i++) {
            int route_global_id = local_link_id_to_global[i];
            solve_res[route_global_id] = local_solve_res[i];
        }
    }
    return solve_res;
}
