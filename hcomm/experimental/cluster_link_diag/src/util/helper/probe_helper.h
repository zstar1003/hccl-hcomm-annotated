/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <limits>
#include <utility>
#include <cstdint>
#include <map>

#include <nlohmann/json.hpp>

#include "topo/mesh_topo.h"
#include "topo/control_topo.h"

using json = nlohmann::json;

class ProbeHelper {
public:
    using HccnDeviceIpListParallel = std::function<std::vector<std::vector<std::string>>(
        const std::vector<std::string>&, const std::vector<int>&)>;
    using TracertPathResult = std::vector<std::vector<std::vector<std::vector<std::string>>>>;
    using TracertPortsMultiParallel = std::function<TracertPathResult(
        const std::vector<std::string>&, const std::vector<std::string>&, const std::vector<std::vector<std::string>>&,
        const std::vector<std::vector<int>>&)>;
    using PinglistInsertMutiParallel = std::function<void(
        const std::vector<std::string>&, const std::vector<std::string>&, const std::vector<std::vector<std::string>>&,
        const std::vector<std::vector<int>>&, const std::vector<std::vector<int>>&, const std::vector<int>&)>;
    using PingpongRawResult = std::vector<std::vector<std::vector<uint64_t>>>;
    using UdPingpongTxMutiParallel = std::function<PingpongRawResult(
        const std::vector<std::string>&, const std::vector<std::string>&, const std::vector<int>&)>;
    using PingpongMetricMatrix = std::vector<std::vector<float>>;
    using LinkMetricVector = std::vector<float>;

    enum class PingpongMetric {
        P90Lat,
        P99Lat,
        MeanLat,
        LogPassRate,
    };

    struct ExplicitPingTask {
        std::string from_ip;
        std::string to_ip;
        int src_port = 0;
        std::string tag;
    };

private:
    ControlTopo* ct;

    std::vector<std::string> device_control_list;
    std::map<std::string, std::string> device_ip_to_control;

    std::map<std::string, std::map<std::string, std::vector<std::vector<std::string>>>> tracert_res;
    std::vector<int> level_ip_index;
    std::vector<int> level_trace_len;
    std::vector<int> level_trace_port_num = {1, 16};
    int tree_probe_sport_count = 1;

    // 获取探测路径经过的层级
    int get_path_level(const std::vector<std::string>& path);
    int get_level_trace_port_num(int level) const;
    // 获取探测到某个层级的链路ip
    std::string get_level_link_ip(const std::vector<std::string>& path, int level, bool near_ip);
    // 将ip_list形式的route转化为id_list形式，方便求解
    std::vector<int> route_to_ids(const std::vector<std::string>& path, const std::string& from, const std::string& to);

    HccnDeviceIpListParallel hccn_device_ip_list_parallel;
    TracertPortsMultiParallel tracert_ports_multi_parallel;
    PinglistInsertMutiParallel pinglist_insert_muti_parallel; // 第二个参数：&src_devIp_list
    UdPingpongTxMutiParallel ud_pingpong_tx_muti_parallel;    // 第二个参数：&src_devIp_list

    // probe_topo
    void gen_topo();
    void mesh_full_trace();
    MeshTopo build_lldp_topo(const std::map<std::string, std::string>& device_lldp_mgmt_ip);
    // 添加额外的tracert
    std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>> pinglist;
    std::vector<ExplicitPingTask> explicit_ping_tasks;
    void gen_pinglist();
    void addition_tracert_for_pinglist();
    void gen_id();

    void get_link_from_mesh_topo();

    int get_return_port(const std::string& from_ip, const std::string& to_ip, int port);
    std::vector<int> get_mesh_link_route_ids(const std::string& from_ip, const std::string& to_ip) const;
    std::vector<int> get_route_ids_for_ping(const std::string& from_ip, const std::string& to_ip, int port) const;

public:
    static constexpr int MAX_PINGLIST_TARGETS_PER_SOURCE = 16;

    MeshTopo meshTopo;
    int global_id = 0;
    std::vector<std::tuple<std::string, std::string, int>> link_list;
    std::vector<std::string> device_ip_list;
    std::vector<std::vector<std::string>> lldp_device_groups;

    size_t configured_device_count() const { return device_control_list.size(); }

    size_t discovered_device_count() const { return device_ip_list.size(); }

    bool device_count_complete() const { return configured_device_count() == discovered_device_count(); }

    ProbeHelper(
        ControlTopo* ct, HccnDeviceIpListParallel hccn_device_ip_list_parallel,
        TracertPortsMultiParallel tracert_ports_multi_parallel,
        PinglistInsertMutiParallel pinglist_insert_muti_parallel,
        UdPingpongTxMutiParallel ud_pingpong_tx_muti_parallel);
    // 所有接口函数
    void set_hccn_device_ip_list_parallel(HccnDeviceIpListParallel func)
    {
        hccn_device_ip_list_parallel = std::move(func);
    }
    void set_tracert_ports_multi_parallel(
        std::function<std::vector<std::vector<std::vector<std::vector<std::string>>>>(
            const std::vector<std::string>&, const std::vector<std::string>&,
            const std::vector<std::vector<std::string>>&, const std::vector<std::vector<int>>&)>
            func)
    {
        tracert_ports_multi_parallel = func;
    }
    void set_pinglist_insert_muti_parallel(
        std::function<void(
            const std::vector<std::string>&, const std::vector<std::string>&,
            const std::vector<std::vector<std::string>>&, const std::vector<std::vector<int>>&,
            const std::vector<std::vector<int>>&, const std::vector<int>&)>
            func)
    {
        pinglist_insert_muti_parallel = func;
    }
    void set_ud_pingpong_tx_muti_parallel(
        std::function<std::vector<std::vector<std::vector<uint64_t>>>(
            const std::vector<std::string>&, const std::vector<std::string>&, const std::vector<int>&)>
            func)
    {
        ud_pingpong_tx_muti_parallel = func;
    }
    void set_tracert_port_config(int tree_probe_sport_count, int fullmesh_sport_count);
    MeshTopo& probe_topo_ring();
    MeshTopo
    probe_lldp_topo(std::function<std::vector<std::string>(const std::vector<std::string>&)> lldp_mgmt_ip_parallel);
    // 探测拓扑,初始化类里的所有私有变量
    MeshTopo& probe_topo();
    void set_mesh_topo(const MeshTopo& meshTopo);
    void set_mesh_topo(const json& j);
    void set_mesh_topo(const std::string& json_path);
    void set_mesh_topo(MeshTopo& meshTopo);
    // 获取pinglist:map<mesh_name, task<from, to, portnum>>
    std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>& get_pinglist();
    static std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>
    limit_pinglist_targets_per_source(
        const std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>& pinglist,
        int max_targets_per_source = MAX_PINGLIST_TARGETS_PER_SOURCE);
    // 设置pinglist到每个节点上
    void set_pinglist(const std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>& pinglist);
    void set_explicit_ping_tasks(const std::vector<ExplicitPingTask>& tasks);
    // 获取 PingPong 原始结果：device<task<metric_vec<uint64_t>>>，metric_vec 下标见 PingpongResult。
    PingpongRawResult get_pingpong_res(int times = 1);
    // 将 3 级 PingPong 原始结果归约为求解使用的 2 级指标矩阵：device<task<float>>。
    PingpongMetricMatrix
    reduce_pingpong_res(const PingpongRawResult& pingpong_res, PingpongMetric metric, int times = 1) const;
    // 求解结果使用连续 link_global_id 作为下标，范围为 [0, global_id)，value 为对应链路指标。
    LinkMetricVector solve_pingpong_res(
        const PingpongMetricMatrix& pingpong_res,
        const std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>& pinglist);
};
