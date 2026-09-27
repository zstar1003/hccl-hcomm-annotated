/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <unistd.h>
#include <numeric>
#include <chrono>
#include <memory>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <map>

#include <CLI/CLI.hpp>

#include "helper/probe_helper.h"
#include "helper/tracert_rpc_helper.h"
#include "file_path/workspace.h"

#include "tool/hccn_pingpong/rpc_func.h"
#include "tool/metrics_collector/metrics_collector.h"
#include "tool/tracert/tracert.h"
#include "topo/control_topo.h"
#include "topo/probe_config.h"
#include "rpc_call/proxy_call.hpp"
#include "rpc_call/proxy_call_paralled.hpp"

namespace {
static const int L2_PATH_SRC_PORT = 49152;

struct DeviceLabelInfo {
    std::vector<std::string> labels;
    std::map<std::string, std::string> label_by_ip;
};

std::string make_device_label(const std::string& control_dev) { return "dev" + control_dev; }

std::string format_path(const std::vector<std::string>& path)
{
    std::string result;
    for (size_t i = 0; i < path.size(); i++) {
        if (i != 0) {
            result += " -> ";
        }
        result += path[i];
    }
    return result;
}

DeviceLabelInfo append_device_metadata(
    json& root, const std::vector<std::string>& device_control_list, const std::vector<std::string>& device_ip_list)
{
    DeviceLabelInfo info;
    root["device_list"] = json::array();
    root["device_map"] = json::object();
    info.labels.reserve(device_control_list.size());
    for (size_t i = 0; i < device_control_list.size(); i++) {
        const std::string label = make_device_label(device_control_list[i]);
        info.labels.emplace_back(label);
        const std::string device_ip = i < device_ip_list.size() ? device_ip_list[i] : "";
        root["device_list"].push_back({
            {"index", i},
            {"label", label},
            {"control_dev", device_control_list[i]},
            {"device_ip", device_ip},
        });
        root["device_map"][label] = device_ip;
        if (!device_ip.empty()) {
            info.label_by_ip[device_ip] = label;
        }
    }
    return info;
}

bool get_target_path_info(
    size_t src_index, size_t target_index, const std::vector<std::vector<int>>& dst_indexes,
    const std::vector<std::string>& device_labels, std::string& dst_label)
{
    if (src_index >= dst_indexes.size() || target_index >= dst_indexes[src_index].size()) {
        return false;
    }
    const int dst_index = dst_indexes[src_index][target_index];
    if (dst_index < 0 || static_cast<size_t>(dst_index) >= device_labels.size()) {
        return false;
    }
    dst_label = device_labels[dst_index];
    return true;
}

void append_directed_pair(
    const std::string& src_ip, const std::string& dst_ip, const std::map<std::string, int>& device_ip_to_index,
    const ProbeRuntimeConfig& probe_config, std::vector<std::vector<std::string>>& dst_list,
    std::vector<std::vector<int>>& dst_indexes, std::vector<std::vector<int>>& port_num, int& directed_pair_count)
{
    if (src_ip == dst_ip) {
        return;
    }
    const int src = device_ip_to_index.at(src_ip);
    const int dst = device_ip_to_index.at(dst_ip);
    dst_list[src].emplace_back(dst_ip);
    dst_indexes[src].emplace_back(dst);
    port_num[src].emplace_back(probe_config.probe_topo_tracert.sport_count);
    directed_pair_count++;
}

json build_allpath_json(
    const std::vector<std::string>& device_control_list, const std::vector<std::string>& device_ip_list,
    const std::vector<std::vector<int>>& dst_indexes,
    const std::vector<std::vector<std::vector<std::vector<std::string>>>>& allpath_res,
    const ProbeRuntimeConfig& probe_config)
{
    json root;
    root["schema"] = "disp_probe.allpath.v1";
    root["scope"] = probe_config.probe_topo_tracert.scope;
    root["sport_begin"] = probe_config.probe_topo_tracert.sport_begin;
    root["sport_count"] = probe_config.probe_topo_tracert.sport_count;

    const auto label_info = append_device_metadata(root, device_control_list, device_ip_list);
    const auto& device_labels = label_info.labels;

    root["paths"] = json::object();
    root["summary"] = json::object();
    int total_pair_count = 0;
    int total_sample_count = 0;
    int total_empty_count = 0;
    int total_unique_path_count = 0;

    for (size_t src_index = 0; src_index < allpath_res.size(); src_index++) {
        const std::string& src_label = device_labels[src_index];
        const auto& src_res = allpath_res[src_index];
        for (size_t target_index = 0; target_index < src_res.size(); target_index++) {
            std::string dst_label;
            if (!get_target_path_info(src_index, target_index, dst_indexes, device_labels, dst_label)) {
                continue;
            }
            const auto& port_paths = src_res[target_index];
            std::set<std::vector<std::string>> unique_paths;
            int empty_count = 0;
            json samples = json::array();
            for (size_t port_index = 0; port_index < port_paths.size(); port_index++) {
                const auto& hops = port_paths[port_index];
                if (hops.empty()) {
                    empty_count++;
                } else {
                    unique_paths.insert(hops);
                }
                samples.push_back({
                    {"sport", probe_config.probe_topo_tracert.sport_begin + static_cast<int>(port_index)},
                    {"hops", hops},
                });
            }

            root["paths"][src_label][dst_label] = std::move(samples);
            root["summary"][src_label][dst_label] = {
                {"samples", port_paths.size()},
                {"empty", empty_count},
                {"unique_paths", unique_paths.size()},
            };

            total_pair_count++;
            total_sample_count += static_cast<int>(port_paths.size());
            total_empty_count += empty_count;
            total_unique_path_count += static_cast<int>(unique_paths.size());
        }
    }

    root["summary_total"] = {
        {"pair_count", total_pair_count},
        {"sample_count", total_sample_count},
        {"empty_count", total_empty_count},
        {"unique_path_count_sum", total_unique_path_count},
    };
    return root;
}

json build_l2_path_json(
    const std::vector<std::string>& device_control_list, const std::vector<std::string>& device_ip_list,
    const std::vector<std::vector<std::string>>& lldp_groups, const std::vector<std::vector<int>>& dst_indexes,
    const std::vector<std::vector<std::vector<std::vector<std::string>>>>& l2_path_res,
    const ProbeRuntimeConfig& probe_config)
{
    json root;
    root["schema"] = "disp_probe.l2_ring_fullmesh_paths.v1";
    root["scope"] = probe_config.probe_topo_tracert.scope;
    root["sport"] = L2_PATH_SRC_PORT;

    const auto label_info = append_device_metadata(root, device_control_list, device_ip_list);
    const auto& device_labels = label_info.labels;
    const auto& label_by_ip = label_info.label_by_ip;

    root["domains"] = json::array();
    for (size_t group_index = 0; group_index < lldp_groups.size(); group_index++) {
        json devices = json::array();
        for (const auto& device_ip : lldp_groups[group_index]) {
            devices.push_back({
                {"label", label_by_ip[device_ip]},
                {"device_ip", device_ip},
            });
        }
        root["domains"].push_back({
            {"name", "TOPO[0," + std::to_string(group_index) + "]"},
            {"devices", std::move(devices)},
        });
    }

    root["paths"] = json::object();
    int pair_count = 0;
    int empty_count = 0;
    for (size_t src_index = 0; src_index < l2_path_res.size(); src_index++) {
        if (src_index >= device_labels.size()) {
            continue;
        }
        const std::string& src_label = device_labels[src_index];
        const auto& src_res = l2_path_res[src_index];
        for (size_t target_index = 0; target_index < src_res.size(); target_index++) {
            std::string dst_label;
            if (!get_target_path_info(src_index, target_index, dst_indexes, device_labels, dst_label)) {
                continue;
            }
            const auto& port_paths = src_res[target_index];
            const std::vector<std::string> hops = port_paths.empty() ? std::vector<std::string>{} : port_paths.front();
            if (hops.empty()) {
                empty_count++;
            }
            root["paths"][src_label][dst_label] = {
                {"sport", L2_PATH_SRC_PORT},
                {"hops", hops},
            };
            pair_count++;
        }
    }

    root["summary_total"] = {
        {"pair_count", pair_count},
        {"empty_count", empty_count},
    };
    return root;
}

void write_json_file(const std::string& path, const json& data)
{
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("[allpath] json path unavailable: " + path);
    }
    ofs << data.dump(2);
}

json make_topology_result(
    const MeshTopo& topology, const ProbeHelper& helper, const ControlTopo& control_topo, const std::string& status)
{
    json result = topology.to_json();
    result["status"] = status;
    result["configured_device_count"] = helper.configured_device_count();
    result["discovered_device_count"] = helper.discovered_device_count();
    result["configured_devices"] = control_topo.LeafList();
    result["discovered_device_ips"] = helper.device_ip_list;
    return result;
}
} // namespace

int main(int argc, char** argv)
{
    // 通过探测得到MeshTopo
    CLI::App app{"test host"};
    std::string rank_table_file = "./control_json/910b2_info.json";
    app.add_option("-f,--file", rank_table_file, "rank table file");
    CLI11_PARSE(app, argc, argv);

    make_abs_path(rank_table_file);
    auto probe_config = load_probe_runtime_config(rank_table_file);
    auto json_path = make_probe_topo_json_path(get_project_path(), probe_config);
    if (probe_config.enabled) {
        std::cout << "[config] probe.scope=" << format_probe_scope(probe_config.probe_topo_tracert.scope)
                  << ", sport_begin=" << probe_config.probe_topo_tracert.sport_begin
                  << ", sport_count=" << probe_config.probe_topo_tracert.sport_count
                  << ", tree_probe_sport_count=" << probe_config.probe_topo_tracert.tree_probe_sport_count
                  << ", topology_optimized=" << (probe_config.probe_topo_tracert.topology_optimized ? "true" : "false")
                  << ", l2_path_aware=" << (probe_config.probe_topo_tracert.l2_path_aware ? "true" : "false")
                  << ", output=" << json_path << std::endl;
    }
    std::filesystem::create_directories(std::filesystem::path(json_path).parent_path());
    std::unique_ptr<ControlTopo> ct_holder;
    if (probe_config.enabled) {
        ct_holder = std::make_unique<ControlTopo>(make_control_topo_json(probe_config.probe_topo_tracert.scope));
    } else {
        ct_holder = std::make_unique<ControlTopo>(rank_table_file);
    }

    // 包装函数
    ControlTopo* ct_p = ct_holder.get();
    const int tracert_sport_begin
        = probe_config.enabled ? probe_config.probe_topo_tracert.sport_begin : DEFAULT_SRC_PORT;
    auto hccn_device_ip_list_parallel
        = [ct_p](const std::vector<std::string>& device_list, const std::vector<int>& device_count_list) {
              return proxy_call_paralled(device_list, ct_p, hccn_device_ip_list, device_count_list);
          };
    auto lldp_mgmt_ip_parallel = [ct_p](const std::vector<std::string>& device_list) {
        return proxy_call_paralled(device_list, ct_p, hccn_lldp_mgmt_ip, device_list);
    };
    int tracert_batch_id = 0;
    auto tracert_ports_multi_parallel
        = [ct_p, tracert_sport_begin, &tracert_batch_id](
              const std::vector<std::string>& device_list, const std::vector<std::string>& src_devIp_list,
              const std::vector<std::vector<std::string>>& dst_list, const std::vector<std::vector<int>>& port_num) {
              const int batch_id = ++tracert_batch_id;
              const int total_port_probe = LogTracertBatchPlan(
                  batch_id, tracert_sport_begin, "src_port_begin", device_list, src_devIp_list, dst_list, port_num);
              std::cout << "[tracert][batch " << batch_id << "] rpc start, total_port_probe=" << total_port_probe
                        << std::endl;

              const auto rpc_start_time = std::chrono::steady_clock::now();
              TracertPathResult res;
              if (tracert_sport_begin == DEFAULT_SRC_PORT) {
                  res = proxy_call_paralled(
                      device_list, ct_p, tracert_ports_multi_by_src_ip, src_devIp_list, dst_list, port_num);
              } else {
                  res = proxy_call_paralled(
                      device_list, ct_p, tracert_ports_multi_by_src_ip_with_sport_begin, src_devIp_list, dst_list,
                      port_num, std::vector<int>(device_list.size(), tracert_sport_begin));
              }
              const auto rpc_end_time = std::chrono::steady_clock::now();

              const auto path_stats = NormalizeTracertPaths(res);
              int success_pair_count = 0;
              for (size_t src_index = 0; src_index < res.size(); src_index++) {
                  auto& v1 = res[src_index];
                  for (size_t target_index = 0; target_index < v1.size(); target_index++) {
                      auto& v2 = v1[target_index];
                      bool pair_success_printed = false;
                      for (size_t port_index = 0; port_index < v2.size(); port_index++) {
                          auto& v3 = v2[port_index];
                          if (!v3.empty() && !pair_success_printed) {
                              const std::string src_control_dev
                                  = src_index < device_list.size() ? device_list[src_index] : "";
                              const std::string src_dev_ip
                                  = src_index < src_devIp_list.size() ? src_devIp_list[src_index] : "";
                              const std::string target_ip
                                  = src_index < dst_list.size() && target_index < dst_list[src_index].size() ?
                                        dst_list[src_index][target_index] :
                                        "";
                              std::cout << "[tracert][batch " << batch_id << "][success] "
                                        << "control_dev=" << src_control_dev << ", src_ip=" << src_dev_ip
                                        << ", target=" << target_ip
                                        << ", sport=" << (tracert_sport_begin + static_cast<int>(port_index))
                                        << ", path=" << format_path(v3) << std::endl;
                              pair_success_printed = true;
                              success_pair_count++;
                          }
                      }
                  }
              }
              const auto elapsed_ms = GetElapsedMs(rpc_start_time, rpc_end_time);
              std::cout << "[tracert][batch " << batch_id << "] rpc done, elapsed_ms=" << elapsed_ms
                        << ", result_path_count=" << path_stats.first << ", empty_path_count=" << path_stats.second
                        << ", success_pair_count=" << success_pair_count << std::endl;
              return res;
          };
    auto pinglist_insert_muti_parallel
        = [ct_p](
              const std::vector<std::string>& device_list, const std::vector<std::string>& src_devIp_list,
              const std::vector<std::vector<std::string>> dst_list, const std::vector<std::vector<int>>& src_ports,
              const std::vector<std::vector<int>>& src_ports_r, const std::vector<int>& rx_nums) {
              return proxy_call_paralled(
                  device_list, ct_p, pinglist_insert_muti, src_devIp_list, dst_list, src_ports, src_ports_r, rx_nums);
          };
    auto ud_pingpong_tx_muti_parallel
        = [ct_p](
              const std::vector<std::string>& device_list, const std::vector<std::string>& src_devIp_list,
              const std::vector<int>& times_list) {
              return proxy_call_paralled(device_list, ct_p, ud_pingpong_tx_muti, src_devIp_list, times_list);
          };

    // 探测拓扑
    ProbeHelper probeHelper(
        ct_p, hccn_device_ip_list_parallel, tracert_ports_multi_parallel, pinglist_insert_muti_parallel,
        ud_pingpong_tx_muti_parallel);
    if (!probeHelper.device_count_complete()) {
        const auto incomplete_result = make_topology_result(probeHelper.meshTopo, probeHelper, *ct_p, "incomplete");
        write_json_file(json_path, incomplete_result);
        if (probe_config.enabled) {
            write_json_file(make_probe_topo_lldp_json_path(get_project_path(), probe_config), incomplete_result);
        }
        std::cerr << "[probe_topo][incomplete] configured_device_count=" << probeHelper.configured_device_count()
                  << ", discovered_device_count=" << probeHelper.discovered_device_count()
                  << ", diagnostic_output=" << json_path << std::endl;
        return 2;
    }
    if (probe_config.enabled) {
        std::cout << "[probe_topo][batch 1] purpose=network-layer-discovery, method=ring-tracert"
                  << ", tree_probe_sport_count=" << probe_config.probe_topo_tracert.tree_probe_sport_count << std::endl;
        probeHelper.set_tracert_port_config(
            probe_config.probe_topo_tracert.tree_probe_sport_count, probe_config.probe_topo_tracert.sport_count);
    }
    auto meshTopo = probe_config.enabled ? probeHelper.probe_topo_ring() : probeHelper.probe_topo();
    write_json_file(json_path, make_topology_result(meshTopo, probeHelper, *ct_p, "complete"));
    std::cout << "mesh_topo.json saved in: " << json_path << std::endl;

    if (probe_config.enabled) {
        const auto lldp_json_path = make_probe_topo_lldp_json_path(get_project_path(), probe_config);
        std::cout << "[probe_topo][batch 2] purpose=merge-layer0-by-lldp, method=hccn_tool-lldp"
                  << ", output=" << lldp_json_path << std::endl;
        auto lldpTopo = probeHelper.probe_lldp_topo(lldp_mgmt_ip_parallel);
        write_json_file(lldp_json_path, make_topology_result(lldpTopo, probeHelper, *ct_p, "complete"));
        std::cout << "mesh_topo_lldp.json saved in: " << lldp_json_path << std::endl;

        const auto allpath_path = make_allpath_json_path(get_project_path(), probe_config);
        const auto& device_control_list = ct_p->LeafList();
        const auto& device_ip_list = probeHelper.device_ip_list;
        const int n = static_cast<int>(device_ip_list.size());
        if (static_cast<int>(device_control_list.size()) != n) {
            throw std::runtime_error("[allpath] device control list and device ip list size mismatch");
        }

        std::map<std::string, int> device_ip_to_index;
        for (int i = 0; i < n; i++) {
            device_ip_to_index[device_ip_list[i]] = i;
        }

        const auto& lldp_groups = probeHelper.lldp_device_groups;
        std::vector<std::vector<std::string>> dst_list(n);
        std::vector<std::vector<int>> dst_indexes(n);
        std::vector<std::vector<int>> port_num(n);
        int directed_pair_count = 0;
        const bool topology_optimized = probe_config.probe_topo_tracert.topology_optimized;
        for (size_t src_group = 0; src_group < lldp_groups.size(); src_group++) {
            if (topology_optimized) {
                if (lldp_groups.size() <= 1) {
                    continue;
                }
                const size_t dst_group = (src_group + 1) % lldp_groups.size();
                const size_t slot_count = std::min(lldp_groups[src_group].size(), lldp_groups[dst_group].size());
                for (size_t slot = 0; slot < slot_count; slot++) {
                    const std::string& src_ip = lldp_groups[src_group][slot];
                    const std::string& dst_ip = lldp_groups[dst_group][slot];
                    append_directed_pair(
                        src_ip, dst_ip, device_ip_to_index, probe_config, dst_list, dst_indexes, port_num,
                        directed_pair_count);
                }
                continue;
            }

            for (size_t dst_group = 0; dst_group < lldp_groups.size(); dst_group++) {
                if (src_group == dst_group) {
                    continue;
                }
                for (const auto& src_ip : lldp_groups[src_group]) {
                    for (const auto& dst_ip : lldp_groups[dst_group]) {
                        append_directed_pair(
                            src_ip, dst_ip, device_ip_to_index, probe_config, dst_list, dst_indexes, port_num,
                            directed_pair_count);
                    }
                }
            }
        }

        std::cout << "[probe_topo][batch 3] purpose=cross-switch-multipath-coverage"
                  << ", method=directed-tracert"
                  << ", topology_optimized=" << (topology_optimized ? "true" : "false")
                  << ", coverage_mode=" << (topology_optimized ? "ring-same-slot" : "full-cross-domain")
                  << ", lldp_group_count=" << lldp_groups.size() << ", directed_pair_count=" << directed_pair_count
                  << ", coverage_count_per_pair=sport_count=" << probe_config.probe_topo_tracert.sport_count
                  << std::endl;

        std::cout << "[allpath] cross-switch tracert start: device_count=" << n
                  << ", directed_pair_count=" << directed_pair_count
                  << ", sport_begin=" << probe_config.probe_topo_tracert.sport_begin
                  << ", sport_count=" << probe_config.probe_topo_tracert.sport_count << ", output=" << allpath_path
                  << std::endl;

        tracert_batch_id = 2;
        auto allpath_res = tracert_ports_multi_parallel(device_control_list, device_ip_list, dst_list, port_num);
        auto allpath_json
            = build_allpath_json(device_control_list, device_ip_list, dst_indexes, allpath_res, probe_config);
        allpath_json["status"] = "complete";
        allpath_json["configured_device_count"] = probeHelper.configured_device_count();
        allpath_json["discovered_device_count"] = probeHelper.discovered_device_count();
        write_json_file(allpath_path, allpath_json);

        std::cout << "[allpath] json saved in: " << allpath_path
                  << ", pair_count=" << allpath_json["summary_total"]["pair_count"]
                  << ", sample_count=" << allpath_json["summary_total"]["sample_count"]
                  << ", empty_count=" << allpath_json["summary_total"]["empty_count"]
                  << ", unique_path_count_sum=" << allpath_json["summary_total"]["unique_path_count_sum"] << std::endl;

        if (probe_config.probe_topo_tracert.l2_path_aware) {
            const auto l2_path_json_path = make_l2_path_json_path(get_project_path(), probe_config);
            std::vector<std::vector<std::string>> l2_dst_list(n);
            std::vector<std::vector<int>> l2_dst_indexes(n);
            std::vector<std::vector<int>> l2_port_num(n);
            int l2_directed_pair_count = 0;
            if (lldp_groups.size() > 1) {
                for (size_t src_group = 0; src_group < lldp_groups.size(); src_group++) {
                    const size_t dst_group = (src_group + 1) % lldp_groups.size();
                    for (const auto& src_ip : lldp_groups[src_group]) {
                        for (const auto& dst_ip : lldp_groups[dst_group]) {
                            if (src_ip == dst_ip) {
                                continue;
                            }
                            const int src = device_ip_to_index.at(src_ip);
                            const int dst = device_ip_to_index.at(dst_ip);
                            l2_dst_list[src].emplace_back(dst_ip);
                            l2_dst_indexes[src].emplace_back(dst);
                            l2_port_num[src].emplace_back(1);
                            l2_directed_pair_count++;
                        }
                    }
                }
            }

            std::cout << "[probe_topo][batch 4] purpose=l2-ring-fullmesh-path-discovery"
                      << ", method=directed-tracert"
                      << ", src_port=" << L2_PATH_SRC_PORT << ", lldp_group_count=" << lldp_groups.size()
                      << ", directed_pair_count=" << l2_directed_pair_count << ", output=" << l2_path_json_path
                      << std::endl;

            std::cout << "[l2-path] ring fullmesh tracert start: device_count=" << n
                      << ", directed_pair_count=" << l2_directed_pair_count << ", sport=" << L2_PATH_SRC_PORT
                      << ", output=" << l2_path_json_path << std::endl;

            tracert_batch_id = 3;
            auto l2_path_res = proxy_call_paralled(
                device_control_list, ct_p, tracert_ports_multi_by_src_ip_with_sport_begin, device_ip_list, l2_dst_list,
                l2_port_num, std::vector<int>(device_control_list.size(), L2_PATH_SRC_PORT));

            int l2_result_path_count = 0;
            int l2_empty_path_count = 0;
            for (auto& v1 : l2_path_res) {
                for (auto& v2 : v1) {
                    for (auto& v3 : v2) {
                        l2_result_path_count++;
                        if (v3.empty()) {
                            l2_empty_path_count++;
                        }
                        if (v3.size() == 1) {
                            v3 = {"", v3[0]};
                        }
                    }
                }
            }

            auto l2_path_json = build_l2_path_json(
                device_control_list, device_ip_list, lldp_groups, l2_dst_indexes, l2_path_res, probe_config);
            l2_path_json["status"] = "complete";
            l2_path_json["configured_device_count"] = probeHelper.configured_device_count();
            l2_path_json["discovered_device_count"] = probeHelper.discovered_device_count();
            write_json_file(l2_path_json_path, l2_path_json);

            std::cout << "[l2-path] json saved in: " << l2_path_json_path
                      << ", pair_count=" << l2_path_json["summary_total"]["pair_count"]
                      << ", empty_count=" << l2_path_json["summary_total"]["empty_count"]
                      << ", result_path_count=" << l2_result_path_count << ", empty_path_count=" << l2_empty_path_count
                      << std::endl;
        } else {
            std::cout << "[probe_topo][batch 4] skip l2-ring-fullmesh-path-discovery"
                      << ", l2_path_aware=false" << std::endl;
        }
    }
}
