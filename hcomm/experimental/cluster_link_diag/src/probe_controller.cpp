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
#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <tuple>
#include <cmath>
#include <atomic>
#include <thread>
#include <iomanip>
#include <ctime>

#include <CLI/CLI.hpp>

#include "helper/probe_helper.h"
#include "helper/probe_plan.h"
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
struct RunningLatencyStats {
    int count = 0;
    double mean = 0.0;
    double m2 = 0.0;

    void add(double value)
    {
        count++;
        const double delta = value - mean;
        mean += delta / count;
        const double delta2 = value - mean;
        m2 += delta * delta2;
    }

    double stddev() const { return count > 1 ? std::sqrt(m2 / static_cast<double>(count - 1)) : 0.0; }
};

std::map<int, int> make_link_level_by_id(const MeshTopo& mesh_topo)
{
    std::map<int, int> link_level_by_id;
    auto record_link_level
        = [&link_level_by_id](const std::vector<std::tuple<std::string, std::string, int>>& links, int level) {
              for (const auto& link : links) {
                  link_level_by_id[std::get<2>(link)] = level;
              }
          };

    for (const auto& mesh_entry : mesh_topo.mesh_map) {
        const auto& mesh = mesh_entry.second;
        for (const auto& kid_link_entry : mesh.kid_to_link) {
            record_link_level(kid_link_entry.second, mesh.level);
        }
        for (const auto& from_kid_entry : mesh.kid_to_kid_link) {
            for (const auto& to_kid_entry : from_kid_entry.second) {
                record_link_level(to_kid_entry.second, mesh.level);
            }
        }
    }
    return link_level_by_id;
}

std::map<int, RunningLatencyStats> make_spatial_latency_stats_by_level(
    const std::vector<std::tuple<std::string, std::string, int, float, float>>& probed_link_list,
    const std::map<int, int>& link_level_by_id)
{
    std::map<int, RunningLatencyStats> stats_by_level;
    for (const auto& link : probed_link_list) {
        const float latency = std::get<3>(link);
        if (!std::isfinite(latency)) {
            continue;
        }
        const auto level_it = link_level_by_id.find(std::get<2>(link));
        if (level_it == link_level_by_id.end()) {
            continue;
        }
        stats_by_level[level_it->second].add(latency);
    }
    return stats_by_level;
}

std::map<std::string, std::string> make_device_label_by_ip(
    const std::vector<std::string>& device_control_list, const std::vector<std::string>& device_ip_list)
{
    std::map<std::string, std::string> labels;
    for (size_t i = 0; i < device_ip_list.size(); i++) {
        std::string label = "dev";
        if (i < device_control_list.size()) {
            label += device_control_list[i];
        } else {
            label += std::to_string(i);
        }
        labels[device_ip_list[i]] = label;
    }
    return labels;
}

std::map<std::string, std::string> make_device_label_by_ip_from_allpath(const std::string& allpath_path)
{
    std::ifstream ifs(allpath_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("[pingpong-plan] allpath json unavailable: " + allpath_path);
    }
    json allpath = json::parse(ifs);
    std::map<std::string, std::string> labels;
    for (const auto& device : allpath.at("device_list")) {
        labels[device.at("device_ip").get<std::string>()] = device.at("label").get<std::string>();
    }
    return labels;
}

std::map<std::string, std::string> try_make_device_label_by_ip_from_allpath(const std::string& allpath_path)
{
    std::ifstream ifs(allpath_path);
    if (!ifs.is_open()) {
        return {};
    }
    json allpath = json::parse(ifs);
    std::map<std::string, std::string> labels;
    if (!allpath.contains("device_list")) {
        return labels;
    }
    for (const auto& device : allpath.at("device_list")) {
        labels[device.at("device_ip").get<std::string>()] = device.at("label").get<std::string>();
    }
    return labels;
}

void print_pingpong_plan(
    const std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>>& pinglist,
    const std::map<std::string, std::string>& device_label_by_ip)
{
    int total_task_count = 0;
    int total_port_count = 0;
    std::cout << "[pingpong-plan] begin" << std::endl;
    for (const auto& [mesh_name, mesh_pinglist] : pinglist) {
        int mesh_port_count = 0;
        for (const auto& [from_ip, to_ip, port_num] : mesh_pinglist) {
            mesh_port_count += port_num;
        }

        std::cout << "[pingpong-plan] mesh=" << mesh_name << ", task_count=" << mesh_pinglist.size()
                  << ", port_count=" << mesh_port_count << std::endl;

        int task_index = 0;
        for (const auto& [from_ip, to_ip, port_num] : mesh_pinglist) {
            const auto from_label_it = device_label_by_ip.find(from_ip);
            const auto to_label_it = device_label_by_ip.find(to_ip);
            const std::string from_label
                = from_label_it == device_label_by_ip.end() ? "unknown" : from_label_it->second;
            const std::string to_label = to_label_it == device_label_by_ip.end() ? "unknown" : to_label_it->second;
            std::cout << "  [" << task_index++ << "] " << from_label << "(" << from_ip << ") -> " << to_label << "("
                      << to_ip << ")"
                      << ", port_num=" << port_num << std::endl;
        }

        total_task_count += static_cast<int>(mesh_pinglist.size());
        total_port_count += mesh_port_count;
    }
    std::cout << "[pingpong-plan] total_mesh_count=" << pinglist.size() << ", total_task_count=" << total_task_count
              << ", total_port_count=" << total_port_count << std::endl;
    std::cout << "[pingpong-plan] end" << std::endl;
}

void print_l2_pingpong_plan(
    const std::vector<ProbeHelper::ExplicitPingTask>& tasks,
    const std::map<std::string, std::string>& device_label_by_ip)
{
    std::cout << "[pingpong-plan][L2] task_count=" << tasks.size() << std::endl;
    for (size_t i = 0; i < tasks.size(); i++) {
        const auto& task = tasks[i];
        std::cout << "  [" << i << "] " << task.tag << " " << device_label_by_ip.at(task.from_ip) << "(" << task.from_ip
                  << ") -> " << device_label_by_ip.at(task.to_ip) << "(" << task.to_ip << ")"
                  << ", src_sport=" << task.src_port << std::endl;
    }
}

void write_l2_result_header(std::ostream& ofs, const std::string& value_name)
{
    ofs << "turn\ttask_index\ttag\tfrom_label\tfrom_ip\tto_label\tto_ip\tsrc_sport\t" << value_name;
    ofs << std::endl;
}

std::vector<int> get_l1_link_ids_for_device(const MeshTopo& mesh_topo, const std::string& device_ip)
{
    std::vector<int> link_ids;
    for (const auto& [mesh_name, mesh] : mesh_topo.mesh_map) {
        auto link_it = mesh.kid_to_link.find(device_ip);
        if (link_it == mesh.kid_to_link.end()) {
            continue;
        }
        for (const auto& link : link_it->second) {
            link_ids.emplace_back(std::get<2>(link));
        }
    }
    std::sort(link_ids.begin(), link_ids.end());
    link_ids.erase(std::unique(link_ids.begin(), link_ids.end()), link_ids.end());
    return link_ids;
}

float sum_solved_link_latency(const std::vector<int>& link_ids, const std::vector<float>& link_lat, bool& ok)
{
    float total = 0.0f;
    if (link_ids.empty()) {
        ok = false;
        return std::numeric_limits<float>::quiet_NaN();
    }
    for (int link_id : link_ids) {
        if (link_id < 0 || link_id >= static_cast<int>(link_lat.size()) || std::isnan(link_lat[link_id])) {
            ok = false;
            return std::numeric_limits<float>::quiet_NaN();
        }
        total += link_lat[link_id];
    }
    ok = true;
    return total;
}

std::pair<std::string, std::string>
get_display_link_endpoint(const MeshTopo& mesh_topo, const std::tuple<std::string, std::string, int>& link)
{
    const int link_id = std::get<2>(link);
    for (const auto& [mesh_name, mesh] : mesh_topo.mesh_map) {
        for (const auto& [kid, links] : mesh.kid_to_link) {
            for (const auto& kid_link : links) {
                if (std::get<2>(kid_link) == link_id) {
                    const std::string tor_ip
                        = !std::get<1>(kid_link).empty() ? std::get<1>(kid_link) : std::get<0>(kid_link);
                    return {kid, tor_ip};
                }
            }
        }
    }
    return {std::get<0>(link), std::get<1>(link)};
}

void print_l2_pingpong_result(
    const std::vector<ProbeHelper::ExplicitPingTask>& tasks,
    const std::map<std::string, std::string>& device_label_by_ip, const std::map<std::string, int>& device_ip_to_index,
    std::vector<int> task_index_by_src, const std::vector<std::vector<float>>& pingpong_p99_lat,
    const std::vector<std::vector<float>>& pingpong_pass_rate, const MeshTopo& mesh_topo,
    const std::vector<float>& link_lat, int turn, std::ostream* l2_latency_file, std::ostream* l2_passrate_file)
{
    std::cout << "[pingpong-result][L2] task_count=" << tasks.size() << ", method=raw_p99_minus_two_l1_links"
              << std::endl;
    for (size_t i = 0; i < tasks.size(); i++) {
        const auto& task = tasks[i];
        const auto from_label_it = device_label_by_ip.find(task.from_ip);
        const auto to_label_it = device_label_by_ip.find(task.to_ip);
        const std::string from_label = from_label_it == device_label_by_ip.end() ? "unknown" : from_label_it->second;
        const std::string to_label = to_label_it == device_label_by_ip.end() ? "unknown" : to_label_it->second;
        auto write_l2_row = [&](std::ostream* ofs, const std::string& value) {
            if (ofs == nullptr) {
                return;
            }
            (*ofs) << turn << "\t" << i << "\t" << task.tag << "\t" << from_label << "\t" << task.from_ip << "\t"
                   << to_label << "\t" << task.to_ip << "\t" << task.src_port << "\t" << value << std::endl;
        };

        auto from_index_it = device_ip_to_index.find(task.from_ip);
        if (from_index_it == device_ip_to_index.end()) {
            std::cout << "  [" << i << "] " << task.tag << " skip: unknown source " << task.from_ip << std::endl;
            write_l2_row(l2_latency_file, "nan");
            write_l2_row(l2_passrate_file, "nan");
            continue;
        }

        const int from_index = from_index_it->second;
        const int task_index = task_index_by_src[from_index]++;
        if (from_index < 0 || from_index >= static_cast<int>(pingpong_p99_lat.size()) || task_index < 0
            || task_index >= static_cast<int>(pingpong_p99_lat[from_index].size())) {
            std::cout << "  [" << i << "] " << task.tag << " skip: missing pingpong result index src=" << from_index
                      << ", task=" << task_index << std::endl;
            write_l2_row(l2_latency_file, "nan");
            write_l2_row(l2_passrate_file, "nan");
            continue;
        }

        bool from_l1_ok = false;
        bool to_l1_ok = false;
        const auto from_l1_ids = get_l1_link_ids_for_device(mesh_topo, task.from_ip);
        const auto to_l1_ids = get_l1_link_ids_for_device(mesh_topo, task.to_ip);
        const float from_l1_lat = sum_solved_link_latency(from_l1_ids, link_lat, from_l1_ok);
        const float to_l1_lat = sum_solved_link_latency(to_l1_ids, link_lat, to_l1_ok);
        const float raw_p99_lat = pingpong_p99_lat[from_index][task_index];
        const float pass_rate = std::exp2f(pingpong_pass_rate[from_index][task_index]);
        const bool l1_ok = from_l1_ok && to_l1_ok;
        const float l1_subtracted = l1_ok ? from_l1_lat + to_l1_lat : std::numeric_limits<float>::quiet_NaN();
        const float l2_path_lat = l1_ok ? raw_p99_lat - l1_subtracted : std::numeric_limits<float>::quiet_NaN();

        std::cout << "  [" << i << "] " << task.tag << " " << from_label << "(" << task.from_ip << ") -> " << to_label
                  << "(" << task.to_ip << ")"
                  << ", src_sport=" << task.src_port << ", raw_p99_lat=" << raw_p99_lat << "ms"
                  << ", l1_subtracted=" << l1_subtracted << "ms"
                  << ", l2_path_lat=" << l2_path_lat << "ms"
                  << ", pass_rate=" << pass_rate << std::endl;
        write_l2_row(l2_latency_file, std::to_string(l2_path_lat));
        write_l2_row(l2_passrate_file, std::to_string(pass_rate));
    }
    if (l2_latency_file != nullptr) {
        l2_latency_file->flush();
    }
    if (l2_passrate_file != nullptr) {
        l2_passrate_file->flush();
    }
}

std::string format_metrics_timestamp()
{
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now - seconds).count();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm{};
    localtime_r(&now_time, &local_tm);

    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%F %T") << "." << std::setw(3) << std::setfill('0') << millis;
    return oss.str();
}

bool topology_result_is_incomplete(const std::string& path)
{
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        return false;
    }
    const json topology = json::parse(ifs);
    return topology.value("status", "complete") == "incomplete";
}
} // namespace

int main(int argc, char** argv)
{
    // 通过探测得到MeshTopo
    auto link_lat_path = get_output_path() + "/link_lat.txt";
    auto link_pass_rate_path = get_output_path() + "/link_pass_rate.txt";
    auto bad_link_path = get_output_path() + "/bad_link.txt";
    auto bad_link_candidate_path = get_output_path() + "/bad_link_candidate.txt";
    auto l2_latency_dir = get_output_path() + "/l2_status";
    auto l2_latency_path = l2_latency_dir + "/l2_path_lat.txt";
    auto l2_passrate_path = l2_latency_dir + "/l2_path_passrate.txt";
    CLI::App app{"test host"};
    std::string rank_table_file = "./control_json/910b2_info.json"; // Default json file
    bool print_pingpong_plan_only = false;
    bool l1_only = false;
    bool no_metrics = false;
    app.add_option("-f,--file", rank_table_file, "rank table file");
    app.add_flag(
        "--print-pingpong-plan", print_pingpong_plan_only, "print pingpong task plan and exit before pingpong");
    app.add_flag("--l1-only", l1_only, "only build and solve L1 pingpong tasks, disable L2 fullmesh tasks and output");
    app.add_flag("--no-metrics", no_metrics, "disable PFC/CNP counter collection");
    CLI11_PARSE(app, argc, argv);

    // 在循环开始前打开文件
    static const int output_width = 25;
    static const int float_width = 15;
    const bool enable_l2_pingpong = !l1_only;
    const bool enable_metrics = !no_metrics;

    make_abs_path(rank_table_file);
    auto probe_config = load_probe_runtime_config(rank_table_file);
    auto json_path = probe_config.enabled ? make_probe_topo_lldp_json_path(get_project_path(), probe_config) :
                                            make_probe_topo_json_path(get_project_path(), probe_config);
    if (probe_config.enabled) {
        std::cout << "[config] probe.scope=" << format_probe_scope(probe_config.probe_controller_pingpong.scope)
                  << ", times=" << probe_config.probe_controller_pingpong.times
                  << ", turns=" << probe_config.probe_controller_pingpong.turns
                  << ", payload_len=" << probe_config.probe_controller_pingpong.payload_len
                  << ", interval_ms=" << probe_config.probe_controller_pingpong.interval_ms
                  << ", topo_input=" << json_path << std::endl;
    }
    if (topology_result_is_incomplete(json_path)) {
        std::cerr << "[probe_controller] topology status is incomplete; skip pingpong and alert generation: "
                  << json_path << std::endl;
        return -1;
    }
    std::unique_ptr<ControlTopo> ct_holder;
    if (probe_config.enabled) {
        ct_holder = std::make_unique<ControlTopo>(make_control_topo_json(probe_config.probe_controller_pingpong.scope));
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
    auto tracert_ports_multi_parallel
        = [ct_p, tracert_sport_begin](
              const std::vector<std::string>& device_list, const std::vector<std::string>& src_devIp_list,
              const std::vector<std::vector<std::string>>& dst_list, const std::vector<std::vector<int>>& port_num) {
              static int tracert_batch_id = 0;
              const int batch_id = ++tracert_batch_id;
              const int total_port_probe = LogTracertBatchPlan(
                  batch_id, tracert_sport_begin, "sport_begin", device_list, src_devIp_list, dst_list, port_num);
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
              const auto elapsed_ms = GetElapsedMs(rpc_start_time, rpc_end_time);
              std::cout << "[tracert][batch " << batch_id << "] rpc done, elapsed_ms=" << elapsed_ms
                        << ", result_path_count=" << path_stats.first << ", empty_path_count=" << path_stats.second
                        << std::endl;
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

    if (print_pingpong_plan_only && probe_config.enabled) {
        const auto allpath_path = make_allpath_json_path(get_project_path(), probe_config);
        MeshTopo lldp_mesh_topo(json_path);
        auto pinglist = probe_plan::build_l0_pinglist_from_lldp_topo(lldp_mesh_topo);
        std::map<std::string, std::string> device_label_by_ip;
        std::vector<ProbeHelper::ExplicitPingTask> l2_ping_tasks;
        if (enable_l2_pingpong) {
            device_label_by_ip = make_device_label_by_ip_from_allpath(allpath_path);
            l2_ping_tasks
                = probe_plan::build_l2_fullmesh_ping_tasks(lldp_mesh_topo, device_label_by_ip, DEFAULT_SRC_PORT);
        } else {
            device_label_by_ip = try_make_device_label_by_ip_from_allpath(allpath_path);
        }
        print_pingpong_plan(pinglist, device_label_by_ip);
        if (enable_l2_pingpong) {
            print_l2_pingpong_plan(l2_ping_tasks, device_label_by_ip);
        } else {
            std::cout << "[pingpong-plan][L2] disabled by --l1-only" << std::endl;
        }
        std::cout << "[pingpong-plan] exit before set_pinglist and pingpong probe" << std::endl;
        return 0;
    }

    // 打印拓扑
    ProbeHelper newProbeHelper(
        ct_p, hccn_device_ip_list_parallel, tracert_ports_multi_parallel, pinglist_insert_muti_parallel,
        ud_pingpong_tx_muti_parallel);
    if (!newProbeHelper.device_count_complete()) {
        std::cerr << "[probe_controller] device count mismatch: configured=" << newProbeHelper.configured_device_count()
                  << ", discovered=" << newProbeHelper.discovered_device_count()
                  << "; skip pingpong and alert generation" << std::endl;
        return -1;
    }
    if (probe_config.enabled) {
        newProbeHelper.set_tracert_port_config(
            probe_config.probe_topo_tracert.tree_probe_sport_count, probe_config.probe_topo_tracert.sport_count);
    }
    newProbeHelper.set_mesh_topo(json_path);
    std::map<std::string, std::vector<std::tuple<std::string, std::string, int>>> pinglist;
    if (probe_config.enabled) {
        pinglist = probe_plan::build_l0_pinglist_from_lldp_topo(newProbeHelper.meshTopo);
    } else {
        pinglist = newProbeHelper.get_pinglist();
    }
    pinglist = ProbeHelper::limit_pinglist_targets_per_source(pinglist);
    auto device_label_by_ip = make_device_label_by_ip(ct_p->LeafList(), newProbeHelper.device_ip_list);
    std::vector<ProbeHelper::ExplicitPingTask> l2_ping_tasks;
    if (enable_l2_pingpong) {
        l2_ping_tasks
            = probe_plan::build_l2_fullmesh_ping_tasks(newProbeHelper.meshTopo, device_label_by_ip, DEFAULT_SRC_PORT);
    }
    print_pingpong_plan(pinglist, device_label_by_ip);
    if (enable_l2_pingpong) {
        print_l2_pingpong_plan(l2_ping_tasks, device_label_by_ip);
    } else {
        std::cout << "[pingpong-plan][L2] disabled by --l1-only" << std::endl;
    }
    if (print_pingpong_plan_only) {
        std::cout << "[pingpong-plan] exit before set_pinglist and pingpong probe" << std::endl;
        return 0;
    }

    std::ofstream lat_file(link_lat_path);
    std::ofstream pass_rate_file(link_pass_rate_path);
    std::ofstream bad_link_file(bad_link_path);
    std::ofstream bad_link_candidate_file(bad_link_candidate_path);
    if (enable_l2_pingpong) {
        std::filesystem::create_directories(l2_latency_dir);
    }
    std::ofstream l2_latency_file;
    std::ofstream l2_passrate_file;
    if (enable_l2_pingpong) {
        l2_latency_file.open(l2_latency_path);
        l2_passrate_file.open(l2_passrate_path);
    }
    // 检查文件是否成功打开
    if (!lat_file.is_open() || !pass_rate_file.is_open() || !bad_link_file.is_open()
        || !bad_link_candidate_file.is_open()
        || (enable_l2_pingpong && (!l2_latency_file.is_open() || !l2_passrate_file.is_open()))) {
        std::cerr << "failed to open output file!" << std::endl;
        return -1;
    }
    lat_file << std::fixed << std::setprecision(3);
    pass_rate_file << std::fixed << std::setprecision(3);
    if (enable_l2_pingpong) {
        l2_latency_file << std::fixed << std::setprecision(3);
        l2_passrate_file << std::fixed << std::setprecision(3);
        write_l2_result_header(l2_latency_file, "l2_path_lat");
        write_l2_result_header(l2_passrate_file, "pass_rate");
    }

    std::vector<std::pair<std::string, std::vector<std::string>>> metrics_collector_routes;
    std::map<std::string, std::ofstream> metrics_counter_files;
    std::atomic<bool> metrics_collector_stop{false};
    std::thread metrics_collector_worker;
    if (enable_metrics) {
        try {
            std::filesystem::create_directories(get_output_path() + "/metrics");
            std::vector<std::string> metrics_counter_name;
            if (!ct_p->ServerTreeList().empty()) {
                metrics_counter_name
                    = proxy_call(ct_p->route_to_device(ct_p->ServerTreeList().front().ip), get_metrics_counter_name);
            }
            if (!metrics_counter_name.empty()) {
                for (size_t i = 1; i < metrics_counter_name.size(); i++) {
                    const auto& counter_name = metrics_counter_name[i];
                    auto ofs = std::ofstream(get_output_path() + "/metrics/" + counter_name + ".txt");
                    if (!ofs.is_open()) {
                        continue;
                    }
                    ofs << "time,port,value,valid" << std::endl;
                    metrics_counter_files.emplace(counter_name, std::move(ofs));
                }
                for (const auto& server : ct_p->ServerTreeList()) {
                    auto route = ct_p->route_to_device(server.ip);
                    const std::string host_name = clean_ip(route.empty() ? server.ip : route.front());
                    metrics_collector_routes.emplace_back(host_name, std::move(route));
                }
            }
            if (!metrics_collector_routes.empty() && !metrics_counter_files.empty()) {
                metrics_collector_worker = std::thread([&, metrics_counter_name]() {
                    auto next_time = std::chrono::steady_clock::now();
                    while (!metrics_collector_stop.load()) {
                        next_time += std::chrono::milliseconds(1000);
                        for (size_t i = 0; i < metrics_collector_routes.size(); i++) {
                            try {
                                const auto& host_name = metrics_collector_routes[i].first;
                                const auto& route = metrics_collector_routes[i].second;
                                auto values = proxy_call(route, get_metrics_counter_value);
                                const std::string timestamp = format_metrics_timestamp();
                                for (const auto& row : values) {
                                    if (row.size() < metrics_counter_name.size()) {
                                        continue;
                                    }
                                    const bool valid = row.size() > metrics_counter_name.size()
                                                       && row[metrics_counter_name.size()] != 0;
                                    const std::string port_name = host_name + "_dev" + std::to_string(row[0]);
                                    for (size_t counter_index = 1; counter_index < metrics_counter_name.size();
                                         counter_index++) {
                                        auto file_it = metrics_counter_files.find(metrics_counter_name[counter_index]);
                                        if (file_it == metrics_counter_files.end()) {
                                            continue;
                                        }
                                        file_it->second << timestamp << "," << port_name << "," << row[counter_index]
                                                        << "," << (valid ? "true" : "false") << std::endl;
                                    }
                                }
                                for (auto& [counter_name, ofs] : metrics_counter_files) {
                                    ofs.flush();
                                }
                            } catch (const std::exception& e) {
                                std::cout << "[metrics][Warning] counter rpc failed: " << e.what() << std::endl;
                            }
                        }
                        std::this_thread::sleep_until(next_time);
                    }
                });
            }
        } catch (const std::exception& e) {
            std::cout << "[metrics][Warning] disable counter collection: " << e.what() << std::endl;
        }
    }

    // 处理pinglist: 这里会通过 RPC 调用 pinglist_insert_muti，在远端完成 HCCN AddTarget 建链。
    if (enable_l2_pingpong) {
        newProbeHelper.set_explicit_ping_tasks(l2_ping_tasks);
    }
    std::cout << "========pingpong set_pinglist/AddTarget========" << std::endl;
    newProbeHelper.set_pinglist(pinglist);
    sleep(2);
    std::cout << "========pingpong set_pinglist/AddTarget done========" << std::endl;
    int pingpong_times = probe_config.enabled ? probe_config.probe_controller_pingpong.times : 50;
    int pingpong_turn = probe_config.enabled ? probe_config.probe_controller_pingpong.turns : 1000;

    auto& link_list = newProbeHelper.link_list;
    std::vector<int> probed_link_index;
    using link_res_type = std::tuple<std::string, std::string, int, float, float>;
    std::vector<link_res_type> probed_link_list;
    int probed_size;
    // pingpong + 求解
    for (int turn = 0; turn < pingpong_turn; turn++) {
        // 探测
        std::cout << "========pingpong========" << std::endl;
        auto pingpong_res = newProbeHelper.get_pingpong_res(pingpong_times);
        auto pingpong_p99_lat
            = newProbeHelper.reduce_pingpong_res(pingpong_res, ProbeHelper::PingpongMetric::P99Lat, pingpong_times);
        auto pingpong_pass_rate = newProbeHelper.reduce_pingpong_res(
            pingpong_res, ProbeHelper::PingpongMetric::LogPassRate, pingpong_times);
        // 打印路径值
        auto& device_ip_list = newProbeHelper.device_ip_list;
        std::vector<int> dst_list_to_task_index(device_ip_list.size(), 0);
        std::map<std::string, int> device_ip_to_index;
        for (int i = 0; i < device_ip_list.size(); i++) {
            auto& from = device_ip_list[i];
            device_ip_to_index[from] = i;
        }
        for (auto& [mesh_name, mesh_pinglist] : pinglist) // 遍历mesh
        {
            for (auto& [from_ip, to_ip, port_num] : mesh_pinglist) // 遍历mesh下每个ping任务
            {
                int from_index = device_ip_to_index[from_ip];
                auto& this_route_id_ports = newProbeHelper.meshTopo.tracert_res_id[from_ip][to_ip];
                for (int port = 0; port < port_num; port++) // 默认任务ports是从0递增的
                {
                    int this_task_index = dst_list_to_task_index[from_index]++; // 和创建pinglist时emplace时的顺序一致
                    float p99_lat = pingpong_p99_lat[from_index][this_task_index];
                    float log_pass_rate = pingpong_pass_rate[from_index][this_task_index];
                    std::cout << "[pingpong-raw][L1] [" << from_ip << "->" << to_ip << "]"
                              << ":" << port << ": round_trip_p99_lat: " << p99_lat
                              << ", pass_rate: " << std::exp2f(log_pass_rate) << std::endl;
                }
            }
        }
        auto l2_task_index_by_src = dst_list_to_task_index;
        // 求解
        auto link_lat = newProbeHelper.solve_pingpong_res(pingpong_p99_lat, pinglist);
        auto link_pass_rate = newProbeHelper.solve_pingpong_res(pingpong_pass_rate, pinglist);
        for (int i = 0; i < link_pass_rate.size(); i++) {
            link_pass_rate[i] = std::exp2f(link_pass_rate[i]);
        }
        if (enable_l2_pingpong) {
            print_l2_pingpong_result(
                l2_ping_tasks, device_label_by_ip, device_ip_to_index, l2_task_index_by_src, pingpong_p99_lat,
                pingpong_pass_rate, newProbeHelper.meshTopo, link_lat, turn,
                enable_l2_pingpong ? &l2_latency_file : nullptr, enable_l2_pingpong ? &l2_passrate_file : nullptr);
        }
        // 解析探测过的链路
        if (probed_link_index.size() == 0) {
            for (int i = 0; i < link_lat.size(); i++) {
                if (!std::isnan(link_lat[i])) {
                    probed_link_index.emplace_back(i);
                }
            }
            probed_size = probed_link_index.size();
            for (int i : probed_link_index) {
                const auto display_link = get_display_link_endpoint(newProbeHelper.meshTopo, link_list[i]);
                probed_link_list.emplace_back(
                    display_link.first, display_link.second, std::get<2>(link_list[i]), link_lat[i], link_pass_rate[i]);
            }
            for (int i = 0; i < probed_size; i++) {
                lat_file << std::setw(output_width)
                         << std::string("[") + std::get<0>(probed_link_list[i]) + "-" + std::get<1>(probed_link_list[i])
                                + "],";
                pass_rate_file << std::setw(output_width)
                               << std::string("[") + std::get<0>(probed_link_list[i]) + "-"
                                      + std::get<1>(probed_link_list[i]) + "],";
            }
            lat_file << std::endl;
            pass_rate_file << std::endl;
        }

        int begin = 0;
        for (int i : probed_link_index) {
            std::get<3>(probed_link_list[begin]) = link_lat[i];
            std::get<4>(probed_link_list[begin]) = link_pass_rate[i];
            begin++;
        }

        for (int i = 0; i < probed_size; i++) {
            auto& first_ip = std::get<0>(probed_link_list[i]);
            auto& second_ip = std::get<1>(probed_link_list[i]);
            float lantency = std::get<3>(probed_link_list[i]);
            float pass_rate = std::get<4>(probed_link_list[i]);
            std::cout << "[pingpong-solved][L1] [" << first_ip << "-" << second_ip << "] ";
            std::cout << "latency: " << lantency << "; ";
            std::cout << "pass rate: " << pass_rate << "; ";
            std::cout << std::endl;
            lat_file << std::setw(float_width) << lantency << std::setw(output_width - float_width) << ",";
            pass_rate_file << std::setw(float_width) << pass_rate << std::setw(output_width - float_width) << ",";
        }
        lat_file << std::endl;
        pass_rate_file << std::endl;

        static std::vector<RunningLatencyStats> l1_latency_stats;
        static std::vector<int> l1_latency_abnormal_streak;
        static std::vector<int> l1_latency_temporal_candidate_streak;
        static std::vector<int> l1_latency_spatial_candidate_streak;
        static std::map<int, int> link_level_by_id;
        if (l1_latency_stats.size() != probed_link_list.size()) {
            l1_latency_stats.assign(probed_link_list.size(), RunningLatencyStats{});
            l1_latency_abnormal_streak.assign(probed_link_list.size(), 0);
            l1_latency_temporal_candidate_streak.assign(probed_link_list.size(), 0);
            l1_latency_spatial_candidate_streak.assign(probed_link_list.size(), 0);
            link_level_by_id = make_link_level_by_id(newProbeHelper.meshTopo);
        }
        const auto spatial_latency_stats_by_level
            = make_spatial_latency_stats_by_level(probed_link_list, link_level_by_id);
        constexpr int min_sigma_samples = 5;
        constexpr double sigma_factor = 3.0;
        constexpr int latency_alarm_streak_threshold = 3;
        constexpr int latency_candidate_streak_threshold = 3;
        for (size_t i = 0; i < probed_link_list.size(); i++) {
            const auto& first_ip = std::get<0>(probed_link_list[i]);
            const auto& second_ip = std::get<1>(probed_link_list[i]);
            const float latency = std::get<3>(probed_link_list[i]);
            const float pass_rate = std::get<4>(probed_link_list[i]);
            if (!std::isfinite(latency)) {
                continue;
            }

            auto& stats = l1_latency_stats[i];
            const double temporal_stddev = stats.stddev();
            const double temporal_threshold = stats.mean + sigma_factor * temporal_stddev;
            const bool temporal_latency_abnormal
                = stats.count >= min_sigma_samples && temporal_stddev > 0.0 && latency > temporal_threshold;
            bool spatial_latency_abnormal = false;
            double spatial_threshold = 0.0;
            const auto level_it = link_level_by_id.find(std::get<2>(probed_link_list[i]));
            if (level_it != link_level_by_id.end()) {
                const auto spatial_stats_it = spatial_latency_stats_by_level.find(level_it->second);
                if (spatial_stats_it != spatial_latency_stats_by_level.end()) {
                    const auto& spatial_stats = spatial_stats_it->second;
                    const double spatial_stddev = spatial_stats.stddev();
                    spatial_threshold = spatial_stats.mean + sigma_factor * spatial_stddev;
                    spatial_latency_abnormal = spatial_stats.count >= min_sigma_samples && spatial_stddev > 0.0
                                               && latency > spatial_threshold;
                }
            }
            if (temporal_latency_abnormal && spatial_latency_abnormal) {
                l1_latency_abnormal_streak[i]++;
            } else {
                l1_latency_abnormal_streak[i] = 0;
            }
            if (temporal_latency_abnormal) {
                l1_latency_temporal_candidate_streak[i]++;
            } else {
                l1_latency_temporal_candidate_streak[i] = 0;
            }
            if (spatial_latency_abnormal) {
                l1_latency_spatial_candidate_streak[i]++;
            } else {
                l1_latency_spatial_candidate_streak[i] = 0;
            }
            if (l1_latency_temporal_candidate_streak[i] >= latency_candidate_streak_threshold
                || l1_latency_spatial_candidate_streak[i] >= latency_candidate_streak_threshold) {
                std::string candidate_msg = "[Candidate] latency abnormal link: ";
                candidate_msg += "[" + first_ip + "-" + second_ip + "] ";
                candidate_msg += "latency: " + std::to_string(latency) + "; ";
                candidate_msg += "pass rate: " + std::to_string(pass_rate) + "; ";
                candidate_msg += "temporal_threshold: " + std::to_string(temporal_threshold) + "; ";
                candidate_msg += "spatial_threshold: " + std::to_string(spatial_threshold) + "; ";
                candidate_msg
                    += "temporal_candidate_streak: " + std::to_string(l1_latency_temporal_candidate_streak[i]) + "; ";
                candidate_msg
                    += "spatial_candidate_streak: " + std::to_string(l1_latency_spatial_candidate_streak[i]) + "; ";
                bad_link_candidate_file << candidate_msg << std::endl;
            }
            if (l1_latency_abnormal_streak[i] >= latency_alarm_streak_threshold) {
                std::string warning_msg = "[Warning] abnormal link: ";
                warning_msg += "[" + first_ip + "-" + second_ip + "] ";
                warning_msg += "latency: " + std::to_string(latency) + "; ";
                warning_msg += "pass rate: " + std::to_string(pass_rate) + "; ";
                warning_msg += "temporal_threshold: " + std::to_string(temporal_threshold) + "; ";
                warning_msg += "spatial_threshold: " + std::to_string(spatial_threshold) + "; ";
                warning_msg += "latency_abnormal_streak: " + std::to_string(l1_latency_abnormal_streak[i]) + "; ";
                std::cout << warning_msg << std::endl;
                bad_link_file << warning_msg << std::endl;
            }
            stats.add(latency);
        }

        for (int i = 0; i < probed_link_list.size(); i++) {
            auto& first_ip = std::get<0>(probed_link_list[i]);
            auto& second_ip = std::get<1>(probed_link_list[i]);
            float latency = std::get<3>(probed_link_list[i]);
            float pass_rate = std::get<4>(probed_link_list[i]);
            if (pass_rate < 0.99) {
                std::string warning_msg = "[Warning] bad link: ";
                warning_msg += "[" + first_ip + "-" + second_ip + "] ";
                warning_msg += "latency: " + std::to_string(latency) + "; ";
                warning_msg += "pass rate: " + std::to_string(pass_rate) + "; ";
                std::cout << warning_msg << std::endl;
                bad_link_file << warning_msg << std::endl;
            }
        }
    }

    metrics_collector_stop.store(true);
    if (metrics_collector_worker.joinable()) {
        metrics_collector_worker.join();
    }
}
