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
#include <vector>
#include <thread>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <filesystem>
#include <utility>
#include <ifaddrs.h>
#include <netinet/in.h>

#include "rpc/server.h"

#include <CLI/CLI.hpp>

#include "rpc_call/proxy_call.hpp"
#include "tool/tracert/tracert.h"
#include "tool/metrics_collector/metrics_collector.h"
#include "tool/hccn_pingpong/rpc_func.h"
#include "tool/hccn_pingpong/hccn_pingpong.h"
#include "tool/cmd/get_IP.h"
#include "topo/probe_config.h"
#include "file_path/workspace.h"

namespace {
std::string resolve_control_json_path(const std::string& input_path)
{
    namespace fs = std::filesystem;

    fs::path input(input_path);
    std::vector<fs::path> candidates;
    if (input.is_absolute()) {
        candidates.emplace_back(input);
    } else {
        candidates.emplace_back(fs::current_path() / input);
        const std::string project_path = get_project_path();
        if (!project_path.empty()) {
            fs::path project_relative = input_path.front() == '.' ? fs::path(input_path.substr(2)) : input;
            candidates.emplace_back(fs::path(project_path) / project_relative);
        }
        candidates.emplace_back(
            fs::path("/root/tarce/disp_probe/disp_probe-main")
            / (input_path.front() == '.' ? fs::path(input_path.substr(2)) : input));
    }

    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (fs::exists(candidate, ec) && !ec) {
            return fs::absolute(candidate, ec).string();
        }
    }

    std::string fallback = input_path;
    make_abs_path(fallback);
    return fallback;
}

std::vector<int>
get_pingpong_init_dev_ids(const ProbeRuntimeConfig& probe_config, const std::vector<std::string>& local_ips)
{
    std::vector<int> dev_ids;
    if (!probe_config.enabled) {
        return dev_ids;
    }

    for (const auto& local_ip : local_ips) {
        auto scope_it = probe_config.probe_controller_pingpong.scope.find(local_ip);
        if (scope_it == probe_config.probe_controller_pingpong.scope.end()) {
            continue;
        }

        std::set<int> seen;
        for (const auto& device_id_str : scope_it->second) {
            int dev_id = std::stoi(device_id_str);
            if (dev_id < 0) {
                throw std::runtime_error("[rpc_host] probe.scope device id must be non-negative");
            }
            if (seen.insert(dev_id).second) {
                dev_ids.emplace_back(dev_id);
            }
        }
        return dev_ids;
    }

    return dev_ids;
}

std::string join_strings(const std::vector<std::string>& items)
{
    std::string result = "[";
    for (size_t i = 0; i < items.size(); i++) {
        if (i != 0) {
            result += ",";
        }
        result += items[i];
    }
    result += "]";
    return result;
}

bool is_ethernet_interface(const std::string& name) { return name.rfind("eth", 0) == 0 || name.rfind("en", 0) == 0; }

std::string get_first_ethernet_interface_name()
{
    struct ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) == -1) {
        return "";
    }

    std::string iface_name;
    for (auto ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr || ifa->ifa_addr->sa_family != AF_INET) {
            continue;
        }

        std::string name(ifa->ifa_name);
        if (is_ethernet_interface(name)) {
            iface_name = std::move(name);
            break;
        }
    }

    freeifaddrs(ifaddr);
    return iface_name;
}
} // namespace

int main(int argc, char* argv[])
{
    std::string net_dev_name = get_first_ethernet_interface_name();
    std::cout << "net_dev_name: " << net_dev_name << std::endl;
    auto net_dev_ip_list = get_ethernet_IPv4_addresses([&net_dev_name](std::string dev) {
        return dev == net_dev_name;
    });

    std::string net_dev_ip;
    if (!net_dev_name.empty() && !net_dev_ip_list.empty()) {
        net_dev_ip = net_dev_ip_list[0];
    }
    std::cout << "net_dev_ip: " << net_dev_ip << std::endl;

    bool is_host = !net_dev_name.empty() && !net_dev_ip_list.empty();

    CLI::App app{"test host"};
    std::string rank_table_file = "./control_json/910b2_info.json";
    app.add_option("-f,--file", rank_table_file, "control json file");
    app.add_option("-d,--dev", net_dev_name, "Network device name");
    app.add_option("-i,--ip", net_dev_ip, "Network device IP");
    int port = RPC_DEFAULT_PORT;
    app.add_option("-p,--port", port, "Port to listen on");
    bool pingpongLocalLog = false;
    std::string pingpongLogDir = "/root/output";
    app.add_flag("--pingpong-local-log", pingpongLocalLog, "write local HCCN pingpong result logs on each rpc_host");
    app.add_option("--pingpong-log-dir", pingpongLogDir, "local HCCN pingpong result log root directory");
    CLI11_PARSE(app, argc, argv);
    if (port <= 0 || port > 65535) {
        throw std::runtime_error("[rpc_host] port must be in [1, 65535]");
    }

    set_hccn_pingpong_local_log(pingpongLocalLog, pingpongLogDir);
    std::cout << "[hccn_pingpong] local_log=" << (pingpongLocalLog ? "on" : "off") << ", log_dir=" << pingpongLogDir
              << std::endl;
    rank_table_file = resolve_control_json_path(rank_table_file);
    ProbeRuntimeConfig probe_config;
    bool has_probe_config = false;
    try {
        probe_config = load_probe_runtime_config(rank_table_file);
        has_probe_config = true;
    } catch (const std::exception& e) {
        std::cerr << "[hccn_pingpong] warning: " << e.what() << ", skip rpc_host pong keepalive init" << std::endl;
    }
    auto local_ip_list = get_ethernet_IPv4_addresses();
    if (!net_dev_ip.empty()
        && std::find(local_ip_list.begin(), local_ip_list.end(), net_dev_ip) == local_ip_list.end()) {
        local_ip_list.emplace_back(net_dev_ip);
    }
    std::cout << "[hccn_pingpong] config=" << rank_table_file << ", local_ips=" << join_strings(local_ip_list)
              << ", probe.scope=" << format_probe_scope(probe_config.probe_controller_pingpong.scope) << std::endl;
    if (has_probe_config) {
        set_hccn_pingpong_payload_len(static_cast<u32>(probe_config.probe_controller_pingpong.payload_len));
        set_hccn_pingpong_interval_ms(static_cast<u32>(probe_config.probe_controller_pingpong.interval_ms));
        std::cout << "[hccn_pingpong] payload_len=" << probe_config.probe_controller_pingpong.payload_len
                  << ", interval_ms=" << probe_config.probe_controller_pingpong.interval_ms << std::endl;
        auto init_dev_ids = get_pingpong_init_dev_ids(probe_config, local_ip_list);
        hccn_pingpong_init_local_devices_for_rpc(init_dev_ids);
        if (!init_dev_ids.empty()) {
            std::cout << "[metrics] start counter thread, dev_ids=";
            for (size_t i = 0; i < init_dev_ids.size(); i++) {
                if (i != 0) {
                    std::cout << ",";
                }
                std::cout << init_dev_ids[i];
            }
            std::cout << std::endl;
            std::thread([init_dev_ids]() {
                metrics_collector_thread(init_dev_ids);
            }).detach();
        }
    }

    const std::string bind_ip = net_dev_ip.empty() ? "127.0.0.1" : net_dev_ip;
    std::cout << "[rpc_host] listen=" << bind_ip << ":" << port << std::endl;
    rpc::server srv(bind_ip, static_cast<uint16_t>(port));

    proxy_bind(srv, tracert_ports_multi_by_src_ip);
    proxy_bind(srv, tracert_ports_multi_by_src_ip_with_sport_begin);
    proxy_bind(srv, hccn_device_ip_list);
    proxy_bind(srv, hccn_lldp_mgmt_ip);

    proxy_bind(srv, pinglist_insert_muti);
    proxy_bind(srv, ud_pingpong_tx_muti);

    proxy_bind(srv, get_metrics_counter_name);
    proxy_bind(srv, get_metrics_counter_value);

    srv.async_run(RPC_DEFAULT_MAX_CONNECTIONS);
    srv.run();
    hccn_pingpong_cleanup();

    for (auto& th : rpc_threads) {
        th.join();
    }

    return 0;
}
