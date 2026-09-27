/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rpc_func.h"
#include "hccn_pingpong.h"

#include <stdexcept>
#include <cstdio>
#include <iostream>
#include <memory>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <chrono>

std::vector<std::thread> rpc_threads;

struct HccnPingTask {
    std::string src_devIp;
    std::vector<std::string> target_devIp_list;
    std::vector<int> src_port;
    int rx_num = 0;
};

static std::map<std::string, HccnPingTask> hccn_ping_tasks;
static std::map<std::string, int> hccn_dev_ids;
static std::mutex hccn_ping_mutex;

static std::string get_hccn_dev_ip(int dev_id)
{
    std::ostringstream cmd;
    cmd << "hccn_tool -i " << dev_id << " -ip -g";
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.str().c_str(), "r"), pclose);
    if (!pipe) {
        return "";
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
        std::string line(buffer);
        const std::string prefix = "ipaddr:";
        if (line.rfind(prefix, 0) == 0) {
            std::string ip = line.substr(prefix.size());
            while (!ip.empty() && (ip.back() == '\n' || ip.back() == '\r' || ip.back() == ' ')) {
                ip.pop_back();
            }
            return ip;
        }
    }
    return "";
}

static int find_hccn_dev_id(const std::string& src_devIp)
{
    auto cached = hccn_dev_ids.find(src_devIp);
    if (cached != hccn_dev_ids.end()) {
        return cached->second;
    }

    for (int dev_id = 0; dev_id < 8; dev_id++) {
        std::ostringstream cmd;
        cmd << "hccn_tool -i " << dev_id << " -ip -g";
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.str().c_str(), "r"), pclose);
        if (!pipe) {
            continue;
        }

        char buffer[256];
        std::string output;
        while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
            output += buffer;
        }

        std::istringstream iss(output);
        std::string line;
        while (std::getline(iss, line)) {
            const std::string prefix = "ipaddr:";
            if (line.rfind(prefix, 0) == 0 && line.substr(prefix.size()) == src_devIp) {
                hccn_dev_ids[src_devIp] = dev_id;
                return dev_id;
            }
        }
    }

    throw std::runtime_error("[hccn_pingpong] cannot find devId for src_devIp=" + src_devIp);
}

void hccn_pingpong_init_local_devices_for_rpc(const std::vector<int>& dev_ids)
{
    if (dev_ids.empty()) {
        std::cout << "[hccn_pingpong] init local devices skipped, empty dev id list" << std::endl;
        return;
    }
    std::cout << "[hccn_pingpong] init local devices start, dev_ids=[";
    for (size_t i = 0; i < dev_ids.size(); i++) {
        if (i != 0) {
            std::cout << ",";
        }
        std::cout << dev_ids[i];
    }
    std::cout << "]" << std::endl;
    std::vector<int> started_dev_ids;
    for (int dev_id : dev_ids) {
        const std::string dev_ip = get_hccn_dev_ip(dev_id);
        if (dev_ip.empty()) {
            std::cout << "[hccn_pingpong] init skip dev=" << dev_id << ", empty ip" << std::endl;
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(hccn_ping_mutex);
            hccn_dev_ids[dev_ip] = dev_id;
        }
        rpc_threads.emplace_back([dev_id, dev_ip]() {
            const int ret = hccn_pingpong_init(static_cast<u32>(dev_id), dev_ip, true);
            std::cout << "[hccn_pingpong] keepalive exit unexpectedly, dev=" << dev_id << ", ip=" << dev_ip
                      << ", ret=" << ret << std::endl;
        });
        rpc_threads.back().detach();
        started_dev_ids.emplace_back(dev_id);
    }

    for (int dev_id : started_dev_ids) {
        bool ready = false;
        for (int retry = 0; retry < 30; retry++) {
            if (hccn_pingpong_is_initialized(static_cast<u32>(dev_id), true)) {
                ready = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        std::cout << "[hccn_pingpong] init dev=" << dev_id << ", ready=" << (ready ? "true" : "false") << std::endl;
    }
    std::cout << "[hccn_pingpong] init local devices done" << std::endl;
}

int pinglist_insert_muti(
    std::string src_devIp, std::vector<std::string> target_devIp_list, std::vector<int> src_hccn_ports,
    std::vector<int> src_hccn_ports_rx, int rx_num)
{
    (void)src_hccn_ports_rx;
    if (target_devIp_list.size() != src_hccn_ports.size()) {
        throw std::runtime_error("[hccn_pingpong] pinglist argument size mismatch");
    }

    if (target_devIp_list.empty()) {
        HccnPingTask task;
        task.src_devIp = src_devIp;
        task.rx_num = rx_num;
        std::lock_guard<std::mutex> lock(hccn_ping_mutex);
        hccn_ping_tasks[src_devIp] = task;
        return 0;
    }

    std::lock_guard<std::mutex> lock(hccn_ping_mutex);

    int dev_id = find_hccn_dev_id(src_devIp);
    if (hccn_pingpong_set_targets(static_cast<u32>(dev_id), src_devIp, target_devIp_list, src_hccn_ports) != 0) {
        throw std::runtime_error("[hccn_pingpong] hccn set targets failed");
    }

    HccnPingTask task;
    task.src_devIp = src_devIp;
    task.target_devIp_list = target_devIp_list;
    task.src_port = src_hccn_ports;
    task.rx_num = rx_num;
    hccn_ping_tasks[src_devIp] = task;

    for (size_t i = 0; i < target_devIp_list.size(); i++) {
        std::cout << "[INFO](pinglist_insert_muti) target[" << i << "]: dstIp=" << target_devIp_list[i]
                  << " srcPort=" << src_hccn_ports[i] << std::endl;
    }

    return static_cast<int>(target_devIp_list.size());
}

std::vector<std::vector<uint64_t>> ud_pingpong_tx_muti(std::string src_devIp, int times)
{
    HccnPingTask task;
    int dev_id = -1;

    {
        std::lock_guard<std::mutex> lock(hccn_ping_mutex);
        auto task_iter = hccn_ping_tasks.find(src_devIp);
        if (task_iter == hccn_ping_tasks.end()) {
            throw std::runtime_error("[hccn_pingpong] pinglist not initialized, src_devIp=" + src_devIp);
        }
        task = task_iter->second;
        dev_id = find_hccn_dev_id(src_devIp);
    }

    auto hccn_results = hccn_pingpong_batch_ping(static_cast<u32>(dev_id), times);

    return hccn_pingpong_to_rpc_result(hccn_results);
}
