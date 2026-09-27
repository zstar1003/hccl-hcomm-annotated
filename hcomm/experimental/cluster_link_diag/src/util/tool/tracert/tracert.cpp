/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tracert.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
static const int HCCN_WAIT_SECONDS = 1;
static const int HCCN_MAX_HOPS = 8;
static const int HCCN_CMD_TIMEOUT_SECONDS = 8;
static const int HCCN_TRACE_ATTEMPTS = 2;

std::string control_device_to_local_device_id(const std::string& device_id);

std::string run_hccn_tool_command(const std::string& cmd)
{
    std::array<char, 4096> buffer{};
    std::string output;
    std::string full_cmd = "(timeout " + std::to_string(HCCN_CMD_TIMEOUT_SECONDS) + "s " + cmd
                           + "; rc=$?; "
                             "if [ \"$rc\" -eq 124 ]; then echo __HCCN_TIMEOUT__; "
                             "elif [ \"$rc\" -ne 0 ]; then echo __HCCN_EXIT_CODE__=$rc; fi) 2>&1";

    auto close_pipe = [](FILE* pipe) {
        if (pipe)
            pclose(pipe);
    };
    std::unique_ptr<FILE, decltype(close_pipe)> pipe(popen(full_cmd.c_str(), "r"), close_pipe);
    if (!pipe)
        throw std::runtime_error("[tracert] popen failed");

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        output += buffer.data();
    }
    return output;
}

std::string clean_ip_token(std::string token)
{
    while (!token.empty() && (token.front() == '(' || token.front() == '['))
        token.erase(token.begin());
    while (!token.empty() && (token.back() == ')' || token.back() == ']' || token.back() == ',' || token.back() == ';'))
        token.pop_back();
    return token;
}

std::string trim_copy(const std::string& value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
        return "";
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

bool looks_like_ipv4(const std::string& ip)
{
    int dots = 0;
    int parts = 0;
    bool has_digit = false;
    for (char c : ip) {
        if (std::isdigit(static_cast<unsigned char>(c))) {
            has_digit = true;
            continue;
        }
        if (c == '.') {
            dots++;
            parts += has_digit ? 1 : 0;
            has_digit = false;
            continue;
        }
        return false;
    }
    parts += has_digit ? 1 : 0;
    return dots == 3 && parts == 4;
}

bool looks_like_unsigned_int(const std::string& value)
{
    return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return std::isdigit(c);
    });
}

std::string require_local_device_id(const std::string& device_id)
{
    const std::string local_device_id = control_device_to_local_device_id(device_id);
    if (!looks_like_unsigned_int(local_device_id)) {
        throw std::invalid_argument("[tracert] invalid device_id: " + device_id);
    }
    return local_device_id;
}

std::vector<std::string> parse_traceroute_path(const std::string& output, const std::string& dst_ip)
{
    std::vector<std::string> path;
    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        std::istringstream line_stream(line);
        std::string hop_token;
        line_stream >> hop_token;
        if (hop_token.empty() || !std::all_of(hop_token.begin(), hop_token.end(), [](unsigned char c) {
                return std::isdigit(c);
            }))
            continue;

        long hop = 0;
        try {
            hop = std::stol(hop_token);
        } catch (const std::exception&) {
            continue;
        }
        if (hop <= 0 || hop > HCCN_MAX_HOPS * 2)
            continue;

        std::string found_ip;
        std::string token;
        while (line_stream >> token) {
            std::string ip = clean_ip_token(token);
            if (looks_like_ipv4(ip)) {
                found_ip = ip;
                break;
            }
        }

        if (found_ip.empty())
            continue;

        if (path.size() < static_cast<size_t>(hop))
            path.resize(static_cast<size_t>(hop));
        path[static_cast<size_t>(hop - 1)] = found_ip;

        if (found_ip == dst_ip)
            break;
    }
    return path;
}

std::string build_traceroute_cmd(int src_device_id, const std::string& target, int sport)
{
    if (!looks_like_ipv4(target)) {
        throw std::invalid_argument("[tracert] invalid target ip: " + target);
    }
    if (src_device_id < 0) {
        throw std::invalid_argument("[tracert] invalid src_device_id: " + std::to_string(src_device_id));
    }
    std::ostringstream cmd;
    cmd << "hccn_tool -i " << src_device_id << " -traceroute -g"
        << " -w " << HCCN_WAIT_SECONDS << " -m " << HCCN_MAX_HOPS << " -sport " << sport << " -d " << target;
    return cmd.str();
}

std::string build_traceroute_reset_cmd(int src_device_id)
{
    std::ostringstream cmd;
    cmd << "hccn_tool -i " << src_device_id << " -traceroute reset";
    return cmd.str();
}

std::string control_device_to_local_device_id(const std::string& device_id)
{
    const auto pos = device_id.find('_');
    return pos == std::string::npos ? device_id : device_id.substr(0, pos);
}

int find_hccn_dev_id(const std::string& srcDevIp)
{
    for (int dev_id = 0; dev_id < 8; dev_id++) {
        std::ostringstream cmd;
        cmd << "hccn_tool -i " << dev_id << " -ip -g";
        std::istringstream lines(run_hccn_tool_command(cmd.str()));
        std::string line;
        while (std::getline(lines, line)) {
            const std::string prefix = "ipaddr:";
            if (line.rfind(prefix, 0) == 0 && line.substr(prefix.size()) == srcDevIp)
                return dev_id;
        }
    }
    throw std::runtime_error("[tracert] cannot find devId for srcDevIp=" + srcDevIp);
}

std::vector<std::string> probe_path_until_success(int src_device_id, const std::string& target, int sport)
{
    for (int attempt = 0; attempt < HCCN_TRACE_ATTEMPTS; attempt++) {
        std::string output = run_hccn_tool_command(build_traceroute_cmd(src_device_id, target, sport));
        std::vector<std::string> path = parse_traceroute_path(output, target);
        if (!path.empty())
            return path;
        if (attempt + 1 < HCCN_TRACE_ATTEMPTS)
            run_hccn_tool_command(build_traceroute_reset_cmd(src_device_id));
    }
    return {};
}
} // namespace

Tracert::Tracert(int src_device_id) : src_device_id_(src_device_id) {}
Tracert::~Tracert() = default;

std::vector<std::vector<std::string>> Tracert::trace(const std::vector<std::pair<std::string, int>>& target_port_list)
{
    std::vector<std::vector<std::string>> results;
    results.reserve(target_port_list.size());
    for (const auto& [target, port] : target_port_list)
        results.push_back(probe_path_until_success(src_device_id_, target, port));
    return results;
}

std::vector<std::vector<std::vector<std::string>>>
tracert_ports_multi_by_src_ip(std::string srcDevIp, std::vector<std::string> targets, std::vector<int> port_num)
{
    return tracert_ports_multi_by_src_ip_with_sport_begin(
        std::move(srcDevIp), std::move(targets), std::move(port_num), DEFAULT_SRC_PORT);
}

std::vector<std::vector<std::vector<std::string>>> tracert_ports_multi_by_src_ip_with_sport_begin(
    std::string srcDevIp, std::vector<std::string> targets, std::vector<int> port_num, int sport_begin)
{
    std::vector<std::pair<std::string, int>> target_port_list;
    std::vector<std::pair<size_t, int>> target_port_index;
    std::vector<std::vector<std::vector<std::string>>> res(targets.size());
    for (size_t i = 0; i < targets.size(); i++) {
        res[i].resize(port_num[i]);
        for (int j = 0; j < port_num[i]; j++) {
            if (targets[i] == srcDevIp) {
                res[i][j] = {targets[i]};
            } else {
                target_port_list.emplace_back(targets[i], sport_begin + j);
                target_port_index.emplace_back(i, j);
            }
        }
    }

    if (target_port_list.empty()) {
        return res;
    }

    Tracert tracer(find_hccn_dev_id(srcDevIp));
    std::vector<std::vector<std::string>> flat_results = tracer.trace(target_port_list);

    for (size_t i = 0; i < flat_results.size(); i++) {
        auto [target_index, port_index] = target_port_index[i];
        res[target_index][port_index] = flat_results[i];
    }
    return res;
}

std::string hccn_lldp_mgmt_ip(std::string device_id)
{
    std::ostringstream cmd;
    cmd << "hccn_tool -i " << require_local_device_id(device_id) << " -lldp -g";
    std::istringstream lines(run_hccn_tool_command(cmd.str()));
    std::string line;
    while (std::getline(lines, line)) {
        line = trim_copy(line);
        const std::string prefix = "IPv4:";
        if (line.rfind(prefix, 0) == 0) {
            std::string ip = trim_copy(line.substr(prefix.size()));
            if (looks_like_ipv4(ip)) {
                return ip;
            }
        }
    }
    return "";
}

std::vector<std::string> hccn_device_ip_list(int device_count)
{
    if (device_count < 0) {
        throw std::runtime_error("[hccn_ip] device_count must be non-negative");
    }

    std::vector<std::string> device_ips;
    device_ips.reserve(static_cast<size_t>(device_count));
    for (int device_id = 0; device_id < device_count; device_id++) {
        std::ostringstream cmd;
        cmd << "hccn_tool -i " << device_id << " -ip -g";
        std::istringstream lines(run_hccn_tool_command(cmd.str()));
        std::string line;
        std::string device_ip;
        while (std::getline(lines, line)) {
            line = trim_copy(line);
            const std::string prefix = "ipaddr:";
            if (line.rfind(prefix, 0) != 0) {
                continue;
            }

            std::string ip = trim_copy(line.substr(prefix.size()));
            if (looks_like_ipv4(ip)) {
                device_ip = std::move(ip);
                break;
            }
        }

        if (device_ip.empty()) {
            throw std::runtime_error("[hccn_ip] cannot resolve device ip for devId=" + std::to_string(device_id));
        }
        device_ips.emplace_back(std::move(device_ip));
    }
    return device_ips;
}
