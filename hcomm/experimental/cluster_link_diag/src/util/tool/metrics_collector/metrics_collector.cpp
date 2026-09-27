/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "metrics_collector.h"

#include <array>
#include <deque>
#include <limits>
#include <memory>
#include <set>

namespace {
static const int kMaxSize = 100;
static const int kIntervalMs = 1000;

const std::vector<std::string> kCounterKeys = {
    "dev_id", "mac_tx_pfc_pkt_num", "mac_rx_pfc_pkt_num", "roce_tx_cnp_pkt_num", "roce_rx_cnp_pkt_num",
};

std::string trim(std::string value)
{
    auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

} // namespace

namespace metrics_collector_detail {
std::map<std::string, uint64_t> parse_hccn_stat_output(const std::string& output)
{
    std::map<std::string, uint64_t> counters;
    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        auto pos = line.find(':');
        if (pos == std::string::npos) {
            continue;
        }

        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));
        if (key.empty() || value.empty()) {
            continue;
        }

        try {
            size_t parsed_chars = 0;
            const uint64_t parsed_value = std::stoull(value, &parsed_chars);
            if (parsed_chars != value.size() || value.front() == '-') {
                continue;
            }
            counters[key] = parsed_value;
        } catch (const std::exception&) {
            continue;
        }
    }
    return counters;
}

std::string build_hccn_stat_cmd(int device_id)
{
    std::ostringstream cmd;
    cmd << "hccn_tool -i " << device_id << " -stat -g";
    return cmd.str();
}

std::string run_hccn_stat_command(const std::string& cmd)
{
    std::array<char, 4096> buffer{};
    std::string output;

    FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("[metrics_collector] popen() failed");
    }

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        output += buffer.data();
    }
    const int status = pclose(pipe);
    if (status != 0) {
        throw std::runtime_error("[metrics_collector] hccn_tool command failed");
    }
    return output;
}

std::vector<uint64_t> collect_counter_values(int device_id, const std::map<std::string, uint64_t>& counters)
{
    std::vector<uint64_t> values;
    values.reserve(kCounterKeys.size() + 1);
    values.emplace_back(static_cast<uint64_t>(device_id));
    bool valid = true;
    for (size_t i = 1; i < kCounterKeys.size(); i++) {
        const auto& key = kCounterKeys[i];
        auto it = counters.find(key);
        if (it == counters.end()) {
            valid = false;
        }
        const uint64_t value = it == counters.end() ? 0 : it->second;
        values.emplace_back(value);
    }
    values.emplace_back(valid ? 1U : 0U);
    return values;
}

std::vector<int> normalize_device_ids(std::vector<int> device_ids)
{
    if (device_ids.empty()) {
        device_ids.emplace_back(0);
    }
    std::set<int> seen;
    std::vector<int> normalized;
    for (int device_id : device_ids) {
        if (device_id < 0) {
            continue;
        }
        if (seen.insert(device_id).second) {
            normalized.emplace_back(device_id);
        }
    }
    if (normalized.empty()) {
        normalized.emplace_back(0);
    }
    return normalized;
}

CounterDelta counter_delta(uint64_t previous, uint64_t current)
{
    if (current >= previous) {
        return {current - previous, false};
    }
    return {current, true};
}
} // namespace metrics_collector_detail

std::vector<std::string> metrics_counter_name = kCounterKeys;
std::mutex rw_mutex;
std::deque<std::vector<uint64_t>> metrics_counter_value;

void metrics_collector_thread() { metrics_collector_thread(std::vector<int>{0}); }

void metrics_collector_thread(std::vector<int> device_ids)
{
    device_ids = metrics_collector_detail::normalize_device_ids(std::move(device_ids));

    auto next_time = std::chrono::steady_clock::now();
    while (true) {
        next_time += std::chrono::milliseconds(kIntervalMs);
        for (int device_id : device_ids) {
            std::vector<uint64_t> values(kCounterKeys.size() + 1, 0);
            values[0] = static_cast<uint64_t>(device_id);
            try {
                std::string output = metrics_collector_detail::run_hccn_stat_command(
                    metrics_collector_detail::build_hccn_stat_cmd(device_id));
                auto counters = metrics_collector_detail::parse_hccn_stat_output(output);
                values = metrics_collector_detail::collect_counter_values(device_id, counters);
            } catch (const std::exception&) {
            }

            std::unique_lock<std::mutex> lock(rw_mutex);
            if (metrics_counter_value.size() >= kMaxSize) {
                metrics_counter_value.pop_front();
            }
            metrics_counter_value.emplace_back(std::move(values));
            lock.unlock();
        }

        std::this_thread::sleep_until(next_time);
    }
}

std::vector<std::string> get_metrics_counter_name()
{
    std::unique_lock<std::mutex> lock(rw_mutex);
    return metrics_counter_name;
}

std::vector<std::vector<uint64_t>> get_metrics_counter_value()
{
    std::unique_lock<std::mutex> lock(rw_mutex);
    auto values = std::vector<std::vector<uint64_t>>(metrics_counter_value.begin(), metrics_counter_value.end());
    metrics_counter_value.clear();
    return values;
}
