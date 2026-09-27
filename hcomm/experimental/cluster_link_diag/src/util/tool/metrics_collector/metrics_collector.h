/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METRICS_COLLECTOR_H
#define METRICS_COLLECTOR_H

#include <iostream>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <chrono>
#include <mutex>
#include <cstdio>
#include <sstream>
#include <regex>
#include <atomic>
#include <utility>

namespace metrics_collector_detail {
struct CounterDelta {
    uint64_t value = 0;
    bool reset_or_wrap = false;
};

std::map<std::string, uint64_t> parse_hccn_stat_output(const std::string& output);
// Row layout: dev_id, four counter values, valid (1=true, 0=false).
std::vector<uint64_t> collect_counter_values(int device_id, const std::map<std::string, uint64_t>& counters);
std::vector<int> normalize_device_ids(std::vector<int> device_ids);
CounterDelta counter_delta(uint64_t previous, uint64_t current);
} // namespace metrics_collector_detail

void metrics_collector_thread();
void metrics_collector_thread(std::vector<int> device_ids);
std::vector<std::string> get_metrics_counter_name();
std::vector<std::vector<uint64_t>> get_metrics_counter_value();

#endif // METRICS_COLLECTOR_H
