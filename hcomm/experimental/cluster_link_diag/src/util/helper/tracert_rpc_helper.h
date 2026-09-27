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

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

using TracertPathResult = std::vector<std::vector<std::vector<std::vector<std::string>>>>;

inline int LogTracertBatchPlan(
    int batchId, int sportBegin, const std::string& sportLabel, const std::vector<std::string>& deviceList,
    const std::vector<std::string>& srcDevIpList, const std::vector<std::vector<std::string>>& dstList,
    const std::vector<std::vector<int>>& portNum)
{
    int totalPortProbe = 0;
    std::cout << "[tracert][batch " << batchId << "] plan: source_count=" << deviceList.size() << ", " << sportLabel
              << "=" << sportBegin << std::endl;
    for (size_t i = 0; i < deviceList.size(); i++) {
        const std::string srcDevIp = i < srcDevIpList.size() ? srcDevIpList[i] : "";
        const size_t targetCount = i < dstList.size() ? dstList[i].size() : 0;
        std::cout << "[tracert][batch " << batchId << "] source " << (i + 1) << "/" << deviceList.size()
                  << ": control_dev=" << deviceList[i] << ", src_ip=" << srcDevIp << ", target_count=" << targetCount
                  << std::endl;

        if (i >= dstList.size()) {
            continue;
        }
        for (size_t j = 0; j < dstList[i].size(); j++) {
            int ports = 0;
            if (i < portNum.size() && j < portNum[i].size()) {
                ports = portNum[i][j];
            }
            totalPortProbe += ports;

            std::cout << "  -> target=" << dstList[i][j] << ", sport=";
            if (ports > 0) {
                const int sportEnd = sportBegin + ports - 1;
                std::cout << sportBegin;
                if (sportEnd != sportBegin) {
                    std::cout << "-" << sportEnd;
                }
            } else {
                std::cout << "none";
            }
            std::cout << ", count=" << ports << std::endl;
        }
    }
    return totalPortProbe;
}

inline std::pair<int, int> NormalizeTracertPaths(TracertPathResult& res)
{
    int resultPathCount = 0;
    int emptyPathCount = 0;
    for (auto& v1 : res) {
        for (auto& v2 : v1) {
            for (auto& v3 : v2) {
                resultPathCount++;
                if (v3.empty()) {
                    emptyPathCount++;
                }
                if (v3.size() == 1) {
                    v3 = {"", v3[0]};
                }
            }
        }
    }
    return {resultPathCount, emptyPathCount};
}

inline long long
GetElapsedMs(std::chrono::steady_clock::time_point startTime, std::chrono::steady_clock::time_point endTime)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
}
