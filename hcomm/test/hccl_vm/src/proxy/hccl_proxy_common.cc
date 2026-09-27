/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "PROXY_STUB"

#include <cstdint>
#include "hccl_proxy_common.h"
#include "sim_log.h"
#include "db_sim_runner_db.h"
#include <fstream>
#include <nlohmann_json/json.hpp>

namespace sim {
bool IsDeviceAddress(void* addr)
{
    uint64_t devPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(addr));
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([devPtr](const sim::VirtualMemBlock& virMem) {
        return (
            (virMem.dev_mapped_ptr <= devPtr) && (devPtr < (virMem.dev_mapped_ptr + virMem.size))
            && (virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_DEV));
    });
    if (!virMemRes.second) {
        HCCL_VM_INFO("this buff offset ptr not found:{:p} in VirtualMemBlock.", addr);
        return false;
    }

    return true;
}

bool ParseKernelJson(const std::string& jsonPath, std::map<std::string, std::string>& out)
{
    // 校验文件后缀是否为.json
    std::string suffix = ".json";
    if (jsonPath.size() < suffix.size() || jsonPath.substr(jsonPath.size() - suffix.size()) != suffix) {
        // 也会加载AIV的.o文件
        HCCL_VM_WARN("path is not json file: {}", jsonPath);
        return true;
    }

    std::ifstream ifs(jsonPath);
    if (!ifs.is_open()) {
        HCCL_VM_ERROR("open failed: {}", jsonPath);
        return false;
    }

    auto j = nlohmann::json::parse(ifs, nullptr, false);
    if (j.is_discarded()) {
        HCCL_VM_ERROR("parse failed: {}", jsonPath);
        return false;
    }

    for (auto& [key, val] : j.items()) {
        if (!val.contains("opInfo") || !val["opInfo"].contains("functionName") || !val["opInfo"].contains("kernelSo")) {
            HCCL_VM_ERROR("skip entry '{}', missing required fields.", key);
            continue;
        }

        std::string funcName = val["opInfo"]["functionName"].get<std::string>();
        std::string kernelSo = val["opInfo"]["kernelSo"].get<std::string>();
        out[funcName] = kernelSo;
    }

    HCCL_VM_INFO("loaded {} kernels from {}", out.size(), jsonPath);
    return true;
}

} // namespace sim
