/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VM_DUMP_RUN_MANIFEST_H
#define HCCL_VM_DUMP_RUN_MANIFEST_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <nlohmann_json/json.hpp>
#include <string>

#include "hccl_types.h"

namespace HcclSim {
class DumpRunManifest {
public:
    static DumpRunManifest& GetInstance()
    {
        static DumpRunManifest instance;
        return instance;
    }

    DumpRunManifest(const DumpRunManifest&) = delete;
    DumpRunManifest& operator=(const DumpRunManifest&) = delete;

    void Reset(const std::string& dataId);
    void SetOpParam(const nlohmann::json& opParam);
    void SetGraphStageStats(const std::string& stage, const nlohmann::json& stageStats);
    void SetMemorySnapshotStats(const nlohmann::json& memoryStats);
    void SetErrorCount(size_t errorCount);
    void SetCheckResult(HcclResult retCode);
    HcclResult Flush() const;

private:
    DumpRunManifest() = default;

    mutable std::mutex m_mutex;
    std::string m_dataId;
    nlohmann::json m_opParam = nullptr;
    nlohmann::json m_graphStages = nlohmann::json::object();
    nlohmann::json m_memoryStats = nullptr;
    nlohmann::json m_errorCount = nullptr;
    uint32_t m_retCode = static_cast<uint32_t>(HcclResult::HCCL_SUCCESS);
};
} // namespace HcclSim

#endif // HCCL_VM_DUMP_RUN_MANIFEST_H
