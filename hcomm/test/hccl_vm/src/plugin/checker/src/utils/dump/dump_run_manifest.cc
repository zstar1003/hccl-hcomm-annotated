/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "dump/dump_manager.h"
#include "dump/dump_run_manifest.h"

namespace HcclSim {
static const std::string RUN_MANIFEST_PATH = "manifest.json";
static const std::string INPUT_GRAPH_STAGE = "input_graph";
static const std::string ORIGIN_GRAPH_STAGE = "origin_graph";
static const std::string REVAMP_GRAPH_STAGE = "revamp_graph";

void DumpRunManifest::Reset(const std::string& dataId)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_dataId = dataId;
    m_opParam = nullptr;
    m_graphStages = nlohmann::json::object();
    m_graphStages[INPUT_GRAPH_STAGE] = nullptr;
    m_graphStages[ORIGIN_GRAPH_STAGE] = nullptr;
    m_graphStages[REVAMP_GRAPH_STAGE] = nullptr;

    m_memoryStats = nullptr;
    m_errorCount = nullptr;
    m_retCode = static_cast<uint32_t>(HcclResult::HCCL_SUCCESS);
}

void DumpRunManifest::SetOpParam(const nlohmann::json& opParam)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_opParam = opParam;
}

void DumpRunManifest::SetGraphStageStats(const std::string& stage, const nlohmann::json& stageStats)
{
    if (stage.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    m_graphStages[stage] = stageStats;
}

void DumpRunManifest::SetMemorySnapshotStats(const nlohmann::json& memoryStats)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_memoryStats = memoryStats;
}

void DumpRunManifest::SetErrorCount(size_t errorCount)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_errorCount = errorCount;
}

void DumpRunManifest::SetCheckResult(HcclResult retCode)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_retCode = static_cast<uint32_t>(retCode);
}

HcclResult DumpRunManifest::Flush() const
{
    DumpManager& dumpManager = DumpManager::GetInstance();
    if (!dumpManager.IsEnabled()) {
        return HcclResult::HCCL_SUCCESS;
    }

    nlohmann::json doc = nlohmann::json::object();
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        doc["data_id"] = m_dataId;
        doc["op_param"] = m_opParam;
        doc["graph_stage_stats"] = m_graphStages;
        doc["memory_snapshot_stats"] = m_memoryStats;
        doc["error_count"] = m_errorCount;
        doc["ret_code"] = m_retCode;
        doc["success"] = (m_retCode == static_cast<uint32_t>(HcclResult::HCCL_SUCCESS));
    }

    return dumpManager.WriteJson(RUN_MANIFEST_PATH, doc);
}
} // namespace HcclSim
