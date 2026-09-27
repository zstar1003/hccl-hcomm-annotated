/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_loader.h"

#include <cstdint>
#include <filesystem>

#include "sim_common_defs.h"
#include "sim_log.h"
#include "sim_common_api.h"
#include "sim_yaml_config.h"
#include "db_sim_op_db_ops.h"

namespace loader {
Loader::Loader() {}

Loader::~Loader() {}

HcclResult Loader::LoadOpTaskFile(const std::string dbPath)
{
    sim::DBConfig config;
    std::string targetPath;

    if (!dbPath.empty()) {
        if (!std::filesystem::exists(dbPath)) {
            HCCL_VM_ERROR("Backup database file not found: {}", dbPath);
            return HcclResult::HCCL_E_PARA;
        }
        targetPath = dbPath;
        HCCL_VM_INFO("Loading from specific backup path: {}", targetPath);
    } else {
        targetPath = InstallPath::ResolveToInstallRoot("data/hccl_vm_data.db");
        std::string absPath = std::filesystem::absolute(targetPath).string();
        HCCL_VM_INFO("Loading using default configuration path: {}", absPath);
    }

    config.dbPath = targetPath;
    sim::SetDbConfig(config);
    HCCL_VM_INFO("Loading using default configuration path:{}.", config.dbPath);
    return HcclResult::HCCL_SUCCESS;
}

// 返回内部已组装好的完整 Pipeline 缓存
const OpPipeline& Loader::GetOpTasks() const { return opTaskCache_; }

// Runner: 每 sync 一次调用一次，只加载下一条 pending(status=0) 记录，根据业务传入的 syncIter，从 pending(status=0)
// 记录中查找并加载对应算子数据
HcclResult Loader::LoadRunnerSingleSync(
    const uint32_t& outSyncIter, std::map<uint32_t, std::vector<sim::CompositeOpDetail>>& compositeDataMap)
{
    if (sim::QueryCompositeOpDetailBySyncIter(outSyncIter, compositeDataMap) != 0) {
        HCCL_VM_ERROR("QueryCompositeOpDetailBySyncIter failed for syncIter: {}", outSyncIter);
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("syncIter={} loaded. Rank Count: {}", outSyncIter, compositeDataMap.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::GetCcuChannelInfo(std::vector<sim::CcuChannelTab>& channels)
{
    if (sim::QueryCcuChannelAll(channels) != 0) {
        HCCL_VM_ERROR("QueryCcuChannelAll failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("Get {} channels", channels.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::GetJettyMapInfo(std::vector<sim::JettyMapTab>& jettyMaps)
{
    if (sim::QueryJettyMapAll(jettyMaps) != 0) {
        HCCL_VM_ERROR("QueryJettyMapAll failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("Get {} jetty maps", jettyMaps.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::GetInstrResInfo(std::vector<sim::CcuInstrResTab>& instrRes)
{
    if (sim::QueryCcuInstrResAll(instrRes) != 0) {
        HCCL_VM_ERROR("QueryCcuInstrResAll failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("Get {} instr res records", instrRes.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::GetSyncInfo(std::vector<sim::SyncRecordTab>& syncRecords)
{
    if (sim::QuerySyncRecordAll(syncRecords) != 0) {
        HCCL_VM_ERROR("QuerySyncRecordAll failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("Get {} sync records", syncRecords.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::GetSyncRecordsByStatus(uint8_t status, std::vector<sim::SyncRecordTab>& syncRecords)
{
    if (sim::QuerySyncRecordByStatus(status, syncRecords) != 0) {
        HCCL_VM_ERROR("QuerySyncRecordByStatus failed for status: {}", status);
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("Get {} sync records for status: {}", syncRecords.size(), status);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::LoadCompositeOpDetailBySyncIter(
    uint32_t syncIter, std::map<uint32_t, std::vector<sim::CompositeOpDetail>>& compositeDataMap)
{
    if (sim::QueryCompositeOpDetailBySyncIter(syncIter, compositeDataMap) != 0) {
        HCCL_VM_ERROR("QueryCompositeOpDetailsBySyncIter failed for syncIter: {}", syncIter);
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("syncIter={}, Rank Count: {}", syncIter, compositeDataMap.size());
    return HcclResult::HCCL_SUCCESS;
}
} // namespace loader
