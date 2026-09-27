/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dump/dump_manager.h"
#include <chrono>
#include <cstdint>
#include <fstream>

#include "file_utils.h"
#include "sim_log.h"
#include "setting_manager.h"

static const std::string DATA_DIR_NAME = "data";
static const std::string INSIGHT_DIR_NAME = "insight";

namespace HcclSim {
HcclResult DumpManager::Initialize(const std::string& dataId)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (dataId.empty()) {
        HCCL_VM_ERROR("dataId is empty.");
        return HcclResult::HCCL_E_PARA;
    }

    m_dataId = dataId;
    m_pluginRootDir = HcclSim::GetCurrentPath();
    if (m_pluginRootDir.empty()) {
        HCCL_VM_ERROR("failed to get current path.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    m_dataDir = HcclSim::JoinPath(m_pluginRootDir, DATA_DIR_NAME);
    m_dumpRootDir = HcclSim::JoinPath(HcclSim::JoinPath(m_dataDir, INSIGHT_DIR_NAME), m_dataId);

    if (!SettingManager::GetInstance().IsInsightDumpEnabled()) {
        HCCL_VM_INFO("insight dump disabled by manifest setting.");
        return HcclResult::HCCL_SUCCESS;
    }
    return PrepareDumpDirs();
}

void DumpManager::Reset()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_dataId.clear();
    m_pluginRootDir.clear();
    m_dataDir.clear();
    m_dumpRootDir.clear();
}

bool DumpManager::IsEnabled() const { return SettingManager::GetInstance().IsInsightDumpEnabled(); }

bool DumpManager::IsMemorySnapshotEnabled() const { return SettingManager::GetInstance().IsMemorySnapshotEnabled(); }

std::string DumpManager::GetDumpRootDir() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_dumpRootDir;
}

HcclResult DumpManager::Write(const std::string& relativePath, const nlohmann::json& document) const
{
    std::string dumpRootDir;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        dumpRootDir = m_dumpRootDir;
    }
    if (!SettingManager::GetInstance().IsInsightDumpEnabled()) {
        return HcclResult::HCCL_SUCCESS;
    }
    return WriteMsgpackFile(dumpRootDir, relativePath, document);
}

HcclResult DumpManager::WriteJson(const std::string& relativePath, const nlohmann::json& document) const
{
    std::string dumpRootDir;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        dumpRootDir = m_dumpRootDir;
    }
    if (!SettingManager::GetInstance().IsInsightDumpEnabled()) {
        return HcclResult::HCCL_SUCCESS;
    }
    return WriteJsonFile(dumpRootDir, relativePath, document);
}

HcclResult DumpManager::WriteMsgpackStream(
    const std::string& relativePath, const std::function<HcclResult(std::ostream&)>& writer) const
{
    std::string dumpRootDir;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        dumpRootDir = m_dumpRootDir;
    }
    if (!SettingManager::GetInstance().IsInsightDumpEnabled()) {
        return HcclResult::HCCL_SUCCESS;
    }
    return WriteMsgpackStreamFile(dumpRootDir, relativePath, writer);
}

HcclResult DumpManager::PrepareDumpDirs() const
{
    auto ret = HcclSim::EnsureDirectory(m_dataDir);
    if (ret != HcclResult::HCCL_SUCCESS) {
        return ret;
    }
    ret = HcclSim::EnsureDirectory(HcclSim::JoinPath(m_dataDir, INSIGHT_DIR_NAME));
    if (ret != HcclResult::HCCL_SUCCESS) {
        return ret;
    }
    return HcclSim::EnsureDirectory(m_dumpRootDir);
}

HcclResult DumpManager::WriteMsgpackFile(
    const std::string& dumpRootDir, const std::string& relativePath, const nlohmann::json& document) const
{
    auto NowMs = []() -> uint64_t {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
    };

    const std::string fullPath = HcclSim::JoinPath(dumpRootDir, relativePath);
    const uint64_t ensureDirBeginMs = NowMs();
    auto ret = HcclSim::EnsureDirectory(HcclSim::GetParentPath(fullPath));
    const uint64_t ensureDirCostMs = NowMs() - ensureDirBeginMs;
    if (ret != HcclResult::HCCL_SUCCESS) {
        return ret;
    }

    std::ofstream out(fullPath.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
    if (!out.is_open()) {
        HCCL_VM_ERROR("failed to open file: {}", fullPath);
        return HcclResult::HCCL_E_INTERNAL;
    }

    const uint64_t serializeBeginMs = NowMs();
    try {
        nlohmann::json::to_msgpack(document, nlohmann::detail::output_adapter<char>(out));
    } catch (const std::exception& ex) {
        HCCL_VM_ERROR("stream msgpack serialize failed: {}, file: {}", ex.what(), fullPath);
        return HcclResult::HCCL_E_INTERNAL;
    }
    const uint64_t serializeCostMs = NowMs() - serializeBeginMs;

    const uint64_t writeBeginMs = NowMs();
    out.flush();
    if (!out.good()) {
        HCCL_VM_ERROR("failed to flush file: {}", fullPath);
        return HcclResult::HCCL_E_INTERNAL;
    }
    const uint64_t writeCostMs = NowMs() - writeBeginMs;
    const std::streampos finalPos = out.tellp();
    const uint64_t msgpackBytes = (finalPos >= static_cast<std::streampos>(0)) ? static_cast<uint64_t>(finalPos) : 0;

    if (relativePath.compare(0, std::string("memory/").size(), "memory/") == 0) {
        HCCL_VM_INFO(
            "path={}, ensure_dir_ms={}, serialize_ms={}, "
            "write_file_ms={}, msgpack_bytes={}",
            relativePath, ensureDirCostMs, serializeCostMs, writeCostMs, msgpackBytes);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult DumpManager::WriteJsonFile(
    const std::string& dumpRootDir, const std::string& relativePath, const nlohmann::json& document) const
{
    const std::string fullPath = HcclSim::JoinPath(dumpRootDir, relativePath);
    auto ret = HcclSim::EnsureDirectory(HcclSim::GetParentPath(fullPath));
    if (ret != HcclResult::HCCL_SUCCESS) {
        return ret;
    }

    std::ofstream out(fullPath.c_str(), std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        HCCL_VM_ERROR("failed to open file: {}", fullPath);
        return HcclResult::HCCL_E_INTERNAL;
    }

    try {
        out << document.dump(2) << std::endl;
    } catch (const std::exception& ex) {
        HCCL_VM_ERROR("json serialize failed: {}, file: {}", ex.what(), fullPath);
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (!out.good()) {
        HCCL_VM_ERROR("failed to write file: {}", fullPath);
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult DumpManager::WriteMsgpackStreamFile(
    const std::string& dumpRootDir, const std::string& relativePath,
    const std::function<HcclResult(std::ostream&)>& writer) const
{
    auto NowMs = []() -> uint64_t {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
    };

    const std::string fullPath = HcclSim::JoinPath(dumpRootDir, relativePath);
    const uint64_t ensureDirBeginMs = NowMs();
    auto ret = HcclSim::EnsureDirectory(HcclSim::GetParentPath(fullPath));
    const uint64_t ensureDirCostMs = NowMs() - ensureDirBeginMs;
    if (ret != HcclResult::HCCL_SUCCESS) {
        return ret;
    }

    std::ofstream out(fullPath.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
    if (!out.is_open()) {
        HCCL_VM_ERROR("failed to open file: {}", fullPath);
        return HcclResult::HCCL_E_INTERNAL;
    }

    const uint64_t serializeBeginMs = NowMs();
    ret = writer(out);
    const uint64_t serializeCostMs = NowMs() - serializeBeginMs;
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("custom writer failed. path: {}", fullPath);
        return ret;
    }

    const uint64_t writeBeginMs = NowMs();
    out.flush();
    if (!out.good()) {
        HCCL_VM_ERROR("failed to flush file: {}", fullPath);
        return HcclResult::HCCL_E_INTERNAL;
    }
    const uint64_t writeCostMs = NowMs() - writeBeginMs;
    const std::streampos finalPos = out.tellp();
    const uint64_t msgpackBytes = (finalPos >= static_cast<std::streampos>(0)) ? static_cast<uint64_t>(finalPos) : 0;

    if (relativePath.compare(0, std::string("memory/").size(), "memory/") == 0) {
        HCCL_VM_INFO(
            "path={}, ensure_dir_ms={}, serialize_ms={}, "
            "write_file_ms={}, msgpack_bytes={}",
            relativePath, ensureDirCostMs, serializeCostMs, writeCostMs, msgpackBytes);
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
