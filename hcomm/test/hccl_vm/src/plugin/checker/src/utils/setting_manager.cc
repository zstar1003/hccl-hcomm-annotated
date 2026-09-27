/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "setting_manager.h"

#include <fstream>

#include <nlohmann_json/json.hpp>

#include "file_utils.h"
#include "sim_log.h"

namespace {

static const std::string MANIFEST_FILE_NAME = "manifest.json";
static const std::string SETTING_KEY_ENABLE_INSIGHT_DUMP = "enable_insight_dump";
static const std::string SETTING_KEY_ENABLE_MEMORY_SNAPSHOT_DUMP = "enable_memory_snapshot_dump";
static const std::string SETTING_KEY_ENABLE_NEW_CHECKER = "enable_new_checker";
static const std::string SETTING_KEY_ENABLE_OLD_CHECKER = "enable_old_checker";
static const std::string SETTING_KEY_ENABLE_BIG_GRAPH_CHECKER = "enable_big_graph_checker";

} // namespace

namespace HcclSim {

HcclResult SettingManager::Refresh()
{
    const std::string pluginRootDir = GetCurrentPath();
    if (pluginRootDir.empty()) {
        HCCL_VM_ERROR("failed to get current path.");
        return HcclResult::HCCL_E_INTERNAL;
    }

    const std::string manifestPath = JoinPath(pluginRootDir, MANIFEST_FILE_NAME);
    CheckerSettings newSettings;

    if (!FileExists(manifestPath)) {
        HCCL_VM_WARN("manifest.json not found, use default settings.");
        std::lock_guard<std::mutex> lock(m_mutex);
        m_settings = newSettings;
        m_pluginRootDir = pluginRootDir;
        m_manifestPath = manifestPath;
        return HcclResult::HCCL_SUCCESS;
    }

    std::ifstream manifestStream(manifestPath.c_str());
    if (!manifestStream.is_open()) {
        HCCL_VM_ERROR("failed to open manifest file: {}", manifestPath);
        return HcclResult::HCCL_E_INTERNAL;
    }

    try {
        nlohmann::json manifest;
        manifestStream >> manifest;
        const nlohmann::json settings = manifest.value("setting", nlohmann::json::object());
        newSettings.enableInsightDump = settings.value(SETTING_KEY_ENABLE_INSIGHT_DUMP, false);
        newSettings.enableMemorySnapshotDump = settings.value(SETTING_KEY_ENABLE_MEMORY_SNAPSHOT_DUMP, true);
        newSettings.enableNewChecker = settings.value(SETTING_KEY_ENABLE_NEW_CHECKER, true);
        newSettings.enableOldChecker = settings.value(SETTING_KEY_ENABLE_OLD_CHECKER, true);
        newSettings.enableBigGraphChecker = settings.value(SETTING_KEY_ENABLE_BIG_GRAPH_CHECKER, false);
    } catch (const std::exception& ex) {
        HCCL_VM_ERROR("parse manifest failed: {}", ex.what());
        return HcclResult::HCCL_E_INTERNAL;
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_settings = newSettings;
        m_pluginRootDir = pluginRootDir;
        m_manifestPath = manifestPath;
    }

    HCCL_VM_INFO(
        "settings refreshed: insight_dump={}, memory_snapshot_dump={}, "
        "new_checker={}, old_checker={}, big_graph_checker={}",
        newSettings.enableInsightDump, newSettings.enableMemorySnapshotDump, newSettings.enableNewChecker,
        newSettings.enableOldChecker, newSettings.enableBigGraphChecker);
    return HcclResult::HCCL_SUCCESS;
}

void SettingManager::Reset()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_settings = CheckerSettings{};
    m_pluginRootDir.clear();
    m_manifestPath.clear();
}

CheckerSettings SettingManager::GetSettings() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_settings;
}

bool SettingManager::IsInsightDumpEnabled() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_settings.enableInsightDump;
}

bool SettingManager::IsMemorySnapshotEnabled() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_settings.enableInsightDump && m_settings.enableMemorySnapshotDump;
}

bool SettingManager::IsNewCheckerEnabled() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_settings.enableNewChecker;
}

bool SettingManager::IsOldCheckerEnabled() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_settings.enableOldChecker;
}

bool SettingManager::IsBigGraphCheckerEnabled() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_settings.enableBigGraphChecker;
}

} // namespace HcclSim
