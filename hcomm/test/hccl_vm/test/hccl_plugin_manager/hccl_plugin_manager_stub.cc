/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_plugin_manager_stub.h"

#include <iostream>
#include <climits>
#include <unistd.h>

#include "hccl_plugin_stub.h"
#include "sim_log.h"

using namespace HcclSim;

HcclPluginManager& HcclPluginManager::GetInstance()
{
    static HcclPluginManager instance;
    return instance;
}

HcclPluginManager::HcclPluginManager()
{
    m_monitorThreadStop = false;
    m_monitorThread = std::thread([this]() {
        this->MonitorThread();
    });
}

HcclPluginManager::~HcclPluginManager()
{
    m_monitorThreadStop = true;
    StopAllPlugins();
    if (m_monitorThread.joinable()) {
        m_monitorThread.join();
    }
}

void HcclPluginManager::MonitorThread()
{
    while (!m_monitorThreadStop) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

HcclVmResult ExitRunnerPlugin() { return HcclVmResult::HCCL_SIM_SUCCESS; }

std::vector<HcclVmResult> HcclPluginManager::StopPlugins(const std::vector<std::string>& tags)
{
    std::vector<HcclVmResult> results;
    std::lock_guard<std::mutex> lock(m_mutex);

    for (const auto& tag : tags) {
        if (tag == "runner") {
            results.push_back(ExitRunnerPlugin());
            continue;
        }

        auto it = m_plugins.find(tag);
        if (it == m_plugins.end()) {
            results.push_back(HcclVmResult::HCCL_SIM_E_NOT_FOUND);
            continue;
        }

        HcclVmResult res = it->second->Stop();
        results.push_back(res);
        m_plugins.erase(it);
    }
    return results;
}

HcclVmResult HcclPluginManager::StopAllPlugins()
{
    std::vector<std::string> allTags;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& pair : m_plugins) {
            allTags.push_back(pair.first);
        }
    }

    if (allTags.empty()) {
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    StopPlugins(allTags);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

std::vector<std::string> HcclPluginManager::GetPluginStatus() const
{
    std::vector<std::string> result;
    std::lock_guard<std::mutex> lock(m_mutex);
    return result;
}

HcclVmResult HcclPluginManager::RegisterPlugin(const std::string& pluginTag)
{
    return HcclVmResult::HCCL_SIM_E_NOT_FOUND;
}

std::vector<HcclVmResult> HcclPluginManager::StartPlugins(const std::vector<std::string>& tags)
{
    std::vector<HcclVmResult> results;
    for (const auto& tag : tags) {
        results.push_back(HcclVmResult::HCCL_SIM_E_NOT_FOUND);
    }
    return results;
}
