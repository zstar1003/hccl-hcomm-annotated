/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_PLUGIN_MANAGER_H
#define SIM_PLUGIN_MANAGER_H

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "sim_common_defs.h"
#include "sim_plugin.h"

class HcclPluginManager {
public:
    static HcclPluginManager& GetInstance();
    // 注册插件 同时通过LoadPlugins加载
    HcclSim::HcclVmResult RegisterPlugin(const std::string& pluginTag);

    HcclSim::HcclVmResult SendMessageToPlugin(
        const std::string& pluginTag, const std::string& action,
        const nlohmann::json& payload = nlohmann::json::object());
    // 广播给所有运行中的插件
    HcclSim::HcclVmResult
    BroadcastToAllPlugin(const std::string& action, const nlohmann::json& payload = nlohmann::json::object());

    // 获取已加载的插件状态
    std::vector<std::string> GetPluginStatus() const;

    // 运行插件
    std::vector<HcclSim::HcclVmResult> StartPlugins(const std::vector<std::string>& tag);
    // 关闭并卸载插件
    std::vector<HcclSim::HcclVmResult> StopPlugins(const std::vector<std::string>& tag);

    HcclSim::HcclVmResult StopAllPlugins();

    // 禁止拷贝
    HcclPluginManager(const HcclPluginManager&) = delete;
    HcclPluginManager& operator=(const HcclPluginManager&) = delete;

private:
    HcclPluginManager();
    ~HcclPluginManager();

    void MonitorThread();
    std::atomic<bool> m_monitorThreadStop{false};
    std::thread m_monitorThread;

    bool GetPluginFolderPath(const std::string& pluginTag, std::string& pluginFolderPath);
    bool IsMatchingPlugin(const std::string& manifestPath, const std::string& targetTag);
    std::map<std::string, std::shared_ptr<HcclPlugin>> m_plugins; // 插件Tag - 插件

    mutable std::mutex m_mutex; // 线程安全锁
};

#endif // SIM_PLUGIN_MANAGER_H
