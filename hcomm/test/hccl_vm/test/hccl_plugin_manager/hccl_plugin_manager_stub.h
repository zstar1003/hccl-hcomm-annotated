/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_PLUGIN_MANAGER_STUB_H
#define HCCL_PLUGIN_MANAGER_STUB_H

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "sim_common_defs.h"

class HcclPlugin;

class HcclPluginManager {
public:
    static HcclPluginManager& GetInstance();
    HcclSim::HcclVmResult RegisterPlugin(const std::string& pluginTag);
    std::vector<std::string> GetPluginStatus() const;
    std::vector<HcclSim::HcclVmResult> StartPlugins(const std::vector<std::string>& tag);
    std::vector<HcclSim::HcclVmResult> StopPlugins(const std::vector<std::string>& tag);
    HcclSim::HcclVmResult StopAllPlugins();

    HcclPluginManager(const HcclPluginManager&) = delete;
    HcclPluginManager& operator=(const HcclPluginManager&) = delete;

private:
    HcclPluginManager();
    ~HcclPluginManager();
    void MonitorThread();
    std::atomic<bool> m_monitorThreadStop{false};
    std::thread m_monitorThread;
    std::map<std::string, std::shared_ptr<HcclPlugin>> m_plugins;
    mutable std::mutex m_mutex;
};

#endif
