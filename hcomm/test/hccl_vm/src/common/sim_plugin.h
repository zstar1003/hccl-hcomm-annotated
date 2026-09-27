/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_PLUGIN_H
#define SIM_PLUGIN_H

#include <json.hpp>
#include <mutex>
#include <string>

#include "sim_common_defs.h"

enum class PLUGIN_MESSAGE_TYPE { BROADCAST = 0, COMMAND, MESSAGE };
// 插件基础接口
class HcclPlugin {
public:
    static const int MAX_SCAN_DEPTH; // 最大扫描深度
    static const std::string PLUGIN_PATH;
    static const std::string MANIFEST_FILE;

    struct Manifest {
        static const std::string pluginName;
        static const std::string pluginVersion;
        static const std::string pluginEntry;
        struct pluginDependency {
            static const std::string hostVersion;
        };
    };

    struct PluginMessage {
        static const std::string messageType;
        static const std::string messageAction;
        static const std::string messagePayload;
    };

    HcclPlugin(const std::string& pluginPath);
    ~HcclPlugin();
    // 禁止拷贝，防止文件描述符重复关闭逻辑混乱
    HcclPlugin(const HcclPlugin&) = delete;
    HcclPlugin& operator=(const HcclPlugin&) = delete;

    // 启动/停止插件
    HcclSim::HcclVmResult Start();
    HcclSim::HcclVmResult Stop();
    // 发送命令
    HcclSim::HcclVmResult SendMessage(
        PLUGIN_MESSAGE_TYPE type, const std::string& action, const nlohmann::json& payload = nlohmann::json::object());

    int32_t GetPid() const { return m_pid; }
    int32_t GetStdinFd() const { return m_stdinFd; }
    bool IsRunning() const;
    std::string GetTag() const;

private:
    std::vector<char*> PrepareArgs(const std::string& command);
    std::string m_pluginPath;
    nlohmann::json m_manifest;
    int32_t m_pid = -1;
    int32_t m_stdinFd = -1;
    std::mutex m_mutex;
};

#endif // SIM_PLUGIN_H
