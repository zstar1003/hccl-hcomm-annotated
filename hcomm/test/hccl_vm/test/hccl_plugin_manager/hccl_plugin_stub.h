/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_PLUGIN_STUB_H
#define HCCL_PLUGIN_STUB_H

#include <string>
#include <vector>

#include "sim_common_defs.h"

class HcclPlugin {
public:
    static const int MAX_SCAN_DEPTH;
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

    HcclPlugin(const std::string& pluginPath);
    ~HcclPlugin();
    HcclPlugin(const HcclPlugin&) = delete;
    HcclPlugin& operator=(const HcclPlugin&) = delete;

    HcclSim::HcclVmResult Start();
    HcclSim::HcclVmResult Stop();
    int32_t GetPid() const;
    int32_t GetStdinFd() const;
    bool IsRunning() const;
    std::string GetTag() const;

private:
    std::string m_pluginPath;
    int32_t m_pid = -1;
    int32_t m_stdinFd = -1;
};

#endif
