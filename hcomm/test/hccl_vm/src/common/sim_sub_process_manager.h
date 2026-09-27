/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_SUB_PROCESS_MANAGER_H
#define SIM_SUB_PROCESS_MANAGER_H

#include <sys/types.h>
#include <cstdint>
#include <mutex>
#include <map>
#include <string>
#include <vector>
#include "sim_aicpu_pipe_msg.h"

namespace sim {

struct SubProcessConfig {
    std::string executable;
    std::vector<std::string> args;
    std::map<std::string, std::string> envVars;
    std::string workDir;
};

class SubProcessManager {
public:
    SubProcessManager() = default;
    ~SubProcessManager();

    SubProcessManager(const SubProcessManager&) = delete;
    SubProcessManager& operator=(const SubProcessManager&) = delete;

    int CreateProcess(const SubProcessConfig& config);
    int DestroyProcess(int timeoutMs = 3000);

    bool IsAlive() const;
    pid_t GetPid() const { return m_pid; }

    int Request(
        uint8_t reqCmd, const void* reqData, uint32_t reqLen, uint8_t& rspCmd, void* rspData, uint32_t rspMaxLen,
        uint32_t& rspLen);

private:
    static constexpr int kH2dReadFd = 200;
    static constexpr int kD2hWriteFd = 201;

    int m_h2dWriteFd = -1;
    int m_d2hReadFd = -1;
    pid_t m_pid = -1;

    mutable std::mutex m_rpcLock;
    mutable std::mutex m_forkLock;

    int WaitForReady();

    int HostSendMsg(uint8_t cmd, const void* data, uint32_t len);
    int HostRecvMsg(uint8_t& outCmd, void* outData, uint32_t maxLen, uint32_t& outLen);
};

SubProcessConfig CreateAicpuDeviceConfig(uint32_t rankId, uint32_t deviceKey = 0);
SubProcessManager& GetAicpuProcMgr();
} // namespace sim

#endif
