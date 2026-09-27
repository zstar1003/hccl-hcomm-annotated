/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_sub_process_manager.h"
#include <sys/wait.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <thread>
#include "sim_log.h"
#include "sim_common_defs.h"
#include "sim_common_api.h"
#include "sim_pipe_io.h"

namespace sim {

thread_local static sim::SubProcessManager g_aiCpuProcMgr;

SubProcessManager& GetAicpuProcMgr() { return g_aiCpuProcMgr; }

SubProcessManager::~SubProcessManager()
{
    if (m_pid > 0) {
        DestroyProcess();
    }
}

int SubProcessManager::CreateProcess(const SubProcessConfig& config)
{
    std::lock_guard<std::mutex> lock(m_forkLock);

    if (m_pid > 0) {
        HCCL_VM_INFO("Process {} already exists, skip.", m_pid);
        return 0;
    }

    sim::PipePair h2dPipe{}, d2hPipe{};
    if (sim::PipeCreate(h2dPipe) != 0) {
        HCCL_VM_ERROR("Failed to create h2d pipe");
        return -1;
    }
    if (sim::PipeCreate(d2hPipe) != 0) {
        HCCL_VM_ERROR("Failed to create d2h pipe");
        sim::PipeClose(h2dPipe.readFd);
        sim::PipeClose(h2dPipe.writeFd);
        return -1;
    }

    pid_t pid = fork();
    if (pid == -1) {
        HCCL_VM_ERROR("fork() failed: {}", strerror(errno));
        sim::PipeClose(h2dPipe.readFd);
        sim::PipeClose(h2dPipe.writeFd);
        sim::PipeClose(d2hPipe.readFd);
        sim::PipeClose(d2hPipe.writeFd);
        return -1;
    }

    // device进程执行参数拼接
    std::vector<char*> argv;
    for (const auto& arg : config.args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }

    std::string h2dReadFdStr = std::to_string(kH2dReadFd);
    std::string d2hWriteFdStr = std::to_string(kD2hWriteFd);
    argv.push_back(const_cast<char*>(h2dReadFdStr.c_str()));
    argv.push_back(const_cast<char*>(d2hWriteFdStr.c_str()));
    argv.push_back(nullptr);

    if (pid == 0) {
        g_logger = nullptr;
        sim::PipeChildSetup(h2dPipe, d2hPipe, kH2dReadFd, kD2hWriteFd);

        for (const auto& env : config.envVars) {
            setenv(env.first.c_str(), env.second.c_str(), 1);
        }

        execvp(config.executable.c_str(), argv.data());
        _exit(127);
    }

    m_pid = pid;
    sim::PipeClose(h2dPipe.readFd);  // 主进程不读 h2dPipe
    sim::PipeClose(d2hPipe.writeFd); // 主进程不写 d2hPipe
    m_h2dWriteFd = h2dPipe.writeFd;
    m_d2hReadFd = d2hPipe.readFd;

    HCCL_VM_INFO("Child process {} created (m_h2dWriteFd={}, m_d2hReadFd={}).", m_pid, m_h2dWriteFd, m_d2hReadFd);

    // 等待device进程管道准备就绪进入Ready状态
    if (WaitForReady() != 0) {
        HCCL_VM_ERROR("Child process {} failed to become READY, aborting.", m_pid);
        DestroyProcess();
        return -1;
    }

    return 0;
}

int SubProcessManager::WaitForReady()
{
    if (m_d2hReadFd < 0 || m_pid < 0) {
        HCCL_VM_ERROR("WaitForReady: child not running.");
        return -1;
    }

    uint8_t rspCmd = 0;
    uint8_t buf[16] = {0};
    uint32_t rspLen = 0;
    if (HostRecvMsg(rspCmd, buf, sizeof(buf), rspLen) != 0) {
        HCCL_VM_ERROR("WaitForReady: child pid={} exited or read failed.", m_pid);
        return -1;
    }
    if (rspCmd != PIPE_RSP_READY) {
        HCCL_VM_ERROR("WaitForReady: unexpected msg cmd=0x{:02x}.", rspCmd);
        return -1;
    }
    HCCL_VM_INFO("Child process {} is READY.", m_pid);
    return 0;
}

int SubProcessManager::DestroyProcess(int timeoutMs)
{
    if (m_pid < 0) {
        HCCL_VM_INFO("Process already stopped.");
        return 0;
    }

    HCCL_VM_INFO("Sending SHUTDOWN to process {}.", m_pid);
    {
        std::lock_guard<std::mutex> lock(m_rpcLock);
        HostSendMsg(PIPE_CMD_SHUTDOWN, nullptr, 0);
    }

    sim::PipeClose(m_h2dWriteFd);

    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        int status;
        pid_t wPid = waitpid(m_pid, &status, WNOHANG);
        if (wPid == m_pid) {
            HCCL_VM_INFO("Child process {} exited with status {}.", m_pid, status);
            break;
        } else if (wPid == -1 && errno == ECHILD) {
            break;
        }

        auto elapsed
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                  .count();
        if (elapsed >= timeoutMs) {
            HCCL_VM_WARN("Child process {} did not exit within {} ms, SIGKILL.", m_pid, timeoutMs);
            kill(m_pid, SIGKILL);
            waitpid(m_pid, &status, 0);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    sim::PipeClose(m_d2hReadFd);

    m_pid = -1;
    return 0;
}

bool SubProcessManager::IsAlive() const
{
    if (m_pid < 0) {
        return false;
    }

    int status = 0;
    pid_t wPid = waitpid(m_pid, &status, WNOHANG);
    return wPid == 0;
}

int SubProcessManager::Request(
    uint8_t reqCmd, const void* reqData, uint32_t reqLen, uint8_t& rspCmd, void* rspData, uint32_t rspMaxLen,
    uint32_t& rspLen)
{
    std::lock_guard<std::mutex> lock(m_rpcLock);
    if (HostSendMsg(reqCmd, reqData, reqLen) != 0) {
        return -1;
    }

    if (HostRecvMsg(rspCmd, rspData, rspMaxLen, rspLen) != 0) {
        return -1;
    }

    return 0;
}

int SubProcessManager::HostSendMsg(uint8_t cmd, const void* data, uint32_t len)
{
    if (m_h2dWriteFd < 0) {
        HCCL_VM_ERROR("HostSendMsg: h2d pipe not open.");
        return -1;
    }

    return PipeSendMsg(m_h2dWriteFd, cmd, data, len);
}

int SubProcessManager::HostRecvMsg(uint8_t& outCmd, void* outData, uint32_t maxLen, uint32_t& outLen)
{
    if (m_d2hReadFd < 0) {
        HCCL_VM_ERROR("HostRecvMsg: d2h pipe not open.");
        return -1;
    }

    return PipeRecvMsg(m_d2hReadFd, outCmd, outData, maxLen, outLen);
}

bool IsAarch64Host()
{
    utsname utsBuf;
    uname(&utsBuf);
    return (std::strstr(utsBuf.machine, "aarch64") != nullptr || std::strstr(utsBuf.machine, "arm") != nullptr);
}

SubProcessConfig CreateAicpuDeviceConfig(uint32_t rankId, uint32_t deviceKey)
{
    SubProcessConfig config;
    std::string rankIdStr = std::to_string(rankId);
    std::string devKeyStr = std::to_string(deviceKey);
    std::string devBinPath = InstallPath::ResolveToInstallRoot("bin/device");

    if (IsAarch64Host()) {
        config.executable = devBinPath;
        config.args.push_back(devBinPath);
    } else {
        config.executable = "qemu-aarch64-static";
        config.args.push_back("qemu-aarch64-static");
        config.args.push_back(devBinPath);
        config.envVars["QEMU_LD_PREFIX"] = "/usr/aarch64-linux-gnu";
    }

    config.args.push_back(rankIdStr);
    config.args.push_back(devKeyStr);

    std::string preloadPath = InstallPath::ResolveToInstallRoot("lib/aarch64/libhccl_device_proxy.so");
    std::string libPath = InstallPath::ResolveToInstallRoot("lib/aarch64");
    const char* ascendHomePath = std::getenv("ASCEND_HOME_PATH");
    if (ascendHomePath != nullptr) {
        libPath += ":";
        libPath += ascendHomePath;
        libPath += "/" + GetArchStr() + "-linux/devlib/device";
    }

    config.envVars["LD_PRELOAD"] = preloadPath;
    config.envVars["LD_LIBRARY_PATH"] = libPath;

    return config;
}

} // namespace sim
