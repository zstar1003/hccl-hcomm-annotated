/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

#include "sim_log.h"

namespace sim {
// Sentinel value written to proxy_ready to signal runner to exit.
// Normal rounds use values in [1, UINT32_MAX-1].
constexpr uint32_t kRunnerExitSignal = UINT32_MAX;

class ProcessSyncer {
public:
    explicit ProcessSyncer(const std::string& syncDir = "/dev/shm/hccl_sync/") : m_syncDir(syncDir)
    {
        if (!m_syncDir.empty() && m_syncDir.back() != '/') {
            m_syncDir += '/';
        }
        m_fileProxyReady = m_syncDir + "proxy_ready.flag";
        m_fileRunnerDone = m_syncDir + "runner_done.flag";
    }

    void Init()
    {
        (void)mkdir(m_syncDir.data(), 0777);
        writeCounter(m_fileProxyReady, 0);
        writeCounter(m_fileRunnerDone, 0);
    }

    // Returns the most recently completed round number (= value in runner_done.flag).
    // Proxies use this to compute their expected targetRound = getCurrentRound() + 1.
    uint32_t getCurrentRound() const { return readCounter(m_fileRunnerDone); }

    // hccl-vm: ready all proxies have arrived, notify runner and wait for ACK.
    void notifyRunnerAndWaitAcknowledge(uint32_t targetRound)
    {
        writeCounter(m_fileProxyReady, targetRound);

        while (readCounter(m_fileProxyReady) != 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    // proxy: wait until runner has completed targetRound.
    void blockProxyUntilRunnerDone(uint32_t targetRound)
    {
        while (true) {
            if (readCounter(m_fileRunnerDone) >= targetRound) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    // proxy: same as blockProxyUntilRunnerDone but with a timeout.
    // Returns true if runner completed targetRound within timeoutMs, false if timed out.
    bool tryBlockProxyUntilRunnerDone(uint32_t targetRound, int timeoutMs)
    {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        while (true) {
            if (readCounter(m_fileRunnerDone) >= targetRound) {
                return true;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    // runner: check if a new task is available. If so, returns targetRound > 0 and
    // clears proxy_ready to 0 (ACK). Returns 0 if no new task.
    // Returns kRunnerExitSignal if hccl-vm has asked runner to exit.
    uint32_t checkProxyReadySignal()
    {
        uint32_t ready = readCounter(m_fileProxyReady);
        if (ready == 0) {
            return 0;
        }
        if (ready == kRunnerExitSignal) {
            return kRunnerExitSignal;
        }
        writeCounter(m_fileProxyReady, 0);
        return ready;
    }

    // runner: signal that targetRound has been completed.
    void notifyProxyToContinue(uint32_t targetRound) { writeCounter(m_fileRunnerDone, targetRound); }

    // hccl-vm: signal runner to exit.
    void signalRunnerExit() { writeCounter(m_fileProxyReady, kRunnerExitSignal); }

    // hccl-vm: reset all state (called on hccl-vm reset/restart).
    void Reset()
    {
        writeCounter(m_fileProxyReady, 0);
        writeCounter(m_fileRunnerDone, 0);
    }

private:
    static void writeCounter(const std::string& filepath, uint32_t value)
    {
        char buf[16] = {0};
        int n = std::snprintf(buf, sizeof(buf), "%u", value);

        std::string tmpPath = filepath + ".tmp";

        int fd = ::open(tmpPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            HCCL_VM_ERROR("[ProcessSyncer] open tmp failed: {}, error: {}", tmpPath.c_str(), std::strerror(errno));
            return;
        }

        ssize_t written = ::write(fd, buf, n);
        if (written != n) {
            HCCL_VM_ERROR("[ProcessSyncer] write failed: {}", std::strerror(errno));
        }

        if (::fsync(fd) != 0) {
            HCCL_VM_ERROR("[ProcessSyncer] fsync failed: {}", std::strerror(errno));
        }
        ::close(fd);

        if (::rename(tmpPath.c_str(), filepath.c_str()) != 0) {
            HCCL_VM_ERROR("[ProcessSyncer] rename failed: {}", std::strerror(errno));
        }
    }

    static uint32_t readCounter(const std::string& filepath)
    {
        std::ifstream f(filepath);
        uint32_t value = 0;
        if (f.is_open()) {
            f >> value;
        }
        return value;
    }

private:
    std::string m_syncDir;
    std::string m_fileProxyReady;
    std::string m_fileRunnerDone;
};
} // namespace sim
