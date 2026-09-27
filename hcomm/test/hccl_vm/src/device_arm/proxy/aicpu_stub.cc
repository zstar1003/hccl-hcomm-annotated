/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <mutex>
#include <thread>
namespace aicpu {
std::mutex g_sqeIdMtx;
constexpr uint32_t INITAL_SQE_ID = 0x80000000U;
uint32_t g_sqeId = INITAL_SQE_ID;
void GetSqeId(const uint32_t num, uint32_t& start, uint32_t& end)
{
    std::lock_guard<std::mutex> lk(g_sqeIdMtx);
    start = g_sqeId;
    g_sqeId += num;
    end = g_sqeId;
    if (start > end) {
        g_sqeId = INITAL_SQE_ID;
        start = g_sqeId;
        g_sqeId += num;
        if (start >= end) {
            g_sqeId = INITAL_SQE_ID;
            return;
        }
    }
    return;
}
} // namespace aicpu

extern "C" {

// 打桩支持AICPU上背景线程，用于销毁通信域等场景
int32_t StartMC2MaintenanceThread(void (*f1)(void*), void* p1, void (*f2)(void*), void* p2)
{
    // 创建一个真正运行的后台线程（非 MC2 固件，而是 std::thread）
    // AicpuDaemonService::ServiceRun 会无限循环直到 command == Stop
    // 守护线程 detach：进程退出时自然死亡
    std::thread(f1, p1).detach();
    return 0;
}

int32_t AicpuCreateCtrlThread(int32_t type, void (*f1)(void*), void* p1, void (*f2)(void*), void* p2)
{
    std::thread(f1, p1).detach();
    return 0;
}

} // extern "C"
