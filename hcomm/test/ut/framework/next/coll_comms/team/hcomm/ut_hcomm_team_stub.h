/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_HCOMM_TEAM_STUB_H
#define UT_HCOMM_TEAM_STUB_H

// hcomm team 系列 UT（ut_hccl_team_c_adpt.cc 不适用；ut_hcomm_team_c_adpt.cc /
// ut_hcomm_team_mgr.cc 共享）的公共段：private/protected 展开下的被测头 include、
// hrtMalloc/hrtFree/hrtMemSyncCopy mockcpp 桩（进程堆内存 + 计数器支持失败注入）、
// 伪 window 句柄分配、HcommTeamMgr 单例清理、公共 fixture 基类 HcommTeamUtBase。

#include <cstdlib>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <securec.h>

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

// 两个 UT 均需访问 HcommTeamMgr 私有成员（锁/map/FreeXxxResources），统一在此展开宏包含
#define private public
#define protected public
#include "hcomm_team_entity_defs.h"
#include "hcomm_team_c_adpt.h"
#include "hcomm_team_mgr.h"
#undef protected
#undef private

#include "hcomm_team.h"
#include "hcomm_res_defs.h"
#include "hcomm_result_defs.h"
#include "adapter_rts_common.h"

namespace hcomm_ut {

// 构造 device 侧 CommMem（malloc 内存，默认 1024B），析构时自动 free——用例内直接用，
// 消除各测试手写 malloc/free 配对。memSize 为 0 视为非法，addr 置空由用例断言发现。
struct DeviceMem : public CommMem {
    static constexpr size_t DEFAULT_MEM_SIZE = 1024;
    explicit DeviceMem(size_t memSize = DEFAULT_MEM_SIZE)
    {
        type = COMM_MEM_TYPE_DEVICE;
        if (memSize == 0) {
            addr = nullptr;
            size = 0;
            return;
        }
        addr = malloc(memSize);
        size = memSize;
    }
    ~DeviceMem()
    {
        free(addr);
        addr = nullptr;
    }
    DeviceMem(const DeviceMem&) = delete;
    DeviceMem& operator=(const DeviceMem&) = delete;
};

// 分配"真实"的 HcommWindow device 副本内存（SyncWindowToDevice 会拷贝到 devWindow 指针，
// 伪句柄（如 0x30002）在 StubHrtMemSyncCopy 中会被解引用导致 SEGFAULT，必须用真实可写内存）。
inline HcclCommSymWindow AllocFakeWindow() { return static_cast<HcclCommSymWindow>(malloc(sizeof(HcommWindow))); }

inline void FreeFakeWindow(HcclCommSymWindow& handle)
{
    if (handle != nullptr) {
        free(handle);
        handle = nullptr;
    }
}

inline uint32_t g_hrtMallocFailAfter = UINT32_MAX;
inline uint32_t g_hrtMallocCallCount = 0;
inline uint32_t g_hrtMemSyncCopyFailAfter = UINT32_MAX;
inline uint32_t g_hrtMemSyncCopyCallCount = 0;
inline uint32_t g_hrtFreeCallCount = 0;

inline void ResetStubCounters()
{
    g_hrtMallocFailAfter = UINT32_MAX;
    g_hrtMallocCallCount = 0;
    g_hrtMemSyncCopyFailAfter = UINT32_MAX;
    g_hrtMemSyncCopyCallCount = 0;
    g_hrtFreeCallCount = 0;
}

inline HcclResult StubHrtMalloc(void** devPtr, u64 size, bool level2Address)
{
    (void)level2Address;
    g_hrtMallocCallCount++;
    if (g_hrtMallocCallCount > g_hrtMallocFailAfter) {
        return HCCL_E_INTERNAL;
    }
    *devPtr = malloc(static_cast<size_t>(size));
    return (*devPtr != nullptr) ? HCCL_SUCCESS : HCCL_E_PTR;
}

inline HcclResult StubHrtFree(void* devPtr)
{
    g_hrtFreeCallCount++;
    free(devPtr);
    return HCCL_SUCCESS;
}

inline HcclResult
StubHrtMemSyncCopy(void* dst, uint64_t destMax, const void* src, uint64_t count, HcclRtMemcpyKind kind)
{
    (void)destMax;
    (void)kind;
    g_hrtMemSyncCopyCallCount++;
    if (g_hrtMemSyncCopyCallCount > g_hrtMemSyncCopyFailAfter) {
        return HCCL_E_INTERNAL;
    }
    if (dst == nullptr || src == nullptr) {
        return HCCL_E_PTR;
    }
    // 伪地址（UT 填的伪指针，如远端 mem.addr / 伪 ChannelHandle）不可解引用；直接跳过拷贝
    // （UT 不验证拷贝内容），避免解引用触发 SEGFAULT。
    if (reinterpret_cast<uintptr_t>(src) < 0x10000 || reinterpret_cast<uintptr_t>(dst) < 0x10000) {
        return HCCL_SUCCESS;
    }
    if (memcpy_s(dst, static_cast<size_t>(count), src, static_cast<size_t>(count)) != EOK) {
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

// 清理 HcommTeamMgr 单例残留（teams_/windows_/windowToTeamMap_），避免用例间串扰。
inline void CleanupHcommTeamMgrSingleton()
{
    auto& mgr = hcomm::HcommTeamMgr::GetInstance();
    {
        std::unique_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
        for (auto& pair : mgr.teams_) {
            if (pair.second != nullptr) {
                mgr.FreeTeamResources(pair.second.get());
            }
        }
        mgr.teams_.clear();
    }
    {
        std::unique_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
        for (auto& pair : mgr.windows_) {
            if (pair.second != nullptr) {
                mgr.FreeWindowResources(pair.second.get());
            }
        }
        mgr.windows_.clear();
    }
}

// hcomm team 系列 UT 公共 fixture：桩注册（SetUp）+ 单例清理与 mock 校验（TearDown）。
// 用例 fixture 继承本类即可获得统一的桩环境。
class HcommTeamUtBase : public testing::Test {
protected:
    void SetUp() override
    {
        ResetStubCounters();
        MOCKER(hrtMalloc).stubs().will(invoke(StubHrtMalloc));
        MOCKER(hrtFree).stubs().will(invoke(StubHrtFree));
        MOCKER(hrtMemSyncCopy).stubs().will(invoke(StubHrtMemSyncCopy));
    }

    void TearDown() override
    {
        CleanupHcommTeamMgrSingleton();
        GlobalMockObject::verify();
    }
};

} // namespace hcomm_ut

#endif // UT_HCOMM_TEAM_STUB_H
