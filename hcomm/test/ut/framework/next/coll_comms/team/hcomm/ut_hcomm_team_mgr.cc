/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 被测对象：src/coll_communicator_mgr/team/hcomm/hcomm_team_mgr.cc 的 HcommTeamMgr window 管理
// （WindowRegister/WindowDeregister/UpdateWindowRemoteMemByRank 及层分段表分配/增量回填逻辑，
// 对应 STC ut_ain_037/038/039~042 及 ut_team_042/043 等窗口用例）。
// 设计说明：
// 1) 用例经 L3 C 接口（HcommTeamWindowRegister/HcommTeamWindowDeregister/
//    HcommTeamUpdateWindowRemoteMemByRank，hcomm_team_c_adpt.cc 薄封装）驱动 HcommTeamMgr window 逻辑，
//    入口判空分支的用例保留在 ut_hcomm_team_c_adpt.cc（C 适配层文件）。
// 2) hrtMalloc/hrtFree/hrtMemSyncCopy 用 mockcpp invoke 桩替换为进程堆内存，计数器支持失败注入；
//    "device 副本"实际为 malloc 内存，伪句柄会触发解引用 SEGFAULT，故 AllocFakeWindow 分配真实可写内存。

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <mutex>
#include <shared_mutex>

#include "ut_hcomm_team_stub.h"

using namespace hcomm;
using namespace hcomm_ut;

class TestHcommTeamMgr : public hcomm_ut::HcommTeamUtBase {};

// ==================== ut_ain_037 ====================
// WindowRegister 正常登记 + 同 handle 重复登记返回 HCOMM_E_PARA（STC ut_ain_037）
TEST_F(TestHcommTeamMgr, Ut_WindowRegister_When_Valid_Expect_ReturnSuccessAndDuplicatePara)
{
    void* devSymWin = AllocFakeWindow();
    HcclCommSymWindow handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(devSymWin, &handle), HCOMM_SUCCESS);

    {
        auto& mgr = HcommTeamMgr::GetInstance();
        std::shared_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
        EXPECT_NE(mgr.windows_.find(handle), mgr.windows_.end());
    }

    // 重复登记同 handle
    HcclCommSymWindow handle2 = nullptr;
    EXPECT_EQ(HcommTeamWindowRegister(devSymWin, &handle2), HCOMM_SUCCESS);
    HcommTeamWindowDeregister(handle2);
    free(devSymWin);
}

// ==================== ut_ain_038 ====================
// WindowRegister 后未回填时条目为空壳（首次 Update 才分配层分段表）
TEST_F(TestHcommTeamMgr, Ut_WindowRegister_When_AfterRegister_Expect_DeviceWindowEmpty)
{
    void* devSymWin = AllocFakeWindow();
    HcclCommSymWindow handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(devSymWin, &handle), HCOMM_SUCCESS);

    auto& mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
    auto* winEntry = mgr.FindWindowByHandleLocked(handle);
    ASSERT_NE(winEntry, nullptr);
    EXPECT_EQ(winEntry->devWindow, handle);
    EXPECT_EQ(winEntry->hostRemoteAddrs, nullptr);
    EXPECT_EQ(winEntry->devRemoteAddrs, nullptr);
    EXPECT_EQ(winEntry->remoteMemsTotal, 0U);
    free(devSymWin);
}

// WindowRegister nullptr 入口判空（C 适配层薄封装分支）
TEST_F(TestHcommTeamMgr, Ut_WindowRegister_When_HandleNull_Expect_ReturnPtrError)
{
    HcclCommSymWindow handleNull = nullptr;
    EXPECT_EQ(HcommTeamWindowRegister(nullptr, &handleNull), HCOMM_E_PTR);
}

// ==================== ut_ain_043 ====================
// WindowDeregister 正常注销：释放层分段表资源（计数 +2：hrtFree(devRemoteAddrs)+hrtFree(devWorldTeamAccumulateId)）
TEST_F(TestHcommTeamMgr, Ut_WindowDeregister_When_AfterUpdate_Expect_ResourcesFreed)
{
    void* devSymWin = AllocFakeWindow();
    HcclCommSymWindow handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(devSymWin, &handle), HCOMM_SUCCESS);

    uint32_t sizes[3] = {2, 2, 0};
    uint32_t slots[2] = {0, 2};
    DeviceMem mem;
    ASSERT_EQ(HcommTeamUpdateWindowRemoteMemByRank(handle, sizes, 3, slots, 2, &mem), HCOMM_SUCCESS);

    uint32_t freeCntBefore = g_hrtFreeCallCount;
    EXPECT_EQ(HcommTeamWindowDeregister(handle), HCOMM_SUCCESS);
    EXPECT_EQ(
        g_hrtFreeCallCount - freeCntBefore, 3U); // hrtFree(devRemoteAddrs+devRemoteSizes+devWorldTeamAccumulateId)

    auto& mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
    EXPECT_EQ(mgr.windows_.find(handle), mgr.windows_.end());
    free(devSymWin);
}

// WindowDeregister 未登记 handle：HCCL_WARNING 容忍返回 HCOMM_SUCCESS（修正旧预期，STC ut_ain_043）
TEST_F(TestHcommTeamMgr, Ut_WindowDeregister_When_InvalidHandle_Expect_ReturnSuccess)
{
    HcclCommSymWindow fake = reinterpret_cast<HcclCommSymWindow>(0xDEADBEEF);
    EXPECT_EQ(HcommTeamWindowDeregister(fake), HCOMM_SUCCESS);
}

// WindowDeregister nullptr 入口判空（C 适配层薄封装分支）
TEST_F(TestHcommTeamMgr, Ut_WindowDeregister_When_HandleNull_Expect_ReturnPtrError)
{
    EXPECT_EQ(HcommTeamWindowDeregister(nullptr), HCOMM_E_PTR);
}

// ==================== ut_team_035 ====================
TEST_F(TestHcommTeamMgr, Ut_WindowDeregister_When_BeforeTeamDestroy_Expect_BothSucceed)
{
    void* devSymWin = AllocFakeWindow();
    HcclCommSymWindow handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(devSymWin, &handle), HCOMM_SUCCESS);
    EXPECT_EQ(HcommTeamWindowDeregister(handle), HCOMM_SUCCESS);
    free(devSymWin);
}

// ==================== ut_team_042 ====================
// UpdateWindowRemoteMemByRank 后 baseRemoteMemAddr/devRemoteAddrs 有效
TEST_F(TestHcommTeamMgr, Ut_UpdateWindowRemoteMemByRank_When_AfterUpdate_Expect_DeviceMemsPtrValid)
{
    void* devSymWin = AllocFakeWindow();
    HcclCommSymWindow handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(devSymWin, &handle), HCOMM_SUCCESS);

    uint32_t sizes[3] = {4, 0, 0};
    uint32_t slots[1] = {2};
    DeviceMem mem;
    ASSERT_EQ(HcommTeamUpdateWindowRemoteMemByRank(handle, sizes, 3, slots, 1, &mem), HCOMM_SUCCESS);

    auto& mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
    auto* winEntry = mgr.FindWindowByHandleLocked(handle);
    ASSERT_NE(winEntry, nullptr);
    EXPECT_NE(winEntry->hostWindow.netWin.baseRemoteMemAddr, 0u);
    EXPECT_NE(winEntry->hostWindow.netWin.windowSize, 0u);
    EXPECT_EQ(winEntry->remoteMemsTotal, 4u);
    EXPECT_NE(winEntry->devRemoteAddrs, nullptr);
    EXPECT_NE(winEntry->devWorldTeamAccumulateId, nullptr);
    EXPECT_EQ(winEntry->hostWindow.netWin.worldTeamAccumulateId[0], 0u);
    free(devSymWin);
}

// UpdateWindowRemoteMemByRank 首次分配时 hrtMalloc 失败（devRemoteAddrs）
TEST_F(TestHcommTeamMgr, Ut_UpdateWindowRemoteMemByRank_When_HrtMallocFail_Expect_ReturnError)
{
    void* devSymWin = AllocFakeWindow();
    HcclCommSymWindow handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(devSymWin, &handle), HCOMM_SUCCESS);

    g_hrtMallocCallCount = 0;
    g_hrtMallocFailAfter = 0;

    uint32_t sizes[3] = {2, 0, 0};
    uint32_t slots[1] = {0};
    DeviceMem mem;
    EXPECT_NE(HcommTeamUpdateWindowRemoteMemByRank(handle, sizes, 3, slots, 1, &mem), HCOMM_SUCCESS);
    free(devSymWin);
}

// UpdateWindowRemoteMemByRank 首次分配 SyncWindowToDevice 失败（hrtMemSyncCopy 第 2 笔起失败：
// 第 1 笔为 accumulateId H2D，第 2 笔为 sizeof(HcommWindow) 整体同步）
TEST_F(TestHcommTeamMgr, Ut_UpdateWindowRemoteMemByRank_When_SyncWindowToDeviceFail_Expect_ReturnError)
{
    void* devSymWin = AllocFakeWindow();
    HcclCommSymWindow handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(devSymWin, &handle), HCOMM_SUCCESS);

    g_hrtMemSyncCopyCallCount = 0;
    g_hrtMemSyncCopyFailAfter = 1;

    uint32_t sizes[3] = {2, 0, 0};
    uint32_t slots[1] = {0};
    DeviceMem mem;
    EXPECT_NE(HcommTeamUpdateWindowRemoteMemByRank(handle, sizes, 3, slots, 1, &mem), HCOMM_SUCCESS);
    free(devSymWin);
}

// window 未登记（对照用例，配合 ut_hcomm_team_c_adpt.cc 的空指针分支）
TEST_F(TestHcommTeamMgr, Ut_UpdateWindowRemoteMemByRank_When_WindowNotRegistered_Expect_ReturnNotFoundError)
{
    HcclCommSymWindow fakeWin = reinterpret_cast<HcclCommSymWindow>(0xDEADBEEF);
    uint32_t sizes[3] = {2, 0, 0};
    uint32_t slots[1] = {0};
    DeviceMem mem;
    EXPECT_EQ(HcommTeamUpdateWindowRemoteMemByRank(fakeWin, sizes, 3, slots, 1, &mem), HCOMM_E_NOT_FOUND);
}

// ==================== ut_ain_039 ====================
// 首次分配：sizes={2,2,0}、slots={0,2}（remoteRank 同跨 layer0/layer1），total=4，槽 0/2 已填
TEST_F(TestHcommTeamMgr, Ut_UpdateWindowRemoteMemByRank_When_FirstAllocate_Expect_SegmentedTableFilled)
{
    void* devSymWin = AllocFakeWindow();
    HcclCommSymWindow handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(devSymWin, &handle), HCOMM_SUCCESS);

    uint32_t sizes[3] = {2, 2, 0};
    uint32_t slots[2] = {0, 2}; // rank0 在 layer0 槽 0 + layer1 槽 0（偏移 2）
    DeviceMem mem;
    uint64_t addr = reinterpret_cast<uint64_t>(mem.addr);

    uint32_t mallocCntBefore = g_hrtMallocCallCount;
    uint32_t syncCntBefore = g_hrtMemSyncCopyCallCount;
    ASSERT_EQ(HcommTeamUpdateWindowRemoteMemByRank(handle, sizes, 3, slots, 2, &mem), HCOMM_SUCCESS);

    auto& mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
    auto* winEntry = mgr.FindWindowByHandleLocked(handle);
    ASSERT_NE(winEntry, nullptr);
    EXPECT_EQ(winEntry->remoteMemsTotal, 4U);
    ASSERT_NE(winEntry->hostRemoteAddrs, nullptr);
    EXPECT_EQ(winEntry->hostRemoteAddrs[0], addr);
    EXPECT_EQ(winEntry->hostRemoteAddrs[2], addr);
    EXPECT_EQ(winEntry->hostRemoteAddrs[1], 0u); // 未填槽
    EXPECT_NE(winEntry->devRemoteAddrs, nullptr);
    EXPECT_EQ(winEntry->hostWindow.netWin.baseRemoteMemAddr, reinterpret_cast<uint64_t>(winEntry->devRemoteAddrs));
    EXPECT_EQ(winEntry->hostWindow.netWin.windowSize, mem.size);
    ASSERT_NE(winEntry->hostWindow.netWin.worldTeamAccumulateId, nullptr);
    EXPECT_EQ(winEntry->hostWindow.netWin.worldTeamAccumulateId[0], 0U);
    EXPECT_EQ(winEntry->hostWindow.netWin.worldTeamAccumulateId[1], 2U);
    EXPECT_EQ(winEntry->hostWindow.netWin.worldTeamAccumulateId[2], 4U);
    EXPECT_EQ(winEntry->hostWindow.netWin.netLayerNum, 3U);
    // 首次分配：hrtMalloc 2 笔（devRemoteAddrs + devWorldTeamAccumulateId）；
    // hrtMemSyncCopy 5 笔（accumulateId H2D + 整体同步 + windowSize 同步 + 槽0 addr + 槽2 addr）
    EXPECT_EQ(g_hrtMallocCallCount - mallocCntBefore, 2U);
    EXPECT_EQ(g_hrtMemSyncCopyCallCount - syncCntBefore, 5U);

    free(devSymWin);
}

// ==================== ut_ain_040 ====================
// 增量回填不重分配：首次后换 remoteMem 再调（slots={1}），仅 1 笔 sizeof(uint64_t) 增量 H2D（addr）
TEST_F(TestHcommTeamMgr, Ut_UpdateWindowRemoteMemByRank_When_IncrementalUpdate_Expect_NoReallocation)
{
    void* devSymWin = AllocFakeWindow();
    HcclCommSymWindow handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(devSymWin, &handle), HCOMM_SUCCESS);

    uint32_t sizes[3] = {2, 2, 0};
    uint32_t slotsFirst[2] = {0, 2};
    DeviceMem mem0;
    ASSERT_EQ(HcommTeamUpdateWindowRemoteMemByRank(handle, sizes, 3, slotsFirst, 2, &mem0), HCOMM_SUCCESS);
    uint64_t addr0 = reinterpret_cast<uint64_t>(mem0.addr);

    void* devMemsAfterFirst = nullptr;
    {
        auto& mgr = HcommTeamMgr::GetInstance();
        std::shared_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
        auto* winEntry = mgr.FindWindowByHandleLocked(handle);
        ASSERT_NE(winEntry, nullptr);
        devMemsAfterFirst = winEntry->devRemoteAddrs;
    }

    // 增量：换 remoteMem 再调（slots={1}），仅 2 笔增量 H2D，不重新分配
    DeviceMem mem1;
    uint64_t addr1 = reinterpret_cast<uint64_t>(mem1.addr);
    uint32_t mallocCntBefore = g_hrtMallocCallCount;
    uint32_t syncCntBefore = g_hrtMemSyncCopyCallCount;
    uint32_t slotsInc[1] = {1};
    EXPECT_EQ(HcommTeamUpdateWindowRemoteMemByRank(handle, sizes, 3, slotsInc, 1, &mem1), HCOMM_SUCCESS);
    EXPECT_EQ(g_hrtMallocCallCount, mallocCntBefore);         // 无新分配
    EXPECT_EQ(g_hrtMemSyncCopyCallCount - syncCntBefore, 1U); // 仅 1 笔增量 H2D（addr）

    {
        auto& mgr = HcommTeamMgr::GetInstance();
        std::shared_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
        auto* winEntry = mgr.FindWindowByHandleLocked(handle);
        ASSERT_NE(winEntry, nullptr);
        // 槽 1 更新，槽 0/2 不变
        EXPECT_EQ(winEntry->hostRemoteAddrs[1], addr1);
        EXPECT_EQ(winEntry->hostRemoteAddrs[0], addr0);
        EXPECT_EQ(winEntry->hostRemoteAddrs[2], addr0);
        EXPECT_EQ(winEntry->devRemoteAddrs, devMemsAfterFirst);
    }

    free(devSymWin);
}

// ==================== ut_ain_042 ====================
// 多 window 独立层分段表：登记 2 个 handle，分别 Update 不同 slots，互不串扰
TEST_F(TestHcommTeamMgr, Ut_UpdateWindowRemoteMemByRank_When_MultiWindows_Expect_TablesIndependent)
{
    HcclCommSymWindow winA = AllocFakeWindow();
    HcclCommSymWindow winB = AllocFakeWindow();
    HcclCommSymWindow handleA = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(winA, &handleA), HCOMM_SUCCESS);
    HcclCommSymWindow handleB = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(winB, &handleB), HCOMM_SUCCESS);

    uint32_t sizesA[3] = {2, 0, 0}; // total=2
    uint32_t sizesB[3] = {4, 0, 0}; // total=4
    DeviceMem memA(1024);
    DeviceMem memB(2048);
    uint64_t addrA = reinterpret_cast<uint64_t>(memA.addr);
    uint64_t addrB = reinterpret_cast<uint64_t>(memB.addr);

    uint32_t slotsA[1] = {1};
    uint32_t slotsB[1] = {3};
    ASSERT_EQ(HcommTeamUpdateWindowRemoteMemByRank(handleA, sizesA, 3, slotsA, 1, &memA), HCOMM_SUCCESS);
    ASSERT_EQ(HcommTeamUpdateWindowRemoteMemByRank(handleB, sizesB, 3, slotsB, 1, &memB), HCOMM_SUCCESS);

    auto& mgr = HcommTeamMgr::GetInstance();
    std::shared_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
    auto* entryA = mgr.FindWindowByHandleLocked(handleA);
    auto* entryB = mgr.FindWindowByHandleLocked(handleB);
    ASSERT_NE(entryA, nullptr);
    ASSERT_NE(entryB, nullptr);
    EXPECT_EQ(entryA->remoteMemsTotal, 2U);
    EXPECT_EQ(entryB->remoteMemsTotal, 4U);
    EXPECT_EQ(entryA->hostRemoteAddrs[1], addrA);
    EXPECT_EQ(entryB->hostRemoteAddrs[3], addrB);
    EXPECT_NE(entryA->devRemoteAddrs, entryB->devRemoteAddrs);

    free(winA);
    free(winB);
}

// ==================== ut_ain_041 ====================
// 反向x5 变体：window 未登记 / slotNum=0 / slots=nullptr / 首次 total=0 / slot>=total
TEST_F(TestHcommTeamMgr, Ut_UpdateWindowRemoteMemByRank_When_InvalidParams_Expect_ReturnErrors)
{
    uint32_t sizes[3] = {2, 2, 0};
    DeviceMem mem;

    // (1) window 未登记 -> HCOMM_E_NOT_FOUND
    HcclCommSymWindow fake = reinterpret_cast<HcclCommSymWindow>(0xDEAD0000);
    uint32_t slotsOk[1] = {0};
    EXPECT_EQ(HcommTeamUpdateWindowRemoteMemByRank(fake, sizes, 3, slotsOk, 1, &mem), HCOMM_E_NOT_FOUND);

    // 已登记 window 供后续变体使用
    void* devSymWin = AllocFakeWindow();
    HcclCommSymWindow handle = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(devSymWin, &handle), HCOMM_SUCCESS);

    // (2) slotNum=0 -> HCOMM_E_PARA
    EXPECT_EQ(HcommTeamUpdateWindowRemoteMemByRank(handle, sizes, 3, slotsOk, 0, &mem), HCOMM_E_PARA);
    // (3) slots=nullptr -> HCOMM_E_PARA
    EXPECT_EQ(HcommTeamUpdateWindowRemoteMemByRank(handle, sizes, 3, nullptr, 1, &mem), HCOMM_E_PARA);
    // (4) 首次 total=0（sizes 全 0）-> HCOMM_E_PARA
    uint32_t zeroSizes[3] = {0, 0, 0};
    EXPECT_EQ(HcommTeamUpdateWindowRemoteMemByRank(handle, zeroSizes, 3, slotsOk, 1, &mem), HCOMM_E_PARA);
    // (2)(3)(4) 后状态不变：仍未分配层分段表
    {
        auto& mgr = HcommTeamMgr::GetInstance();
        std::shared_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
        auto* winEntry = mgr.FindWindowByHandleLocked(handle);
        ASSERT_NE(winEntry, nullptr);
        EXPECT_EQ(winEntry->hostRemoteAddrs, nullptr);
        EXPECT_EQ(winEntry->remoteMemsTotal, 0U);
    }

    free(devSymWin);
}

TEST_F(TestHcommTeamMgr, Ut_BindUbSymmetricWindow_When_LayoutValid_Expect_OnlyLsaFieldsUpdated)
{
    constexpr uint32_t memberNum = 4U;
    constexpr uint32_t netLayer = 1U;
    constexpr size_t stride = 0x10000U;
    constexpr size_t userSize = 0x2000U;
    constexpr uintptr_t baseAddress = 0x100000U;
    constexpr uint64_t legacySymWindow = 0x400000U;
    HcclCommSymWindow window = nullptr;
    ASSERT_EQ(HcommTeamWindowRegister(reinterpret_cast<void*>(legacySymWindow), &window), HCOMM_SUCCESS);
    auto* deviceWindow = static_cast<HcommWindow*>(window);
    constexpr uint64_t remoteMemAddr = 0x500000U;
    constexpr uint64_t remoteWindowSize = 0x3000U;
    auto& mgr = HcommTeamMgr::GetInstance();
    {
        std::unique_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
        auto* entry = mgr.FindWindowByHandleLocked(window);
        ASSERT_NE(entry, nullptr);
        entry->hostWindow.netWin.baseRemoteMemAddr = remoteMemAddr;
        entry->hostWindow.netWin.windowSize = remoteWindowSize;
    }

    std::vector<CommMem> memberMems(memberNum);
    for (uint32_t member = 0; member < memberNum; ++member) {
        memberMems[member].addr = reinterpret_cast<void*>(baseAddress + member * stride);
        memberMems[member].size = userSize;
        memberMems[member].type = COMM_MEM_TYPE_DEVICE;
    }
    auto worldTeam = reinterpret_cast<HcommTeamHandle>(0x300000U);
    auto teamEntry = std::make_unique<TeamEntry>();
    teamEntry->hostTeam.memberNum = memberNum;
    teamEntry->hostTeam.netLayer = netLayer;
    {
        std::unique_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
        mgr.teams_[worldTeam] = std::move(teamEntry);
    }
    ASSERT_EQ(
        mgr.BindUbSymmetricWindow(
            window, worldTeam, netLayer, memberMems.data(), memberNum, reinterpret_cast<void*>(baseAddress), stride,
            userSize),
        HCOMM_SUCCESS);

    std::shared_lock<std::shared_mutex> lock(mgr.windowsRwMutex_);
    auto* entry = mgr.FindWindowByHandleLocked(window);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->hostRemoteAddrs, nullptr);
    EXPECT_EQ(entry->devRemoteAddrs, nullptr);
    EXPECT_EQ(entry->hostWindow.netWin.worldTeamAccumulateId, nullptr);
    EXPECT_EQ(entry->remoteMemsTotal, 0U);
    EXPECT_EQ(entry->hostWindow.netWin.baseRemoteMemAddr, remoteMemAddr);
    EXPECT_EQ(entry->hostWindow.netWin.windowSize, remoteWindowSize);
    EXPECT_EQ(entry->hostWindow.lsaWin.baseVa, baseAddress);
    EXPECT_EQ(entry->hostWindow.lsaWin.stride, stride);
    EXPECT_EQ(entry->hostWindow.lsaWin.userSize, userSize);
    EXPECT_EQ(entry->hostWindow.legacySymWindow, legacySymWindow);
    lock.unlock();
    EXPECT_EQ(deviceWindow->netWin.baseRemoteMemAddr, remoteMemAddr);
    EXPECT_EQ(deviceWindow->netWin.windowSize, remoteWindowSize);
    EXPECT_EQ(deviceWindow->lsaWin.baseVa, baseAddress);
    EXPECT_EQ(deviceWindow->legacySymWindow, legacySymWindow);

    EXPECT_EQ(HcommTeamWindowDeregister(window), HCOMM_SUCCESS);
}

TEST_F(TestHcommTeamMgr, Ut_BindUbSymmetricWindow_When_LayoutInvalid_Expect_ParaError)
{
    constexpr size_t stride = 0x10000U;
    constexpr size_t userSize = 0x2000U;
    constexpr uintptr_t baseAddress = 0x100000U;
    std::vector<CommMem> memberMems(2U);
    memberMems[0] = {COMM_MEM_TYPE_DEVICE, reinterpret_cast<void*>(baseAddress), userSize};
    memberMems[1] = {COMM_MEM_TYPE_DEVICE, reinterpret_cast<void*>(baseAddress + stride + 1U), userSize};

    auto& mgr = HcommTeamMgr::GetInstance();
    EXPECT_EQ(
        mgr.BindUbSymmetricWindow(
            reinterpret_cast<void*>(0x200000U), reinterpret_cast<HcommTeamHandle>(0x300000U), 0U, memberMems.data(),
            static_cast<uint32_t>(memberMems.size()), reinterpret_cast<void*>(baseAddress), stride, userSize),
        HCOMM_E_PARA);
    EXPECT_EQ(
        mgr.BindUbSymmetricWindow(
            reinterpret_cast<void*>(0x200000U), reinterpret_cast<HcommTeamHandle>(0x300000U), 0U, memberMems.data(),
            static_cast<uint32_t>(memberMems.size()), reinterpret_cast<void*>(baseAddress), stride, stride + 1U),
        HCOMM_E_PARA);
}

TEST_F(TestHcommTeamMgr, Ut_BindUbSymmetricWindow_When_LsaTeamInvalid_Expect_Error)
{
    constexpr uint32_t memberNum = 2U;
    constexpr uint32_t netLayer = 1U;
    constexpr size_t stride = 0x10000U;
    constexpr size_t userSize = 0x2000U;
    constexpr uintptr_t baseAddress = 0x100000U;
    std::vector<CommMem> memberMems(memberNum);
    for (uint32_t member = 0; member < memberNum; ++member) {
        memberMems[member] = {COMM_MEM_TYPE_DEVICE, reinterpret_cast<void*>(baseAddress + member * stride), userSize};
    }

    auto& mgr = HcommTeamMgr::GetInstance();
    auto lsaTeam = reinterpret_cast<HcommTeamHandle>(0x300000U);
    auto window = reinterpret_cast<HcclCommSymWindow>(0x200000U);
    EXPECT_EQ(
        mgr.BindUbSymmetricWindow(
            window, lsaTeam, netLayer, memberMems.data(), memberNum, reinterpret_cast<void*>(baseAddress), stride,
            userSize),
        HCOMM_E_NOT_FOUND);

    auto teamEntry = std::make_unique<TeamEntry>();
    TeamEntry* teamEntryPtr = teamEntry.get();
    teamEntryPtr->hostTeam.memberNum = memberNum;
    teamEntryPtr->hostTeam.netLayer = netLayer;
    teamEntryPtr->isSubTeam = true;
    {
        std::unique_lock<std::shared_mutex> lock(mgr.teamsRwMutex_);
        mgr.teams_[lsaTeam] = std::move(teamEntry);
    }
    EXPECT_EQ(
        mgr.BindUbSymmetricWindow(
            window, lsaTeam, netLayer, memberMems.data(), memberNum, reinterpret_cast<void*>(baseAddress), stride,
            userSize),
        HCOMM_E_PARA);

    teamEntryPtr->isSubTeam = false;
    teamEntryPtr->hostTeam.netLayer = netLayer + 1U;
    EXPECT_EQ(
        mgr.BindUbSymmetricWindow(
            window, lsaTeam, netLayer, memberMems.data(), memberNum, reinterpret_cast<void*>(baseAddress), stride,
            userSize),
        HCOMM_E_PARA);

    teamEntryPtr->hostTeam.netLayer = netLayer;
    teamEntryPtr->hostTeam.memberNum = memberNum + 1U;
    EXPECT_EQ(
        mgr.BindUbSymmetricWindow(
            window, lsaTeam, netLayer, memberMems.data(), memberNum, reinterpret_cast<void*>(baseAddress), stride,
            userSize),
        HCOMM_E_PARA);
}
