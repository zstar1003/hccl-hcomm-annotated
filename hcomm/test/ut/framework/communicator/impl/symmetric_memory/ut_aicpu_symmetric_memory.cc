/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#ifndef private
#define private public
#define protected public
#endif
#include "symmetric_memory.h"
#undef private
#undef protected
#include "hccl_sym_win.h"
#include "hcomm/hcomm_team_entity_defs.h"

using namespace hccl;

namespace {
// 构造一个 mode 为 URMA 的 SymmetricWindow，仅填充本接口访问的字段（mode/remoteMems/remoteMemNum），其余置零。
SymmetricWindow BuildUrmaSymmetricWindow(CommMem* remoteMems, u32 remoteMemNum)
{
    SymmetricWindow win{};
    win.mode = SymmetricMemoryMode::URMA;
    win.remoteMems = remoteMems;
    win.remoteMemNum = remoteMemNum;
    return win;
}

// 构造一个合法的 CommMem。
CommMem BuildValidCommMem(void* addr, uint64_t size, CommMemType type = COMM_MEM_TYPE_DEVICE)
{
    CommMem mem{};
    mem.type = type;
    mem.addr = addr;
    mem.size = size;
    return mem;
}

// AIN 重构后 winHandle 为 HcommWindow*，SymmetricWindow 挂在 legacySymWindow 字段（device 指针语义，
// UT 中直接存 host 地址）。构造仅填充本接口访问字段的包装句柄。
HcommWindow WrapToHcommWindow(SymmetricWindow* symWin)
{
    HcommWindow win{};
    win.header.magicWord = HCOMM_WINDOW_MAGIC_WORD;
    win.header.version = HCOMM_WINDOW_VERSION;
    win.header.size = sizeof(HcommWindow);
    win.legacySymWindow = reinterpret_cast<uint64_t>(symWin);
    return win;
}
} // namespace

class AicpuSymmetricMemoryTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AicpuSymmetricMemoryTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "AicpuSymmetricMemoryTest TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "AicpuSymmetricMemoryTest Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "AicpuSymmetricMemoryTest Test TearDown" << std::endl;
    }
};

// 测试目的：winHandle为空指针时参数校验失败，预期返回HCCL_E_PTR
TEST_F(AicpuSymmetricMemoryTest, Ut_HcclSymWinGetRemoteAddr_When_WinHandleIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    void* ptr = nullptr;
    HcclResult ret = HcclSymWinGetRemoteAddr(nullptr, 0, 0, &ptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 测试目的：输出指针ptr为空时参数校验失败，预期返回HCCL_E_PTR
TEST_F(AicpuSymmetricMemoryTest, Ut_HcclSymWinGetRemoteAddr_When_PtrIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    SymmetricWindow win{};
    HcommWindow hWin = WrapToHcommWindow(&win);
    HcclResult ret = HcclSymWinGetRemoteAddr(reinterpret_cast<HcclCommSymWindow>(&hWin), 0, 0, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 测试目的：窗口模式非URMA时不支持该接口，预期返回HCCL_E_NOT_SUPPORT且*ptr置为nullptr
TEST_F(AicpuSymmetricMemoryTest, Ut_HcclSymWinGetRemoteAddr_When_ModeIsNotURMA_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    SymmetricWindow win{};
    win.mode = SymmetricMemoryMode::HCCS;        // 非 URMA
    void* ptr = reinterpret_cast<void*>(0xDEAD); // 哨兵值
    HcommWindow hWin = WrapToHcommWindow(&win);
    HcclResult ret = HcclSymWinGetRemoteAddr(reinterpret_cast<HcclCommSymWindow>(&hWin), 0, 0, &ptr);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ptr, nullptr); // 验证 *ptr 被置为 nullptr
}

// 测试目的：URMA模式下remoteMems为空指针时参数校验失败，预期返回HCCL_E_PTR
TEST_F(AicpuSymmetricMemoryTest, Ut_HcclSymWinGetRemoteAddr_When_RemoteMemsIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    SymmetricWindow win = BuildUrmaSymmetricWindow(nullptr, 0);
    void* ptr = nullptr;
    HcommWindow hWin = WrapToHcommWindow(&win);
    HcclResult ret = HcclSymWinGetRemoteAddr(reinterpret_cast<HcclCommSymWindow>(&hWin), 0, 0, &ptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 测试目的：peerRank越界(>=remoteMemNum)时参数校验失败，预期返回HCCL_E_PARA
TEST_F(AicpuSymmetricMemoryTest, Ut_HcclSymWinGetRemoteAddr_When_PeerRankOutOfRange_Expect_ReturnIsHCCL_E_PARA)
{
    CommMem remoteMems[2] = {};
    SymmetricWindow win = BuildUrmaSymmetricWindow(remoteMems, 2);
    void* ptr = nullptr;
    HcommWindow hWin = WrapToHcommWindow(&win);
    // peerRank=2 等于 remoteMemNum, >= 成立
    HcclResult ret = HcclSymWinGetRemoteAddr(reinterpret_cast<HcclCommSymWindow>(&hWin), 0, 2, &ptr);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 测试目的：远端内存addr为空指针时参数校验失败，预期返回HCCL_E_PARA（覆盖||短路左侧）
TEST_F(AicpuSymmetricMemoryTest, Ut_HcclSymWinGetRemoteAddr_When_RemoteMemAddrIsNull_Expect_ReturnIsHCCL_E_PARA)
{
    CommMem remoteMems[2] = {};
    remoteMems[1] = BuildValidCommMem(nullptr, 1024, COMM_MEM_TYPE_DEVICE);
    SymmetricWindow win = BuildUrmaSymmetricWindow(remoteMems, 2);
    void* ptr = nullptr;
    HcommWindow hWin = WrapToHcommWindow(&win);
    HcclResult ret = HcclSymWinGetRemoteAddr(reinterpret_cast<HcclCommSymWindow>(&hWin), 0, 1, &ptr);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 测试目的：远端内存type为COMM_MEM_TYPE_INVALID时参数校验失败，预期返回HCCL_E_PARA（覆盖||短路右侧）
TEST_F(AicpuSymmetricMemoryTest, Ut_HcclSymWinGetRemoteAddr_When_RemoteMemTypeIsInvalid_Expect_ReturnIsHCCL_E_PARA)
{
    CommMem remoteMems[2] = {};
    remoteMems[1] = BuildValidCommMem(reinterpret_cast<void*>(0x5000000), 1024, COMM_MEM_TYPE_INVALID);
    SymmetricWindow win = BuildUrmaSymmetricWindow(remoteMems, 2);
    void* ptr = nullptr;
    HcommWindow hWin = WrapToHcommWindow(&win);
    HcclResult ret = HcclSymWinGetRemoteAddr(reinterpret_cast<HcclCommSymWindow>(&hWin), 0, 1, &ptr);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 测试目的：offset越界(>=remoteMem.size)时参数校验失败，预期返回HCCL_E_PARA
TEST_F(AicpuSymmetricMemoryTest, Ut_HcclSymWinGetRemoteAddr_When_OffsetOutOfSize_Expect_ReturnIsHCCL_E_PARA)
{
    CommMem remoteMems[2] = {};
    remoteMems[1] = BuildValidCommMem(reinterpret_cast<void*>(0x5000000), 1024, COMM_MEM_TYPE_DEVICE);
    SymmetricWindow win = BuildUrmaSymmetricWindow(remoteMems, 2);
    void* ptr = nullptr;
    HcommWindow hWin = WrapToHcommWindow(&win);
    // offset=1024 等于 size, >= 成立
    HcclResult ret = HcclSymWinGetRemoteAddr(reinterpret_cast<HcclCommSymWindow>(&hWin), 1024, 1, &ptr);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 测试目的：全部参数合法时获取远端内存地址，预期返回HCCL_SUCCESS且*ptr等于remoteMem.addr+offset
TEST_F(AicpuSymmetricMemoryTest, Ut_HcclSymWinGetRemoteAddr_When_AllValid_Expect_ReturnIsHCCL_SUCCESS)
{
    const uint64_t offset = 256;
    const uint64_t size = 1024;
    void* baseAddr = reinterpret_cast<void*>(0x5000000);
    CommMem remoteMems[2] = {};
    remoteMems[1] = BuildValidCommMem(baseAddr, size, COMM_MEM_TYPE_DEVICE);
    SymmetricWindow win = BuildUrmaSymmetricWindow(remoteMems, 2);
    void* ptr = nullptr;
    HcommWindow hWin = WrapToHcommWindow(&win);
    HcclResult ret = HcclSymWinGetRemoteAddr(reinterpret_cast<HcclCommSymWindow>(&hWin), offset, 1, &ptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ptr, reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(baseAddr) + offset));
}

TEST_F(AicpuSymmetricMemoryTest, Ut_HcclSymWinGetPeerPointer_When_UbWindowValid_Expect_UseLsaMemberId)
{
    constexpr uintptr_t baseAddress = 0x280000000000ULL;
    constexpr uint64_t stride = 0x40000000ULL;
    constexpr uint64_t userSize = 0x200000ULL;
    constexpr size_t offset = 0x80U;
    HcommWindow window{};
    window.header.magicWord = HCOMM_WINDOW_MAGIC_WORD;
    window.header.version = HCOMM_WINDOW_VERSION;
    window.header.size = sizeof(HcommWindow);
    window.lsaWin.baseVa = baseAddress;
    window.lsaWin.stride = stride;
    window.lsaWin.userSize = userSize;

    void* ptr = nullptr;
    EXPECT_EQ(HcclSymWinGetPeerPointer(&window, offset, 1U, &ptr), HCCL_SUCCESS);
    EXPECT_EQ(ptr, reinterpret_cast<void*>(baseAddress + stride + offset));
}

TEST_F(AicpuSymmetricMemoryTest, Ut_HcclSymWinGetPeerPointer_When_UbAndUrmaCoexist_Expect_UseLsaMemberId)
{
    constexpr uintptr_t baseAddress = 0x280000000000ULL;
    constexpr uint64_t stride = 0x40000000ULL;
    HcommWindow window{};
    window.header.magicWord = HCOMM_WINDOW_MAGIC_WORD;
    window.header.version = HCOMM_WINDOW_VERSION;
    window.header.size = sizeof(HcommWindow);
    window.lsaWin.baseVa = baseAddress;
    window.lsaWin.stride = stride;
    window.lsaWin.userSize = 0x200000ULL;
    window.legacySymWindow = 0x100000U;

    void* ptr = nullptr;
    EXPECT_EQ(HcclSymWinGetPeerPointer(&window, 0x80U, 1U, &ptr), HCCL_SUCCESS);
    EXPECT_EQ(ptr, reinterpret_cast<void*>(baseAddress + stride + 0x80U));
}

TEST_F(AicpuSymmetricMemoryTest, Ut_HcclSymWinGetPeerPointer_When_UbParameterInvalid_Expect_ParaError)
{
    constexpr uintptr_t baseAddress = 0x280000000000ULL;
    constexpr uint64_t stride = 0x40000000ULL;
    constexpr uint64_t userSize = 0x200000ULL;
    HcommWindow window{};
    window.header.magicWord = HCOMM_WINDOW_MAGIC_WORD;
    window.header.version = HCOMM_WINDOW_VERSION;
    window.header.size = sizeof(HcommWindow);
    window.lsaWin.baseVa = baseAddress;
    window.lsaWin.stride = stride;
    window.lsaWin.userSize = userSize;

    void* ptr = reinterpret_cast<void*>(0xDEADU);
    EXPECT_EQ(HcclSymWinGetPeerPointer(&window, userSize, 1U, &ptr), HCCL_E_PARA);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(AicpuSymmetricMemoryTest, Ut_HcclSymWinGetPeerPointer_When_LegacyWindowValid_Expect_UseRankId)
{
    constexpr uintptr_t baseAddress = 0x100000U;
    constexpr size_t stride = 0x10000U;
    SymmetricWindow window{};
    window.mode = SymmetricMemoryMode::HCCS;
    window.rankSize = 4U;
    window.baseVa = reinterpret_cast<void*>(baseAddress);
    window.stride = stride;
    window.userSize = 0x1000U;

    void* ptr = nullptr;
    EXPECT_EQ(HcclSymWinGetPeerPointer(&window, 0x40U, 2U, &ptr), HCCL_SUCCESS);
    EXPECT_EQ(ptr, reinterpret_cast<void*>(baseAddress + 2U * stride + 0x40U));
}
