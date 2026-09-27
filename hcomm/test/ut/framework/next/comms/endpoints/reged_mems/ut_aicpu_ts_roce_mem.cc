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
#include <securec.h>
#include <string>
#include <vector>

#include "hccl_network.h"
#include "local_rdma_rma_buffer.h"

#define private public
#include "aicpu_ts_roce_reged_mem_mgr.h"
#undef private

using namespace hcomm;

class AicpuTsRoceRegedMemMgrTest : public testing::Test {
protected:
    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
};

TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_RegisterMemory_When_NetDevNull_Returns_E_PTR)
{
    AicpuTsRoceRegedMemMgr mgr(nullptr, nullptr);
    HcommMem mem{};
    mem.addr = reinterpret_cast<void*>(0x1000U);
    mem.size = 4096U;
    mem.type = COMM_MEM_TYPE_DEVICE;
    void* handle = nullptr;
    EXPECT_EQ(mgr.RegisterMemory(&mem, "t", &handle), HCCL_E_PTR);
}

TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_MemoryExport_When_MemHandleNull_Returns_E_PTR)
{
    AicpuTsRoceRegedMemMgr mgr(nullptr, nullptr);
    EndpointDesc ep{};
    void* desc = nullptr;
    uint32_t len = 0;
    EXPECT_EQ(mgr.MemoryExport(ep, nullptr, &desc, &len), HCCL_E_PTR);
}

TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_MemoryExport_When_MemDescOutNull_Returns_E_PTR)
{
    AicpuTsRoceRegedMemMgr mgr(nullptr, nullptr);
    EndpointDesc ep{};
    void* fake = reinterpret_cast<void*>(0x1);
    uint32_t len = 0;
    EXPECT_EQ(mgr.MemoryExport(ep, fake, nullptr, &len), HCCL_E_PTR);
}

TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_MemoryExport_When_MemHandleUnregistered_Returns_NOT_FOUND)
{
    hccl::HcclIpAddress localIp;
    ASSERT_EQ(localIp.SetReadableAddress("127.0.0.1"), HCCL_SUCCESS);
    hccl::NetDevContext netCtx;
    ASSERT_EQ(netCtx.Init(NicType::DEVICE_NIC_TYPE, 0, 0, localIp), HCCL_SUCCESS);

    AicpuTsRoceRegedMemMgr mgr(reinterpret_cast<HcclNetDev>(&netCtx), nullptr);
    EndpointDesc ep{};
    auto localRdmaRmaBuffer = std::make_shared<hccl::LocalRdmaRmaBuffer>(
        reinterpret_cast<HcclNetDevCtx>(&netCtx), reinterpret_cast<void*>(0xD2000ULL), 4096U, hccl::RmaMemType::DEVICE);
    void* memHandle = localRdmaRmaBuffer.get();

    mgr.allRegisteredBuffers_.emplace_back(localRdmaRmaBuffer, false);
    mgr.allRegisteredBuffers_.clear();

    void* desc = nullptr;
    uint32_t len = 0;
    EXPECT_EQ(mgr.MemoryExport(ep, memHandle, &desc, &len), HCCL_E_NOT_FOUND);
    EXPECT_EQ(desc, nullptr);
    EXPECT_EQ(len, 0U);
}

TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_MemoryImport_When_DescTooShort_Returns_E_PARA)
{
    AicpuTsRoceRegedMemMgr mgr(nullptr, nullptr);
    char buf[4] = {};
    HcommMem out{};
    EXPECT_EQ(mgr.MemoryImport(buf, sizeof(buf), &out), HCCL_E_PARA);
}

TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_MemoryImport_When_OutMemNull_Returns_E_PTR)
{
    AicpuTsRoceRegedMemMgr mgr(nullptr, nullptr);
    char buf[sizeof(EndpointDesc)] = {};
    EXPECT_EQ(mgr.MemoryImport(buf, sizeof(buf), nullptr), HCCL_E_PTR);
}

TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_MemoryUnimport_When_MemDescNull_Returns_E_PTR)
{
    AicpuTsRoceRegedMemMgr mgr(nullptr, nullptr);
    EXPECT_EQ(mgr.MemoryUnimport(nullptr, sizeof(EndpointDesc)), HCCL_E_PTR);
}

TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_GetAllMemHandles_When_CountOutNull_Returns_E_PTR)
{
    AicpuTsRoceRegedMemMgr mgr(nullptr, nullptr);
    void* handles = nullptr;
    EXPECT_EQ(mgr.GetAllMemHandles(&handles, nullptr), HCCL_E_PTR);
}

TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_GetAllMemDetails_When_NetDevNull_Returns_E_PTR)
{
    AicpuTsRoceRegedMemMgr mgr(nullptr, nullptr);
    std::vector<RoceMemDetails> local;
    std::vector<RoceMemDetails> remote;
    EXPECT_EQ(mgr.GetAllMemDetails(local, remote), HCCL_E_PTR);
}

TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_GetAllMemHandles_When_NoRecords_Returns_SUCCESS)
{
    AicpuTsRoceRegedMemMgr mgr(nullptr, nullptr);
    void* handles = reinterpret_cast<void*>(0xdeadbeefULL);
    uint32_t n = 99U;
    ASSERT_EQ(mgr.GetAllMemHandles(&handles, &n), HCCL_SUCCESS);
    EXPECT_EQ(n, 0U);
    EXPECT_EQ(handles, nullptr);
}

TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_UnregisterMemory_When_MemHandleNull_Returns_E_PTR)
{
    AicpuTsRoceRegedMemMgr mgr(nullptr, nullptr);
    EXPECT_EQ(mgr.UnregisterMemory(nullptr), HCCL_E_PTR);
}

TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_MemoryUnimport_When_RemoteMgrMissing_Returns_NOT_FOUND)
{
    AicpuTsRoceRegedMemMgr mgr(nullptr, nullptr);
    EndpointDesc ep{};
    ep.protocol = COMM_PROTOCOL_ROCE;
    ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    std::vector<char> buf(sizeof(EndpointDesc));
    ASSERT_EQ(memcpy_s(buf.data(), buf.size(), &ep, sizeof(ep)), EOK);
    EXPECT_EQ(mgr.MemoryUnimport(buf.data(), static_cast<uint32_t>(buf.size())), HCCL_E_NOT_FOUND);
}

TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_GetParamsFromMemDesc_WhenValid_Returns_SUCCESS)
{
    AicpuTsRoceRegedMemMgr mgr(nullptr, nullptr);
    EndpointDesc ep{};
    ep.protocol = COMM_PROTOCOL_ROCE;
    ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    const std::string rdmaBlob = "rdma-serialized-bytes";
    std::vector<char> buf;
    buf.insert(buf.end(), rdmaBlob.begin(), rdmaBlob.end());
    std::vector<char> tail(sizeof(EndpointDesc));
    ASSERT_EQ(memcpy_s(tail.data(), tail.size(), &ep, sizeof(ep)), EOK);
    buf.insert(buf.end(), tail.begin(), tail.end());

    EndpointDesc outEp{};
    std::string outRdma;
    ASSERT_EQ(mgr.GetParamsFromMemDesc(buf.data(), static_cast<uint32_t>(buf.size()), outEp, outRdma), HCCL_SUCCESS);
    EXPECT_EQ(outRdma, rdmaBlob);
    EXPECT_EQ(std::memcmp(&outEp, &ep, sizeof(EndpointDesc)), 0);
}

TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_RegisterMemory_WithNetDev_MockLocalRdmaInit_ThenAgain_Returns_SUCCESS)
{
    MOCKER_CPP(&hccl::LocalRdmaRmaBuffer::Init).stubs().will(returnValue(HCCL_SUCCESS));

    hccl::HcclIpAddress localIp;
    ASSERT_EQ(localIp.SetReadableAddress("127.0.0.1"), HCCL_SUCCESS);
    hccl::NetDevContext netCtx;
    ASSERT_EQ(netCtx.Init(NicType::DEVICE_NIC_TYPE, 0, 0, localIp), HCCL_SUCCESS);

    AicpuTsRoceRegedMemMgr mgr(reinterpret_cast<HcclNetDev>(&netCtx), nullptr);
    HcommMem mem{};
    mem.addr = reinterpret_cast<void*>(0x9000ULL);
    mem.size = 4096U;
    mem.type = COMM_MEM_TYPE_DEVICE;
    void* h1 = nullptr;
    ASSERT_EQ(mgr.RegisterMemory(&mem, "t", &h1), HCCL_SUCCESS);
    ASSERT_NE(h1, nullptr);

    void* h2 = nullptr;
    EXPECT_EQ(mgr.RegisterMemory(&mem, "t", &h2), HCCL_SUCCESS);
    EXPECT_NE(h2, h1);
    auto* aliasBuffer = static_cast<hccl::LocalRdmaRmaBuffer*>(h2);
    EXPECT_TRUE(aliasBuffer->IsAlias());

    EXPECT_EQ(mgr.UnregisterMemory(h2), HCCL_SUCCESS);
    EXPECT_EQ(mgr.UnregisterMemory(h1), HCCL_SUCCESS);
}

TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_RegisterMemory_When_LocalRdmaInitFails_Returns_Error)
{
    MOCKER_CPP(&hccl::LocalRdmaRmaBuffer::Init).stubs().will(returnValue(HCCL_E_INTERNAL));

    hccl::HcclIpAddress localIp;
    ASSERT_EQ(localIp.SetReadableAddress("127.0.0.1"), HCCL_SUCCESS);
    hccl::NetDevContext netCtx;
    ASSERT_EQ(netCtx.Init(NicType::DEVICE_NIC_TYPE, 0, 0, localIp), HCCL_SUCCESS);

    AicpuTsRoceRegedMemMgr mgr(reinterpret_cast<HcclNetDev>(&netCtx), nullptr);
    HcommMem mem{};
    mem.addr = reinterpret_cast<void*>(0xA000ULL);
    mem.size = 4096U;
    mem.type = COMM_MEM_TYPE_DEVICE;
    void* handle = nullptr;
    EXPECT_EQ(mgr.RegisterMemory(&mem, "init_fail", &handle), HCCL_E_INTERNAL);
    EXPECT_EQ(handle, nullptr);
}

TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_RegisterMemory_When_AliasRegistered_Expect_UnregisterChildFirstSuccess)
{
    MOCKER_CPP(&hccl::LocalRdmaRmaBuffer::Init).stubs().will(returnValue(HCCL_SUCCESS));

    hccl::HcclIpAddress localIp;
    ASSERT_EQ(localIp.SetReadableAddress("127.0.0.1"), HCCL_SUCCESS);
    hccl::NetDevContext netCtx;
    ASSERT_EQ(netCtx.Init(NicType::DEVICE_NIC_TYPE, 0, 0, localIp), HCCL_SUCCESS);

    AicpuTsRoceRegedMemMgr mgr(reinterpret_cast<HcclNetDev>(&netCtx), nullptr);
    HcommMem mem{};
    mem.addr = reinterpret_cast<void*>(0xB000ULL);
    mem.size = 4096U;
    mem.type = COMM_MEM_TYPE_DEVICE;

    void* parentHandle = nullptr;
    ASSERT_EQ(mgr.RegisterMemory(&mem, "parent", &parentHandle), HCCL_SUCCESS);
    ASSERT_NE(parentHandle, nullptr);

    void* childHandle = nullptr;
    ASSERT_EQ(mgr.RegisterMemory(&mem, "child", &childHandle), HCCL_SUCCESS);
    ASSERT_NE(childHandle, nullptr);
    EXPECT_NE(childHandle, parentHandle);
    auto* childBuffer = static_cast<hccl::LocalRdmaRmaBuffer*>(childHandle);
    EXPECT_TRUE(childBuffer->IsAlias());

    EXPECT_EQ(mgr.UnregisterMemory(childHandle), HCCL_SUCCESS);
    EXPECT_EQ(mgr.UnregisterMemory(parentHandle), HCCL_SUCCESS);
}

// 父+子集注册 → 先解注册父(soft-delete) → 验证allRegisteredBuffers_状态 → 再解注册子
TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_UnregisterParentFirst_Expect_ParentSoftDeleted)
{
    MOCKER_CPP(&hccl::LocalRdmaRmaBuffer::Init).stubs().will(returnValue(HCCL_SUCCESS));

    hccl::HcclIpAddress localIp;
    ASSERT_EQ(localIp.SetReadableAddress("127.0.0.1"), HCCL_SUCCESS);
    hccl::NetDevContext netCtx;
    ASSERT_EQ(netCtx.Init(NicType::DEVICE_NIC_TYPE, 0, 0, localIp), HCCL_SUCCESS);

    AicpuTsRoceRegedMemMgr mgr(reinterpret_cast<HcclNetDev>(&netCtx), nullptr);
    HcommMem mem0;
    mem0.addr = reinterpret_cast<void*>(0xC000ULL);
    mem0.size = 4096U;
    mem0.type = COMM_MEM_TYPE_DEVICE;
    void* parentHandle = nullptr;
    ASSERT_EQ(mgr.RegisterMemory(&mem0, "parent", &parentHandle), HCCL_SUCCESS);
    ASSERT_NE(parentHandle, nullptr);
    auto* parentBuf = static_cast<hccl::LocalRdmaRmaBuffer*>(parentHandle);
    EXPECT_FALSE(parentBuf->IsAlias());

    // 子集注册（alias）
    void* childHandle = nullptr;
    ASSERT_EQ(mgr.RegisterMemory(&mem0, "child", &childHandle), HCCL_SUCCESS);
    ASSERT_NE(childHandle, nullptr);
    EXPECT_NE(childHandle, parentHandle);
    auto* childBuf = static_cast<hccl::LocalRdmaRmaBuffer*>(childHandle);
    EXPECT_TRUE(childBuf->IsAlias());
    EXPECT_EQ(mgr.allRegisteredBuffers_.size(), 2u);

    // 先解注册父（ref 2→1，IsInTree=true，父标记为soft-deleted）
    EXPECT_EQ(mgr.UnregisterMemory(parentHandle), HCCL_SUCCESS);

    // 验证：父在allRegisteredBuffers_中标记为true（soft-deleted），子标记为false
    EXPECT_EQ(mgr.allRegisteredBuffers_.size(), 2u);
    auto itParent
        = std::find_if(mgr.allRegisteredBuffers_.begin(), mgr.allRegisteredBuffers_.end(), [parentBuf](const auto& e) {
              return e.first.get() == parentBuf;
          });
    ASSERT_NE(itParent, mgr.allRegisteredBuffers_.end());
    EXPECT_TRUE(itParent->second);

    auto itChild
        = std::find_if(mgr.allRegisteredBuffers_.begin(), mgr.allRegisteredBuffers_.end(), [childBuf](const auto& e) {
              return e.first.get() == childBuf;
          });
    ASSERT_NE(itChild, mgr.allRegisteredBuffers_.end());
    EXPECT_FALSE(itChild->second);

    // 再解注册子（ref 1→0，子从allRegisteredBuffers_擦除）
    EXPECT_EQ(mgr.UnregisterMemory(childHandle), HCCL_SUCCESS);

    // 验证：子已从allRegisteredBuffers_中移除
    itChild
        = std::find_if(mgr.allRegisteredBuffers_.begin(), mgr.allRegisteredBuffers_.end(), [childBuf](const auto& e) {
              return e.first.get() == childBuf;
          });
    EXPECT_EQ(itChild, mgr.allRegisteredBuffers_.end());
}

// GetAllMemHandles: 注册后 → 解注册后 → 验证hcclBufRecords_与句柄数一致
TEST_F(AicpuTsRoceRegedMemMgrTest, Ut_GetAllMemHandles_When_RecordsExist_Expect_CorrectCount)
{
    MOCKER_CPP(&hccl::LocalRdmaRmaBuffer::Init).stubs().will(returnValue(HCCL_SUCCESS));

    hccl::HcclIpAddress localIp;
    ASSERT_EQ(localIp.SetReadableAddress("127.0.0.1"), HCCL_SUCCESS);
    hccl::NetDevContext netCtx;
    ASSERT_EQ(netCtx.Init(NicType::DEVICE_NIC_TYPE, 0, 0, localIp), HCCL_SUCCESS);

    AicpuTsRoceRegedMemMgr memMgr(reinterpret_cast<HcclNetDev>(&netCtx), nullptr);
    HcommMem mem{};
    mem.addr = reinterpret_cast<void*>(0xD000ULL);
    mem.size = 4096U;
    mem.type = COMM_MEM_TYPE_DEVICE;

    void* handles = nullptr;
    uint32_t count = 99U;
    EXPECT_EQ(memMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 0U);

    void* h1 = nullptr;
    ASSERT_EQ(memMgr.RegisterMemory(&mem, "t1", &h1), HCCL_SUCCESS);
    EXPECT_EQ(memMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 1U);
    EXPECT_NE(handles, nullptr);

    void* h2 = nullptr;
    ASSERT_EQ(memMgr.RegisterMemory(&mem, "t2", &h2), HCCL_SUCCESS);
    EXPECT_EQ(memMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 2U);

    EXPECT_EQ(memMgr.UnregisterMemory(h1), HCCL_SUCCESS);
    EXPECT_EQ(memMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 1U);

    EXPECT_EQ(memMgr.UnregisterMemory(h2), HCCL_SUCCESS);
    EXPECT_EQ(memMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 0U);
}
