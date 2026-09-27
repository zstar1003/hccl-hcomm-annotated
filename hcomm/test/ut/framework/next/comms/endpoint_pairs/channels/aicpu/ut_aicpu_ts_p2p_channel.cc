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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "aicpu/aicpu_ts_p2p_channel.h"
#include "endpoint.h"
#include "buffer/local_ipc_rma_buffer_v2.h"
#include "makebufs_helper.h"
#undef private
#undef protected
using namespace hcomm;

namespace {
class StubEndpointForP2pChannel : public Endpoint {
public:
    StubEndpointForP2pChannel() : Endpoint(MakeDesc()) {}

    HcclResult Init() override { return HCCL_SUCCESS; }
    // 内存方法在 RegedMemMgr 上，Endpoint 不再 override
    RegedMemMgr* GetRegedMemMgr() override { return nullptr; }
    void* GetRdmaHandle() override { return ctxHandle_; }
    bool IsCtxHandleValid() const override { return ctxHandle_ != nullptr; }

private:
    void* ctxHandle_{nullptr};
    static EndpointDesc MakeDesc()
    {
        EndpointDesc desc{};
        desc.protocol = COMM_PROTOCOL_PCIE;
        desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        desc.loc.device.devPhyId = 3;
        return desc;
    }
};

class NullBufRmaBuffer : public Hccl::LocalRmaBuffer {
public:
    NullBufRmaBuffer() : Hccl::LocalRmaBuffer(nullptr, Hccl::RmaType::IPC) {}
    std::string Describe() const override { return "NullBufRmaBuffer"; }
};

std::shared_ptr<Hccl::LocalIpcRmaBuffer> MakeLocalIpcRmaBuffer(uintptr_t addr, u64 size, const char* tag)
{
    auto buffer = std::make_shared<Hccl::Buffer>(addr, size, HCCL_MEM_TYPE_DEVICE, tag);
    return std::make_shared<Hccl::LocalIpcRmaBuffer>(buffer);
}

HcommResult StubP2pGetAllMemHandlesOne(EndpointHandle, void** memHandles, uint32_t* memHandleNum)
{
    static auto localBuffer = MakeLocalIpcRmaBuffer(0x4000U, 0x1000U, "p2p_all");
    static std::shared_ptr<Hccl::LocalIpcRmaBuffer> localBuffers[1] = {localBuffer};
    *memHandles = localBuffers;
    *memHandleNum = 1;
    return HCCL_SUCCESS;
}
} // namespace

class AicpuTsP2pChannelTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(AicpuTsP2pChannelTest, UT_MakeRmaBufferVecFromMemHandles_When_NullHandle_Expect_ReturnHCCL_E_PTR)
{
    HcommMemHandle memHandles[1] = {nullptr};
    std::vector<Hccl::LocalRmaBuffer*> bufferVec;

    EXPECT_EQ(MakeRmaBufferVecFromMemHandles(memHandles, 1, bufferVec, "AicpuTsP2pChannel"), HCCL_E_PTR);
    EXPECT_TRUE(bufferVec.empty());
}

TEST_F(AicpuTsP2pChannelTest, UT_MakeRmaBufferVecFromMemHandles_When_ZeroHandles_Expect_EmptyBufferVec)
{
    std::vector<Hccl::LocalRmaBuffer*> bufferVec{reinterpret_cast<Hccl::LocalRmaBuffer*>(0x1)};

    ASSERT_EQ(MakeRmaBufferVecFromMemHandles(nullptr, 0, bufferVec, "AicpuTsP2pChannel"), HCCL_SUCCESS);
    EXPECT_TRUE(bufferVec.empty());
}

TEST_F(AicpuTsP2pChannelTest, UT_MakeRmaBufferVecFromMemHandles_When_ValidHandle_Expect_FillBufferVec)
{
    auto localRmaBuffer = MakeLocalIpcRmaBuffer(0x1000U, 0x2000U, "p2p_ut");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localRmaBuffer.get())};
    std::vector<Hccl::LocalRmaBuffer*> bufferVec;

    ASSERT_EQ(MakeRmaBufferVecFromMemHandles(memHandles, 1, bufferVec, "AicpuTsP2pChannel"), HCCL_SUCCESS);
    ASSERT_EQ(bufferVec.size(), 1U);
    EXPECT_EQ(bufferVec[0], localRmaBuffer.get());
}

TEST_F(AicpuTsP2pChannelTest, UT_MakeRmaBufferVecFromMemHandles_When_GetBufNull_Expect_ReturnHCCL_E_PTR)
{
    NullBufRmaBuffer nullBuf;
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(&nullBuf)};
    std::vector<Hccl::LocalRmaBuffer*> bufferVec;

    EXPECT_EQ(MakeRmaBufferVecFromMemHandles(memHandles, 1, bufferVec, "AicpuTsP2pChannel"), HCCL_E_PTR);
    EXPECT_TRUE(bufferVec.empty());
}

TEST_F(AicpuTsP2pChannelTest, UT_ParseInputParam_When_ExchangeAllMemsFalse_Expect_FillCommonRes)
{
    StubEndpointForP2pChannel endpoint;
    auto localRmaBuffer = MakeLocalIpcRmaBuffer(0x3000U, 0x1000U, "p2p_parse");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localRmaBuffer.get())};
    HcommChannelDesc desc{};
    desc.exchangeAllMems = false;
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;

    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.commonRes_.bufferVec.size(), 1U);
    EXPECT_EQ(ch.commonRes_.bufferVec[0], localRmaBuffer.get());
}

TEST_F(AicpuTsP2pChannelTest, UT_ParseInputParam_When_ExchangeAllMemsTrue_Expect_FillCommonRes)
{
    StubEndpointForP2pChannel endpoint;
    HcommChannelDesc desc{};
    desc.exchangeAllMems = true;
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);
    MOCKER(HcommMemGetAllMemHandles).stubs().will(invoke(StubP2pGetAllMemHandlesOne));

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.commonRes_.bufferVec.size(), 1U);
    EXPECT_EQ(ch.commonRes_.bufferVec[0]->GetAddr(), 0x4000U);
}

// P2P 收编走 V2 后，host V2 分流依赖 GetChannelKind()==AICPU_TS_PCIE 识别；对齐
// ut_aicpu_ts_roce_channel_v2.cc 的 Ut_When_GetChannelKind_Expect_RoceV2。
TEST_F(AicpuTsP2pChannelTest, Ut_When_GetChannelKind_Expect_Pcie)
{
    StubEndpointForP2pChannel endpoint;
    HcommChannelDesc desc{};
    AicpuTsP2pChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    EXPECT_EQ(ch.GetChannelKind(), HcommChannelKind::AICPU_TS_PCIE);
}
