/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include <vector>
#include <iostream>

#define private public

#include "cpu_urma_endpoint.h"
#include "host_cpu_urma_channel.h"
#include "sockets/socket_mgr.h"
#include "buffer/local_ub_rma_buffer.h"
#include "host_ub_connection.h"
#include "ip_address.h"
#include "op_mode.h"
#include "rdma_handle_manager.h"
#include "hcomm_res.h"
#include "hcomm_c_adpt.h"
#include "socket.h"
#include "urma_api.h"
#include "dlurma_function.h"
#include "hcomm_adapter_urma.h"
#include "env_config/env_config_v2.h"

using namespace hcomm;

namespace {
std::shared_ptr<Hccl::LocalUbRmaBuffer> MakeHostLocalBuffer(uintptr_t addr, u64 size, const char* tag)
{
    auto buffer = std::make_shared<Hccl::Buffer>(addr, size, HCCL_MEM_TYPE_DEVICE, tag);
    return std::make_shared<Hccl::LocalUbRmaBuffer>(buffer);
}

class NonUbRmaBuffer : public Hccl::LocalRmaBuffer {
public:
    explicit NonUbRmaBuffer(std::shared_ptr<Hccl::Buffer> buffer) : Hccl::LocalRmaBuffer(buffer, Hccl::RmaType::RDMA) {}

    std::string Describe() const override { return "NonUbRmaBuffer"; }
};

HcommResult StubHostGetAllMemHandlesOne(EndpointHandle, void** memHandles, uint32_t* memHandleNum)
{
    static std::shared_ptr<Hccl::Buffer> buffer
        = std::make_shared<Hccl::Buffer>(0x730000U, 0x1000U, HCCL_MEM_TYPE_DEVICE, "host_all");
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffer = std::make_shared<Hccl::LocalUbRmaBuffer>(buffer);
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffers[1] = {localBuffer};
    *memHandles = localBuffers;
    *memHandleNum = 1;
    return HCCL_SUCCESS;
}
} // namespace

class HostCpuUrmaChannelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HostCpuUrmaChannelTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "HostCpuUrmaChannelTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in HostCpuUrmaChannelTest SetUP" << std::endl;
        Hccl::IpAddress localIp("1.0.0.0");
        Hccl::IpAddress remoteIp("2.0.0.0");

        MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
        // 出参 devPhyId 需赋合法值：批次1后它作为 GetDeviceResMgr(devPhyId) 的设备数组下标
        unsigned int devicePhyId = 0U;
        MOCKER(hrtGetDevicePhyIdByIndex)
            .stubs()
            .with(mockcpp::any(), outBound(devicePhyId))
            .will(returnValue(HCCL_SUCCESS));

        RdmaHandle rdmaHandle = (void*)0x1000000;
        MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle));
        MOCKER(RaSocketSetWhiteListStatus).stubs().will(returnValue(0));

        EndpointDesc epDesc{};
        epDesc.protocol = COMM_PROTOCOL_UB_CTP;
        epDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        epDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
        epDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
        endpoint = std::make_unique<CpuUrmaEndpoint>(epDesc);
        endpoint->Init();
        endpointHandle = static_cast<EndpointHandle>(endpoint.get());

        channelDesc.remoteEndpoint = epDesc;
        channelDesc.exchangeAllMems = false;
        channelDesc.port = 60001;

        fakeSocket = new Hccl::Socket(
            nullptr, localIp, 60001, remoteIp, "_0_1_", Hccl::SocketRole::SERVER, Hccl::NicType::HOST_NIC_TYPE);
        channelDesc.socket = static_cast<void*>(fakeSocket);

        localBufferPtr = std::make_shared<Hccl::Buffer>(666);
        rdmaHandle_ = rdmaHandle;
    }

    virtual void TearDown()
    {
        unsetenv("HCCL_DFS_CONFIG");
        GlobalMockObject::verify();
        std::cout << "A Test case in HostCpuUrmaChannelTest TearDown" << std::endl;
        delete fakeSocket;
    }

    std::unique_ptr<CpuUrmaEndpoint> endpoint;
    EndpointHandle endpointHandle{};
    HcommChannelDesc channelDesc{};
    Hccl::Socket* fakeSocket;
    std::shared_ptr<Hccl::Buffer> localBufferPtr;
    RdmaHandle rdmaHandle_{nullptr};
};

TEST_F(HostCpuUrmaChannelTest, Ut_When_ParseInputParam_NullEndpoint_Expect_HCCL_E_PTR)
{
    channelDesc.socket = nullptr;
    auto impl = std::make_unique<HostCpuUrmaChannel>(nullptr, channelDesc);
    EXPECT_EQ(impl->Init(), HCCL_E_PTR);
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_StartListen_Fail_Expect_Error)
{
    channelDesc.role = HCOMM_SOCKET_ROLE_SERVER;
    MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_E_INTERNAL)));
    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    EXPECT_NE(impl->Init(), HCCL_SUCCESS);
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_GetRemoteMems_Expect_Success)
{
    MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle_));
    MOCKER(RaSocketSetWhiteListStatus).stubs().will(returnValue(0));

    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    ASSERT_EQ(impl->Init(), HCCL_SUCCESS);

    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    char** memInfos = nullptr;
    // GetRemoteMems depends on memTransport_, which is initialized in Init()
    // This test verifies the function can be called
    EXPECT_NO_THROW(impl->GetRemoteMems(&memNum, &remoteMem, &memInfos));
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_Clean_Expect_NotSupport)
{
    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    EXPECT_EQ(impl->Clean(), HCCL_E_NOT_SUPPORT);
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_Resume_Expect_NotSupport)
{
    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    EXPECT_EQ(impl->Resume(), HCCL_E_NOT_SUPPORT);
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_GetSplitNum_ZeroLen_Expect_Error)
{
    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    uint64_t splitNum = 0;
    uint64_t maxJettyWrDataLen = 256 * 1024 * 1024; // 256MB
    EXPECT_EQ(impl->GetSplitNum(0, maxJettyWrDataLen, splitNum), HCCL_E_PARA);
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_GetSplitNum_ValidLen_Expect_Success)
{
    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    uint64_t splitNum = 0;
    uint64_t maxJettyWrDataLen = 256 * 1024 * 1024; // 256MB
    // Test with len = 256MB (should be 1 split)
    EXPECT_EQ(impl->GetSplitNum(256 * 1024 * 1024, maxJettyWrDataLen, splitNum), HCCL_SUCCESS);
    EXPECT_EQ(splitNum, 1);

    // Test with len = 512MB (should be 2 splits)
    EXPECT_EQ(impl->GetSplitNum(512 * 1024 * 1024, maxJettyWrDataLen, splitNum), HCCL_SUCCESS);
    EXPECT_EQ(splitNum, 2);

    // Test with len = 300MB (should be 2 splits)
    EXPECT_EQ(impl->GetSplitNum(300 * 1024 * 1024, maxJettyWrDataLen, splitNum), HCCL_SUCCESS);
    EXPECT_EQ(splitNum, 2);
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_BuildSocket_WithSocket_Expect_Success)
{
    MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle_));
    MOCKER(RaSocketSetWhiteListStatus).stubs().will(returnValue(0));

    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    // socket_ is already set in SetUp, BuildSocket should return success
    EXPECT_EQ(impl->Init(), HCCL_SUCCESS);
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_ParseInputParam_NullSocket_Expect_Success)
{
    channelDesc.socket = nullptr;
    MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle_));
    MOCKER(RaSocketSetWhiteListStatus).stubs().will(returnValue(0));
    MOCKER_CPP(&hcomm::SocketMgr::GetSocket)
        .stubs()
        .with(mockcpp::any(), outBound(fakeSocket))
        .will(returnValue(HCCL_SUCCESS));

    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    // ParseInputParam should succeed, BuildSocket will create a new socket
    EXPECT_NO_THROW(impl->Init());
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_Init_WithDefaultPort_Expect_Success)
{
    channelDesc.port = 0; // Use default port
    MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle_));
    MOCKER(RaSocketSetWhiteListStatus).stubs().will(returnValue(0));

    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    EXPECT_EQ(impl->Init(), HCCL_SUCCESS);
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_GetLocSeg_EmptyBuffers_Expect_Error)
{
    MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle_));
    MOCKER(RaSocketSetWhiteListStatus).stubs().will(returnValue(0));

    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    ASSERT_EQ(impl->Init(), HCCL_SUCCESS);

    // commonRes_.bufferVec is empty after Init with exchangeAllMems = false
    u64 seg = 0;
    void* addr = reinterpret_cast<void*>(0x1000);
    EXPECT_EQ(impl->GetLocSeg(addr, 1024, &seg), HCCL_E_INTERNAL);
}

TEST_F(HostCpuUrmaChannelTest, UT_GetLocSeg_When_BufferMatches_Expect_ReturnHCCL_SUCCESS)
{
    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    auto localBuffer = MakeHostLocalBuffer(0x710000U, 0x2000U, "host_match");
    impl->commonRes_.bufferVec.push_back(localBuffer.get());

    u64 seg = 0;
    void* addr = reinterpret_cast<void*>(0x710100U);
    EXPECT_EQ(impl->GetLocSeg(addr, 0x100, &seg), HCCL_SUCCESS);
    EXPECT_EQ(seg, localBuffer->GetTargetSeg());
}

TEST_F(HostCpuUrmaChannelTest, UT_GetLocSeg_When_BufferDoesNotMatch_Expect_ReturnHCCL_E_INTERNAL)
{
    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    auto localBuffer = MakeHostLocalBuffer(0x720000U, 0x1000U, "host_not_match");
    impl->commonRes_.bufferVec.push_back(localBuffer.get());

    u64 seg = 0xFF;
    void* addr = reinterpret_cast<void*>(0x721100U);
    EXPECT_EQ(impl->GetLocSeg(addr, 0x100, &seg), HCCL_E_INTERNAL);
    EXPECT_EQ(seg, 0xFFU);
}

TEST_F(HostCpuUrmaChannelTest, UT_GetLocSeg_When_MatchedBufferIsNotUb_Expect_ReturnHCCL_E_PTR)
{
    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    auto buffer = std::make_shared<Hccl::Buffer>(0x725000U, 0x1000U, HCCL_MEM_TYPE_DEVICE, "host_non_ub");
    NonUbRmaBuffer nonUbBuffer(buffer);
    impl->commonRes_.bufferVec.push_back(&nonUbBuffer);

    u64 seg = 0xFF;
    void* addr = reinterpret_cast<void*>(0x725100U);
    EXPECT_EQ(impl->GetLocSeg(addr, 0x100, &seg), HCCL_E_PTR);
    EXPECT_EQ(seg, 0xFFU);
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_ChannelFence_NoWqe_Expect_Success)
{
    MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle_));
    MOCKER(RaSocketSetWhiteListStatus).stubs().will(returnValue(0));

    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    ASSERT_EQ(impl->Init(), HCCL_SUCCESS);

    // wqeNum_ is 0, ChannelFence should return success immediately
    EXPECT_EQ(impl->ChannelFence(), HCCL_SUCCESS);
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_Init_WithExchangeAllMems_Expect_Success)
{
    channelDesc.exchangeAllMems = true;
    MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle_));
    MOCKER(RaSocketSetWhiteListStatus).stubs().will(returnValue(0));
    MOCKER(HcommMemGetAllMemHandles).stubs().will(invoke(StubHostGetAllMemHandlesOne));

    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    ASSERT_EQ(impl->Init(), HCCL_SUCCESS);
    ASSERT_EQ(impl->commonRes_.bufferVec.size(), 1U);
    EXPECT_EQ(impl->commonRes_.bufferVec[0]->GetAddr(), 0x730000U);
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_ChannelFence_HasWqe_PollFails_Expect_NetworkError)
{
    MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle_));
    MOCKER(RaSocketSetWhiteListStatus).stubs().will(returnValue(0));

    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    ASSERT_EQ(impl->Init(), HCCL_SUCCESS);

    // wqeNum_ > 0 跳过 line 354 的提前返回，使 line 358 的超时时间计算得以执行
    impl->wqeNum_ = 1;
    // Mock HrtUrmaPollJfc 返回负数（错误），触发 line 362-365 快速返回 HCCL_E_NETWORK
    MOCKER(HrtUrmaPollJfc).stubs().will(returnValue(-1));

    HcclResult ret = impl->ChannelFence();
    EXPECT_EQ(ret, HCCL_E_NETWORK);
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_ChannelDrain_NoWqe_Expect_SuccessWithoutFenceFlag)
{
    MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle_));
    MOCKER(RaSocketSetWhiteListStatus).stubs().will(returnValue(0));

    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    ASSERT_EQ(impl->Init(), HCCL_SUCCESS);

    EXPECT_EQ(impl->ChannelDrain(), HCCL_SUCCESS);
    EXPECT_EQ(impl->wqeNum_, 0U);
    EXPECT_FALSE(impl->fenceFlag_);
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_ChannelDrain_HasWqe_Expect_PollAllWithoutFenceFlag)
{
    MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle_));
    MOCKER(RaSocketSetWhiteListStatus).stubs().will(returnValue(0));

    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    ASSERT_EQ(impl->Init(), HCCL_SUCCESS);

    impl->wqeNum_ = 2;
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    MOCKER_CPP(&Hccl::EnvRtsConfig::GetExecTimeOut).stubs().will(returnValue(static_cast<uint32_t>(1)));
    MOCKER(HrtUrmaPollJfc).stubs().will(returnValue(1)).then(returnValue(1));

    EXPECT_EQ(impl->ChannelDrain(), HCCL_SUCCESS);
    EXPECT_EQ(impl->wqeNum_, 0U);
    EXPECT_FALSE(impl->fenceFlag_);
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_ChannelFence_HasWqe_Expect_PollAllAndSetFenceFlag)
{
    MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle_));
    MOCKER(RaSocketSetWhiteListStatus).stubs().will(returnValue(0));

    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    ASSERT_EQ(impl->Init(), HCCL_SUCCESS);

    impl->wqeNum_ = 1;
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    MOCKER_CPP(&Hccl::EnvRtsConfig::GetExecTimeOut).stubs().will(returnValue(static_cast<uint32_t>(1)));
    MOCKER(HrtUrmaPollJfc).stubs().will(returnValue(1));

    EXPECT_EQ(impl->ChannelFence(), HCCL_SUCCESS);
    EXPECT_EQ(impl->wqeNum_, 0U);
    EXPECT_TRUE(impl->fenceFlag_);
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_ChannelDrain_PollFails_Expect_NetworkError)
{
    MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle_));
    MOCKER(RaSocketSetWhiteListStatus).stubs().will(returnValue(0));

    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    ASSERT_EQ(impl->Init(), HCCL_SUCCESS);

    impl->wqeNum_ = 1;
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    MOCKER_CPP(&Hccl::EnvRtsConfig::GetExecTimeOut).stubs().will(returnValue(static_cast<uint32_t>(1)));
    MOCKER(HrtUrmaPollJfc).stubs().will(returnValue(-1));

    EXPECT_EQ(impl->ChannelDrain(), HCCL_E_NETWORK);
    EXPECT_EQ(impl->wqeNum_, 1U);
    EXPECT_FALSE(impl->fenceFlag_);
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_ChannelDrain_PollExcessCqe_Expect_InternalError)
{
    MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle_));
    MOCKER(RaSocketSetWhiteListStatus).stubs().will(returnValue(0));

    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    ASSERT_EQ(impl->Init(), HCCL_SUCCESS);

    impl->wqeNum_ = 1;
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    MOCKER_CPP(&Hccl::EnvRtsConfig::GetExecTimeOut).stubs().will(returnValue(static_cast<uint32_t>(1)));
    MOCKER(HrtUrmaPollJfc).stubs().will(returnValue(2));

    EXPECT_EQ(impl->ChannelDrain(), HCCL_E_INTERNAL);
}

TEST_F(HostCpuUrmaChannelTest, Ut_When_ChannelDrain_NoCqeUntilTimeout_Expect_Timeout)
{
    MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle_));
    MOCKER(RaSocketSetWhiteListStatus).stubs().will(returnValue(0));

    auto impl = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc);
    ASSERT_EQ(impl->Init(), HCCL_SUCCESS);

    impl->wqeNum_ = 1;
    MOCKER(HrtUrmaPollJfc).stubs().will(returnValue(0));
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    MOCKER_CPP(&Hccl::EnvRtsConfig::GetExecTimeOut).stubs().will(returnValue(static_cast<uint32_t>(0)));

    EXPECT_EQ(impl->ChannelDrain(), HCCL_E_TIMEOUT);
}
