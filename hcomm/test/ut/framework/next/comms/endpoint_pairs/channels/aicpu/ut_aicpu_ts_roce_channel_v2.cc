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
#include <vector>
#include <iostream>
#include <cstring>
#include "aicpu_ts_roce_channel_v2.h"
#include "../sockets/socket_mgr.h"
#include "aicpu/dev_rdma_connection_v2.h"
#include "socket.h"
#include "orion_adapter_hccp.h"
#include "hccp.h"
#include "hccp_common.h"
#include "hcomm_c_adpt.h"
#include "buffer/local_rdma_rma_buffer_v2.h"
#include "rdma_handle_manager.h"
#include "cpu_roce_endpoint.h"
#include "binary_stream.h"
#include "mem_device_pub.h"
#include "aicpu_res_package_helper.h"
#include "exchange_rdma_buffer_dto.h"
#include "channels/host/exchange_rdma_conn_dto.h"
#include "acl/acl_rt.h"

#define private public

using namespace hcomm;

class AicpuTsRoceChannelV2Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AicpuTsRoceChannelV2Test tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AicpuTsRoceChannelV2Test tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in AicpuTsRoceChannelV2Test SetUP" << std::endl;
        Hccl::DevType dev = Hccl::DevType::DEV_TYPE_950;
        MOCKER(Hccl::HrtGetDevice).stubs().will(returnValue(0));
        MOCKER(Hccl::HrtGetDeviceType).stubs().will(returnValue(dev));
        MOCKER(Hccl::HrtGetDevicePhyIdByUserDevId)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(static_cast<Hccl::DevId>(0)));
        RdmaHandle rdmaHandle = (void*)0x1000000;
        MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));

        Hccl::IpAddress localIp("1.0.0.0");
        Hccl::IpAddress remoteIp("2.0.0.0");
        fakeSocket = new Hccl::Socket(
            nullptr, localIp, 60001, remoteIp, "_0_1_", Hccl::SocketRole::SERVER, Hccl::NicType::HOST_NIC_TYPE);

        MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
        MOCKER(Hccl::HrtRaNdaQpCreate)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
        MOCKER(Hccl::HrtRaNdaCqCreate)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
        MOCKER(Hccl::HrtRaNdaCqDestroy).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER(Hccl::HrtRaQpDestroy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER(RaGetQpAttr).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
        MOCKER_CPP(&AicpuTsRoceChannelV2::BuildNotifyValueBuffer).stubs().will(returnValue(HCCL_SUCCESS));

        EndpointDesc endpointDesc{};
        endpointDesc.protocol = COMM_PROTOCOL_ROCE;
        endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
        endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
        endpoint = std::make_unique<CpuRoceEndpoint>(endpointDesc);
        endpoint->Init();
        endpointHandle = static_cast<EndpointHandle>(endpoint.get());

        EndpointDesc endpointDesc2;
        endpointDesc2.protocol = COMM_PROTOCOL_ROCE;
        endpointDesc2.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        endpointDesc2.commAddr.addr = remoteIp.GetBinaryAddress().addr;
        endpointDesc2.loc.locType = ENDPOINT_LOC_TYPE_HOST;
        channelDesc.remoteEndpoint = endpointDesc2;
        channelDesc.notifyNum = 3;
        channelDesc.port = 60001;
        void* fsocket = static_cast<void*>(fakeSocket);
        channelDesc.socket = fsocket;

        localBufferPtr = std::make_shared<Hccl::Buffer>(666);
        localRdmaRmaBuffer = std::make_shared<Hccl::LocalRdmaRmaBuffer>(localBufferPtr, rdmaHandle);
        void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
        channelDesc.memHandles = &memHandle;
        channelDesc.memHandleNum = 1;
        channelDesc.exchangeAllMems = false;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in AicpuTsRoceChannelV2Test TearDown" << std::endl;
        delete fakeSocket;
    }

    std::shared_ptr<Hccl::Buffer> localBufferPtr;
    std::shared_ptr<Hccl::LocalRdmaRmaBuffer> localRdmaRmaBuffer;
    std::unique_ptr<CpuRoceEndpoint> endpoint;
    EndpointHandle endpointHandle{};
    HcommChannelDesc channelDesc{};
    Hccl::Socket* fakeSocket;
};

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_Normal_Init_Expect_HCCL_SUCCESS)
{
    std::cout << "Start Ut_When_Normal_Init_Expect_HCCL_SUCCESS" << std::endl;

    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&DevRdmaConnectionV2::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&DevRdmaConnectionV2::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsRoceChannelV2::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsRoceChannelV2::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsRoceChannelV2::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsRoceChannelV2::NotifyVecUnpack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsRoceChannelV2::ConnVecUnpackProc).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsRoceChannelV2::RmtBufferVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    // construct
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    // Init
    EXPECT_EQ(channel->Init(), HCCL_SUCCESS);
    // connect
    hcomm::ChannelStatus status = channel->GetStatus();
    EXPECT_EQ(channel->rdmaStatus_, AicpuTsRoceChannelV2::RdmaStatus::SOCKET_OK);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = channel->GetStatus();
    EXPECT_EQ(channel->rdmaStatus_, AicpuTsRoceChannelV2::RdmaStatus::QP_CREATED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = channel->GetStatus();
    EXPECT_EQ(channel->rdmaStatus_, AicpuTsRoceChannelV2::RdmaStatus::DATA_EXCHANGE);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = channel->GetStatus();
    EXPECT_EQ(channel->rdmaStatus_, AicpuTsRoceChannelV2::RdmaStatus::CONN_OK);
    EXPECT_EQ(status, ChannelStatus::READY);

    std::cout << "End Ut_When_Normal_Init_Expect_HCCL_SUCCESS" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_GetNotifyNum_Expect_Correct)
{
    std::cout << "Start Ut_When_GetNotifyNum_Expect_Correct" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();

    uint32_t notifyNum = 0;
    HcclResult ret = channel->GetNotifyNum(&notifyNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(notifyNum, channelDesc.notifyNum);

    std::cout << "End Ut_When_GetNotifyNum_Expect_Correct" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_GetBufferNum_Expect_Correct)
{
    std::cout << "Start Ut_When_GetBufferNum_Expect_Correct" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();

    uint32_t bufferNum = 0;
    HcclResult ret = channel->GetBufferNum(&bufferNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_GetBufferNum_Expect_Correct" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_Describe_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_Describe_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();

    std::string desc = channel->Describe();
    std::cout << desc << std::endl;

    std::cout << "End Ut_When_Describe_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_GetQpNum_Expect_Success)
{
    std::cout << "Start Ut_When_GetQpNum_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();

    uint32_t qpNum = 0;
    HcclResult ret = channel->GetQpNum(&qpNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_GetQpNum_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_BuildAndGetLocNotifyInfo_Expect_Success)
{
    std::cout << "Start Ut_When_BuildAndGetLocNotifyInfo_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;

    RegedNotifyEntity* notify = nullptr;
    HcclResult ret = channel->BuildAndGetLocNotifyInfo(&notify);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_BuildAndGetLocNotifyInfo_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_BuildAndGetRmtNotifyInfo_Expect_Success)
{
    std::cout << "Start Ut_When_BuildAndGetRmtNotifyInfo_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;

    RegedNotifyEntity* notify = nullptr;
    HcclResult ret = channel->BuildAndGetRmtNotifyInfo(&notify);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_BuildAndGetRmtNotifyInfo_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_BuildAndGetLocBufInfo_Expect_Success)
{
    std::cout << "Start Ut_When_BuildAndGetLocBufInfo_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;

    std::vector<RegedBufferEntity> locBufList;
    locBufList.resize(channel->bufferNum_);
    RegedBufferEntity* bufferEntity = nullptr;
    HcclResult ret = channel->BuildAndGetLocBufInfo(locBufList, &bufferEntity);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_BuildAndGetLocBufInfo_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_BuildAndGetRmtBufInfo_Expect_Success)
{
    std::cout << "Start Ut_When_BuildAndGetRmtBufInfo_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    std::vector<RegedBufferEntity> rmtBufList;
    rmtBufList.resize(channel->bufferNum_);
    RegedBufferEntity* bufferEntity = nullptr;
    HcclResult ret = channel->BuildAndGetRmtBufInfo(rmtBufList, &bufferEntity);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_BuildAndGetRmtBufInfo_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_BuildAndGetSqContext_Expect_Success)
{
    std::cout << "Start Ut_When_BuildAndGetSqContext_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;

    std::vector<SqContext> sqList;
    sqList.resize(channel->connNum_);
    SqContext* sqContext = nullptr;
    HcclResult ret = channel->BuildAndGetSqContext(sqList, &sqContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_BuildAndGetSqContext_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_BuildAndGetCqContext_Expect_Success)
{
    std::cout << "Start Ut_When_BuildAndGetCqContext_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;

    std::vector<CqContext> cqList;
    cqList.resize(channel->connNum_);
    CqContext* cqContext = nullptr;
    HcclResult ret = channel->BuildAndGetCqContext(cqList, &cqContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::cout << "End Ut_When_BuildAndGetCqContext_Expect_Success" << std::endl;
}

static hccl::DeviceMem StubDeviceMemAlloc(u64 size, bool)
{
    void* p = std::malloc(static_cast<size_t>(size));
    return hccl::DeviceMem(p, size, false);
}

static void StubHrtMemSyncCopy(void* dst, uint64_t dstMax, const void* src, uint64_t count, rtMemcpyKind_t)
{
    if (dst != nullptr && src != nullptr && dstMax >= count) {
        (void)memcpy(dst, src, static_cast<size_t>(count));
    }
}

static bool g_slabAllocated = false;
static void* g_slabPtr = nullptr;
static size_t g_slabSize = 0;

aclError StubAclrtMalloc(void** devPtr, size_t size, aclrtMemMallocPolicy, aclrtMallocConfig*)
{
    if (devPtr == nullptr) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    *devPtr = std::malloc(size);
    if (*devPtr == nullptr) {
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }
    g_slabAllocated = true;
    g_slabPtr = *devPtr;
    g_slabSize = size;
    return ACL_SUCCESS;
}

aclError StubAclrtMallocFail(void** devPtr, size_t, aclrtMemMallocPolicy, aclrtMallocConfig*)
{
    if (devPtr != nullptr) {
        *devPtr = nullptr;
    }
    return ACL_ERROR_RT_MEMORY_ALLOCATION;
}

aclError StubAclrtFree(void* devPtr)
{
    std::free(devPtr);
    if (devPtr == g_slabPtr) {
        g_slabAllocated = false;
        g_slabPtr = nullptr;
        g_slabSize = 0;
    }
    return ACL_SUCCESS;
}

static void StubHrtMemcpyVoid(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind) {}

static void StubHrtMemcpyReal(void* dst, uint64_t dstMax, const void* src, uint64_t count, Hccl::tagRtMemcpyKind)
{
    if (dst != nullptr && src != nullptr && dstMax >= count) {
        (void)std::memcpy(dst, src, static_cast<size_t>(count));
    }
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_BuildSocket_NullSocket_Expect_Success)
{
    std::cout << "Start Ut_When_BuildSocket_NullSocket_Expect_Success" << std::endl;
    channelDesc.socket = nullptr;
    channelDesc.port = 0;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    MOCKER_CPP(&hcomm::SocketMgr::GetSocket)
        .stubs()
        .with(mockcpp::any(), outBound(fakeSocket))
        .will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(channel->Init(), HCCL_SUCCESS);
    std::cout << "End Ut_When_BuildSocket_NullSocket_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_ModifyQp_Expect_Success)
{
    std::cout << "Start Ut_When_ModifyQp_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    MOCKER_CPP(&DevRdmaConnectionV2::ParseRmtExchangeDto).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&DevRdmaConnectionV2::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    HcclResult ret = channel->ModifyQp();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::cout << "End Ut_When_ModifyQp_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_BuildAndGetDevChannelEntity_Expect_Success)
{
    std::cout << "Start Ut_When_BuildAndGetDevChannelEntity_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyReal)));

    uint64_t devChannelEntityPtr = 0;
    HcclResult ret = channel->BuildAndGetDevChannelEntity(&devChannelEntityPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(devChannelEntityPtr, 0);
    channel->FreeDeviceMemories();
    std::cout << "End Ut_When_BuildAndGetDevChannelEntity_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_BuildSlab_Expect_Success)
{
    std::cout << "Start Ut_When_BuildSlab_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyReal)));

    g_slabAllocated = false;
    uint64_t devChannelEntityPtr = 0;
    HcclResult ret = channel->BuildAndGetDevChannelEntity(&devChannelEntityPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(devChannelEntityPtr, 0);
    EXPECT_TRUE(g_slabAllocated);
    EXPECT_NE(channel->devChannelEntitySlab_, nullptr);
    EXPECT_GT(channel->devChannelEntitySlabSize_, sizeof(ChannelEntity));

    ChannelEntity* devEntity = reinterpret_cast<ChannelEntity*>(devChannelEntityPtr);
    EXPECT_EQ(devEntity->engine, static_cast<uint32_t>(COMM_ENGINE_AICPU));
    EXPECT_EQ(devEntity->sqNum, 1u);
    EXPECT_EQ(devEntity->cqNum, 1u);

    channel->FreeDeviceMemories();
    std::cout << "End Ut_When_BuildSlab_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_BuildSlab_CachedReturn_Expect_SamePtr)
{
    std::cout << "Start Ut_When_BuildSlab_CachedReturn_Expect_SamePtr" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyReal)));

    uint64_t firstPtr = 0;
    HcclResult ret = channel->BuildAndGetDevChannelEntity(&firstPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(firstPtr, 0);

    // Second call should return same pointer without re-allocating
    g_slabAllocated = false;
    uint64_t secondPtr = 0;
    ret = channel->BuildAndGetDevChannelEntity(&secondPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(secondPtr, firstPtr);
    EXPECT_FALSE(g_slabAllocated);

    channel->FreeDeviceMemories();
    std::cout << "End Ut_When_BuildSlab_CachedReturn_Expect_SamePtr" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_BuildSlab_AclrtMallocFail_Expect_MemoryError)
{
    std::cout << "Start Ut_When_BuildSlab_AclrtMallocFail_Expect_MemoryError" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMallocFail));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyVoid)));

    uint64_t devChannelEntityPtr = 0;
    HcclResult ret = channel->BuildAndGetDevChannelEntity(&devChannelEntityPtr);
    EXPECT_EQ(ret, HCCL_E_MEMORY);
    EXPECT_EQ(devChannelEntityPtr, 0);
    EXPECT_EQ(channel->devChannelEntitySlab_, nullptr);
    EXPECT_EQ(channel->devChannelEntitySlabSize_, 0);

    std::cout << "End Ut_When_BuildSlab_AclrtMallocFail_Expect_MemoryError" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_Clean_FreesSlab_CanRebuild)
{
    std::cout << "Start Ut_When_Clean_FreesSlab_CanRebuild" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyReal)));

    // First build
    uint64_t firstPtr = 0;
    HcclResult ret = channel->BuildAndGetDevChannelEntity(&firstPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(channel->devChannelEntitySlab_, nullptr);

    // Clean should free the slab
    channel->Clean();
    EXPECT_EQ(channel->devChannelEntitySlab_, nullptr);
    EXPECT_EQ(channel->devChannelEntitySlabSize_, 0);

    // Rebuild should succeed
    g_slabAllocated = false;
    uint64_t secondPtr = 0;
    ret = channel->BuildAndGetDevChannelEntity(&secondPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(secondPtr, 0);
    EXPECT_TRUE(g_slabAllocated);

    channel->FreeDeviceMemories();
    std::cout << "End Ut_When_Clean_FreesSlab_CanRebuild" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_GetRemoteMem_NonEmpty_Expect_Success)
{
    std::cout << "Start Ut_When_GetRemoteMem_NonEmpty_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));
    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    char** memInfos = nullptr;
    HcclResult ret = channel->GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memNum, 1u);
    std::cout << "End Ut_When_GetRemoteMems_NonEmpty_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_GetRemoteMems_CacheValid_Expect_Success)
{
    std::cout << "Start Ut_When_GetRemoteMems_CacheValid_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x2000000));
    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = channel->GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = channel->GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::cout << "End Ut_When_GetRemoteMems_CacheValid_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_GetRemoteMems_NullPtr_Expect_Error)
{
    std::cout << "Start Ut_When_GetRemoteMems_NullPtr_Expect_Error" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    char** memInfos = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = channel->GetRemoteMems(&memNum, nullptr, &memInfos);
    EXPECT_NE(ret, HCCL_SUCCESS);
    CommMem* remoteMem = nullptr;
    ret = channel->GetRemoteMems(&memNum, &remoteMem, nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
    ret = channel->GetRemoteMems(nullptr, &remoteMem, &memInfos);
    EXPECT_NE(ret, HCCL_SUCCESS);
    std::cout << "End Ut_When_GetRemoteMems_NullPtr_Expect_Error" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_GetRemoteMems_Empty_Expect_Success)
{
    std::cout << "Start Ut_When_GetRemoteMems_Empty_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->rmtRmaBuffers_.clear();
    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = channel->GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memNum, 0U);
    std::cout << "End Ut_When_GetRemoteMems_Empty_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_GetLocalNotifyUniqueIds_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetLocalNotifyUniqueIds_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    auto result = channel->GetLocalNotifyUniqueIds();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetLocalNotifyUniqueIds_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_GetRemoteNotifyUniqueIds_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetRemoteNotifyUniqueIds_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->remoteNotifies_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));
    auto result = channel->GetRemoteNotifyUniqueIds();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetRemoteNotifyUniqueIds_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_GetNotifyValueBufferUniqueIds_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetNotifyValueBufferUniqueIds_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    // BuildNotifyValueBuffer is mocked in SetUp, so create notifyValueBuffer_ here
    channel->notifyValueBuffer_ = std::make_unique<Hccl::LocalRdmaRmaBuffer>(localBufferPtr, (void*)0x1000000);
    auto result = channel->GetNotifyValueBufferUniqueIds();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetNotifyValueBufferUniqueIds_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_GetLocBufferUniqueIds_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetLocBufferUniqueIds_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    auto result = channel->GetLocBufferUniqueIds();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetLocBufferUniqueIds_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_GetLocBufferUniqueIds_NullBuffer_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetLocBufferUniqueIds_NullBuffer_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->localRmaBuffers_.push_back(nullptr);
    channel->bufferNum_ = channel->localRmaBuffers_.size();
    auto result = channel->GetLocBufferUniqueIds();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetLocBufferUniqueIds_NullBuffer_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_GetRmtBufferUniqueIds_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetRmtBufferUniqueIds_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));
    auto result = channel->GetRmtBufferUniqueIds();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetRmtBufferUniqueIds_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_GetRmtBufferUniqueIds_NullBuffer_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetRmtBufferUniqueIds_NullBuffer_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->rmtRmaBuffers_.push_back(nullptr);
    auto result = channel->GetRmtBufferUniqueIds();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetRmtBufferUniqueIds_NullBuffer_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_GetConnUniqueIds_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetConnUniqueIds_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    auto result = channel->GetConnUniqueIds();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetConnUniqueIds_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_GetUniqueId_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_GetUniqueId_Expect_NotEmpty" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));
    // BuildNotifyValueBuffer is mocked in SetUp, so create notifyValueBuffer_ here
    channel->notifyValueBuffer_ = std::make_unique<Hccl::LocalRdmaRmaBuffer>(localBufferPtr, (void*)0x1000000);
    auto result = channel->GetUniqueId();
    EXPECT_GT(result.size(), 0u);
    std::cout << "End Ut_When_GetUniqueId_Expect_NotEmpty" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_GetChannelKind_Expect_RoceV2)
{
    std::cout << "Start Ut_When_GetChannelKind_Expect_RoceV2" << std::endl;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    EXPECT_EQ(channel->GetChannelKind(), HcommChannelKind::AICPU_TS_ROCE_V2);
    std::cout << "End Ut_When_GetChannelKind_Expect_RoceV2" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_Clean_Expect_Success)
{
    std::cout << "Start Ut_When_Clean_Expect_Success" << std::endl;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    EXPECT_EQ(channel->Clean(), HCCL_SUCCESS);
    std::cout << "End Ut_When_Clean_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_Resume_Expect_Success)
{
    std::cout << "Start Ut_When_Resume_Expect_Success" << std::endl;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    EXPECT_EQ(channel->Resume(), HCCL_SUCCESS);
    std::cout << "End Ut_When_Resume_Expect_Success" << std::endl;
}

TEST_F(AicpuTsRoceChannelV2Test, Ut_When_H2DResPack_Expect_Success)
{
    std::cout << "Start Ut_When_H2DResPack_Expect_Success" << std::endl;
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));
    // BuildNotifyValueBuffer is mocked in SetUp, so create notifyValueBuffer_ here
    channel->notifyValueBuffer_ = std::make_unique<Hccl::LocalRdmaRmaBuffer>(localBufferPtr, (void*)0x1000000);
    std::vector<char> buffer;
    HcclResult ret = channel->H2DResPack(buffer);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::cout << "End Ut_When_H2DResPack_Expect_Success" << std::endl;
}

// ===================== PreAllocDevChannelEntity / FillDevChannelEntity UT =====================

// PreAllocDevChannelEntity: devChannelEntityPtr 为 nullptr
TEST_F(AicpuTsRoceChannelV2Test, Ut_PreAllocDevChannelEntity_When_PtrNull_Returns_E_PTR)
{
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    EXPECT_EQ(channel->PreAllocDevChannelEntity(nullptr), HCCL_E_PTR);
}

// FillDevChannelEntity: 未预分配（slab 为 nullptr）→ E_INTERNAL
TEST_F(AicpuTsRoceChannelV2Test, Ut_FillDevChannelEntity_When_NotPreAllocated_Returns_E_INTERNAL)
{
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    EXPECT_EQ(channel->FillDevChannelEntity(), HCCL_E_INTERNAL);
}

// FillDevChannelEntity: 已预分配但 channel 未 READY → E_INTERNAL
TEST_F(AicpuTsRoceChannelV2Test, Ut_FillDevChannelEntity_When_NotReady_Returns_E_INTERNAL)
{
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->channelStatus_ = ChannelStatus::INIT;
    // 模拟已预分配 slab（设置一个非空指针）
    channel->devChannelEntitySlab_ = reinterpret_cast<void*>(0x1000);
    channel->devChannelEntitySlabSize_ = 4096;
    EXPECT_EQ(channel->FillDevChannelEntity(), HCCL_E_INTERNAL);
    channel->devChannelEntitySlab_ = nullptr;
}

// ===================== PreAllocDevChannelEntity / FillDevChannelEntity 正常路径 UT =====================

// PreAllocDevChannelEntity: 正常分配 slab
TEST_F(AicpuTsRoceChannelV2Test, Ut_PreAllocDevChannelEntity_When_Normal_ReturnsSuccess)
{
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyReal)));

    uint64_t devChannelEntityPtr = 0;
    EXPECT_EQ(channel->PreAllocDevChannelEntity(&devChannelEntityPtr), HCCL_SUCCESS);
    EXPECT_NE(devChannelEntityPtr, 0);
    channel->FreeDeviceMemories();
}

// PreAllocDevChannelEntity: 已分配则返回缓存指针
TEST_F(AicpuTsRoceChannelV2Test, Ut_PreAllocDevChannelEntity_When_AlreadyBuilt_ReturnsCached)
{
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyReal)));

    uint64_t firstPtr = 0;
    ASSERT_EQ(channel->PreAllocDevChannelEntity(&firstPtr), HCCL_SUCCESS);
    EXPECT_NE(firstPtr, 0);
    uint64_t secondPtr = 0;
    EXPECT_EQ(channel->PreAllocDevChannelEntity(&secondPtr), HCCL_SUCCESS);
    EXPECT_EQ(secondPtr, firstPtr);
    channel->FreeDeviceMemories();
}

// FillDevChannelEntity: 先 PreAlloc 再 Fill，验证成功
TEST_F(AicpuTsRoceChannelV2Test, Ut_FillDevChannelEntity_When_PreAllocatedAndReady_ReturnsSuccess)
{
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto channel = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, CommEngine::COMM_ENGINE_AICPU);
    channel->Init();
    channel->channelStatus_ = ChannelStatus::READY;
    channel->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((void*)0x1000000));

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyReal)));

    uint64_t devChannelEntityPtr = 0;
    ASSERT_EQ(channel->PreAllocDevChannelEntity(&devChannelEntityPtr), HCCL_SUCCESS);
    EXPECT_EQ(channel->FillDevChannelEntity(), HCCL_SUCCESS);
    channel->FreeDeviceMemories();
}
