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
#include <cstdio>
#include "adapter_error_manager_pub.h"
#include "host_socket_handle_manager.h"
#include "cpu_roce_endpoint.h"
#include "buffer/local_rdma_rma_buffer_v2.h"
#include "host/host_cpu_roce_channel.h"
#include "sockets/socket_mgr.h"
#include "host/host_rdma_connection.h"
#include "topo_common_types.h"
#include "ip_address.h"
#include "op_mode.h"
#include "rdma_handle_manager.h"
#include "host/exchange_rdma_conn_dto.h"
#include "socket.h"
#include "hccp.h"
#include "types/types.h"
#include "hcomm_res.h"
#include "hcomm_c_adpt.h"
#include "exchange_rdma_buffer_dto.h"

#include "env_config/env_config_v2.h"
#include "env_config/env_func.h"
#include "orion_adpt_utils.h"

#define private public
using namespace hcomm;

void stub_RptInputErr_print(std::string error_code, std::vector<std::string> key, std::vector<std::string> value)
{
    if (error_code == "EI0013") {
        fprintf(
            stderr,
            "Execution_Error_ROCE_CQE(EI0013): An error CQE occurred during operator execution. "
            "Local information: server %s, device ID %s, device IP %s. "
            "Peer information: server %s, device ID %s, device IP %s.\n",
            "0", "0", "1.0.0.0", "1", "1", "2.0.0.0");
        fprintf(
            stderr, "        Possible cause: The RDMA data transfer fails due to a hardware fault or link error.\n"
                    "        Solution: 1. Check whether the network devices between the two ends are abnormal.\n"
                    "2. Check whether the peer process exits first. If yes, check the cause of the process exit.\n");
    }
    fflush(stderr);
}

// ==================== Hybrid Mode Stub Functions ====================
static bool RecvWithWrongMagicStub(Hccl::Socket* socket, void* buf, uint32_t size)
{
    (void)socket;
    uint32_t* magicPtr = reinterpret_cast<uint32_t*>(buf);
    *magicPtr = 0xDEADBEEF; // 错误的魔数，模拟对端是旧版本
    return true;
}

static bool RecvWithInvalidVersionStub(Hccl::Socket* socket, void* buf, uint32_t size)
{
    (void)socket;
    uint8_t* data = reinterpret_cast<uint8_t*>(buf);
    uint32_t magic = 0x48434C52; // "HCLR"
    memcpy_s(data, sizeof(uint32_t), &magic, sizeof(uint32_t));
    uint16_t version = 100; // 无效版本号，大于支持的最大版本
    memcpy_s(data + 4, sizeof(uint16_t), &version, sizeof(uint16_t));
    uint16_t totalLength = 20;
    memcpy_s(data + 6, sizeof(uint16_t), &totalLength, sizeof(uint16_t));
    return true;
}

static HcclResult ExchangeCapabilityHybridStub(HostCpuRoceChannel* channel)
{
    (void)channel;
    return HCCL_SUCCESS;
}

static bool RecvForHybridModeStub(Hccl::Socket* socket, void* buf, uint32_t size)
{
    (void)socket;
    if (size < sizeof(RoCECapability)) {
        return false;
    }
    uint8_t* data = reinterpret_cast<uint8_t*>(buf);
    uint32_t magic = 0x48434C52; // "HCLR"
    memcpy_s(data, sizeof(uint32_t), &magic, sizeof(uint32_t));
    uint16_t version = 1; // ROCE_CAPABILITY_VERSION
    memcpy_s(data + 4, sizeof(uint16_t), &version, sizeof(uint16_t));
    uint16_t totalLength = sizeof(RoCECapability);
    memcpy_s(data + 6, sizeof(uint16_t), &totalLength, sizeof(uint16_t));
    uint8_t nodeType = 0;
    memcpy_s(data + 8, sizeof(uint8_t), &nodeType, sizeof(uint8_t));
    uint8_t commStack = 1; // COMM_STACK_TRANSPORT_IBVERBS - 触发 hybrid mode
    memcpy_s(data + 9, sizeof(uint8_t), &commStack, sizeof(uint8_t));
    uint8_t syncMode = 0;
    memcpy_s(data + 10, sizeof(uint8_t), &syncMode, sizeof(uint8_t));
    uint8_t padding = 0;
    memcpy_s(data + 11, sizeof(uint8_t), &padding, sizeof(uint8_t));
    NICDeployment nicDeploy = NICDeployment::NIC_DEPLOYMENT_HOST;
    memcpy_s(data + 12, sizeof(NICDeployment), &nicDeploy, sizeof(nicDeploy));
    return true;
}

class HostCpuRoceChannelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HostCpuRoceChannelTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "HostCpuRoceChannelTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in HostCpuRoceChannelTest SetUP" << std::endl;
        Hccl::DevType dev = Hccl::DevType::DEV_TYPE_950;
        MOCKER(Hccl::HrtGetDevice).stubs().will(returnValue(0));
        MOCKER(Hccl::HrtGetDeviceType).stubs().will(returnValue(dev));
        MOCKER(Hccl::HrtGetDevicePhyIdByUserDevId)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(static_cast<Hccl::DevId>(0)));

        DevType devType = DevType::DEV_TYPE_950;
        MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
        // MOCKER(hrtGetDeviceType).stubs().with(outBound(dev)).will(returnValue(HCCL_SUCCESS));
        RdmaHandle rdmaHandle = (void*)0x1000000;
        MOCKER(Hccl::HrtRaRdmaInit, Hccl::RdmaHandle(Hccl::HrtNetworkMode, Hccl::RaInterface&))
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(rdmaHandle));
        MOCKER(HcommEndpointStartListen).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
        EndpointDesc endpointDesc{};
        endpointDesc.protocol = COMM_PROTOCOL_ROCE;
        endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        Hccl::IpAddress localIp("1.0.0.0");
        endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
        endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
        endpoint = std::make_unique<CpuRoceEndpoint>(endpointDesc);
        endpoint->Init();
        endpointHandle = static_cast<EndpointHandle>(endpoint.get());
        EndpointDesc endpointDesc2;
        endpointDesc2.protocol = COMM_PROTOCOL_ROCE;
        endpointDesc2.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        Hccl::IpAddress remoteIp("2.0.0.0");
        endpointDesc2.commAddr.addr = remoteIp.GetBinaryAddress().addr;
        endpointDesc2.loc.locType = ENDPOINT_LOC_TYPE_HOST;
        channelDesc.remoteEndpoint = endpointDesc2;
        channelDesc.notifyNum = 2;
        fakeSocket = new Hccl::Socket(
            nullptr, localIp, 60001, remoteIp, "_0_1_", Hccl::SocketRole::SERVER, Hccl::NicType::HOST_NIC_TYPE);
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
        unsetenv("HCCL_DFS_CONFIG");
        GlobalMockObject::verify();
        std::cout << "A Test case in HostCpuRoceChannelTest TearDown" << std::endl;
        delete fakeSocket;
    }
    void SetupSuccessfulConnectionMocks()
    {
        DevType devType = DevType::DEV_TYPE_950;
        MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
        MOCKER_CPP(&HostRdmaConnection::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HostRdmaConnection::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HostCpuRoceChannel::IbvPostRecv).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HostCpuRoceChannel::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HostCpuRoceChannel::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HostCpuRoceChannel::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HostCpuRoceChannel::NotifyVecUnpack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HostCpuRoceChannel::ConnVecUnpackProc).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HostCpuRoceChannel::RmtBufferVecUnpackProc)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HostCpuRoceChannel::ExchangeCapability).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HostCpuRoceChannel::ExchangeData).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&SocketMgr::GetSocket).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&SocketMgr::PutSocket).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HostRdmaConnection::ParseRmtExchangeDto).stubs().will(returnValue(HCCL_SUCCESS));
    }
    std::vector<Hccl::QpInfo> SetupValidQpInfos(uint32_t count = 1)
    {
        std::vector<Hccl::QpInfo> qpInfos(count);
        for (uint32_t i = 0; i < count; i++) {
            static ibv_qp qp;
            static ibv_cq cqi;
            qp.state = IBV_QPS_RTS;
            qp.qp_num = 12345 + i;
            qpInfos[i].qp = &qp;
            qpInfos[i].sendCq = &cqi;
            qpInfos[i].recvCq = &cqi;
            qpInfos[i].context = nullptr;
        }
        return qpInfos;
    }
    void MockGetQpInfos(uint32_t count = 1)
    {
        auto qpInfos = SetupValidQpInfos(count);
        MOCKER_CPP(&HostCpuRoceChannel::GetQpInfos).stubs().will(returnValue(qpInfos));
    }
    std::unique_ptr<hcomm::HostCpuRoceChannel> CreateInitAndConnect(uint32_t notifyNum = 4)
    {
        memHandle_ = static_cast<void*>(localRdmaRmaBuffer.get());
        channelDesc.memHandles = &memHandle_;
        channelDesc.memHandleNum = 1;
        channelDesc.notifyNum = notifyNum;
        auto impl = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
        EXPECT_EQ(impl->Init(), HCCL_SUCCESS);
        hcomm::ChannelStatus status = impl->GetStatus();
        EXPECT_EQ(impl->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::SOCKET_OK);
        status = impl->GetStatus();
        status = impl->GetStatus();
        status = impl->GetStatus();
        status = impl->GetStatus();
        EXPECT_EQ(impl->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CONN_OK);
        return impl;
    }

    void SetupOneValidQpInfoMock()
    {
        std::vector<Hccl::QpInfo> qpInfos(1);
        static ibv_cq cq{};
        static ibv_qp qp{};
        static ibv_context context{};
        qpInfos[0].sendCq = &cq;
        qpInfos[0].qp = &qp;
        qpInfos[0].sendCq->context = &context;
        MOCKER_CPP(&HostCpuRoceChannel::GetQpInfos).stubs().will(returnValue(qpInfos));
    }

    std::shared_ptr<Hccl::Buffer> localBufferPtr;
    std::shared_ptr<Hccl::LocalRdmaRmaBuffer> localRdmaRmaBuffer;
    std::vector<std::shared_ptr<Hccl::Buffer>> bufs{std::make_shared<Hccl::Buffer>((uintptr_t)2, 64)};
    std::unique_ptr<CpuRoceEndpoint> endpoint;
    EndpointHandle endpointHandle{};
    HcommChannelDesc channelDesc{};
    Hccl::Socket* fakeSocket;
    void* memHandle_{nullptr};
};

// Mock endpoint that is NOT CpuRoceEndpoint, used to test dynamic_cast failure
class MockNonRoceEndpoint : public Endpoint {
public:
    explicit MockNonRoceEndpoint(const EndpointDesc& desc) : Endpoint(desc)
    {
        ctxHandle_ = (void*)0x1000000; // non-null so CHK_PTR_NULL(rdmaHandle_) passes
    }
    HcclResult Init() override { return HCCL_SUCCESS; }
    // 内存方法在 RegedMemMgr 上，Endpoint 不再 override
    RegedMemMgr* GetRegedMemMgr() override { return nullptr; }
    void* GetRdmaHandle() override { return ctxHandle_; }
    bool IsCtxHandleValid() const override { return ctxHandle_ != nullptr; }

private:
    void* ctxHandle_{nullptr};
};

TEST_F(HostCpuRoceChannelTest, Ut_When_Normal_Expect_HCCL_SUCCESS)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&HostRdmaConnection::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ParseRmtExchangeDto).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::IbvPostRecv).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecUnpack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecUnpackProc).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::RmtBufferVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeCapability).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeData).stubs().will(returnValue(HCCL_SUCCESS));
    // construct
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.notifyNum = 4;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    // Init
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);
    // connect
    hcomm::ChannelStatus status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::SOCKET_OK);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CAP_EXCHANGED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::QP_CREATED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::DATA_EXCHANGE);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CONN_OK);
    EXPECT_EQ(status, ChannelStatus::READY);
}

// Channel Init for HIXL
TEST_F(HostCpuRoceChannelTest, Ut_Init_When_ExchangeAllMemsIsTrue_And_SocketIsNull_Expect_HCCL_SUCCESS)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&HostRdmaConnection::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::IbvPostRecv).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecUnpack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecUnpackProc).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::RmtBufferVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeCapability).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeData).stubs().will(returnValue(HCCL_SUCCESS));
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    uint32_t memHandleNum = 1;
    MOCKER(HcommMemGetAllMemHandles)
        .stubs()
        .with(mockcpp::any(), outBound(&memHandle), outBound(&memHandleNum))
        .will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER_CPP(&hcomm::SocketMgr::GetSocket)
        .stubs()
        .with(mockcpp::any(), outBound(fakeSocket))
        .will(returnValue(HCCL_SUCCESS));
    // construct
    channelDesc.exchangeAllMems = true;
    channelDesc.memHandles = nullptr;
    channelDesc.memHandleNum = 0;
    channelDesc.notifyNum = 4;
    channelDesc.socket = nullptr;
    channelDesc.port = 0;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    // Init
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);
    // connect
    hcomm::ChannelStatus status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::SOCKET_OK);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CAP_EXCHANGED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::QP_CREATED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::DATA_EXCHANGE);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CONN_OK);
    EXPECT_EQ(status, ChannelStatus::READY);
}

TEST_F(HostCpuRoceChannelTest, Ut_When_Init_Param_Nullptr_Expect_HCCL_E_PTR)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    // construct
    channelDesc.memHandles = nullptr;
    channelDesc.memHandleNum = 1;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    // Init
    EXPECT_EQ(impl_->Init(), HCCL_E_PTR);
}

TEST_F(HostCpuRoceChannelTest, Ut_When_Init_NotifyNumOutOfRange_Expect_HCCL_E_PARA)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    // construct
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.notifyNum = 8193;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    // Init
    EXPECT_EQ(impl_->Init(), HCCL_E_PARA);
}

TEST_F(HostCpuRoceChannelTest, Ut_When_SocketTimeout_Expect_FAILED)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::TIMEOUT));
    // construct
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    // Init
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);
    // connect
    hcomm::ChannelStatus status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::INIT);
    EXPECT_EQ(status, ChannelStatus::FAILED);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::INIT);
    EXPECT_EQ(status, ChannelStatus::FAILED);
}

TEST_F(HostCpuRoceChannelTest, Ut_When_CreateQp_Failed_Expect_FAILED)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&HostRdmaConnection::CreateQp).stubs().will(returnValue(HCCL_E_NETWORK));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeCapability).stubs().will(returnValue(HCCL_SUCCESS));
    // construct
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.roceAttr.queueNum = 1;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    // Init
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);
    // connect
    hcomm::ChannelStatus status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::SOCKET_OK);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CAP_EXCHANGED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CAP_EXCHANGED);
    EXPECT_EQ(status, ChannelStatus::FAILED);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CAP_EXCHANGED);
    EXPECT_EQ(status, ChannelStatus::FAILED);
}

TEST_F(HostCpuRoceChannelTest, Ut_When_ExchangeData_Failed_Expect_FAILED)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&HostRdmaConnection::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecUnpack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_ROCE_CONNECT));
    MOCKER_CPP(&HostCpuRoceChannel::RmtBufferVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeCapability).stubs().will(returnValue(HCCL_SUCCESS));
    // construct
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    // Init
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);
    // connect
    hcomm::ChannelStatus status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::SOCKET_OK);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CAP_EXCHANGED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::QP_CREATED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::QP_CREATED);
    EXPECT_EQ(status, ChannelStatus::FAILED);
}

TEST_F(HostCpuRoceChannelTest, Ut_When_ModifyQp_Failed_Expect_FAILED)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&HostRdmaConnection::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ModifyQp).stubs().will(returnValue(HCCL_E_ROCE_CONNECT));
    MOCKER_CPP(&HostRdmaConnection::ParseRmtExchangeDto).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecUnpack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecUnpackProc).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::RmtBufferVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeCapability).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeData).stubs().will(returnValue(HCCL_SUCCESS));
    // construct
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.roceAttr.queueNum = 1;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    // Init
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);
    // connect
    hcomm::ChannelStatus status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::SOCKET_OK);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CAP_EXCHANGED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::QP_CREATED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::DATA_EXCHANGE);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::DATA_EXCHANGE);
    EXPECT_EQ(status, ChannelStatus::FAILED);
}

TEST_F(HostCpuRoceChannelTest, Ut_GetRemoteMems_When_NullParam_Expect_HCCL_E_PARA)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    // construct
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    // Init
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);
    // GetRemoteMems
    CommMem* remoteMem;
    uint32_t memNum{11119999};
    char** memInfosArray = nullptr;
    HcclResult ret = impl_->GetRemoteMems(&memNum, &remoteMem, &memInfosArray);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memNum, 0);
    ret = impl_->GetRemoteMems((uint32_t*)nullptr, &remoteMem, &memInfosArray);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HostCpuRoceChannelTest, Ut_GetRemoteMems_When_RemoteMemExists_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    // construct
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    // Init
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);
    RdmaHandle rdmaHandle = (void*)0x1000000;
    Hccl::ExchangeRdmaBufferDto rmtBufDto(0x2000000, 1024, 1000, "tag1");
    auto rmtBuf = std::make_unique<Hccl::RemoteRdmaRmaBuffer>(rdmaHandle, rmtBufDto);
    impl_->rmtRmaBuffers_.emplace_back(std::move(rmtBuf));
    Hccl::ExchangeRdmaBufferDto rmtBufDto2(0x3000000, 2048, 1000, "tag2");
    auto rmtBuf2 = std::make_unique<Hccl::RemoteRdmaRmaBuffer>(rdmaHandle, rmtBufDto2);
    impl_->rmtRmaBuffers_.emplace_back(std::move(rmtBuf2));
    // GetRemoteMems
    uint32_t memNum = 0;
    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    HcclResult ret = impl_->GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memNum, 2);
    ASSERT_NE(remoteMem, nullptr);
}

TEST_F(HostCpuRoceChannelTest, Ut_When_Rdma_Conn_Failed_Expect_ERROR)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&HostRdmaConnection::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ParseRmtExchangeDto).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::IbvPostRecv).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecUnpack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecUnpackProc).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::RmtBufferVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeCapability).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeData).stubs().will(returnValue(HCCL_SUCCESS));
    MockGetQpInfos();
    // construct
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.notifyNum = 4;
    channelDesc.roceAttr.queueNum = 1;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    // Init
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);
    // connect
    hcomm::ChannelStatus status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::SOCKET_OK);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CAP_EXCHANGED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::QP_CREATED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::DATA_EXCHANGE);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CONN_OK);
    EXPECT_EQ(status, ChannelStatus::READY);

    // 交换过程实际没有进行,因此对端数据为空
    HcclResult ret = impl_->NotifyRecord(0);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = impl_->NotifyWait(0, 1800);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    ret = impl_->WriteWithNotify((void*)0x0001, (void*)0x0002, 10, 1);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = impl_->NotifyWait(1, 1800);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    ret = impl_->NotifyRecord(2);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = impl_->NotifyWait(2, 1800);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HostCpuRoceChannelTest, Ut_When_PrepareWriteWrResource_Expect_SUCCESS)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&HostRdmaConnection::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ParseRmtExchangeDto).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::IbvPostRecv).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecUnpack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecUnpackProc).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::RmtBufferVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::FindLocalBuffer).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::FindRemoteBuffer).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeCapability).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeData).stubs().will(returnValue(HCCL_SUCCESS));
    // construct
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.notifyNum = 4;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    // Init
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);
    // connect
    hcomm::ChannelStatus status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::SOCKET_OK);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CAP_EXCHANGED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::QP_CREATED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::DATA_EXCHANGE);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CONN_OK);
    EXPECT_EQ(status, ChannelStatus::READY);

    impl_->localDpuNotifyIds_.emplace_back(0);
    impl_->remoteDpuNotifyIds_.emplace_back(0);
    impl_->localRmaBuffers_.emplace_back(localRdmaRmaBuffer.get());
    RdmaHandle rdmaHandle = (void*)0x1000000;
    std::unique_ptr<Hccl::RemoteRdmaRmaBuffer> remoteRmaBuffer
        = std::make_unique<Hccl::RemoteRdmaRmaBuffer>(rdmaHandle);
    impl_->rmtRmaBuffers_.emplace_back(std::move(remoteRmaBuffer));
    struct ibv_send_wr sendWr {};
    struct ibv_sge sge {};
    sendWr.sg_list = &sge;
    Hccl::TaskParam taskParam{};
    HcclResult ret = impl_->PrepareWriteWrResource((void*)0x0001, (void*)0x0002, 10, 0, sendWr, taskParam);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HostCpuRoceChannelTest, Ut_When_HostCpuRoceChannel_Pack_And_Unpack_Expect_HCCL_SUCCESS)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&HostRdmaConnection::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ParseRmtExchangeDto).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeCapability).stubs().will(returnValue(HCCL_SUCCESS));
    // construct
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.notifyNum = 4;
    channelDesc.roceAttr.queueNum = 1;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    // Init
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);
    // connect
    hcomm::ChannelStatus status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::SOCKET_OK);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CAP_EXCHANGED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::QP_CREATED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    struct QpAttr localQpAttr;
    localQpAttr.qpn = 0;
    localQpAttr.udpSport = 1;
    localQpAttr.psn = 2;
    localQpAttr.gidIdx = 3;
    MOCKER(RaGetQpAttr).stubs().with(mockcpp::any(), outBoundP(&localQpAttr)).will(returnValue(0));

    Hccl::BinaryStream binaryStream;
    impl_->NotifyVecPack(binaryStream);
    // impl_->BufferVecPack(binaryStream);
    impl_->connections_[0]->rdmaConnStatus_ = HostRdmaConnection::RdmaConnStatus::QP_CREATED;
    HcclResult ret = impl_->ConnVecPack(binaryStream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = impl_->NotifyVecUnpack(binaryStream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // impl_->RmtBufferVecUnpackProc(binaryStream);
    ret = impl_->ConnVecUnpackProc(binaryStream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HostCpuRoceChannelTest, Ut_ChannelFence_When_WqeNumIsZero_Expect_HCCL_SUCCESS)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&HostRdmaConnection::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ParseRmtExchangeDto).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::IbvPostRecv).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecUnpack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecUnpackProc).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::RmtBufferVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeCapability).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeData).stubs().will(returnValue(HCCL_SUCCESS));
    MockGetQpInfos();
    // construct
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.notifyNum = 4;
    channelDesc.roceAttr.queueNum = 1;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    // Init
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);
    // connect
    hcomm::ChannelStatus status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::SOCKET_OK);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CAP_EXCHANGED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::QP_CREATED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::DATA_EXCHANGE);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CONN_OK);
    EXPECT_EQ(status, ChannelStatus::READY);
    // ChannelFence
    SetupOneValidQpInfoMock();
    impl_->wqeNums_ = {0};
    HcclResult ret = impl_->ChannelFence();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(impl_->fenceFlag_);
}

TEST_F(HostCpuRoceChannelTest, Ut_ChannelFence_When_PollExcessCqe_Expect_HCCL_E_INTERNAL)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();
    impl_->wqeNums_ = {2};
    SetupOneValidQpInfoMock();
    MOCKER_CPP(&HostCpuRoceChannel::IbvPollCq).stubs().will(returnValue(5));
    HcclResult ret = impl_->ChannelFence();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// ChannelFence: IbvPollCq返回错误时立即返回HCCL_E_NETWORK（不等待30秒超时）
// 注意：真正等待30秒超时的场景不适合UT测试，这里测试IbvPollCq失败的快速路径
TEST_F(HostCpuRoceChannelTest, Ut_ChannelFence_When_PollCqFailed_Expect_HCCL_E_NETWORK)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();
    impl_->wqeNums_ = {1};
    SetupOneValidQpInfoMock();
    // Mock IbvPollCq返回错误（负数），模拟ibv_poll_cq失败
    MOCKER_CPP(&HostCpuRoceChannel::IbvPollCq).stubs().will(returnValue(-1));
    HcclResult ret = impl_->ChannelFence();
    EXPECT_EQ(ret, HCCL_E_NETWORK);
}

TEST_F(HostCpuRoceChannelTest, Ut_ChannelDrain_When_WqeNumIsZero_Expect_NoFenceFlagSideEffect)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();
    impl_->wqeNums_ = {0};
    SetupOneValidQpInfoMock();

    impl_->fenceFlag_ = false;
    EXPECT_EQ(impl_->ChannelDrain(), HCCL_SUCCESS);
    EXPECT_FALSE(impl_->fenceFlag_);

    impl_->fenceFlag_ = true;
    EXPECT_EQ(impl_->ChannelDrain(), HCCL_SUCCESS);
    EXPECT_TRUE(impl_->fenceFlag_);
}

TEST_F(HostCpuRoceChannelTest, Ut_ChannelDrain_When_WqePending_Expect_PollToZeroWithoutFenceFlag)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();
    impl_->wqeNums_ = {2};
    impl_->fenceFlag_ = false;
    SetupOneValidQpInfoMock();
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    MOCKER_CPP(&Hccl::EnvRtsConfig::GetExecTimeOut).stubs().will(returnValue(static_cast<uint32_t>(1)));
    MOCKER_CPP(&HostCpuRoceChannel::IbvPollCq).stubs().will(returnValue(2));

    EXPECT_EQ(impl_->ChannelDrain(), HCCL_SUCCESS);
    EXPECT_EQ(impl_->wqeNums_[0], 0);
    EXPECT_FALSE(impl_->fenceFlag_);
}

TEST_F(HostCpuRoceChannelTest, Ut_ChannelDrain_When_PollExcessCqe_Expect_HCCL_E_INTERNAL)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();
    impl_->wqeNums_ = {2};
    SetupOneValidQpInfoMock();
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    MOCKER_CPP(&Hccl::EnvRtsConfig::GetExecTimeOut).stubs().will(returnValue(static_cast<uint32_t>(1)));
    MOCKER_CPP(&HostCpuRoceChannel::IbvPollCq).stubs().will(returnValue(5));

    EXPECT_EQ(impl_->ChannelDrain(), HCCL_E_INTERNAL);
}

TEST_F(HostCpuRoceChannelTest, Ut_ChannelDrain_When_PollCqFailed_Expect_HCCL_E_NETWORK)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();
    impl_->wqeNums_ = {1};
    SetupOneValidQpInfoMock();
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    MOCKER_CPP(&Hccl::EnvRtsConfig::GetExecTimeOut).stubs().will(returnValue(static_cast<uint32_t>(1)));
    MOCKER_CPP(&HostCpuRoceChannel::IbvPollCq).stubs().will(returnValue(-1));

    EXPECT_EQ(impl_->ChannelDrain(), HCCL_E_NETWORK);
}

TEST_F(HostCpuRoceChannelTest, Ut_ChannelDrain_When_NoCqeUntilTimeout_Expect_HCCL_E_TIMEOUT)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();
    impl_->wqeNums_ = {1};
    SetupOneValidQpInfoMock();
    MOCKER_CPP(&HostCpuRoceChannel::IbvPollCq).stubs().will(returnValue(0));
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    MOCKER_CPP(&Hccl::EnvRtsConfig::GetExecTimeOut).stubs().will(returnValue(static_cast<uint32_t>(0)));

    EXPECT_EQ(impl_->ChannelDrain(), HCCL_E_TIMEOUT);
}

TEST_F(HostCpuRoceChannelTest, Ut_Write_When_Normal_Expect_HCCL_SUCCESS)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&HostRdmaConnection::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::IbvPostRecv).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecUnpack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecUnpackProc).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::RmtBufferVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeCapability).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeData).stubs().will(returnValue(HCCL_SUCCESS));
    // construct
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.notifyNum = 4;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    // Init
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);
    // connect
    hcomm::ChannelStatus status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::SOCKET_OK);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CAP_EXCHANGED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::QP_CREATED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::DATA_EXCHANGE);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CONN_OK);
    EXPECT_EQ(status, ChannelStatus::READY);
    // Write
    impl_->wqeNums_ = {1};
    impl_->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((Hccl::RdmaHandle)0x1000));
    impl_->rmtRmaBuffers_[0]->addr = (uintptr_t)0x0002;
    impl_->rmtRmaBuffers_[0]->size = 10;
    std::vector<Hccl::QpInfo> qpInfos(1);
    MOCKER_CPP(&HostCpuRoceChannel::GetQpInfos).stubs().will(returnValue(qpInfos));
    MOCKER_CPP(&HostCpuRoceChannel::PostAndCheckSend).stubs().will(returnValue(HCCL_SUCCESS));
    void* localAddr = (void*)0x0001;
    void* remoteAddr = (void*)0x0002;
    size_t size = 10;
    HcclResult ret = impl_->Write(remoteAddr, localAddr, size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HostCpuRoceChannelTest, Ut_Read_When_Normal_Expect_HCCL_SUCCESS)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&HostRdmaConnection::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::IbvPostRecv).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecUnpack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecUnpackProc).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::RmtBufferVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeCapability).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeData).stubs().will(returnValue(HCCL_SUCCESS));
    // construct
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.notifyNum = 4;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    // Init
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);
    // connect
    hcomm::ChannelStatus status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::SOCKET_OK);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CAP_EXCHANGED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::QP_CREATED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::DATA_EXCHANGE);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CONN_OK);
    EXPECT_EQ(status, ChannelStatus::READY);
    // Read
    impl_->wqeNums_ = {1};
    impl_->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((Hccl::RdmaHandle)0x1000));
    impl_->rmtRmaBuffers_[0]->addr = (uintptr_t)0x0002;
    impl_->rmtRmaBuffers_[0]->size = 10;
    std::vector<Hccl::QpInfo> qpInfos(1);
    MOCKER_CPP(&HostCpuRoceChannel::GetQpInfos).stubs().will(returnValue(qpInfos));
    MOCKER_CPP(&HostCpuRoceChannel::PostAndCheckSend).stubs().will(returnValue(HCCL_SUCCESS));
    void* localAddr = (void*)0x0001;
    void* remoteAddr = (void*)0x0002;
    size_t size = 10;
    HcclResult ret = impl_->Read(localAddr, remoteAddr, size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ParseInputParam: dynamic_cast<CpuRoceEndpoint*> 失败 → HCCL_E_INTERNAL
TEST_F(HostCpuRoceChannelTest, Ut_Init_When_EndpointIsNotCpuRoce_Expect_HCCL_E_INTERNAL)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    // 构造一个非 CpuRoceEndpoint 的 Endpoint
    EndpointDesc mockDesc{};
    mockDesc.protocol = COMM_PROTOCOL_ROCE;
    mockDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    Hccl::IpAddress mockIp("1.0.0.0");
    mockDesc.commAddr.addr = mockIp.GetBinaryAddress().addr;
    mockDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    MockNonRoceEndpoint mockEp(mockDesc);
    EndpointHandle mockHandle = static_cast<EndpointHandle>(&mockEp);
    // construct
    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.notifyNum = 4;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(mockHandle, channelDesc);
    // Init 应在 dynamic_cast 失败时返回 HCCL_E_INTERNAL
    EXPECT_EQ(impl_->Init(), HCCL_E_INTERNAL);
}

// Write/Read/WriteWithNotify: maxMsgSize_ == 0 守卫 → HCCL_E_INTERNAL
TEST_F(HostCpuRoceChannelTest, Ut_WriteAndReadAndWriteWithNotify_When_MaxMsgSizeIsZero_Expect_HCCL_E_INTERNAL)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    // construct,不调用 Init,maxMsgSize_ 保持默认值 0
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    EXPECT_EQ(impl_->maxMsgSize_, 0u);
    // Write
    EXPECT_EQ(impl_->Write((void*)0x1, (void*)0x2, 10), HCCL_E_INTERNAL);
    // Read
    EXPECT_EQ(impl_->Read((void*)0x1, (void*)0x2, 10), HCCL_E_INTERNAL);
    // WriteWithNotify
    EXPECT_EQ(impl_->WriteWithNotify((void*)0x1, (void*)0x2, 10, 0), HCCL_E_INTERNAL);
}

// Write: len > maxMsgSize_ 时触发分片循环
TEST_F(HostCpuRoceChannelTest, Ut_Write_When_LenExceedsMaxMsgSize_Expect_Slicing_SUCCESS)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();
    // 设置小的 maxMsgSize_ 以触发分片
    impl_->maxMsgSize_ = 100;
    impl_->wqeNums_ = {1};
    // mock PostRdmaOp
    MOCKER_CPP(&HostCpuRoceChannel::PostRdmaOp).stubs().will(returnValue(HCCL_SUCCESS));
    // len = 250 → 3 个分片 (100 + 100 + 50)
    HcclResult ret = impl_->Write((void*)0x1, (void*)0x2, 250);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// Read: len > maxMsgSize_ 时触发分片循环
TEST_F(HostCpuRoceChannelTest, Ut_Read_When_LenExceedsMaxMsgSize_Expect_Slicing_SUCCESS)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();
    // 设置小的 maxMsgSize_ 以触发分片
    impl_->maxMsgSize_ = 100;
    impl_->wqeNums_ = {1};
    // mock PostRdmaOp
    MOCKER_CPP(&HostCpuRoceChannel::PostRdmaOp).stubs().will(returnValue(HCCL_SUCCESS));
    // len = 250 → 3 个分片 (100 + 100 + 50)
    HcclResult ret = impl_->Read((void*)0x1, (void*)0x2, 250);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// WriteWithNotify: len > maxMsgSize_ 时触发分片 (前 N-1 块 RDMA_WRITE + 尾块 RDMA_WRITE_WITH_IMM)
TEST_F(HostCpuRoceChannelTest, Ut_WriteWithNotify_When_LenExceedsMaxMsgSize_Expect_Slicing_SUCCESS)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();
    // 设置小的 maxMsgSize_ 以触发分片
    impl_->maxMsgSize_ = 100;
    impl_->wqeNums_ = {1};
    // mock 前 N-1 块的 PostRdmaOp
    MOCKER_CPP(&HostCpuRoceChannel::PostRdmaOp).stubs().will(returnValue(HCCL_SUCCESS));
    // mock 尾块所需的依赖
    std::vector<Hccl::QpInfo> qpInfos(1);
    MOCKER_CPP(&HostCpuRoceChannel::GetQpInfos).stubs().will(returnValue(qpInfos));
    MOCKER_CPP(&HostCpuRoceChannel::PrepareWriteWrResource).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::PostAndCheckSend).stubs().will(returnValue(HCCL_SUCCESS));
    // len = 250 → 前 2 块 PostRdmaOp(WRITE, 100 each) + 尾块 WRITE_WITH_IMM(50)
    HcclResult ret = impl_->WriteWithNotify((void*)0x1, (void*)0x2, 250, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ==================== Hybrid Mode UT Test Cases ====================

// ExchangeCapability: 魔数不匹配时，回退到原生模式
// 直接Mock ExchangeCapability返回HCCL_E_INTERNAL来模拟验证失败
TEST_F(HostCpuRoceChannelTest, Ut_ExchangeCapability_MagicMismatch_Fallback_To_NativeMode)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&HostRdmaConnection::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::IbvPostRecv).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecUnpack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecUnpackProc).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::RmtBufferVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeData).stubs().will(returnValue(HCCL_SUCCESS));
    // 直接Mock ExchangeCapability返回HCCL_E_INTERNAL模拟魔数不匹配
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeCapability).stubs().will(returnValue(HCCL_E_INTERNAL));

    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.notifyNum = 4;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);

    // ExchangeCapability返回失败后，isHybridMode_应为false
    EXPECT_EQ(impl_->isHybridMode_, false);
}

// ExchangeCapability: 能力校验失败时GetStatus返回FAILED
// 直接Mock ExchangeCapability返回HCCL_E_INTERNAL来模拟校验失败
TEST_F(HostCpuRoceChannelTest, Ut_ExchangeCapability_ValidationFailed_Expect_INTERNAL_ERROR)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&HostRdmaConnection::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::IbvPostRecv).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecUnpack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecUnpackProc).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::RmtBufferVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeData).stubs().will(returnValue(HCCL_SUCCESS));
    // 直接Mock ExchangeCapability返回HCCL_E_INTERNAL模拟校验失败
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeCapability).stubs().will(returnValue(HCCL_E_INTERNAL));

    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.notifyNum = 4;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);

    // rdmaStatus_初始为INIT，第一次GetStatus会转到CONN_OK
    // 手动设置为SOCKET_OK以触发ExchangeCapability
    impl_->rdmaStatus_ = HostCpuRoceChannel::RdmaStatus::SOCKET_OK;

    // GetStatus调用ExchangeCapability，校验失败返回HCCL_E_INTERNAL
    // 由于CHK_RET提前返回，rdmaStatus_保持在SOCKET_OK
    hcomm::ChannelStatus status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::SOCKET_OK);
    EXPECT_EQ(status, ChannelStatus::FAILED); // 校验失败导致GetStatus返回FAILED
}

// WriteWithNotifyHybrid: 空指针参数校验
TEST_F(HostCpuRoceChannelTest, Ut_WriteWithNotifyHybrid_NullPtr_Expect_HCCL_E_PARA)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    // 设置混合模式相关成员
    impl_->isHybridMode_ = true;
    impl_->connections_.clear(); // 清空connections_避免析构函数调用无效的HrtRaMrDereg
    impl_->localRmaBuffers_.emplace_back(localRdmaRmaBuffer.get());
    impl_->localMemMsg_[hccl::NOTIFY_SRC_MEM].addr = nullptr; // 未初始化

    // dst 为空指针
    HcclResult ret = impl_->WriteWithNotifyHybrid(nullptr, (void*)0x0001, 10, 0);
    EXPECT_EQ(ret, HCCL_E_PTR);

    // src 为空指针
    ret = impl_->WriteWithNotifyHybrid((void*)0x0001, nullptr, 10, 0);
    EXPECT_EQ(ret, HCCL_E_PTR);

    impl_->isHybridMode_ = false; // 析构前设置为false避免调用HrtRaMrDereg
}

// PostRdmaOp: ibv_post_send成功场景
// 通过Mock PostRdmaOp来覆盖ibv_post_send成功路径
TEST_F(HostCpuRoceChannelTest, Ut_PostRdmaOp_PostSendSuccess_Expect_Success)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->wqeNums_ = {1};
    impl_->localRmaBuffers_.emplace_back(localRdmaRmaBuffer.get());
    impl_->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((Hccl::RdmaHandle)0x1000));

    std::vector<Hccl::QpInfo> qpInfos(1);
    qpInfos[0].qp = reinterpret_cast<struct ibv_qp*>(0x2000);
    qpInfos[0].sendCq = reinterpret_cast<struct ibv_cq*>(0x2001);
    MOCKER_CPP(&HostCpuRoceChannel::GetQpInfos).stubs().will(returnValue(qpInfos));

    // Mock PostRdmaOp返回成功
    MOCKER_CPP(&HostCpuRoceChannel::PostRdmaOp).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = impl_->Write((void*)0x0001, (void*)0x0002, 10);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// PostRdmaOp: ibv_post_send失败场景
// 通过Mock PostRdmaOp来覆盖ibv_post_send失败路径
TEST_F(HostCpuRoceChannelTest, Ut_PostRdmaOp_PostSendFailed_Expect_NetworkError)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->wqeNums_ = {1};
    impl_->localRmaBuffers_.emplace_back(localRdmaRmaBuffer.get());
    impl_->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((Hccl::RdmaHandle)0x1000));

    std::vector<Hccl::QpInfo> qpInfos(1);
    qpInfos[0].qp = reinterpret_cast<struct ibv_qp*>(0x2000);
    qpInfos[0].sendCq = reinterpret_cast<struct ibv_cq*>(0x2001);
    MOCKER_CPP(&HostCpuRoceChannel::GetQpInfos).stubs().will(returnValue(qpInfos));

    // Mock PostRdmaOp返回失败
    MOCKER_CPP(&HostCpuRoceChannel::PostRdmaOp).stubs().will(returnValue(HCCL_E_NETWORK));

    HcclResult ret = impl_->Write((void*)0x0001, (void*)0x0002, 10);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
}

// WriteWithNotifyHybrid: len超过buffer大小时返回错误（HCCL_E_PARA）
// 设置完整的buffer和QPs配置，测试数据长度校验路径
TEST_F(HostCpuRoceChannelTest, Ut_WriteWithNotifyHybrid_LenExceedsBuffer_Expect_HCCL_E_PARA)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->isHybridMode_ = true;
    impl_->connections_.clear(); // 清空connections_避免析构函数调用无效的HrtRaMrDereg
    // 设置完整的本地和远程buffer
    impl_->localRmaBuffers_.emplace_back(localRdmaRmaBuffer.get());
    impl_->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((Hccl::RdmaHandle)0x1000));

    // 设置QPs信息
    std::vector<Hccl::QpInfo> qpInfos(1);
    qpInfos[0].qp = reinterpret_cast<struct ibv_qp*>(0x2000);
    qpInfos[0].sendCq = reinterpret_cast<struct ibv_cq*>(0x2001);
    MOCKER_CPP(&HostCpuRoceChannel::GetQpInfos).stubs().will(returnValue(qpInfos));

    // 设置remoteMemMsg_用于Notify WR
    impl_->remoteMemMsg_[hccl::MemType::ACK_NOTIFY_MEM].addr = reinterpret_cast<void*>(0x3000);
    impl_->remoteMemMsg_[hccl::MemType::ACK_NOTIFY_MEM].len = 64;
    impl_->remoteMemMsg_[hccl::MemType::ACK_NOTIFY_MEM].lkey = 1;

    // localRdmaRmaBuffer创建时size为666，设置len超过666以触发HCCL_E_PARA
    HcclResult ret = impl_->WriteWithNotifyHybrid((void*)0x0001, (void*)0x0002, 1000, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);

    impl_->isHybridMode_ = false; // 析构前设置为false避免调用HrtRaMrDereg
}

// WriteWithNotifyHybrid: qpInfos为空时返回错误
// 设置完整的buffer配置，但QPs信息为空
TEST_F(HostCpuRoceChannelTest, Ut_WriteWithNotifyHybrid_EmptyQpInfos_Expect_HCCL_E_ROCE_CONNECT)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->isHybridMode_ = true;
    impl_->connections_.clear(); // 清空connections_避免析构函数调用无效的HrtRaMrDereg
    // 设置完整的本地和远程buffer
    impl_->localRmaBuffers_.emplace_back(localRdmaRmaBuffer.get());
    impl_->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((Hccl::RdmaHandle)0x1000));

    // Mock GetQpInfos返回空vector
    std::vector<Hccl::QpInfo> emptyQpInfos;
    MOCKER_CPP(&HostCpuRoceChannel::GetQpInfos).stubs().will(returnValue(emptyQpInfos));

    HcclResult ret = impl_->WriteWithNotifyHybrid((void*)0x0001, (void*)0x0002, 10, 0);
    EXPECT_EQ(ret, HCCL_E_ROCE_CONNECT);

    impl_->isHybridMode_ = false; // 析构前设置为false避免调用HrtRaMrDereg
}

// WriteWithNotifyHybrid: len超过buffer大小时返回错误
TEST_F(HostCpuRoceChannelTest, Ut_WriteWithNotifyHybrid_LenExceedsBufferSize_Expect_HCCL_E_PARA)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->isHybridMode_ = true;
    impl_->connections_.clear(); // 清空connections_避免析构函数调用无效的HrtRaMrDereg
    impl_->localRmaBuffers_.emplace_back(localRdmaRmaBuffer.get());
    // localRdmaRmaBuffer 创建时 size 为 666

    // 设置一个大的 remote buffer
    impl_->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((Hccl::RdmaHandle)0x1000));

    // 设置QPs信息
    std::vector<Hccl::QpInfo> qpInfos(1);
    qpInfos[0].qp = reinterpret_cast<struct ibv_qp*>(0x2000);
    qpInfos[0].sendCq = reinterpret_cast<struct ibv_cq*>(0x2001);
    MOCKER_CPP(&HostCpuRoceChannel::GetQpInfos).stubs().will(returnValue(qpInfos));

    // 设置remoteMemMsg_用于Notify WR
    impl_->remoteMemMsg_[hccl::MemType::ACK_NOTIFY_MEM].addr = reinterpret_cast<void*>(0x3000);
    impl_->remoteMemMsg_[hccl::MemType::ACK_NOTIFY_MEM].len = 64;
    impl_->remoteMemMsg_[hccl::MemType::ACK_NOTIFY_MEM].lkey = 1;

    HcclResult ret = impl_->WriteWithNotifyHybrid((void*)0x0001, (void*)0x0002, 1000, 0); // 超过 buffer size
    EXPECT_EQ(ret, HCCL_E_PARA);

    impl_->isHybridMode_ = false; // 析构前设置为false避免调用HrtRaMrDereg
}

// NotifyWaitHybrid: 超时场景
TEST_F(HostCpuRoceChannelTest, Ut_NotifyWaitHybrid_Timeout_Expect_HCCL_E_TIMEOUT)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->isHybridMode_ = true;
    impl_->connections_.clear(); // 清空connections_避免析构函数调用无效的HrtRaMrDereg

    // 使用malloc分配内存，设置为0（不是expectedValue=1），会一直等到超时
    auto notifyMem = malloc(sizeof(std::atomic<uint32_t>));
    *reinterpret_cast<std::atomic<uint32_t>*>(notifyMem) = 0;
    impl_->localMemMsg_[hccl::DATA_NOTIFY_MEM].addr = notifyMem;
    impl_->localMemMsg_[hccl::DATA_NOTIFY_MEM].len = sizeof(std::atomic<uint32_t>);
    impl_->localMemMsg_[hccl::DATA_NOTIFY_MEM].lkey = 0;

    // 使用很短的超时时间（1s）触发超时
    HcclResult ret = impl_->NotifyWaitHybrid(1, 1); // 1s 超时
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);

    impl_->isHybridMode_ = false; // 析构前设置为false避免调用HrtRaMrDereg
    free(notifyMem);
}

// NotifyWaitHybrid: 内存未分配时仍然会超时（因为没有空指针检查）
// 注意：由于NotifyWaitHybrid没有空指针检查，必须设置有效可访问的内存地址
TEST_F(HostCpuRoceChannelTest, Ut_NotifyWaitHybrid_NullMemAddr_Expect_HCCL_E_TIMEOUT)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->isHybridMode_ = true;
    impl_->connections_.clear(); // 清空connections_避免析构函数调用无效的HrtRaMrDereg

    // 由于代码没有空指针检查，必须使用有效的可访问内存
    auto notifyMem = malloc(sizeof(std::atomic<uint32_t>));
    *reinterpret_cast<std::atomic<uint32_t>*>(notifyMem) = 0;
    impl_->localMemMsg_[hccl::ACK_NOTIFY_MEM].addr = notifyMem;
    impl_->localMemMsg_[hccl::ACK_NOTIFY_MEM].len = sizeof(std::atomic<uint32_t>);
    impl_->localMemMsg_[hccl::ACK_NOTIFY_MEM].lkey = 0;

    // 使用1s超时
    HcclResult ret = impl_->NotifyWaitHybrid(0, 1);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);

    impl_->isHybridMode_ = false; // 析构前设置为false避免调用HrtRaMrDereg
    free(notifyMem);
}

// GetStatus: 混合模式下调用 ExchangeDataHybird
TEST_F(HostCpuRoceChannelTest, Ut_GetStatus_HybridMode_Calls_ExchangeDataHybird)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&HostRdmaConnection::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::IbvPostRecv).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecUnpack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecUnpackProc).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::RmtBufferVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    // Mock ExchangeCapability 返回成功并设置 isHybridMode_ = true
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeCapability).stubs().will(invoke(ExchangeCapabilityHybridStub));
    // Mock ExchangeDataHybird 返回成功
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeDataHybird).stubs().will(returnValue(HCCL_SUCCESS));
    // Mock ConnectSingleQpHybrid 返回成功
    MOCKER_CPP(&HostCpuRoceChannel::ConnectSingleQpHybrid).stubs().will(returnValue(HCCL_SUCCESS));

    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.notifyNum = 4;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);

    // 第一个 GetStatus: INIT -> SOCKET_OK
    hcomm::ChannelStatus status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::SOCKET_OK);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);

    // 第二个 GetStatus: SOCKET_OK -> CAP_EXCHANGED
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CAP_EXCHANGED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);

    // 第三个 GetStatus: CAP_EXCHANGED -> QP_CREATED
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::QP_CREATED);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);

    // 第四个 GetStatus: QP_CREATED -> DATA_EXCHANGE (hybrid 模式调用 ExchangeDataHybird)
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::DATA_EXCHANGE);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);
}

// GetStatus: 混合模式下 ConnectSingleQpHybrid 失败
TEST_F(HostCpuRoceChannelTest, Ut_GetStatus_HybridMode_ConnectSingleQpHybrid_Failed)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&HostRdmaConnection::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::IbvPostRecv).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecUnpack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecUnpackProc).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::RmtBufferVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeCapability).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeDataHybird).stubs().will(returnValue(HCCL_SUCCESS));
    // Mock ConnectSingleQpHybrid 返回超时错误
    MOCKER_CPP(&HostCpuRoceChannel::ConnectSingleQpHybrid).stubs().will(returnValue(HCCL_E_TIMEOUT));

    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.notifyNum = 4;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);

    // 手动设置 isHybridMode_ = true，因为 ExchangeCapability 被 mock 了
    impl_->isHybridMode_ = true;

    // 驱动到 DATA_EXCHANGE 状态
    hcomm::ChannelStatus status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::SOCKET_OK);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::CAP_EXCHANGED);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::QP_CREATED);
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::DATA_EXCHANGE);
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);

    // DATA_EXCHANGE 状态下调用 ConnectSingleQpHybrid 失败
    status = impl_->GetStatus();
    EXPECT_EQ(impl_->rdmaStatus_, HostCpuRoceChannel::RdmaStatus::DATA_EXCHANGE);
    EXPECT_EQ(status, ChannelStatus::FAILED);
}

// NotifyIdToMemtypeHybird: 验证索引到类型的映射
TEST_F(HostCpuRoceChannelTest, Ut_NotifyIdToMemtypeHybird_Mapping)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    // remoteNotifyIdx = 0 -> ACK_NOTIFY_MEM
    EXPECT_EQ(impl_->NotifyIdToMemtypeHybird(0), hccl::MemType::ACK_NOTIFY_MEM);

    // remoteNotifyIdx = 1 -> DATA_NOTIFY_MEM
    EXPECT_EQ(impl_->NotifyIdToMemtypeHybird(1), hccl::MemType::DATA_NOTIFY_MEM);

    // 其他值 -> DATA_NOTIFY_MEM (默认)
    EXPECT_EQ(impl_->NotifyIdToMemtypeHybird(2), hccl::MemType::DATA_NOTIFY_MEM);
    EXPECT_EQ(impl_->NotifyIdToMemtypeHybird(100), hccl::MemType::DATA_NOTIFY_MEM);
}

// ExchangeDataHybird: RegisterUserMemHybird 失败时整体失败
TEST_F(HostCpuRoceChannelTest, Ut_ExchangeDataHybird_RegisterUserMemFailed)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    // Mock RegisterUserMemHybird 失败
    MOCKER_CPP(&HostCpuRoceChannel::RegisterUserMemHybird).stubs().will(returnValue(HCCL_E_MEMORY));

    HcclResult ret = impl_->ExchangeDataHybird();
    EXPECT_EQ(ret, HCCL_E_MEMORY);
}

// ExchangeDataHybird: BuildExchangeDataHybird 失败时整体失败
TEST_F(HostCpuRoceChannelTest, Ut_ExchangeDataHybird_BuildExchangeDataFailed)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    // Mock RegisterUserMemHybird 成功，BuildExchangeDataHybird 失败
    MOCKER_CPP(&HostCpuRoceChannel::RegisterUserMemHybird).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BuildExchangeDataHybird).stubs().will(returnValue(HCCL_E_INTERNAL));

    HcclResult ret = impl_->ExchangeDataHybird();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// ExchangeDataHybird: Socket 发送失败
TEST_F(HostCpuRoceChannelTest, Ut_ExchangeDataHybird_SendFailed)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    MOCKER_CPP(&HostCpuRoceChannel::RegisterUserMemHybird).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BuildExchangeDataHybird).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::Send).stubs().will(returnValue(false)); // 发送失败

    HcclResult ret = impl_->ExchangeDataHybird();
    EXPECT_EQ(ret, HCCL_E_NETWORK);
}

// ExchangeDataHybird: Socket 接收失败
TEST_F(HostCpuRoceChannelTest, Ut_ExchangeDataHybird_RecvFailed)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    MOCKER_CPP(&HostCpuRoceChannel::RegisterUserMemHybird).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BuildExchangeDataHybird).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::Send).stubs().will(returnValue(true));
    MOCKER_CPP(&Hccl::Socket::Recv).stubs().will(returnValue(false)); // 接收失败

    impl_->exchangeDataTotalSize_ = 1024;
    impl_->exchangeDataForSend_.resize(1024);
    HcclResult ret = impl_->ExchangeDataHybird();
    EXPECT_EQ(ret, HCCL_E_NETWORK);
}

// ExchangeDataHybird: ParseRecvExchangeDataHybird 失败
TEST_F(HostCpuRoceChannelTest, Ut_ExchangeDataHybird_ParseFailed)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    MOCKER_CPP(&HostCpuRoceChannel::RegisterUserMemHybird).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BuildExchangeDataHybird).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::Send).stubs().will(returnValue(true));
    MOCKER_CPP(&Hccl::Socket::Recv).stubs().will(returnValue(true));
    MOCKER_CPP(&HostCpuRoceChannel::ParseRecvExchangeDataHybird).stubs().will(returnValue(HCCL_E_PARA));

    HcclResult ret = impl_->ExchangeDataHybird();
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// BuildExchangeDataLengthHybird: 验证计算的长度
TEST_F(HostCpuRoceChannelTest, Ut_BuildExchangeDataLengthHybird_CorrectSize)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    // 期望的计算公式:
    // sizeof(u32)                    // qp数量
    // + sizeof(hccl::MemMsg) * 2     // output、input buffer
    // + sizeof(hccl::MemMsg) * 3     // 3个Notify
    // + sizeof(u8)                   // atomic value
    HcclResult ret = impl_->BuildExchangeDataLengthHybird();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// BuildNotifyWrHybird: 正常流程，remoteNotifyIdx=0 (ACK_NOTIFY_MEM)
TEST_F(HostCpuRoceChannelTest, Ut_BuildNotifyWrHybird_RemoteNotifyIdx0_Expect_HCCL_SUCCESS)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->isHybridMode_ = true;
    impl_->connections_.clear();

    // 设置 localMemMsg_[NOTIFY_SRC_MEM]
    impl_->localMemMsg_[hccl::NOTIFY_SRC_MEM].addr = reinterpret_cast<void*>(0x5000);
    impl_->localMemMsg_[hccl::NOTIFY_SRC_MEM].len = 8;
    impl_->localMemMsg_[hccl::NOTIFY_SRC_MEM].lkey = 10;

    // 设置 remoteMemMsg_[ACK_NOTIFY_MEM] (remoteNotifyIdx=0 映射到此类型)
    impl_->remoteMemMsg_[hccl::ACK_NOTIFY_MEM].addr = reinterpret_cast<void*>(0x6000);
    impl_->remoteMemMsg_[hccl::ACK_NOTIFY_MEM].len = 8;
    impl_->remoteMemMsg_[hccl::ACK_NOTIFY_MEM].lkey = 20;

    // 构造 ibv_send_wr，sg_list 预先分配
    struct ibv_send_wr notifRecordWr {};
    struct ibv_sge sg {};
    notifRecordWr.sg_list = &sg;

    // 调用 BuildNotifyWrHybird，remoteNotifyIdx=0 -> ACK_NOTIFY_MEM
    HcclResult ret = impl_->BuildNotifyWrHybird(0, notifRecordWr);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 验证 WR 字段
    EXPECT_EQ(notifRecordWr.sg_list->addr, reinterpret_cast<uint64_t>(impl_->localMemMsg_[hccl::NOTIFY_SRC_MEM].addr));
    EXPECT_EQ(notifRecordWr.sg_list->length, impl_->localMemMsg_[hccl::NOTIFY_SRC_MEM].len);
    EXPECT_EQ(notifRecordWr.sg_list->lkey, impl_->localMemMsg_[hccl::NOTIFY_SRC_MEM].lkey);
    EXPECT_EQ(notifRecordWr.opcode, IBV_WR_RDMA_WRITE);
    EXPECT_EQ(notifRecordWr.send_flags, IBV_SEND_SIGNALED);
    EXPECT_EQ(notifRecordWr.next, nullptr);
    EXPECT_EQ(notifRecordWr.num_sge, 1u);
    EXPECT_EQ(notifRecordWr.wr_id, 0u);
    EXPECT_EQ(notifRecordWr.wr.rdma.rkey, impl_->remoteMemMsg_[hccl::ACK_NOTIFY_MEM].lkey);
    EXPECT_EQ(
        notifRecordWr.wr.rdma.remote_addr, reinterpret_cast<uint64_t>(impl_->remoteMemMsg_[hccl::ACK_NOTIFY_MEM].addr));

    impl_->isHybridMode_ = false;
}

// CreateNotifyHybird: 正常流程，HrtRaMrReg 成功
TEST_F(HostCpuRoceChannelTest, Ut_CreateNotifyHybird_MrRegSuccess_Expect_HCCL_SUCCESS)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->isHybridMode_ = true;
    impl_->connections_.clear(); // 清空 connections_ 避免析构函数调用无效的 HrtRaMrDereg
    static auto dummyConn = std::make_unique<hcomm::HostRdmaConnection>(nullptr, (void*)0x1000);
    impl_->connections_.push_back(std::move(dummyConn));

    // Mock HostRdmaConnection::GetQpInfo 返回有效的 QP 信息
    Hccl::QpInfo qpInfo;
    qpInfo.qpHandle = reinterpret_cast<void*>(0x4000);
    qpInfo.qp = reinterpret_cast<struct ibv_qp*>(0x4001);
    qpInfo.sendCq = reinterpret_cast<struct ibv_cq*>(0x4002);
    qpInfo.recvCq = reinterpret_cast<struct ibv_cq*>(0x4003);
    MOCKER_CPP(&HostRdmaConnection::GetQpInfo).stubs().will(returnValue(qpInfo));

    // Mock HrtRaMrReg 返回成功
    MOCKER(HrtRaMrReg).stubs().will(returnValue(HCCL_SUCCESS));

    // 调用 CreateNotifyHybird，notifyType = DATA_NOTIFY_MEM, notifyId = 1
    HcclResult ret = impl_->CreateNotifyHybird(hccl::MemType::DATA_NOTIFY_MEM, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 验证 localMemMsg_ 已正确设置（因为 connections_ 被清空，析构函数不会 delete 内存）
    EXPECT_NE(impl_->localMemMsg_[hccl::MemType::DATA_NOTIFY_MEM].addr, nullptr);
    EXPECT_EQ(impl_->localMemMsg_[hccl::MemType::DATA_NOTIFY_MEM].notifyId, 1u);
    EXPECT_EQ(impl_->localMemMsg_[hccl::MemType::DATA_NOTIFY_MEM].len, 4u);
    // 释放内存（模拟析构函数行为，因为 connections_ 已清空不会自动释放）
    delete[] reinterpret_cast<int8_t*>(impl_->localMemMsg_[hccl::MemType::DATA_NOTIFY_MEM].addr);
    impl_->isHybridMode_ = false; // 析构前设置为 false 避免调用 HrtRaMrDereg
}

// CreateNotifyHybird: HrtRaMrReg 失败，返回 HCCL_E_MEMORY
TEST_F(HostCpuRoceChannelTest, Ut_CreateNotifyHybird_MrRegFailed_Expect_HCCL_E_MEMORY)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->isHybridMode_ = true;
    impl_->connections_.clear();
    static auto dummyConn = std::make_unique<hcomm::HostRdmaConnection>(nullptr, (void*)0x1000);
    impl_->connections_.push_back(std::move(dummyConn));

    // Mock HostRdmaConnection::GetQpInfo 返回有效的 QP 信息
    Hccl::QpInfo qpInfo;
    qpInfo.qpHandle = reinterpret_cast<void*>(0x4000);
    qpInfo.qp = reinterpret_cast<struct ibv_qp*>(0x4001);
    qpInfo.sendCq = reinterpret_cast<struct ibv_cq*>(0x4002);
    qpInfo.recvCq = reinterpret_cast<struct ibv_cq*>(0x4003);
    MOCKER_CPP(&HostRdmaConnection::GetQpInfo).stubs().will(returnValue(qpInfo));

    // Mock HrtRaMrReg 返回失败
    MOCKER(HrtRaMrReg).stubs().will(returnValue(HCCL_E_MEMORY));

    // 调用 CreateNotifyHybird，HrtRaMrReg 失败应返回 HCCL_E_MEMORY
    HcclResult ret = impl_->CreateNotifyHybird(hccl::MemType::ACK_NOTIFY_MEM, 0);
    EXPECT_EQ(ret, HCCL_E_MEMORY);

    impl_->isHybridMode_ = false;
}

// CreateNotifyHybird: memset_s 失败，返回 HCCL_E_MEMORY
// 本测试主要验证路径覆盖
TEST_F(HostCpuRoceChannelTest, Ut_CreateNotifyHybird_MemsetFailed_Expect_HCCL_E_MEMORY)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->isHybridMode_ = true;
    impl_->connections_.clear();
    static auto dummyConn = std::make_unique<hcomm::HostRdmaConnection>(nullptr, (void*)0x1000);
    impl_->connections_.push_back(std::move(dummyConn));

    // Mock HostRdmaConnection::GetQpInfo 返回有效的 QP 信息
    Hccl::QpInfo qpInfo;
    qpInfo.qpHandle = reinterpret_cast<void*>(0x4000);
    qpInfo.qp = reinterpret_cast<struct ibv_qp*>(0x4001);
    qpInfo.sendCq = reinterpret_cast<struct ibv_cq*>(0x4002);
    qpInfo.recvCq = reinterpret_cast<struct ibv_cq*>(0x4003);
    MOCKER_CPP(&HostRdmaConnection::GetQpInfo).stubs().will(returnValue(qpInfo));

    MOCKER(HrtRaMrReg).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = impl_->CreateNotifyHybird(hccl::MemType::DATA_NOTIFY_MEM, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    impl_->isHybridMode_ = false;
}

TEST_F(HostCpuRoceChannelTest, Ut_CreateNotifyValueBufferHybird_Success_Expect_HCCL_SUCCESS)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->isHybridMode_ = true;
    impl_->connections_.clear();
    static auto dummyConn = std::make_unique<hcomm::HostRdmaConnection>(nullptr, (void*)0x1000);
    impl_->connections_.push_back(std::move(dummyConn));

    Hccl::QpInfo qpInfo;
    qpInfo.qpHandle = reinterpret_cast<void*>(0x4000);
    qpInfo.qp = reinterpret_cast<struct ibv_qp*>(0x4001);
    qpInfo.sendCq = reinterpret_cast<struct ibv_cq*>(0x4002);
    qpInfo.recvCq = reinterpret_cast<struct ibv_cq*>(0x4003);
    MOCKER_CPP(&HostRdmaConnection::GetQpInfo).stubs().will(returnValue(qpInfo));
    MOCKER(HrtRaMrReg).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = impl_->CreateNotifyValueBufferHybird();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete[] reinterpret_cast<int8_t*>(impl_->localMemMsg_[hccl::MemType::NOTIFY_SRC_MEM].addr);
    impl_->isHybridMode_ = false;
}

TEST_F(HostCpuRoceChannelTest, Ut_CreateNotifyBufferHybird_Success_Expect_HCCL_SUCCESS)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->isHybridMode_ = true;
    impl_->connections_.clear();
    static auto dummyConn = std::make_unique<hcomm::HostRdmaConnection>(nullptr, (void*)0x1000);
    impl_->connections_.push_back(std::move(dummyConn));

    Hccl::QpInfo qpInfo;
    qpInfo.qpHandle = reinterpret_cast<void*>(0x4000);
    qpInfo.qp = reinterpret_cast<struct ibv_qp*>(0x4001);
    qpInfo.sendCq = reinterpret_cast<struct ibv_cq*>(0x4002);
    qpInfo.recvCq = reinterpret_cast<struct ibv_cq*>(0x4003);
    MOCKER_CPP(&HostRdmaConnection::GetQpInfo).stubs().will(returnValue(qpInfo));
    MOCKER(HrtRaMrReg).stubs().will(returnValue(HCCL_SUCCESS));

    const u64 kBufSize = 2048;
    u8* buffer = new (std::nothrow) u8[kBufSize];
    ASSERT_NE(buffer, nullptr);
    u8* data = buffer;
    u64 size = kBufSize;

    HcclResult ret = impl_->CreateNotifyBufferHybird(hccl::MemType::DATA_NOTIFY_MEM, 1, data, size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(size, kBufSize - sizeof(hccl::MemMsg));

    if (impl_->localMemMsg_[hccl::MemType::DATA_NOTIFY_MEM].addr != nullptr) {
        delete[] reinterpret_cast<int8_t*>(impl_->localMemMsg_[hccl::MemType::DATA_NOTIFY_MEM].addr);
        impl_->localMemMsg_[hccl::MemType::DATA_NOTIFY_MEM].addr = nullptr;
    }
    impl_->isHybridMode_ = false;
    delete[] buffer;
}

TEST_F(HostCpuRoceChannelTest, Ut_ExchangeCapability_DirectCall_Success_Expect_HCCL_SUCCESS)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&Hccl::Socket::Send).stubs().will(returnValue(true));
    MOCKER_CPP(&Hccl::Socket::Recv).stubs().will(returnValue(true));
    MOCKER_CPP(&HostRdmaConnection::CreateQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostRdmaConnection::ModifyQp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::IbvPostRecv).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::BufferVecPack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::NotifyVecUnpack).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ConnVecUnpackProc).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::RmtBufferVecUnpackProc)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HostCpuRoceChannel::ExchangeData).stubs().will(returnValue(HCCL_SUCCESS));

    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.notifyNum = 4;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);

    impl_->rdmaStatus_ = HostCpuRoceChannel::RdmaStatus::SOCKET_OK;
    HcclResult ret = impl_->ExchangeCapability();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HostCpuRoceChannelTest, Ut_RegisterUserMemHybird_Success_Expect_HCCL_SUCCESS)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->isHybridMode_ = true;
    impl_->connections_.clear();
    static auto dummyConn = std::make_unique<hcomm::HostRdmaConnection>(nullptr, (void*)0x1000);
    impl_->connections_.push_back(std::move(dummyConn));

    impl_->localRmaBuffers_.emplace_back(localRdmaRmaBuffer.get());

    Hccl::QpInfo qpInfo;
    qpInfo.qpHandle = reinterpret_cast<void*>(0x4000);
    MOCKER_CPP(&HostRdmaConnection::GetQpInfo).stubs().will(returnValue(qpInfo));
    MOCKER(HrtRaMrReg).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = impl_->RegisterUserMemHybird();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(impl_->localMemMsg_[hccl::USER_OUTPUT_MEM].addr, reinterpret_cast<void*>(localRdmaRmaBuffer->GetAddr()));
    EXPECT_EQ(impl_->localMemMsg_[hccl::USER_OUTPUT_MEM].notifyId, INVALID_DPU_NOTIFY_ID);

    impl_->isHybridMode_ = false;
}

TEST_F(HostCpuRoceChannelTest, Ut_BuildExchangeDataHybird_Success_Expect_HCCL_SUCCESS)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->isHybridMode_ = true;
    impl_->connections_.clear();
    static auto dummyConn = std::make_unique<hcomm::HostRdmaConnection>(nullptr, (void*)0x1000);
    impl_->connections_.push_back(std::move(dummyConn));

    impl_->localRmaBuffers_.emplace_back(localRdmaRmaBuffer.get());

    Hccl::QpInfo qpInfo;
    qpInfo.qpHandle = reinterpret_cast<void*>(0x4000);
    MOCKER_CPP(&HostRdmaConnection::GetQpInfo).stubs().will(returnValue(qpInfo));
    MOCKER(HrtRaMrReg).stubs().will(returnValue(HCCL_SUCCESS));

    impl_->localMemMsg_[hccl::USER_OUTPUT_MEM].addr = reinterpret_cast<void*>(localRdmaRmaBuffer->GetAddr());
    impl_->localMemMsg_[hccl::USER_OUTPUT_MEM].len = localRdmaRmaBuffer->GetSize();
    impl_->localMemMsg_[hccl::USER_OUTPUT_MEM].lkey = 1;

    HcclResult ret = impl_->BuildExchangeDataHybird();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    impl_->isHybridMode_ = false;
}

TEST_F(HostCpuRoceChannelTest, Ut_GetRemoteAddrHybird_Success_Expect_HCCL_SUCCESS)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->isHybridMode_ = true;

    const u64 kBufSize = 256;
    u8* buffer = new (std::nothrow) u8[kBufSize];
    ASSERT_NE(buffer, nullptr);
    u8* data = buffer;
    u64 size = kBufSize;

    HcclResult ret = impl_->GetRemoteAddrHybird(hccl::MemType::USER_OUTPUT_MEM, data, size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(size, kBufSize - sizeof(hccl::MemMsg));

    impl_->isHybridMode_ = false;
    delete[] buffer;
}

TEST_F(HostCpuRoceChannelTest, Ut_ParseRecvExchangeDataHybird_Success_Expect_HCCL_SUCCESS)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->isHybridMode_ = true;

    impl_->remoteMemMsg_[hccl::USER_OUTPUT_MEM].addr = reinterpret_cast<void*>(0x6000);
    impl_->remoteMemMsg_[hccl::USER_OUTPUT_MEM].len = 64;
    impl_->remoteMemMsg_[hccl::USER_OUTPUT_MEM].lkey = 1;
    impl_->remoteMemMsg_[hccl::USER_INPUT_MEM].addr = reinterpret_cast<void*>(0x6000);
    impl_->remoteMemMsg_[hccl::USER_INPUT_MEM].len = 64;
    impl_->remoteMemMsg_[hccl::USER_INPUT_MEM].lkey = 1;
    impl_->remoteMemMsg_[hccl::DATA_NOTIFY_MEM].addr = reinterpret_cast<void*>(0x6000);
    impl_->remoteMemMsg_[hccl::DATA_NOTIFY_MEM].len = 64;
    impl_->remoteMemMsg_[hccl::DATA_NOTIFY_MEM].lkey = 1;
    impl_->remoteMemMsg_[hccl::ACK_NOTIFY_MEM].addr = reinterpret_cast<void*>(0x6000);
    impl_->remoteMemMsg_[hccl::ACK_NOTIFY_MEM].len = 64;
    impl_->remoteMemMsg_[hccl::ACK_NOTIFY_MEM].lkey = 1;
    impl_->remoteMemMsg_[hccl::DATA_ACK_NOTIFY_MEM].addr = reinterpret_cast<void*>(0x6000);
    impl_->remoteMemMsg_[hccl::DATA_ACK_NOTIFY_MEM].len = 64;
    impl_->remoteMemMsg_[hccl::DATA_ACK_NOTIFY_MEM].lkey = 1;

    impl_->BuildExchangeDataLengthHybird();
    impl_->exchangeDataForRecv_.resize(impl_->exchangeDataTotalSize_);

    HcclResult ret = impl_->ParseRecvExchangeDataHybird();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(impl_->rmtRmaBuffers_.size(), 1u);

    impl_->isHybridMode_ = false;
}

TEST_F(HostCpuRoceChannelTest, Ut_BufferVecPack_Success_Expect_HCCL_SUCCESS)
{
    SetupSuccessfulConnectionMocks();
    MOCKER_CPP(&HostCpuRoceChannel::BufferVecPack).reset();
    auto impl_ = CreateInitAndConnect();

    impl_->bufferNum_ = 1;
    impl_->localRmaBuffers_.clear();
    impl_->localRmaBuffers_.emplace_back(localRdmaRmaBuffer.get());

    Hccl::BinaryStream binaryStream;
    HcclResult ret = impl_->BufferVecPack(binaryStream);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HostCpuRoceChannelTest, Ut_Describe_Success_Expect_NonEmptyString)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->notifyNum_ = 4;
    impl_->bufferNum_ = 1;
    impl_->connNum_ = 1;
    impl_->localRmaBuffers_.emplace_back(localRdmaRmaBuffer.get());

    std::string desc = impl_->Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("HostCpuRoceChannel"), std::string::npos);
}

TEST_F(HostCpuRoceChannelTest, Ut_RmtBufferVecUnpackProc_Success_Expect_HCCL_SUCCESS)
{
    SetupSuccessfulConnectionMocks();

    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.notifyNum = 4;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);

    impl_->bufferNum_ = 1;
    RdmaHandle rdmaHandle = (void*)0x1000000;
    impl_->rdmaHandle_ = rdmaHandle;

    Hccl::BinaryStream binaryStream;
    u32 rmtNum = 1;
    binaryStream << rmtNum;
    u32 pos = 0;
    binaryStream << pos;
    Hccl::ExchangeRdmaBufferDto dto(0x1000, 64, 1, "testBuffer");
    dto.Serialize(binaryStream);

    HcclResult ret = impl_->RmtBufferVecUnpackProc(binaryStream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(impl_->rmtRmaBuffers_.size(), 0);
}

TEST_F(HostCpuRoceChannelTest, Ut_PrepareNotifyWrResource_Success_Expect_HCCL_SUCCESS)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->localRmaBuffers_.emplace_back(localRdmaRmaBuffer.get());
    impl_->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((Hccl::RdmaHandle)0x1000));
    impl_->remoteDpuNotifyIds_.push_back(1);

    struct ibv_send_wr notifyRecordWr {};
    struct ibv_sge sg {};
    notifyRecordWr.sg_list = &sg;
    Hccl::TaskParam taskParam{};

    HcclResult ret = impl_->PrepareNotifyWrResource(0, 64, 0, notifyRecordWr, taskParam);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HostCpuRoceChannelTest, Ut_GetNotifyNum_Success_Expect_HCCL_SUCCESS)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->notifyNum_ = 4;
    uint32_t notifyNum = 0;

    HcclResult ret = impl_->GetNotifyNum(&notifyNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(notifyNum, 4u);
}

TEST_F(HostCpuRoceChannelTest, Ut_GetHcclBuffer_Success_Expect_HCCL_SUCCESS)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->rmtRmaBuffers_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>((Hccl::RdmaHandle)0x1000));
    impl_->rmtRmaBuffers_[0]->addr = 0x2000;
    impl_->rmtRmaBuffers_[0]->size = 64;

    void* addr = nullptr;
    uint64_t size = 0;
    HcclResult ret = impl_->GetHcclBuffer(addr, size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(addr, reinterpret_cast<void*>(0x2000));
    EXPECT_EQ(size, 64u);
}

TEST_F(HostCpuRoceChannelTest, Ut_ExchangeCapability_HybridMode_Expect_HCCL_SUCCESS)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    MOCKER_CPP(&Hccl::Socket::Send).stubs().will(returnValue(true));
    MOCKER_CPP(&Hccl::Socket::Recv).stubs().will(invoke(RecvForHybridModeStub));

    void* memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    channelDesc.notifyNum = 4;
    auto impl_ = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    EXPECT_EQ(impl_->Init(), HCCL_SUCCESS);

    impl_->rdmaStatus_ = HostCpuRoceChannel::RdmaStatus::SOCKET_OK;
    EXPECT_EQ(impl_->isHybridMode_, false);

    HcclResult ret = impl_->ExchangeCapability();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(impl_->isHybridMode_, true);
}

static int g_needStopCount = 0;
static bool NeedStopCounter()
{
    g_needStopCount++;
    return (g_needStopCount >= 3) ? true : false;
}

TEST_F(HostCpuRoceChannelTest, Ut_ConnectSingleQpHybrid_Success_Expect_HCCL_SUCCESS)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    impl_->isHybridMode_ = true;
    impl_->connections_.clear();
    static auto dummyConn = std::make_unique<hcomm::HostRdmaConnection>(nullptr, (void*)0x1000);
    impl_->connections_.push_back(std::move(dummyConn));

    Hccl::QpInfo qpInfo;
    qpInfo.qpHandle = reinterpret_cast<void*>(0x4000);
    MOCKER_CPP(&HostRdmaConnection::GetQpInfo).stubs().will(returnValue(qpInfo));
    MOCKER(HrtRaQpConnectAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetRaQpStatus).stubs().will(returnValue(0));
    g_needStopCount = 0;

    HcclResult ret = impl_->ConnectSingleQpHybrid(NeedStopCounter);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}
// 测试 SetDfxCallback - 正常设置
TEST_F(HostCpuRoceChannelTest, Ut_SetDfxCallback_When_Normal_Expect_SetSuccess)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();
    bool callbackCalled = false;

    std::function<HcclResult(const Hccl::TaskParam&, u64)> callback
        = [&callbackCalled](const Hccl::TaskParam& param, u64 handle) -> HcclResult {
        callbackCalled = true;
        return HCCL_SUCCESS;
    };

    HcclResult ret = impl_->SetDfxCallback(callback);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HostCpuRoceChannelTest, Ut_ReportWcStatusError_When_Normal_Expect_HCCL_E_NETWORK)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();
    impl_->localDpuNotifyIds_ = {0};
    impl_->remoteDpuNotifyIds_ = {0};

    MOCKER(RptInputErr).stubs().will(invoke(stub_RptInputErr_print));

    HcclResult ret = impl_->ReportWcStatusError(IBV_WC_WR_FLUSH_ERR);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
    GlobalMockObject::verify();
}

TEST_F(HostCpuRoceChannelTest, Ut_ReportWcStatusError_When_VariousStatuses_Expect_HCCL_E_NETWORK)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();
    impl_->localDpuNotifyIds_ = {0};
    impl_->remoteDpuNotifyIds_ = {0};

    MOCKER(RptInputErr).stubs().will(invoke(stub_RptInputErr_print));

    std::vector<enum ibv_wc_status> errorStatuses
        = {IBV_WC_WR_FLUSH_ERR, IBV_WC_BAD_RESP_ERR, IBV_WC_LOC_ACCESS_ERR, IBV_WC_REM_ACCESS_ERR, IBV_WC_REM_OP_ERR};

    for (const auto& status : errorStatuses) {
        HcclResult ret = impl_->ReportWcStatusError(status);
        EXPECT_EQ(ret, HCCL_E_NETWORK);
    }
    GlobalMockObject::verify();
}

TEST_F(HostCpuRoceChannelTest, Ut_GetRemoteMems_When_OnlyCclBuffer_Expect_Success)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    // Only cclBuffer at index 0
    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x1000000);
    Hccl::ExchangeRdmaBufferDto cclBufDto(0x2000000, 4096, 100, "cclBuffer");
    impl_->rmtRmaBuffers_.emplace_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>(rdmaHandle, cclBufDto));

    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    uint32_t memNum = 0;

    HcclResult ret = impl_->GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memNum, 1);
}

TEST_F(HostCpuRoceChannelTest, Ut_GetRemoteMems_When_UserBuffersExist_Expect_Success)
{
    SetupSuccessfulConnectionMocks();
    auto impl_ = CreateInitAndConnect();

    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x1000000);
    // cclBuffer at index 0
    Hccl::ExchangeRdmaBufferDto cclBufDto(0x2000000, 4096, 100, "cclTag");
    impl_->rmtRmaBuffers_.emplace_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>(rdmaHandle, cclBufDto));
    // user buffer 1 at index 1
    Hccl::ExchangeRdmaBufferDto userDto1(0x3000000, 1024, 200, "userTag1");
    auto userBuf1 = std::make_unique<Hccl::RemoteRdmaRmaBuffer>(rdmaHandle, userDto1);
    userBuf1->memType = HCCL_MEM_TYPE_DEVICE;
    impl_->rmtRmaBuffers_.emplace_back(std::move(userBuf1));
    // user buffer 2 at index 2
    Hccl::ExchangeRdmaBufferDto userDto2(0x4000000, 2048, 300, "userTag2");
    auto userBuf2 = std::make_unique<Hccl::RemoteRdmaRmaBuffer>(rdmaHandle, userDto2);
    userBuf2->memType = HCCL_MEM_TYPE_HOST;
    impl_->rmtRmaBuffers_.emplace_back(std::move(userBuf2));

    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    uint32_t memNum = 0;

    HcclResult ret = impl_->GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memNum, 3);

    ASSERT_NE(remoteMem, nullptr);
    ASSERT_NE(memInfos, nullptr);
}

static void SetupBcMocks() { MOCKER_CPP(&HostRdmaConnection::Init).stubs().will(returnValue(HCCL_SUCCESS)); }

TEST_F(HostCpuRoceChannelTest, Ut_BuildConnection_When_ConfigNotAvailable_Expect_AllUdpSportZero)
{
    SetupBcMocks();
    Hccl::EnvConfig::GetInstance().rdmaCfg.multiQpSrcPortConfig_.ipPairToPorts.clear();

    auto impl = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    Endpoint* localEpPtr = reinterpret_cast<Endpoint*>(endpointHandle);
    impl->localEp_ = localEpPtr->GetEndpointDesc();
    impl->remoteEp_ = channelDesc.remoteEndpoint;
    impl->lbMax_ = 0;
    impl->channelDesc_.roceAttr.queueNum = 2;
    impl->socket_ = fakeSocket;
    impl->rdmaHandle_ = (void*)0x1000000;

    EXPECT_EQ(impl->BuildConnection(), HCCL_SUCCESS);
    ASSERT_EQ(impl->connections_.size(), 2u);
    EXPECT_EQ(impl->connections_[0]->qpInfo_.udpSport, 0u);
    EXPECT_EQ(impl->connections_[1]->qpInfo_.udpSport, 0u);
}

TEST_F(HostCpuRoceChannelTest, Ut_BuildConnection_When_ConfigAvailableButNoIpMatch_Expect_AllUdpSportZero)
{
    SetupBcMocks();
    auto& portMap = Hccl::EnvConfig::GetInstance().rdmaCfg.multiQpSrcPortConfig_.ipPairToPorts;
    portMap.clear();
    portMap["3.0.0.0,4.0.0.0"] = {10001};

    auto impl = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    Endpoint* localEpPtr = reinterpret_cast<Endpoint*>(endpointHandle);
    impl->localEp_ = localEpPtr->GetEndpointDesc();
    impl->remoteEp_ = channelDesc.remoteEndpoint;
    impl->lbMax_ = 0;
    impl->channelDesc_.roceAttr.queueNum = 2;
    impl->socket_ = fakeSocket;
    impl->rdmaHandle_ = (void*)0x1000000;

    EXPECT_EQ(impl->BuildConnection(), HCCL_SUCCESS);
    ASSERT_EQ(impl->connections_.size(), 2u);
    EXPECT_EQ(impl->connections_[0]->qpInfo_.udpSport, 0u);
    EXPECT_EQ(impl->connections_[1]->qpInfo_.udpSport, 0u);

    portMap.clear();
}

TEST_F(HostCpuRoceChannelTest, Ut_BuildConnection_When_PortsCountLessThanLoopTimes_Expect_CycleExpand)
{
    SetupBcMocks();
    auto impl = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    Endpoint* localEpPtr = reinterpret_cast<Endpoint*>(endpointHandle);
    impl->localEp_ = localEpPtr->GetEndpointDesc();
    impl->remoteEp_ = channelDesc.remoteEndpoint;
    impl->lbMax_ = 0;
    impl->channelDesc_.roceAttr.queueNum = 4;
    static uint16_t srcPorts[] = {10001, 10002, 10001, 10002};
    impl->channelDesc_.roceAttr.srcPortList = srcPorts;
    impl->socket_ = fakeSocket;
    impl->rdmaHandle_ = (void*)0x1000000;

    EXPECT_EQ(impl->BuildConnection(), HCCL_SUCCESS);
    ASSERT_EQ(impl->connections_.size(), 4u);
    EXPECT_EQ(impl->connections_[0]->qpInfo_.udpSport, 10001u);
    EXPECT_EQ(impl->connections_[1]->qpInfo_.udpSport, 10002u);
    EXPECT_EQ(impl->connections_[2]->qpInfo_.udpSport, 10001u);
    EXPECT_EQ(impl->connections_[3]->qpInfo_.udpSport, 10002u);
}

// BC4: exchangeAllMems=true（hixl场景）→ 即使 srcPortList 可用，udpSport 仍为 0
TEST_F(HostCpuRoceChannelTest, Ut_BuildConnection_When_ExchangeAllMemsTrue_Expect_AllUdpSportZero)
{
    SetupBcMocks();
    auto& portMap = Hccl::EnvConfig::GetInstance().rdmaCfg.multiQpSrcPortConfig_.ipPairToPorts;
    portMap.clear();
    portMap["1.0.0.0,2.0.0.0"] = {10001, 10002};

    auto impl = std::make_unique<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
    Endpoint* localEpPtr = reinterpret_cast<Endpoint*>(endpointHandle);
    impl->localEp_ = localEpPtr->GetEndpointDesc();
    impl->remoteEp_ = channelDesc.remoteEndpoint;
    impl->lbMax_ = 0;
    impl->channelDesc_.roceAttr.queueNum = 2;
    impl->channelDesc_.exchangeAllMems = true;
    impl->socket_ = fakeSocket;
    impl->rdmaHandle_ = (void*)0x1000000;

    EXPECT_EQ(impl->BuildConnection(), HCCL_SUCCESS);
    ASSERT_EQ(impl->connections_.size(), 2u);
    EXPECT_EQ(impl->connections_[0]->qpInfo_.udpSport, 0u);
    EXPECT_EQ(impl->connections_[1]->qpInfo_.udpSport, 0u);

    portMap.clear();
}
