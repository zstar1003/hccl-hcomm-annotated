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

#define private public
#define protected public
#include "aicpu/aicpu_ts_urma_channel.h"
#include "endpoint.h"
#include "buffer/local_ub_rma_buffer.h"
#undef private
#undef protected
using namespace hcomm;

namespace {
class StubEndpointForUrmaChannel : public Endpoint {
public:
    StubEndpointForUrmaChannel() : Endpoint(MakeDesc()) { ctxHandle_ = reinterpret_cast<void*>(0x1); }

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
        Hccl::IpAddress localIp("127.0.0.1");
        desc.protocol = COMM_PROTOCOL_UB_CTP;
        desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        desc.commAddr.addr = localIp.GetBinaryAddress().addr;
        desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        desc.loc.device.devPhyId = 3;
        return desc;
    }
};

std::shared_ptr<Hccl::LocalUbRmaBuffer> MakeUrmaLocalBuffer(uintptr_t addr, u64 size, const char* tag)
{
    auto buffer = std::make_shared<Hccl::Buffer>(addr, size, HCCL_MEM_TYPE_DEVICE, tag);
    return std::make_shared<Hccl::LocalUbRmaBuffer>(buffer);
}

HcommResult StubUrmaGetAllMemHandlesOne(EndpointHandle, void** memHandles, uint32_t* memHandleNum)
{
    static std::shared_ptr<Hccl::Buffer> buffer
        = std::make_shared<Hccl::Buffer>(0x620000U, 0x1000U, HCCL_MEM_TYPE_DEVICE, "urma_all");
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffer = std::make_shared<Hccl::LocalUbRmaBuffer>(buffer);
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffers[1] = {localBuffer};
    *memHandles = localBuffers;
    *memHandleNum = 1;
    return HCCL_SUCCESS;
}

HcommResult StubUrmaGetAllMemHandlesWithNullSlot(EndpointHandle, void** memHandles, uint32_t* memHandleNum)
{
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffers[1] = {nullptr};
    *memHandles = localBuffers;
    *memHandleNum = 1;
    return HCCL_SUCCESS;
}

void AttachFakeUbMemTransport(AicpuTsUrmaChannel& ch)
{
    Hccl::BaseMemTransport::Attribution attr{};
    Hccl::IpAddress locAddrIp{"1.1.1.1"};
    Hccl::IpAddress rmtAddrIp{"2.2.2.2"};
    Hccl::Socket sock{nullptr, locAddrIp, 0, rmtAddrIp, "ut", Hccl::SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE};
    Hccl::BaseMemTransport::CommonLocRes commonLocRes{};
    auto conn = std::make_unique<Hccl::DevUbConnection>(nullptr, locAddrIp, rmtAddrIp, Hccl::OpMode::OPBASE);
    commonLocRes.connVec.push_back(conn.get());
    RdmaHandle rdmaHandle = nullptr;
    Hccl::BaseMemTransport::LocCntNotifyRes locCntNotifyRes{};
    Hccl::LinkData linkData(
        Hccl::PortDeploymentType::HOST_NET, Hccl::LinkProtocol::UB_TP, 0, 0, Hccl::IpAddress(), Hccl::IpAddress());

    ch.connections_.push_back(std::move(conn));
    ch.memTransport_ = std::make_unique<Hccl::UbMemTransport>(
        commonLocRes, attr, linkData, sock, rdmaHandle, locCntNotifyRes, false);
}

std::vector<Hccl::LocalRmaBuffer*> g_capturedUpdateBuffers;

HcclResult StubUrmaUpdateMemInfo(Hccl::UbMemTransport*, std::vector<Hccl::LocalRmaBuffer*>& bufferVec)
{
    g_capturedUpdateBuffers = bufferVec;
    return HCCL_SUCCESS;
}
} // namespace

class AicpuTsUrmaChannelTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(AicpuTsUrmaChannelTest, Ut_Clean_WithoutInit_Returns_SUCCESS)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);

    auto ret = ch.Clean();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AicpuTsUrmaChannelTest, Ut_Resume_MockedBuilds_Returns_SUCCESS)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);

    // Mock private helper methods BuildConnection and BuildUbMemTransport
    MOCKER_CPP(&AicpuTsUrmaChannel::BuildSocket, HcclResult(AicpuTsUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&AicpuTsUrmaChannel::BuildConnection, HcclResult(AicpuTsUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&AicpuTsUrmaChannel::ResetLocalNotifies, HcclResult(AicpuTsUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&AicpuTsUrmaChannel::BuildUbMemTransport, HcclResult(AicpuTsUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    auto ret = ch.Resume();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AicpuTsUrmaChannelTest, Ut_ResetLocalNotifies_When_Empty_Expect_SUCCESS)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);

    EXPECT_EQ(ch.ResetLocalNotifies(), HCCL_SUCCESS);
}

TEST_F(AicpuTsUrmaChannelTest, Ut_Resume_When_ResetLocalNotifiesFailed_Expect_ReturnFailed)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);

    MOCKER_CPP(&AicpuTsUrmaChannel::BuildSocket, HcclResult(AicpuTsUrmaChannel::*)())
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsUrmaChannel::BuildConnection, HcclResult(AicpuTsUrmaChannel::*)())
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsUrmaChannel::ResetLocalNotifies, HcclResult(AicpuTsUrmaChannel::*)())
        .stubs()
        .will(returnValue(HCCL_E_INTERNAL));
    MOCKER_CPP(&AicpuTsUrmaChannel::BuildUbMemTransport, HcclResult(AicpuTsUrmaChannel::*)()).expects(never());

    EXPECT_EQ(ch.Resume(), HCCL_E_INTERNAL);
}

TEST_F(AicpuTsUrmaChannelTest, UT_ParseInputParam_When_ExchangeAllMemsFalse_Expect_FillCommonRes)
{
    StubEndpointForUrmaChannel endpoint;
    auto localBuffer = MakeUrmaLocalBuffer(0x610000U, 0x1000U, "urma_desc");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};
    HcommChannelDesc desc{};
    desc.exchangeAllMems = false;
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AicpuTsUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.commonRes_.bufferVec.size(), 1U);
    EXPECT_EQ(ch.commonRes_.bufferVec[0], localBuffer.get());
}

TEST_F(AicpuTsUrmaChannelTest, UT_ParseInputParam_When_ExchangeAllMemsTrue_Expect_FillCommonRes)
{
    StubEndpointForUrmaChannel endpoint;
    HcommChannelDesc desc{};
    desc.exchangeAllMems = true;
    AicpuTsUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(HcommMemGetAllMemHandles).stubs().will(invoke(StubUrmaGetAllMemHandlesOne));

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.commonRes_.bufferVec.size(), 1U);
    EXPECT_EQ(ch.commonRes_.bufferVec[0]->GetAddr(), 0x620000U);
}

TEST_F(AicpuTsUrmaChannelTest, UT_ParseInputParam_When_ExchangeAllMemsTrueAndNullSlot_Expect_ReturnHCCL_E_PTR)
{
    StubEndpointForUrmaChannel endpoint;
    HcommChannelDesc desc{};
    desc.exchangeAllMems = true;
    AicpuTsUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(HcommMemGetAllMemHandles).stubs().will(invoke(StubUrmaGetAllMemHandlesWithNullSlot));

    EXPECT_EQ(ch.ParseInputParam(), HCCL_E_PTR);
}

TEST_F(AicpuTsUrmaChannelTest, UT_UpdateMemInfo_When_ValidMemHandles_Expect_UpdateTransportAndCommonRes)
{
    g_capturedUpdateBuffers.clear();
    HcommChannelDesc desc{};
    AicpuTsUrmaChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    AttachFakeUbMemTransport(ch);
    auto initBuffer = MakeUrmaLocalBuffer(0x620000U, 0x1000U, "urma_init");
    ch.commonRes_.bufferVec.push_back(initBuffer.get());
    auto localBuffer = MakeUrmaLocalBuffer(0x630000U, 0x1000U, "urma_update");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};

    MOCKER_CPP(
        &Hccl::UbMemTransport::UpdateMemInfo, HcclResult(Hccl::UbMemTransport::*)(std::vector<Hccl::LocalRmaBuffer*>&))
        .stubs()
        .with(mockcpp::any())
        .will(invoke(StubUrmaUpdateMemInfo));

    EXPECT_EQ(ch.UpdateMemInfo(memHandles, 1), HCCL_SUCCESS);
    ASSERT_EQ(g_capturedUpdateBuffers.size(), 1U);
    EXPECT_EQ(g_capturedUpdateBuffers[0], localBuffer.get());
    ASSERT_EQ(ch.commonRes_.bufferVec.size(), 2U);
    EXPECT_EQ(ch.commonRes_.bufferVec[0], initBuffer.get());
    EXPECT_EQ(ch.commonRes_.bufferVec[1], localBuffer.get());
}

TEST_F(AicpuTsUrmaChannelTest, UT_UpdateMemInfo_When_NullHandle_Expect_ReturnHCCL_E_PTR)
{
    HcommChannelDesc desc{};
    AicpuTsUrmaChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    AttachFakeUbMemTransport(ch);
    HcommMemHandle memHandles[1] = {nullptr};

    EXPECT_EQ(ch.UpdateMemInfo(memHandles, 1), HCCL_E_PTR);
}

TEST_F(AicpuTsUrmaChannelTest, Ut_GetStatus_DfxInfo_TEST)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);

    // 1. 构造依赖
    Hccl::BaseMemTransport::Attribution attr{};
    Hccl::IpAddress locAddrIp{"1.1.1.1"};
    Hccl::IpAddress rmtAddrIp{"2.2.2.2"};
    Hccl::Socket sock{nullptr, locAddrIp, 0, rmtAddrIp, "ut", Hccl::SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE};
    Hccl::BaseMemTransport::CommonLocRes commonLocRes{};
    std::vector<Hccl::RmaConnection*> conns;
    auto conn = std::make_unique<Hccl::DevUbConnection>(nullptr, locAddrIp, rmtAddrIp, Hccl::OpMode::OPBASE);
    conns.push_back(conn.get()); // 仅占位，测试不涉及具体连接对象
    commonLocRes.connVec = conns;
    RdmaHandle rdmaHandle = nullptr;
    Hccl::BaseMemTransport::LocCntNotifyRes locCntNotifyRes{};

    Hccl::PortDeploymentType portDeploymentType = Hccl::PortDeploymentType::HOST_NET;
    Hccl::LinkProtocol linkProtocol = Hccl::LinkProtocol::UB_TP;
    Hccl::IpAddress locAddr;
    Hccl::IpAddress rmtAddr;
    uint32_t locDevPhyId = 0;
    uint32_t rmtDevPhyId = 0;
    Hccl::LinkData ld = Hccl::LinkData(portDeploymentType, linkProtocol, locDevPhyId, rmtDevPhyId, locAddr, rmtAddr);

    auto memTransport
        = std::make_unique<Hccl::UbMemTransport>(commonLocRes, attr, ld, sock, rdmaHandle, locCntNotifyRes, false);
    memTransport->connNum = 1;
    memTransport->baseStatus = Hccl::TransportStatus::READY;
    ch.memTransport_ = std::move(memTransport);

    MOCKER_CPP(&Hccl::UbMemTransport::Describe, HcclResult(Hccl::UbMemTransport::*)(std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS))
        .then(returnValue(HCCL_E_PARA)); // 连续两次调用，第一次成功，第二次失败

    auto ret = ch.GetStatus();
    EXPECT_EQ(ret, ChannelStatus::READY);

    // 再次调用GetStatus，验证isFirstPrintChannelInfo_逻辑分支覆盖
    ch.isFirstPrintChannelInfo_ = true; // 重置为true，模拟第一次打印
    ret = ch.GetStatus();
    EXPECT_EQ(ret, ChannelStatus::FAILED);

    GlobalMockObject::verify();
    GlobalMockObject::reset();
}

TEST_F(AicpuTsUrmaChannelTest, Ut_StartListen_When_RoleNotServer_Expect_SUCCESS)
{
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_CLIENT;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUrmaChannel ch(ep, desc);

    auto ret = ch.StartListen();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
