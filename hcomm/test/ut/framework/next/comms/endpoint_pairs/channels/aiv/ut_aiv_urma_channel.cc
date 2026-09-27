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
#include <cstdint>
#include <cstdlib>
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "aiv_urma_channel.h"
#include "../sockets/socket_mgr.h"
#include "aiv_urma_transport.h"
#undef protected
#undef private
#include "hcomm_c_adpt.h"
#include "endpoint.h"
#include "exchange_ub_buffer_dto.h"
#include "runtime_api_exception.h"

using namespace hcomm;
using namespace Hccl;

namespace {

uint32_t g_listenPort = 0;
constexpr u32 STUB_TOKEN_VALUE = 0x12345678U;
constexpr u32 STUB_TOKEN_ID = 0x87654321U;

HcommResult StubEndpointStartListen(EndpointHandle, uint32_t port, const HcommEndpointListenConfig*)
{
    g_listenPort = port;
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

LinkData MakeDefaultLinkData()
{
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    return LinkData(portType, 0, 1, 0, 0);
}

class StubEndpointForAivUrmaChannel : public Endpoint {
public:
    explicit StubEndpointForAivUrmaChannel(const EndpointDesc& desc, void* rdmaHandle) : Endpoint(desc)
    {
        ctxHandle_ = rdmaHandle;
    }

    HcclResult Init() override { return HCCL_SUCCESS; }
    // 内存方法在 RegedMemMgr 上，Endpoint 不再 override
    RegedMemMgr* GetRegedMemMgr() override { return nullptr; }
    void* GetRdmaHandle() override { return ctxHandle_; }
    bool IsCtxHandleValid() const override { return ctxHandle_ != nullptr; }

private:
    void* ctxHandle_{nullptr};
};

class StubLocalUbRmaBufferWithExchangeDto : public LocalUbRmaBuffer {
public:
    explicit StubLocalUbRmaBufferWithExchangeDto(const std::shared_ptr<Buffer>& buffer) : LocalUbRmaBuffer(buffer) {}

    std::unique_ptr<Serializable> GetExchangeDto() override
    {
        return std::make_unique<ExchangeUbBufferDto>(
            GetAddr(), GetSize(), STUB_TOKEN_VALUE, STUB_TOKEN_ID, 0, UINT32_MAX);
    }
};

HcommResult StubHcommMemGetAllMemHandlesFail(EndpointHandle, void**, uint32_t*) { return HCCL_E_INTERNAL; }

HcommResult StubHcommMemGetAllMemHandlesEmpty(EndpointHandle, void** memHandles, uint32_t* memHandleNum)
{
    *memHandles = nullptr;
    *memHandleNum = 0;
    return HCCL_SUCCESS;
}

HcommResult StubHcommMemGetAllMemHandlesOne(EndpointHandle, void** memHandles, uint32_t* memHandleNum)
{
    static std::shared_ptr<Hccl::Buffer> buffer = std::make_shared<Hccl::Buffer>(0x12340000U, 0x2000U, "aiv_urma_ut");
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffer = std::make_shared<Hccl::LocalUbRmaBuffer>(buffer);
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffers[1] = {localBuffer};
    *memHandles = localBuffers;
    *memHandleNum = 1;
    return HCCL_SUCCESS;
}

static hccl::DeviceMem StubDeviceMemAlloc(u64 size, bool /*level2Address*/)
{
    void* ptr = std::malloc(static_cast<size_t>(size));
    // UT uses host memory as fake device memory. Keep owner=false because DeviceMem dtor calls hrtFree for owner
    // memory.
    return hccl::DeviceMem(ptr, size, false);
}

aclError StubAclrtMalloc(void** devPtr, size_t size, aclrtMemMallocPolicy /*policy*/, aclrtMallocConfig*)
{
    if (devPtr == nullptr) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    *devPtr = std::malloc(size);
    return (*devPtr == nullptr) ? ACL_ERROR_RT_MEMORY_ALLOCATION : ACL_SUCCESS;
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
    return ACL_SUCCESS;
}

bool IsPtrInRange(const void* ptr, const void* base, size_t size)
{
    uintptr_t ptrValue = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t baseValue = reinterpret_cast<uintptr_t>(base);
    return ptrValue >= baseValue && ptrValue < baseValue + size;
}

void BuildReadyTransport(AivUrmaChannel& ch, DevUbCtpConnection& conn)
{
    BaseMemTransport::CommonLocRes commonRes{};
    conn.status = RmaConnStatus::READY;
    conn.jettyId = 3;
    conn.dbAddr = 0x1000;
    conn.sqBuffVa = 0x2000;
    conn.sqDepth = 4;
    conn.tpn = 5;
    conn.cqInfo_.id = 6;
    conn.cqInfo_.va = 0x3000;
    conn.cqInfo_.cqeSize = 64;
    conn.cqInfo_.cqDepth = 8;
    conn.cqInfo_.swdbAddr = 0x4000;
    commonRes.connVec.emplace_back(&conn);

    BaseMemTransport::Attribution attr{};
    LinkData linkData = MakeDefaultLinkData();
    static Socket socket(
        nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    ch.transport_
        = std::make_unique<AivUrmaTransport>(commonRes, attr, linkData, socket, reinterpret_cast<RdmaHandle>(0x1));
    ch.transport_->transportStatus_ = TransportStatus::READY;
}

static void StubHrtMemcpyReal(void* dst, uint64_t dstMax, const void* src, uint64_t count, Hccl::tagRtMemcpyKind)
{
    if (dst != nullptr && src != nullptr && dstMax >= count) {
        (void)std::memcpy(dst, src, static_cast<size_t>(count));
    }
}
} // namespace

class AivUrmaChannelTest : public testing::Test {
protected:
    static void TearDownTestCase() { GlobalMockObject::verify(); }

    void TearDown() override { GlobalMockObject::verify(); }

    HcommChannelDesc MakeDefaultDesc()
    {
        HcommChannelDesc desc{};
        desc.roceAttr.queueNum = 1;
        return desc;
    }
};

TEST_F(AivUrmaChannelTest, Ut_Clean_WhenCalled_Returns_SUCCESS)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());

    EXPECT_EQ(ch.Clean(), HCCL_SUCCESS);
    EXPECT_EQ(ch.Clean(), HCCL_SUCCESS);
}

TEST_F(AivUrmaChannelTest, Ut_Resume_WhenBuildsMocked_Returns_SUCCESS)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());

    MOCKER_CPP(&AivUrmaChannel::BuildConnection, HcclResult(AivUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AivUrmaChannel::BuildAivUrmaTransport, HcclResult(AivUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(ch.Resume(), HCCL_SUCCESS);
}

TEST_F(AivUrmaChannelTest, Ut_Init_WhenBuildsMocked_Returns_SUCCESS)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());

    MOCKER_CPP(&AivUrmaChannel::ParseInputParam, HcclResult(AivUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AivUrmaChannel::StartListen, HcclResult(AivUrmaChannel::*)())
        .expects(once())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AivUrmaChannel::BuildSocket, HcclResult(AivUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AivUrmaChannel::BuildAttr, HcclResult(AivUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AivUrmaChannel::BuildConnection, HcclResult(AivUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AivUrmaChannel::BuildAivUrmaTransport, HcclResult(AivUrmaChannel::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(ch.Init(), HCCL_SUCCESS);
}

TEST_F(AivUrmaChannelTest, Ut_BuildAttr_WhenCalled_FillsAivOpBaseAttr)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    ch.localEp_.loc.device.devPhyId = 3;

    EXPECT_EQ(ch.BuildAttr(), HCCL_SUCCESS);
    EXPECT_EQ(ch.attr_.devicePhyId, 3);
    EXPECT_EQ(ch.attr_.opMode, OpMode::OPBASE);
    EXPECT_EQ(ch.attr_.opAcceState, AcceleratorState::AIV);
}

TEST_F(AivUrmaChannelTest, Ut_BuildConnection_When_UbProtocol_Expect_RetainsConfiguredSqDepth)
{
    enum class ConnectionType {
        TP,
        CTP,
        UBG,
    };

    struct TestCase {
        CommProtocol protocol;
        ConnectionType connectionType;
    };

    const TestCase testCases[] = {
        {COMM_PROTOCOL_UBC_TP, ConnectionType::TP},
        {COMM_PROTOCOL_UBC_CTP, ConnectionType::CTP},
        {COMM_PROTOCOL_UBG, ConnectionType::UBG},
    };

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    for (const auto& testCase : testCases) {
        SCOPED_TRACE(testing::Message() << "protocol=" << testCase.protocol);
        HcommChannelDesc desc = MakeDefaultDesc();
        desc.ubAttr.sqDepth = 512U;
        AivUrmaChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
        ch.localEp_.protocol = testCase.protocol;
        ch.localEp_.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        ch.localEp_.commAddr.addr = IpAddress("1.0.0.1").GetBinaryAddress().addr;
        ch.remoteEp_.protocol = testCase.protocol;
        ch.remoteEp_.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        ch.remoteEp_.commAddr.addr = IpAddress("2.0.0.1").GetBinaryAddress().addr;
        ch.rdmaHandle_ = reinterpret_cast<RdmaHandle>(0x1);
        ch.devBaseAttr_.sqMaxDepth = 1024U;

        ASSERT_EQ(ch.BuildConnection(), HCCL_SUCCESS);
        ASSERT_EQ(ch.connections_.size(), 1U);
        EXPECT_EQ(ch.connections_[0]->GetSqDepth(), 512U);
        switch (testCase.connectionType) {
            case ConnectionType::TP:
                EXPECT_NE(dynamic_cast<DevUbTpConnection*>(ch.connections_[0].get()), nullptr);
                break;
            case ConnectionType::CTP:
                EXPECT_NE(dynamic_cast<DevUbCtpConnection*>(ch.connections_[0].get()), nullptr);
                break;
            case ConnectionType::UBG:
                EXPECT_NE(dynamic_cast<DevUbRtpConnection*>(ch.connections_[0].get()), nullptr);
                break;
        }
    }
}

TEST_F(AivUrmaChannelTest, Ut_BuildSocket_WhenSocketExists_Returns_SUCCESS)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    ch.socket_ = &socket;

    EXPECT_EQ(ch.BuildSocket(), HCCL_SUCCESS);
    ch.socket_ = nullptr;
}

TEST_F(AivUrmaChannelTest, Ut_BuildSocket_WhenSocketNull_GetsSocketFromSocketMgr)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 61001;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    AivUrmaChannel ch(ep, desc);
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    ch.socket_ = nullptr;
    ch.localEp_.loc.device.devPhyId = 3;

    MOCKER_CPP(&SocketMgr::GetSocket).stubs().with(mockcpp::any(), outBound(&socket)).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(ch.BuildSocket(), HCCL_SUCCESS);
    EXPECT_EQ(ch.socket_, &socket);
    ASSERT_NE(ch.socketConfigHolder_, nullptr);
    EXPECT_EQ(ch.socketConfig_, ch.socketConfigHolder_.get());
    EXPECT_EQ(ch.socketConfig_->listeningPort, desc.port);
    EXPECT_EQ(ch.socketConfig_->GetRole(), SocketRole::SERVER);
    ch.socket_ = nullptr;
}

TEST_F(AivUrmaChannelTest, Ut_BuildSocket_WhenRoleReserved_PreservesNoRankIdMatching)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 61002;
    desc.role = HCOMM_SOCKET_ROLE_RESERVED;
    AivUrmaChannel ch(ep, desc);
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    ch.socket_ = nullptr;
    ch.localEp_.loc.device.devPhyId = 3;

    MOCKER_CPP(&SocketMgr::GetSocket).stubs().with(mockcpp::any(), outBound(&socket)).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(ch.BuildSocket(), HCCL_SUCCESS);
    EXPECT_EQ(ch.socket_, &socket);
    ASSERT_NE(ch.socketConfigHolder_, nullptr);
    EXPECT_TRUE(ch.socketConfig_->noRankId);
    EXPECT_EQ(ch.socketConfig_->listeningPort, desc.port);
    SocketRole expectedRole = ch.socketConfig_->link.GetLocalAddr() < ch.socketConfig_->link.GetRemoteAddr() ?
                                  SocketRole::SERVER :
                                  SocketRole::CLIENT;
    EXPECT_EQ(ch.socketConfig_->GetRole(), expectedRole);
    ch.socket_ = nullptr;
}

TEST_F(AivUrmaChannelTest, Ut_StartListen_WhenPortConfigured_UsesConfiguredPort)
{
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 61001;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, desc);
    g_listenPort = 0;

    MOCKER(HcommEndpointStartListen).stubs().will(invoke(StubEndpointStartListen));

    EXPECT_EQ(ch.StartListen(), HCCL_SUCCESS);
    EXPECT_EQ(g_listenPort, desc.port);
}

TEST_F(AivUrmaChannelTest, Ut_StartListen_WhenPortZero_UsesDefaultPort)
{
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 0;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, desc);
    g_listenPort = 0;

    MOCKER(HcommEndpointStartListen).stubs().will(invoke(StubEndpointStartListen));

    EXPECT_EQ(ch.StartListen(), HCCL_SUCCESS);
    EXPECT_EQ(g_listenPort, 60001);
}

TEST_F(AivUrmaChannelTest, Ut_StartListen_WhenRoleClient_DoesNotStartListen)
{
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 61001;
    desc.role = HCOMM_SOCKET_ROLE_CLIENT;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, desc);
    g_listenPort = 0;

    MOCKER(HcommEndpointStartListen).expects(never());

    EXPECT_EQ(ch.StartListen(), HCCL_SUCCESS);
    EXPECT_EQ(g_listenPort, 0);
}

TEST_F(AivUrmaChannelTest, Ut_StartListen_WhenRoleReserved_DoesNotStartListen)
{
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.port = 61002;
    desc.role = HCOMM_SOCKET_ROLE_RESERVED;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, desc);
    g_listenPort = 0;

    MOCKER(HcommEndpointStartListen).expects(never());

    EXPECT_EQ(ch.StartListen(), HCCL_SUCCESS);
    EXPECT_EQ(g_listenPort, 0);
}

TEST_F(AivUrmaChannelTest, Ut_PutSocketIfNeeded_WhenCalledTwice_OnlyKeepsSocketNull)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    std::string socketTag = "ut";
    Hccl::SocketConfig socketConfig(MakeDefaultLinkData(), socketTag, true);
    ch.socket_ = &socket;
    ch.socketConfig_ = &socketConfig;

    MOCKER_CPP(&SocketMgr::PutSocket).stubs().will(returnValue(HCCL_SUCCESS));

    ch.PutSocketIfNeeded();
    EXPECT_EQ(ch.socket_, nullptr);

    ch.PutSocketIfNeeded();
    EXPECT_EQ(ch.socket_, nullptr);
}

TEST_F(AivUrmaChannelTest, Ut_UnsupportedDataApis_WhenCalled_Return_NOT_SUPPORT)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    uint32_t notifyNum = 1;

    EXPECT_EQ(ch.GetNotifyNum(&notifyNum), HCCL_SUCCESS);
    EXPECT_EQ(ch.NotifyRecord(0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.NotifyWait(0, 0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.WriteWithNotify(nullptr, nullptr, 0, 0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.Write(nullptr, nullptr, 0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.Read(nullptr, nullptr, 0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.ChannelFence(), HCCL_E_NOT_SUPPORT);
}

TEST_F(AivUrmaChannelTest, Ut_Init_WhenEndpointNull_Returns_E_PTR)
{
    AivUrmaChannel ch(nullptr, MakeDefaultDesc());

    EXPECT_EQ(ch.Init(), HCCL_E_PTR);
}

TEST_F(AivUrmaChannelTest, Ut_ParseInputParam_WhenExchangeAllMemsGetFailed_ReturnsError)
{
    EndpointDesc local{};
    local.loc.device.devPhyId = 3;
    StubEndpointForAivUrmaChannel endpoint(local, reinterpret_cast<void*>(0x1));
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.exchangeAllMems = true;
    AivUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(HcommMemGetAllMemHandles).stubs().will(invoke(StubHcommMemGetAllMemHandlesFail));

    EXPECT_EQ(ch.ParseInputParam(), HCCL_E_INTERNAL);
    EXPECT_TRUE(ch.commonRes_.bufferVec.empty());
}

TEST_F(AivUrmaChannelTest, Ut_ParseInputParam_WhenExchangeAllMemsEmpty_ReturnsSuccess)
{
    EndpointDesc local{};
    local.loc.device.devPhyId = 3;
    StubEndpointForAivUrmaChannel endpoint(local, reinterpret_cast<void*>(0x1));
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.exchangeAllMems = true;
    AivUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(HcommMemGetAllMemHandles).stubs().will(invoke(StubHcommMemGetAllMemHandlesEmpty));

    EXPECT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_TRUE(ch.commonRes_.bufferVec.empty());
    EXPECT_EQ(ch.devicePhyId_, 3);
}

TEST_F(AivUrmaChannelTest, UT_ParseInputParam_When_ExchangeAllMemsHasOneHandle_Expect_FillCommonRes)
{
    EndpointDesc local{};
    local.loc.device.devPhyId = 3;
    StubEndpointForAivUrmaChannel endpoint(local, reinterpret_cast<void*>(0x1));
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.exchangeAllMems = true;
    AivUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(HcommMemGetAllMemHandles).stubs().will(invoke(StubHcommMemGetAllMemHandlesOne));

    EXPECT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.commonRes_.bufferVec.size(), 1U);
    ASSERT_NE(ch.commonRes_.bufferVec[0], nullptr);
    EXPECT_EQ(ch.commonRes_.bufferVec[0]->GetAddr(), 0x12340000U);
    EXPECT_EQ(ch.commonRes_.bufferVec[0]->GetSize(), 0x2000U);
    EXPECT_FALSE(ch.commonRes_.bufferVec[0]->GetBuf()->GetMemInfo().empty());
}

TEST_F(AivUrmaChannelTest, UT_ParseInputParam_When_ExchangeAllMemsFalse_Expect_FillCommonRes)
{
    EndpointDesc local{};
    local.loc.device.devPhyId = 3;
    StubEndpointForAivUrmaChannel endpoint(local, reinterpret_cast<void*>(0x1));
    auto buffer = std::make_shared<Hccl::Buffer>(0x22340000U, 0x1000U, "aiv_urma_desc");
    auto localBuffer = std::make_shared<Hccl::LocalUbRmaBuffer>(buffer);
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};
    HcommChannelDesc desc = MakeDefaultDesc();
    desc.exchangeAllMems = false;
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AivUrmaChannel ch(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    EXPECT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.commonRes_.bufferVec.size(), 1U);
    EXPECT_EQ(ch.commonRes_.bufferVec[0], localBuffer.get());
}

TEST_F(AivUrmaChannelTest, Ut_BuildChannelEntityToDevice_WhenDevPtrNull_Returns_E_PTR)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());

    EXPECT_EQ(ch.BuildChannelEntityToDevice(nullptr), HCCL_E_PTR);
}

TEST_F(AivUrmaChannelTest, Ut_BuildChannelEntityToDevice_WhenTransportNull_Returns_E_PTR)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    void* devChannelPtr = nullptr;

    EXPECT_EQ(ch.BuildChannelEntityToDevice(&devChannelPtr), HCCL_E_PTR);
    EXPECT_EQ(devChannelPtr, nullptr);
}

TEST_F(AivUrmaChannelTest, Ut_GetStatus_WhenTransportStatusTimeout_Returns_SOCKET_TIMEOUT)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());

    BaseMemTransport::CommonLocRes commonRes{};
    DevUbCtpConnection conn(reinterpret_cast<RdmaHandle>(0x1), IpAddress(), IpAddress(), OpMode::OPBASE);
    commonRes.connVec.emplace_back(&conn);
    BaseMemTransport::Attribution attr{};
    LinkData linkData = MakeDefaultLinkData();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    ch.transport_
        = std::make_unique<AivUrmaTransport>(commonRes, attr, linkData, socket, reinterpret_cast<RdmaHandle>(0x1));
    SocketStatus fakeSocketStatus = SocketStatus::TIMEOUT;

    MOCKER_CPP(&Socket::GetAsyncStatus, SocketStatus(Socket::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(fakeSocketStatus));

    EXPECT_EQ(ch.GetStatus(), ChannelStatus::SOCKET_TIMEOUT);
}

TEST_F(AivUrmaChannelTest, Ut_BuildChannelEntityToDevice_WhenTransportReady_ReturnsDevicePtr)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    DevUbCtpConnection conn(reinterpret_cast<RdmaHandle>(0x1), IpAddress(), IpAddress(), OpMode::OPBASE);
    BuildReadyTransport(ch, conn);

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));

    void* devChannelPtr = nullptr;
    EXPECT_EQ(ch.BuildChannelEntityToDevice(&devChannelPtr), HCCL_SUCCESS);
    ASSERT_NE(devChannelPtr, nullptr);
    EXPECT_EQ(ch.devChannelEntity_, devChannelPtr);
    ASSERT_NE(ch.devChannelEntitySlab_, nullptr);
    EXPECT_EQ(ch.devChannelEntitySlab_, devChannelPtr);
    EXPECT_GT(ch.devChannelEntitySlabSize_, sizeof(ChannelEntity));
    EXPECT_TRUE(ch.deviceMemories_.empty());

    ChannelEntity* devChannel = reinterpret_cast<ChannelEntity*>(devChannelPtr);
    EXPECT_EQ(devChannel->engine, COMM_ENGINE_AIV);
    EXPECT_EQ(devChannel->sqNum, 1);
    EXPECT_EQ(devChannel->cqNum, 1);
    EXPECT_NE(devChannel->sqContextAddr, nullptr);
    EXPECT_NE(devChannel->cqContextAddr, nullptr);
    EXPECT_TRUE(IsPtrInRange(devChannel->sqContextAddr, ch.devChannelEntitySlab_, ch.devChannelEntitySlabSize_));
    EXPECT_TRUE(IsPtrInRange(devChannel->cqContextAddr, ch.devChannelEntitySlab_, ch.devChannelEntitySlabSize_));
    EXPECT_TRUE(IsPtrInRange(
        reinterpret_cast<void*>(devChannel->sqContextAddr[0].contextInfo.ubJfs.headAddr), ch.devChannelEntitySlab_,
        ch.devChannelEntitySlabSize_));
    EXPECT_TRUE(IsPtrInRange(
        reinterpret_cast<void*>(devChannel->sqContextAddr[0].contextInfo.ubJfs.tailAddr), ch.devChannelEntitySlab_,
        ch.devChannelEntitySlabSize_));
    EXPECT_TRUE(IsPtrInRange(
        reinterpret_cast<void*>(devChannel->cqContextAddr[0].contextInfo.ubJfc.headAddr), ch.devChannelEntitySlab_,
        ch.devChannelEntitySlabSize_));
    EXPECT_TRUE(IsPtrInRange(
        reinterpret_cast<void*>(devChannel->cqContextAddr[0].contextInfo.ubJfc.tailAddr), ch.devChannelEntitySlab_,
        ch.devChannelEntitySlabSize_));
}

TEST_F(AivUrmaChannelTest, Ut_BuildChannelEntityToDevice_WhenAclrtMallocFail_ReturnsMemoryError)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    DevUbCtpConnection conn(reinterpret_cast<RdmaHandle>(0x1), IpAddress(), IpAddress(), OpMode::OPBASE);
    BuildReadyTransport(ch, conn);

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMallocFail));

    void* devChannelPtr = nullptr;
    EXPECT_EQ(ch.BuildChannelEntityToDevice(&devChannelPtr), HCCL_E_MEMORY);
    EXPECT_EQ(devChannelPtr, nullptr);
    EXPECT_EQ(ch.devChannelEntitySlab_, nullptr);
    EXPECT_EQ(ch.devChannelEntity_, nullptr);
}

TEST_F(AivUrmaChannelTest, Ut_Clean_WhenDeviceChannelEntityBuilt_ReleasesSlab)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    DevUbCtpConnection conn(reinterpret_cast<RdmaHandle>(0x1), IpAddress(), IpAddress(), OpMode::OPBASE);
    BuildReadyTransport(ch, conn);

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));

    void* devChannelPtr = nullptr;
    ASSERT_EQ(ch.BuildChannelEntityToDevice(&devChannelPtr), HCCL_SUCCESS);
    ASSERT_NE(ch.devChannelEntitySlab_, nullptr);

    EXPECT_EQ(ch.Clean(), HCCL_SUCCESS);
    EXPECT_EQ(ch.devChannelEntitySlab_, nullptr);
    EXPECT_EQ(ch.devChannelEntitySlabSize_, 0);
    EXPECT_EQ(ch.devChannelEntity_, nullptr);
}

class AivUrmaTransportTest : public testing::Test {
protected:
    static void TearDownTestCase() { GlobalMockObject::verify(); }

    void TearDown() override { GlobalMockObject::verify(); }

    std::unique_ptr<DevUbCtpConnection> MakeConn()
    {
        return std::make_unique<DevUbCtpConnection>(
            reinterpret_cast<RdmaHandle>(0x1), IpAddress(), IpAddress(), OpMode::OPBASE);
    }

    std::unique_ptr<AivUrmaTransport> MakeTransport(DevUbConnection& conn, Socket& socket)
    {
        commonRes_.connVec.clear();
        commonRes_.bufferVec.clear();
        commonRes_.connVec.emplace_back(&conn);
        attr_.opAcceState = AcceleratorState::AIV;
        LinkData linkData = MakeDefaultLinkData();
        return std::make_unique<AivUrmaTransport>(
            commonRes_, attr_, linkData, socket, reinterpret_cast<RdmaHandle>(0x1));
    }

    BaseMemTransport::CommonLocRes commonRes_{};
    BaseMemTransport::Attribution attr_{};
};

TEST_F(AivUrmaTransportTest, Ut_Construct_WhenConnNull_ThrowsInvalidParamsException)
{
    BaseMemTransport::CommonLocRes commonRes{};
    commonRes.connVec.emplace_back(nullptr);
    BaseMemTransport::Attribution attr{};
    LinkData linkData = MakeDefaultLinkData();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);

    EXPECT_THROW(
        AivUrmaTransport transport(commonRes, attr, linkData, socket, reinterpret_cast<RdmaHandle>(0x1)),
        InvalidParamsException);
}

TEST_F(AivUrmaTransportTest, Ut_DescribeAndLinkDesc_WhenCalled_ReturnsNonEmpty)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);

    EXPECT_FALSE(transport->Describe().empty());
    EXPECT_FALSE(transport->GetLinkDescInfo().empty());
}

TEST_F(AivUrmaTransportTest, Ut_CheckCommonLocRes_WhenConnValid_Returns)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);

    EXPECT_NO_THROW(transport->CheckLocBuffer(commonRes_));
    EXPECT_NO_THROW(transport->CheckLocConn(commonRes_));
    EXPECT_NO_THROW(transport->CheckCommonLocRes(commonRes_));
}

TEST_F(AivUrmaTransportTest, Ut_HandshakePackUnpack_WhenEmptyMsg_Returns)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    BinaryStream binaryStream;

    transport->HandshakeMsgPack(binaryStream);
    EXPECT_NO_THROW(transport->HandshakeMsgUnpack(binaryStream));
    EXPECT_EQ(transport->rmtOpAcceState_, AcceleratorState::AIV);
}

TEST_F(AivUrmaTransportTest, Ut_HandshakeUnpack_WhenAcceleratorMismatch_Throws)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    BinaryStream binaryStream;
    binaryStream << static_cast<uint32_t>(AcceleratorState::CCU_MS);
    binaryStream << std::vector<char>{};

    EXPECT_THROW(transport->HandshakeMsgUnpack(binaryStream), InvalidParamsException);
}

TEST_F(AivUrmaTransportTest, Ut_HandshakeUnpack_WhenMsgSizeMismatch_Throws)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    BinaryStream binaryStream;
    binaryStream << static_cast<uint32_t>(AcceleratorState::AIV);
    binaryStream << std::vector<char>{'x'};

    EXPECT_THROW(transport->HandshakeMsgUnpack(binaryStream), InvalidParamsException);
}

TEST_F(AivUrmaTransportTest, Ut_BufferVecPack_WhenCalled_FillsBinaryStream)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    BinaryStream binaryStream;

    EXPECT_NO_THROW(transport->BufferVecPack(binaryStream));
}

TEST_F(AivUrmaTransportTest, Ut_BufferVecPack_WhenBufferVecHasLocalBuffer_PreservesRegisteredToken)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    auto buffer = std::make_shared<Hccl::Buffer>(0x99880000U, 0x1000U, "aiv_urma_qpt");
    auto localBuffer = std::make_shared<StubLocalUbRmaBufferWithExchangeDto>(buffer);
    transport->commonLocRes_.bufferVec.emplace_back(localBuffer.get());
    BinaryStream binaryStream;

    EXPECT_NO_THROW(transport->BufferVecPack(binaryStream));
    uint32_t bufferNum = 0;
    uint32_t pos = 0;
    ExchangeUbBufferDto dto;
    binaryStream >> bufferNum >> pos;
    dto.Deserialize(binaryStream);

    EXPECT_EQ(bufferNum, 1U);
    EXPECT_EQ(pos, 0U);
    EXPECT_EQ(dto.tokenId, STUB_TOKEN_ID);
    EXPECT_EQ(dto.tokenValue, STUB_TOKEN_VALUE);
}

TEST_F(AivUrmaTransportTest, Ut_IsResReadyAndConnsReady_WhenConnReady_ReturnsTrue)
{
    auto conn = MakeConn();
    conn->status = RmaConnStatus::READY;
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);

    EXPECT_TRUE(transport->IsResReady());
    EXPECT_TRUE(transport->IsConnsReady());
}

TEST_F(AivUrmaTransportTest, Ut_PrepareGetStatus_WhenAlreadyReady_ReturnsFalse)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->transportStatus_ = TransportStatus::READY;

    EXPECT_FALSE(transport->PrepareGetStatus());
    EXPECT_EQ(transport->transportStatus_, TransportStatus::READY);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenInit_SetsSocketOk)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::INIT;

    transport->ProcessUrmaStatus();

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::SOCKET_OK);
    EXPECT_EQ(transport->transportStatus_, TransportStatus::SOCKET_OK);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenSocketOkAndResNotReady_StaysSocketOk)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::SOCKET_OK;

    MOCKER_CPP(&AivUrmaTransport::IsResReady).stubs().will(returnValue(false));

    transport->ProcessUrmaStatus();

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::SOCKET_OK);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenSocketOkAndResReady_SendsData)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::SOCKET_OK;

    MOCKER_CPP(&AivUrmaTransport::IsResReady).stubs().will(returnValue(true));
    MOCKER_CPP(&AivUrmaTransport::SendExchangeData).stubs().will(ignoreReturnValue());

    transport->ProcessUrmaStatus();

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::SEND_DATA);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenSendData_RecvsData)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::SEND_DATA;

    MOCKER_CPP(&AivUrmaTransport::RecvExchangeData).stubs().will(ignoreReturnValue());

    transport->ProcessUrmaStatus();

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::RECV_DATA);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenRecvDataNeedsFinish_ProcessesData)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::RECV_DATA;

    MOCKER_CPP(&AivUrmaTransport::RecvDataProcess).stubs().will(returnValue(true));

    transport->ProcessUrmaStatus();

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::PROCESS_DATA);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenRecvDataNeedsNoFinish_BecomesReady)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::RECV_DATA;

    MOCKER_CPP(&AivUrmaTransport::RecvDataProcess).stubs().will(returnValue(false));

    transport->ProcessUrmaStatus();

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::RECV_FIN);
    EXPECT_EQ(transport->transportStatus_, TransportStatus::READY);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenProcessDataAndConnsReady_SendsFinish)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::PROCESS_DATA;

    MOCKER_CPP(&AivUrmaTransport::IsConnsReady).stubs().will(returnValue(true));
    MOCKER_CPP(&AivUrmaTransport::SendFinish).stubs().will(ignoreReturnValue());

    transport->ProcessUrmaStatus();

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::CONN_OK);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenConnOk_RecvsFinish)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::CONN_OK;

    MOCKER_CPP(&AivUrmaTransport::RecvFinish).stubs().will(ignoreReturnValue());

    transport->ProcessUrmaStatus();

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::SEND_FIN);
}

TEST_F(AivUrmaTransportTest, Ut_ProcessUrmaStatus_WhenSendFin_BecomesReady)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->urmaStatus_ = AivUrmaTransport::UrmaStatus::SEND_FIN;

    transport->ProcessUrmaStatus();

    EXPECT_EQ(transport->urmaStatus_, AivUrmaTransport::UrmaStatus::RECV_FIN);
    EXPECT_EQ(transport->transportStatus_, TransportStatus::READY);
}

TEST_F(AivUrmaTransportTest, Ut_RmtBufferVecUnpackProc_WhenZeroSizeDto_AppendsNullBuffer)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    BinaryStream binaryStream;
    ExchangeUbBufferDto dto{};
    binaryStream << static_cast<uint32_t>(1);
    binaryStream << static_cast<uint32_t>(0);
    dto.Serialize(binaryStream);
    AivUrmaTransport::RemoteBufferVec bufferVec;

    transport->RmtBufferVecUnpackProc(1, binaryStream, bufferVec);

    ASSERT_EQ(bufferVec.size(), 1);
    EXPECT_EQ(bufferVec[0], nullptr);
}

TEST_F(AivUrmaTransportTest, Ut_RmtBufferVecUnpackProc_WhenNumMismatch_Throws)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    BinaryStream binaryStream;
    binaryStream << static_cast<uint32_t>(2);
    AivUrmaTransport::RemoteBufferVec bufferVec;

    EXPECT_THROW(transport->RmtBufferVecUnpackProc(1, binaryStream, bufferVec), InvalidParamsException);
}

TEST_F(AivUrmaTransportTest, Ut_ConnVecUnpackProc_WhenConnNumZero_ReturnsFalse)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->connNum_ = 0;
    BinaryStream binaryStream;
    binaryStream << static_cast<uint32_t>(0);

    EXPECT_FALSE(transport->ConnVecUnpackProc(binaryStream));
}

TEST_F(AivUrmaTransportTest, Ut_ConnVecUnpackProc_WhenConnNumMismatch_Throws)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    BinaryStream binaryStream;
    binaryStream << static_cast<uint32_t>(2);

    EXPECT_THROW(transport->ConnVecUnpackProc(binaryStream), InvalidParamsException);
}

TEST_F(AivUrmaTransportTest, Ut_GetRemoteMems_WhenParamNull_Returns_E_PARA)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    uint32_t memNum = 0;

    EXPECT_EQ(transport->GetRemoteMems(&memNum, nullptr, &memInfos), HCCL_E_PARA);
    EXPECT_EQ(transport->GetRemoteMems(&memNum, &remoteMem, nullptr), HCCL_E_PARA);
    EXPECT_EQ(transport->GetRemoteMems(nullptr, &remoteMem, &memInfos), HCCL_E_PARA);
}

TEST_F(AivUrmaTransportTest, Ut_GetRemoteMems_WhenNoRemoteBuffer_Returns_SUCCESS)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    CommMem* remoteMem = reinterpret_cast<CommMem*>(0x1);
    char** memInfos = reinterpret_cast<char**>(0x1);
    uint32_t memNum = 0;

    EXPECT_EQ(transport->GetRemoteMems(&memNum, &remoteMem, &memInfos), HCCL_SUCCESS);
}

TEST_F(AivUrmaTransportTest, Ut_GetRemoteMems_WhenOnlyReservedRemoteBuffer_Returns_SUCCESS)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    void* rdmaHandle = (void*)0x100;
    auto remoteRmaBuffer = std::make_unique<Hccl::RemoteUbRmaBuffer>(rdmaHandle);
    transport->rmtBufferVec_.push_back(std::move(remoteRmaBuffer));
    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    uint32_t memNum = 1;

    EXPECT_EQ(transport->GetRemoteMems(&memNum, &remoteMem, &memInfos), HCCL_SUCCESS);
    EXPECT_EQ(memNum, 1);
}

TEST_F(AivUrmaTransportTest, Ut_GetStatus_WhenSocketTimeout_Returns_SOCKET_TIMEOUT)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    SocketStatus fakeSocketStatus = SocketStatus::TIMEOUT;

    MOCKER_CPP(&Socket::GetAsyncStatus, SocketStatus(Socket::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(fakeSocketStatus));

    EXPECT_EQ(transport->GetStatus(), TransportStatus::SOCKET_TIMEOUT);
}

TEST_F(AivUrmaTransportTest, Ut_GetHostChannelEntity_WhenNotReady_ThrowsInternalException)
{
    auto conn = MakeConn();
    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    ChannelEntity hostChannel{};

    EXPECT_THROW(transport->GetHostChannelEntity(&hostChannel), InternalException);
}

TEST_F(AivUrmaTransportTest, Ut_GetHostChannelEntity_WhenReady_FillsSqAndCqContext)
{
    auto conn = MakeConn();
    conn->status = RmaConnStatus::READY;
    conn->jettyId = 3;
    conn->dbAddr = 0x1000;
    conn->sqBuffVa = 0x2000;
    conn->sqDepth = 4;
    conn->tpn = 5;
    conn->cqInfo_.id = 6;
    conn->cqInfo_.va = 0x3000;
    conn->cqInfo_.cqeSize = 64;
    conn->cqInfo_.cqDepth = 8;
    conn->cqInfo_.swdbAddr = 0x4000;

    Socket socket(nullptr, IpAddress(), 0, IpAddress(), "ut", SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    auto transport = MakeTransport(*conn, socket);
    transport->transportStatus_ = TransportStatus::READY;
    ChannelEntity hostChannel{};

    using DeviceMemAllocRetByValue = hccl::DeviceMem (*)(u64, bool);
    mockcpp::mockAPI<DeviceMemAllocRetByValue>::get(
        "hccl::DeviceMem::alloc", "AivUrmaTransport_DeviceMem_alloc_by_value",
        static_cast<DeviceMemAllocRetByValue>(&hccl::DeviceMem::alloc))
        .stubs()
        .will(invoke(StubDeviceMemAlloc));

    transport->GetHostChannelEntity(&hostChannel);

    ASSERT_EQ(hostChannel.sqNum, 1);
    ASSERT_NE(hostChannel.sqContextAddr, nullptr);
    EXPECT_EQ(hostChannel.sqContextAddr[0].type, SQ_CONTEXT_TYPE_UB_JFS);
    EXPECT_EQ(hostChannel.sqContextAddr[0].contextInfo.ubJfs.jfsID, conn->jettyId);
    EXPECT_EQ(hostChannel.sqContextAddr[0].contextInfo.ubJfs.dbVa, conn->dbAddr);
    EXPECT_EQ(hostChannel.sqContextAddr[0].contextInfo.ubJfs.sqVa, conn->sqBuffVa);
    EXPECT_EQ(hostChannel.sqContextAddr[0].contextInfo.ubJfs.tpID, conn->tpn);
    EXPECT_EQ(hostChannel.sqContextAddr[0].contextInfo.ubJfs.wqeSize, 64);
    EXPECT_NE(hostChannel.sqContextAddr[0].contextInfo.ubJfs.headAddr, 0);
    EXPECT_NE(hostChannel.sqContextAddr[0].contextInfo.ubJfs.tailAddr, 0);

    ASSERT_EQ(hostChannel.cqNum, 1);
    ASSERT_NE(hostChannel.cqContextAddr, nullptr);
    EXPECT_EQ(hostChannel.cqContextAddr[0].type, CQ_CONTEXT_TYPE_UB_JFC);
    EXPECT_EQ(hostChannel.cqContextAddr[0].contextInfo.ubJfc.jfcID, conn->cqInfo_.id);
    EXPECT_EQ(hostChannel.cqContextAddr[0].contextInfo.ubJfc.scqVa, conn->cqInfo_.va);
    EXPECT_EQ(hostChannel.cqContextAddr[0].contextInfo.ubJfc.cqeSize, conn->cqInfo_.cqeSize);
    EXPECT_EQ(hostChannel.cqContextAddr[0].contextInfo.ubJfc.cqDepth, conn->cqInfo_.cqDepth);
    EXPECT_EQ(hostChannel.cqContextAddr[0].contextInfo.ubJfc.dbVa, conn->cqInfo_.swdbAddr);
    EXPECT_NE(hostChannel.cqContextAddr[0].contextInfo.ubJfc.headAddr, 0);
    EXPECT_NE(hostChannel.cqContextAddr[0].contextInfo.ubJfc.tailAddr, 0);
}

// ===================== PreAllocChannelEntityToDevice / FillChannelEntityToDevice UT =====================

// PreAllocChannelEntityToDevice: devChannelPtr 为 nullptr
TEST_F(AivUrmaChannelTest, Ut_PreAllocChannelEntityToDevice_When_DevPtrNull_Returns_E_PTR)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    EXPECT_EQ(ch.PreAllocChannelEntityToDevice(nullptr), HCCL_E_PTR);
}

// PreAllocChannelEntityToDevice: transport_ 为 nullptr
TEST_F(AivUrmaChannelTest, Ut_PreAllocChannelEntityToDevice_When_TransportNull_Returns_E_PTR)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    void* devChannelPtr = nullptr;
    EXPECT_EQ(ch.PreAllocChannelEntityToDevice(&devChannelPtr), HCCL_E_PTR);
    EXPECT_EQ(devChannelPtr, nullptr);
}

// FillChannelEntityToDevice: 未预分配（slab 为 nullptr）→ E_INTERNAL
TEST_F(AivUrmaChannelTest, Ut_FillChannelEntityToDevice_When_NotPreAllocated_Returns_E_INTERNAL)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    EXPECT_EQ(ch.FillChannelEntityToDevice(), HCCL_E_INTERNAL);
}

// ===================== PreAllocChannelEntityToDevice / FillChannelEntityToDevice 正常路径 UT =====================

// PreAllocChannelEntityToDevice: transport ready，正常分配 slab
TEST_F(AivUrmaChannelTest, Ut_PreAllocChannelEntityToDevice_When_TransportReady_ReturnsDevicePtr)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    DevUbCtpConnection conn(reinterpret_cast<RdmaHandle>(0x1), IpAddress(), IpAddress(), OpMode::OPBASE);
    BuildReadyTransport(ch, conn);

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));

    void* devChannelPtr = nullptr;
    EXPECT_EQ(ch.PreAllocChannelEntityToDevice(&devChannelPtr), HCCL_SUCCESS);
    ASSERT_NE(devChannelPtr, nullptr);
    ASSERT_NE(ch.devChannelEntitySlab_, nullptr);
    EXPECT_EQ(ch.devChannelEntity_, devChannelPtr);
    EXPECT_GT(ch.devChannelEntitySlabSize_, 0);
}

// PreAllocChannelEntityToDevice: 已分配则返回缓存指针
TEST_F(AivUrmaChannelTest, Ut_PreAllocChannelEntityToDevice_When_AlreadyBuilt_ReturnsCached)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    DevUbCtpConnection conn(reinterpret_cast<RdmaHandle>(0x1), IpAddress(), IpAddress(), OpMode::OPBASE);
    BuildReadyTransport(ch, conn);

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));

    void* firstPtr = nullptr;
    ASSERT_EQ(ch.PreAllocChannelEntityToDevice(&firstPtr), HCCL_SUCCESS);
    void* secondPtr = nullptr;
    EXPECT_EQ(ch.PreAllocChannelEntityToDevice(&secondPtr), HCCL_SUCCESS);
    EXPECT_EQ(secondPtr, firstPtr);
}

// FillChannelEntityToDevice: 先 PreAlloc 再 Fill，验证成功
TEST_F(AivUrmaChannelTest, Ut_FillChannelEntityToDevice_When_PreAllocated_ReturnsSuccess)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AivUrmaChannel ch(ep, MakeDefaultDesc());
    DevUbCtpConnection conn(reinterpret_cast<RdmaHandle>(0x1), IpAddress(), IpAddress(), OpMode::OPBASE);
    BuildReadyTransport(ch, conn);

    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(StubAclrtMalloc));
    MOCKER(aclrtFree).stubs().will(invoke(StubAclrtFree));
    using HrtMemcpyType = void (*)(void*, uint64_t, const void*, uint64_t, Hccl::tagRtMemcpyKind);
    MOCKER((Hccl::HrtMemcpy)).stubs().will(invoke(static_cast<HrtMemcpyType>(StubHrtMemcpyReal)));

    void* devChannelPtr = nullptr;
    ASSERT_EQ(ch.PreAllocChannelEntityToDevice(&devChannelPtr), HCCL_SUCCESS);
    EXPECT_EQ(ch.FillChannelEntityToDevice(), HCCL_SUCCESS);
    ASSERT_NE(ch.devChannelEntity_, nullptr);
}
