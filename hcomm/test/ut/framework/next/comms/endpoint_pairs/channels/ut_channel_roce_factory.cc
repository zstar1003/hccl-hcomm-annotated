/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <arpa/inet.h>

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "aicpu/aicpu_ts_roce_channel.h"
#include "aicpu/aicpu_ts_urma_channel.h"
#include "aiv/aiv_urma_channel.h"
#include "aiv/aiv_ub_mem_channel.h"
#include "ccu_urma_channel.h"
#include "channel.h"
#include "host/host_cpu_roce_channel.h"
#include "hcomm_c_adpt.h"
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"
#include "endpoint.h"
#include "tp_manager.h"

using namespace hcomm;

namespace {
/* Member-function pointer types for MOCKER_CPP_VIRTUAL (vtable hook). MOCKER_CPP on virtual Init is unsafe when
 * Init is invoked through Channel* / unique_ptr<Channel> inside Channel::CreateChannel. */
using HostCpuRoceChannelInitFp = HcclResult (HostCpuRoceChannel::*)();
using AicpuTsRoceChannelInitFp = HcclResult (AicpuTsRoceChannel::*)();
using AicpuTsUrmaChannelInitFp = HcclResult (AicpuTsUrmaChannel::*)();
using AivUbMemChannelInitFp = HcclResult (AivUbMemChannel::*)();
using CcuUrmaChannelInitFp = HcclResult (CcuUrmaChannel::*)();

class StubEndpointForChannelFactory : public Endpoint {
public:
    explicit StubEndpointForChannelFactory(const EndpointDesc& desc, void* rdma = reinterpret_cast<void*>(0x10U))
        : Endpoint(desc),
          ctxHandle_(rdma)
    {}
    HcclResult Init() override { return HCCL_SUCCESS; }
    // 内存方法在 RegedMemMgr 上，Endpoint 不再 override；返回空 RegedMemMgr。
    RegedMemMgr* GetRegedMemMgr() override { return nullptr; }
    void* GetRdmaHandle() override { return ctxHandle_; }
    bool IsCtxHandleValid() const override { return ctxHandle_ != nullptr; }

private:
    void* ctxHandle_{nullptr};
};
} // namespace

class UtChannelRoceFactory : public testing::Test {
protected:
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(UtChannelRoceFactory, CreateChannel_NullEndpoint_AicpuTs_Roce_Returns_E_PTR)
{
    HcommChannelDesc desc{};
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    std::shared_ptr<Channel> ch;
    EXPECT_EQ(Channel::CreateChannel(nullptr, COMM_ENGINE_AICPU_TS, desc, ch), HCCL_E_PTR);
    EXPECT_EQ(ch.get(), nullptr);
}

TEST_F(UtChannelRoceFactory, CreateChannel_NullEndpoint_Aicpu_Roce_Returns_E_PTR)
{
    HcommChannelDesc desc{};
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    std::shared_ptr<Channel> ch;
    EXPECT_EQ(Channel::CreateChannel(nullptr, COMM_ENGINE_AICPU, desc, ch), HCCL_E_PTR);
    EXPECT_EQ(ch.get(), nullptr);
}

TEST_F(UtChannelRoceFactory, CreateChannel_InvalidEngine_Returns_NOT_FOUND)
{
    EndpointDesc ep{};
    ep.protocol = COMM_PROTOCOL_ROCE;
    ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ep.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.20.30.40", &ep.commAddr.addr), 1);
    StubEndpointForChannelFactory stub(ep);
    HcommChannelDesc desc{};
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    std::shared_ptr<Channel> ch;
    EXPECT_EQ(
        Channel::CreateChannel(reinterpret_cast<EndpointHandle>(&stub), static_cast<CommEngine>(0x7fff0001), desc, ch),
        HCCL_E_NOT_FOUND);
    EXPECT_EQ(ch.get(), nullptr);
}

TEST_F(UtChannelRoceFactory, CreateChannel_CpuTs_Returns_NOT_SUPPORT)
{
    EndpointDesc ep{};
    ep.protocol = COMM_PROTOCOL_ROCE;
    ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    StubEndpointForChannelFactory stub(ep);
    HcommChannelDesc desc{};
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    std::shared_ptr<Channel> ch;
    EXPECT_EQ(
        Channel::CreateChannel(reinterpret_cast<EndpointHandle>(&stub), COMM_ENGINE_CPU_TS, desc, ch),
        HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.get(), nullptr);
}

TEST_F(UtChannelRoceFactory, CreateChannel_CpuNonRoce_Returns_NOT_SUPPORT)
{
    EndpointDesc ep{};
    ep.protocol = COMM_PROTOCOL_ROCE;
    ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    StubEndpointForChannelFactory stub(ep);
    HcommChannelDesc desc{};
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_UB_MEM;
    std::shared_ptr<Channel> ch;
    EXPECT_EQ(
        Channel::CreateChannel(reinterpret_cast<EndpointHandle>(&stub), COMM_ENGINE_CPU, desc, ch), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.get(), nullptr);
}

TEST_F(UtChannelRoceFactory, CreateChannel_CpuRoce_MockInit_Returns_SUCCESS)
{
    EndpointDesc ep{};
    ep.protocol = COMM_PROTOCOL_ROCE;
    ep.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    ep.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.20.30.41", &ep.commAddr.addr), 1);
    StubEndpointForChannelFactory stub(ep);
    HcommChannelDesc desc{};
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    auto* raw = new HostCpuRoceChannel(reinterpret_cast<EndpointHandle>(&stub), desc);
    std::unique_ptr<Channel> ch(raw);
    MOCKER_CPP_VIRTUAL(*raw, &HostCpuRoceChannel::Init, HostCpuRoceChannelInitFp)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    ASSERT_EQ(ch->Init(), HCCL_SUCCESS);
    ASSERT_NE(ch.get(), nullptr);
}

TEST_F(UtChannelRoceFactory, CreateChannel_AicpuTsRoce_MockInit_Returns_SUCCESS)
{
    EndpointDesc ep{};
    ep.protocol = COMM_PROTOCOL_ROCE;
    ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ep.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.20.30.42", &ep.commAddr.addr), 1);
    StubEndpointForChannelFactory stub(ep);
    HcommChannelDesc desc{};
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    auto* raw = new AicpuTsRoceChannel(reinterpret_cast<EndpointHandle>(&stub), desc);
    std::unique_ptr<Channel> ch(raw);
    MOCKER_CPP_VIRTUAL(*raw, &AicpuTsRoceChannel::Init, AicpuTsRoceChannelInitFp)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    ASSERT_EQ(ch->Init(), HCCL_SUCCESS);
    ASSERT_NE(ch.get(), nullptr);
}

TEST_F(UtChannelRoceFactory, CreateChannel_AicpuNonRoce_MockUrmaInit_Returns_SUCCESS)
{
    EndpointDesc ep{};
    ep.protocol = COMM_PROTOCOL_ROCE;
    ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    StubEndpointForChannelFactory stub(ep);
    HcommChannelDesc desc{};
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_UB_MEM;
    auto* raw = new AicpuTsUrmaChannel(reinterpret_cast<EndpointHandle>(&stub), desc);
    std::unique_ptr<Channel> ch(raw);
    MOCKER_CPP_VIRTUAL(*raw, &AicpuTsUrmaChannel::Init, AicpuTsUrmaChannelInitFp)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    ASSERT_EQ(ch->Init(), HCCL_SUCCESS);
    ASSERT_NE(ch.get(), nullptr);
}

TEST_F(UtChannelRoceFactory, CreateChannel_AivUbMem_MockInit_Returns_SUCCESS)
{
    EndpointDesc ep{};
    ep.protocol = COMM_PROTOCOL_ROCE;
    ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    StubEndpointForChannelFactory stub(ep);
    HcommChannelDesc desc{};
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_UB_MEM;
    auto* raw = new AivUbMemChannel(reinterpret_cast<EndpointHandle>(&stub), desc);
    std::unique_ptr<Channel> ch(raw);
    MOCKER_CPP_VIRTUAL(*raw, &AivUbMemChannel::Init, AivUbMemChannelInitFp).stubs().will(returnValue(HCCL_SUCCESS));
    ASSERT_EQ(ch->Init(), HCCL_SUCCESS);
    ASSERT_NE(ch.get(), nullptr);
}

TEST_F(UtChannelRoceFactory, CreateChannel_AivUbRtp_On950_ReturnsAivUrmaChannel)
{
    EndpointDesc ep{};
    ep.protocol = COMM_PROTOCOL_UB_RTP;
    ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ep.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.20.30.43", &ep.commAddr.addr), 1);
    StubEndpointForChannelFactory stub(ep);

    HcommChannelDesc desc{};
    desc.remoteEndpoint = ep;
    desc.ubAttr.sqDepth = UB_SQ_DEPTH_NOT_SET;
    Hccl::Socket socket(
        nullptr, Hccl::IpAddress(), 0, Hccl::IpAddress(), "ut", Hccl::SocketRole::CLIENT,
        Hccl::NicType::DEVICE_NIC_TYPE);
    desc.socket = &socket;

    DevType deviceType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::TpManager::Init).stubs();

    std::shared_ptr<Channel> channel;
    EXPECT_EQ(
        Channel::CreateChannel(reinterpret_cast<EndpointHandle>(&stub), COMM_ENGINE_AIV, desc, channel), HCCL_SUCCESS);
    ASSERT_NE(channel, nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<AivUrmaChannel>(channel), nullptr);
}

TEST_F(UtChannelRoceFactory, CreateChannel_CcuUrma_MockInit_Returns_SUCCESS)
{
    EndpointDesc ep{};
    ep.protocol = COMM_PROTOCOL_ROCE;
    ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    StubEndpointForChannelFactory stub(ep);
    HcommChannelDesc desc{};
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_UBC_TP;
    auto* raw = new CcuUrmaChannel(reinterpret_cast<EndpointHandle>(&stub), desc);
    std::unique_ptr<Channel> ch(raw);
    MOCKER_CPP_VIRTUAL(*raw, &CcuUrmaChannel::Init, CcuUrmaChannelInitFp).stubs().will(returnValue(HCCL_SUCCESS));
    ASSERT_EQ(ch->Init(), HCCL_SUCCESS);
    ASSERT_NE(ch.get(), nullptr);
}

namespace {
class ChannelDefaultsTestDouble : public Channel {
public:
    HcclResult Init() override { return HCCL_SUCCESS; }
    HcclResult GetNotifyNum(uint32_t* notifyNum) const override
    {
        *notifyNum = 0U;
        return HCCL_SUCCESS;
    }
    HcclResult GetRemoteMems(uint32_t*, CommMem**, char***) override { return HCCL_SUCCESS; }
    ChannelStatus GetStatus() override { return ChannelStatus::INIT; }
    const HcommChannelDesc& GetChannelDesc() const override
    {
        static HcommChannelDesc defaultDesc{};
        return defaultDesc;
    }
    HcclResult Clean() override { return HCCL_SUCCESS; }
    HcclResult Resume() override { return HCCL_SUCCESS; }
    HcclResult NotifyRecord(uint32_t) override { return HCCL_E_NOT_SUPPORT; }
    HcclResult NotifyWait(uint32_t, uint32_t) override { return HCCL_E_NOT_SUPPORT; }
    HcclResult WriteWithNotify(void*, const void*, uint64_t, uint32_t) override { return HCCL_E_NOT_SUPPORT; }
    HcclResult Write(void*, const void*, uint64_t) override { return HCCL_E_NOT_SUPPORT; }
    HcclResult Read(void*, const void*, uint64_t) override { return HCCL_E_NOT_SUPPORT; }
    HcclResult ChannelFence() override { return HCCL_E_NOT_SUPPORT; }
};
} // namespace

TEST_F(UtChannelRoceFactory, Channel_Base_GetChannelKind_Returns_INVALID)
{
    ChannelDefaultsTestDouble ch;
    EXPECT_EQ(ch.GetChannelKind(), HcommChannelKind::INVALID);
}

TEST_F(UtChannelRoceFactory, Channel_Base_Serialize_Returns_NOT_SUPPORT)
{
    ChannelDefaultsTestDouble ch;
    std::shared_ptr<hccl::DeviceMem> out;
    EXPECT_EQ(ch.Serialize(out), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(out.get(), nullptr);
}

TEST_F(UtChannelRoceFactory, TransportStatusToChannelStatus_AllStates_Returns_Mapped)
{
    EndpointDesc localEp{};
    localEp.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.20.30.1", &localEp.commAddr.addr), 1);

    HcommChannelDesc channelDesc{};
    channelDesc.remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.20.30.2", &channelDesc.remoteEndpoint.commAddr.addr), 1);
    channelDesc.channelName = "ut-socket-tag";

    EXPECT_EQ(
        Channel::TransportStatusToChannelStatus(Hccl::TransportStatus::INIT, localEp, channelDesc),
        ChannelStatus::INIT);
    EXPECT_EQ(
        Channel::TransportStatusToChannelStatus(Hccl::TransportStatus::SOCKET_OK, localEp, channelDesc),
        ChannelStatus::SOCKET_OK);
    EXPECT_EQ(
        Channel::TransportStatusToChannelStatus(Hccl::TransportStatus::SOCKET_TIMEOUT, localEp, channelDesc),
        ChannelStatus::SOCKET_TIMEOUT);
    EXPECT_EQ(
        Channel::TransportStatusToChannelStatus(Hccl::TransportStatus::READY, localEp, channelDesc),
        ChannelStatus::READY);
}

TEST_F(UtChannelRoceFactory, TransportStatusToChannelStatus_InvalidStatus_Returns_FAILED)
{
    EndpointDesc localEp{};
    HcommChannelDesc channelDesc{};
    EXPECT_EQ(
        Channel::TransportStatusToChannelStatus(Hccl::TransportStatus::INVALID, localEp, channelDesc),
        ChannelStatus::FAILED);
}
