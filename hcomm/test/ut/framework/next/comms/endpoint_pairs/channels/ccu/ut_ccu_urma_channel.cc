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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#include "env_config/env_config_v2.h"

#define private public
#include "ccu_urma_channel.h"
#include "local_ub_rma_buffer.h"
#include "orion_adpt_utils.h"
#undef private

using namespace hcomm;

namespace hcomm {
HcclResult BuildBufferInfos(
    HcommMemHandle* memHandles, uint32_t memHandleNum, std::vector<CcuTransport::CclBufferInfo>& bufferInfos);
}

class CcuUrmaChannelTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

static UrmaEndpoint& TestUrmaEndpoint()
{
    static UrmaEndpoint endpoint{EndpointDesc{}};
    return endpoint;
}

TEST_F(CcuUrmaChannelTest, Ut_Clean_When_ImplIsNull_Expect_HCCL_E_PTR)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    CcuUrmaChannel ch(ep, desc);

    auto ret = ch.Clean();
    EXPECT_EQ(ret, HcclResult::HCCL_E_PTR);
}

// Minimal test-only connection deriving from CcuConnection to avoid heavy Init()
class TestCcuConnection : public CcuConnection {
public:
    TestCcuConnection(
        const CommAddr& locAddr, const CommAddr& rmtAddr, const CcuChannelInfo& channelInfo,
        const std::vector<CcuJetty*>& ccuJettys, uint32_t qos = Hccl::UB_QOS_DEFAULT)
        : CcuConnection(locAddr, rmtAddr, channelInfo, ccuJettys, qos)
    {}
    // Do not call Init(); use default base behavior for Clean()
};

TEST_F(CcuUrmaChannelTest, Ut_Clean_When_ImplIsPresent_Expect_HCCL_SUCCESS)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    CcuUrmaChannel ch(ep, desc);

    // Prepare minimal safe objects to construct a CcuTransport without heavy Init
    CommAddr locAddr{};
    CommAddr rmtAddr{};
    CcuChannelInfo channelInfo{};
    std::vector<CcuJetty*> jettys{};

    // Create a test connection (does not call Init)
    std::unique_ptr<CcuConnection> conn(
        new TestCcuConnection(locAddr, rmtAddr, channelInfo, jettys, Hccl::UB_QOS_DEFAULT));

    // Fake socket pointer (not dereferenced by Clean())
    Hccl::Socket* fakeSocket = reinterpret_cast<Hccl::Socket*>(0x1);

    // Prepare a simple buffer info
    CcuTransport::CclBufferInfo bufInfo(0x1000, 0x100, 1, 1);

    // Construct transport instance (constructor does not call Init)
    std::unique_ptr<CcuTransport> transport(new CcuTransport(fakeSocket, std::move(conn), bufInfo));

    // Inject into channel (we used #define private public when included)
    ch.impl_ = std::move(transport);

    auto ret = ch.Clean();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuUrmaChannelTest, Ut_Resume_When_Called_Expect_HCCL_SUCCESS)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    CcuUrmaChannel ch(ep, desc);

    auto ret = ch.Resume();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuUrmaChannelTest, Ut_BuildBufferInfos_When_LocalUbHandle_Expect_BufferInfoFieldsFromRmaBuffer)
{
    auto rawBuffer = std::make_shared<Hccl::Buffer>(0x34560, 0x200, HCCL_MEM_TYPE_HOST, "ccu_user");
    auto localRmaBuffer = std::make_shared<Hccl::LocalUbRmaBuffer>(rawBuffer);
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localRmaBuffer.get())};

    std::vector<CcuTransport::CclBufferInfo> bufferInfos;
    ASSERT_EQ(BuildBufferInfos(memHandles, 1, bufferInfos), HCCL_SUCCESS);
    ASSERT_EQ(bufferInfos.size(), 1U);
    EXPECT_EQ(bufferInfos[0].addr, localRmaBuffer->GetAddr());
    EXPECT_EQ(bufferInfos[0].size, static_cast<uint32_t>(localRmaBuffer->GetSize()));
    EXPECT_EQ(bufferInfos[0].tokenId, localRmaBuffer->GetTokenId());
    EXPECT_EQ(bufferInfos[0].tokenValue, localRmaBuffer->GetTokenValue());
    EXPECT_EQ(bufferInfos[0].type, COMM_MEM_TYPE_HOST);
    EXPECT_EQ(std::string(bufferInfos[0].memInfo.data()), rawBuffer->GetMemInfo());
}

TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_DfxDescribe_Expect_ReadyOrFailed)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(1);
    CcuUrmaChannel ch(ep, desc);

    ch.ccuEndpoint_ = &TestUrmaEndpoint();
    ch.socket_ = reinterpret_cast<Hccl::Socket*>(1);
    ch.memHandles_.push_back(reinterpret_cast<HcommMemHandle>(0x1));

    // 1. 构造依赖
    CommAddr locAddr{}, rmtAddr{};
    CcuChannelInfo channelInfo{};
    std::vector<CcuJetty*> jettys{};

    auto conn = std::make_unique<CcuConnection>(locAddr, rmtAddr, channelInfo, jettys, Hccl::UB_QOS_DEFAULT);
    auto fakeSocket = reinterpret_cast<Hccl::Socket*>(1);
    CcuTransport::CclBufferInfo bufInfo(0x1000, 0x100, 1, 1);

    // 2. 创建 transport
    auto transport = std::make_unique<CcuTransport>(fakeSocket, std::move(conn), bufInfo);
    ch.impl_ = std::move(transport);
    ch.impl_->SetLocResStatus(CcuTransport::CcuResStatus::RES_OK);

    // 3. 从 ch.impl_ 操作，不要再用旧的 transport 指针！
    ch.impl_->transStatus_ = CcuTransport::TransStatus::SOCKET_TIMEOUT;
    auto ret = ch.GetStatus();
    EXPECT_NE(ret, ChannelStatus::INIT);

    ch.impl_->transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;
    ret = ch.GetStatus();
    EXPECT_NE(ret, ChannelStatus::INIT);

    // 4. Mock 1：成功
    ch.impl_->transStatus_ = CcuTransport::TransStatus::READY;
    MOCKER_CPP(&CcuConnection::Describe, HcclResult(CcuConnection::*)(std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    ret = ch.GetStatus();
    EXPECT_EQ(ret, ChannelStatus::READY);
    GlobalMockObject::verify();
    GlobalMockObject::reset(); // <-- 必须重置 Mock！

    // 5. Mock 2：失败
    ch.impl_->transStatus_ = CcuTransport::TransStatus::READY;
    ch.isFirstPrintChannelInfo_ = true; // 需要重置为true才能触发Describe的调用
    MOCKER_CPP(&CcuConnection::Describe, HcclResult(CcuConnection::*)(std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_E_PARA));
    ret = ch.GetStatus();
    EXPECT_EQ(ret, ChannelStatus::FAILED);
    GlobalMockObject::verify();
    GlobalMockObject::reset();
}

// ==================== CcuUrmaChannel::GetStatus 资源不足终态映射 UT ====================

struct ChannelWithTransport {
    std::unique_ptr<CcuUrmaChannel> ch;
    CcuTransport* impl;
};

ChannelWithTransport MakeChannelWithTransport()
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(1);
    auto ch = std::make_unique<CcuUrmaChannel>(ep, desc);
    ch->ccuEndpoint_ = &TestUrmaEndpoint();
    ch->socket_ = reinterpret_cast<Hccl::Socket*>(1);
    ch->memHandles_.push_back(reinterpret_cast<HcommMemHandle>(0x1));
    CommAddr locAddr{}, rmtAddr{};
    CcuChannelInfo channelInfo{};
    std::vector<CcuJetty*> jettys{};
    auto conn = std::make_unique<CcuConnection>(locAddr, rmtAddr, channelInfo, jettys, Hccl::UB_QOS_DEFAULT);
    auto fakeSocket = reinterpret_cast<Hccl::Socket*>(1);
    CcuTransport::CclBufferInfo bufInfo(0x1000, 0x100, 1, 1);
    auto transport = std::make_unique<CcuTransport>(fakeSocket, std::move(conn), bufInfo);
    // STC V2.0: locResStatus_ 改用公共 setter, 不直塞私有成员
    transport->SetLocResStatus(CcuTransport::CcuResStatus::RES_OK);
    CcuTransport* impl = transport.get();
    ch->impl_ = std::move(transport);
    ch->isFirstPrintChannelInfo_ = false;
    return {std::move(ch), impl};
}

TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_LocUnavail_Expect_ResLocUnavailable)
{
    auto ctx = MakeChannelWithTransport();
    // 白盒保留: transStatus_/rmtResStatus_ 无公共 setter, 只能直塞私有成员
    ctx.impl->transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;
    // STC V2.0: locResStatus_ 改用公共 setter
    ctx.impl->SetLocResStatus(CcuTransport::CcuResStatus::RES_UNAVAIL);
    ctx.impl->rmtResStatus_ = CcuTransport::CcuResStatus::RES_OK;
    EXPECT_EQ(ctx.ch->GetStatus(), ChannelStatus::RES_LOC_UNAVAIL);
}

TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_RmtUnavail_Expect_ResRmtUnavailable)
{
    auto ctx = MakeChannelWithTransport();
    // 白盒保留: transStatus_/rmtResStatus_ 无公共 setter, 只能直塞私有成员
    ctx.impl->transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;
    // STC V2.0: locResStatus_ 改用公共 setter
    ctx.impl->SetLocResStatus(CcuTransport::CcuResStatus::RES_OK);
    ctx.impl->rmtResStatus_ = CcuTransport::CcuResStatus::RES_UNAVAIL;
    EXPECT_EQ(ctx.ch->GetStatus(), ChannelStatus::RES_RMT_UNAVAIL);
}

TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_BothUnavail_Expect_LocPriority)
{
    auto ctx = MakeChannelWithTransport();
    // 白盒保留: transStatus_/rmtResStatus_ 无公共 setter, 只能直塞私有成员
    ctx.impl->transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;
    // STC V2.0: locResStatus_ 改用公共 setter
    ctx.impl->SetLocResStatus(CcuTransport::CcuResStatus::RES_UNAVAIL);
    ctx.impl->rmtResStatus_ = CcuTransport::CcuResStatus::RES_UNAVAIL;
    EXPECT_EQ(ctx.ch->GetStatus(), ChannelStatus::RES_LOC_UNAVAIL);
}

TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_NoUnavail_Expect_Failed)
{
    auto ctx = MakeChannelWithTransport();
    // 白盒保留: transStatus_/rmtResStatus_ 无公共 setter, 只能直塞私有成员
    ctx.impl->transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;
    // STC V2.0: locResStatus_ 改用公共 setter
    ctx.impl->SetLocResStatus(CcuTransport::CcuResStatus::RES_OK);
    ctx.impl->rmtResStatus_ = CcuTransport::CcuResStatus::RES_OK;
    EXPECT_EQ(ctx.ch->GetStatus(), ChannelStatus::FAILED);
}

// TC-CH-016: 本端硬失败(RES_FAILED)终态映射为 FAILED (非资源不足, 不触发回退)
TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_LocFailed_Expect_Failed)
{
    auto ctx = MakeChannelWithTransport();
    ctx.impl->transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;
    ctx.impl->SetLocResStatus(CcuTransport::CcuResStatus::RES_FAILED);
    ctx.impl->rmtResStatus_ = CcuTransport::CcuResStatus::RES_OK;
    EXPECT_EQ(ctx.ch->GetStatus(), ChannelStatus::FAILED);
}

// TC-CH-017: 对端硬失败(RES_FAILED)终态映射为 FAILED
TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_RmtFailed_Expect_Failed)
{
    auto ctx = MakeChannelWithTransport();
    ctx.impl->transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;
    ctx.impl->SetLocResStatus(CcuTransport::CcuResStatus::RES_OK);
    ctx.impl->rmtResStatus_ = CcuTransport::CcuResStatus::RES_FAILED;
    EXPECT_EQ(ctx.ch->GetStatus(), ChannelStatus::FAILED);
}

TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_Ready_Expect_Ready)
{
    auto ctx = MakeChannelWithTransport();
    ctx.impl->transStatus_ = CcuTransport::TransStatus::READY;
    EXPECT_EQ(ctx.ch->GetStatus(), ChannelStatus::READY);
}

TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_SocketTimeout_Expect_SocketTimeout)
{
    auto ctx = MakeChannelWithTransport();
    ctx.impl->transStatus_ = CcuTransport::TransStatus::SOCKET_TIMEOUT;
    EXPECT_EQ(ctx.ch->GetStatus(), ChannelStatus::SOCKET_TIMEOUT);
}

TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_ConstructFails_Expect_MsgOnlyTransport)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    CcuUrmaChannel ch(ep, desc);

    ch.ccuEndpoint_ = &TestUrmaEndpoint();
    ch.socket_ = reinterpret_cast<Hccl::Socket*>(0x1);
    ch.memHandles_ = {reinterpret_cast<HcommMemHandle>(0x2)};
    ch.channelStatus_ = ChannelStatus::INIT;

    // 第一次 GetStatus: 触发 TryPrepareAndConstruct, 硬失败 -> msg-only transport(RES_FAILED)
    EXPECT_EQ(ch.GetStatus(), ChannelStatus::INIT);
    ASSERT_NE(ch.impl_, nullptr);
    EXPECT_EQ(ch.impl_->GetLocResStatus(), CcuTransport::CcuResStatus::RES_FAILED);
    EXPECT_EQ(ch.impl_->transStatus_, CcuTransport::TransStatus::INIT);

    // 第二次 GetStatus: locResStatus_ != RES_UNKNOWN, 不会重复构造 transport
    CcuTransport* implPtr = ch.impl_.get();
    EXPECT_EQ(ch.GetStatus(), ChannelStatus::INIT);
    EXPECT_EQ(ch.impl_.get(), implPtr);
}

// 未初始化路径: endpoint/socket/memHandles 缺失时直接失败, 不再进入懒建链
TEST_F(CcuUrmaChannelTest, Ut_GetStatus_When_Uninitialized_Expect_Failed)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    CcuUrmaChannel ch(ep, desc);

    EXPECT_EQ(ch.GetStatus(), ChannelStatus::FAILED);
    EXPECT_EQ(ch.impl_, nullptr);
}
