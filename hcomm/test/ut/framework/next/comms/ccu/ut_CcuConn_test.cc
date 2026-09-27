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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#define protected public

#include "ccu_conn.h"

#include "hccp_async.h"

#include "port.h"
#include "socket.h"
#include "ccu_jetty_.h"
#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"
#include "rdma_handle_manager.h"
#include "internal_exception.h"
#include "hccl_types.h"
#include "adapter_rts.h"
#include "hcom_common.h"
#include "env_config/env_config_v2.h"

#undef private
#undef protected

using namespace Hccl;

class CcuConnTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuConnTest tests set up." << std::endl; }

    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        std::cout << "CcuConnTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuConnTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuConnTest TearDown" << std::endl;
    }
};

pair<unique_ptr<hcomm::CcuConnection>, vector<unique_ptr<hcomm::CcuJetty>>>
MockMakeCcuConnection(hcomm::TpProtocol tpProtocol)
{
    constexpr uint64_t fakeMemAddr = 0x12345678;

    const uint32_t fakeTaJettyId = 1025;
    const uint64_t fakeSqBufVa = fakeMemAddr;
    const uint32_t fakeSqBufSize = 1024;
    const uint32_t fakeSqDepth = 4;
    const IpAddress locAddrIp{"1.1.1.1"};
    CommAddr locAddr{};
    CommAddr rmtAddr{};
    locAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    rmtAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    locAddr.addr.s_addr = 0;
    rmtAddr.addr.s_addr = 0;

    hcomm::CcuChannelInfo channelInfo;
    channelInfo.channelId = 1;
    channelInfo.dieId = 1;

    vector<unique_ptr<hcomm::CcuJetty>> ccuJettys;
    vector<hcomm::CcuJetty*> ccuJettyPtrs;
    for (uint32_t i = 0; i < 2; i++) {
        hcomm::CcuJettyInfo jettyInfo;
        jettyInfo.jettyCtxId = 1 + i;
        jettyInfo.taJettyId = fakeTaJettyId + i;
        jettyInfo.sqDepth = fakeSqDepth;
        jettyInfo.wqeBBStartId = 16;
        jettyInfo.sqBufVa = fakeSqBufVa + i;
        jettyInfo.sqBufSize = fakeSqBufSize + i;
        channelInfo.jettyInfos.push_back(jettyInfo);
        auto ccuJetty = make_unique<hcomm::CcuJetty>(locAddrIp, jettyInfo);
        ccuJettyPtrs.emplace_back(ccuJetty.get());
        ccuJettys.emplace_back(std::move(ccuJetty));
    }

    constexpr uint32_t kJettyQos = ::EnvConfig::UB_QOS_DEFAULT;
    unique_ptr<hcomm::CcuConnection> connection;
    if (tpProtocol == hcomm::TpProtocol::CTP) {
        connection = make_unique<hcomm::CcuCtpConnection>(locAddr, rmtAddr, channelInfo, ccuJettyPtrs, kJettyQos);
    } else {
        connection = make_unique<hcomm::CcuRtpConnection>(locAddr, rmtAddr, channelInfo, ccuJettyPtrs, kJettyQos);
    }

    return {std::move(connection), std::move(ccuJettys)};
}

#define SETUP_CCU_JETTY_MOCKS()                                                                 \
    do {                                                                                        \
        MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(0));                             \
        MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<s32>(0)));    \
        void* rdmaHandle = reinterpret_cast<void*>(0x300);                                      \
        MOCKER_CPP(&Hccl::RdmaHandleManager::GetByIp).stubs().will(returnValue(rdmaHandle));    \
        MOCKER_CPP(&Hccl::RdmaHandleManager::IsHandleValid).stubs().will(returnValue(true));    \
        MOCKER_CPP(&Hccl::RdmaHandleManager::GetJfcHandle)                                      \
            .stubs()                                                                            \
            .will(returnValue(static_cast<Hccl::JfcHandle>(0x400ULL)));                         \
        MOCKER_CPP(&Hccl::RdmaHandleManager::GetTokenIdInfo)                                    \
            .stubs()                                                                            \
            .will(returnValue(std::make_pair(static_cast<Hccl::TokenIdHandle>(0x500ULL), 0U))); \
    } while (0)

TEST_F(CcuConnTest, Ut_GetStatus_When_CreateAndImportJettySuccess_Expect_Return_Connected)
{
    auto resPair = MockMakeCcuConnection(hcomm::TpProtocol::RTP);
    auto connection = resPair.first.get();

    std::string testDfxMsg = "";
    HcclResult ret = connection->Describe(testDfxMsg);
    MOCKER(HrtRaGetTpAttrAsync).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    connection->Describe(testDfxMsg);
    GlobalMockObject::verify();

    MOCKER(HrtRaGetTpAttrAsync)
        .stubs()
        .will(returnValue(HcclResult::HCCL_E_NOT_SUPPORT))
        .then(returnValue(HcclResult::HCCL_E_INTERNAL));
    ret = connection->Describe(testDfxMsg);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
    ret = connection->Describe(testDfxMsg);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuConnTest, Ut_GetTpAttr_CTP_ReturnsSuccess)
{
    auto resPair = MockMakeCcuConnection(hcomm::TpProtocol::CTP);
    auto connection = resPair.first.get();
    connection->tpProtocol_ = hcomm::TpProtocol::CTP;

    HcclResult ret = connection->GetTpAttr();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(CcuConnTest, Ut_GetTpAttr_RTP_TpMgrReturnsAgain_ReturnsAgain)
{
    auto resPair = MockMakeCcuConnection(hcomm::TpProtocol::RTP);
    auto connection = resPair.first.get();
    connection->tpProtocol_ = hcomm::TpProtocol::RTP;
    connection->tpInfo_.tpHandle = 0x1234;
    connection->ctxHandle_ = (hcomm::CtxHandle)0x5678;
    connection->devPhyId_ = 0;

    MOCKER_CPP(&hcomm::TpMgr::GetTpAttr).stubs().will(returnValue(HcclResult::HCCL_E_AGAIN));

    HcclResult ret = connection->GetTpAttr();
    EXPECT_EQ(ret, HcclResult::HCCL_E_AGAIN);

    GlobalMockObject::verify();
}

TEST_F(CcuConnTest, Ut_GetTpAttr_RTP_TpMgrReturnsSuccess_ReturnsSuccess)
{
    auto resPair = MockMakeCcuConnection(hcomm::TpProtocol::RTP);
    auto connection = resPair.first.get();
    connection->tpProtocol_ = hcomm::TpProtocol::RTP;
    connection->tpInfo_.tpHandle = 0x1234;
    connection->ctxHandle_ = (hcomm::CtxHandle)0x5678;
    connection->devPhyId_ = 0;

    MOCKER_CPP(&hcomm::TpMgr::GetTpAttr).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    HcclResult ret = connection->GetTpAttr();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(CcuConnTest, Ut_GetTaTimeOut_RTP_UsesCalcTaTimeout)
{
    auto resPair = MockMakeCcuConnection(hcomm::TpProtocol::RTP);
    auto connection = resPair.first.get();
    connection->tpProtocol_ = hcomm::TpProtocol::RTP;
    connection->tpAttrInfo_.tpAttr.at = 2;
    connection->tpAttrInfo_.tpAttr.retryTimesInit = 1;

    MOCKER_CPP(static_cast<uint8_t (*)(hcomm::TpProtocol, uint8_t, uint32_t)>(&hcomm::TpMgr::CalcTaTimeout))
        .stubs()
        .will(returnValue(static_cast<uint8_t>(24)));

    HcclResult ret = connection->GetTaTimeOut();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(connection->errTimeout_, 24);

    GlobalMockObject::verify();
}

TEST_F(CcuConnTest, Ut_UpdateInitStatus_TP_ATTR_GETTING_State_Transitions)
{
    auto resPair = MockMakeCcuConnection(hcomm::TpProtocol::RTP);
    auto connection = resPair.first.get();
    connection->tpProtocol_ = hcomm::TpProtocol::RTP;
    connection->innerStatus_ = hcomm::CcuConnection::InnerStatus::TP_ATTR_GETTING;
    connection->tpInfo_.tpHandle = 0x1234;
    connection->ctxHandle_ = (hcomm::CtxHandle)0x5678;
    connection->devPhyId_ = 0;
    connection->tpAttrInfo_.tpAttr.at = 2;

    MOCKER_CPP(&hcomm::TpMgr::GetTpAttr).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(static_cast<uint8_t (*)(hcomm::TpProtocol, uint8_t, uint32_t)>(&hcomm::TpMgr::CalcTaTimeout))
        .stubs()
        .will(returnValue(static_cast<uint8_t>(16)));
    MOCKER_CPP(&hcomm::CcuJetty::CreateJetty).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    HcclResult ret = connection->UpdateInitStatus();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(connection->innerStatus_, hcomm::CcuConnection::InnerStatus::JETTY_CREATING);

    ret = connection->UpdateInitStatus();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(connection->innerStatus_, hcomm::CcuConnection::InnerStatus::EXCHANGEABLE);

    GlobalMockObject::verify();
}

HcclResult StubTpMgrGetTpInfoWithMappedPriority(hcomm::TpMgr*, const hcomm::GetTpInfoParam&, hcomm::TpInfo& tpInfo)
{
    tpInfo.tpHandle = 0x4321ULL;
    tpInfo.hasMappedJettyPriority = true;
    tpInfo.mappedJettyPriority = 4U;
    return HcclResult::HCCL_SUCCESS;
}

TEST_F(CcuConnTest, Ut_CcuJetty_SetMappedJettyPriority_When_NotCreated_SetsQos)
{
    constexpr uint64_t fakeMemAddr = 0x12345678;
    const uint32_t fakeTaJettyId = 1025;
    const IpAddress locAddrIp{"2.2.2.2"};
    hcomm::CcuJettyInfo jettyInfo{};
    jettyInfo.taJettyId = fakeTaJettyId;
    jettyInfo.sqBufVa = fakeMemAddr;
    jettyInfo.sqBufSize = 1024;
    jettyInfo.sqDepth = 4;

    SETUP_CCU_JETTY_MOCKS();

    hcomm::CcuJetty jetty(locAddrIp, jettyInfo);
    ASSERT_EQ(jetty.Init(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(jetty.SetMappedJettyPriority(4U), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(jetty.GetCreateJettyParam().qos, static_cast<uint8_t>(4U));

    GlobalMockObject::verify();
}

TEST_F(CcuConnTest, Ut_CcuJetty_SetMappedJettyPriority_When_Conflict_Expect_Internal)
{
    const IpAddress locAddrIp{"3.3.3.3"};
    hcomm::CcuJettyInfo jettyInfo{};
    jettyInfo.taJettyId = 2048;
    jettyInfo.sqBufVa = 0x1000;
    jettyInfo.sqBufSize = 512;
    jettyInfo.sqDepth = 4;

    SETUP_CCU_JETTY_MOCKS();

    hcomm::CcuJetty jetty(locAddrIp, jettyInfo);
    ASSERT_EQ(jetty.Init(), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(jetty.SetMappedJettyPriority(2U), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(jetty.SetMappedJettyPriority(5U), HcclResult::HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}

TEST_F(CcuConnTest, Ut_UpdateInitStatus_WithMappedPriority_SetsJettyQos)
{
    auto resPair = MockMakeCcuConnection(hcomm::TpProtocol::RTP);
    auto connection = resPair.first.get();
    auto& jetty = *resPair.second.front();
    connection->tpProtocol_ = hcomm::TpProtocol::RTP;
    connection->innerStatus_ = hcomm::CcuConnection::InnerStatus::INIT;

    MOCKER_CPP(&hcomm::TpMgr::GetTpInfo).stubs().will(invoke(StubTpMgrGetTpInfoWithMappedPriority));
    MOCKER_CPP(&hcomm::TpMgr::GetTpAttr).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(static_cast<uint8_t (*)(hcomm::TpProtocol, uint8_t, uint32_t)>(&hcomm::TpMgr::CalcTaTimeout))
        .stubs()
        .will(returnValue(static_cast<uint8_t>(16)));
    MOCKER_CPP(&hcomm::CcuJetty::CreateJetty).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    HcclResult ret = connection->UpdateInitStatus();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(connection->innerStatus_, hcomm::CcuConnection::InnerStatus::TP_ATTR_GETTING);
    EXPECT_EQ(jetty.GetCreateJettyParam().qos, static_cast<uint8_t>(4U));

    GlobalMockObject::verify();
}

hcomm::GetTpInfoParam gCapturedConnTpParam{};

HcclResult StubTpMgrGetTpInfoCaptureParam(hcomm::TpMgr*, const hcomm::GetTpInfoParam& param, hcomm::TpInfo& tpInfo)
{
    gCapturedConnTpParam = param;
    tpInfo.tpHandle = 0x4321ULL;
    tpInfo.hasMappedJettyPriority = true;
    tpInfo.mappedJettyPriority = 3U;
    return HcclResult::HCCL_SUCCESS;
}

TEST_F(CcuConnTest, Ut_MakeGetTpInfoParam_When_QosAboveSeven_Expect_ClampsToDefault)
{
    gCapturedConnTpParam = hcomm::GetTpInfoParam{};
    auto resPair = MockMakeCcuConnection(hcomm::TpProtocol::RTP);
    auto connection = resPair.first.get();
    connection->qos_ = 9U;
    connection->innerStatus_ = hcomm::CcuConnection::InnerStatus::INIT;

    MOCKER_CPP(&hcomm::TpMgr::GetTpInfo).stubs().will(invoke(StubTpMgrGetTpInfoCaptureParam));

    HcclResult ret = connection->UpdateInitStatus();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(gCapturedConnTpParam.qos, static_cast<uint32_t>(::EnvConfig::UB_QOS_DEFAULT));

    GlobalMockObject::verify();
}
