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
#include <cstring>
#include <memory>

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "ccu_device_pub.h"
#include "ccu_comp.h"
#include "hccl_common.h"
#include "buffer.h"
#include "local_ub_rma_buffer.h"
#include "orion_adapter_hccp.h"
#include "rdma_handle_manager.h"
#include "env_config/env_config_v2.h"

#include "unified_platform/ccu/ccu_device/ccu_component/ccu_component.h"

#include "ccu_pfe_cfg_mgr.h"

#define private public
#define protected public
#include "ccu_comp.h"
#undef protected
#undef private

using namespace hcomm;

class CcuCompPubTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    static void TearDownTestCase()
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    void SetUp() override
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    void TearDown() override
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }

    // Helpers to stub CcuComponent member functions
    void StubSetTaskKill(const HcclResult ret)
    {
        MOCKER_CPP(&CcuComponent::SetTaskKill).stubs().will(returnValue(ret));
        MOCKER_CPP(&Hccl::CcuComponent::SetTaskKill).stubs().will(returnValue(ret));
    }
    void StubSetTaskKillDone(const HcclResult ret)
    {
        MOCKER_CPP(&CcuComponent::SetTaskKillDone).stubs().will(returnValue(ret));
        MOCKER_CPP(&Hccl::CcuComponent::SetTaskKillDone).stubs().will(returnValue(ret));
    }
    void StubCleanTaskKillState(const HcclResult ret)
    {
        MOCKER_CPP(&CcuComponent::CleanTaskKillState).stubs().will(returnValue(ret));
        MOCKER_CPP(&Hccl::CcuComponent::CleanTaskKillState).stubs().will(returnValue(ret));
    }
    void StubCleanDieCkes(const HcclResult ret)
    {
        MOCKER_CPP(&CcuComponent::CleanDieCkes).stubs().with(mockcpp::any()).will(returnValue(ret));
        MOCKER_CPP(&Hccl::CcuComponent::CleanDieCkes).stubs().with(mockcpp::any()).will(returnValue(ret));
    }
};

// CcuSetTaskKill: invalid deviceLogicId -> HCCL_E_PARA
TEST_F(CcuCompPubTest, Ut_CcuSetTaskKillWhenDeviceIdInvalidExpectHcclEPara)
{
    const int32_t badId = static_cast<int32_t>(MAX_MODULE_DEVICE_NUM); // out of range
    auto ret = CcuSetTaskKill(-1);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
    ret = CcuSetTaskKill(badId);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
}

// CcuSetTaskKill: valid deviceLogicId forwards to CcuComponent::SetTaskKill
TEST_F(CcuCompPubTest, Ut_CcuSetTaskKillWhenUnderlyingSucceedsExpectSuccess)
{
    StubSetTaskKill(HcclResult::HCCL_SUCCESS);
    auto ret = CcuSetTaskKill(0);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuCompPubTest, Ut_CcuSetTaskKillWhenUnderlyingFailsExpectFailure)
{
    StubSetTaskKill(HcclResult::HCCL_E_INTERNAL);
    auto ret = CcuSetTaskKill(0);
    EXPECT_EQ(ret, HcclResult::HCCL_E_INTERNAL);
}

// CcuSetTaskKillDone: invalid deviceLogicId -> HCCL_E_PARA
TEST_F(CcuCompPubTest, Ut_CcuSetTaskKillDoneWhenDeviceIdInvalidExpectHcclEPara)
{
    auto ret = CcuSetTaskKillDone(-2);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
    ret = CcuSetTaskKillDone(static_cast<int32_t>(MAX_MODULE_DEVICE_NUM));
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
}

// CcuSetTaskKillDone: forwards to CcuComponent::SetTaskKillDone
TEST_F(CcuCompPubTest, Ut_CcuSetTaskKillDoneWhenUnderlyingSucceedsExpectSuccess)
{
    StubSetTaskKillDone(HcclResult::HCCL_SUCCESS);
    auto ret = CcuSetTaskKillDone(0);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuCompPubTest, Ut_CcuSetTaskKillDoneWhenUnderlyingFailsExpectFailure)
{
    StubSetTaskKillDone(HcclResult::HCCL_E_INTERNAL);
    auto ret = CcuSetTaskKillDone(0);
    EXPECT_EQ(ret, HcclResult::HCCL_E_INTERNAL);
}

// CcuCleanTaskKillState: invalid deviceLogicId -> HCCL_E_PARA
TEST_F(CcuCompPubTest, Ut_CcuCleanTaskKillStateWhenDeviceIdInvalidExpectHcclEPara)
{
    auto ret = CcuCleanTaskKillState(-5);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
    ret = CcuCleanTaskKillState(static_cast<int32_t>(MAX_MODULE_DEVICE_NUM));
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
}

// CcuCleanTaskKillState: forwards to CcuComponent::CleanTaskKillState
TEST_F(CcuCompPubTest, Ut_CcuCleanTaskKillStateWhenUnderlyingSucceedsExpectSuccess)
{
    StubCleanTaskKillState(HcclResult::HCCL_SUCCESS);
    auto ret = CcuCleanTaskKillState(0);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuCompPubTest, Ut_CcuCleanTaskKillStateWhenUnderlyingFailsExpectFailure)
{
    StubCleanTaskKillState(HcclResult::HCCL_E_INTERNAL);
    auto ret = CcuCleanTaskKillState(0);
    EXPECT_EQ(ret, HcclResult::HCCL_E_INTERNAL);
}

// CcuCleanDieCkes: invalid deviceLogicId -> HCCL_E_PARA
TEST_F(CcuCompPubTest, Ut_CcuCleanDieCkesWhenDeviceIdInvalidExpectHcclEPara)
{
    auto ret = CcuCleanDieCkes(-1, 0);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
    ret = CcuCleanDieCkes(static_cast<int32_t>(MAX_MODULE_DEVICE_NUM), 1);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
}

// CcuCleanDieCkes: forwards to CcuComponent::CleanDieCkes
TEST_F(CcuCompPubTest, Ut_CcuCleanDieCkesWhenUnderlyingSucceedsExpectSuccess)
{
    StubCleanDieCkes(HcclResult::HCCL_SUCCESS);
    auto ret = CcuCleanDieCkes(0, 1);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuCompPubTest, Ut_CcuCleanDieCkesWhenUnderlyingFailsExpectFailure)
{
    StubCleanDieCkes(HcclResult::HCCL_E_INTERNAL);
    auto ret = CcuCleanDieCkes(0, 1);
    EXPECT_EQ(ret, HcclResult::HCCL_E_INTERNAL);
}

namespace {

uint32_t gCapturedLoopJettyQos = 0U;
GetTpInfoParam gCapturedLoopTpParam{};

HcclResult
StubHccpUbCreateJettyCaptureQos(const CtxHandle, const HrtRaUbCreateJettyParam& req, HrtRaUbJettyCreatedOutParam&)
{
    gCapturedLoopJettyQos = req.qos;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult StubHccpUbTpImportJettyOk(
    const CtxHandle, u8*, const u32, const u32, const JettyImportCfg&, HrtRaUbJettyImportedOutParam&)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult StubTpMgrGetLoopTpInfo(TpMgr*, const GetTpInfoParam& param, TpInfo& tpInfo)
{
    gCapturedLoopTpParam = param;
    tpInfo.tpHandle = 0xABCDULL;
    tpInfo.hasMappedJettyPriority = true;
    tpInfo.mappedJettyPriority = 2U;
    return HcclResult::HCCL_SUCCESS;
}

CommAddr MakeLoopCommAddr(const char* dotted)
{
    CommAddr commAddr{};
    commAddr.type = COMM_ADDR_TYPE_IP_V4;
    EXPECT_EQ(inet_pton(AF_INET, dotted, &commAddr.addr), 1);
    return commAddr;
}

void PrepareLoopJettyTestFixture(
    CcuComponent& comp, const uint8_t dieId, const CommAddr& commAddr, const uint32_t mappedJettyPriority)
{
    comp.devPhyId_ = 0U;
    comp.userDevId_ = 0;

    const std::pair<Hccl::TokenIdHandle, uint32_t> fakeTokenInfo
        = std::make_pair(reinterpret_cast<Hccl::TokenIdHandle>(0x88888888ULL), 1U);
    MOCKER_CPP(&Hccl::RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(fakeTokenInfo));
    MOCKER(Hccl::HrtRaUbLocalMemReg).stubs().will(returnValue(Hccl::HrtRaUbLocalMemRegOutParam()));

    auto buffer = std::make_shared<Hccl::Buffer>(0x1000ULL, 4096ULL);
    comp.ccuRmaBufferMap_[dieId]
        = std::make_unique<Hccl::LocalUbRmaBuffer>(buffer, reinterpret_cast<RdmaHandle>(0x200));

    TpInfo tpInfo{};
    tpInfo.tpHandle = 0xABCDULL;
    tpInfo.hasMappedJettyPriority = true;
    tpInfo.mappedJettyPriority = mappedJettyPriority;
    comp.tpInfoMap_[dieId] = tpInfo;

    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 1U;
    comp.tpAttrInfoMap_[dieId] = tpAttrInfo;

    MOCKER_CPP(&Hccl::RdmaHandleManager::GetByIp).stubs().will(returnValue(reinterpret_cast<RdmaHandle>(0x300)));
    MOCKER_CPP(&Hccl::RdmaHandleManager::GetJfcHandle)
        .stubs()
        .will(returnValue(static_cast<Hccl::JfcHandle>(0x400ULL)));
    MOCKER(HccpUbCreateJetty).stubs().will(invoke(StubHccpUbCreateJettyCaptureQos));
    MOCKER(HccpUbTpImportJetty).stubs().will(invoke(StubHccpUbTpImportJettyOk));
}

} // namespace

TEST_F(CcuCompPubTest, Ut_CreateAndImportLoopJettys_When_MappedPrioritySet_Expect_QosApplied)
{
    gCapturedLoopJettyQos = 0U;
    CcuComponent comp{};
    const uint8_t dieId = 0U;
    const CommAddr commAddr = MakeLoopCommAddr("127.0.0.1");
    PrepareLoopJettyTestFixture(comp, dieId, commAddr, 5U);

    JettyInfo jettyInfo{};
    jettyInfo.taJettyId = 100U;
    jettyInfo.sqBufVa = 0x2000ULL;
    jettyInfo.sqBufSize = 1024U;
    jettyInfo.sqDepth = 4U;

    EXPECT_EQ(comp.CreateAndImportLoopJettys(dieId, commAddr, {jettyInfo}), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(gCapturedLoopJettyQos, 5U);
}

TEST_F(CcuCompPubTest, Ut_CreateAndImportLoopJettys_When_NoMappedPriority_Expect_DefaultQos)
{
    gCapturedLoopJettyQos = 0U;
    CcuComponent comp{};
    const uint8_t dieId = 1U;
    const CommAddr commAddr = MakeLoopCommAddr("127.0.0.2");
    PrepareLoopJettyTestFixture(comp, dieId, commAddr, 0U);
    comp.tpInfoMap_[dieId].hasMappedJettyPriority = false;

    JettyInfo jettyInfo{};
    jettyInfo.taJettyId = 101U;
    jettyInfo.sqBufVa = 0x3000ULL;
    jettyInfo.sqBufSize = 512U;
    jettyInfo.sqDepth = 4U;

    EXPECT_EQ(comp.CreateAndImportLoopJettys(dieId, commAddr, {jettyInfo}), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(gCapturedLoopJettyQos, static_cast<uint32_t>(EnvConfig::UB_QOS_DEFAULT));
}

TEST_F(CcuCompPubTest, Ut_RequestNewLoopTpInfo_When_Called_Expect_LoopGetTpInfoParamFlags)
{
    gCapturedLoopTpParam = GetTpInfoParam{};
    CcuComponent comp{};
    comp.devPhyId_ = 0U;
    comp.userDevId_ = 0;

    MOCKER_CPP(&TpMgr::GetTpInfo).stubs().will(invoke(StubTpMgrGetLoopTpInfo));

    const CommAddr commAddr = MakeLoopCommAddr("10.0.0.8");
    TpInfo tpInfo{};
    EXPECT_EQ(comp.GetLoopTpInfo(1, commAddr, tpInfo), HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(gCapturedLoopTpParam.loopFirstTpLowestSl);
    EXPECT_TRUE(gCapturedLoopTpParam.ccuLoopbackGetTpInfo);
    EXPECT_EQ(gCapturedLoopTpParam.tpProtocol, TpProtocol::RTP);
    EXPECT_EQ(tpInfo.mappedJettyPriority, 2U);
}
