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
#include <mockcpp/mockcpp.hpp>

#include "ccu_jetty_ctx_mgr_v2.h"
#include "ccu_wqebb_mgr_v2.h"
#include "ccu_res_specs.h"
#include "ccu_pfe_cfg_mgr.h"
#include "hcomm_adapter_hccp.h"
#include "hccl_common.h"

#define private public
#define protected public

using namespace hcomm;

static void SetupCcuResSpecsForJetty(int32_t devLogicId, uint8_t dieId)
{
    auto& specs = CcuResSpecifications::GetInstance(devLogicId);
    specs.initFlag_ = true;
    specs.ccuVersion_ = CcuVersion::CCU_V2;
    specs.dieEnableFlags_[dieId] = true;
    specs.resSpecs_[dieId].jettyNum = 128;
    specs.resSpecs_[dieId].wqeBBNum = 4096;
    specs.resSpecs_[dieId].channelNum = 1024;
    specs.resSpecs_[dieId].resourceAddr = 0xE7FFBF800000;
    specs.resSpecs_[dieId].pfeNum = 20;
    specs.resSpecs_[dieId].channelJettyMap = CcuChannelJettyMap(8, 1);
}

static void InjectPfeStrategy(CcuJettyCtxMgrV2& mgr, uint32_t feId)
{
    PfeJettyStrategy strategy{};
    strategy.feId = feId;
    strategy.pfeId = feId;
    strategy.startTaJettyId = CCU_START_TA_JETTY_ID;
    strategy.size = 128;
    strategy.startLocalJettyCtxId = 0;
    mgr.pfeMgr_.pfeJettyMap_[feId] = strategy;
}

class CcuJettyCtxMgrV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
    void SetUp() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        SetupCcuResSpecsForJetty(0, 0);
    }
    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
};

TEST_F(CcuJettyCtxMgrV2Test, Ut_Init_When_Normal_Expect_ReturnSuccess)
{
    CcuJettyCtxMgrV2 mgr(0, 0, 0);
    EXPECT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(mgr.jettySpecNum_, 128U);
    EXPECT_NE(mgr.ccuResBaseVa_, 0ULL);
    EXPECT_NE(mgr.wqeBBMgr_, nullptr);
}

TEST_F(CcuJettyCtxMgrV2Test, Ut_Alloc_When_NoPfeStrategy_Expect_ReturnError)
{
    CcuJettyCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);

    std::vector<JettyInfo> jettyInfos;
    EXPECT_NE(mgr.Alloc(2, 1, 32, jettyInfos), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuJettyCtxMgrV2Test, Ut_Alloc_When_Normal_Expect_ReturnSuccess)
{
    CcuJettyCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);
    InjectPfeStrategy(mgr, 2);

    std::vector<JettyInfo> jettyInfos;
    EXPECT_EQ(mgr.Alloc(2, 1, 32, jettyInfos), HcclResult::HCCL_SUCCESS);
    ASSERT_FALSE(jettyInfos.empty());
    EXPECT_EQ(jettyInfos[0].jettyCtxId, 0);
    EXPECT_EQ(jettyInfos[0].taJettyId, CCU_START_TA_JETTY_ID);
    EXPECT_EQ(jettyInfos[0].sqDepth, CCU_V2_FIXED_SQ_SIZE / CCU_WQE_NUM_PER_SQE);
}

TEST_F(CcuJettyCtxMgrV2Test, Ut_Alloc_When_SqSizeNot32_Expect_ForcedTo32)
{
    CcuJettyCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);
    InjectPfeStrategy(mgr, 2);

    std::vector<JettyInfo> jettyInfos;
    EXPECT_EQ(mgr.Alloc(2, 1, 64, jettyInfos), HcclResult::HCCL_SUCCESS);
    ASSERT_FALSE(jettyInfos.empty());
    EXPECT_EQ(jettyInfos[0].sqDepth, CCU_V2_FIXED_SQ_SIZE / CCU_WQE_NUM_PER_SQE);
}

TEST_F(CcuJettyCtxMgrV2Test, Ut_Alloc_When_AllGroupsUsed_Expect_ReturnHCCL_E_UNAVAIL)
{
    CcuJettyCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);
    InjectPfeStrategy(mgr, 2);
    mgr.pfeMgr_.pfeJettyMap_[2].size = 1;

    std::vector<JettyInfo> jettyInfos;
    ASSERT_EQ(mgr.Alloc(2, 1, 32, jettyInfos), HcclResult::HCCL_SUCCESS);

    std::vector<JettyInfo> jettyInfos2;
    EXPECT_EQ(mgr.Alloc(2, 1, 32, jettyInfos2), HcclResult::HCCL_E_UNAVAIL);
}

TEST_F(CcuJettyCtxMgrV2Test, Ut_Config_When_NotAllocated_Expect_ReturnHCCL_E_PARA)
{
    CcuJettyCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);
    InjectPfeStrategy(mgr, 2);

    JettyInfo info{};
    info.jettyCtxId = 0;
    std::vector<JettyInfo> jettyInfos{info};
    JettyCfg cfg{};
    cfg.jettyCtxId = 0;
    std::vector<JettyCfg> jettyCfgs{cfg};

    EXPECT_EQ(mgr.Config(2, jettyInfos, jettyCfgs), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuJettyCtxMgrV2Test, Ut_Config_When_AlreadyConfigured_Expect_ReturnSuccess)
{
    CcuJettyCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);
    InjectPfeStrategy(mgr, 2);

    std::vector<JettyInfo> jettyInfos;
    ASSERT_EQ(mgr.Alloc(2, 1, 32, jettyInfos), HcclResult::HCCL_SUCCESS);

    JettyCfg cfg{};
    cfg.jettyCtxId = jettyInfos[0].jettyCtxId;
    std::vector<JettyCfg> jettyCfgs{cfg};

    MOCKER(hcomm::HccpRaTlvCcuCustomChannel).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    EXPECT_EQ(mgr.Config(2, jettyInfos, jettyCfgs), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(mgr.Config(2, jettyInfos, jettyCfgs), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuJettyCtxMgrV2Test, Ut_Config_When_CfgSizeMismatch_Expect_ReturnHCCL_E_PARA)
{
    CcuJettyCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);
    InjectPfeStrategy(mgr, 2);

    std::vector<JettyInfo> jettyInfos;
    ASSERT_EQ(mgr.Alloc(2, 1, 32, jettyInfos), HcclResult::HCCL_SUCCESS);

    std::vector<JettyCfg> jettyCfgs;
    EXPECT_EQ(mgr.Config(2, jettyInfos, jettyCfgs), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuJettyCtxMgrV2Test, Ut_Release_When_EmptyJettyInfos_Expect_ReturnSuccess)
{
    CcuJettyCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);

    std::vector<JettyInfo> jettyInfos;
    EXPECT_EQ(mgr.Release(2, jettyInfos), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuJettyCtxMgrV2Test, Ut_Release_When_UseCntGreaterThanOne_Expect_Decrement)
{
    CcuJettyCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);
    InjectPfeStrategy(mgr, 2);

    std::vector<JettyInfo> jettyInfos;
    ASSERT_EQ(mgr.Alloc(2, 1, 32, jettyInfos), HcclResult::HCCL_SUCCESS);
    EXPECT_GT(mgr.ctxGroups_[0].useCnt, 1U);

    EXPECT_EQ(mgr.Release(2, jettyInfos), HcclResult::HCCL_SUCCESS);
    EXPECT_GT(mgr.ctxGroups_[0].useCnt, 0U);
}

TEST_F(CcuJettyCtxMgrV2Test, Ut_Release_When_LastUseCnt_Expect_FullRelease)
{
    CcuJettyCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);
    InjectPfeStrategy(mgr, 2);

    std::vector<JettyInfo> jettyInfos;
    ASSERT_EQ(mgr.Alloc(2, 1, 32, jettyInfos), HcclResult::HCCL_SUCCESS);

    uint32_t useCnt = mgr.ctxGroups_[0].useCnt;
    for (uint32_t i = 0; i < useCnt; i++) {
        EXPECT_EQ(mgr.Release(2, jettyInfos), HcclResult::HCCL_SUCCESS);
    }
    EXPECT_EQ(mgr.ctxGroups_[0].useCnt, 0U);
}

TEST_F(CcuJettyCtxMgrV2Test, Ut_Release_When_NotAllocated_Expect_ReturnHCCL_E_PARA)
{
    CcuJettyCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);
    InjectPfeStrategy(mgr, 2);

    JettyInfo info{};
    info.jettyCtxId = 0;
    std::vector<JettyInfo> jettyInfos{info};

    EXPECT_EQ(mgr.Release(2, jettyInfos), HcclResult::HCCL_E_PARA);
}
