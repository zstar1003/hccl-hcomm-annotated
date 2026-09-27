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

#include "ccu_channel_ctx_mgr_v2.h"
#include "ccu_jetty_ctx_mgr_v2.h"
#include "ccu_res_specs.h"
#include "ccu_pfe_cfg_mgr.h"
#include "hcomm_adapter_hccp.h"
#include "hccl_common.h"

#define private public
#define protected public

using namespace hcomm;

static void SetupCcuResSpecsForChannel(int32_t devLogicId, uint8_t dieId)
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

static void InjectPfeStrategyForChannel(CcuChannelCtxMgrV2& mgr, uint32_t feId)
{
    PfeJettyStrategy strategy{};
    strategy.feId = feId;
    strategy.pfeId = feId;
    strategy.startTaJettyId = CCU_START_TA_JETTY_ID;
    strategy.size = 128;
    strategy.startLocalJettyCtxId = 0;
    mgr.jettyCtxMgr_.pfeMgr_.pfeJettyMap_[feId] = strategy;
}

class CcuChannelCtxMgrV2Test : public testing::Test {
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
        SetupCcuResSpecsForChannel(0, 0);
    }
    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
};

TEST_F(CcuChannelCtxMgrV2Test, Ut_Init_When_Normal_Expect_ReturnSuccess)
{
    CcuChannelCtxMgrV2 mgr(0, 0, 0);
    EXPECT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(mgr.channelResInfos_.size(), 1024U);
}

TEST_F(CcuChannelCtxMgrV2Test, Ut_Alloc_When_NoPfeStrategy_Expect_ReturnError)
{
    CcuChannelCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);

    ChannelPara para{};
    para.feId = 2;
    para.jettyNum = 1;
    para.sqSize = 32;
    std::vector<ChannelInfo> channelInfos;
    EXPECT_NE(mgr.Alloc(para, channelInfos), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuChannelCtxMgrV2Test, Ut_Alloc_When_Normal_Expect_ReturnSuccess)
{
    CcuChannelCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);
    InjectPfeStrategyForChannel(mgr, 2);

    ChannelPara para{};
    para.feId = 2;
    para.jettyNum = 1;
    para.sqSize = 32;
    std::vector<ChannelInfo> channelInfos;
    EXPECT_EQ(mgr.Alloc(para, channelInfos), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(channelInfos.size(), 8U);
    EXPECT_EQ(channelInfos[0].channelId, 0U);
    EXPECT_EQ(channelInfos[0].dieId, 0);
}

TEST_F(CcuChannelCtxMgrV2Test, Ut_Alloc_When_JettyNumMismatch_Expect_AdjustedToMapValue)
{
    CcuChannelCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);
    InjectPfeStrategyForChannel(mgr, 2);

    ChannelPara para{};
    para.feId = 2;
    para.jettyNum = 4;
    para.sqSize = 32;
    std::vector<ChannelInfo> channelInfos;
    EXPECT_EQ(mgr.Alloc(para, channelInfos), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(channelInfos.size(), 8U);
}

TEST_F(CcuChannelCtxMgrV2Test, Ut_Config_When_NotAllocated_Expect_ReturnHCCL_E_PARA)
{
    CcuChannelCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);

    ChannelCfg cfg{};
    cfg.channelId = 0;
    EXPECT_EQ(mgr.Config(cfg), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuChannelCtxMgrV2Test, Ut_Config_When_Normal_Expect_ReturnSuccess)
{
    CcuChannelCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);
    InjectPfeStrategyForChannel(mgr, 2);

    ChannelPara para{};
    para.feId = 2;
    para.jettyNum = 1;
    para.sqSize = 32;
    std::vector<ChannelInfo> channelInfos;
    ASSERT_EQ(mgr.Alloc(para, channelInfos), HcclResult::HCCL_SUCCESS);

    ChannelCfg cfg{};
    cfg.channelId = channelInfos[0].channelId;
    cfg.tpn = 0x12340000;
    cfg.jettyCfgs.resize(channelInfos[0].jettyInfos.size());
    for (size_t i = 0; i < cfg.jettyCfgs.size(); i++) {
        cfg.jettyCfgs[i].jettyCtxId = channelInfos[0].jettyInfos[i].jettyCtxId;
    }

    MOCKER(hcomm::HccpRaTlvCcuCustomChannel).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    EXPECT_EQ(mgr.Config(cfg), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuChannelCtxMgrV2Test, Ut_Config_When_DriverFails_Expect_ReturnHCCL_E_NETWORK)
{
    CcuChannelCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);
    InjectPfeStrategyForChannel(mgr, 2);

    ChannelPara para{};
    para.feId = 2;
    para.jettyNum = 1;
    para.sqSize = 32;
    std::vector<ChannelInfo> channelInfos;
    ASSERT_EQ(mgr.Alloc(para, channelInfos), HcclResult::HCCL_SUCCESS);

    ChannelCfg cfg{};
    cfg.channelId = channelInfos[0].channelId;
    cfg.jettyCfgs.resize(channelInfos[0].jettyInfos.size());
    for (size_t i = 0; i < cfg.jettyCfgs.size(); i++) {
        cfg.jettyCfgs[i].jettyCtxId = channelInfos[0].jettyInfos[i].jettyCtxId;
    }

    MOCKER(hcomm::HccpRaTlvCcuCustomChannel).stubs().will(returnValue(HcclResult::HCCL_E_NETWORK));
    EXPECT_EQ(mgr.Config(cfg), HcclResult::HCCL_E_NETWORK);
}

TEST_F(CcuChannelCtxMgrV2Test, Ut_Release_When_NotAllocated_Expect_ReturnHCCL_E_PARA)
{
    CcuChannelCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(mgr.Release(0), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuChannelCtxMgrV2Test, Ut_Release_When_Normal_Expect_ReturnSuccess)
{
    CcuChannelCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);
    InjectPfeStrategyForChannel(mgr, 2);

    ChannelPara para{};
    para.feId = 2;
    para.jettyNum = 1;
    para.sqSize = 32;
    std::vector<ChannelInfo> channelInfos;
    ASSERT_EQ(mgr.Alloc(para, channelInfos), HcclResult::HCCL_SUCCESS);

    MOCKER(hcomm::HccpRaTlvCcuCustomChannel).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    EXPECT_EQ(mgr.Release(channelInfos[0].channelId), HcclResult::HCCL_SUCCESS);
    EXPECT_FALSE(mgr.channelResInfos_[channelInfos[0].channelId].allocated);
}

TEST_F(CcuChannelCtxMgrV2Test, Ut_Config_When_ChannelIdOutOfRange_Expect_ReturnHCCL_E_PARA)
{
    CcuChannelCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);

    ChannelCfg cfg{};
    cfg.channelId = 1028;
    EXPECT_EQ(mgr.Config(cfg), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuChannelCtxMgrV2Test, Ut_Release_When_ChannelIdOutOfRange_Expect_ReturnHCCL_E_PARA)
{
    CcuChannelCtxMgrV2 mgr(0, 0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(mgr.Release(1028), HcclResult::HCCL_E_PARA);
}
