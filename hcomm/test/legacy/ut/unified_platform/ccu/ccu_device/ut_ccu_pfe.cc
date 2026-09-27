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

#include "hccp_ctx.h"
#include "orion_adapter_hccp.h"
#include "ccu_res_specs_legacy.h"
#include "ccu_eid_info.h"
#include "ccu_pfe_cfg_generator.h"
#include "ccu_pfe_mgr_legacy.h"
#include "network_api_exception.h"
#include "internal_exception.h"
#include "ccu_device_manager.h"

#undef private
#undef protected

using namespace Hccl;

class CcuPfeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GlobalMockObject::reset();
        std::cout << "CcuPfeTest SetUP" << std::endl;
    }

    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        std::cout << "CcuPfeTest TearDown" << std::endl;
    }

    virtual void SetUp()
    {
        GlobalMockObject::reset();
        std::cout << "A Test case in CcuPfeTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuPfeTest TearDown" << std::endl;
    }
};

TEST_F(CcuPfeTest, GetDevEidInfoListPass)
{
    u32 num = 2;

    MOCKER(RaGetDevEidInfoNum).stubs().with(mockcpp::any(), outBoundP(&num, sizeof(num))).will(returnValue(0));

    MOCKER(RaGetDevEidInfoList)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&num, sizeof(num)))
        .will(returnValue(0));

    HRaInfo info(HrtNetworkMode::HDC, 0);
    vector<HrtDevEidInfo> eidInfoList = {};
    EXPECT_EQ(0, eidInfoList.size());
    eidInfoList = HrtRaGetDevEidInfoList(info);
    EXPECT_NE(0, eidInfoList.size());
}

TEST_F(CcuPfeTest, GetDevEidInfoListFail)
{
    u32 num = 2;

    MOCKER(RaGetDevEidInfoNum).stubs().with(mockcpp::any(), outBoundP(&num, sizeof(num))).will(returnValue(1));

    MOCKER(RaGetDevEidInfoList)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&num, sizeof(num)))
        .will(returnValue(0));

    HRaInfo info(HrtNetworkMode::HDC, 0);
    EXPECT_THROW(HrtRaGetDevEidInfoList(info), NetworkApiException);
}

TEST_F(CcuPfeTest, GetDevEidInfoListFail1)
{
    u32 num = 2;

    MOCKER(RaGetDevEidInfoNum).stubs().with(mockcpp::any(), outBoundP(&num, sizeof(num))).will(returnValue(0));

    MOCKER(RaGetDevEidInfoList)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&num, sizeof(num)))
        .will(returnValue(1));

    HRaInfo info(HrtNetworkMode::HDC, 0);
    EXPECT_THROW(HrtRaGetDevEidInfoList(info), NetworkApiException);
}

TEST_F(CcuPfeTest, HcclEidInfoPass)
{
    s32 phyDeviceId = 8;

    vector<HrtDevEidInfo> eidInfoListStbu;
    HrtDevEidInfo eidInfo;
    eidInfo.name = "udma0";
    eidInfo.dieId = 0;
    eidInfo.funcId = 3;

    eidInfoListStbu.push_back(eidInfo);

    MOCKER(HrtRaGetDevEidInfoList).stubs().with(mockcpp::any()).will(returnValue(eidInfoListStbu));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(0));

    vector<HrtDevEidInfo> eidInfoList = {};
    EXPECT_EQ(0, eidInfoList.size());

    HcclResult result = CcuEidInfo::GetInstance(phyDeviceId).GetEidInfo(phyDeviceId, eidInfoList);

    EXPECT_NE(0, eidInfoList.size());
    EXPECT_EQ(HCCL_SUCCESS, result);
}

TEST_F(CcuPfeTest, GetPfeJettyCtxCfg)
{
    s32 phyDeviceId = 8;
    u8 die_id = 0;

    vector<HrtDevEidInfo> eidInfoListStbu = {};
    HrtDevEidInfo eidInfo;
    eidInfo.name = "udma0";
    eidInfo.dieId = 0;
    eidInfo.funcId = 3;

    eidInfoListStbu.push_back(eidInfo);

    MOCKER(HrtRaGetDevEidInfoList).stubs().with(mockcpp::any()).will(returnValue(eidInfoListStbu));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(0));

    CcuResSpecifications::GetInstance(phyDeviceId).dieEnableFlags[die_id] = true;
    CcuResSpecifications::GetInstance(phyDeviceId).resSpecs[die_id].jettyNum = 128;

    vector<PfeJettyCtxCfg> pfeJettyCtxCfg = {};
    EXPECT_EQ(0, pfeJettyCtxCfg.size());
    pfeJettyCtxCfg = CcuPfeCfgGenerator::GetInstance(phyDeviceId).GetPfeJettyCtxCfg(die_id);
    EXPECT_NE(0, pfeJettyCtxCfg.size());
    CcuResSpecifications::GetInstance(phyDeviceId).ifInit = false;
}

TEST_F(CcuPfeTest, GetPfeJettyStrategy)
{
    s32 logicDevId = 8;
    u32 phyDeviceId = 8;
    u8 dieId = 0;
    u8 feId = 3;

    std::vector<PfeJettyCtxCfg> pfeJettyCtxCfgListStub;
    PfeJettyCtxCfg pfeJettyCtxCfg;
    pfeJettyCtxCfg.feId = feId;
    pfeJettyCtxCfg.startTaJettyId = (CCU_START_TA_JETTY_ID + ONE_CCU_PFE_USE_JETTY_NUM);
    pfeJettyCtxCfg.size = ONE_CCU_PFE_USE_JETTY_NUM;

    pfeJettyCtxCfgListStub.push_back(pfeJettyCtxCfg);

    CustomChannelInfoIn inBuff;
    inBuff.data.dataInfo.dataLen = 512;
    inBuff.data.dataInfo.dataArraySize = 64;

    MOCKER(HrtRaTlvRequestForCustomChannel)
        .stubs()
        .with(
            mockcpp::any(), mockcpp::any(), outBoundP(reinterpret_cast<void*>(&inBuff), sizeof(inBuff)),
            outBoundP(reinterpret_cast<void*>(&inBuff), sizeof(inBuff)));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(0));

    MOCKER_CPP(&CcuPfeCfgGenerator::GetPfeJettyCtxCfg).stubs().will(returnValue(pfeJettyCtxCfgListStub));
    auto& ccuResSpecs = CcuResSpecifications::GetInstance(logicDevId);
    ccuResSpecs.dieEnableFlags[dieId] = true;

    CcuPfeMgr ccuPfeManager(logicDevId, dieId, phyDeviceId);
    PfeJettyStrategy pfeJettyCtxCfgRet = {};
    EXPECT_EQ(ccuPfeManager.GetPfeStrategy(feId, pfeJettyCtxCfgRet), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(feId, pfeJettyCtxCfgRet.feId);
    EXPECT_EQ(ONE_CCU_PFE_USE_JETTY_NUM, pfeJettyCtxCfgRet.size);

    ccuResSpecs.ifInit = false; // 恢复未初始化状态
}
