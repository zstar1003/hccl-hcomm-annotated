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

#include "ccu_wqebb_mgr_v2.h"
#include "ccu_res_specs.h"
#include "hccl_common.h"

#define private public
#define protected public

using namespace hcomm;

static void SetupCcuResSpecsForWqeBB(int32_t devLogicId, uint8_t dieId, uint32_t jettyNum, uint32_t wqeBBNum)
{
    auto& specs = CcuResSpecifications::GetInstance(devLogicId);
    specs.initFlag_ = true;
    specs.ccuVersion_ = CcuVersion::CCU_V2;
    specs.dieEnableFlags_[dieId] = true;
    specs.resSpecs_[dieId].jettyNum = jettyNum;
    specs.resSpecs_[dieId].wqeBBNum = wqeBBNum;
    specs.resSpecs_[dieId].resourceAddr = 0xE7FFBF800000;
}

class CcuWqeBBMgrV2Test : public testing::Test {
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
    }
    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
};

TEST_F(CcuWqeBBMgrV2Test, Ut_Init_When_JettyNumIsZero_Expect_ReturnHCCL_E_UNAVAIL)
{
    SetupCcuResSpecsForWqeBB(0, 0, 0, 4096);
    CcuWqeBBMgrV2 mgr(0, 0);
    EXPECT_EQ(mgr.Init(), HcclResult::HCCL_E_UNAVAIL);
}

TEST_F(CcuWqeBBMgrV2Test, Ut_Init_When_Normal_Expect_ReturnSuccess)
{
    SetupCcuResSpecsForWqeBB(0, 0, 128, 4096);
    CcuWqeBBMgrV2 mgr(0, 0);
    EXPECT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(mgr.allocatedWqeBBBlocks_.size(), 4096U / 32U);
}

TEST_F(CcuWqeBBMgrV2Test, Ut_Alloc_When_Normal_Expect_ReturnSuccess)
{
    SetupCcuResSpecsForWqeBB(0, 0, 128, 4096);
    CcuWqeBBMgrV2 mgr(0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);

    WqeBBReq req{};
    req.jettyCtxStartId = 0;
    req.sqSize = 32;
    ResInfo wqeBBInfo{};
    EXPECT_EQ(mgr.Alloc(req, wqeBBInfo), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(wqeBBInfo.startId, 0U * CCU_V2_FIXED_SQ_SIZE);
    EXPECT_EQ(wqeBBInfo.num, CCU_V2_FIXED_SQ_SIZE);
}

TEST_F(CcuWqeBBMgrV2Test, Ut_Alloc_When_StartIdExceedsSize_Expect_ReturnHCCL_E_INTERNAL)
{
    SetupCcuResSpecsForWqeBB(0, 0, 128, 4096);
    CcuWqeBBMgrV2 mgr(0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);

    WqeBBReq req{};
    req.jettyCtxStartId = static_cast<uint32_t>(mgr.allocatedWqeBBBlocks_.size()) + 1;
    req.sqSize = 32;
    ResInfo wqeBBInfo{};
    EXPECT_EQ(mgr.Alloc(req, wqeBBInfo), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(CcuWqeBBMgrV2Test, Ut_Alloc_When_AlreadyAllocated_Expect_ReturnHCCL_E_INTERNAL)
{
    SetupCcuResSpecsForWqeBB(0, 0, 128, 4096);
    CcuWqeBBMgrV2 mgr(0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);

    WqeBBReq req{};
    req.jettyCtxStartId = 0;
    req.sqSize = 32;
    ResInfo wqeBBInfo{};
    ASSERT_EQ(mgr.Alloc(req, wqeBBInfo), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(mgr.Alloc(req, wqeBBInfo), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(CcuWqeBBMgrV2Test, Ut_Release_When_Normal_Expect_ReturnSuccess)
{
    SetupCcuResSpecsForWqeBB(0, 0, 128, 4096);
    CcuWqeBBMgrV2 mgr(0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);

    WqeBBReq req{};
    req.jettyCtxStartId = 0;
    req.sqSize = 32;
    ResInfo wqeBBInfo{};
    ASSERT_EQ(mgr.Alloc(req, wqeBBInfo), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(mgr.Release(wqeBBInfo), HcclResult::HCCL_SUCCESS);
    EXPECT_FALSE(mgr.allocatedWqeBBBlocks_[0]);
}

TEST_F(CcuWqeBBMgrV2Test, Ut_Release_When_StartIdExceedsSize_Expect_ReturnHCCL_E_INTERNAL)
{
    SetupCcuResSpecsForWqeBB(0, 0, 128, 4096);
    CcuWqeBBMgrV2 mgr(0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);

    ResInfo wqeBBInfo{};
    wqeBBInfo.startId = (static_cast<uint32_t>(mgr.allocatedWqeBBBlocks_.size()) + 1) * CCU_V2_FIXED_SQ_SIZE;
    wqeBBInfo.num = CCU_V2_FIXED_SQ_SIZE;
    EXPECT_EQ(mgr.Release(wqeBBInfo), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(CcuWqeBBMgrV2Test, Ut_Release_When_NotAllocated_Expect_ReturnErrorPara)
{
    SetupCcuResSpecsForWqeBB(0, 0, 128, 4096);
    CcuWqeBBMgrV2 mgr(0, 0);
    ASSERT_EQ(mgr.Init(), HcclResult::HCCL_SUCCESS);

    ResInfo wqeBBInfo{};
    wqeBBInfo.startId = 0;
    wqeBBInfo.num = CCU_V2_FIXED_SQ_SIZE;
    EXPECT_EQ(mgr.Release(wqeBBInfo), HcclResult::HCCL_E_PARA);
}
