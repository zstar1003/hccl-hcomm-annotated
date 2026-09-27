/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define private public
#define protected public

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <chrono>

#include "ccu_res_specs_legacy.h"
#include "hccl_common_v2.h"

#undef private
#undef protected

#include "ccu_res_specs_test_common.h"

using namespace Hccl;

TEST_F(CcuResSpecsTest, St_Init_When_CcuDriverOk_Expect_Return_Ok)
{
    MOCKER(HrtRaTlvRequestForCustomChannel).stubs().will(invoke(MockCcuDriverInterfaceReturnDieEnableStub));

    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(MAX_MODULE_DEVICE_NUM));
    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    ccuResSpecs.Init(devLogicId);
    for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
        EXPECT_EQ(ccuResSpecs.dieEnableFlags[dieId], true);
    }
}

TEST_F(CcuResSpecsTest, St_InitGetCcuVersion_When_CcuV1_Expect_Return_Ok)
{
    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    const uint8_t dieId = 1;
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuOneDieResource(ccuResSpecs, devLogicId, dieId, ccuVersion);

    EXPECT_EQ(ccuResSpecs.GetCcuVersion(), CcuVersion::CCU_V1);
}

TEST_F(CcuResSpecsTest, St_GetDieEnableFlag_When_DieIsValid_Expect_Return_Ok)
{
    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    const uint8_t dieId = 1;
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuOneDieResource(ccuResSpecs, devLogicId, dieId, ccuVersion);

    bool dieFlag = false;
    auto ret = ccuResSpecs.GetDieEnableFlag(dieId, dieFlag);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(dieFlag, true);
}

TEST_F(CcuResSpecsTest, St_GetDieEnableFlag_When_DieIsValidButDisable_Expect_Return_Ok)
{
    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    const uint8_t dieId = 1;
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuOneDieResource(ccuResSpecs, devLogicId, dieId, ccuVersion);
    ccuResSpecs.dieEnableFlags[dieId] = false;

    bool dieFlag = false;
    auto ret = ccuResSpecs.GetDieEnableFlag(dieId, dieFlag);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(dieFlag, false);
}

TEST_F(CcuResSpecsTest, St_PublicFunc_When_DieIsInvalid_Expect_Return_ErrorPara)
{
    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    const uint8_t dieId = 1;
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuOneDieResource(ccuResSpecs, devLogicId, dieId, ccuVersion);

    uint8_t invalidDieId = MAX_CCU_IODIE_NUM;
    uint64_t resourceAddr = 0;
    auto ret = ccuResSpecs.GetResourceAddr(invalidDieId, resourceAddr);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
    EXPECT_EQ(resourceAddr, 0);
}

TEST_F(CcuResSpecsTest, St_PublicFunc_When_DieIsDisable_Expect_Return_ErrorPara)
{
    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    const uint8_t dieId = 1;
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuOneDieResource(ccuResSpecs, devLogicId, dieId, ccuVersion);
    ccuResSpecs.dieEnableFlags[dieId] = false;

    uint8_t invalidDieId = MAX_CCU_IODIE_NUM;
    uint64_t resourceAddr = 0;
    auto ret = ccuResSpecs.GetResourceAddr(dieId, resourceAddr);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
    EXPECT_EQ(resourceAddr, 0);
}

TEST_F(CcuResSpecsTest, St_GetResourceAddr_When_DieIsValid_Expect_Return_Ok)
{
    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    const uint8_t dieId = 1;
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuOneDieResource(ccuResSpecs, devLogicId, dieId, ccuVersion);

    uint64_t resourceAddr = 0;
    auto ret = ccuResSpecs.GetResourceAddr(dieId, resourceAddr);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_NE(resourceAddr, 0);
}

TEST_F(CcuResSpecsTest, St_GetResourceAddr_When_DieIsValidAndCcuV1_Expect_Return_Ok)
{
    // 该用例按正常ccu满规格设计
    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    const uint8_t dieId = 1;
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuOneDieResource(ccuResSpecs, devLogicId, dieId, ccuVersion);
    ccuResSpecs.ccuVersion = CcuVersion::CCU_V1;

    uint64_t xnBaseAddr = 0;
    auto ret = ccuResSpecs.GetXnBaseAddr(dieId, xnBaseAddr);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    const uint64_t correctOffset = CCU_V1_CCUM_OFFSET + 0x108000;
    EXPECT_EQ(xnBaseAddr, ccuResSpecs.resSpecs[dieId].resourceAddr + correctOffset);
}

TEST_F(CcuResSpecsTest, St_GetResSpec_When_InitOk_Expect_Return_Ok)
{
    using GetResSpecFunc = HcclResult (CcuResSpecifications::*)(const uint8_t, uint32_t&) const;
    constexpr GetResSpecFunc GET_RES_SPEC_FUNC_ARRAY[]
        = {&CcuResSpecifications::GetLoopEngineNum, &CcuResSpecifications::GetMsNum,
           &CcuResSpecifications::GetCkeNum,        &CcuResSpecifications::GetXnNum,
           &CcuResSpecifications::GetGsaNum,        &CcuResSpecifications::GetInstructionNum,
           &CcuResSpecifications::GetMissionNum,    &CcuResSpecifications::GetMsId,
           &CcuResSpecifications::GetMissionKey,    &CcuResSpecifications::GetChannelNum,
           &CcuResSpecifications::GetJettyNum,      &CcuResSpecifications::GetPfeReservedNum,
           &CcuResSpecifications::GetPfeNum,        &CcuResSpecifications::GetWqeBBNum};

    CcuResSpecifications ccuResSpecs;
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM; // 避免影响其他用例
    const uint8_t dieId = 1;
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuOneDieResource(ccuResSpecs, devLogicId, dieId, ccuVersion);
    ccuResSpecs.ccuVersion = CcuVersion::CCU_V1;

    for (const auto& getFunc : GET_RES_SPEC_FUNC_ARRAY) {
        uint32_t capacity = 0;
        std::cout << "Test GetResSpecFunc: " << getFunc << std::endl;
        EXPECT_EQ((ccuResSpecs.*getFunc)(dieId, capacity), HcclResult::HCCL_SUCCESS);
        EXPECT_NE(capacity, 0);
    }
}
