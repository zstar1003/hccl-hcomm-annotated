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
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/mokc.h>

#include "orion_adapter_hccp.h"
#include "hccp_tlv_hdc_manager.h"

#define private public
#define protected public
#include "ccu_dfx.h"
#include "ccu_context_mgr_imp.h"
#include "ccu_ctx.h"
#include "task_param.h"
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_error_handler.h"
#undef private
#undef protected

using namespace std;
using namespace Hccl;

class CcuDfxTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuDfxTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CcuDfxTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CcuDfxTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuDfxTest TearDown" << std::endl;
    }
};

TEST_F(CcuDfxTest, get_ccu_error_msg_should_fail_when_param_error)
{
    ParaCcu ccuTaskParam;
    ccuTaskParam.dieId = 0;
    ccuTaskParam.missionId = 0;
    ccuTaskParam.executeId = 0;
    vector<CcuErrorInfo> errorInfo{};
    uint16_t status = 0;
    uint16_t instrId = 0;
    EXPECT_EQ(GetCcuErrorMsg(100, status, ccuTaskParam, "", errorInfo), HcclResult::HCCL_E_PARA);
}

void GetCcuErrorMsgExcptionStub(
    int32_t deviceId, const ParaCcu& ccuTaskParam, const std::string& groupRankContent, vector<CcuErrorInfo>& errorInfo)
{
    THROW<CcuApiException>("API failed.");
}

TEST_F(CcuDfxTest, get_ccu_error_msg_should_fail_when_throw_exception)
{
    ParaCcu ccuTaskParam;
    ccuTaskParam.dieId = 0;
    ccuTaskParam.missionId = 0;
    ccuTaskParam.executeId = 0;
    vector<CcuErrorInfo> errorInfo{};
    uint16_t status = 0xffff;
    uint16_t instrId = 0;

    MOCKER_CPP(&CtxMgrImp::Init).stubs();
    CcuContext* ctx{nullptr};
    MOCKER_CPP(&CtxMgrImp::GetCtx).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(ctx));

    MOCKER_CPP(&HccpTlvHdcManager::GetTlvHandle)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(reinterpret_cast<void*>(0x1)));
    MOCKER(HrtRaTlvRequestForCustomChannel).stubs();

    EXPECT_EQ(GetCcuErrorMsg(1, status, ccuTaskParam, "", errorInfo), HcclResult::HCCL_E_INTERNAL);
}
