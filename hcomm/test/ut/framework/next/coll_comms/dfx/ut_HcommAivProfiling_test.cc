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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "hccl/hccl_diag.h"
#include "hccl_comm_pub.h"
#include "hcomm_adapter_hccp.h"

using namespace hccl;
using namespace hcomm;

class HcclReportAivKernelTest : public testing::Test {
protected:
    virtual void SetUp() override {}

    virtual void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(HcclReportAivKernelTest, Ut_HcclReportAivKernel_When_CommIsNull_Expect_ReturnHCCL_E_PTR)
{
    HcclResult ret = HcclReportAivKernel(nullptr, 12345);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

class HccpBatchQueryJettyStatusTest : public testing::Test {
protected:
    virtual void SetUp() override {}

    virtual void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(HccpBatchQueryJettyStatusTest, Ut_HccpBatchQueryJettyStatus_When_SizeMismatch_Expect_ReturnHCCL_E_PARA)
{
    CtxHandle ctxHandle = reinterpret_cast<CtxHandle>(1);
    std::vector<JettyHandle> jettyHandles;
    jettyHandles.push_back(reinterpret_cast<JettyHandle>(1));
    jettyHandles.push_back(reinterpret_cast<JettyHandle>(2));
    jettyHandles.push_back(reinterpret_cast<JettyHandle>(3));
    std::vector<JettyStatus> jettyAttrs;
    u32 num = 2;
    HcclResult ret = HccpBatchQueryJettyStatus(ctxHandle, jettyHandles, jettyAttrs, num);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HccpBatchQueryJettyStatusTest, Ut_HccpBatchQueryJettyStatus_When_NumIsZero_Expect_ReturnHCCL_SUCCESS)
{
    CtxHandle ctxHandle = reinterpret_cast<CtxHandle>(1);
    std::vector<JettyHandle> jettyHandles;
    std::vector<JettyStatus> jettyAttrs;
    u32 num = 0;
    HcclResult ret = HccpBatchQueryJettyStatus(ctxHandle, jettyHandles, jettyAttrs, num);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
