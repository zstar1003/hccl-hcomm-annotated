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
#include "hcomm_primitives.h"

#include "op_base.h"

using namespace hccl;

class UtCpuHcommFenceOnThread : public testing::Test {
protected:
    virtual void SetUp() override {}

    virtual void TearDown() override { GlobalMockObject::verify(); }

    // thread is unused by HcommFenceOnThread, any value is acceptable.
    ThreadHandle thread = static_cast<ThreadHandle>(0x01);
    int32_t res{HCCL_E_RESERVED};
};

TEST_F(UtCpuHcommFenceOnThread, Ut_HcommFenceOnThread_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    MOCKER(&HcommFlushV2).stubs().will(returnValue(HCCL_SUCCESS));
    res = HcommFenceOnThread(thread);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtCpuHcommFenceOnThread, Ut_HcommFenceOnThread_When_Thread_IsNull_Expect_ReturnIsHCCL_SUCCESS)
{
    MOCKER(&HcommFlushV2).stubs().will(returnValue(HCCL_SUCCESS));
    // thread is cast to void — nullptr is acceptable.
    res = HcommFenceOnThread(0);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtCpuHcommFenceOnThread, Ut_HcommFenceOnThread_When_FlushV2_Fails_Expect_ErrorCodePropagated)
{
    MOCKER(&HcommFlushV2).stubs().will(returnValue(HCCL_E_INTERNAL));
    res = HcommFenceOnThread(thread);
    EXPECT_EQ(res, HCCL_E_INTERNAL);
}
