/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ut_aicpu_ts_base.h"

using namespace hccl;

class UtAicpuTsHcommThreadResAcquireTimeOut : public UtAicpuTsBase {
protected:
    static void SetUpTestCase() { std::cout << "UtAicpuTsHcommThreadResAcquireTimeOut tests set up." << std::endl; }

    static void TearDownTestCase()
    {
        std::cout << "UtAicpuTsHcommThreadResAcquireTimeOut tests tear down." << std::endl;
    }

    virtual void SetUp() override
    {
        std::cout << "A Test case in UtAicpuTsHcommThreadResAcquireTimeOut SetUp" << std::endl;
        UtAicpuTsBase::SetUp();
    }

    uint32_t timeout = 2000;
    int32_t res{HCCL_E_RESERVED};
};

TEST_F(UtAicpuTsHcommThreadResAcquireTimeOut, Ut_HcommThreadResAcquireTimeOut_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    res = HcommThreadResAcquireTimeOut(timeout);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(
    UtAicpuTsHcommThreadResAcquireTimeOut,
    Ut_HcommThreadResAcquireTimeOut_When_Timeout_IsZero_Expect_ReturnIsHCCL_SUCCESS)
{
    timeout = 0;
    res = HcommThreadResAcquireTimeOut(timeout);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(
    UtAicpuTsHcommThreadResAcquireTimeOut,
    Ut_HcommThreadResAcquireTimeOut_When_Timeout_IsLarge_Expect_ReturnIsHCCL_SUCCESS)
{
    timeout = 100000;
    res = HcommThreadResAcquireTimeOut(timeout);
    EXPECT_EQ(res, HCCL_SUCCESS);
}
