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

extern HcclResult HandleDispatchAllStreams();

class UtAicpuTsHandleDispatchAllStreams : public UtAicpuTsBase {
protected:
    static void SetUpTestCase() { std::cout << "UtAicpuTsHandleDispatchAllStreams tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "UtAicpuTsHandleDispatchAllStreams tests tear down." << std::endl; }

    virtual void SetUp() override
    {
        std::cout << "A Test case in UtAicpuTsHandleDispatchAllStreams SetUp" << std::endl;
        UtAicpuTsBase::SetUp();
    }

    virtual void TearDown() override
    {
        UtAicpuTsBase::TearDown();
        std::cout << "A Test case in UtAicpuTsHandleDispatchAllStreams TearDown" << std::endl;
    }

    int32_t res{HCCL_E_RESERVED};
};

TEST_F(UtAicpuTsHandleDispatchAllStreams, Ut_HandleDispatchAllStreams_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    res = HandleDispatchAllStreams();
    EXPECT_EQ(res, HCCL_SUCCESS);
}
