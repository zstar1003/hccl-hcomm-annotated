/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_api_base_test.h"

class HcomGetCommHandleByGroupTest : public BaseInit {
public:
    void SetUp() override { BaseInit::SetUp(); }

    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(HcomGetCommHandleByGroupTest, Ut_HcomGetCommHandleByGroup_When_GroupIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    HcclComm commHandle = nullptr;
    HcclResult ret = HcomGetCommHandleByGroup(nullptr, &commHandle);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcomGetCommHandleByGroupTest, Ut_HcomGetCommHandleByGroup_When_CommHandleIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    char group[] = "hccl_world_group";
    HcclResult ret = HcomGetCommHandleByGroup(group, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcomGetCommHandleByGroupTest, Ut_HcomGetCommHandleByGroup_When_GroupNotExist_Expect_ReturnIsHCCL_E_NOT_FOUND)
{
    char group[] = "non_existent_group";
    HcclComm commHandle = nullptr;
    HcclResult ret = HcomGetCommHandleByGroup(group, &commHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}
