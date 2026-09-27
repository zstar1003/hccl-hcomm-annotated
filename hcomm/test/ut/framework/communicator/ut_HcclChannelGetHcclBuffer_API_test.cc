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
class HcclChannelGetHcclBufferTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        UT_USE_RANK_TABLE_910_1SERVER_1RANK;
        UT_COMM_CREATE_DEFAULT(comm);
    }
    void TearDown() override
    {
        Ut_Comm_Destroy(comm);
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(HcclChannelGetHcclBufferTest, Ut_HcclChannelGetHcclBuffer_When_Params_Nullptr_Return_EPTR)
{
    // 测试接口入参为空指针的场景
    HcclResult ret = HcclChannelGetHcclBuffer(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclChannelGetHcclBuffer(comm, 0, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);

    char* buffer[1];
    ret = HcclChannelGetHcclBuffer(comm, 0, (void**)&buffer, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}
