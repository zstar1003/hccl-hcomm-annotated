/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#ifndef private
#define private public
#define protected public
#endif
#include "hccl_aicpu_utils.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

class UT_Hccl_Aicpu_Utils : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UT_Hccl_Aicpu_Utils SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "UT_Hccl_Aicpu_Utils TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(UT_Hccl_Aicpu_Utils, ut_PrintHcclOpResParam_When_resParamNull_Expect_NoCore)
{
    HcclAicpuUtils::PrintHcclOpResParam(nullptr);
}

TEST_F(UT_Hccl_Aicpu_Utils, ut_Getkey_When_userAddrNull_Expect_HCCL_E_PTR)
{
    AicpuComContext ctx;
    u32 remoteRankId;
    void* userAddr = nullptr;
    u64 length;
    u32 outKey;
    int32_t keyType;

    HcclResult ret = HcclAicpuUtils::Getkey(ctx, remoteRankId, userAddr, length, outKey, keyType);
    EXPECT_EQ(ret, HCCL_E_PTR);
}
