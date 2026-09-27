/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>

#include "hccl/hcom.h"

namespace error_message {
int32_t RegisterFormatErrorMessage(const char* error_msg, size_t error_msg_len);
}

class HcclInnerStubLevel2Test : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(HcclInnerStubLevel2Test, GetWorkflowMode_ReturnsOpBase)
{
    HcclWorkflowMode mode = GetWorkflowMode();
    EXPECT_EQ(mode, HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
}

TEST_F(HcclInnerStubLevel2Test, GetWorkflowMode_ConsistentResults)
{
    HcclWorkflowMode mode1 = GetWorkflowMode();
    HcclWorkflowMode mode2 = GetWorkflowMode();
    EXPECT_EQ(mode1, mode2);
}

TEST_F(HcclInnerStubLevel2Test, RegisterFormatErrorMessage_ValidInput)
{
    const char* errorMsg = "Test error message";
    size_t len = strlen(errorMsg);
    int32_t result = error_message::RegisterFormatErrorMessage(errorMsg, len);
    EXPECT_EQ(result, 0);
}

TEST_F(HcclInnerStubLevel2Test, RegisterFormatErrorMessage_NullPointer)
{
    int32_t result = error_message::RegisterFormatErrorMessage(nullptr, 0);
    EXPECT_EQ(result, 0);
}

TEST_F(HcclInnerStubLevel2Test, RegisterFormatErrorMessage_EmptyString)
{
    int32_t result = error_message::RegisterFormatErrorMessage("", 0);
    EXPECT_EQ(result, 0);
}

TEST_F(HcclInnerStubLevel2Test, RegisterFormatErrorMessage_LongString)
{
    std::string longMsg(2000, 'x');
    int32_t result = error_message::RegisterFormatErrorMessage(longMsg.c_str(), longMsg.size());
    EXPECT_EQ(result, 0);
}

TEST_F(HcclInnerStubLevel2Test, RegisterFormatErrorMessage_MultipleCalls)
{
    for (int i = 0; i < 100; i++) {
        int32_t result = error_message::RegisterFormatErrorMessage("test", 4);
        EXPECT_EQ(result, 0);
    }
}
