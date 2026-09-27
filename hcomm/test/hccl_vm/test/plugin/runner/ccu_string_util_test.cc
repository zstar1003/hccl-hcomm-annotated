/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdarg>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "ccu_string_util.h"

using namespace HcclSim;

class StringUtilTest : public testing::Test {
protected:
};

TEST_F(StringUtilTest, StringFormat_WithValidFormat)
{
    std::string result = StringFormat("test %d", 123);
    EXPECT_EQ(result, "test 123");
}

TEST_F(StringUtilTest, StringFormat_WithMultipleArgs)
{
    std::string result = StringFormat("%s %d %s", "hello", 42, "world");
    EXPECT_EQ(result, "hello 42 world");
}

TEST_F(StringUtilTest, StringFormat_WithNoArgs)
{
    std::string result = StringFormat("plain string");
    EXPECT_EQ(result, "plain string");
}

TEST_F(StringUtilTest, StringFormat_WithFloat)
{
    std::string result = StringFormat("float: %.2f", 3.14159);
    EXPECT_EQ(result, "float: 3.14");
}

TEST_F(StringUtilTest, StringFormat_WithHex)
{
    std::string result = StringFormat("hex: 0x%x", 255);
    EXPECT_EQ(result, "hex: 0xff");
}

TEST_F(StringUtilTest, StringFormat_WithMultipleIntegers)
{
    std::string result = StringFormat("%d + %d = %d", 1, 2, 3);
    EXPECT_EQ(result, "1 + 2 = 3");
}

TEST_F(StringUtilTest, StringFormat_WithPercent)
{
    std::string result = StringFormat("100%% complete");
    EXPECT_EQ(result, "100% complete");
}

TEST_F(StringUtilTest, StringFormat_WithChar)
{
    std::string result = StringFormat("char: %c", 'A');
    EXPECT_EQ(result, "char: A");
}

TEST_F(StringUtilTest, StringFormat_WithUnsignedInt)
{
    std::string result = StringFormat("unsigned: %u", 4294967295U);
    EXPECT_EQ(result, "unsigned: 4294967295");
}

TEST_F(StringUtilTest, StringFormat_WithPointer)
{
    void* ptr = reinterpret_cast<void*>(0x1234);
    std::string result = StringFormat("ptr: %p", ptr);
    EXPECT_TRUE(result.find("1234") != std::string::npos);
}

TEST_F(StringUtilTest, StringFormat_WithLongLong)
{
    std::string result = StringFormat("long long: %lld", 9223372036854775807LL);
    EXPECT_TRUE(result.find("9223372036854775807") != std::string::npos);
}

TEST_F(StringUtilTest, StringFormat_WithEmptyFormat)
{
    std::string result = StringFormat("");
    EXPECT_EQ(result, "");
}

TEST_F(StringUtilTest, StringFormat_WithNegativeNumber)
{
    std::string result = StringFormat("negative: %d", -12345);
    EXPECT_EQ(result, "negative: -12345");
}

TEST_F(StringUtilTest, StringFormat_WithZero)
{
    std::string result = StringFormat("zero: %d", 0);
    EXPECT_EQ(result, "zero: 0");
}

TEST_F(StringUtilTest, StringFormat_WithComplexFormat)
{
    std::string result = StringFormat("[%s] code=%d, value=0x%x", "test", 42, 0xABCD);
    EXPECT_EQ(result, "[test] code=42, value=0xabcd");
}
