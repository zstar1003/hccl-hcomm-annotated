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

#include "ccu_fp16.h"

class FwkTypesTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(FwkTypesTest, FP16_DefaultConstructor)
{
    FP16 x;
    EXPECT_FLOAT_EQ(x.to_float(), 0.0f);
}

TEST_F(FwkTypesTest, FP16_FloatConstructor_Positive)
{
    FP16 x(1.5f);
    EXPECT_FLOAT_EQ(x.to_float(), 1.5f);
}

TEST_F(FwkTypesTest, FP16_FloatConstructor_Negative)
{
    FP16 x(-3.25f);
    EXPECT_FLOAT_EQ(x.to_float(), -3.25f);
}

TEST_F(FwkTypesTest, FP16_FloatConstructor_Zero)
{
    FP16 x(0.0f);
    EXPECT_FLOAT_EQ(x.to_float(), 0.0f);
}

TEST_F(FwkTypesTest, FP16_HalfConstructor_FromFloat)
{
    FP16 intermediate(2.75f);
    half h = intermediate.value;
    FP16 x(h);
    EXPECT_FLOAT_EQ(x.to_float(), 2.75f);
}

TEST_F(FwkTypesTest, FP16_HalfConstructor_Zero)
{
    FP16 intermediate(0.0f);
    half h = intermediate.value;
    FP16 x(h);
    EXPECT_FLOAT_EQ(x.to_float(), 0.0f);
}

TEST_F(FwkTypesTest, FP16_IntConstructor_Positive)
{
    FP16 x(5);
    EXPECT_FLOAT_EQ(x.to_float(), 5.0f);
}

TEST_F(FwkTypesTest, FP16_IntConstructor_Negative)
{
    FP16 x(-10);
    EXPECT_FLOAT_EQ(x.to_float(), -10.0f);
}

TEST_F(FwkTypesTest, FP16_IntConstructor_Zero)
{
    FP16 x(0);
    EXPECT_FLOAT_EQ(x.to_float(), 0.0f);
}

TEST_F(FwkTypesTest, FP16_Addition)
{
    FP16 a(3.0f);
    FP16 b(4.0f);
    FP16 c = a + b;
    EXPECT_FLOAT_EQ(c.to_float(), 7.0f);
}

TEST_F(FwkTypesTest, FP16_Addition_Negative)
{
    FP16 a(5.0f);
    FP16 b(-3.0f);
    FP16 c = a + b;
    EXPECT_FLOAT_EQ(c.to_float(), 2.0f);
}

TEST_F(FwkTypesTest, FP16_Subtraction)
{
    FP16 a(10.0f);
    FP16 b(3.0f);
    FP16 c = a - b;
    EXPECT_FLOAT_EQ(c.to_float(), 7.0f);
}

TEST_F(FwkTypesTest, FP16_Subtraction_Negative)
{
    FP16 a(2.0f);
    FP16 b(5.0f);
    FP16 c = a - b;
    EXPECT_FLOAT_EQ(c.to_float(), -3.0f);
}

TEST_F(FwkTypesTest, FP16_Multiplication)
{
    FP16 a(3.0f);
    FP16 b(4.0f);
    FP16 c = a * b;
    EXPECT_FLOAT_EQ(c.to_float(), 12.0f);
}

TEST_F(FwkTypesTest, FP16_Multiplication_ByZero)
{
    FP16 a(100.0f);
    FP16 b(0.0f);
    FP16 c = a * b;
    EXPECT_FLOAT_EQ(c.to_float(), 0.0f);
}

TEST_F(FwkTypesTest, FP16_AdditionAssignment)
{
    FP16 a(3.0f);
    FP16 b(4.0f);
    a += b;
    EXPECT_FLOAT_EQ(a.to_float(), 7.0f);
}

TEST_F(FwkTypesTest, FP16_AdditionAssignment_Chained)
{
    FP16 a(1.0f);
    a += FP16(2.0f);
    a += FP16(3.0f);
    EXPECT_FLOAT_EQ(a.to_float(), 6.0f);
}

TEST_F(FwkTypesTest, FP16_GreaterThan_True)
{
    FP16 a(5.0f);
    FP16 b(2.0f);
    EXPECT_TRUE(a > b);
}

TEST_F(FwkTypesTest, FP16_GreaterThan_False)
{
    FP16 a(1.0f);
    FP16 b(3.0f);
    EXPECT_FALSE(a > b);
}

TEST_F(FwkTypesTest, FP16_GreaterThan_Equal)
{
    FP16 a(4.0f);
    FP16 b(4.0f);
    EXPECT_FALSE(a > b);
}

TEST_F(FwkTypesTest, FP16_LessThan_True)
{
    FP16 a(2.0f);
    FP16 b(5.0f);
    EXPECT_TRUE(a < b);
}

TEST_F(FwkTypesTest, FP16_LessThan_False)
{
    FP16 a(7.0f);
    FP16 b(3.0f);
    EXPECT_FALSE(a < b);
}

TEST_F(FwkTypesTest, FP16_LessThan_Equal)
{
    FP16 a(4.0f);
    FP16 b(4.0f);
    EXPECT_FALSE(a < b);
}

TEST_F(FwkTypesTest, FP16_Equality_True)
{
    FP16 a(4.5f);
    FP16 b(4.5f);
    EXPECT_TRUE(a == b);
}

TEST_F(FwkTypesTest, FP16_Equality_False)
{
    FP16 a(1.0f);
    FP16 b(2.0f);
    EXPECT_FALSE(a == b);
}

TEST_F(FwkTypesTest, FP16_Fabs_Positive)
{
    FP16 x(3.5f);
    FP16 result = FP16::fabs(x);
    EXPECT_FLOAT_EQ(result.to_float(), 3.5f);
}

TEST_F(FwkTypesTest, FP16_Fabs_Negative)
{
    FP16 x(-3.5f);
    FP16 result = FP16::fabs(x);
    EXPECT_FLOAT_EQ(result.to_float(), 3.5f);
}

TEST_F(FwkTypesTest, FP16_Fabs_Zero)
{
    FP16 x(0.0f);
    FP16 result = FP16::fabs(x);
    EXPECT_FLOAT_EQ(result.to_float(), 0.0f);
}

TEST_F(FwkTypesTest, FP16_FloatConstructor_VerySmall)
{
    FP16 x(1.0f / 256.0f);
    EXPECT_FLOAT_EQ(x.to_float(), 1.0f / 256.0f);
}

TEST_F(FwkTypesTest, FP16_HalfConstructor_FromInt)
{
    FP16 intermediate(static_cast<float>(42));
    half h = intermediate.value;
    FP16 x(h);
    EXPECT_FLOAT_EQ(x.to_float(), 42.0f);
}

TEST_F(FwkTypesTest, FP16_OperatorOutput)
{
    FP16 x(3.14f);
    std::ostringstream oss;
    oss << x;
    std::string out = oss.str();
    EXPECT_FALSE(out.empty());
}
