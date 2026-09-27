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
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/mokc.h>

#include "ccu_assist.h"
#include "log.h"
#include "orion_adapter_rts.h"
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_microcode.h"
using namespace Hccl;
using namespace CcuRep;

class CcuAssistTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuAssistTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CcuAssistTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CcuAssistTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuAssistTest TearDown" << std::endl;
    }
};

TEST_F(CcuAssistTest, Test)
{
    EXPECT_EQ(GetLoopParam(0, 0, 2), 2);
    EXPECT_EQ(GetParallelParam(15, 0, 1), 540434154307715072);
    EXPECT_EQ(GetOffsetParam(0, 8, 1), 8193);
    EXPECT_EQ(GetToken(0, 0, 1), 4503599627370496);
    EXPECT_EQ(GetExpansionParam(1), 18014398509481984);
    EXPECT_EQ(GetCcuReduceType(ReduceOp::SUM), 0);
    EXPECT_EQ(GetCcuDataType(DataType::INT32, ReduceOp::SUM), 9);
    EXPECT_EQ(GetUBReduceType(ReduceOp::SUM), 10);
    EXPECT_EQ(GetUBDataType(DataType::FP32), 7);
    EXPECT_EQ(GetReduceExpansionNum(ReduceOp::SUM, DataType::HIF8, DataType::FP32), 4);
    EXPECT_EQ(GetReduceTypeStr(DataType::FP32, ReduceOp::SUM), "fp32_sum");
}

TEST_F(CcuAssistTest, Test1)
{
    EXPECT_THROW(GetCcuDataType(DataType::BF16_SAT, ReduceOp::SUM), CcuApiException);
    EXPECT_THROW(GetCcuDataType(DataType::BF16_SAT, ReduceOp::MIN), CcuApiException);
    EXPECT_THROW(GetUBReduceType(ReduceOp::PROD), CcuApiException);
    EXPECT_THROW(GetUBDataType(DataType::BF16_SAT), CcuApiException);
    EXPECT_THROW(GetCcuReduceType(ReduceOp::PROD), CcuApiException);
}
