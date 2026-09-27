/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_assist_v1.h"
#include "ccu_microcode_v1.h"
#include "ccu_api_exception.h"
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

namespace hcomm {
namespace CcuRep {
    namespace {

        class CcuAssistTest : public ::testing::Test {
        protected:
            void TearDown() override { GlobalMockObject::verify(); }
        };

        TEST_F(CcuAssistTest, GetLoopParam)
        {
            uint64_t loopCtxId = 5;
            uint64_t gsaOffset = 100;
            uint64_t loopIterNum = 200;
            uint64_t result = GetLoopParam(loopCtxId, gsaOffset, loopIterNum);

            EXPECT_GT(result, 0);
        }

        TEST_F(CcuAssistTest, GetLoopParam_ZeroInputs)
        {
            uint64_t result = GetLoopParam(0, 0, 0);
            EXPECT_EQ(result, 0);
        }

        TEST_F(CcuAssistTest, GetToken)
        {
            uint64_t tokenId = 10;
            uint64_t tokenValue = 0xABCD;
            uint64_t tokenValid = 1;
            uint64_t result = GetToken(tokenId, tokenValue, tokenValid);

            EXPECT_GT(result, 0);
        }

        TEST_F(CcuAssistTest, GetToken_ValidZero)
        {
            uint64_t result = GetToken(0, 0, 0);
            EXPECT_EQ(result, 0);
        }

        TEST_F(CcuAssistTest, GetCcuReduceType_SUM)
        {
            uint16_t result = GetCcuReduceType(Hccl::ReduceOp::SUM);
            EXPECT_EQ(result, CCU_REDUCE_SUM);
        }

        TEST_F(CcuAssistTest, GetCcuReduceType_MAX)
        {
            uint16_t result = GetCcuReduceType(Hccl::ReduceOp::MAX);
            EXPECT_EQ(result, CCU_REDUCE_MAX);
        }

        TEST_F(CcuAssistTest, GetCcuReduceType_MIN)
        {
            uint16_t result = GetCcuReduceType(Hccl::ReduceOp::MIN);
            EXPECT_EQ(result, CCU_REDUCE_MIN);
        }

        TEST_F(CcuAssistTest, GetCcuReduceType_Unsupported)
        {
            EXPECT_THROW(GetCcuReduceType(Hccl::ReduceOp::PROD), Hccl::CcuApiException);
        }

        TEST_F(CcuAssistTest, GetCcuDataType_FP32_SUM)
        {
            uint16_t result = GetCcuDataType(Hccl::DataType::FP32, Hccl::ReduceOp::SUM);
            EXPECT_EQ(result, 0);
        }

        TEST_F(CcuAssistTest, GetCcuDataType_FP16_SUM)
        {
            uint16_t result = GetCcuDataType(Hccl::DataType::FP16, Hccl::ReduceOp::SUM);
            EXPECT_EQ(result, 1);
        }

        TEST_F(CcuAssistTest, GetCcuDataType_INT32_SUM)
        {
            uint16_t result = GetCcuDataType(Hccl::DataType::INT32, Hccl::ReduceOp::SUM);
            EXPECT_EQ(result, 9);
        }

        TEST_F(CcuAssistTest, GetCcuDataType_FP32_MAX)
        {
            uint16_t result = GetCcuDataType(Hccl::DataType::FP32, Hccl::ReduceOp::MAX);
            EXPECT_EQ(result, 0);
        }

        TEST_F(CcuAssistTest, GetCcuDataType_INT32_MAX)
        {
            uint16_t result = GetCcuDataType(Hccl::DataType::INT32, Hccl::ReduceOp::MAX);
            EXPECT_EQ(result, 9);
        }

        TEST_F(CcuAssistTest, GetCcuDataType_UnsupportedType_SUM)
        {
            EXPECT_THROW(GetCcuDataType(Hccl::DataType::BF16_SAT, Hccl::ReduceOp::SUM), Hccl::CcuApiException);
        }

        TEST_F(CcuAssistTest, GetCcuDataType_UnsupportedType_MAX)
        {
            EXPECT_THROW(GetCcuDataType(Hccl::DataType::FP8E4M3, Hccl::ReduceOp::MAX), Hccl::CcuApiException);
        }

        TEST_F(CcuAssistTest, GetUBReduceType_SUM)
        {
            uint16_t result = GetUBReduceType(Hccl::ReduceOp::SUM);
            EXPECT_EQ(result, 10);
        }

        TEST_F(CcuAssistTest, GetUBReduceType_MAX)
        {
            uint16_t result = GetUBReduceType(Hccl::ReduceOp::MAX);
            EXPECT_EQ(result, 8);
        }

        TEST_F(CcuAssistTest, GetUBReduceType_MIN)
        {
            uint16_t result = GetUBReduceType(Hccl::ReduceOp::MIN);
            EXPECT_EQ(result, 9);
        }

        TEST_F(CcuAssistTest, GetUBReduceType_Unsupported)
        {
            EXPECT_THROW(GetUBReduceType(Hccl::ReduceOp::PROD), Hccl::CcuApiException);
        }

        TEST_F(CcuAssistTest, GetUBDataType_FP32)
        {
            uint16_t result = GetUBDataType(Hccl::DataType::FP32);
            EXPECT_EQ(result, 7);
        }

        TEST_F(CcuAssistTest, GetUBDataType_FP16)
        {
            uint16_t result = GetUBDataType(Hccl::DataType::FP16);
            EXPECT_EQ(result, 6);
        }

        TEST_F(CcuAssistTest, GetUBDataType_INT32)
        {
            uint16_t result = GetUBDataType(Hccl::DataType::INT32);
            EXPECT_EQ(result, 2);
        }

        TEST_F(CcuAssistTest, GetUBDataType_Unsupported)
        {
            EXPECT_THROW(GetUBDataType(Hccl::DataType::HIF8), Hccl::CcuApiException);
        }

    } // namespace
} // namespace CcuRep
} // namespace hcomm
