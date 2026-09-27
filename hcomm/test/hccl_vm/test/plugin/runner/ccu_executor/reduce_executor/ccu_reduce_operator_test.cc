/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>

#include "ccu_reduce_operator.h"
#include "ccu_simulator_base.h"

extern bool
ReduceAddProcess(const void* srcBuf, void* dstBuf, uint64_t length, uint16_t dataType, RunnerCcuVersion version);
extern bool
ReduceMaxProcess(const void* srcBuf, void* dstBuf, uint64_t length, uint16_t dataType, RunnerCcuVersion version);
extern bool
ReduceMinProcess(const void* srcBuf, void* dstBuf, uint64_t length, uint16_t dataType, RunnerCcuVersion version);

using namespace hcomm::CcuRep;

class ReduceOperatorTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ReduceOperatorTest, ReduceAddProcessInt8)
{
    int8_t src[4] = {1, 2, -3, 4};
    int8_t dst[4] = {10, -5, 3, 0};
    ReduceAddProcess(src, dst, 4, TransMemReduceDataTypeV1::INT8_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 11);
    EXPECT_EQ(dst[1], -3);
    EXPECT_EQ(dst[2], 0);
    EXPECT_EQ(dst[3], 4);
}

TEST_F(ReduceOperatorTest, ReduceAddProcessInt16)
{
    int16_t src[3] = {100, -200, 300};
    int16_t dst[3] = {1000, 2000, -500};
    ReduceAddProcess(src, dst, 3, TransMemReduceDataTypeV1::INT16_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 1100);
    EXPECT_EQ(dst[1], 1800);
    EXPECT_EQ(dst[2], -200);
}

TEST_F(ReduceOperatorTest, ReduceAddProcessInt32)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    ReduceAddProcess(src, dst, 4, TransMemReduceDataTypeV1::INT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 11);
    EXPECT_EQ(dst[1], 22);
    EXPECT_EQ(dst[2], 33);
    EXPECT_EQ(dst[3], 44);
}

TEST_F(ReduceOperatorTest, ReduceAddProcessUint8)
{
    uint8_t src[4] = {1, 2, 3, 4};
    uint8_t dst[4] = {10, 20, 30, 40};
    ReduceAddProcess(src, dst, 4, TransMemReduceDataTypeV1::UINT8_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 11);
    EXPECT_EQ(dst[1], 22);
    EXPECT_EQ(dst[2], 33);
    EXPECT_EQ(dst[3], 44);
}

TEST_F(ReduceOperatorTest, ReduceAddProcessUint16)
{
    uint16_t src[3] = {100, 200, 300};
    uint16_t dst[3] = {1000, 2000, 3000};
    ReduceAddProcess(src, dst, 3, TransMemReduceDataTypeV1::UINT16_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 1100);
    EXPECT_EQ(dst[1], 2200);
    EXPECT_EQ(dst[2], 3300);
}

TEST_F(ReduceOperatorTest, ReduceAddProcessUint32)
{
    uint32_t src[3] = {100, 200, 300};
    uint32_t dst[3] = {1000, 2000, 3000};
    ReduceAddProcess(src, dst, 3, TransMemReduceDataTypeV1::UINT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 1100);
    EXPECT_EQ(dst[1], 2200);
    EXPECT_EQ(dst[2], 3300);
}

TEST_F(ReduceOperatorTest, ReduceAddProcessFP32)
{
    float src[4] = {1.5f, 2.5f, 3.5f, 4.5f};
    float dst[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    ReduceAddProcess(src, dst, 4, TransMemReduceDataTypeV1::FP32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_FLOAT_EQ(dst[0], 11.5f);
    EXPECT_FLOAT_EQ(dst[1], 22.5f);
    EXPECT_FLOAT_EQ(dst[2], 33.5f);
    EXPECT_FLOAT_EQ(dst[3], 44.5f);
}

TEST_F(ReduceOperatorTest, ReduceAddProcessUnsupportedInvalid)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceAddProcess(src, dst, 4, TransMemReduceDataTypeV1::INVALID_V1, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceAddProcessUnsupportedBF16)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceAddProcess(src, dst, 4, TransMemReduceDataTypeV1::BF16_V1, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceAddProcessUnsupportedFP16Normal)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceAddProcess(src, dst, 4, TransMemReduceDataTypeV1::FP16_NORMAL_V1, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceAddProcessUnsupportedFP16Sat)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceAddProcess(src, dst, 4, TransMemReduceDataTypeV1::FP16_SAT_V1, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessInt8)
{
    int8_t src[3] = {-5, 10, 3};
    int8_t dst[3] = {10, -20, 30};
    ReduceMaxProcess(src, dst, 3, TransMemReduceDataTypeV1::INT8_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 10);
    EXPECT_EQ(dst[1], 10);
    EXPECT_EQ(dst[2], 30);
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessInt16)
{
    int16_t src[3] = {100, -200, 300};
    int16_t dst[3] = {50, 2000, -500};
    ReduceMaxProcess(src, dst, 3, TransMemReduceDataTypeV1::INT16_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 100);
    EXPECT_EQ(dst[1], 2000);
    EXPECT_EQ(dst[2], 300);
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessInt32)
{
    int32_t src[4] = {5, 100, 30, 10};
    int32_t dst[4] = {10, 20, 300, 40};
    ReduceMaxProcess(src, dst, 4, TransMemReduceDataTypeV1::INT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 10);
    EXPECT_EQ(dst[1], 100);
    EXPECT_EQ(dst[2], 300);
    EXPECT_EQ(dst[3], 40);
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessUint8)
{
    uint8_t src[3] = {5, 200, 3};
    uint8_t dst[3] = {10, 20, 30};
    ReduceMaxProcess(src, dst, 3, TransMemReduceDataTypeV1::UINT8_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 10);
    EXPECT_EQ(dst[1], 200);
    EXPECT_EQ(dst[2], 30);
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessUint16)
{
    uint16_t src[3] = {100, 2000, 300};
    uint16_t dst[3] = {1000, 200, 3000};
    ReduceMaxProcess(src, dst, 3, TransMemReduceDataTypeV1::UINT16_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 1000);
    EXPECT_EQ(dst[1], 2000);
    EXPECT_EQ(dst[2], 3000);
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessUint32)
{
    uint32_t src[3] = {100, 2000, 300};
    uint32_t dst[3] = {1000, 200, 3000};
    ReduceMaxProcess(src, dst, 3, TransMemReduceDataTypeV1::UINT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 1000);
    EXPECT_EQ(dst[1], 2000);
    EXPECT_EQ(dst[2], 3000);
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessFP32)
{
    float src[4] = {1.0f, 200.0f, 3.0f, 40.0f};
    float dst[4] = {10.0f, 20.0f, 30.0f, 4.0f};
    ReduceMaxProcess(src, dst, 4, TransMemReduceDataTypeV1::FP32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_FLOAT_EQ(dst[0], 10.0f);
    EXPECT_FLOAT_EQ(dst[1], 200.0f);
    EXPECT_FLOAT_EQ(dst[2], 30.0f);
    EXPECT_FLOAT_EQ(dst[3], 40.0f);
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessUnsupportedBF16)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceMaxProcess(src, dst, 4, TransMemReduceDataTypeV1::BF16_V1, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessUnsupportedFP16Normal)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceMaxProcess(src, dst, 4, TransMemReduceDataTypeV1::FP16_NORMAL_V1, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessUnsupportedInvalid)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceMaxProcess(src, dst, 4, TransMemReduceDataTypeV1::INVALID_V1, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceMinProcessInt8)
{
    int8_t src[3] = {-5, 10, 3};
    int8_t dst[3] = {10, -20, 30};
    ReduceMinProcess(src, dst, 3, TransMemReduceDataTypeV1::INT8_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], -5);
    EXPECT_EQ(dst[1], -20);
    EXPECT_EQ(dst[2], 3);
}

TEST_F(ReduceOperatorTest, ReduceMinProcessInt16)
{
    int16_t src[3] = {100, -200, 300};
    int16_t dst[3] = {50, 2000, -500};
    ReduceMinProcess(src, dst, 3, TransMemReduceDataTypeV1::INT16_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 50);
    EXPECT_EQ(dst[1], -200);
    EXPECT_EQ(dst[2], -500);
}

TEST_F(ReduceOperatorTest, ReduceMinProcessInt32)
{
    int32_t src[4] = {5, 1, 30, 10};
    int32_t dst[4] = {10, 20, 3, 40};
    ReduceMinProcess(src, dst, 4, TransMemReduceDataTypeV1::INT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 5);
    EXPECT_EQ(dst[1], 1);
    EXPECT_EQ(dst[2], 3);
    EXPECT_EQ(dst[3], 10);
}

TEST_F(ReduceOperatorTest, ReduceMinProcessUint8)
{
    uint8_t src[3] = {5, 1, 3};
    uint8_t dst[3] = {10, 20, 30};
    ReduceMinProcess(src, dst, 3, TransMemReduceDataTypeV1::UINT8_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 5);
    EXPECT_EQ(dst[1], 1);
    EXPECT_EQ(dst[2], 3);
}

TEST_F(ReduceOperatorTest, ReduceMinProcessUint16)
{
    uint16_t src[3] = {100, 2000, 300};
    uint16_t dst[3] = {1000, 200, 3000};
    ReduceMinProcess(src, dst, 3, TransMemReduceDataTypeV1::UINT16_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 100);
    EXPECT_EQ(dst[1], 200);
    EXPECT_EQ(dst[2], 300);
}

TEST_F(ReduceOperatorTest, ReduceMinProcessUint32)
{
    uint32_t src[3] = {100, 2000, 300};
    uint32_t dst[3] = {1000, 200, 3000};
    ReduceMinProcess(src, dst, 3, TransMemReduceDataTypeV1::UINT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 100);
    EXPECT_EQ(dst[1], 200);
    EXPECT_EQ(dst[2], 300);
}

TEST_F(ReduceOperatorTest, ReduceMinProcessFP32)
{
    float src[4] = {5.0f, 1.0f, 30.0f, 10.0f};
    float dst[4] = {10.0f, 20.0f, 3.0f, 40.0f};
    ReduceMinProcess(src, dst, 4, TransMemReduceDataTypeV1::FP32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_FLOAT_EQ(dst[0], 5.0f);
    EXPECT_FLOAT_EQ(dst[1], 1.0f);
    EXPECT_FLOAT_EQ(dst[2], 3.0f);
    EXPECT_FLOAT_EQ(dst[3], 10.0f);
}

TEST_F(ReduceOperatorTest, ReduceMinProcessUnsupportedFP16Normal)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceMinProcess(src, dst, 4, TransMemReduceDataTypeV1::FP16_NORMAL_V1, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceMinProcessUnsupportedBF16)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceMinProcess(src, dst, 4, TransMemReduceDataTypeV1::BF16_V1, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceMinProcessUnsupportedInvalid)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceMinProcess(src, dst, 4, TransMemReduceDataTypeV1::INVALID_V1, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceProcessOpMax)
{
    int32_t src[4] = {5, 100, 30, 10};
    int32_t dst[4] = {10, 20, 300, 40};
    ReduceProcess(src, dst, 4, TransMemReduceDataTypeV1::INT32_V1, 8, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 10);
    EXPECT_EQ(dst[1], 100);
    EXPECT_EQ(dst[2], 300);
    EXPECT_EQ(dst[3], 40);
}

TEST_F(ReduceOperatorTest, ReduceProcessOpMin)
{
    int32_t src[4] = {5, 1, 30, 10};
    int32_t dst[4] = {10, 20, 3, 40};
    ReduceProcess(src, dst, 4, TransMemReduceDataTypeV1::INT32_V1, 9, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 5);
    EXPECT_EQ(dst[1], 1);
    EXPECT_EQ(dst[2], 3);
    EXPECT_EQ(dst[3], 10);
}

TEST_F(ReduceOperatorTest, ReduceProcessOpSum)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    ReduceProcess(src, dst, 4, TransMemReduceDataTypeV1::INT32_V1, 10, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 11);
    EXPECT_EQ(dst[1], 22);
    EXPECT_EQ(dst[2], 33);
    EXPECT_EQ(dst[3], 44);
}

TEST_F(ReduceOperatorTest, ReduceProcessOpEqualNotSupported)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceProcess(src, dst, 4, TransMemReduceDataTypeV1::INT32_V1, 11, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceProcessOpDefaultNotSupported)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceProcess(src, dst, 4, TransMemReduceDataTypeV1::INT32_V1, 99, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceAddProcessZeroLength)
{
    int32_t src[1] = {1};
    int32_t dst[1] = {10};
    ReduceAddProcess(src, dst, 0, TransMemReduceDataTypeV1::INT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 10);
}

TEST_F(ReduceOperatorTest, ReduceAddProcessSingleElement)
{
    int32_t src[1] = {5};
    int32_t dst[1] = {10};
    ReduceAddProcess(src, dst, 1, TransMemReduceDataTypeV1::INT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 15);
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessZeroLength)
{
    int32_t src[1] = {5};
    int32_t dst[1] = {10};
    ReduceMaxProcess(src, dst, 0, TransMemReduceDataTypeV1::INT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 10);
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessSingleElement)
{
    int32_t src[1] = {5};
    int32_t dst[1] = {10};
    ReduceMaxProcess(src, dst, 1, TransMemReduceDataTypeV1::INT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 10);
}

TEST_F(ReduceOperatorTest, ReduceMinProcessZeroLength)
{
    int32_t src[1] = {5};
    int32_t dst[1] = {10};
    ReduceMinProcess(src, dst, 0, TransMemReduceDataTypeV1::INT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 10);
}

TEST_F(ReduceOperatorTest, ReduceMinProcessSingleElement)
{
    int32_t src[1] = {5};
    int32_t dst[1] = {10};
    ReduceMinProcess(src, dst, 1, TransMemReduceDataTypeV1::INT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 5);
}

TEST_F(ReduceOperatorTest, ReduceAddProcessInt8Negative)
{
    int8_t src[4] = {-1, -2, -3, -4};
    int8_t dst[4] = {-10, -20, -30, -40};
    ReduceAddProcess(src, dst, 4, TransMemReduceDataTypeV1::INT8_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], -11);
    EXPECT_EQ(dst[1], -22);
    EXPECT_EQ(dst[2], -33);
    EXPECT_EQ(dst[3], -44);
}

TEST_F(ReduceOperatorTest, ReduceAddProcessFP32Negative)
{
    float src[3] = {-1.5f, -2.5f, -3.5f};
    float dst[3] = {10.0f, -20.0f, 0.0f};
    ReduceAddProcess(src, dst, 3, TransMemReduceDataTypeV1::FP32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_FLOAT_EQ(dst[0], 8.5f);
    EXPECT_FLOAT_EQ(dst[1], -22.5f);
    EXPECT_FLOAT_EQ(dst[2], -3.5f);
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessInt8AllNegative)
{
    int8_t src[3] = {-5, -10, -3};
    int8_t dst[3] = {-10, -20, -1};
    ReduceMaxProcess(src, dst, 3, TransMemReduceDataTypeV1::INT8_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], -5);
    EXPECT_EQ(dst[1], -10);
    EXPECT_EQ(dst[2], -1);
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessFP32Equal)
{
    float src[3] = {10.0f, 20.0f, 30.0f};
    float dst[3] = {10.0f, 20.0f, 30.0f};
    ReduceMaxProcess(src, dst, 3, TransMemReduceDataTypeV1::FP32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_FLOAT_EQ(dst[0], 10.0f);
    EXPECT_FLOAT_EQ(dst[1], 20.0f);
    EXPECT_FLOAT_EQ(dst[2], 30.0f);
}

TEST_F(ReduceOperatorTest, ReduceMinProcessInt8AllNegative)
{
    int8_t src[3] = {-5, -10, -3};
    int8_t dst[3] = {-10, -20, -1};
    ReduceMinProcess(src, dst, 3, TransMemReduceDataTypeV1::INT8_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], -10);
    EXPECT_EQ(dst[1], -20);
    EXPECT_EQ(dst[2], -3);
}

TEST_F(ReduceOperatorTest, ReduceMinProcessFP32Equal)
{
    float src[3] = {10.0f, 20.0f, 30.0f};
    float dst[3] = {10.0f, 20.0f, 30.0f};
    ReduceMinProcess(src, dst, 3, TransMemReduceDataTypeV1::FP32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_FLOAT_EQ(dst[0], 10.0f);
    EXPECT_FLOAT_EQ(dst[1], 20.0f);
    EXPECT_FLOAT_EQ(dst[2], 30.0f);
}

TEST_F(ReduceOperatorTest, ReduceProcessOpMaxWithFP32)
{
    float src[3] = {5.0f, 200.0f, 3.0f};
    float dst[3] = {10.0f, 20.0f, 30.0f};
    ReduceProcess(src, dst, 3, TransMemReduceDataTypeV1::FP32_V1, 8, RunnerCcuVersion::CCU_V1);
    EXPECT_FLOAT_EQ(dst[0], 10.0f);
    EXPECT_FLOAT_EQ(dst[1], 200.0f);
    EXPECT_FLOAT_EQ(dst[2], 30.0f);
}

TEST_F(ReduceOperatorTest, ReduceProcessOpMinWithFP32)
{
    float src[3] = {5.0f, 200.0f, 3.0f};
    float dst[3] = {10.0f, 20.0f, 30.0f};
    ReduceProcess(src, dst, 3, TransMemReduceDataTypeV1::FP32_V1, 9, RunnerCcuVersion::CCU_V1);
    EXPECT_FLOAT_EQ(dst[0], 5.0f);
    EXPECT_FLOAT_EQ(dst[1], 20.0f);
    EXPECT_FLOAT_EQ(dst[2], 3.0f);
}

TEST_F(ReduceOperatorTest, ReduceProcessOpSumWithFP32)
{
    float src[3] = {1.5f, 2.5f, 3.5f};
    float dst[3] = {10.0f, 20.0f, 30.0f};
    ReduceProcess(src, dst, 3, TransMemReduceDataTypeV1::FP32_V1, 10, RunnerCcuVersion::CCU_V1);
    EXPECT_FLOAT_EQ(dst[0], 11.5f);
    EXPECT_FLOAT_EQ(dst[1], 22.5f);
    EXPECT_FLOAT_EQ(dst[2], 33.5f);
}

TEST_F(ReduceOperatorTest, ReduceProcessOpMaxUnsupportedType)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceProcess(src, dst, 4, TransMemReduceDataTypeV1::BF16_V1, 8, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceProcessOpMinUnsupportedType)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceProcess(src, dst, 4, TransMemReduceDataTypeV1::FP16_NORMAL_V1, 9, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceProcessOpSumUnsupportedType)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceProcess(src, dst, 4, TransMemReduceDataTypeV1::INVALID_V1, 10, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceProcessOpZeroNotSupported)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceProcess(src, dst, 4, TransMemReduceDataTypeV1::INT32_V1, 0, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceProcessOpOneNotSupported)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceProcess(src, dst, 4, TransMemReduceDataTypeV1::INT32_V1, 1, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceProcessOpSevenNotSupported)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceProcess(src, dst, 4, TransMemReduceDataTypeV1::INT32_V1, 7, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceProcessOpTwelveNotSupported)
{
    int32_t src[4] = {1, 2, 3, 4};
    int32_t dst[4] = {10, 20, 30, 40};
    EXPECT_FALSE(ReduceProcess(src, dst, 4, TransMemReduceDataTypeV1::INT32_V1, 12, RunnerCcuVersion::CCU_V1));
}

TEST_F(ReduceOperatorTest, ReduceAddProcessUint8Overflow)
{
    uint8_t src[2] = {200, 255};
    uint8_t dst[2] = {100, 1};
    ReduceAddProcess(src, dst, 2, TransMemReduceDataTypeV1::UINT8_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], static_cast<uint8_t>(200 + 100));
    EXPECT_EQ(dst[1], static_cast<uint8_t>(255 + 1));
}

TEST_F(ReduceOperatorTest, ReduceAddProcessInt16LargeValues)
{
    int16_t src[2] = {10000, -10000};
    int16_t dst[2] = {20000, 20000};
    ReduceAddProcess(src, dst, 2, TransMemReduceDataTypeV1::INT16_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 30000);
    EXPECT_EQ(dst[1], 10000);
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessUint8SameValue)
{
    uint8_t src[3] = {10, 20, 30};
    uint8_t dst[3] = {10, 20, 30};
    ReduceMaxProcess(src, dst, 3, TransMemReduceDataTypeV1::UINT8_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 10);
    EXPECT_EQ(dst[1], 20);
    EXPECT_EQ(dst[2], 30);
}

TEST_F(ReduceOperatorTest, ReduceMinProcessUint16SameValue)
{
    uint16_t src[3] = {100, 200, 300};
    uint16_t dst[3] = {100, 200, 300};
    ReduceMinProcess(src, dst, 3, TransMemReduceDataTypeV1::UINT16_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 100);
    EXPECT_EQ(dst[1], 200);
    EXPECT_EQ(dst[2], 300);
}

TEST_F(ReduceOperatorTest, ReduceAddProcessFP32Zero)
{
    float src[3] = {0.0f, 0.0f, 0.0f};
    float dst[3] = {10.0f, -20.0f, 0.0f};
    ReduceAddProcess(src, dst, 3, TransMemReduceDataTypeV1::FP32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_FLOAT_EQ(dst[0], 10.0f);
    EXPECT_FLOAT_EQ(dst[1], -20.0f);
    EXPECT_FLOAT_EQ(dst[2], 0.0f);
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessFP32Negative)
{
    float src[3] = {-5.0f, -1.0f, -30.0f};
    float dst[3] = {-10.0f, -20.0f, -3.0f};
    ReduceMaxProcess(src, dst, 3, TransMemReduceDataTypeV1::FP32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_FLOAT_EQ(dst[0], -5.0f);
    EXPECT_FLOAT_EQ(dst[1], -1.0f);
    EXPECT_FLOAT_EQ(dst[2], -3.0f);
}

TEST_F(ReduceOperatorTest, ReduceMinProcessFP32Negative)
{
    float src[3] = {-5.0f, -1.0f, -30.0f};
    float dst[3] = {-10.0f, -20.0f, -3.0f};
    ReduceMinProcess(src, dst, 3, TransMemReduceDataTypeV1::FP32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_FLOAT_EQ(dst[0], -10.0f);
    EXPECT_FLOAT_EQ(dst[1], -20.0f);
    EXPECT_FLOAT_EQ(dst[2], -30.0f);
}

TEST_F(ReduceOperatorTest, ReduceAddProcessUint32LargeValues)
{
    uint32_t src[2] = {1000000u, 2000000u};
    uint32_t dst[2] = {3000000u, 4000000u};
    ReduceAddProcess(src, dst, 2, TransMemReduceDataTypeV1::UINT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 4000000u);
    EXPECT_EQ(dst[1], 6000000u);
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessUint32LargeValues)
{
    uint32_t src[2] = {5000000u, 1000000u};
    uint32_t dst[2] = {3000000u, 4000000u};
    ReduceMaxProcess(src, dst, 2, TransMemReduceDataTypeV1::UINT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 5000000u);
    EXPECT_EQ(dst[1], 4000000u);
}

TEST_F(ReduceOperatorTest, ReduceMinProcessUint32LargeValues)
{
    uint32_t src[2] = {5000000u, 1000000u};
    uint32_t dst[2] = {3000000u, 4000000u};
    ReduceMinProcess(src, dst, 2, TransMemReduceDataTypeV1::UINT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 3000000u);
    EXPECT_EQ(dst[1], 1000000u);
}

TEST_F(ReduceOperatorTest, ReduceAddProcessInt32Negative)
{
    int32_t src[3] = {-1, -2, -3};
    int32_t dst[3] = {10, -20, 0};
    ReduceAddProcess(src, dst, 3, TransMemReduceDataTypeV1::INT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 9);
    EXPECT_EQ(dst[1], -22);
    EXPECT_EQ(dst[2], -3);
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessInt32Negative)
{
    int32_t src[3] = {-1, -20, 30};
    int32_t dst[3] = {10, -5, -30};
    ReduceMaxProcess(src, dst, 3, TransMemReduceDataTypeV1::INT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], 10);
    EXPECT_EQ(dst[1], -5);
    EXPECT_EQ(dst[2], 30);
}

TEST_F(ReduceOperatorTest, ReduceMinProcessInt32Negative)
{
    int32_t src[3] = {-1, -20, 30};
    int32_t dst[3] = {10, -5, -30};
    ReduceMinProcess(src, dst, 3, TransMemReduceDataTypeV1::INT32_V1, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst[0], -1);
    EXPECT_EQ(dst[1], -20);
    EXPECT_EQ(dst[2], -30);
}

TEST_F(ReduceOperatorTest, ReduceAddProcessAllSupportedDataTypes)
{
    int8_t src8[2] = {1, 2};
    int8_t dst8[2] = {10, 20};
    ReduceAddProcess(src8, dst8, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::INT8_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst8[0], 11);

    int16_t src16[2] = {1, 2};
    int16_t dst16[2] = {10, 20};
    ReduceAddProcess(
        src16, dst16, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::INT16_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst16[0], 11);

    int32_t src32[2] = {1, 2};
    int32_t dst32[2] = {10, 20};
    ReduceAddProcess(
        src32, dst32, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::INT32_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst32[0], 11);

    uint8_t usrc8[2] = {1, 2};
    uint8_t udst8[2] = {10, 20};
    ReduceAddProcess(
        usrc8, udst8, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::UINT8_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(udst8[0], 11);

    uint16_t usrc16[2] = {1, 2};
    uint16_t udst16[2] = {10, 20};
    ReduceAddProcess(
        usrc16, udst16, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::UINT16_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(udst16[0], 11);

    uint32_t usrc32[2] = {1, 2};
    uint32_t udst32[2] = {10, 20};
    ReduceAddProcess(
        usrc32, udst32, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::UINT32_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(udst32[0], 11);

    float fsrc[2] = {1.0f, 2.0f};
    float fdst[2] = {10.0f, 20.0f};
    ReduceAddProcess(fsrc, fdst, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::FP32_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_FLOAT_EQ(fdst[0], 11.0f);
}

TEST_F(ReduceOperatorTest, ReduceMaxProcessAllSupportedDataTypes)
{
    int8_t src8[2] = {5, 2};
    int8_t dst8[2] = {10, 20};
    ReduceMaxProcess(src8, dst8, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::INT8_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst8[0], 10);

    int16_t src16[2] = {5, 2};
    int16_t dst16[2] = {10, 20};
    ReduceMaxProcess(
        src16, dst16, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::INT16_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst16[0], 10);

    int32_t src32[2] = {5, 2};
    int32_t dst32[2] = {10, 20};
    ReduceMaxProcess(
        src32, dst32, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::INT32_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst32[0], 10);

    uint8_t usrc8[2] = {5, 2};
    uint8_t udst8[2] = {10, 20};
    ReduceMaxProcess(
        usrc8, udst8, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::UINT8_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(udst8[0], 10);

    uint16_t usrc16[2] = {5, 2};
    uint16_t udst16[2] = {10, 20};
    ReduceMaxProcess(
        usrc16, udst16, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::UINT16_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(udst16[0], 10);

    uint32_t usrc32[2] = {5, 2};
    uint32_t udst32[2] = {10, 20};
    ReduceMaxProcess(
        usrc32, udst32, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::UINT32_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(udst32[0], 10);

    float fsrc[2] = {5.0f, 2.0f};
    float fdst[2] = {10.0f, 20.0f};
    ReduceMaxProcess(fsrc, fdst, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::FP32_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_FLOAT_EQ(fdst[0], 10.0f);
}

TEST_F(ReduceOperatorTest, ReduceMinProcessAllSupportedDataTypes)
{
    int8_t src8[2] = {5, 2};
    int8_t dst8[2] = {10, 20};
    ReduceMinProcess(src8, dst8, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::INT8_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst8[0], 5);

    int16_t src16[2] = {5, 2};
    int16_t dst16[2] = {10, 20};
    ReduceMinProcess(
        src16, dst16, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::INT16_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst16[0], 5);

    int32_t src32[2] = {5, 2};
    int32_t dst32[2] = {10, 20};
    ReduceMinProcess(
        src32, dst32, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::INT32_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dst32[0], 5);

    uint8_t usrc8[2] = {5, 2};
    uint8_t udst8[2] = {10, 20};
    ReduceMinProcess(
        usrc8, udst8, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::UINT8_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(udst8[0], 5);

    uint16_t usrc16[2] = {5, 2};
    uint16_t udst16[2] = {10, 20};
    ReduceMinProcess(
        usrc16, udst16, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::UINT16_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(udst16[0], 5);

    uint32_t usrc32[2] = {5, 2};
    uint32_t udst32[2] = {10, 20};
    ReduceMinProcess(
        usrc32, udst32, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::UINT32_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(udst32[0], 5);

    float fsrc[2] = {5.0f, 2.0f};
    float fdst[2] = {10.0f, 20.0f};
    ReduceMinProcess(fsrc, fdst, 2, static_cast<uint16_t>(TransMemReduceDataTypeV1::FP32_V1), RunnerCcuVersion::CCU_V1);
    EXPECT_FLOAT_EQ(fdst[0], 5.0f);
}

TEST_F(ReduceOperatorTest, ReduceProcessAllOpsWithInt8)
{
    int8_t src[2] = {3, 10};
    int8_t dstMax[2] = {5, 2};
    int8_t dstMin[2] = {5, 2};
    int8_t dstSum[2] = {5, 2};

    ReduceProcess(src, dstMax, 2, TransMemReduceDataTypeV1::INT8_V1, 8, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dstMax[0], 5);
    EXPECT_EQ(dstMax[1], 10);

    ReduceProcess(src, dstMin, 2, TransMemReduceDataTypeV1::INT8_V1, 9, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dstMin[0], 3);
    EXPECT_EQ(dstMin[1], 2);

    ReduceProcess(src, dstSum, 2, TransMemReduceDataTypeV1::INT8_V1, 10, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dstSum[0], 8);
    EXPECT_EQ(dstSum[1], 12);
}

TEST_F(ReduceOperatorTest, ReduceProcessAllOpsWithUint16)
{
    uint16_t src[2] = {3, 10};
    uint16_t dstMax[2] = {5, 2};
    uint16_t dstMin[2] = {5, 2};
    uint16_t dstSum[2] = {5, 2};

    ReduceProcess(src, dstMax, 2, TransMemReduceDataTypeV1::UINT16_V1, 8, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dstMax[0], 5);
    EXPECT_EQ(dstMax[1], 10);

    ReduceProcess(src, dstMin, 2, TransMemReduceDataTypeV1::UINT16_V1, 9, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dstMin[0], 3);
    EXPECT_EQ(dstMin[1], 2);

    ReduceProcess(src, dstSum, 2, TransMemReduceDataTypeV1::UINT16_V1, 10, RunnerCcuVersion::CCU_V1);
    EXPECT_EQ(dstSum[0], 8);
    EXPECT_EQ(dstSum[1], 12);
}
