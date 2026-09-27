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
#include <cstring>
#include <gtest/gtest.h>

#include "hccl_task_reduce_process.h"

using namespace VirtualRunTime;

class HcclTaskReduceProcessTest : public testing::Test {
protected:
    void SetUp() override
    {
        // 初始化测试数据
    }

    void TearDown() override
    {
        // 清理测试数据
    }
};

// Test: MemReduceSum with INT8 data type
TEST_F(HcclTaskReduceProcessTest, MemReduceSum_Int8)
{
    int8_t src[] = {1, 2, 3, 4};
    int8_t dst[] = {10, 20, 30, 40};
    MemReduceSum(src, dst, sizeof(src), HCCL_DATA_TYPE_INT8);
    EXPECT_EQ(dst[0], 11);
    EXPECT_EQ(dst[1], 22);
    EXPECT_EQ(dst[2], 33);
    EXPECT_EQ(dst[3], 44);
}

// Test: MemReduceSum with INT16 data type
TEST_F(HcclTaskReduceProcessTest, MemReduceSum_Int16)
{
    int16_t src[] = {100, 200, 300, 400};
    int16_t dst[] = {10, 20, 30, 40};
    MemReduceSum(src, dst, sizeof(src), HCCL_DATA_TYPE_INT16);
    EXPECT_EQ(dst[0], 110);
    EXPECT_EQ(dst[1], 220);
    EXPECT_EQ(dst[2], 330);
    EXPECT_EQ(dst[3], 440);
}

// Test: MemReduceSum with INT32 data type
TEST_F(HcclTaskReduceProcessTest, MemReduceSum_Int32)
{
    int32_t src[] = {1000, 2000, 3000, 4000};
    int32_t dst[] = {100, 200, 300, 400};
    MemReduceSum(src, dst, sizeof(src), HCCL_DATA_TYPE_INT32);
    EXPECT_EQ(dst[0], 1100);
    EXPECT_EQ(dst[1], 2200);
    EXPECT_EQ(dst[2], 3300);
    EXPECT_EQ(dst[3], 4400);
}

// Test: MemReduceSum with INT64 data type
TEST_F(HcclTaskReduceProcessTest, MemReduceSum_Int64)
{
    int64_t src[] = {10000, 20000, 30000, 40000};
    int64_t dst[] = {1000, 2000, 3000, 4000};
    MemReduceSum(src, dst, sizeof(src), HCCL_DATA_TYPE_INT64);
    EXPECT_EQ(dst[0], 11000);
    EXPECT_EQ(dst[1], 22000);
    EXPECT_EQ(dst[2], 33000);
    EXPECT_EQ(dst[3], 44000);
}

// Test: MemReduceSum with UINT8 data type
TEST_F(HcclTaskReduceProcessTest, MemReduceSum_Uint8)
{
    uint8_t src[] = {1, 2, 3, 4};
    uint8_t dst[] = {10, 20, 30, 40};
    MemReduceSum(src, dst, sizeof(src), HCCL_DATA_TYPE_UINT8);
    EXPECT_EQ(dst[0], 11);
    EXPECT_EQ(dst[1], 22);
    EXPECT_EQ(dst[2], 33);
    EXPECT_EQ(dst[3], 44);
}

// Test: MemReduceSum with UINT16 data type
TEST_F(HcclTaskReduceProcessTest, MemReduceSum_Uint16)
{
    uint16_t src[] = {100, 200, 300, 400};
    uint16_t dst[] = {10, 20, 30, 40};
    MemReduceSum(src, dst, sizeof(src), HCCL_DATA_TYPE_UINT16);
    EXPECT_EQ(dst[0], 110);
    EXPECT_EQ(dst[1], 220);
    EXPECT_EQ(dst[2], 330);
    EXPECT_EQ(dst[3], 440);
}

// Test: MemReduceSum with UINT32 data type
TEST_F(HcclTaskReduceProcessTest, MemReduceSum_Uint32)
{
    uint32_t src[] = {1000, 2000, 3000, 4000};
    uint32_t dst[] = {100, 200, 300, 400};
    MemReduceSum(src, dst, sizeof(src), HCCL_DATA_TYPE_UINT32);
    EXPECT_EQ(dst[0], 1100);
    EXPECT_EQ(dst[1], 2200);
    EXPECT_EQ(dst[2], 3300);
    EXPECT_EQ(dst[3], 4400);
}

// Test: MemReduceSum with UINT64 data type
TEST_F(HcclTaskReduceProcessTest, MemReduceSum_Uint64)
{
    uint64_t src[] = {10000, 20000, 30000, 40000};
    uint64_t dst[] = {1000, 2000, 3000, 4000};
    MemReduceSum(src, dst, sizeof(src), HCCL_DATA_TYPE_UINT64);
    EXPECT_EQ(dst[0], 11000);
    EXPECT_EQ(dst[1], 22000);
    EXPECT_EQ(dst[2], 33000);
    EXPECT_EQ(dst[3], 44000);
}

// Test: MemReduceSum with FP32 data type
TEST_F(HcclTaskReduceProcessTest, MemReduceSum_Fp32)
{
    float src[] = {1.5f, 2.5f, 3.5f, 4.5f};
    float dst[] = {10.0f, 20.0f, 30.0f, 40.0f};
    MemReduceSum(src, dst, sizeof(src), HCCL_DATA_TYPE_FP32);
    EXPECT_FLOAT_EQ(dst[0], 11.5f);
    EXPECT_FLOAT_EQ(dst[1], 22.5f);
    EXPECT_FLOAT_EQ(dst[2], 33.5f);
    EXPECT_FLOAT_EQ(dst[3], 44.5f);
}

// Test: MemReduceMin with INT8 data type
TEST_F(HcclTaskReduceProcessTest, MemReduceMin_Int8)
{
    int8_t src[] = {5, 3, 8, 1};
    int8_t dst[] = {10, 20, 30, 40};
    MemReduceMin(src, dst, sizeof(src), HCCL_DATA_TYPE_INT8);
    // MemReduceMin只比较index 1到count-1，dst[0]不会被修改
    EXPECT_EQ(dst[1], 3); // min(20, 3)
    EXPECT_EQ(dst[2], 8); // min(30, 8)
    EXPECT_EQ(dst[3], 1); // min(40, 1)
}

// Test: MemReduceMin with INT32 data type
TEST_F(HcclTaskReduceProcessTest, MemReduceMin_Int32)
{
    int32_t src[] = {100, 50, 200, 25};
    int32_t dst[] = {1000, 2000, 3000, 4000};
    MemReduceMin(src, dst, sizeof(src), HCCL_DATA_TYPE_INT32);
    EXPECT_EQ(dst[1], 50);
    EXPECT_EQ(dst[2], 200);
    EXPECT_EQ(dst[3], 25);
}

// Test: MemReduceMax with INT8 data type
TEST_F(HcclTaskReduceProcessTest, MemReduceMax_Int8)
{
    int8_t src[] = {50, 30, 80, 10};
    int8_t dst[] = {10, 20, 30, 40};
    MemReduceMax(src, dst, sizeof(src), HCCL_DATA_TYPE_INT8);
    EXPECT_EQ(dst[1], 30); // max(20, 30)
    EXPECT_EQ(dst[2], 80); // max(30, 80)
    EXPECT_EQ(dst[3], 40); // max(40, 10) -  remains 40 since 10 < 40
}

// Test: MemReduceMax with INT32 data type
TEST_F(HcclTaskReduceProcessTest, MemReduceMax_Int32)
{
    int32_t src[] = {5000, 3000, 8000, 1000};
    int32_t dst[] = {1000, 2000, 3000, 4000};
    MemReduceMax(src, dst, sizeof(src), HCCL_DATA_TYPE_INT32);
    EXPECT_EQ(dst[1], 3000);
    EXPECT_EQ(dst[2], 8000);
    EXPECT_EQ(dst[3], 4000);
}

// Test: MemReduceSum with zero length
TEST_F(HcclTaskReduceProcessTest, MemReduceSum_ZeroLength)
{
    int32_t src[] = {1, 2, 3, 4};
    int32_t dst[] = {10, 20, 30, 40};
    MemReduceSum(src, dst, 0, HCCL_DATA_TYPE_INT32);
    // 长度为0，不应该修改任何数据
    EXPECT_EQ(dst[0], 10);
    EXPECT_EQ(dst[1], 20);
    EXPECT_EQ(dst[2], 30);
    EXPECT_EQ(dst[3], 40);
}

// Test: MemReduceSum with single element
TEST_F(HcclTaskReduceProcessTest, MemReduceSum_SingleElement)
{
    int32_t src[] = {100};
    int32_t dst[] = {50};
    MemReduceSum(src, dst, sizeof(src), HCCL_DATA_TYPE_INT32);
    EXPECT_EQ(dst[0], 150);
}

// Test: MemReduceMin with single element
TEST_F(HcclTaskReduceProcessTest, MemReduceMin_SingleElement)
{
    int32_t src[] = {100};
    int32_t dst[] = {50};
    MemReduceMin(src, dst, sizeof(src), HCCL_DATA_TYPE_INT32);
    // 只有一个元素，循环不会执行，dst保持不变
    EXPECT_EQ(dst[0], 50);
}

// Test: MemReduceMax with single element
TEST_F(HcclTaskReduceProcessTest, MemReduceMax_SingleElement)
{
    int32_t src[] = {100};
    int32_t dst[] = {50};
    MemReduceMax(src, dst, sizeof(src), HCCL_DATA_TYPE_INT32);
    // 只有一个元素，循环不会执行，dst保持不变
    EXPECT_EQ(dst[0], 50);
}

// Test: MemReduceSum with negative numbers
TEST_F(HcclTaskReduceProcessTest, MemReduceSum_NegativeNumbers)
{
    int32_t src[] = {-10, -20, -30, -40};
    int32_t dst[] = {100, 200, 300, 400};
    MemReduceSum(src, dst, sizeof(src), HCCL_DATA_TYPE_INT32);
    EXPECT_EQ(dst[0], 90);
    EXPECT_EQ(dst[1], 180);
    EXPECT_EQ(dst[2], 270);
    EXPECT_EQ(dst[3], 360);
}
