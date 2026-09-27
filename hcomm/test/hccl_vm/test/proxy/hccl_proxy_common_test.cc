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

#include "hccl/hccl_common.h"
#include "hccl/hccl_types.h"
#include "hccl_proxy_common.h"

using u32 = uint32_t;

class HcclProxyCommonTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(HcclProxyCommonTest, GetDataTypeSize_Int8)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_INT8, size);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(size, 1);
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_Int16)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_INT16, size);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(size, 2);
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_Int32)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_INT32, size);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(size, 4);
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_Int64)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_INT64, size);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(size, 8);
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_Uint8)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_UINT8, size);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(size, 1);
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_Uint16)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_UINT16, size);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(size, 2);
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_Uint32)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_UINT32, size);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(size, 4);
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_Uint64)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_UINT64, size);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(size, 8);
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_Fp16)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_FP16, size);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(size, 2);
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_Fp32)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_FP32, size);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(size, 4);
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_Fp64)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_FP64, size);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(size, 8);
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_Bfp16)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_BFP16, size);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(size, 2);
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_Int128)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_INT128, size);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(size, 16);
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_Hif8)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_HIF8, size);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(size, 1);
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_Fp8E4M3)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_FP8E4M3, size);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(size, 1);
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_Fp8E5M2)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_FP8E5M2, size);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(size, 1);
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_Fp8E8M0)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_FP8E8M0, size);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(size, 1);
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_InvalidType)
{
    uint32_t size = 0;
    int result = sim::GetDataTypeSize(HcclDataType::HCCL_DATA_TYPE_RESERVED, size);
    EXPECT_EQ(result, 1); // Returns error for invalid type
}

TEST_F(HcclProxyCommonTest, GetDataTypeSize_AllValidTypes)
{
    // Test all valid data types
    std::vector<HcclDataType> validTypes = {
        HcclDataType::HCCL_DATA_TYPE_INT8,    HcclDataType::HCCL_DATA_TYPE_INT16,  HcclDataType::HCCL_DATA_TYPE_INT32,
        HcclDataType::HCCL_DATA_TYPE_FP16,    HcclDataType::HCCL_DATA_TYPE_FP32,   HcclDataType::HCCL_DATA_TYPE_INT64,
        HcclDataType::HCCL_DATA_TYPE_UINT64,  HcclDataType::HCCL_DATA_TYPE_UINT8,  HcclDataType::HCCL_DATA_TYPE_UINT16,
        HcclDataType::HCCL_DATA_TYPE_UINT32,  HcclDataType::HCCL_DATA_TYPE_FP64,   HcclDataType::HCCL_DATA_TYPE_BFP16,
        HcclDataType::HCCL_DATA_TYPE_INT128,  HcclDataType::HCCL_DATA_TYPE_HIF8,   HcclDataType::HCCL_DATA_TYPE_FP8E4M3,
        HcclDataType::HCCL_DATA_TYPE_FP8E5M2, HcclDataType::HCCL_DATA_TYPE_FP8E8M0};

    for (auto type : validTypes) {
        uint32_t size = 0;
        int result = sim::GetDataTypeSize(type, size);
        EXPECT_EQ(result, 0);
        EXPECT_GT(size, 0u);
    }
}

TEST_F(HcclProxyCommonTest, DataTypeSizeMap_ContainsExpectedEntries)
{
    EXPECT_EQ(sim::DATA_TYPE_SIZE_MAP.size(), 17u);
    EXPECT_TRUE(sim::DATA_TYPE_SIZE_MAP.count(HcclDataType::HCCL_DATA_TYPE_INT8) > 0);
    EXPECT_TRUE(sim::DATA_TYPE_SIZE_MAP.count(HcclDataType::HCCL_DATA_TYPE_FP16) > 0);
    EXPECT_TRUE(sim::DATA_TYPE_SIZE_MAP.count(HcclDataType::HCCL_DATA_TYPE_INT128) > 0);
}
