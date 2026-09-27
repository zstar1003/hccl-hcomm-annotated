/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <gtest/gtest.h>

#include "hccl/dtype_common.h"

namespace ops_hccl {
bool RunIndependentOpExpansion(DevType deviceType);
bool IsAiCpuMode(DevType deviceType, u32 rankSize);
} // namespace ops_hccl

class ScatterStubLevel2Test : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { unsetenv("HCCL_OP_EXPANSION_MODE"); }
};

TEST_F(ScatterStubLevel2Test, RunIndependentOpExpansion_ReturnsTrue)
{
    EXPECT_TRUE(ops_hccl::RunIndependentOpExpansion(DevType::DEV_TYPE_910B));
}

TEST_F(ScatterStubLevel2Test, RunIndependentOpExpansion_MultipleDeviceTypes)
{
    EXPECT_TRUE(ops_hccl::RunIndependentOpExpansion(DevType::DEV_TYPE_910B));
    EXPECT_TRUE(ops_hccl::RunIndependentOpExpansion(DevType::DEV_TYPE_910_93));
    EXPECT_TRUE(ops_hccl::RunIndependentOpExpansion(DevType::DEV_TYPE_310P3));
    EXPECT_TRUE(ops_hccl::RunIndependentOpExpansion(static_cast<DevType>(999)));
}

TEST_F(ScatterStubLevel2Test, IsAiCpuMode_NoEnvVar)
{
    unsetenv("HCCL_OP_EXPANSION_MODE");
    EXPECT_FALSE(ops_hccl::IsAiCpuMode(DevType::DEV_TYPE_910B, 8));
}

TEST_F(ScatterStubLevel2Test, IsAiCpuMode_EnvVarAiCpu)
{
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    EXPECT_TRUE(ops_hccl::IsAiCpuMode(DevType::DEV_TYPE_910B, 8));
}

TEST_F(ScatterStubLevel2Test, IsAiCpuMode_EnvVarOther)
{
    setenv("HCCL_OP_EXPANSION_MODE", "OTHER", 1);
    EXPECT_FALSE(ops_hccl::IsAiCpuMode(DevType::DEV_TYPE_910B, 8));
}

TEST_F(ScatterStubLevel2Test, IsAiCpuMode_EnvVarEmpty)
{
    setenv("HCCL_OP_EXPANSION_MODE", "", 1);
    EXPECT_FALSE(ops_hccl::IsAiCpuMode(DevType::DEV_TYPE_910B, 8));
}

TEST_F(ScatterStubLevel2Test, IsAiCpuMode_VariousRankSizes)
{
    unsetenv("HCCL_OP_EXPANSION_MODE");
    for (u32 rankSize = 1; rankSize <= 32; rankSize++) {
        EXPECT_FALSE(ops_hccl::IsAiCpuMode(DevType::DEV_TYPE_910B, rankSize));
    }
}

TEST_F(ScatterStubLevel2Test, IsAiCpuMode_ConsistentBehavior)
{
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    EXPECT_TRUE(ops_hccl::IsAiCpuMode(DevType::DEV_TYPE_910B, 1));
    EXPECT_TRUE(ops_hccl::IsAiCpuMode(DevType::DEV_TYPE_910B, 2));
    EXPECT_TRUE(ops_hccl::IsAiCpuMode(DevType::DEV_TYPE_910B, 8));
    EXPECT_TRUE(ops_hccl::IsAiCpuMode(DevType::DEV_TYPE_910B, 16));
    EXPECT_TRUE(ops_hccl::IsAiCpuMode(DevType::DEV_TYPE_910B, 1024));
}
