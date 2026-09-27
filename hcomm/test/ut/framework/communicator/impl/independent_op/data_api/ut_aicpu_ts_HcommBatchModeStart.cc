/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file ut_aicpu_ts_HcommBatchModeStart.cc
 * @brief HCCL HcommBatchModeStart接口单元测试文件
 *
 * 本文件用于测试HCCL通信库中的HcommBatchModeStart接口，该接口用于启动批量操作模式。
 * 批量操作模式允许将多个通信操作合并为一个批次执行，以提高通信效率。
 *
 * 测试的接口：HcommBatchModeStart
 * - 功能：启动批量操作模式，开始收集通信操作到批次中
 * - 参数：batchTag（批次标签，用于标识不同的批量操作）
 */

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#define private public
#include "aicpu_ts_thread.h"
#include "aicpu_ts_thread_interface.h"
#undef private

using namespace hccl;

/**
 * @class UtAicpuTsHcommBatchModeStart
 * @brief HcommBatchModeStart接口测试类
 *
 * 该测试类用于验证HcommBatchModeStart接口的基本功能。
 * 测试场景包括正常调用启动批量操作。
 */
class UtAicpuTsHcommBatchModeStart : public testing::Test {
protected:
    virtual void TearDown() override { GlobalMockObject::verify(); }

    int32_t res{HCCL_E_RESERVED};
};

/**
 * @test Ut_HcommBatchModeStart_When_Normal_Expect_ReturnHCCL_SUCCESS
 * @brief 测试正常情况下，启动批量操作应返回成功
 *
 * 测试场景：传入有效的批次标签，调用批量模式启动接口
 * 预期结果：返回HCCL_SUCCESS，表示批量操作成功启动
 */
TEST_F(UtAicpuTsHcommBatchModeStart, Ut_HcommBatchModeStart_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    const char* batchTag = "test_batch";
    res = HcommBatchModeStart(batchTag);
    EXPECT_EQ(res, HCCL_SUCCESS);
}
