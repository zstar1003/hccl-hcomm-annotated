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
 * @file ut_aicpu_ts_HcommFlush.cc
 * @brief HcommFlush接口的单元测试文件
 *
 * 本文件用于测试HcommFlush接口的功能，该接口用于刷新通信缓冲区和确保数据一致性。
 * 测试场景：验证HcommFlush接口的返回值。
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
 * @class UtAicpuTsHcommFlush
 * @brief HcommFlush接口的测试类
 *
 * 该测试类用于验证HcommFlush接口的功能，包括：
 * - 测试正常情况下调用Flush操作
 */
class UtAicpuTsHcommFlush : public testing::Test {
protected:
    virtual void TearDown() override { GlobalMockObject::verify(); }

    int32_t res{HCCL_E_RESERVED};
};

/**
 * @test Ut_HcommFlush_When_Normal_Expect_ReturnHCCL_E_NOT_SUPPORT
 * @brief 测试正常情况下调用HcommFlush接口
 *
 * 测试场景：调用HcommFlush刷新操作
 * 预期结果：返回HCCL_E_NOT_SUPPORT，表示该接口在当前版本不支持
 */
TEST_F(UtAicpuTsHcommFlush, Ut_HcommFlush_When_Normal_Expect_ReturnHCCL_E_NOT_SUPPORT)
{
    res = HcommFlush();
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);
}
