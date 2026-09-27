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
 * @file ut_aicpu_ts_HcommProfilingReportDeviceOp.cc
 * @brief HcommProfilingReportDeviceOp接口的单元测试文件
 *
 * 本文件用于测试HcommProfilingReportDeviceOp接口的功能，该接口用于上报设备侧算子性能数据，
 * 帮助开发者分析通信算子的性能瓶颈。
 * 测试场景：验证空指针检查和正常情况下的调用。
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
 * @class UtAicpuTsHcommProfilingReportDeviceOp
 * @brief HcommProfilingReportDeviceOp接口的测试类
 *
 * 该测试类用于验证HcommProfilingReportDeviceOp接口的功能，包括：
 * - 测试组名为空时的错误处理
 * - 测试正常情况下上报设备算子性能数据
 */
class UtAicpuTsHcommProfilingReportDeviceOp : public testing::Test {
protected:
    virtual void TearDown() override { GlobalMockObject::verify(); }

    HcclResult res{HCCL_E_RESERVED};
};

/**
 * @test Ut_HcommProfilingReportDeviceOp_When_GroupName_IsNull_Expect_ReturnHCCL_E_PTR
 * @brief 测试组名为空时的错误处理
 *
 * 测试场景：使用空指针作为组名调用HcommProfilingReportDeviceOp
 * 预期结果：返回HCCL_E_PTR，表示检测到空指针错误
 */
TEST_F(
    UtAicpuTsHcommProfilingReportDeviceOp,
    Ut_HcommProfilingReportDeviceOp_When_GroupName_IsNull_Expect_ReturnHCCL_E_PTR)
{
    res = HcommProfilingReportDeviceOp(nullptr);
    EXPECT_EQ(res, HCCL_E_PTR);
}

/**
 * @test Ut_HcommProfilingReportDeviceOp_When_Normal_Expect_ReturnHCCL_SUCCESS
 * @brief 测试正常情况下上报设备算子性能数据
 *
 * 测试场景：使用有效的组名调用HcommProfilingReportDeviceOp
 * 预期结果：返回HCCL_SUCCESS，表示性能数据上报成功
 */
TEST_F(UtAicpuTsHcommProfilingReportDeviceOp, Ut_HcommProfilingReportDeviceOp_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    const char* groupName = "test_group";
    res = HcommProfilingReportDeviceOp(groupName);
    EXPECT_EQ(res, HCCL_SUCCESS);
}
