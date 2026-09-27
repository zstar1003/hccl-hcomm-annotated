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
 * @file ut_aicpu_ts_HcommProfilingReportKernelStartTask.cc
 * @brief HcommProfilingReportKernelStartTask接口的单元测试文件
 *
 * 本文件用于测试HcommProfilingReportKernelStartTask接口的功能，该接口用于上报内核任务开始时的性能数据，
 * 与HcommProfilingReportKernelEndTask配合使用，用于计算内核执行时长。
 * 测试场景：验证线程句柄和组名的空指针检查。
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
 * @class UtAicpuTsHcommProfilingReportKernelStartTask
 * @brief HcommProfilingReportKernelStartTask接口的测试类
 *
 * 该测试类用于验证HcommProfilingReportKernelStartTask接口的功能，包括：
 * - 测试组名为空时的错误处理
 * - 测试线程句柄为空时的错误处理
 */
class UtAicpuTsHcommProfilingReportKernelStartTask : public testing::Test {
protected:
    virtual void SetUp() override
    {
        threadOnDevice.devType_ = DevType::DEV_TYPE_950;
        threadOnDevice.pImpl_ = std::make_unique<Hccl::IAicpuTsThread>();
    }

    virtual void TearDown() override { GlobalMockObject::verify(); }

    AicpuTsThread threadOnDevice{StreamType::STREAM_TYPE_DEVICE, 0, NotifyLoadType::DEVICE_NOTIFY};
    ThreadHandle thread = reinterpret_cast<ThreadHandle>(&threadOnDevice);
    HcclResult res{HCCL_E_RESERVED};
};

/**
 * @test Ut_HcommProfilingReportKernelStartTask_When_GroupName_IsNull_Expect_ReturnHCCL_E_PTR
 * @brief 测试组名为空时的错误处理
 *
 * 测试场景：使用有效的线程句柄但空组名调用HcommProfilingReportKernelStartTask
 * 预期结果：返回HCCL_E_PTR，表示检测到组名为空指针
 */
TEST_F(
    UtAicpuTsHcommProfilingReportKernelStartTask,
    Ut_HcommProfilingReportKernelStartTask_When_GroupName_IsNull_Expect_ReturnHCCL_E_PTR)
{
    uint64_t threadHandle = reinterpret_cast<uint64_t>(thread);
    res = HcommProfilingReportKernelStartTask(threadHandle, nullptr);
    EXPECT_EQ(res, HCCL_E_PTR);
}

/**
 * @test Ut_HcommProfilingReportKernelStartTask_When_ThreadPtr_IsNull_Expect_ReturnHCCL_E_PTR
 * @brief 测试线程句柄为空时的错误处理
 *
 * 测试场景：使用空线程句柄（0）和有效组名调用HcommProfilingReportKernelStartTask
 * 预期结果：返回HCCL_E_PTR，表示检测到线程句柄为空指针
 */
TEST_F(
    UtAicpuTsHcommProfilingReportKernelStartTask,
    Ut_HcommProfilingReportKernelStartTask_When_ThreadPtr_IsNull_Expect_ReturnHCCL_E_PTR)
{
    const char* groupName = "test_group";
    res = HcommProfilingReportKernelStartTask(0, groupName);
    EXPECT_EQ(res, HCCL_E_PTR);
}
