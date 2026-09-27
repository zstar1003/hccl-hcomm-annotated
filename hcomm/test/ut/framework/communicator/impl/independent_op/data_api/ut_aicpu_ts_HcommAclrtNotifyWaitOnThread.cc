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
 * @file ut_aicpu_ts_HcommAclrtNotifyWaitOnThread.cc
 * @brief HCCL HcommAclrtNotifyWaitOnThread接口单元测试文件
 *
 * 本文件用于测试HCCL通信库中的HcommAclrtNotifyWaitOnThread接口，该接口用于
 * 在指定线程上等待ACL RT通知事件，实现线程间的同步等待机制。
 *
 * 测试的接口：HcommAclrtNotifyWaitOnThread
 * - 功能：在指定线程上等待通知事件
 * - 参数：thread（线程句柄）、notifyId（通知ID）、timeOut（超时时间）
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
 * @class UtAicpuTsHcommAclrtNotifyWaitOnThread
 * @brief HcommAclrtNotifyWaitOnThread接口测试类
 *
 * 该测试类用于验证HcommAclrtNotifyWaitOnThread接口的参数校验和基本功能。
 * 测试场景包括正常调用和空线程句柄校验。
 */
class UtAicpuTsHcommAclrtNotifyWaitOnThread : public testing::Test {
protected:
    virtual void SetUp() override
    {
        threadOnDevice.devType_ = DevType::DEV_TYPE_950;
        threadOnDevice.pImpl_ = std::make_unique<Hccl::IAicpuTsThread>();
    }

    virtual void TearDown() override { GlobalMockObject::verify(); }

    AicpuTsThread threadOnDevice{StreamType::STREAM_TYPE_DEVICE, 0, NotifyLoadType::DEVICE_NOTIFY};
    ThreadHandle thread = reinterpret_cast<ThreadHandle>(&threadOnDevice);
    uint64_t notifyId = 12345;
    uint32_t timeOut = 100;
    int32_t res{HCCL_E_RESERVED};
};

/**
 * @test Ut_HcommAclrtNotifyWaitOnThread_When_Normal_Expect_ReturnIsHCCL_SUCCESS
 * @brief 测试正常情况下，在线程上等待通知应返回成功
 *
 * 测试场景：传入有效的线程句柄、通知ID和超时时间
 * 预期结果：返回HCCL_SUCCESS，表示等待操作成功完成
 */
TEST_F(UtAicpuTsHcommAclrtNotifyWaitOnThread, Ut_HcommAclrtNotifyWaitOnThread_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    res = HcommAclrtNotifyWaitOnThread(thread, notifyId, timeOut);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

/**
 * @test Ut_HcommAclrtNotifyWaitOnThread_When_Thread_IsNull_Expect_ReturnIsHCCL_E_PTR
 * @brief 测试当线程句柄为空时，HcommAclrtNotifyWaitOnThread应返回空指针错误
 *
 * 测试场景：传入空线程句柄（0）
 * 预期结果：返回HCCL_E_PTR错误码，表示空指针错误
 */
TEST_F(
    UtAicpuTsHcommAclrtNotifyWaitOnThread, Ut_HcommAclrtNotifyWaitOnThread_When_Thread_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommAclrtNotifyWaitOnThread(0, notifyId, timeOut);
    EXPECT_EQ(res, HCCL_E_PTR);
}
