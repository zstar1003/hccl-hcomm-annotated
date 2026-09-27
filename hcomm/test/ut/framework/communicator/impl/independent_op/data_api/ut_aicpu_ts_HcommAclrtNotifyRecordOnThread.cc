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
 * @file ut_aicpu_ts_HcommAclrtNotifyRecordOnThread.cc
 * @brief HCCL HcommAclrtNotifyRecordOnThread接口单元测试文件
 *
 * 本文件用于测试HCCL通信库中的HcommAclrtNotifyRecordOnThread接口，该接口用于
 * 在指定线程上记录ACL RT通知事件，用于线程间的同步机制。
 *
 * 测试的接口：HcommAclrtNotifyRecordOnThread
 * - 功能：在指定线程上记录通知事件
 * - 参数：thread（线程句柄）、notifyId（通知ID）
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
 * @class UtAicpuTsHcommAclrtNotifyRecordOnThread
 * @brief HcommAclrtNotifyRecordOnThread接口测试类
 *
 * 该测试类用于验证HcommAclrtNotifyRecordOnThread接口的参数校验和基本功能。
 * 测试场景包括正常调用和空线程句柄校验。
 */
class UtAicpuTsHcommAclrtNotifyRecordOnThread : public testing::Test {
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
    int32_t res{HCCL_E_RESERVED};
};

/**
 * @test Ut_HcommAclrtNotifyRecordOnThread_When_Normal_Expect_ReturnIsHCCL_SUCCESS
 * @brief 测试正常情况下，在线程上记录通知应返回成功
 *
 * 测试场景：传入有效的线程句柄和通知ID
 * 预期结果：返回HCCL_SUCCESS，表示通知记录成功
 */
TEST_F(
    UtAicpuTsHcommAclrtNotifyRecordOnThread, Ut_HcommAclrtNotifyRecordOnThread_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    res = HcommAclrtNotifyRecordOnThread(thread, notifyId);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

/**
 * @test Ut_HcommAclrtNotifyRecordOnThread_When_Thread_IsNull_Expect_ReturnIsHCCL_E_PTR
 * @brief 测试当线程句柄为空时，HcommAclrtNotifyRecordOnThread应返回空指针错误
 *
 * 测试场景：传入空线程句柄（0）
 * 预期结果：返回HCCL_E_PTR错误码，表示空指针错误
 */
TEST_F(
    UtAicpuTsHcommAclrtNotifyRecordOnThread,
    Ut_HcommAclrtNotifyRecordOnThread_When_Thread_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommAclrtNotifyRecordOnThread(0, notifyId);
    EXPECT_EQ(res, HCCL_E_PTR);
}
