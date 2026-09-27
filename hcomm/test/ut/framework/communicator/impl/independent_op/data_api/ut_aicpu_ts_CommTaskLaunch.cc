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
 * @file ut_aicpu_ts_CommTaskLaunch.cc
 * @brief HCCL CommTaskLaunch接口单元测试文件
 *
 * 本文件用于测试HCCL通信库中的CommTaskLaunch接口，该接口用于启动通信任务线程。
 * CommTaskLaunch负责初始化并启动多个通信线程，以执行集合通信操作。
 *
 * 测试的接口：CommTaskLaunch
 * - 功能：启动指定数量的通信任务线程
 * - 参数：threads（线程数组指针）、threadNum（线程数量）
 */

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#define private public
#include "aicpu_ts_thread.h"
#include "aicpu_ts_thread_interface.h"
#include "dispatcher_aicpu.h"
#undef private

using namespace hccl;

/**
 * @class UtAicpuTsCommTaskLaunch
 * @brief CommTaskLaunch接口测试类
 *
 * 该测试类用于验证CommTaskLaunch接口的参数校验和基本功能。
 * 测试场景包括空指针校验和参数有效性校验。
 */
class UtAicpuTsCommTaskLaunch : public testing::Test {
protected:
    virtual void SetUp() override
    {
        MOCKER_CPP(&Hccl::IAicpuTsThread::SdmaCopy).stubs().will(returnValue(HCCL_SUCCESS));
    }

    virtual void TearDown() override { GlobalMockObject::verify(); }

    AicpuTsThread threadOnDevice{StreamType::STREAM_TYPE_DEVICE, 0, NotifyLoadType::DEVICE_NOTIFY};
    ThreadHandle thread = reinterpret_cast<ThreadHandle>(&threadOnDevice);
    int32_t res{HCCL_E_RESERVED};
};

/**
 * @test Ut_CommTaskLaunch_When_Threads_IsNull_Expect_ReturnHCCL_E_PTR
 * @brief 测试当线程数组指针为空时，CommTaskLaunch应返回参数错误
 *
 * 测试场景：传入空指针作为线程数组
 * 预期结果：返回HCCL_E_PTR错误码，表示参数错误
 */
TEST_F(UtAicpuTsCommTaskLaunch, Ut_CommTaskLaunch_When_Threads_IsNull_Expect_ReturnHCCL_E_PTR)
{
    res = CommTaskLaunch(nullptr, 1);
    EXPECT_EQ(res, HCCL_E_PTR);
}

/**
 * @test Ut_CommTaskLaunch_When_ThreadNum_IsZero_Expect_ReturnHCCL_E_PARA
 * @brief 测试当线程数量为零时，CommTaskLaunch应返回参数错误
 *
 * 测试场景：传入线程数量为0
 * 预期结果：返回HCCL_E_PARA错误码，表示参数错误
 */
TEST_F(UtAicpuTsCommTaskLaunch, Ut_CommTaskLaunch_When_ThreadNum_IsZero_Expect_ReturnHCCL_E_PARA)
{
    res = CommTaskLaunch(&thread, 0);
    EXPECT_EQ(res, HCCL_E_PARA);
}
