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
 * @file ut_aicpu_ts_HcommChannelNotifyWait.cc
 * @brief HcommChannelNotifyWait接口的单元测试文件
 *
 * 本文件用于测试HcommChannelNotifyWait接口的功能，该接口用于在设备侧线程上执行通知等待操作。
 * 测试场景：验证在AICPU线程环境下Channel通知等待功能的返回值。
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
 * @class UtAicpuTsHcommChannelNotifyWait
 * @brief HcommChannelNotifyWait接口的测试类
 *
 * 该测试类用于验证HcommChannelNotifyWait接口的功能，包括：
 * - 测试正常情况下调用该接口的返回值
 */
class UtAicpuTsHcommChannelNotifyWait : public testing::Test {
protected:
    virtual void TearDown() override { GlobalMockObject::verify(); }

    std::vector<char> uniqueId;
    Hccl::UbTransportLiteImpl transportOnDevice{uniqueId};
    ChannelHandle channel = reinterpret_cast<ChannelHandle>(&transportOnDevice);
    uint32_t localNotifyIdx = 0;
    uint32_t timeout = 100;
    int32_t res{HCCL_E_RESERVED};
};

/**
 * @test Ut_HcommChannelNotifyWait_When_Normal_Expect_ReturnHCCL_E_NOT_SUPPORT
 * @brief 测试正常情况下调用HcommChannelNotifyWait接口
 *
 * 测试场景：使用有效的channel句柄、本地通知索引和超时时间调用HcommChannelNotifyWait
 * 预期结果：返回HCCL_E_NOT_SUPPORT，表示该接口在当前版本不支持
 */
TEST_F(UtAicpuTsHcommChannelNotifyWait, Ut_HcommChannelNotifyWait_When_Normal_Expect_ReturnHCCL_E_NOT_SUPPORT)
{
    res = HcommChannelNotifyWait(channel, localNotifyIdx, timeout);
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);
}
