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
 * @file ut_aicpu_ts_HcommChannelNotifyRecord.cc
 * @brief HCCL HcommChannelNotifyRecord接口单元测试文件
 *
 * 本文件用于测试HCCL通信库中的HcommChannelNotifyRecord接口，该接口用于在通信通道上
 * 记录通知事件，用于通道间的同步机制。
 *
 * 测试的接口：HcommChannelNotifyRecord
 * - 功能：在指定通道上记录通知事件
 * - 参数：channel（通道句柄）、notifyIdx（通知索引）
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
 * @class UtAicpuTsHcommChannelNotifyRecord
 * @brief HcommChannelNotifyRecord接口测试类
 *
 * 该测试类用于验证HcommChannelNotifyRecord接口的基本功能。
 * 测试场景验证该接口在当前平台上的支持状态。
 */
class UtAicpuTsHcommChannelNotifyRecord : public testing::Test {
protected:
    virtual void TearDown() override { GlobalMockObject::verify(); }

    std::vector<char> uniqueId;
    Hccl::UbTransportLiteImpl transportOnDevice{uniqueId};
    ChannelHandle channel = reinterpret_cast<ChannelHandle>(&transportOnDevice);
    uint32_t notifyIdx = 0;
    int32_t res{HCCL_E_RESERVED};
};

/**
 * @test Ut_HcommChannelNotifyRecord_When_Normal_Expect_ReturnHCCL_E_NOT_SUPPORT
 * @brief 测试HcommChannelNotifyRecord接口在当前平台上的支持状态
 *
 * 测试场景：传入有效的通道句柄和通知索引调用HcommChannelNotifyRecord接口
 * 预期结果：返回HCCL_E_NOT_SUPPORT错误码，表示该接口在当前平台不支持
 */
TEST_F(UtAicpuTsHcommChannelNotifyRecord, Ut_HcommChannelNotifyRecord_When_Normal_Expect_ReturnHCCL_E_NOT_SUPPORT)
{
    res = HcommChannelNotifyRecord(channel, notifyIdx);
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);
}
