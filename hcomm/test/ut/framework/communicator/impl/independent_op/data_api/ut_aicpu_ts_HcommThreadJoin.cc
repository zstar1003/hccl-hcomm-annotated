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
 * @file ut_aicpu_ts_HcommThreadJoin.cc
 * @brief HcommThreadJoin接口的单元测试文件
 *
 * 本文件对HcommThreadJoin函数进行单元测试，验证其在不同输入参数下的行为是否符合预期。
 * HcommThreadJoin是HCCL（Huawei Collective Communication Library）通信子系统中
 * 用于等待线程结束的核心接口。
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
 * @class UtAicpuTsHcommThreadJoin
 * @brief HcommThreadJoin单元测试类
 *
 * 继承自Google Test的Test类，用于构建HcommThreadJoin接口的单元测试用例。
 * 该测试类提供了公共的测试基础设施，包括：
 * - SetUp：每个测试用例执行前的初始化操作，设置设备类型和线程实现
 * - TearDown：每个测试用例执行后的清理操作，验证mock对象是否被正确调用
 * - threadOnDevice：设备侧线程对象，用于模拟真实线程环境
 * - thread：线程句柄
 * - timeout：等待超时时间
 * - res：用于存储接口返回值的成员变量
 */
class UtAicpuTsHcommThreadJoin : public testing::Test {
protected:
    virtual void SetUp() override
    {
        threadOnDevice.devType_ = DevType::DEV_TYPE_950;
        threadOnDevice.pImpl_ = std::make_unique<Hccl::IAicpuTsThread>();
    }

    virtual void TearDown() override { GlobalMockObject::verify(); }

    AicpuTsThread threadOnDevice{StreamType::STREAM_TYPE_DEVICE, 0, NotifyLoadType::DEVICE_NOTIFY};
    ThreadHandle thread = reinterpret_cast<ThreadHandle>(&threadOnDevice);
    uint32_t timeout = 1;
    int32_t res{HCCL_E_RESERVED};
};

/**
 * @brief 测试HcommThreadJoin接口传入空线程句柄时的行为
 *
 * 测试场景：当thread参数为空句柄（0）时，接口应返回HCCL_E_PTR错误码。
 * 这是对接口参数校验能力的验证，确保在传入无效参数时能够正确处理。
 *
 * 预期结果：返回HCCL_E_PTR（空指针错误码）
 */
TEST_F(UtAicpuTsHcommThreadJoin, Ut_HcommThreadJoin_When_Thread_IsNull_Expect_ReturnHCCL_E_PTR)
{
    res = HcommThreadJoin(0, timeout);
    EXPECT_EQ(res, HCCL_E_PTR);
}
