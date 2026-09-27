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
 * @file ut_aicpu_ts_HcommSetLaunchMode.cc
 * @brief HcommSetLaunchMode接口的单元测试文件
 *
 * 本文件对HcommSetLaunchMode函数进行单元测试，验证其在不同输入参数下的行为是否符合预期。
 * HcommSetLaunchMode是HCCL（Huawei Collective Communication Library）通信子系统中
 * 用于设置通信启动模式的核心接口。
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
 * @class UtAicpuTsHcommSetLaunchMode
 * @brief HcommSetLaunchMode单元测试类
 *
 * 继承自Google Test的Test类，用于构建HcommSetLaunchMode接口的单元测试用例。
 * 该测试类提供了公共的测试基础设施，包括：
 * - TearDown：每个测试用例执行后的清理操作，验证mock对象是否被正确调用
 * - res：用于存储接口返回值的成员变量
 */
class UtAicpuTsHcommSetLaunchMode : public testing::Test {
protected:
    virtual void TearDown() override { GlobalMockObject::verify(); }

    int32_t res{HCCL_E_RESERVED};
};

/**
 * @brief 测试HcommSetLaunchMode接口正常调用时的行为
 *
 * 测试场景：传入有效的launchTag（启动标签）和HCOMM_LAUNCH_MODE_BATCH启动模式，
 * 验证接口能够成功设置启动模式。
 *
 * 预期结果：返回HCCL_SUCCESS（成功码）
 */
TEST_F(UtAicpuTsHcommSetLaunchMode, Ut_HcommSetLaunchMode_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    const char* launchTag = "test_launch";
    res = HcommSetLaunchMode(launchTag, HcommLaunchMode::HCOMM_LAUNCH_MODE_BATCH);
    EXPECT_EQ(res, HCCL_SUCCESS);
}
