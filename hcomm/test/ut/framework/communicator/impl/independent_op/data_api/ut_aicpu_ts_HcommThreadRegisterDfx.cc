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
 * @file ut_aicpu_ts_HcommThreadRegisterDfx.cc
 * @brief HcommThreadRegisterDfx接口的单元测试文件
 *
 * 本文件对HcommThreadRegisterDfx函数进行单元测试，验证其在不同输入参数下的行为是否符合预期。
 * HcommThreadRegisterDfx是HCCL（Huawei Collective Communication Library）通信子系统中
 * 用于注册线程级DFX（Debug调试与诊断）回调函数的核心接口。
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
 * @class UtAicpuTsHcommThreadRegisterDfx
 * @brief HcommThreadRegisterDfx单元测试类
 *
 * 继承自Google Test的Test类，用于构建HcommThreadRegisterDfx接口的单元测试用例。
 * 该测试类提供了公共的测试基础设施，包括：
 * - SetUp：每个测试用例执行前的初始化操作，设置设备类型和线程实现
 * - TearDown：每个测试用例执行后的清理操作，验证mock对象是否被正确调用
 * - threadOnDevice：设备侧线程对象，用于模拟真实线程环境
 * - thread：线程句柄
 * - res：用于存储接口返回值的成员变量
 */
class UtAicpuTsHcommThreadRegisterDfx : public testing::Test {
protected:
    virtual void SetUp() override
    {
        threadOnDevice.devType_ = DevType::DEV_TYPE_950;
        threadOnDevice.pImpl_ = std::make_unique<Hccl::IAicpuTsThread>();
    }

    virtual void TearDown() override { GlobalMockObject::verify(); }

    AicpuTsThread threadOnDevice{StreamType::STREAM_TYPE_DEVICE, 0, NotifyLoadType::DEVICE_NOTIFY};
    ThreadHandle thread = reinterpret_cast<ThreadHandle>(&threadOnDevice);
    int32_t res{HCCL_E_RESERVED};
};

/**
 * @brief 测试HcommThreadRegisterDfx接口正常调用时的行为
 *
 * 测试场景：传入有效的线程句柄和一个lambda回调函数，验证接口能够成功注册DFX回调。
 *
 * 预期结果：返回HCCL_SUCCESS（成功码）
 */
TEST_F(UtAicpuTsHcommThreadRegisterDfx, Ut_HcommThreadRegisterDfx_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> callback
        = [](u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle) {
              return HCCL_SUCCESS;
          };
    res = HcommThreadRegisterDfx(thread, callback);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

/**
 * @brief 测试HcommThreadRegisterDfx接口传入空线程句柄时的行为
 *
 * 测试场景：当thread参数为空句柄（0）时，接口应返回HCCL_E_PTR错误码。
 * 这是对接口参数校验能力的验证，确保在传入无效参数时能够正确处理。
 *
 * 预期结果：返回HCCL_E_PTR（空指针错误码）
 */
TEST_F(UtAicpuTsHcommThreadRegisterDfx, Ut_HcommThreadRegisterDfx_When_Thread_IsNull_Expect_ReturnHCCL_E_PTR)
{
    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> callback
        = [](u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle) {
              return HCCL_SUCCESS;
          };
    res = HcommThreadRegisterDfx(0, callback);
    EXPECT_EQ(res, HCCL_E_PTR);
}
