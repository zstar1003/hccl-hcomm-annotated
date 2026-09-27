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
 * @file ut_aicpu_ts_HcommChannelRegisterDfx.cc
 * @brief HcommChannelRegisterDfx接口的单元测试文件
 *
 * 本文件用于测试HcommChannelRegisterDfx接口的功能，该接口用于向Channel注册DFX（调试和故障诊断）回调函数。
 * 测试场景：验证Channel句柄有效性和空指针检查。
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
 * @class UtAicpuTsHcommChannelRegisterDfx
 * @brief HcommChannelRegisterDfx接口的测试类
 *
 * 该测试类用于验证HcommChannelRegisterDfx接口的功能，包括：
 * - 测试正常情况下注册DFX回调函数
 * - 测试Channel句柄为空的错误处理
 */
class UtAicpuTsHcommChannelRegisterDfx : public testing::Test {
protected:
    virtual void TearDown() override { GlobalMockObject::verify(); }

    std::vector<char> uniqueId;
    Hccl::UbTransportLiteImpl transportOnDevice{uniqueId};
    ChannelHandle channel = reinterpret_cast<ChannelHandle>(&transportOnDevice);
    int32_t res{HCCL_E_RESERVED};
};

/**
 * @test Ut_HcommChannelRegisterDfx_When_Normal_Expect_ReturnHCCL_SUCCESS
 * @brief 测试正常情况下注册DFX回调函数
 *
 * 测试场景：使用有效的channel句柄和回调函数调用HcommChannelRegisterDfx
 * 预期结果：返回HCCL_SUCCESS，表示DFX回调函数注册成功
 */
TEST_F(UtAicpuTsHcommChannelRegisterDfx, Ut_HcommChannelRegisterDfx_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> callback
        = [](u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle) {
              return HCCL_SUCCESS;
          };
    res = HcommChannelRegisterDfx(channel, callback);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

/**
 * @test Ut_HcommChannelRegisterDfx_When_Channel_IsNull_Expect_ReturnHCCL_E_PTR
 * @brief 测试Channel句柄为空时的错误处理
 *
 * 测试场景：使用空channel句柄（0）调用HcommChannelRegisterDfx
 * 预期结果：返回HCCL_E_PTR，表示检测到空指针错误
 */
TEST_F(UtAicpuTsHcommChannelRegisterDfx, Ut_HcommChannelRegisterDfx_When_Channel_IsNull_Expect_ReturnHCCL_E_PTR)
{
    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> callback
        = [](u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle) {
              return HCCL_SUCCESS;
          };
    res = HcommChannelRegisterDfx(0, callback);
    EXPECT_EQ(res, HCCL_E_PTR);
}
