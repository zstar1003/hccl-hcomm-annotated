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
 * @file ut_aicpu_ts_HcommReadNbiOnThread.cc
 * @brief HcommReadNbiOnThread接口的单元测试文件
 *
 * 本文件用于测试HcommReadNbiOnThread接口的功能，该接口用于在指定线程上执行NBI读取操作，
 * 支持在线程上下文中进行高效的非阻塞数据传输。
 * 测试场景：验证线程上下文下源地址、目标地址的有效性检查。
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
 * @class UtAicpuTsHcommReadNbiOnThread
 * @brief HcommReadNbiOnThread接口的测试类
 *
 * 该测试类用于验证HcommReadNbiOnThread接口的功能，包括：
 * - 测试源地址为空时的错误处理
 * - 测试目标地址为空时的错误处理
 * - 测试所有参数有效时的调用
 */
class UtAicpuTsHcommReadNbiOnThread : public testing::Test {
protected:
    virtual void TearDown() override { GlobalMockObject::verify(); }

    AicpuTsThread threadOnDevice{StreamType::STREAM_TYPE_DEVICE, 0, NotifyLoadType::DEVICE_NOTIFY};
    ThreadHandle thread = reinterpret_cast<ThreadHandle>(&threadOnDevice);
    std::vector<char> uniqueId;
    Hccl::UbTransportLiteImpl transportOnDevice{uniqueId};
    ChannelHandle channel = reinterpret_cast<ChannelHandle>(&transportOnDevice);
    uint64_t tempDst[6] = {0};
    uint64_t tempSrc[6] = {1, 1, 4, 5, 1, 4};
    void* dst = reinterpret_cast<void*>(tempDst);
    void* src = reinterpret_cast<void*>(tempSrc);
    uint64_t len = sizeof(tempDst);
    int32_t res{HCCL_E_RESERVED};
};

/**
 * @test Ut_HcommReadNbiOnThread_When_Src_IsNull_Expect_ReturnIsHCCL_E_NOT_SUPPORT
 * @brief 测试线程上下文中源地址为空时的错误处理
 *
 * 测试场景：使用有效的线程和channel，但空源地址调用HcommReadNbiOnThread
 * 预期结果：返回HCCL_E_NOT_SUPPORT，表示该接口在当前版本不支持或检测到无效参数
 */
TEST_F(UtAicpuTsHcommReadNbiOnThread, Ut_HcommReadNbiOnThread_When_Src_IsNull_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    res = HcommReadNbiOnThread(thread, channel, nullptr, src, len);
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);
}

/**
 * @test Ut_HcommReadNbiOnThread_When_Dst_IsNull_Expect_ReturnIsHCCL_E_NOT_SUPPORT
 * @brief 测试线程上下文中目标地址为空时的错误处理
 *
 * 测试场景：使用有效的线程和channel，但空目标地址调用HcommReadNbiOnThread
 * 预期结果：返回HCCL_E_NOT_SUPPORT，表示该接口在当前版本不支持或检测到无效参数
 */
TEST_F(UtAicpuTsHcommReadNbiOnThread, Ut_HcommReadNbiOnThread_When_Dst_IsNull_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    res = HcommReadNbiOnThread(thread, channel, dst, nullptr, len);
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);
}

/**
 * @test Ut_HcommReadNbiOnThread_When_AllValid_Expect_ReturnIsHCCL_E_NOT_SUPPORT
 * @brief 测试线程上下文中所有参数有效时的正常调用
 *
 * 测试场景：使用有效的线程、channel、目标地址、源地址和长度调用HcommReadNbiOnThread
 * 预期结果：返回HCCL_E_NOT_SUPPORT，表示该接口在当前版本不支持
 */
TEST_F(UtAicpuTsHcommReadNbiOnThread, Ut_HcommReadNbiOnThread_When_AllValid_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    res = HcommReadNbiOnThread(thread, channel, dst, src, len);
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);
}
