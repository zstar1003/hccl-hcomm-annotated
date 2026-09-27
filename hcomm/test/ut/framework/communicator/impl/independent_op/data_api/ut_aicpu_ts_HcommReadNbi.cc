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
 * @file ut_aicpu_ts_HcommReadNbi.cc
 * @brief HcommReadNbi接口的单元测试文件
 *
 * 本文件用于测试HcommReadNbi接口的功能，该接口用于执行NBI（Non-Blocking Interface）读取操作，
 * 实现高效的数据传输，适用于大规模集合通信场景。
 * 测试场景：验证源地址、目标地址和参数的有效性检查。
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
 * @class UtAicpuTsHcommReadNbi
 * @brief HcommReadNbi接口的测试类
 *
 * 该测试类用于验证HcommReadNbi接口的功能，包括：
 * - 测试源地址为空时的错误处理
 * - 测试目标地址为空时的错误处理
 * - 测试所有参数有效时的调用
 */
class UtAicpuTsHcommReadNbi : public testing::Test {
protected:
    virtual void TearDown() override { GlobalMockObject::verify(); }

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
 * @test Ut_HcommReadNbi_When_Src_IsNull_Expect_ReturnIsHCCL_E_NOT_SUPPORT
 * @brief 测试源地址为空时的错误处理
 *
 * 测试场景：使用空源地址调用HcommReadNbi
 * 预期结果：返回HCCL_E_NOT_SUPPORT，表示该接口在当前版本不支持或检测到无效参数
 */
TEST_F(UtAicpuTsHcommReadNbi, Ut_HcommReadNbi_When_Src_IsNull_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    res = HcommReadNbi(channel, dst, nullptr, len);
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);
}

/**
 * @test Ut_HcommReadNbi_When_Dst_IsNull_Expect_ReturnIsHCCL_E_NOT_SUPPORT
 * @brief 测试目标地址为空时的错误处理
 *
 * 测试场景：使用空目标地址调用HcommReadNbi
 * 预期结果：返回HCCL_E_NOT_SUPPORT，表示该接口在当前版本不支持或检测到无效参数
 */
TEST_F(UtAicpuTsHcommReadNbi, Ut_HcommReadNbi_When_Dst_IsNull_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    res = HcommReadNbi(channel, dst, nullptr, len);
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);
}

/**
 * @test Ut_HcommReadNbi_When_AllValid_Expect_ReturnIsHCCL_E_NOT_SUPPORT
 * @brief 测试所有参数有效时的正常调用
 *
 * 测试场景：使用有效的channel、目标地址、源地址和长度调用HcommReadNbi
 * 预期结果：返回HCCL_E_NOT_SUPPORT，表示该接口在当前版本不支持
 */
TEST_F(UtAicpuTsHcommReadNbi, Ut_HcommReadNbi_When_AllValid_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    res = HcommReadNbi(channel, dst, src, len);
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);
}
