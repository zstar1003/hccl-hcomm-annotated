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
 * @file ut_aicpu_ts_HcommWriteWithNotifyNbi.cc
 * @brief HcommWriteWithNotifyNbi接口的单元测试文件
 *
 * 本文件对HcommWriteWithNotifyNbi函数进行单元测试，验证其在不同输入参数下的行为是否符合预期。
 * HcommWriteWithNotifyNbi是HCCL（Huawei Collective Communication Library）通信子系统中
 * 用于非缓存直接写入数据到通信通道并触发通知事件的核心接口。
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
 * @class UtAicpuTsHcommWriteWithNotifyNbi
 * @brief HcommWriteWithNotifyNbi单元测试类
 *
 * 继承自Google Test的Test类，用于构建HcommWriteWithNotifyNbi接口的单元测试用例。
 * 该测试类提供了公共的测试基础设施，包括：
 * - TearDown：每个测试用例执行后的清理操作，验证mock对象是否被正确调用
 * - channel：通信通道句柄
 * - dst：目标缓冲区地址
 * - src：源缓冲区地址
 * - len：数据长度
 * - notifyIdx：通知索引
 * - res：用于存储接口返回值的成员变量
 */
class UtAicpuTsHcommWriteWithNotifyNbi : public testing::Test {
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
    uint32_t notifyIdx = 0;
    int32_t res{HCCL_E_RESERVED};
};

/**
 * @brief 测试HcommWriteWithNotifyNbi接口传入空源缓冲区时的行为
 *
 * 测试场景：当src参数为空指针时，接口应返回HCCL_E_NOT_SUPPORT错误码。
 * 这是对接口参数校验能力的验证，确保在传入无效参数时能够正确处理。
 *
 * 预期结果：返回HCCL_E_NOT_SUPPORT（不支持错误码）
 */
TEST_F(UtAicpuTsHcommWriteWithNotifyNbi, Ut_HcommWriteWithNotifyNbi_When_Src_IsNull_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    res = HcommWriteWithNotifyNbi(channel, nullptr, src, len, notifyIdx);
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);
}

/**
 * @brief 测试HcommWriteWithNotifyNbi接口传入空目标缓冲区时的行为
 *
 * 测试场景：当dst参数为空指针时，接口应返回HCCL_E_NOT_SUPPORT错误码。
 * 这是对接口参数校验能力的验证，确保在传入无效参数时能够正确处理。
 *
 * 预期结果：返回HCCL_E_NOT_SUPPORT（不支持错误码）
 */
TEST_F(UtAicpuTsHcommWriteWithNotifyNbi, Ut_HcommWriteWithNotifyNbi_When_Dst_IsNull_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    res = HcommWriteWithNotifyNbi(channel, dst, nullptr, len, notifyIdx);
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);
}

/**
 * @brief 测试HcommWriteWithNotifyNbi接口所有参数都有效时的行为
 *
 * 测试场景：当所有参数都有效时，接口应返回HCCL_E_NOT_SUPPORT错误码。
 * 这表明当前实现不支持该接口或该接口在当前版本中尚未实现。
 *
 * 预期结果：返回HCCL_E_NOT_SUPPORT（不支持错误码）
 */
TEST_F(UtAicpuTsHcommWriteWithNotifyNbi, Ut_HcommWriteWithNotifyNbi_When_AllValid_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    res = HcommWriteWithNotifyNbi(channel, dst, src, len, notifyIdx);
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);
}
