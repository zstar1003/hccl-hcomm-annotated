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
 * @file ut_aicpu_ts_CommWriteReduceWithNotify.cc
 * @brief HCCL CommWriteReduceWithNotify接口单元测试文件
 *
 * 本文件用于测试HCCL通信库中的CommWriteReduceWithNotify接口，该接口用于执行带通知的
 * 写如减操作（Write-Reduce），在多线程通信场景中实现数据聚合和同步。
 *
 * 测试的接口：CommWriteReduceWithNotify
 * - 功能：执行带通知的写如减操作，结合远程通知进行同步
 * - 参数：thread（线程句柄）、channel（通道句柄）、dst（目标地址）、src（源地址）、
 *        count（数据元素数量）、dataType（数据类型）、reduceOp（归约操作类型）、
 *        remoteNotifyIdx（远程通知索引）
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
 * @class UtAicpuTsCommWriteReduceWithNotify
 * @brief CommWriteReduceWithNotify接口测试类
 *
 * 该测试类用于验证CommWriteReduceWithNotify接口的参数校验和基本功能。
 * 测试场景包括源地址空指针、目的地址空指针和不支持的归约操作类型校验。
 */
class UtAicpuTsCommWriteReduceWithNotify : public testing::Test {
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
    uint64_t count = 6;
    HcommDataType dataType = HcommDataType::HCOMM_DATA_TYPE_FP32;
    HcommReduceOp reduceOp = HcommReduceOp::HCOMM_REDUCE_SUM;
    uint32_t remoteNotifyIdx = 0;
    int32_t res{HCCL_E_RESERVED};
};

/**
 * @test Ut_CommWriteReduceWithNotify_When_Src_IsNull_Expect_ReturnHCCL_E_PTR
 * @brief 测试当源地址为空时，CommWriteReduceWithNotify应返回参数错误
 *
 * 测试场景：传入空指针作为源地址
 * 预期结果：返回HCCL_E_PTR错误码，表示参数错误
 */
TEST_F(UtAicpuTsCommWriteReduceWithNotify, Ut_CommWriteReduceWithNotify_When_Src_IsNull_Expect_ReturnHCCL_E_PTR)
{
    res = CommWriteReduceWithNotify(thread, channel, nullptr, src, count, dataType, reduceOp, remoteNotifyIdx);
    EXPECT_EQ(res, HCCL_E_PTR);
}

/**
 * @test Ut_CommWriteReduceWithNotify_When_Dst_IsNull_Expect_ReturnHCCL_E_PTR
 * @brief 测试当目标地址为空时，CommWriteReduceWithNotify应返回参数错误
 *
 * 测试场景：传入空指针作为目标地址
 * 预期结果：返回HCCL_E_PTR错误码，表示参数错误
 */
TEST_F(UtAicpuTsCommWriteReduceWithNotify, Ut_CommWriteReduceWithNotify_When_Dst_IsNull_Expect_ReturnHCCL_E_PTR)
{
    res = CommWriteReduceWithNotify(thread, channel, dst, nullptr, count, dataType, reduceOp, remoteNotifyIdx);
    EXPECT_EQ(res, HCCL_E_PTR);
}

/**
 * @test Ut_CommWriteReduceWithNotify_When_UnsupportedReduce_Expect_ReturnHCCL_E_PARA
 * @brief 测试当使用不支持的归约操作时，CommWriteReduceWithNotify应返回参数错误
 *
 * 测试场景：使用HCOMM_REDUCE_PROD（乘积）操作，该操作可能不被支持
 * 预期结果：返回HCCL_E_PARA错误码，表示参数错误
 */
TEST_F(UtAicpuTsCommWriteReduceWithNotify, Ut_CommWriteReduceWithNotify_When_UnsupportedReduce_Expect_ReturnHCCL_E_PARA)
{
    HcommReduceOp unsupportedOp = HcommReduceOp::HCOMM_REDUCE_PROD;
    res = CommWriteReduceWithNotify(thread, channel, dst, src, count, dataType, unsupportedOp, remoteNotifyIdx);
    EXPECT_EQ(res, HCCL_E_PARA);
}
