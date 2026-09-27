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
 * @file ut_aicpu_ts_CommFence.cc
 * @brief HCCL CommFence接口单元测试文件
 *
 * 本文件用于测试HCCL通信库中的CommFence接口，该接口用于在通信线程之间插入栅障，
 * 确保所有线程完成当前操作后再继续执行后续操作。
 *
 * 测试的接口：CommFence
 * - 功能：在指定通道上插入通信栅障
 * - 参数：thread（线程句柄）、channel（通道句柄）
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
 * @class UtAicpuTsCommFence
 * @brief CommFence接口测试类
 *
 * 该测试类用于验证CommFence接口的参数校验和基本功能。
 * 测试场景包括空指针校验等边界条件。
 */
class UtAicpuTsCommFence : public testing::Test {
protected:
    virtual void SetUp() override
    {
        threadOnDevice.devType_ = DevType::DEV_TYPE_950;
        threadOnDevice.pImpl_ = std::make_unique<Hccl::IAicpuTsThread>();
    }

    virtual void TearDown() override { GlobalMockObject::verify(); }

    AicpuTsThread threadOnDevice{StreamType::STREAM_TYPE_DEVICE, 0, NotifyLoadType::DEVICE_NOTIFY};
    ThreadHandle thread = reinterpret_cast<ThreadHandle>(&threadOnDevice);
    std::vector<char> uniqueId;
    Hccl::UbTransportLiteImpl transportOnDevice{uniqueId};
    ChannelHandle channel = reinterpret_cast<ChannelHandle>(&transportOnDevice);
    HcclResult res{HCCL_E_RESERVED};
};

/**
 * @test Ut_CommFence_When_Thread_IsNull_Expect_ReturnHCCL_E_PTR
 * @brief 测试当线程句柄为空时，CommFence应返回空指针错误
 *
 * 测试场景：传入空线程句柄（0）给CommFence接口
 * 预期结果：返回HCCL_E_PTR错误码，表示空指针错误
 */
TEST_F(UtAicpuTsCommFence, Ut_CommFence_When_Thread_IsNull_Expect_ReturnHCCL_E_PTR)
{
    res = CommFence(0, channel);
    EXPECT_EQ(res, HCCL_E_PTR);
}
