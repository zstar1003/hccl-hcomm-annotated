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
 * @file ut_aicpu_ts_CommTaskPrepare.cc
 * @brief HCCL CommTaskPrepare接口单元测试文件
 *
 * 本文件用于测试HCCL通信库中的CommTaskPrepare接口，该接口用于准备通信任务。
 * CommTaskPrepare负责初始化通信任务的上下文和配置信息。
 *
 * 测试的接口：CommTaskPrepare
 * - 功能：准备通信任务，初始化任务上下文
 * - 参数：key（任务密钥）、keyLen（密钥长度）
 */

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#define private public
#include "aicpu_ts_thread.h"
#include "aicpu_ts_thread_interface.h"
#include "dispatcher_aicpu.h"
#undef private

using namespace hccl;

/**
 * @class UtAicpuTsCommTaskPrepare
 * @brief CommTaskPrepare接口测试类
 *
 * 该测试类用于验证CommTaskPrepare接口的参数校验和基本功能。
 * 测试场景包括空密钥校验和有效密钥输入。
 */
class UtAicpuTsCommTaskPrepare : public testing::Test {
protected:
    virtual void SetUp() override
    {
        MOCKER_CPP(&Hccl::IAicpuTsThread::SdmaCopy).stubs().will(returnValue(HCCL_SUCCESS));
    }

    virtual void TearDown() override { GlobalMockObject::verify(); }

    int32_t res{HCCL_E_RESERVED};
};

/**
 * @test Ut_CommTaskPrepare_When_KeyIsNullAndKeyLenIsZero_Expect_ReturnHCCL_SUCCESS
 * @brief 测试当密钥为空且长度为0时，CommTaskPrepare应返回成功
 *
 * 测试场景：传入空指针和长度为0的参数，表示使用默认配置
 * 预期结果：返回HCCL_SUCCESS，表示操作成功
 */
TEST_F(UtAicpuTsCommTaskPrepare, Ut_CommTaskPrepare_When_KeyIsNullAndKeyLenIsZero_Expect_ReturnHCCL_SUCCESS)
{
    char keyBuffer[] = "test_key";
    res = CommTaskPrepare(nullptr, 0);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

/**
 * @test Ut_CommTaskPrepare_When_KeyIsValid_Expect_ReturnHCCL_SUCCESS
 * @brief 测试当提供有效密钥时，CommTaskPrepare应返回成功
 *
 * 测试场景：传入有效的密钥字符串和正确的密钥长度
 * 预期结果：返回HCCL_SUCCESS，表示任务准备成功
 */
TEST_F(UtAicpuTsCommTaskPrepare, Ut_CommTaskPrepare_When_KeyIsValid_Expect_ReturnHCCL_SUCCESS)
{
    char keyBuffer[] = "test_key";
    uint32_t keyLen = strlen(keyBuffer);
    res = CommTaskPrepare(keyBuffer, keyLen);
    EXPECT_EQ(res, HCCL_SUCCESS);
}
