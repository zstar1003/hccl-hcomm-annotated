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
 * @file ut_aicpu_ts_HcommReleaseComm.cc
 * @brief HcommReleaseComm接口的单元测试文件
 *
 * 本文件用于测试HcommReleaseComm接口的功能，该接口用于释放Hcomm通信资源，
 * 包括关闭通道、释放内存等，确保通信资源的正确清理。
 * 测试场景：验证通信ID为空时的参数检查。
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
 * @class UtAicpuTsHcommReleaseComm
 * @brief HcommReleaseComm接口的测试类
 *
 * 该测试类用于验证HcommReleaseComm接口的功能，包括：
 * - 测试通信ID为空时的错误处理
 */
class UtAicpuTsHcommReleaseComm : public testing::Test {
protected:
    virtual void TearDown() override { GlobalMockObject::verify(); }

    int32_t res{HCCL_E_RESERVED};
};

/**
 * @test Ut_HcommReleaseComm_When_CommId_IsNull_Expect_ReturnHCCL_E_PTR
 * @brief 测试通信ID为空时的错误处理
 *
 * 测试场景：使用空指针作为通信ID调用HcommReleaseComm
 * 预期结果：返回HCCL_E_PTR，表示检测到无效参数
 */
TEST_F(UtAicpuTsHcommReleaseComm, Ut_HcommReleaseComm_When_CommId_IsNull_Expect_ReturnHCCL_E_PTR)
{
    res = HcommReleaseComm(nullptr);
    EXPECT_EQ(res, HCCL_E_PTR);
}
