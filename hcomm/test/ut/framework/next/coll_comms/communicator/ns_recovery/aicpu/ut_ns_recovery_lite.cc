/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <memory>

#include "ns_recovery/aicpu/ns_recovery_lite.h"
#include "hdc_pub.h"

using namespace hccl;

class NsRecoveryLiteTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // 初始化 NsRecoveryLite 并用两个默认构造的 HDCommunicate 填充 hdcHandler_
    void InitWithDefaultHdc(NsRecoveryLite& lite)
    {
        auto h2d = std::make_shared<HDCommunicate>();
        auto d2h = std::make_shared<HDCommunicate>();
        lite.Init(h2d, d2h);
    }
};

TEST_F(NsRecoveryLiteTest, Ut_IsNeedCleanWhenDefaultAndToggledExpectReflect)
{
    NsRecoveryLite lite;
    // 默认不需要清理
    EXPECT_FALSE(lite.IsNeedClean());

    // 设置需要清理
    lite.SetNeedClean(true);
    EXPECT_TRUE(lite.IsNeedClean());

    // 取消需要清理
    lite.SetNeedClean(false);
    EXPECT_FALSE(lite.IsNeedClean());
}

TEST_F(NsRecoveryLiteTest, Ut_BackGroundGetCmdWhenHdcGetFailsExpectNoneReturned)
{
    NsRecoveryLite lite;
    // 使用默认 HDCommunicate（未初始化共享内存），会导致 HcclAicpuHdcHandler::GetKfcCommand 返回非 HCCL_SUCCESS
    InitWithDefaultHdc(lite);

    auto cmd = lite.BackGroundGetCmd();
    // 在失败路径中，函数会返回默认初始化的 NONE
    EXPECT_EQ(cmd, Hccl::KfcCommand::NONE);
}

TEST_F(NsRecoveryLiteTest, Ut_BackGroundSetStatusWhenCalledWithHdcExpectNoCrash)
{
    NsRecoveryLite lite;
    InitWithDefaultHdc(lite);

    // 仅验证调用不崩溃；内部会尝试向 d2hTransfer 写入状态，写失败时会记录错误但不抛出
    lite.BackGroundSetStatus(Hccl::KfcStatus::STOP_LAUNCH_DONE, Hccl::KfcErrType::NONE);
    SUCCEED();
}
