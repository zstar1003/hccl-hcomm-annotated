/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ut_aicpu_ts_base.h"
#include "ub_transport_lite_impl.h"

using namespace hccl;

class UtAicpuTsHcommChannelNotifyWaitOnThread : public UtAicpuTsBase {
protected:
    static void SetUpTestCase() { std::cout << "UtAicpuTsHcommChannelNotifyWaitOnThread tests set up." << std::endl; }

    static void TearDownTestCase()
    {
        std::cout << "UtAicpuTsHcommChannelNotifyWaitOnThread tests tear down." << std::endl;
    }

    virtual void SetUp() override
    {
        std::cout << "A Test case in UtAicpuTsHcommChannelNotifyWaitOnThread SetUp" << std::endl;
        UtAicpuTsBase::SetUp();
    }

    virtual void TearDown() override
    {
        UtAicpuTsBase::TearDown();
        std::cout << "A Test case in UtAicpuTsHcommChannelNotifyWaitOnThread TearDown" << std::endl;
    }

    std::vector<char> uniqueId;
    Hccl::UbTransportLiteImpl transportOnDevice{uniqueId};
    ChannelHandle channel = reinterpret_cast<ChannelHandle>(&transportOnDevice);
    uint32_t notifyIdx = 0;
    uint32_t timeout = 1800;
    int32_t res{HCCL_E_RESERVED};
};

TEST_F(
    UtAicpuTsHcommChannelNotifyWaitOnThread, Ut_HcommChannelNotifyWaitOnThread_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    res = HcommChannelNotifyWaitOnThread(thread, channel, notifyIdx, timeout);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(
    UtAicpuTsHcommChannelNotifyWaitOnThread,
    Ut_HcommChannelNotifyWaitOnThread_When_Thread_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommChannelNotifyWaitOnThread(0, channel, notifyIdx, timeout);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(
    UtAicpuTsHcommChannelNotifyWaitOnThread,
    Ut_HcommChannelNotifyWaitOnThread_When_Channel_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommChannelNotifyWaitOnThread(thread, 0, notifyIdx, timeout);
    EXPECT_EQ(res, HCCL_E_PTR);
}
