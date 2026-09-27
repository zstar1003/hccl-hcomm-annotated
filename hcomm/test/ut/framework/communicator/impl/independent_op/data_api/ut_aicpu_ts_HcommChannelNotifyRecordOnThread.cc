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

class UtAicpuTsHcommChannelNotifyRecordOnThread : public UtAicpuTsBase {
protected:
    static void SetUpTestCase() { std::cout << "UtAicpuTsHcommChannelNotifyRecordOnThread tests set up." << std::endl; }

    static void TearDownTestCase()
    {
        std::cout << "UtAicpuTsHcommChannelNotifyRecordOnThread tests tear down." << std::endl;
    }

    virtual void SetUp() override
    {
        std::cout << "A Test case in UtAicpuTsHcommChannelNotifyRecordOnThread SetUp" << std::endl;
        UtAicpuTsBase::SetUp();
    }

    virtual void TearDown() override
    {
        UtAicpuTsBase::TearDown();
        std::cout << "A Test case in UtAicpuTsHcommChannelNotifyRecordOnThread TearDown" << std::endl;
    }

    std::vector<char> uniqueId;
    Hccl::UbTransportLiteImpl transportOnDevice{uniqueId};
    ChannelHandle channel = reinterpret_cast<ChannelHandle>(&transportOnDevice);
    uint32_t notifyIdx = 0;
    int32_t res{HCCL_E_RESERVED};
};

TEST_F(
    UtAicpuTsHcommChannelNotifyRecordOnThread,
    Ut_HcommChannelNotifyRecordOnThread_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    res = HcommChannelNotifyRecordOnThread(thread, channel, notifyIdx);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(
    UtAicpuTsHcommChannelNotifyRecordOnThread,
    Ut_HcommChannelNotifyRecordOnThread_When_Thread_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommChannelNotifyRecordOnThread(0, channel, notifyIdx);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(
    UtAicpuTsHcommChannelNotifyRecordOnThread,
    Ut_HcommChannelNotifyRecordOnThread_When_Channel_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommChannelNotifyRecordOnThread(thread, 0, notifyIdx);
    EXPECT_EQ(res, HCCL_E_PTR);
}
