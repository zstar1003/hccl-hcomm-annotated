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

class UtAicpuTsHcommSetNotifyWaitTimeOut : public UtAicpuTsBase {
protected:
    static void SetUpTestCase() { std::cout << "UtAicpuTsHcommSetNotifyWaitTimeOut tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "UtAicpuTsHcommSetNotifyWaitTimeOut tests tear down." << std::endl; }

    virtual void SetUp() override
    {
        std::cout << "A Test case in UtAicpuTsHcommSetNotifyWaitTimeOut SetUp" << std::endl;
        UtAicpuTsBase::SetUp();

        MOCKER_CPP(&Hccl::IAicpuTsThread::NotifyWait, HcclResult(Hccl::IAicpuTsThread::*)(uint32_t, uint32_t) const)
            .stubs()
            .will(returnValue(HCCL_SUCCESS));
    }

    std::vector<char> uniqueId;
    Hccl::UbTransportLiteImpl transportOnDevice{uniqueId};
    ChannelHandle channel = reinterpret_cast<ChannelHandle>(&transportOnDevice);
    uint32_t timeout = 2000;
    int32_t res{HCCL_E_RESERVED};
};

TEST_F(UtAicpuTsHcommSetNotifyWaitTimeOut, Ut_HcommSetNotifyWaitTimeOut_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    res = HcommSetNotifyWaitTimeOut(timeout);
    EXPECT_EQ(res, HCCL_SUCCESS);

    res = HcommThreadNotifyWaitOnThreadWithDefaultTimeout(thread, 0);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtAicpuTsHcommSetNotifyWaitTimeOut, Ut_HcommSetNotifyWaitTimeOut_When_Timeout_IsZero_Expect_ReturnIsHCCL_SUCCESS)
{
    timeout = 0;
    res = HcommSetNotifyWaitTimeOut(timeout);
    EXPECT_EQ(res, HCCL_SUCCESS);

    res = HcommChannelNotifyWaitOnThreadWithDefaultTimeout(thread, channel, 0);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(
    UtAicpuTsHcommSetNotifyWaitTimeOut, Ut_HcommSetNotifyWaitTimeOut_When_Timeout_IsLarge_Expect_ReturnIsHCCL_SUCCESS)
{
    timeout = 100000;
    res = HcommSetNotifyWaitTimeOut(timeout);
    EXPECT_EQ(res, HCCL_SUCCESS);

    res = HcommThreadNotifyWaitOnThreadWithDefaultTimeout(thread, 0);
    EXPECT_EQ(res, HCCL_SUCCESS);
}
