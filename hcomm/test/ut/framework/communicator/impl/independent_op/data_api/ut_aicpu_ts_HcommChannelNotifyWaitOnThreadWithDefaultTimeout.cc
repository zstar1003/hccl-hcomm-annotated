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

class UtAicpuTsHcommChannelNotifyWaitOnThreadWithDefaultTimeout : public UtAicpuTsBase {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UtAicpuTsHcommChannelNotifyWaitOnThreadWithDefaultTimeout tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UtAicpuTsHcommChannelNotifyWaitOnThreadWithDefaultTimeout tests tear down." << std::endl;
    }

    virtual void SetUp() override
    {
        std::cout << "A Test case in UtAicpuTsHcommChannelNotifyWaitOnThreadWithDefaultTimeout SetUp" << std::endl;
        UtAicpuTsBase::SetUp();

        MOCKER_CPP(&Hccl::IAicpuTsThread::NotifyWait, HcclResult(Hccl::IAicpuTsThread::*)(uint32_t, uint32_t) const)
            .stubs()
            .will(returnValue(HCCL_SUCCESS));
    }

    std::vector<char> uniqueId;
    Hccl::UbTransportLiteImpl transportOnDevice{uniqueId};
    ChannelHandle channel = reinterpret_cast<ChannelHandle>(&transportOnDevice);
    uint32_t localNotifyIdx = 0;
    int32_t res{HCCL_E_RESERVED};
};

TEST_F(
    UtAicpuTsHcommChannelNotifyWaitOnThreadWithDefaultTimeout,
    Ut_HcommChannelNotifyWaitOnThreadWithDefaultTimeout_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    uint32_t customTimeout = 3000;
    res = HcommSetNotifyWaitTimeOut(customTimeout);
    EXPECT_EQ(res, HCCL_SUCCESS);

    res = HcommChannelNotifyWaitOnThreadWithDefaultTimeout(thread, channel, localNotifyIdx);
    EXPECT_EQ(res, HCCL_SUCCESS);
}
