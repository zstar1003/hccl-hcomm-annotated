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

using namespace hccl;

class UtAicpuTsHcommThreadNotifyWaitOnThread : public UtAicpuTsBase {
protected:
    static void SetUpTestCase() { std::cout << "UtAicpuTsHcommThreadNotifyWaitOnThread tests set up." << std::endl; }

    static void TearDownTestCase()
    {
        std::cout << "UtAicpuTsHcommThreadNotifyWaitOnThread tests tear down." << std::endl;
    }

    virtual void SetUp() override
    {
        std::cout << "A Test case in UtAicpuTsHcommThreadNotifyWaitOnThread SetUp" << std::endl;
        UtAicpuTsBase::SetUp();

        MOCKER_CPP(&Hccl::IAicpuTsThread::NotifyWait, HcclResult(Hccl::IAicpuTsThread::*)(uint32_t, uint32_t) const)
            .stubs()
            .will(returnValue(HCCL_SUCCESS));
    }

    virtual void TearDown() override
    {
        UtAicpuTsBase::TearDown();
        std::cout << "A Test case in UtAicpuTsHcommThreadNotifyWaitOnThread TearDown" << std::endl;
    }

    uint32_t notifyIdx = 0;
    uint32_t timeOut = 1800;
    int32_t res{HCCL_E_RESERVED};
};

TEST_F(UtAicpuTsHcommThreadNotifyWaitOnThread, Ut_HcommThreadNotifyWaitOnThread_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    res = HcommThreadNotifyWaitOnThread(thread, notifyIdx, timeOut);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(
    UtAicpuTsHcommThreadNotifyWaitOnThread,
    Ut_HcommThreadNotifyWaitOnThread_When_Thread_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommThreadNotifyWaitOnThread(0, notifyIdx, timeOut);
    EXPECT_EQ(res, HCCL_E_PTR);
}
