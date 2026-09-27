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

class UtAicpuTsHcommWriteWithNotifyOnThread : public UtAicpuTsBase {
protected:
    static void SetUpTestCase() { std::cout << "UtAicpuTsHcommWriteWithNotifyOnThread tests set up." << std::endl; }

    static void TearDownTestCase()
    {
        std::cout << "UtAicpuTsHcommWriteWithNotifyOnThread tests tear down." << std::endl;
    }

    virtual void SetUp() override
    {
        std::cout << "A Test case in UtAicpuTsHcommWriteWithNotifyOnThread SetUp" << std::endl;
        UtAicpuTsBase::SetUp();
    }

    virtual void TearDown() override
    {
        UtAicpuTsBase::TearDown();
        std::cout << "A Test case in UtAicpuTsHcommWriteWithNotifyOnThread TearDown" << std::endl;
    }

    std::vector<char> uniqueId;
    Hccl::UbTransportLiteImpl transportOnDevice{uniqueId};
    ChannelHandle channel = reinterpret_cast<ChannelHandle>(&transportOnDevice);
    uint64_t tempDst[6] = {0};
    uint64_t tempSrc[6] = {1, 1, 4, 5, 1, 4};
    void* dst = reinterpret_cast<void*>(tempDst);
    void* src = reinterpret_cast<void*>(tempSrc);
    uint64_t len = sizeof(tempDst);
    uint32_t notifyIdx = 0;
    int32_t res{HCCL_E_RESERVED};
};

TEST_F(UtAicpuTsHcommWriteWithNotifyOnThread, Ut_HcommWriteWithNotifyOnThread_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    res = HcommWriteWithNotifyOnThread(thread, channel, dst, src, len, notifyIdx);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(
    UtAicpuTsHcommWriteWithNotifyOnThread, Ut_HcommWriteWithNotifyOnThread_When_Thread_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommWriteWithNotifyOnThread(0, channel, dst, src, len, notifyIdx);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(
    UtAicpuTsHcommWriteWithNotifyOnThread,
    Ut_HcommWriteWithNotifyOnThread_When_Channel_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommWriteWithNotifyOnThread(thread, 0, dst, src, len, notifyIdx);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtAicpuTsHcommWriteWithNotifyOnThread, Ut_HcommWriteWithNotifyOnThread_When_Dst_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommWriteWithNotifyOnThread(thread, channel, nullptr, src, len, notifyIdx);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtAicpuTsHcommWriteWithNotifyOnThread, Ut_HcommWriteWithNotifyOnThread_When_Src_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommWriteWithNotifyOnThread(thread, channel, dst, nullptr, len, notifyIdx);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(
    UtAicpuTsHcommWriteWithNotifyOnThread,
    Ut_HcommWriteWithNotifyOnThread_When_BuildLocRmaBufferLite_Fail_Expect_ReturnIsHCCL_E_INTERNAL)
{
    GlobalMockObject::verify();
    auto* const transportLitePtr = reinterpret_cast<Hccl::UbTransportLiteImpl*>(channel);
    MOCKER_CPP_VIRTUAL(transportLitePtr, &Hccl::UbTransportLiteImpl::BuildLocRmaBufferLite)
        .stubs()
        .will(returnValue(HCCL_E_INTERNAL));

    res = HcommWriteWithNotifyOnThread(thread, channel, dst, src, len, notifyIdx);
    EXPECT_EQ(res, HCCL_E_INTERNAL);
}
