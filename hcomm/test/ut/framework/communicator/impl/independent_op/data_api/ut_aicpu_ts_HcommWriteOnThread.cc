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
#include "exception_util.h"
#include "hccl_exception.h"

using namespace hccl;

class UtAicpuTsHcommWriteOnThread : public UtAicpuTsBase {
protected:
    static void SetUpTestCase() { std::cout << "UtAicpuTsHcommWriteOnThread tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "UtAicpuTsHcommWriteOnThread tests tear down." << std::endl; }

    virtual void SetUp() override
    {
        std::cout << "A Test case in UtAicpuTsHcommWriteOnThread SetUp" << std::endl;
        UtAicpuTsBase::SetUp();
    }

    virtual void TearDown() override
    {
        UtAicpuTsBase::TearDown();
        std::cout << "A Test case in UtAicpuTsHcommWriteOnThread TearDown" << std::endl;
    }

    uint64_t tempDst[6] = {0};
    uint64_t tempSrc[6] = {1, 1, 4, 5, 1, 4};
    void* dst = reinterpret_cast<void*>(tempDst);
    void* src = reinterpret_cast<void*>(tempSrc);
    uint64_t len = sizeof(tempDst);
    std::vector<char> uniqueId;
    Hccl::UbTransportLiteImpl transportDev{uniqueId};
    ChannelHandle devHandle = reinterpret_cast<ChannelHandle>(&transportDev);
    int32_t res{HCCL_E_RESERVED};
};

TEST_F(UtAicpuTsHcommWriteOnThread, Ut_HcommWriteOnThread_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    res = HcommWriteOnThread(thread, devHandle, dst, src, len);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtAicpuTsHcommWriteOnThread, Ut_HcommWriteOnThread_When_Thread_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommWriteOnThread(0, devHandle, dst, src, len);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(
    UtAicpuTsHcommWriteOnThread, Ut_HcommWriteOnThread_When_BuildLocRmaBufferLite_Fail_Expect_ReturnIsHCCL_E_INTERNAL)
{
    GlobalMockObject::verify();
    auto* const transportLitePtr = reinterpret_cast<Hccl::UbTransportLiteImpl*>(devHandle);
    MOCKER_CPP_VIRTUAL(transportLitePtr, &Hccl::UbTransportLiteImpl::BuildLocRmaBufferLite)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    res = HcommWriteOnThread(thread, devHandle, dst, src, len);
    EXPECT_EQ(res, HCCL_E_INTERNAL);
}

TEST_F(
    UtAicpuTsHcommWriteOnThread, Ut_HcommWriteOnThread_When_GetRmtRmaBufSliceLite_Throw_Expect_ReturnIsHCCL_E_INTERNAL)
{
    MOCKER_CPP_VIRTUAL(transportDev, &Hccl::UbTransportLiteImpl::Write)
        .stubs()
        .will(throws(Hccl::InternalException("test exception")));
    res = HcommWriteOnThread(thread, devHandle, dst, src, len);
    EXPECT_EQ(res, HCCL_E_INTERNAL);
}
