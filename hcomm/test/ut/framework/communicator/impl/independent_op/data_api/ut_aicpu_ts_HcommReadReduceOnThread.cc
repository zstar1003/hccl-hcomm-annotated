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

class UtAicpuTsHcommReadReduceOnThread : public UtAicpuTsBase {
protected:
    static void SetUpTestCase() { std::cout << "UtAicpuTsHcommReadReduceOnThread tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "UtAicpuTsHcommReadReduceOnThread tests tear down." << std::endl; }

    virtual void SetUp() override
    {
        std::cout << "A Test case in UtAicpuTsHcommReadReduceOnThread SetUp" << std::endl;
        UtAicpuTsBase::SetUp();
    }

    virtual void TearDown() override
    {
        UtAicpuTsBase::TearDown();
        std::cout << "A Test case in UtAicpuTsHcommReadReduceOnThread TearDown" << std::endl;
    }

    uint64_t tempDst[6] = {0};
    uint64_t tempSrc[6] = {1, 1, 4, 5, 1, 4};
    void* dst = reinterpret_cast<void*>(tempDst);
    void* src = reinterpret_cast<void*>(tempSrc);
    std::vector<char> uniqueId;
    Hccl::UbTransportLiteImpl transportDev{uniqueId};
    ChannelHandle devHandle = reinterpret_cast<ChannelHandle>(&transportDev);
    uint64_t count = 1;
    HcommDataType dataType = HCOMM_DATA_TYPE_INT8;
    HcommReduceOp reduceOp = HCOMM_REDUCE_SUM;
    int32_t res{HCCL_E_RESERVED};
};

TEST_F(UtAicpuTsHcommReadReduceOnThread, Ut_HcommReadReduceOnThread_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    res = HcommReadReduceOnThread(thread, devHandle, dst, src, count, dataType, reduceOp);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtAicpuTsHcommReadReduceOnThread, Ut_HcommReadReduceOnThread_When_Thread_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommReadReduceOnThread(0, devHandle, dst, src, count, dataType, reduceOp);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtAicpuTsHcommReadReduceOnThread, Ut_HcommReadReduceOnThread_When_DataType_IsInvalid_Expect_ReturnIsHCCL_E_PARA)
{
    dataType = HCOMM_DATA_TYPE_RESERVED;
    res = HcommReadReduceOnThread(thread, devHandle, dst, src, count, dataType, reduceOp);
    EXPECT_EQ(res, HCCL_E_PARA);
}

TEST_F(
    UtAicpuTsHcommReadReduceOnThread,
    Ut_HcommReadReduceOnThread_When_BuildLocRmaBufferLite_Fail_Expect_ReturnIsHCCL_E_INTERNAL)
{
    GlobalMockObject::verify();
    auto* const transportLitePtr = reinterpret_cast<Hccl::UbTransportLiteImpl*>(devHandle);
    MOCKER_CPP_VIRTUAL(transportLitePtr, &Hccl::UbTransportLiteImpl::BuildLocRmaBufferLite)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    res = HcommReadReduceOnThread(thread, devHandle, dst, src, count, dataType, reduceOp);
    EXPECT_EQ(res, HCCL_E_INTERNAL);
}
