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

class UtAicpuTsHcommLocalReduceOnThread : public UtAicpuTsBase {
protected:
    static void SetUpTestCase() { std::cout << "UtAicpuTsHcommLocalReduceOnThread tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "UtAicpuTsHcommLocalReduceOnThread tests tear down." << std::endl; }

    virtual void SetUp() override
    {
        std::cout << "A Test case in UtAicpuTsHcommLocalReduceOnThread SetUp" << std::endl;
        UtAicpuTsBase::SetUp();

        MOCKER_CPP(
            &Hccl::IAicpuTsThread::SdmaReduce,
            HcclResult(Hccl::IAicpuTsThread::*)(uint64_t, uint64_t, uint64_t, uint32_t, uint32_t) const)
            .stubs()
            .will(returnValue(HCCL_SUCCESS));
    }

    virtual void TearDown() override
    {
        UtAicpuTsBase::TearDown();
        std::cout << "A Test case in UtAicpuTsHcommLocalReduceOnThread TearDown" << std::endl;
    }

    void* dst = reinterpret_cast<void*>(0x2345);
    void* src = reinterpret_cast<void*>(0x2345);
    uint64_t count = 1;
    HcommDataType dataType = HCOMM_DATA_TYPE_FP16;
    HcommReduceOp reduceOp = HCOMM_REDUCE_SUM;
    int32_t res{HCCL_E_RESERVED};
};

TEST_F(UtAicpuTsHcommLocalReduceOnThread, Ut_HcommLocalReduceOnThread_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    res = HcommLocalReduceOnThread(thread, dst, src, count, dataType, reduceOp);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtAicpuTsHcommLocalReduceOnThread, Ut_HcommLocalReduceOnThread_When_Thread_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommLocalReduceOnThread(0, dst, src, count, dataType, reduceOp);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtAicpuTsHcommLocalReduceOnThread, Ut_HcommLocalReduceOnThread_When_Dst_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommLocalReduceOnThread(thread, nullptr, src, count, dataType, reduceOp);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtAicpuTsHcommLocalReduceOnThread, Ut_HcommLocalReduceOnThread_When_Src_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommLocalReduceOnThread(thread, dst, nullptr, count, dataType, reduceOp);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtAicpuTsHcommLocalReduceOnThread, Ut_HcommLocalReduceOnThread_When_CountEquals4GB_Expect_ReturnIsHCCL_SUCCESS)
{
    uint64_t count4GB = 0x100000000ULL;
    void* largeDst = reinterpret_cast<void*>(0x100000);
    void* largeSrc = reinterpret_cast<void*>(0x200000);

    res = HcommLocalReduceOnThread(thread, largeDst, largeSrc, count4GB, dataType, reduceOp);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtAicpuTsHcommLocalReduceOnThread, Ut_HcommLocalReduceOnThread_When_CountExceeds4GB_Expect_ReturnIsHCCL_SUCCESS)
{
    uint64_t countExceeds4GB = 0x100000001ULL;
    void* largeDst = reinterpret_cast<void*>(0x100000);
    void* largeSrc = reinterpret_cast<void*>(0x200000);

    res = HcommLocalReduceOnThread(thread, largeDst, largeSrc, countExceeds4GB, dataType, reduceOp);
    EXPECT_EQ(res, HCCL_SUCCESS);
}
