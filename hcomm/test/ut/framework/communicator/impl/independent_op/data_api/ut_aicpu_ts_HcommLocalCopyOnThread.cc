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

class UtAicpuTsHcommLocalCopyOnThread : public UtAicpuTsBase {
protected:
    static void SetUpTestCase() { std::cout << "UtAicpuTsHcommLocalCopyOnThread tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "UtAicpuTsHcommLocalCopyOnThread tests tear down." << std::endl; }

    virtual void SetUp() override
    {
        std::cout << "A Test case in UtAicpuTsHcommLocalCopyOnThread SetUP" << std::endl;
        UtAicpuTsBase::SetUp();

        MOCKER_CPP(
            &Hccl::IAicpuTsThread::SdmaCopy, HcclResult(Hccl::IAicpuTsThread::*)(uint64_t, uint64_t, uint64_t) const)
            .stubs()
            .will(returnValue(HCCL_SUCCESS));
    }

    virtual void TearDown() override
    {
        UtAicpuTsBase::TearDown();
        std::cout << "A Test case in UtAicpuTsHcommLocalCopyOnThread TearDown" << std::endl;
    }

    void* dst = reinterpret_cast<void*>(0x2345);
    void* src = reinterpret_cast<void*>(0x2345);
    uint64_t len = 1;
    int32_t res{HCCL_E_RESERVED};
};

TEST_F(UtAicpuTsHcommLocalCopyOnThread, Ut_HcommLocalCopyOnThread_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    res = HcommLocalCopyOnThread(thread, dst, src, len);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtAicpuTsHcommLocalCopyOnThread, Ut_HcommLocalCopyOnThread_When_Thread_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommLocalCopyOnThread(0, dst, src, len);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtAicpuTsHcommLocalCopyOnThread, Ut_HcommLocalCopyOnThread_When_Dst_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommLocalCopyOnThread(thread, nullptr, src, len);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtAicpuTsHcommLocalCopyOnThread, Ut_HcommLocalCopyOnThread_When_Src_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommLocalCopyOnThread(thread, dst, nullptr, len);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtAicpuTsHcommLocalCopyOnThread, Ut_HcommLocalCopyOnThread_When_SizeEquals4GB_Expect_ReturnIsHCCL_SUCCESS)
{
    uint64_t size4GB = 0x100000000ULL;
    void* largeDst = reinterpret_cast<void*>(0x100000);
    void* largeSrc = reinterpret_cast<void*>(0x200000);

    res = HcommLocalCopyOnThread(thread, largeDst, largeSrc, size4GB);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtAicpuTsHcommLocalCopyOnThread, Ut_HcommLocalCopyOnThread_When_SizeExceeds4GB_Expect_ReturnIsHCCL_SUCCESS)
{
    uint64_t sizeExceeds4GB = 0x100000001ULL;
    void* largeDst = reinterpret_cast<void*>(0x100000);
    void* largeSrc = reinterpret_cast<void*>(0x200000);

    res = HcommLocalCopyOnThread(thread, largeDst, largeSrc, sizeExceeds4GB);
    EXPECT_EQ(res, HCCL_SUCCESS);
}
