/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_types.h>
#include <hccl/hccl_res_expt.h>
#include "hccl_api_base_test.h"
#include "log.h"

#define private public

using namespace hccl;

class HcclGetRemoteIpcHcclBufTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        UT_USE_RANK_TABLE_910_1SERVER_1RANK;
        UT_COMM_CREATE_DEFAULT(comm);
    }
    void TearDown() override
    {
        Ut_Comm_Destroy(comm);
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

HcclResult GetRemoteCCLBufStub(HcclCommunicator* This, uint32_t remoteRank, void** addr, uint64_t* size)
{
    *addr = reinterpret_cast<void*>(0x12345678);
    *size = 0;
    return HCCL_SUCCESS;
}

HcclResult GetRemoteCCLBufStubWithNullAddr(HcclCommunicator* This, uint32_t remoteRank, void** addr, uint64_t* size)
{
    *addr = nullptr;
    *size = 0;
    return HCCL_SUCCESS;
}

HcclResult GetRemoteCCLBufStubWithValidSize(HcclCommunicator* This, uint32_t remoteRank, void** addr, uint64_t* size)
{
    *addr = reinterpret_cast<void*>(0x12345678);
    *size = 4096;
    return HCCL_SUCCESS;
}

HcclResult GetRemoteCCLBufStubWithFailure(HcclCommunicator* This, uint32_t remoteRank, void** addr, uint64_t* size)
{
    *addr = nullptr;
    *size = 0;
    return HCCL_E_INTERNAL;
}

TEST_F(HcclGetRemoteIpcHcclBufTest, ut_HcclGetRemoteIpcHcclBuf_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    void* addr = nullptr;
    uint64_t size = 0;

    MOCKER_CPP(&HcclCommunicator::GetRemoteCCLBuf).stubs().will(invoke(GetRemoteCCLBufStub));

    HcclResult ret = HcclGetRemoteIpcHcclBuf(comm, 1, &addr, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclGetRemoteIpcHcclBufTest, ut_HcclGetRemoteIpcHcclBuf_When_ParamIsNullptr_Expect_ReturnIsHCCL_E_PTR)
{
    void* addr = nullptr;
    uint64_t size = 0;

    HcclResult ret = HcclGetRemoteIpcHcclBuf(nullptr, 1, &addr, &size);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclGetRemoteIpcHcclBuf(comm, 1, nullptr, &size);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclGetRemoteIpcHcclBuf(comm, 1, &addr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclGetRemoteIpcHcclBufTest, ut_HcclGetRemoteIpcHcclBuf_When_RemoteRankIsInvalid_Expect_ReturnIsHCCL_E_PTR)
{
    void* addr = nullptr;
    uint64_t size = 0;

    HcclResult ret = HcclGetRemoteIpcHcclBuf(comm, 16, &addr, &size);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclGetRemoteIpcHcclBufTest, ut_HcclGetRemoteIpcHcclBuf_When_RemoteRankEqualsMax_Expect_ReturnIsHCCL_E_PARA)
{
    void* addr = nullptr;
    uint64_t size = 0;

    HcclResult ret = HcclGetRemoteIpcHcclBuf(comm, 128 * 1024, &addr, &size);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclGetRemoteIpcHcclBufTest, ut_HcclGetRemoteIpcHcclBuf_When_RemoteRank0_Expect_ReturnIsHCCL_SUCCESS)
{
    void* addr = nullptr;
    uint64_t size = 0;

    MOCKER_CPP(&HcclCommunicator::GetRemoteCCLBuf).stubs().will(invoke(GetRemoteCCLBufStub));

    HcclResult ret = HcclGetRemoteIpcHcclBuf(comm, 0, &addr, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclGetRemoteIpcHcclBufTest, ut_HcclGetRemoteIpcHcclBuf_When_ValidAddrAndSize_Expect_ReturnIsHCCL_SUCCESS)
{
    void* addr = nullptr;
    uint64_t size = 0;

    MOCKER_CPP(&HcclCommunicator::GetRemoteCCLBuf).stubs().will(invoke(GetRemoteCCLBufStubWithValidSize));

    HcclResult ret = HcclGetRemoteIpcHcclBuf(comm, 1, &addr, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(addr, reinterpret_cast<void*>(0x12345678));
    EXPECT_EQ(size, 4096);
    GlobalMockObject::verify();
}

TEST_F(HcclGetRemoteIpcHcclBufTest, ut_HcclGetRemoteIpcHcclBuf_When_GetRemoteCCLBufFails_Expect_ReturnIsError)
{
    void* addr = nullptr;
    uint64_t size = 0;

    MOCKER_CPP(&HcclCommunicator::GetRemoteCCLBuf).stubs().will(invoke(GetRemoteCCLBufStubWithFailure));

    HcclResult ret = HcclGetRemoteIpcHcclBuf(comm, 1, &addr, &size);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    GlobalMockObject::verify();
}
