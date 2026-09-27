/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_api_base_test.h"

class HcclAlltoAllVCTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        UT_USE_1SERVER_1RANK_AS_DEFAULT;
        MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
        HcclCommunicator commun_mock;
        MOCKER_CPP_VIRTUAL(commun_mock, &HcclCommunicator::AlltoAllVCOutPlace)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

protected:
    s8* sendBuf = nullptr;
    u64* sendCountMatrix = nullptr;
    s8* recvBuf = nullptr;
};

TEST_F(HcclAlltoAllVCTest, Ut_HcclAlltoAllVC_When_GroupModeSuccess_Expect_ReturnIsHCCL_SUCCESS)
{
    MOCKER(taskAppend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    Ut_Buf_Create(sendBuf, HCCL_COM_DATA_SIZE);
    sendCountMatrix = (u64*)sal_malloc(1 * sizeof(u64));
    sendCountMatrix[0] = HCCL_COM_DATA_SIZE;
    Ut_Buf_Create(recvBuf, HCCL_COM_DATA_SIZE);
    UT_COMM_CREATE_DEFAULT(comm);
    UT_STREAM_CREATE_DEFAULT(stream);
    hcclGroupDepth = 1;

    HcclResult ret = HcclAlltoAllVCInner(
        sendBuf, sendCountMatrix, HCCL_DATA_TYPE_INT8, recvBuf, HCCL_DATA_TYPE_INT8, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclGroupDepth = 0;
    Ut_Stream_SynchronizeAndDestroy(stream);
    Ut_Comm_Destroy(comm);
    sal_free(sendBuf);
    sal_free(sendCountMatrix);
    sal_free(recvBuf);
}
