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

class HcclReduceScatterVTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        UT_USE_1SERVER_1RANK_AS_DEFAULT;
        // 将enableEntryLog默认返回为true
        MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
        // MOCK掉对communicator层的依赖，保证分层测试
        HcclCommunicator commun_mock;
        MOCKER_CPP_VIRTUAL(commun_mock, &HcclCommunicator::ReduceScatterVOutPlace)
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
    u64* sendCounts = nullptr;
    u64* sendDispls = nullptr;
    s8* recvBuf = nullptr;
    u64 recvCount = 0;
};

TEST_F(HcclReduceScatterVTest, Ut_HcclReduceScatterV_When_SendBufIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    UT_SET_SENDBUFV_RECVBUF_COUNT(0, 1, HCCL_COM_DATA_SIZE, 1, 0, HCCL_COM_DATA_SIZE, HCCL_COM_DATA_SIZE);
    UT_COMM_CREATE_DEFAULT(comm);
    UT_STREAM_CREATE_DEFAULT(stream);

    HcclResult ret = HcclReduceScatterVInner(
        sendBuf, sendCounts, sendDispls, recvBuf, recvCount, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_E_PTR);

    UT_UNSET_SENDBUFV_RECVBUF_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, stream);
}

TEST_F(HcclReduceScatterVTest, Ut_HcclReduceScatterV_When_SendCountsIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    UT_SET_SENDBUFV_RECVBUF_COUNT(HCCL_COM_DATA_SIZE, 0, 0, 1, 0, HCCL_COM_DATA_SIZE, HCCL_COM_DATA_SIZE);
    UT_COMM_CREATE_DEFAULT(comm);
    UT_STREAM_CREATE_DEFAULT(stream);

    HcclResult ret = HcclReduceScatterVInner(
        sendBuf, sendCounts, sendDispls, recvBuf, recvCount, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_E_PTR);

    UT_UNSET_SENDBUFV_RECVBUF_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, stream);
}

TEST_F(HcclReduceScatterVTest, Ut_HcclReduceScatterV_When_SendDisplsIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    UT_SET_SENDBUFV_RECVBUF_COUNT(
        HCCL_COM_DATA_SIZE, 1, HCCL_COM_DATA_SIZE, 0, 0, HCCL_COM_DATA_SIZE, HCCL_COM_DATA_SIZE);
    UT_COMM_CREATE_DEFAULT(comm);
    UT_STREAM_CREATE_DEFAULT(stream);

    HcclResult ret = HcclReduceScatterVInner(
        sendBuf, sendCounts, sendDispls, recvBuf, recvCount, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_E_PTR);

    UT_UNSET_SENDBUFV_RECVBUF_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, stream);
}

TEST_F(HcclReduceScatterVTest, Ut_HcclReduceScatterV_When_SendCountsZero_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_SET_SENDBUFV_RECVBUF_COUNT(HCCL_COM_DATA_SIZE, 1, 0, 1, 0, HCCL_COM_DATA_SIZE, HCCL_COM_DATA_SIZE);
    UT_COMM_CREATE_DEFAULT(comm);
    UT_STREAM_CREATE_DEFAULT(stream);

    HcclResult ret = HcclReduceScatterVInner(
        sendBuf, sendCounts, sendDispls, recvBuf, recvCount, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    UT_UNSET_SENDBUFV_RECVBUF_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, stream);
}

TEST_F(HcclReduceScatterVTest, Ut_HcclReduceScatterV_When_RecvBufIsNullAndRecvCountGEZero_Expect_ReturnIsHCCL_E_PTR)
{
    UT_SET_SENDBUFV_RECVBUF_COUNT(HCCL_COM_DATA_SIZE, 1, HCCL_COM_DATA_SIZE, 1, 0, 0, HCCL_COM_DATA_SIZE);
    UT_COMM_CREATE_DEFAULT(comm);
    UT_STREAM_CREATE_DEFAULT(stream);

    HcclResult ret = HcclReduceScatterVInner(
        sendBuf, sendCounts, sendDispls, recvBuf, recvCount, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_E_PTR);

    UT_UNSET_SENDBUFV_RECVBUF_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, stream);
}

TEST_F(HcclReduceScatterVTest, Ut_HcclReduceScatterV_When_RecvBufIsNullAndRecvCountIsZero_Expect_ReturnIsHCCL_E_PARA)
{
    UT_SET_SENDBUFV_RECVBUF_COUNT(HCCL_COM_DATA_SIZE, 1, HCCL_COM_DATA_SIZE, 1, 0, 0, 0);
    UT_COMM_CREATE_DEFAULT(comm);
    UT_STREAM_CREATE_DEFAULT(stream);

    HcclResult ret = HcclReduceScatterVInner(
        sendBuf, sendCounts, sendDispls, recvBuf, recvCount, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_E_PARA);

    UT_UNSET_SENDBUFV_RECVBUF_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, stream);
}

TEST_F(HcclReduceScatterVTest, Ut_HcclReduceScatterV_When_RecvCountIsTooLarge_Expect_ReturnIsHCCL_E_PARA)
{
    UT_SET_SENDBUFV_RECVBUF_COUNT(
        HCCL_COM_DATA_SIZE, 1, HCCL_COM_DATA_SIZE, 1, 0, HCCL_COM_DATA_SIZE, SYS_MAX_COUNT + 1);
    UT_COMM_CREATE_DEFAULT(comm);
    UT_STREAM_CREATE_DEFAULT(stream);

    HcclResult ret = HcclReduceScatterVInner(
        sendBuf, sendCounts, sendDispls, recvBuf, recvCount, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_E_PARA);

    UT_UNSET_SENDBUFV_RECVBUF_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, stream);
}

TEST_F(HcclReduceScatterVTest, Ut_HcclReduceScatterV_When_CommIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    UT_SET_SENDBUFV_RECVBUF_COUNT(
        HCCL_COM_DATA_SIZE, 1, HCCL_COM_DATA_SIZE, 1, 0, HCCL_COM_DATA_SIZE, HCCL_COM_DATA_SIZE);
    Ut_Device_Set(0);

    UT_STREAM_CREATE_DEFAULT(stream);

    HcclResult ret = HcclReduceScatterVInner(
        sendBuf, sendCounts, sendDispls, recvBuf, recvCount, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_E_PTR);

    UT_UNSET_SENDBUFV_RECVBUF_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, stream);
}

TEST_F(HcclReduceScatterVTest, Ut_HcclReduceScatterV_When_DeviceNotSupport_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    UT_SET_SENDBUFV_RECVBUF_COUNT(
        HCCL_COM_DATA_SIZE, 1, HCCL_COM_DATA_SIZE, 1, 0, HCCL_COM_DATA_SIZE, HCCL_COM_DATA_SIZE);
    UT_COMM_CREATE_DEFAULT(comm);
    UT_STREAM_CREATE_DEFAULT(stream);
    DevType deviceType = DevType::DEV_TYPE_910_93 /*or DevType::DEV_TYPE_910B*/;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcclReduceScatterVInner(
        sendBuf, sendCounts, sendDispls, recvBuf, recvCount, HCCL_DATA_TYPE_INT16 /*or HCCL_DATA_TYPE_BFP16*/,
        HCCL_REDUCE_PROD, comm, stream);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    UT_UNSET_SENDBUFV_RECVBUF_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, stream);
}

TEST_F(HcclReduceScatterVTest, Ut_HcclReduceScatterV_When_RankDataNotEqual_Expect_ReturnIsHCCL_E_PARA)
{
    UT_SET_SENDBUFV_RECVBUF_COUNT(
        HCCL_COM_DATA_SIZE + 1, 1, HCCL_COM_DATA_SIZE + 1, 1, 0, HCCL_COM_DATA_SIZE, HCCL_COM_DATA_SIZE);
    UT_COMM_CREATE_DEFAULT(comm);
    UT_STREAM_CREATE_DEFAULT(stream);

    HcclResult ret = HcclReduceScatterVInner(
        sendBuf, sendCounts, sendDispls, recvBuf, recvCount, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_E_PARA);

    UT_UNSET_SENDBUFV_RECVBUF_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, stream);
}

TEST_F(HcclReduceScatterVTest, Ut_HcclReduceScatterV_When_DataSize1KB_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_SET_SENDBUFV_RECVBUF_COUNT(
        HCCL_COM_DATA_SIZE, 1, HCCL_COM_DATA_SIZE, 1, 0, HCCL_COM_DATA_SIZE, HCCL_COM_DATA_SIZE);
    UT_COMM_CREATE_DEFAULT(comm);
    UT_STREAM_CREATE_DEFAULT(stream);

    HcclResult ret = HcclReduceScatterVInner(
        sendBuf, sendCounts, sendDispls, recvBuf, recvCount, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    UT_UNSET_SENDBUFV_RECVBUF_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, stream);
}

TEST_F(HcclReduceScatterVTest, Ut_HcclReduceScatterV_When_DataSize300MB_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_SET_SENDBUFV_RECVBUF_COUNT(
        HCCL_COM_BIG_DATA_SIZE, 1, HCCL_COM_BIG_DATA_SIZE, 1, 0, HCCL_COM_BIG_DATA_SIZE, HCCL_COM_BIG_DATA_SIZE);
    UT_COMM_CREATE_DEFAULT(comm);
    UT_STREAM_CREATE_DEFAULT(stream);

    HcclResult ret = HcclReduceScatterVInner(
        sendBuf, sendCounts, sendDispls, recvBuf, recvCount, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    UT_UNSET_SENDBUFV_RECVBUF_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, stream);
}

TEST_F(HcclReduceScatterVTest, Ut_HcclReduceScatterV_When_Exec20times_Expect_ReturnIsHCCL_SUCCESS)
{
    constexpr int LOOP_TIMES = 20;
    UT_SET_SENDBUFV_RECVBUF_COUNT(
        HCCL_COM_DATA_SIZE, 1, HCCL_COM_DATA_SIZE, 1, 0, HCCL_COM_DATA_SIZE, HCCL_COM_DATA_SIZE);
    UT_COMM_CREATE_DEFAULT(comm);
    UT_STREAM_CREATE_DEFAULT(stream);

    for (u64 k = 0; k < LOOP_TIMES; k++) {
        HcclResult ret = HcclReduceScatterVInner(
            sendBuf, sendCounts, sendDispls, recvBuf, recvCount, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        Ut_Stream_Synchronize(stream);
    }

    UT_UNSET_SENDBUFV_RECVBUF_COMM_STREAM(comm, stream);
}

TEST_F(HcclReduceScatterVTest, Ut_HcclReduceScatterV_When_2Server4Rank_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_SET_SENDBUFV_RECVBUF_COUNT(
        HCCL_COM_DATA_SIZE, rankNum, HCCL_COM_DATA_SIZE, rankNum, HCCL_COM_DATA_SIZE, HCCL_COM_DATA_SIZE,
        HCCL_COM_DATA_SIZE);
    UT_COMM_CREATE_DEFAULT(comm);
    UT_STREAM_CREATE_DEFAULT(stream);

    HcclResult ret = HcclReduceScatterVInner(
        sendBuf, sendCounts, sendDispls, recvBuf, recvCount, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    UT_UNSET_SENDBUFV_RECVBUF_COMM_STREAM(comm, stream);
}

TEST_F(HcclReduceScatterVTest, Ut_HcclReduceScatterV_When_GroupModeSuccess_Expect_ReturnIsHCCL_SUCCESS)
{
    MOCKER(taskAppend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    UT_SET_SENDBUFV_RECVBUF_COUNT(
        HCCL_COM_DATA_SIZE, 1, HCCL_COM_DATA_SIZE, 1, 0, HCCL_COM_DATA_SIZE, HCCL_COM_DATA_SIZE);
    UT_COMM_CREATE_DEFAULT(comm);
    UT_STREAM_CREATE_DEFAULT(stream);
    hcclGroupDepth = 1;

    HcclResult ret = HcclReduceScatterVInner(
        sendBuf, sendCounts, sendDispls, recvBuf, recvCount, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclGroupDepth = 0;
    UT_UNSET_SENDBUFV_RECVBUF_COMM_STREAM_WITHSTREAMSYNCHRONIZEFIRST(comm, stream);
}
