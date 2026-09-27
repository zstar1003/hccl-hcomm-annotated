/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include "aicpu/aicpu_hccl_sqcqv1.h"
#include "hccl_common.h"
#include "dispatcher_task_types.h"

using namespace std;
using namespace hccl;

class RtsqInteract_Sqcqv1_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RtsqInteract_Sqcqv1_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "RtsqInteract_Sqcqv1_UT TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "RtsqInteract_Sqcqv1_UT Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "RtsqInteract_Sqcqv1_UT Test TearDown" << std::endl;
    }
};

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneRdmaDbSendSqeV1_When_DbAddrIsZero_Expect_SqeTypeIsInvalid)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    AddOneRdmaDbSendSqeV1(1, 1, 0, 0, 100, 1, sqe, &sqeType);
    EXPECT_EQ(sqeType, RDMA_DB_SEND_SQE);
    rtStarsWriteValueSqe_t* sqeStruct = reinterpret_cast<rtStarsWriteValueSqe_t*>(sqe);
    EXPECT_EQ(sqeStruct->header.type, RT_STARS_SQE_TYPE_INVALID);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneRdmaDbSendSqeV1_When_NormalParams_Expect_SqeTypeIsRdmaDbSend)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint64_t dbAddr = 0x1000000ULL;
    AddOneRdmaDbSendSqeV1(1, 1, 0x12345678, dbAddr, 100, 2, sqe, &sqeType);
    EXPECT_EQ(sqeType, RDMA_DB_SEND_SQE);
    rtStarsWriteValueSqe_t* sqeStruct = reinterpret_cast<rtStarsWriteValueSqe_t*>(sqe);
    EXPECT_EQ(sqeStruct->header.type, RT_STARS_SQE_TYPE_WRITE_VALUE);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneRdmaDbSendSqeV1_When_NormalParams_Expect_SqeFieldsAreCorrect)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint64_t dbInfo = 0xABCD;
    uint64_t dbAddr = 0x1000000ULL;
    uint32_t length = 256;
    uint8_t rdmaType = 3;
    uint16_t streamId = 5;
    uint16_t taskId = 10;

    AddOneRdmaDbSendSqeV1(streamId, taskId, dbInfo, dbAddr, length, rdmaType, sqe, &sqeType);

    rtStarsWriteValueSqe_t* sqeStruct = reinterpret_cast<rtStarsWriteValueSqe_t*>(sqe);
    EXPECT_EQ(sqeStruct->header.type, RT_STARS_SQE_TYPE_WRITE_VALUE);
    EXPECT_EQ(sqeStruct->header.rtStreamId, streamId);
    EXPECT_EQ(sqeStruct->header.taskId, taskId);
    EXPECT_EQ(sqeStruct->sub_type, RT_STARS_WRITE_VALUE_SUB_TYPE_RDMA_DB_SEND);
    EXPECT_EQ(sqeStruct->write_addr_low, static_cast<uint32_t>(dbAddr & MASK_32_BIT));
    EXPECT_EQ(sqeStruct->write_addr_high, static_cast<uint32_t>((dbAddr >> UINT32_BIT_NUM) & MASK_17_BIT));
    EXPECT_EQ(sqeStruct->rdmaWrLenth, length);
    EXPECT_EQ(sqeStruct->rdmaType, rdmaType);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneNotifyWaitSqeV1)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    u64 notifyId = 100;
    dfx::DfxTimeOutConfig dfxTimeOutConfig = {};
    dfxTimeOutConfig.useCredit = true;
    dfxTimeOutConfig.sqeCreditTimeOut = 240;

    AddOneNotifyWaitSqeV1(streamId, taskId, notifyId, sqe, &sqeType, dfxTimeOutConfig);

    EXPECT_EQ(sqeType, NOTIFY_SQE);
    rtStarsNotifySqeV1_t* sqeStruct = reinterpret_cast<rtStarsNotifySqeV1_t*>(sqe);
    EXPECT_EQ(sqeStruct->header.type, RT_STARS_SQE_TYPE_NOTIFY_WAIT);
    EXPECT_EQ(sqeStruct->header.rtStreamId, streamId);
    EXPECT_EQ(sqeStruct->header.taskId, taskId);
    EXPECT_EQ(sqeStruct->notify_id, notifyId);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneRecordSqeV1)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    u64 notifyId = 100;

    AddOneRecordSqeV1(streamId, taskId, notifyId, sqe, &sqeType);

    EXPECT_EQ(sqeType, NOTIFY_SQE);
    rtStarsNotifySqeV1_t* sqeStruct = reinterpret_cast<rtStarsNotifySqeV1_t*>(sqe);
    EXPECT_EQ(sqeStruct->header.type, RT_STARS_SQE_TYPE_NOTIFY_RECORD);
    EXPECT_EQ(sqeStruct->header.rtStreamId, streamId);
    EXPECT_EQ(sqeStruct->header.taskId, taskId);
    EXPECT_EQ(sqeStruct->notify_id, notifyId);
    EXPECT_EQ(sqeStruct->kernel_credit, RT_STARS_DEFAULT_KERNEL_CREDIT);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneWriteValueRecordSqeV1)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    u64 notifyWRAddr = 0x1000000UL;

    AddOneWriteValueRecordSqeV1(streamId, taskId, notifyWRAddr, sqe, &sqeType);

    EXPECT_EQ(sqeType, WRITE_VALUE_SQE);
    rtStarsWriteValueSqe_t* sqeStruct = reinterpret_cast<rtStarsWriteValueSqe_t*>(sqe);
    EXPECT_EQ(sqeStruct->header.type, RT_STARS_SQE_TYPE_WRITE_VALUE);
    EXPECT_EQ(sqeStruct->header.rtStreamId, streamId);
    EXPECT_EQ(sqeStruct->header.taskId, taskId);
    EXPECT_EQ(sqeStruct->kernel_credit, RT_STARS_DEFAULT_KERNEL_CREDIT);
    EXPECT_EQ(sqeStruct->awsize, RT_STARS_WRITE_VALUE_SIZE_TYPE_32BIT);
    EXPECT_EQ(sqeStruct->write_value_part0, 1U);
    EXPECT_EQ(sqeStruct->sub_type, RT_STARS_WRITE_VALUE_SUB_TYPE_NOTIFY_RECORD_IPC_NO_PCIE);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneMemcpySqeV1_WithReduce)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    void* src = reinterpret_cast<void*>(0x1000UL);
    void* dst = reinterpret_cast<void*>(0x2000UL);
    uint32_t length = 256;
    aclDataType runtimeDataType = ACL_FLOAT16;
    aclrtReduceKind rtReduceOp = ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM;
    uint32_t partId = 1;
    uint32_t ssid = 2;
    uint32_t devId = 0;
    u64 overflowAddr = 0x3000UL;
    uint8_t linkType = 0;
    uint32_t hcclQos = 3;

    AddOneMemcpySqeV1(
        streamId, taskId, src, length, runtimeDataType, rtReduceOp, dst, partId, ssid, devId, overflowAddr, linkType,
        sqe, &sqeType, hcclQos);

    EXPECT_EQ(sqeType, MEMCPY_ASYNC_SQE);
    rtStarsMemcpyAsyncSqe_t* sqeStruct = reinterpret_cast<rtStarsMemcpyAsyncSqe_t*>(sqe);
    EXPECT_EQ(sqeStruct->header.type, RT_STARS_SQE_TYPE_SDMA);
    EXPECT_EQ(sqeStruct->header.rtStreamId, streamId);
    EXPECT_EQ(sqeStruct->header.taskId, taskId);
    EXPECT_EQ(sqeStruct->length, length);
    EXPECT_NE(sqeStruct->opcode, 0);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneMemcpySqeV1_WithLinkSio)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    void* src = reinterpret_cast<void*>(0x1000UL);
    void* dst = reinterpret_cast<void*>(0x2000UL);
    uint32_t length = 256;
    aclDataType runtimeDataType = ACL_FLOAT16;
    aclrtReduceKind rtReduceOp = ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM;
    uint32_t partId = 1;
    uint32_t ssid = 2;
    uint32_t devId = 0;
    u64 overflowAddr = 0x3000UL;
    uint8_t linkType = static_cast<uint8_t>(hccl::LinkType::LINK_SIO);
    uint32_t hcclQos = 10;

    AddOneMemcpySqeV1(
        streamId, taskId, src, length, runtimeDataType, rtReduceOp, dst, partId, ssid, devId, overflowAddr, linkType,
        sqe, &sqeType, hcclQos);

    EXPECT_EQ(sqeType, MEMCPY_ASYNC_SQE);
    rtStarsMemcpyAsyncSqe_t* sqeStruct = reinterpret_cast<rtStarsMemcpyAsyncSqe_t*>(sqe);
    EXPECT_EQ(sqeStruct->qos, SDMA_QOS_DEFAULT);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneEventResetSqeV1)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    int32_t eventId = 3;
    int64_t phyChipId = 0;
    int64_t phyDieId = 0;
    u64 eventAddr = 0x1000000UL;

    AddOneEventResetSqeV1(streamId, eventId, taskId, phyChipId, phyDieId, eventAddr, sqe, &sqeType);

    EXPECT_EQ(sqeType, WRITE_VALUE_SQE);
    rtStarsWriteValueSqe_t* sqeStruct = reinterpret_cast<rtStarsWriteValueSqe_t*>(sqe);
    EXPECT_EQ(sqeStruct->header.type, RT_STARS_SQE_TYPE_WRITE_VALUE);
    EXPECT_EQ(sqeStruct->header.rtStreamId, streamId);
    EXPECT_EQ(sqeStruct->header.taskId, taskId);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneEventRecordSqeV1)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    int32_t eventId = 3;

    AddOneEventRecordSqeV1(streamId, eventId, taskId, sqe, &sqeType);

    EXPECT_EQ(sqeType, EVENT_SQE);
    rtStarsEventSqe_t* sqeStruct = reinterpret_cast<rtStarsEventSqe_t*>(sqe);
    EXPECT_EQ(sqeStruct->header.type, RT_STARS_SQE_TYPE_EVENT_RECORD);
    EXPECT_EQ(sqeStruct->kernel_credit, RT_STARS_DEFAULT_KERNEL_CREDIT);
    EXPECT_EQ(sqeStruct->header.rtStreamId, streamId);
    EXPECT_EQ(sqeStruct->eventId, eventId);
    EXPECT_EQ(sqeStruct->header.taskId, taskId);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneEventWaitSqeV1)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    int32_t eventId = 3;

    AddOneEventWaitSqeV1(streamId, eventId, taskId, sqe, &sqeType);

    EXPECT_EQ(sqeType, EVENT_SQE);
    rtStarsEventSqe_t* sqeStruct = reinterpret_cast<rtStarsEventSqe_t*>(sqe);
    EXPECT_EQ(sqeStruct->header.type, RT_STARS_SQE_TYPE_EVENT_WAIT);
    EXPECT_EQ(sqeStruct->kernel_credit, RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT);
    EXPECT_EQ(sqeStruct->header.rtStreamId, streamId);
    EXPECT_EQ(sqeStruct->eventId, eventId);
    EXPECT_EQ(sqeStruct->header.taskId, taskId);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneFlipPlaceHolderSqeV1)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t flipNum = 10;
    uint16_t taskId = 2;

    AddOneFlipPlaceHolderSqeV1(streamId, flipNum, taskId, sqe, &sqeType);

    EXPECT_EQ(sqeType, FLIP_PLACEHOLDER_SQE);
    rtStarsPlaceHolderSqe_t* sqeStruct = reinterpret_cast<rtStarsPlaceHolderSqe_t*>(sqe);
    EXPECT_EQ(sqeStruct->header.type, RT_STARS_SQE_TYPE_PLACE_HOLDER);
    EXPECT_EQ(sqeStruct->header.rtStreamId, streamId);
    EXPECT_EQ(sqeStruct->header.taskId, taskId);
    EXPECT_EQ(sqeStruct->header.blockDim, RT_TASK_TYPE_FLIP);
    EXPECT_EQ(sqeStruct->kernel_credit, RT_STARS_DEFAULT_KERNEL_CREDIT);
    EXPECT_EQ(sqeStruct->u.flip_task_info.flipNumReport, flipNum);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneCacheMemcpyPlaceHolderSqeV1)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    void* src = reinterpret_cast<void*>(0x1000UL);
    void* dst = reinterpret_cast<void*>(0x2000UL);
    uint8_t linkType = 0;
    uint32_t hcclQos = 3;

    AddOneCacheMemcpyPlaceHolderSqeV1(streamId, taskId, src, dst, linkType, sqe, &sqeType, hcclQos);

    EXPECT_EQ(sqeType, CACHE_MEMCPY_PLACEHOLDER_SQE);
    rtStarsPlaceHolderSqe_t* sqeStruct = reinterpret_cast<rtStarsPlaceHolderSqe_t*>(sqe);
    EXPECT_EQ(sqeStruct->header.type, RT_STARS_SQE_TYPE_PLACE_HOLDER);
    EXPECT_EQ(sqeStruct->header.rtStreamId, streamId);
    EXPECT_EQ(sqeStruct->header.taskId, taskId);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneCacheMemcpyPlaceHolderSqeV1_WithLinkSio)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    void* src = reinterpret_cast<void*>(0x1000UL);
    void* dst = reinterpret_cast<void*>(0x2000UL);
    uint8_t linkType = static_cast<uint8_t>(hccl::LinkType::LINK_SIO);
    uint32_t hcclQos = 10;

    AddOneCacheMemcpyPlaceHolderSqeV1(streamId, taskId, src, dst, linkType, sqe, &sqeType, hcclQos);

    EXPECT_EQ(sqeType, CACHE_MEMCPY_PLACEHOLDER_SQE);
    rtStarsPlaceHolderSqe_t* sqeStruct = reinterpret_cast<rtStarsPlaceHolderSqe_t*>(sqe);
    EXPECT_EQ(sqeStruct->u.cache_memcpy_task_info.qos, SDMA_QOS_DEFAULT);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneCacheNotifyWaitPlaceholderSqeV1)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    u64 notifyId = 100;
    dfx::DfxTimeOutConfig dfxTimeOutConfig = {};
    dfxTimeOutConfig.useCredit = true;
    dfxTimeOutConfig.sqeCreditTimeOut = 240;

    AddOneCacheNotifyWaitPlaceholderSqeV1(streamId, taskId, notifyId, sqe, &sqeType, dfxTimeOutConfig);

    EXPECT_EQ(sqeType, CACHE_NOTIFY_PLACEHOLDER_SQE);
    rtStarsPlaceHolderSqe_t* sqeStruct = reinterpret_cast<rtStarsPlaceHolderSqe_t*>(sqe);
    EXPECT_EQ(sqeStruct->u.cache_notify_task_info.is_wait, 1);
    EXPECT_EQ(sqeStruct->u.cache_notify_task_info.notify_id, notifyId);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneCacheNotifyRecordPlaceholderSqeV1)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    u64 notifyId = 100;

    AddOneCacheNotifyRecordPlaceholderSqeV1(streamId, taskId, notifyId, sqe, &sqeType);

    EXPECT_EQ(sqeType, CACHE_NOTIFY_PLACEHOLDER_SQE);
    rtStarsPlaceHolderSqe_t* sqeStruct = reinterpret_cast<rtStarsPlaceHolderSqe_t*>(sqe);
    EXPECT_EQ(sqeStruct->u.cache_notify_task_info.is_wait, 0);
    EXPECT_EQ(sqeStruct->u.cache_notify_task_info.notify_id, notifyId);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneCacheWriteValuePlaceholderSqeV1)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    u64 notifyWRAddr = 0x1000000UL;

    AddOneCacheWriteValuePlaceholderSqeV1(streamId, taskId, notifyWRAddr, sqe, &sqeType);

    EXPECT_EQ(sqeType, CACHE_WRITE_VALUE_PLACEHOLDER_SQE);
    rtStarsPlaceHolderSqe_t* sqeStruct = reinterpret_cast<rtStarsPlaceHolderSqe_t*>(sqe);
    EXPECT_EQ(
        sqeStruct->u.cache_write_value_task_info.write_addr_low, static_cast<uint32_t>(notifyWRAddr & MASK_32_BIT));
    EXPECT_EQ(
        sqeStruct->u.cache_write_value_task_info.write_addr_high,
        static_cast<uint32_t>((notifyWRAddr >> UINT32_BIT_NUM) & MASK_17_BIT));
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneCacheMemcpyRecordPlaceholderSqeV1_WithReduce)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    void* src = reinterpret_cast<void*>(0x1000UL);
    void* dst = reinterpret_cast<void*>(0x2000UL);
    uint32_t length = 256;
    aclDataType runtimeDataType = ACL_FLOAT16;
    aclrtReduceKind rtReduceOp = ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM;
    uint32_t partId = 1;
    uint32_t ssid = 2;
    uint32_t devId = 0;
    u64 overflowAddr = 0x3000UL;
    uint8_t linkType = 0;
    uint32_t hcclQos = 3;

    AddOneCacheMemcpyRecordPlaceholderSqeV1(
        streamId, taskId, src, length, runtimeDataType, rtReduceOp, dst, partId, ssid, devId, overflowAddr, linkType,
        sqe, &sqeType, hcclQos);

    EXPECT_EQ(sqeType, CACHE_MEMCPY_RECORD_PLACEHOLDER_SQE);
    rtStarsPlaceHolderSqe_t* sqeStruct = reinterpret_cast<rtStarsPlaceHolderSqe_t*>(sqe);
    EXPECT_NE(sqeStruct->u.cache_memcpy_record_task_info.opcode, 0);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_AddOneCacheMemcpyRecordPlaceholderSqeV1_WithLinkSio)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    void* src = reinterpret_cast<void*>(0x1000UL);
    void* dst = reinterpret_cast<void*>(0x2000UL);
    uint32_t length = 256;
    aclDataType runtimeDataType = ACL_FLOAT16;
    aclrtReduceKind rtReduceOp = ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM;
    uint32_t partId = 1;
    uint32_t ssid = 2;
    uint32_t devId = 0;
    u64 overflowAddr = 0x3000UL;
    uint8_t linkType = static_cast<uint8_t>(hccl::LinkType::LINK_SIO);
    uint32_t hcclQos = 10;

    AddOneCacheMemcpyRecordPlaceholderSqeV1(
        streamId, taskId, src, length, runtimeDataType, rtReduceOp, dst, partId, ssid, devId, overflowAddr, linkType,
        sqe, &sqeType, hcclQos);

    EXPECT_EQ(sqeType, CACHE_MEMCPY_RECORD_PLACEHOLDER_SQE);
    rtStarsPlaceHolderSqe_t* sqeStruct = reinterpret_cast<rtStarsPlaceHolderSqe_t*>(sqe);
    EXPECT_EQ(sqeStruct->u.cache_memcpy_record_task_info.qos, SDMA_QOS_DEFAULT);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_SetCachePlaceholderHeaderV1)
{
    uint8_t sqe[64] = {0};
    uint16_t streamId = 1;
    uint16_t taskId = 2;

    SetCachePlaceholderHeaderV1(streamId, taskId, sqe);

    rtStarsPlaceHolderSqe_t* sqeStruct = reinterpret_cast<rtStarsPlaceHolderSqe_t*>(sqe);
    EXPECT_EQ(sqeStruct->header.type, RT_STARS_SQE_TYPE_PLACE_HOLDER);
    EXPECT_EQ(sqeStruct->header.ie, 0U);
    EXPECT_EQ(sqeStruct->header.preP, 0U);
    EXPECT_EQ(sqeStruct->header.postP, 0U);
    EXPECT_EQ(sqeStruct->header.wrCqe, 0U);
    EXPECT_EQ(sqeStruct->header.blockDim, RT_TASK_TYPE_FLIP);
    EXPECT_EQ(sqeStruct->header.rtStreamId, streamId);
    EXPECT_EQ(sqeStruct->header.taskId, taskId);
    EXPECT_EQ(sqeStruct->kernel_credit, RT_STARS_DEFAULT_KERNEL_CREDIT);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_TranslateOpcode_WithAdd)
{
    uint8_t opCode = RT_STARS_MEMCPY_ASYNC_OP_KIND_ADD;
    uint8_t reduceType = 0;
    TranslateOpcode(opCode, reduceType);
    EXPECT_EQ(reduceType, HCCL_REDUCE_SUM);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_TranslateOpcode_WithMax)
{
    uint8_t opCode = RT_STARS_MEMCPY_ASYNC_OP_KIND_MAX;
    uint8_t reduceType = 0;
    TranslateOpcode(opCode, reduceType);
    EXPECT_EQ(reduceType, HCCL_REDUCE_MAX);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_TranslateOpcode_WithMin)
{
    uint8_t opCode = RT_STARS_MEMCPY_ASYNC_OP_KIND_MIN;
    uint8_t reduceType = 0;
    TranslateOpcode(opCode, reduceType);
    EXPECT_EQ(reduceType, HCCL_REDUCE_MIN);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_TranslateOpcode_WithCpy)
{
    uint8_t opCode = RT_STARS_MEMCPY_ASYNC_OP_KIND_CPY;
    uint8_t reduceType = 0;
    TranslateOpcode(opCode, reduceType);
    EXPECT_EQ(reduceType, HCCL_REDUCE_RESERVED);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_TranslateOpcode_WithEqual)
{
    uint8_t opCode = RT_STARS_MEMCPY_ASYNC_OP_KIND_EQUAL;
    uint8_t reduceType = 0;
    TranslateOpcode(opCode, reduceType);
    EXPECT_EQ(reduceType, HCCL_REDUCE_RESERVED);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_GetTimeOutValue_UseCredit)
{
    dfx::DfxTimeOutConfig dfxTimeOutConfig = {};
    dfxTimeOutConfig.useCredit = true;
    dfxTimeOutConfig.sqeCreditTimeOut = 240;

    auto result = GetTimeOutValue(dfxTimeOutConfig);
    EXPECT_EQ(result.first, 240UL);
    EXPECT_EQ(result.second, dfx::kTimeOutTimeInvalid);
}

TEST_F(RtsqInteract_Sqcqv1_UT, Ut_GetTimeOutValue_UseSoftSync)
{
    dfx::DfxTimeOutConfig dfxTimeOutConfig = {};
    dfxTimeOutConfig.useCredit = false;
    dfxTimeOutConfig.sqeTimeOutTimeOut = 960;

    auto result = GetTimeOutValue(dfxTimeOutConfig);
    EXPECT_EQ(result.first, dfx::kCreditTimeInvalid);
    EXPECT_EQ(result.second, 960UL);
}
