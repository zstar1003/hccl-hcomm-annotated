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
#include "aicpu/aicpu_hccl_sqcqv2.h"
#include "hccl_common.h"
#include "dispatcher_task_types.h"

using namespace std;
using namespace hccl;

class RtsqInteract_Sqcqv2_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RtsqInteract_Sqcqv2_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "RtsqInteract_Sqcqv2_UT TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "RtsqInteract_Sqcqv2_UT Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "RtsqInteract_Sqcqv2_UT Test TearDown" << std::endl;
    }
};

TEST_F(RtsqInteract_Sqcqv2_UT, Ut_AddOneNotifyWaitSqeV2)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    u64 notifyId = 100;
    dfx::DfxTimeOutConfig dfxTimeOutConfig = {};

    AddOneNotifyWaitSqeV2(streamId, taskId, notifyId, sqe, &sqeType, dfxTimeOutConfig);

    EXPECT_EQ(sqeType, NOTIFY_SQE_V2);
    rtStarsNotifySqeV2_t* sqeStruct = reinterpret_cast<rtStarsNotifySqeV2_t*>(sqe);
    EXPECT_EQ(sqeStruct->header.type, RT_HW_STARS_SQE_TYPE_NOTIFY_WAIT);
    EXPECT_EQ(sqeStruct->header.rt_stream_id, streamId);
    EXPECT_EQ(sqeStruct->header.task_id, taskId);
    EXPECT_EQ(sqeStruct->notify_id, notifyId);
    EXPECT_EQ(sqeStruct->kernel_credit, RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT);
}

TEST_F(RtsqInteract_Sqcqv2_UT, Ut_AddOneRecordSqeV2)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    u64 notifyId = 100;

    AddOneRecordSqeV2(streamId, taskId, notifyId, sqe, &sqeType);

    EXPECT_EQ(sqeType, NOTIFY_SQE_V2);
    rtStarsNotifySqeV2_t* sqeStruct = reinterpret_cast<rtStarsNotifySqeV2_t*>(sqe);
    EXPECT_EQ(sqeStruct->header.type, RT_HW_STARS_SQE_TYPE_NOTIFY_RECORD);
    EXPECT_EQ(sqeStruct->header.rt_stream_id, streamId);
    EXPECT_EQ(sqeStruct->header.task_id, taskId);
    EXPECT_EQ(sqeStruct->notify_id, notifyId);
    EXPECT_EQ(sqeStruct->kernel_credit, RT_STARS_DEFAULT_KERNEL_CREDIT);
}

TEST_F(RtsqInteract_Sqcqv2_UT, Ut_AddOneWriteValueRecordSqeV2)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    u64 notifyWRAddr = 0x1000000UL;

    AddOneWriteValueRecordSqeV2(streamId, taskId, notifyWRAddr, sqe, &sqeType);

    EXPECT_EQ(sqeType, WRITE_VALUE_SQE_V2);
    rtStarsWriteValueSqeV2_t* sqeStruct = reinterpret_cast<rtStarsWriteValueSqeV2_t*>(sqe);
    EXPECT_EQ(sqeStruct->header.type, RT_HW_STARS_SQE_TYPE_WRITE_VALUE);
    EXPECT_EQ(sqeStruct->header.rt_stream_id, streamId);
    EXPECT_EQ(sqeStruct->header.task_id, taskId);
    EXPECT_EQ(sqeStruct->awsize, RT_STARS_WRITE_VALUE_SIZE_TYPE_64BIT);
    EXPECT_EQ(sqeStruct->awprot, 2);
    EXPECT_EQ(sqeStruct->write_val[0], 1U);
    EXPECT_EQ(sqeStruct->reg_addr_low, static_cast<uint32_t>(notifyWRAddr & MASK_32_BIT));
    EXPECT_EQ(sqeStruct->reg_addr_high, static_cast<uint16_t>(notifyWRAddr >> UINT32_BIT_NUM));
}

TEST_F(RtsqInteract_Sqcqv2_UT, Ut_AddOneMemcpySqeV2_WithReduce)
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

    AddOneMemcpySqeV2(
        streamId, taskId, src, length, runtimeDataType, rtReduceOp, dst, partId, ssid, devId, overflowAddr, linkType,
        sqe, &sqeType, hcclQos);

    EXPECT_EQ(sqeType, MEMCPY_ASYNC_SQE_V2);
    rtStarsMemcpyAsyncSqeV2_t* sqeStruct = reinterpret_cast<rtStarsMemcpyAsyncSqeV2_t*>(sqe);
    EXPECT_EQ(sqeStruct->type, RT_HW_STARS_SQE_TYPE_SDMA);
    EXPECT_EQ(sqeStruct->rt_stream_id, streamId);
    EXPECT_EQ(sqeStruct->task_id, taskId);
    EXPECT_EQ(sqeStruct->length, length);
    EXPECT_EQ(sqeStruct->kernel_credit, RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT);
    EXPECT_NE(sqeStruct->opcode, 0);
}

TEST_F(RtsqInteract_Sqcqv2_UT, Ut_AddOneMemcpySqeV2_WithoutReduce)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    void* src = reinterpret_cast<void*>(0x1000UL);
    void* dst = reinterpret_cast<void*>(0x2000UL);
    uint32_t length = 256;
    aclDataType runtimeDataType = ACL_FLOAT16;
    aclrtReduceKind rtReduceOp = ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX;
    uint32_t partId = 1;
    uint32_t ssid = 2;
    uint32_t devId = 1;
    u64 overflowAddr = 0x3000UL;
    uint8_t linkType = 0;
    uint32_t hcclQos = 3;

    AddOneMemcpySqeV2(
        streamId, taskId, src, length, runtimeDataType, rtReduceOp, dst, partId, ssid, devId, overflowAddr, linkType,
        sqe, &sqeType, hcclQos);

    EXPECT_EQ(sqeType, MEMCPY_ASYNC_SQE_V2);
    rtStarsMemcpyAsyncSqeV2_t* sqeStruct = reinterpret_cast<rtStarsMemcpyAsyncSqeV2_t*>(sqe);
    EXPECT_EQ(sqeStruct->type, RT_HW_STARS_SQE_TYPE_SDMA);
    EXPECT_EQ(sqeStruct->rt_stream_id, streamId);
}

TEST_F(RtsqInteract_Sqcqv2_UT, Ut_AddOneMemcpySqeV2_WithLinkSio)
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

    AddOneMemcpySqeV2(
        streamId, taskId, src, length, runtimeDataType, rtReduceOp, dst, partId, ssid, devId, overflowAddr, linkType,
        sqe, &sqeType, hcclQos);

    EXPECT_EQ(sqeType, MEMCPY_ASYNC_SQE_V2);
    rtStarsMemcpyAsyncSqeV2_t* sqeStruct = reinterpret_cast<rtStarsMemcpyAsyncSqeV2_t*>(sqe);
    EXPECT_EQ(sqeStruct->qos, SDMA_QOS_DEFAULT);
}

TEST_F(RtsqInteract_Sqcqv2_UT, Ut_AddOneEventResetSqeV2)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    int32_t eventId = 3;
    int64_t phyChipId = 0;
    int64_t phyDieId = 0;
    u64 addr = 0x1000000UL;

    AddOneEventResetSqeV2(streamId, eventId, taskId, phyChipId, phyDieId, addr, sqe, &sqeType);

    EXPECT_EQ(sqeType, WRITE_VALUE_SQE_V2);
    rtStarsWriteValueSqeV2_t* sqeStruct = reinterpret_cast<rtStarsWriteValueSqeV2_t*>(sqe);
    EXPECT_EQ(sqeStruct->header.type, RT_STARS_SQE_TYPE_WRITE_VALUE);
    EXPECT_EQ(sqeStruct->header.rt_stream_id, streamId);
    EXPECT_EQ(sqeStruct->header.task_id, taskId);
    EXPECT_EQ(sqeStruct->awprot, 0x2);
    EXPECT_EQ(sqeStruct->awsize, RT_STARS_WRITE_VALUE_SIZE_TYPE_64BIT);
    EXPECT_EQ(sqeStruct->reg_addr_low, static_cast<uint32_t>(addr & MASK_32_BIT));
    EXPECT_EQ(sqeStruct->reg_addr_high, static_cast<uint32_t>((addr >> UINT32_BIT_NUM) & MASK_17_BIT));
}

TEST_F(RtsqInteract_Sqcqv2_UT, Ut_AddOneEventRecordSqeV2)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    int32_t eventId = 3;

    AddOneEventRecordSqeV2(streamId, eventId, taskId, sqe, &sqeType);

    EXPECT_EQ(sqeType, EVENT_SQE_V2);
    rtStarsEventSqeV2_t* sqeStruct = reinterpret_cast<rtStarsEventSqeV2_t*>(sqe);
    EXPECT_EQ(sqeStruct->type, RT_STARS_SQE_TYPE_EVENT_RECORD);
    EXPECT_EQ(sqeStruct->kernel_credit, RT_STARS_DEFAULT_KERNEL_CREDIT);
    EXPECT_EQ(sqeStruct->rt_stream_id, streamId);
    EXPECT_EQ(sqeStruct->event_id, eventId);
    EXPECT_EQ(sqeStruct->task_id, taskId);
}

TEST_F(RtsqInteract_Sqcqv2_UT, Ut_AddOneEventWaitSqeV2)
{
    uint8_t sqe[64] = {0};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    int32_t eventId = 3;

    AddOneEventWaitSqeV2(streamId, eventId, taskId, sqe, &sqeType);

    EXPECT_EQ(sqeType, EVENT_SQE_V2);
    rtStarsEventSqeV2_t* sqeStruct = reinterpret_cast<rtStarsEventSqeV2_t*>(sqe);
    EXPECT_EQ(sqeStruct->type, RT_STARS_SQE_TYPE_EVENT_WAIT);
    EXPECT_EQ(sqeStruct->kernel_credit, RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT);
    EXPECT_EQ(sqeStruct->rt_stream_id, streamId);
    EXPECT_EQ(sqeStruct->event_id, eventId);
    EXPECT_EQ(sqeStruct->task_id, taskId);
}
