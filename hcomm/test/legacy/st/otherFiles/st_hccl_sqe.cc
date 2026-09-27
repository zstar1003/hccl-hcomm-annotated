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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "hccl_sqe.h"
#include "driver/ascend_hal.h"

using namespace Hccl;

class HcclSqeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclSqeTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "HcclSqeTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in HcclSqeTest SetUp" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in HcclSqeTest TearDown" << std::endl;
    }
};

TEST_F(HcclSqeTest, hccl_notify_wait_sqe_create_instance)
{
    // Given
    HcclNotifyWaitSqe notifyWaitSqe;

    // when
    RtStarsNotifySqe* sqe = reinterpret_cast<RtStarsNotifySqe*>(notifyWaitSqe.GetSqe());

    // then
    EXPECT_EQ(sqe->header.type, static_cast<u8>(RtStarsSqeType::RT_STARS_SQE_TYPE_NOTIFY_WAIT));
    EXPECT_EQ(sqe->kernelCredit, RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT);
}

TEST_F(HcclSqeTest, hccl_notify_wait_sqe_config)
{
    // Given
    HcclNotifyWaitSqe notifyWaitSqe;
    u16 streamId = 1;
    u16 taskId = 0;
    u64 notifyId = 1;

    // when
    notifyWaitSqe.Config(streamId, taskId, notifyId);
    RtStarsNotifySqe* sqe = reinterpret_cast<RtStarsNotifySqe*>(notifyWaitSqe.GetSqe());

    // then
    EXPECT_EQ(sqe->header.rtStreamId, streamId);
    EXPECT_EQ(sqe->camelBack, notifyId);
    EXPECT_EQ(sqe->header.taskId, taskId);
}

TEST_F(HcclSqeTest, hccl_write_value_sqe_create_instance)
{
    // given
    HcclWriteValueSqe writeValueSqe;

    // when
    RtStarsWriteValueSqe* sqe = reinterpret_cast<RtStarsWriteValueSqe*>(writeValueSqe.GetSqe());

    // then
    EXPECT_EQ(sqe->header.type, static_cast<u8>(RtStarsSqeType::RT_STARS_SQE_TYPE_WRITE_VALUE));
    EXPECT_EQ(sqe->kernelCredit, RT_STARS_DEFAULT_KERNEL_CREDIT);
    EXPECT_EQ(sqe->awsize, static_cast<u8>(RtStarsWriteValueSizeType::RT_STARS_WRITE_VALUE_SIZE_TYPE_32BIT));
    EXPECT_EQ(sqe->writeValuePart0, 1U);
    EXPECT_EQ(
        sqe->subType,
        static_cast<u8>(RtStarsWriteValueSubType::RT_STARS_WRITE_VALUE_SUB_TYPE_NOTIFY_RECORD_IPC_NO_PCIE));
}

TEST_F(HcclSqeTest, hccl_write_value_sqe_config)
{
    // given
    HcclWriteValueSqe writeValueSqe;
    u16 streamId = 1;
    u16 taskId = 0;
    u64 notifyWRAddr = 0x0000000f00000001;

    // when
    writeValueSqe.Config(streamId, taskId, notifyWRAddr);
    RtStarsWriteValueSqe* sqe = reinterpret_cast<RtStarsWriteValueSqe*>(writeValueSqe.GetSqe());

    // then
    EXPECT_EQ(sqe->header.rtStreamId, streamId);
    EXPECT_EQ(sqe->header.taskId, taskId);
    EXPECT_EQ(sqe->writeAddrLow, GetAddrLow(notifyWRAddr));
    EXPECT_EQ(sqe->writeAddrHigh, GetAddrHigh(notifyWRAddr) & MASK_17_BIT);
}

TEST_F(HcclSqeTest, hccl_sdma_sqe_create_instance)
{
    // given
    HcclSdmaSqe sdmaSqe;

    // when
    RtStarsMemcpyAsyncSqe* sqe = reinterpret_cast<RtStarsMemcpyAsyncSqe*>(sdmaSqe.GetSqe());

    // then
    EXPECT_EQ(sqe->header.type, static_cast<u8>(RtStarsSqeType::RT_STARS_SQE_TYPE_SDMA));
    EXPECT_EQ(sqe->kernelCredit, RT_STARS_DEFAULT_KERNEL_CREDIT);
    EXPECT_EQ(sqe->sssv, 1U);
    EXPECT_EQ(sqe->dssv, 1U);
    EXPECT_EQ(sqe->sns, 1U);
    EXPECT_EQ(sqe->dns, 1U);
    EXPECT_EQ(sqe->qos, 6);
}

TEST_F(HcclSqeTest, hccl_sdma_sqe_config)
{
    // given
    HcclSdmaSqe sdmaSqe;
    u16 streamId = 1;
    u16 taskId = 0;
    u64 src = 0x01;
    u64 dst = 0x0f;
    u32 length = 10;
    RtDataType dataType = RtDataType::RT_DATA_TYPE_FP32;
    RtReduceKind reduceOp = RtReduceKind::RT_MEMCPY_SDMA_AUTOMATIC_ADD;
    u32 partId = 0;

    // when
    sdmaSqe.Config(streamId, taskId, src, length, dataType, reduceOp, dst, partId);
    RtStarsMemcpyAsyncSqe* sqe = reinterpret_cast<RtStarsMemcpyAsyncSqe*>(sdmaSqe.GetSqe());

    // then
    EXPECT_EQ(sqe->header.rtStreamId, streamId);
    EXPECT_EQ(sqe->header.taskId, taskId);
    EXPECT_EQ(sqe->opcode, 0x71);
    EXPECT_EQ(sqe->length, length);
    EXPECT_EQ(sqe->srcAddrLow, GetAddrLow(src));
    EXPECT_EQ(sqe->srcAddrHigh, GetAddrHigh(src));
    EXPECT_EQ(sqe->dstAddrLow, GetAddrLow(dst));
    EXPECT_EQ(sqe->dstAddrHigh, GetAddrHigh(dst));
    EXPECT_EQ(sqe->partid, partId);
}

TEST_F(HcclSqeTest, get_addr_low)
{
    // given
    u64 addr = 0x0000000012345678;

    // when
    u32 addrLow = GetAddrLow(addr);

    // then
    EXPECT_EQ(addrLow, 0x12345678);
}

TEST_F(HcclSqeTest, get_addr_high)
{
    // given
    u64 addr = 0x1234567800000000;

    // when
    u32 addrHigh = GetAddrHigh(addr);

    // then
    EXPECT_EQ(addrHigh, 0x12345678);
}

TEST_F(HcclSqeTest, should_return_valid_notify_sqe_when_creating_instance)
{
    // given
    HcclNotifyRecordSqe notifyRecordSqe;

    // when
    RtStarsNotifySqe* sqe = reinterpret_cast<RtStarsNotifySqe*>(notifyRecordSqe.GetSqe());

    // then
    EXPECT_EQ(sqe->header.type, static_cast<u8>(RtStarsSqeType::RT_STARS_SQE_TYPE_NOTIFY_RECORD));
    EXPECT_EQ(sqe->kernelCredit, RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT);
}

TEST_F(HcclSqeTest, should_return_valid_notify_sqe_when_Config)
{
    // Given
    HcclNotifyRecordSqe notifyRecordSqe;
    u16 streamId = 1;
    u16 taskId = 0;
    u64 notifyId = 1;

    // when
    notifyRecordSqe.Config(streamId, taskId, notifyId);
    RtStarsNotifySqe* sqe = reinterpret_cast<RtStarsNotifySqe*>(notifyRecordSqe.GetSqe());

    // then
    EXPECT_EQ(sqe->header.rtStreamId, streamId);
    EXPECT_EQ(sqe->camelBack, notifyId);
    EXPECT_EQ(sqe->header.taskId, taskId);
}
