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
#include "hccl_sqe_v82.h"
#include "driver/ascend_hal.h"

using namespace Hccl;

class HcclSqeTestV82 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclSqeTestV82 SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "HcclSqeTestV82 TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in HcclSqeTestV82 SetUp" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in HcclSqeTestV82 TearDown" << std::endl;
    }
};

TEST_F(HcclSqeTestV82, hccl_ub_db_send_sqe)
{
    // Given
    HcclUBDmaDBSqe ubDmaDBsqe;
    ubDmaDBsqe.Config(1, 2, 3, 4, 5, 6);
    // when
    Rt91095StarsUbdmaDBmodeSqe* sqe = reinterpret_cast<Rt91095StarsUbdmaDBmodeSqe*>(ubDmaDBsqe.GetSqe());

    // then
    EXPECT_EQ(sqe->header.rtStreamId, 1);
    EXPECT_EQ(sqe->header.taskId, 2);
    EXPECT_EQ(sqe->doorbellNum, 1);
}

TEST_F(HcclSqeTestV82, hccl_UBNotifyWaitSqe)
{
    // Given
    HcclNotifyWaitSqe notifyWaitSqe;
    u16 streamId = 1;
    u16 taskId = 0;
    u64 notifyId = 1;

    HcclUBNotifyWaitSqe ubnotifyWaitSqe;
    ubnotifyWaitSqe.Config(streamId, taskId, notifyId);
    // when
    Rt91095StarsNotifySqe* sqe = reinterpret_cast<Rt91095StarsNotifySqe*>(ubnotifyWaitSqe.GetSqe());

    // then
    EXPECT_EQ(sqe->header.rtStreamId, streamId);
    EXPECT_EQ(sqe->notifyId, notifyId);
    EXPECT_EQ(sqe->header.taskId, taskId);
}

TEST_F(HcclSqeTestV82, hccl_ub_Notify_Record_Sqe)
{
    // Given
    HcclUBNotifyRecordSqe notifyRecordSqe;
    u16 streamId = 1;
    u16 taskId = 0;
    u64 notifyId = 1;

    notifyRecordSqe.Config(streamId, taskId, notifyId);
    // when
    Rt91095StarsNotifySqe* sqe = reinterpret_cast<Rt91095StarsNotifySqe*>(notifyRecordSqe.GetSqe());

    // then
    EXPECT_EQ(sqe->header.rtStreamId, streamId);
    EXPECT_EQ(sqe->notifyId, notifyId);
    EXPECT_EQ(sqe->header.taskId, taskId);
}

TEST_F(HcclSqeTestV82, hccl_ub_cnt_notify_nto1_Record_Sqe)
{
    // Given
    HcclUBCntNotifyNto1RecordSqe cntNotifyNto1RecordSqe;
    u16 streamId = 1;
    u16 taskId = 0;
    u64 notifyId = 1;
    u32 value = 1;

    cntNotifyNto1RecordSqe.Config(streamId, taskId, notifyId, value);
    // when
    Rt91095StarsNotifySqe* sqe = reinterpret_cast<Rt91095StarsNotifySqe*>(cntNotifyNto1RecordSqe.GetSqe());

    // then
    EXPECT_EQ(sqe->header.rtStreamId, streamId);
    EXPECT_EQ(sqe->notifyId, notifyId);
    EXPECT_EQ(sqe->header.taskId, taskId);
    EXPECT_EQ(sqe->cntValue, value);
}

TEST_F(HcclSqeTestV82, hccl_ub_cnt_notify_nto1_Wait_Sqe)
{
    // Given
    HcclUBCntNotifyNto1WaitSqe cntNotifyNto1WaitSqe;
    u16 streamId = 1;
    u16 taskId = 0;
    u64 notifyId = 1;
    u32 value = 1;

    cntNotifyNto1WaitSqe.Config(streamId, taskId, notifyId, value);
    // when
    Rt91095StarsNotifySqe* sqe = reinterpret_cast<Rt91095StarsNotifySqe*>(cntNotifyNto1WaitSqe.GetSqe());

    // then
    EXPECT_EQ(sqe->header.rtStreamId, streamId);
    EXPECT_EQ(sqe->notifyId, notifyId);
    EXPECT_EQ(sqe->header.taskId, taskId);
    EXPECT_EQ(sqe->cntValue, value);
}

TEST_F(HcclSqeTestV82, hccl_ub_cnt_notify_1ton_Record_Sqe)
{
    // Given
    HcclUBCntNotify1toNRecordSqe cntNotify1toNRecordSqe;
    u16 streamId = 1;
    u16 taskId = 0;
    u64 notifyId = 1;
    u32 value = 1;

    cntNotify1toNRecordSqe.Config(streamId, taskId, notifyId, value);
    // when
    Rt91095StarsNotifySqe* sqe = reinterpret_cast<Rt91095StarsNotifySqe*>(cntNotify1toNRecordSqe.GetSqe());

    // then
    EXPECT_EQ(sqe->header.rtStreamId, streamId);
    EXPECT_EQ(sqe->notifyId, notifyId);
    EXPECT_EQ(sqe->header.taskId, taskId);
    EXPECT_EQ(sqe->cntValue, value);
}

TEST_F(HcclSqeTestV82, hccl_ub_cnt_notify_1ton_Wait_Sqe)
{
    // Given
    HcclUBCntNotify1toNWaitSqe cntNotify1toNWaitSqe;
    u16 streamId = 1;
    u16 taskId = 0;
    u64 notifyId = 1;
    u32 value = 1;

    cntNotify1toNWaitSqe.Config(streamId, taskId, notifyId, value);
    // when
    Rt91095StarsNotifySqe* sqe = reinterpret_cast<Rt91095StarsNotifySqe*>(cntNotify1toNWaitSqe.GetSqe());

    // then
    EXPECT_EQ(sqe->header.rtStreamId, streamId);
    EXPECT_EQ(sqe->notifyId, notifyId);
    EXPECT_EQ(sqe->header.taskId, taskId);
    EXPECT_EQ(sqe->cntValue, value);
}

TEST_F(HcclSqeTestV82, hccl_ub_memcpy_sqe)
{
    // given

    HcclUBMemcpySqe ubMemcpySqe;
    u16 streamId = 1;
    u16 taskId = 0;
    RtDataType rtDataType = RtDataType::RT_DATA_TYPE_INT16;
    RtReduceKind rtReduceOp = RtReduceKind::RT_MEMCPY_SDMA_AUTOMATIC_ADD;
    u64 count = 0;
    u64 src = 0x01;
    u64 dst = 0x0f;
    u32 partId = 0;

    // when
    ubMemcpySqe.Config(streamId, taskId, rtDataType, rtReduceOp, count, &src, &dst, partId);
    Rt91095StarsMemcpySqe* sqe = reinterpret_cast<Rt91095StarsMemcpySqe*>(ubMemcpySqe.GetSqe());

    // then
    EXPECT_EQ(sqe->header.rtStreamId, streamId);
    EXPECT_EQ(sqe->header.taskId, taskId);
    EXPECT_EQ(sqe->u.strideMode0.srcAddrLow, 1);
    EXPECT_EQ(sqe->u.strideMode0.srcAddrHigh, 0);
    EXPECT_EQ(sqe->u.strideMode0.dstAddrLow, 0xf);
    EXPECT_EQ(sqe->u.strideMode0.dstAddrHigh, 0);
    EXPECT_EQ(sqe->opcode, 0x11);
}
