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
#define private public
#define protected public
#include "rtsq_a5.h"
#include "binary_stream.h"
#include "sqe.h"
#include "ascend_hal.h"
#include "drv_api_exception.h"
#undef private
#undef protected

using namespace Hccl;
class RtsqA5Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RtsqA5 tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RtsqA5 tests tear down." << std::endl; }

    virtual void SetUp()
    {
        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(&mockSq)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(0));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();

        std::cout << "A Test case in RtsqA5 SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RtsqA5 TearDown" << std::endl;
    }

    u32 fakedevPhyId = 0;
    u32 fakeStreamId = 1;
    u32 fakeSqId = 2;
    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
};

class IsRtsqQueueSpaceSufficientTest : public RtsqA5Test {
protected:
    void SetUp() override
    {
        RtsqA5Test::SetUp();
        // 初始化测试环境
        pendingSqeCnt = 5; // 假设pendingSqeCnt为5
    }
    u32 pendingSqeCnt; // 模拟pendingSqeCnt的值
};

TEST_F(IsRtsqQueueSpaceSufficientTest, Ut_IsRtsqQueueSpaceSufficient_When_HeadEqualTail_ExpectFalse)
{
    // 准备测试数据
    RtsqA5 fakeRtsqA5(fakedevPhyId, fakeStreamId, fakeSqId);
    fakeRtsqA5.pendingSqeCnt = pendingSqeCnt;
    fakeRtsqA5.sqDepth_ = pendingSqeCnt + 1; // 设置availableSpace为pendingSqeCnt + 1
    fakeRtsqA5.sqHead_ = 0;                  // 设置sqHead_与sqTail为0，使GetTailToHeadDist返回sqDepth
    fakeRtsqA5.sqTail_ = 0;                  // 以上四个变量控制GetTailToHeadDist方法的输出

    // 模拟QuerySqHead方法
    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(fakeRtsqA5.sqHead_));

    // 执行测试
    bool result = fakeRtsqA5.IsRtsqQueueSpaceSufficient();

    // 验证结果
    EXPECT_FALSE(result);
}

TEST_F(IsRtsqQueueSpaceSufficientTest, Ut_IsRtsqQueueSpaceSufficient_When_HeadEqualTail_ExpectTrue)
{
    // 准备测试数据
    RtsqA5 fakeRtsqA5(fakedevPhyId, fakeStreamId, fakeSqId);
    fakeRtsqA5.pendingSqeCnt = pendingSqeCnt;
    fakeRtsqA5.sqDepth_ = pendingSqeCnt + 2; // 设置availableSpace为pendingSqeCnt + 2
    fakeRtsqA5.sqHead_ = 0;                  // 设置sqHead_与sqTail为0，使GetTailToHeadDist返回sqDepth
    fakeRtsqA5.sqTail_ = 0;                  // 以上四个变量控制GetTailToHeadDist方法的输出

    // 模拟QuerySqHead方法
    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(fakeRtsqA5.sqHead_));

    // 执行测试
    bool result = fakeRtsqA5.IsRtsqQueueSpaceSufficient();

    // 验证结果
    EXPECT_TRUE(result);
}

TEST_F(RtsqA5Test, launch_task_no_loop_back)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.LaunchTask(); // 没有SQE ，直接返回

    u32 oldTail = 0;
    u32 oldHead = 0;
    rtsq.sqTail_ = oldTail;
    rtsq.sqHead_ = oldHead;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;

    rtsq.RefreshInfo();
    u32 newTail = (rtsq.sqTail_ + rtsq.pendingSqeCnt) % rtsq.sqDepth_;
    rtsq.LaunchTask();

    EXPECT_EQ(rtsq.sqTail_, newTail);
}

TEST_F(RtsqA5Test, launch_task_with_loop_back)
{
    memset_s(mockSq, sizeof(mockSq), 0, sizeof(mockSq));
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 oldTail = AC_SQE_MAX_CNT - 1;
    u32 oldHead = AC_SQE_MAX_CNT - 2;
    rtsq.sqTail_ = oldTail;
    rtsq.sqHead_ = oldHead;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;

    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(oldHead));
    rtsq.RefreshInfo();
    rtsq.RefreshInfo();
    u32 newTail = (rtsq.sqTail_ + rtsq.pendingSqeCnt) % rtsq.sqDepth_;
    rtsq.LaunchTask();

    EXPECT_EQ(rtsq.sqTail_, newTail);
}

TEST_F(RtsqA5Test, notify_wait)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 fakeNotifyId = 0;
    rtsq.NotifyWait(fakeNotifyId);
}

TEST_F(RtsqA5Test, notify_record_local)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 fakeNotifyId = 0;
    rtsq.NotifyRecordLoc(fakeNotifyId);
}

TEST_F(RtsqA5Test, cnt_1ton_notify_wait)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 fakeNotifyId = 0;
    u32 fakeValue = 1;
    rtsq.Cnt1toNNotifyWait(fakeNotifyId, fakeValue);
}

TEST_F(RtsqA5Test, cnt_1ton_notify_record)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 fakeNotifyId = 0;
    u32 fakeValue = 1;
    rtsq.Cnt1toNNotifyRecord(fakeNotifyId, fakeValue);
}

TEST_F(RtsqA5Test, cnt_nto1_notify_wait)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 fakeNotifyId = 0;
    u32 fakeValue = 1;
    rtsq.CntNto1NotifyWait(fakeNotifyId, fakeValue);
}

TEST_F(RtsqA5Test, cnt_nto1_notify_record)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 fakeNotifyId = 0;
    u32 fakeValue = 1;
    rtsq.CntNto1NotifyRecord(fakeNotifyId, fakeValue);
}

TEST_F(RtsqA5Test, sdma_copy)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u64 srcAddr = 0x100;
    u64 dstAddr = 0x200;
    u32 size = 0x300;
    u32 partId = 0x400;
    rtsq.SdmaCopy(srcAddr, dstAddr, size, partId);
}

TEST_F(RtsqA5Test, sdma_reduce)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u64 srcAddr = 0x100;
    u64 dstAddr = 0x200;
    u32 size = 0x300;
    u32 partId = 0x400;
    ReduceIn reduceIn(DataType::INT8, ReduceOp::MAX);
    rtsq.SdmaReduce(srcAddr, dstAddr, size, partId, reduceIn);
}

TEST_F(RtsqA5Test, sdma_reduce_failed)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u64 srcAddr = 0x100;
    u64 dstAddr = 0x200;
    u32 size = 0x300;
    u32 partId = 0x400;
    ReduceIn reduceIn(DataType::UINT8, ReduceOp::MAX);

    EXPECT_THROW(rtsq.SdmaReduce(srcAddr, dstAddr, size, partId, reduceIn), Hccl::InternalException);
}

TEST_F(RtsqA5Test, ub_db_send)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 dieId = 0;
    u32 funcId = 0;
    UbJettyLiteId jettyId(18, 18, 18);
    u32 piVal = 0;
    rtsq.UbDbSend(jettyId, piVal);
}

TEST_F(RtsqA5Test, ub_direct_send)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 dieId = 0;
    u32 funcId = 0;
    UbJettyLiteId jettyId(18, 18, 18);
    u32 dwqeSize = 128;
    u8 dwqe[128]{0};
    rtsq.UbDirectSend(jettyId, dwqeSize, dwqe);
}

TEST_F(RtsqA5Test, ub_write_value)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u64 dbAddr = 0;
    u32 piVal = 0;
    rtsq.UbWriteValue(dbAddr, piVal);
}

TEST_F(RtsqA5Test, query_sq_status_by_type)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    GlobalMockObject::reset();
    MOCKER(halSqCqQuery).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(rtsq.QuerySqStatusByType(RtsqBase::QueryDrvSqCqPtopType::CQE_STATUS), DrvApiException);
}

TEST_F(RtsqA5Test, query_sq_base_addr)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    GlobalMockObject::reset();
    MOCKER(halSqCqQuery).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(rtsq.QuerySqBaseAddr(), DrvApiException);
}

TEST_F(RtsqA5Test, config_sq_status_by_type)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    GlobalMockObject::reset();
    MOCKER(halSqCqConfig).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(rtsq.ConfigSqStatusByType(RtsqBase::ConfigDrvSqCqPtopType::TAIL, 1), DrvApiException);
}
