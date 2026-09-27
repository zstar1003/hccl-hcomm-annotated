/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <thread>
#include <vector>
#include <algorithm>

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#define private public
#define protected public
#include "rtsq_a5.h"
#include "binary_stream.h"
#include "config_plf_log_v2.h"
#include "env_config_v2.h"
#include "sqe.h"
#include "ascend_hal.h"
#include "drv_api_exception.h"
#include "rtsq_base.h"
#include "internal_exception.h"
#include "sqe_build_a5.h"
#include "adapter_rts_common.h"
#undef protected
#undef private

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
        MOCKER_CPP(&RtsqBase::QuerySqDepth)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(static_cast<u32>(AC_SQE_MAX_CNT)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(1)));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();

        std::cout << "A Test case in RtsqA5 SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        SetPlfDebugConfigValue(0);
        // 重置SQE profiling置位开关，防止用例异常路径(THROW)泄漏状态污染后续用例
        Hccl::SetSqeProfilingEnabled(false);
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

TEST_F(RtsqA5Test, launch_task_should_dump_generated_sqes_when_task_debug_enabled)
{
    SetPlfDebugConfigValue(PLF_TASK);
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(AC_SQE_MAX_CNT - 1));

    rtsq.NotifyWait(0);
    rtsq.SdmaCopy(0x100, 0x200, 0x300, 0x400);
    rtsq.UbDbSend(UbJettyLiteId(18, 18, 18), 0);

    EXPECT_NO_THROW(rtsq.LaunchTask());
}

TEST_F(RtsqA5Test, notify_wait)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 fakeNotifyId = 0;
    rtsq.NotifyWait(fakeNotifyId);
}

TEST_F(RtsqA5Test, notify_wait_with_timeout)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    u32 fakeNotifyId = 0;
    u32 fakeTimeout = 68;
    rtsq.NotifyWait(fakeNotifyId, fakeTimeout);
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
    EXPECT_THROW(rtsq.QuerySqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_CQE_STATUS), DrvApiException);
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
    EXPECT_THROW(rtsq.ConfigSqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_TAIL, 1), DrvApiException);
}

TEST_F(RtsqA5Test, Ut_CopyLocBufToSq_THROW)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(2));
    rtsq.sqTail_ = 6;
    rtsq.sqDepth_ = 16;
    rtsq.pendingSqeCnt = 8;
    MOCKER(memcpy_sOptAsm).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(rtsq.CopySqeBufToSq(rtsq.locBuf), InternalException);

    rtsq.pendingSqeCnt = 11;
    MOCKER(memcpy_sOptAsm).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(rtsq.CopySqeBufToSq(rtsq.locBuf), InternalException);

    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(10));
    rtsq.sqTail_ = 1;
    rtsq.sqDepth_ = 16;
    rtsq.pendingSqeCnt = 8;
    MOCKER(memcpy_sOptAsm).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(rtsq.CopySqeBufToSq(rtsq.locBuf), InternalException);
}

TEST_F(RtsqA5Test, Ut_TryLaunchTask_When_PendingSqeCnt_IsZero_Expect_NoAction)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 0;

    rtsq.TryLaunchTask();

    EXPECT_EQ(rtsq.pendingSqeCnt, 0);
}

TEST_F(RtsqA5Test, Ut_TryLaunchTask_When_HasEnoughSpace_Expect_LaunchTask)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);

    rtsq.sqTail_ = 0;
    rtsq.sqHead_ = 0;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;
    rtsq.pendingSqeCnt = 5;

    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(10));

    rtsq.TryLaunchTask();

    EXPECT_EQ(rtsq.pendingSqeCnt, 0);
}

TEST_F(RtsqA5Test, UT_CheckLaunchTaskStatus_CallbackReturnSuspending_Expect_PendingSqeCntResetAndNoThrow)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqTail_ = 8;
    rtsq.sqHead_ = 16;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;
    rtsq.pendingSqeCnt = 16;
    rtsq.sqFullTimeout_ = 10;

    auto startTime = std::chrono::steady_clock::now();
    auto curTime = startTime + std::chrono::seconds(5);
    rtsq.checkExecStatusCallback_ = [](bool isTimeout) {
        return HCCL_E_SUSPENDING;
    };

    EXPECT_NO_THROW(rtsq.CheckLaunchTaskStatus(startTime, curTime));
    EXPECT_EQ(rtsq.pendingSqeCnt, 0);
}

TEST_F(RtsqA5Test, UT_CheckLaunchTaskStatus_CallbackReturnSuspendingWithTimeout_Expect_NoThrow)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.sqTail_ = 8;
    rtsq.sqHead_ = 16;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;
    rtsq.pendingSqeCnt = 16;
    rtsq.sqFullTimeout_ = 1;

    auto startTime = std::chrono::steady_clock::now() - std::chrono::seconds(10);
    auto curTime = std::chrono::steady_clock::now();
    rtsq.checkExecStatusCallback_ = [](bool isTimeout) {
        return HCCL_E_SUSPENDING;
    };

    EXPECT_NO_THROW(rtsq.CheckLaunchTaskStatus(startTime, curTime));
    EXPECT_EQ(rtsq.pendingSqeCnt, 0);
}

TEST_F(RtsqA5Test, UT_CheckLaunchTaskStatus_CallbackReturnSuspendingZeroPending_Expect_NormalReturn)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 0;
    rtsq.sqFullTimeout_ = 10;

    auto startTime = std::chrono::steady_clock::now();
    auto curTime = startTime + std::chrono::seconds(5);
    rtsq.checkExecStatusCallback_ = [](bool isTimeout) {
        return HCCL_E_SUSPENDING;
    };

    EXPECT_NO_THROW(rtsq.CheckLaunchTaskStatus(startTime, curTime));
    EXPECT_EQ(rtsq.pendingSqeCnt, 0);
}

TEST_F(RtsqA5Test, UT_CheckLaunchTaskStatus_CallbackReturnErrorWithNoTimeout_Expect_ThrowException)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 16;
    rtsq.sqFullTimeout_ = 10;

    auto startTime = std::chrono::steady_clock::now();
    auto curTime = startTime + std::chrono::seconds(5);
    rtsq.checkExecStatusCallback_ = [](bool isTimeout) {
        return HCCL_E_INTERNAL;
    };

    EXPECT_THROW(rtsq.CheckLaunchTaskStatus(startTime, curTime), Hccl::InternalException);
}

TEST_F(RtsqA5Test, UT_CheckLaunchTaskStatus_CallbackNullWithTimeout_Expect_ThrowException)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 16;
    rtsq.sqFullTimeout_ = 1;
    rtsq.checkExecStatusCallback_ = nullptr;

    auto startTime = std::chrono::steady_clock::now() - std::chrono::seconds(10);
    auto curTime = std::chrono::steady_clock::now();

    EXPECT_THROW(rtsq.CheckLaunchTaskStatus(startTime, curTime), Hccl::InternalException);
}

// ===================== RefreshSqeHeaderTaskField Tests =====================

TEST_F(RtsqA5Test, Ut_RefreshSqeHeaderTaskField_SetsTaskFields)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.taskId_ = 0x00010002U;

    std::vector<uint8_t> sqeBuf(AC_SQE_SIZE, 0);
    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeBuf.data());
    header->rtStreamId = 0;
    header->taskId = 0;

    rtsq.RefreshSqeHeaderTaskField(header);

    EXPECT_EQ(header->rtStreamId, static_cast<uint16_t>(0x0002));
    EXPECT_EQ(header->taskId, static_cast<uint16_t>(0x0001));
}

TEST_F(RtsqA5Test, Ut_RefreshSqeHeaderTaskField_IncrementsTaskId)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    u32 initialTaskId = rtsq.taskId_;

    std::vector<uint8_t> sqeBuf(AC_SQE_SIZE, 0);
    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeBuf.data());

    rtsq.RefreshSqeHeaderTaskField(header);

    EXPECT_EQ(rtsq.taskId_, initialTaskId + 1);
}

TEST_F(RtsqA5Test, Ut_RefreshSqeHeaderTaskField_MultipleCalls_TaskIdIncrementsEachTime)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    u32 initialTaskId = rtsq.taskId_;

    std::vector<uint8_t> sqeBuf(AC_SQE_SIZE, 0);
    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeBuf.data());

    rtsq.RefreshSqeHeaderTaskField(header);
    u32 taskIdAfterFirst = rtsq.taskId_;

    rtsq.RefreshSqeHeaderTaskField(header);
    u32 taskIdAfterSecond = rtsq.taskId_;

    EXPECT_EQ(taskIdAfterFirst, initialTaskId + 1);
    EXPECT_EQ(taskIdAfterSecond, initialTaskId + 2);
}

TEST_F(RtsqA5Test, Ut_RefreshSqeHeaderTaskField_PreservesOtherHeaderFields)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.taskId_ = 0;

    std::vector<uint8_t> sqeBuf(AC_SQE_SIZE, 0);
    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeBuf.data());
    header->type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    header->wrCqe = 1;
    header->numBlocks = 0xABCD;

    rtsq.RefreshSqeHeaderTaskField(header);

    EXPECT_EQ(header->type, static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT));
    EXPECT_EQ(header->wrCqe, 1U);
    EXPECT_EQ(header->numBlocks, static_cast<uint16_t>(0xABCD));
}

// ===================== LaunchNewTask Tests =====================

TEST_F(RtsqA5Test, Ut_LaunchNewTask_PendingSqeCntNotZero_ThrowsException)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 5;
    rtsq.sqTail_ = 0;
    rtsq.sqHead_ = 0;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;

    std::vector<uint8_t> sqeArray(AC_SQE_SIZE * 2, 0);
    EXPECT_THROW(rtsq.LaunchNewTask(sqeArray.data(), 2), Hccl::InternalException);
}

TEST_F(RtsqA5Test, Ut_LaunchNewTask_Success_UpdatesTailAndResetsPendingSqeCnt)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 0;
    rtsq.sqTail_ = 0;
    rtsq.sqHead_ = 0;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;

    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(static_cast<u32>(0)));

    std::vector<uint8_t> sqeArray(AC_SQE_SIZE * 3, 0);
    u32 sqeCount = 3;
    u32 expectedTail = (rtsq.sqTail_ + sqeCount) % rtsq.sqDepth_;

    EXPECT_NO_THROW(rtsq.LaunchNewTask(sqeArray.data(), sqeCount));

    EXPECT_EQ(rtsq.sqTail_, expectedTail);
    EXPECT_EQ(rtsq.pendingSqeCnt, 0U);
}

TEST_F(RtsqA5Test, Ut_LaunchNewTask_Success_CopiesSqeToSqBuffer)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 0;
    rtsq.sqTail_ = 0;
    rtsq.sqHead_ = 0;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;

    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(static_cast<u32>(0)));

    std::vector<uint8_t> sqeArray(AC_SQE_SIZE * 2, 0);
    auto* header0 = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeArray.data());
    header0->type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    auto* header1 = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeArray.data() + AC_SQE_SIZE);
    header1->type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);

    EXPECT_NO_THROW(rtsq.LaunchNewTask(sqeArray.data(), 2));

    auto* sqHeader0 = reinterpret_cast<Rt91095StarsSqeHeader*>(mockSq);
    auto* sqHeader1 = reinterpret_cast<Rt91095StarsSqeHeader*>(mockSq + AC_SQE_SIZE);
    EXPECT_EQ(sqHeader0->type, static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT));
    EXPECT_EQ(sqHeader1->type, static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA));
}

TEST_F(RtsqA5Test, Ut_LaunchNewTask_WithLoopback_Success)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 0;
    rtsq.sqTail_ = AC_SQE_MAX_CNT - 1; // 255
    rtsq.sqHead_ = 4;                  // availableSpace = 256 - (255 - 4) = 5 > 3
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;

    std::vector<uint8_t> sqeArray(AC_SQE_SIZE * 3, 0);
    u32 sqeCount = 3;
    u32 expectedTail = (rtsq.sqTail_ + sqeCount) % rtsq.sqDepth_;

    EXPECT_NO_THROW(rtsq.LaunchNewTask(sqeArray.data(), sqeCount));

    EXPECT_EQ(rtsq.sqTail_, expectedTail);
    EXPECT_EQ(rtsq.pendingSqeCnt, 0U);
}

TEST_F(RtsqA5Test, Ut_LaunchNewTask_SingleSqe_Success)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.pendingSqeCnt = 0;
    rtsq.sqTail_ = 0;
    rtsq.sqHead_ = 0;
    rtsq.sqDepth_ = AC_SQE_MAX_CNT;

    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().will(returnValue(static_cast<u32>(0)));

    std::vector<uint8_t> sqeArray(AC_SQE_SIZE, 0);
    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeArray.data());
    header->type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);

    EXPECT_NO_THROW(rtsq.LaunchNewTask(sqeArray.data(), 1));

    EXPECT_EQ(rtsq.sqTail_, 1U);
    EXPECT_EQ(rtsq.pendingSqeCnt, 0U);

    auto* sqHeader = reinterpret_cast<Rt91095StarsSqeHeader*>(mockSq);
    EXPECT_EQ(sqHeader->type, static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD));
}

TEST_F(RtsqA5Test, Ut_SqeProfilingDisabled_SqeProfStaysZero)
{
    Hccl::SetSqeProfilingEnabled(false);

    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.NotifyWait(0);

    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(rtsq.locBuf);
    EXPECT_EQ(header->sqeProf, 0U);
}

TEST_F(RtsqA5Test, Ut_SqeProfilingEnabled_SqeProfSetToOne)
{
    Hccl::SetSqeProfilingEnabled(true);

    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.NotifyWait(0);

    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(rtsq.locBuf);
    EXPECT_EQ(header->sqeProf, 1U);

    Hccl::SetSqeProfilingEnabled(false);
}

TEST_F(RtsqA5Test, Ut_NotifyWait_WithProfiling_SqeProfSetToOne)
{
    Hccl::SetSqeProfilingEnabled(true);

    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.NotifyWait(0);

    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(rtsq.locBuf);
    EXPECT_EQ(header->sqeProf, 1U);

    Hccl::SetSqeProfilingEnabled(false);
}

TEST_F(RtsqA5Test, Ut_SdmaCopy_WithProfiling_SqeProfSetToOne)
{
    Hccl::SetSqeProfilingEnabled(true);

    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.SdmaCopy(0x100, 0x200, 0x300, 0x400);

    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(rtsq.locBuf);
    EXPECT_EQ(header->sqeProf, 1U);

    Hccl::SetSqeProfilingEnabled(false);
}

TEST_F(RtsqA5Test, Ut_NotifyWait_WithoutProfiling_SqeProfStaysZero)
{
    Hccl::SetSqeProfilingEnabled(false);

    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.NotifyWait(0);

    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(rtsq.locBuf);
    EXPECT_EQ(header->sqeProf, 0U);
}

TEST_F(RtsqA5Test, Ut_SqeProfiling_ToggleOffThenBuild_SqeProfStaysZero)
{
    // 开关推送关闭后，后续构建的SQE不再置位
    Hccl::SetSqeProfilingEnabled(true);
    Hccl::SetSqeProfilingEnabled(false);

    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    rtsq.NotifyWait(0);

    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(rtsq.locBuf);
    EXPECT_EQ(header->sqeProf, 0U);
}

TEST_F(RtsqA5Test, Ut_NotifyWait_EnabledOnOtherThread_SqeProfStaysZero)
{
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    // 开关为线程级变量：其他线程置1不影响本线程，本线程未经报到口开启时读初始值0
    std::thread toggleThread([]() {
        Hccl::SetSqeProfilingEnabled(true);
    });
    toggleThread.join();

    std::thread buildThread([&rtsq]() {
        rtsq.NotifyWait(0);
    });
    buildThread.join();

    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(rtsq.locBuf);
    EXPECT_EQ(header->sqeProf, 0U);

    // 同线程内开启后构建，置位生效：第二次NotifyWait写入locBuf第二槽位，须取新槽位header断言
    Hccl::SetSqeProfilingEnabled(true);
    rtsq.NotifyWait(0);
    auto* secondHeader = reinterpret_cast<Rt91095StarsSqeHeader*>(rtsq.locBuf + RTSQ_SQE_SIZE);
    EXPECT_EQ(secondHeader->sqeProf, 1U);

    Hccl::SetSqeProfilingEnabled(false);
}

TEST_F(RtsqA5Test, Ut_InlineBuildFunctions_WithProfiling_SqeProfSetToOne)
{
    Hccl::SetSqeProfilingEnabled(true);
    std::vector<uint8_t> sqeBuf(128, 0);
    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeBuf.data());
    auto resetAndBuild = [&sqeBuf, header](auto buildFunc) {
        std::fill(sqeBuf.begin(), sqeBuf.end(), 0);
        buildFunc();
        return header->sqeProf;
    };

    // 内联Build函数10个（.h）
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeNotifyWait(1, 1, 1, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeNotifyWait(1, 1, 1, 68, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeNotifyRecord(1, 1, 1, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeCnt1toNNotifyRecord(1, 1, 1, 8, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeCnt1toNNotifyWait(1, 1, 1, 8, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeCntNto1NotifyRecord(1, 1, 1, 8, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeCntNto1NotifyWait(1, 1, 1, 8, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeSdmaCopy(1, 1, 0x100, 0x200, 0x30, 0, 0, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeUbDbSend(1, 1, Hccl::UbJettyLiteId(0, 1, 2), 3, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeP2pWriteValue(1, 1, 0x1000, 0x5a, sqeBuf.data());
              }));

    Hccl::SetSqeProfilingEnabled(false);
}

TEST_F(RtsqA5Test, Ut_NonInlineBuild_WithProfiling_SqeProfSetToOne)
{
    Hccl::SetSqeProfilingEnabled(true);
    // COND类型SQE约68字节，buffer开128字节覆盖全部SQE类型
    std::vector<uint8_t> sqeBuf(128, 0);
    auto* header1 = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeBuf.data());
    auto resetAndBuild = [&sqeBuf, header1](auto buildFunc) {
        std::fill(sqeBuf.begin(), sqeBuf.end(), 0);
        buildFunc();
        return header1->sqeProf;
    };

    // 非内联Build函数3个（.cc），last=true/false分别覆盖clear与nop分支
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeCCoreNotifyWait(1, 1, 0x2000, 0x2004, true, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeCCoreNotifyWait(1, 1, 0x2000, 0x2004, false, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeCCoreNotifyRecord(1, 1, 0x3000, 0x3004, sqeBuf.data());
              }));
    EXPECT_EQ(1U, resetAndBuild([&]() {
                  Hccl::BuildA5SqeRdmaDbSend(1, 1, 0x4000, 0x1234, sqeBuf.data());
              }));

    Hccl::SetSqeProfilingEnabled(false);
}

TEST_F(RtsqA5Test, Ut_NonInlineBuild_WithoutProfiling_SqeProfStaysZero)
{
    Hccl::SetSqeProfilingEnabled(false);
    std::vector<uint8_t> sqeBuf(128, 0);
    auto* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeBuf.data());

    Hccl::BuildA5SqeCCoreNotifyWait(1, 1, 0x2000, 0x2004, true, sqeBuf.data());
    EXPECT_EQ(header->sqeProf, 0U);

    std::fill(sqeBuf.begin(), sqeBuf.end(), 0);
    Hccl::BuildA5SqeCCoreNotifyRecord(1, 1, 0x3000, 0x3004, sqeBuf.data());
    EXPECT_EQ(header->sqeProf, 0U);

    std::fill(sqeBuf.begin(), sqeBuf.end(), 0);
    Hccl::BuildA5SqeRdmaDbSend(1, 1, 0x4000, 0x1234, sqeBuf.data());
    EXPECT_EQ(header->sqeProf, 0U);
}
