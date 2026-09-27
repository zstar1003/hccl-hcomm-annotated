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
#define private public
#include "sqe_mgr.h"
#include "null_ptr_exception.h"
#include "internal_exception.h"
#include "drv_api_exception.h"
#undef private

using namespace Hccl;

class SqeMgrTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        sqeManager = new SqeMgr(1);
        sqId = 1;
        std::cout << "SqeMgrTest SetUP" << std::endl;
    }

    static void TearDownTestCase()
    {
        delete sqeManager;
        std::cout << "SqeMgrTest TearDown" << std::endl;
    }

    virtual void SetUp() { std::cout << "A Test case in SqeMgrTest SetUp" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in SqeMgrTest TearDown" << std::endl;
    }

    static SqeMgr* sqeManager;
    static u32 sqId;
    static u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT];
};

SqeMgr* SqeMgrTest::sqeManager = nullptr;
u32 SqeMgrTest::sqId = 1;
u8 SqeMgrTest::mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT] = {0};

drvError_t halSqCqQuery(uint32_t devId, halSqCqQueryInfo* info) { return DRV_ERROR_NOT_SUPPORT; }

TEST_F(SqeMgrTest, sqe_mgr_begin)
{
    // given
    MOCKER_CPP(&SqeMgr::QuerySqDepth).stubs().with(mockcpp::any()).will(returnValue(AC_SQE_MAX_CNT));
    MOCKER_CPP(&SqeMgr::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
    MOCKER_CPP(&SqeMgr::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
    MOCKER_CPP(&SqeMgr::QuerySqBaseAddr).stubs().with(mockcpp::any()).will(returnValue(reinterpret_cast<u64>(&mockSq)));

    // when
    sqeManager->Begin(sqId);
    SqInfo* sqInfo = sqeManager->sqInfos[sqId].get();

    // then
    EXPECT_EQ(sqInfo->sqeCnt, 0);
    EXPECT_EQ(sqInfo->sqDepth, AC_SQE_MAX_CNT);
    EXPECT_EQ(sqInfo->sqTail, 0);
    EXPECT_EQ(sqInfo->sqHead, 0);
    EXPECT_EQ(sqInfo->sqBaseAddr, reinterpret_cast<u64>(&mockSq));
}

TEST_F(SqeMgrTest, sqe_mgr_add)
{
    HcclResult ret = sqeManager->Add(sqId, nullptr);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PTR);

    // given
    HcclNotifyWaitSqe* notifyWaitSqe = new HcclNotifyWaitSqe();
    u16 streamId = 1;
    u16 taskId = 0;
    u64 notifyId = 1;
    notifyWaitSqe->Config(streamId, taskId, notifyId);

    // when
    sqeManager->Add(sqId, notifyWaitSqe);
    SqInfo* sqInfo = sqeManager->sqInfos[sqId].get();

    // then
    EXPECT_EQ(sqInfo->sqeCnt, 1);
    RtStarsNotifySqe* rtSqe = reinterpret_cast<RtStarsNotifySqe*>(sqInfo->sqeBuffer);
    EXPECT_EQ(rtSqe->header.type, static_cast<u8>(RtStarsSqeType::RT_STARS_SQE_TYPE_NOTIFY_WAIT));
    EXPECT_EQ(rtSqe->kernelCredit, RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT);
    EXPECT_EQ(rtSqe->header.rtStreamId, streamId);
    EXPECT_EQ(rtSqe->camelBack, notifyId);
    EXPECT_EQ(rtSqe->header.taskId, taskId);
    delete notifyWaitSqe;
}

TEST_F(SqeMgrTest, sqe_mgr_commit_no_loop_back)
{
    // given
    MOCKER_CPP(&SqeMgr::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&SqeMgr::ConfigSqTail).stubs().with(mockcpp::any(), mockcpp::any());
    u16 streamId = 1;
    u16 taskId = 0;
    u64 notifyId = 1;
    SqInfo* sqInfo = sqeManager->sqInfos[sqId].get();
    u32 newTail = (sqInfo->sqTail + sqInfo->sqeCnt) % sqInfo->sqDepth;

    // when
    sqeManager->Commit(sqId);

    // then
    EXPECT_EQ(sqInfo->sqTail, newTail);
    RtStarsNotifySqe* rtSqe = reinterpret_cast<RtStarsNotifySqe*>(mockSq);
    EXPECT_EQ(rtSqe->header.type, static_cast<u8>(RtStarsSqeType::RT_STARS_SQE_TYPE_NOTIFY_WAIT));
    EXPECT_EQ(rtSqe->kernelCredit, RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT);
    EXPECT_EQ(rtSqe->header.rtStreamId, streamId);
    EXPECT_EQ(rtSqe->camelBack, notifyId);
    EXPECT_EQ(rtSqe->header.taskId, taskId);
    EXPECT_EQ(sqInfo->sqeCnt, 0);
}

TEST_F(SqeMgrTest, sqe_mgr_commit_invalid_sqId)
{
    // given
    u32 sqId = 99;

    // when
    HcclResult ret = sqeManager->Commit(sqId);

    // then
    EXPECT_EQ(ret, HcclResult::HCCL_E_NOT_FOUND);
}

TEST_F(SqeMgrTest, sqe_mgr_commit_with_loop_back)
{
    // given
    MOCKER_CPP(&SqeMgr::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&SqeMgr::ConfigSqTail).stubs().with(mockcpp::any(), mockcpp::any());
    // clear current sq
    memset_s(mockSq, sizeof(mockSq), 0, sizeof(mockSq));
    sqeManager->Begin(sqId);

    // manually reset sqtail and sqhead to create loopback condition
    SqInfo* sqInfo = sqeManager->sqInfos[sqId].get();
    u32 oldTail = AC_SQE_MAX_CNT - 1;
    u32 oldHead = AC_SQE_MAX_CNT - 2;
    sqInfo->sqTail = oldTail;
    sqInfo->sqHead = oldHead;

    u16 streamId = 1;
    u16 taskId = 0;
    u64 notifyId = 1;
    HcclNotifyWaitSqe* notifyWaitSqe1 = new HcclNotifyWaitSqe();
    notifyWaitSqe1->Config(streamId, taskId, notifyId);
    HcclNotifyWaitSqe* notifyWaitSqe2 = new HcclNotifyWaitSqe();
    notifyWaitSqe2->Config(streamId, taskId, notifyId);

    // when
    sqeManager->Add(sqId, notifyWaitSqe1);
    sqeManager->Add(sqId, notifyWaitSqe2);

    // then
    EXPECT_EQ(sqInfo->sqeCnt, 2);
    u32 newTail = (sqInfo->sqTail + sqInfo->sqeCnt) % sqInfo->sqDepth;

    // when
    sqeManager->Commit(sqId);

    // then
    EXPECT_EQ(sqInfo->sqTail, newTail);
    RtStarsNotifySqe* rtSqe1
        = reinterpret_cast<RtStarsNotifySqe*>(reinterpret_cast<u8*>(mockSq) + oldTail * AC_SQE_SIZE);
    RtStarsNotifySqe* rtSqe2 = reinterpret_cast<RtStarsNotifySqe*>(mockSq);
    EXPECT_EQ(rtSqe1->header.type, static_cast<u8>(RtStarsSqeType::RT_STARS_SQE_TYPE_NOTIFY_WAIT));
    EXPECT_EQ(rtSqe1->kernelCredit, RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT);
    EXPECT_EQ(rtSqe1->header.rtStreamId, streamId);
    EXPECT_EQ(rtSqe1->camelBack, notifyId);
    EXPECT_EQ(rtSqe1->header.taskId, taskId);
    EXPECT_EQ(rtSqe2->header.type, static_cast<u8>(RtStarsSqeType::RT_STARS_SQE_TYPE_NOTIFY_WAIT));
    EXPECT_EQ(rtSqe2->kernelCredit, RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT);
    EXPECT_EQ(rtSqe2->header.rtStreamId, streamId);
    EXPECT_EQ(rtSqe2->camelBack, notifyId);
    EXPECT_EQ(rtSqe2->header.taskId, taskId);
    EXPECT_EQ(sqInfo->sqeCnt, 0);

    delete notifyWaitSqe1;
    delete notifyWaitSqe2;
}

TEST_F(SqeMgrTest, calling_multiple_begin_in_one_round_should_return_error)
{
    // given
    sqeManager->Begin(sqId);
    HcclNotifyWaitSqe* notifyWaitSqe = new HcclNotifyWaitSqe();
    u16 streamId = 1;
    u16 taskId = 0;
    u64 notifyId = 1;
    notifyWaitSqe->Config(streamId, taskId, notifyId);

    // when
    sqeManager->Add(sqId, notifyWaitSqe);

    // then
    EXPECT_EQ(sqeManager->Begin(sqId), HcclResult::HCCL_E_INTERNAL);

    delete notifyWaitSqe;
}

TEST_F(SqeMgrTest, test_query_functions)
{
    halSqCqQueryInfo queryInfo;
    queryInfo.tsId = 0;
    queryInfo.sqId = 0;
    queryInfo.cqId = 0;
    queryInfo.type = DRV_NORMAL_TYPE;
    queryInfo.prop = DRV_SQCQ_PROP_SQ_BASE;
    queryInfo.value[0] = 0;
    queryInfo.value[1] = 0;

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any(), outBoundP(&queryInfo, sizeof(queryInfo))).will(returnValue(0));
    auto head = sqeManager->QuerySqHead(0);
    auto tail = sqeManager->QuerySqTail(0);
    auto depth = sqeManager->QuerySqDepth(0);
    auto addr = sqeManager->QuerySqBaseAddr(0);

    EXPECT_EQ(head, 0);
    EXPECT_EQ(tail, 0);
    EXPECT_EQ(depth, 0);
    EXPECT_EQ(addr, 0);
}

drvError_t halSqCqConfig(uint32_t devId, struct halSqCqConfigInfo* info) { return DRV_ERROR_NOT_SUPPORT; }

TEST_F(SqeMgrTest, test_config_functions) { EXPECT_NO_THROW(sqeManager->ConfigSqTail(0, 1)); }

TEST_F(SqeMgrTest, test_AddSqeToBuffer_functions)
{
    EXPECT_THROW(sqeManager->AddSqeToBuffer(nullptr, nullptr), Hccl::NullPtrException);
}

TEST_F(SqeMgrTest, sqe_mgr_add_nullptr)
{
    HcclResult ret = sqeManager->Add(sqId, nullptr);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PTR);

    // given
    HcclNotifyWaitSqe* notifyWaitSqe = new HcclNotifyWaitSqe();
    u16 streamId = 1;
    u16 taskId = 0;
    u64 notifyId = 1;
    notifyWaitSqe->Config(streamId, taskId, notifyId);

    // when
    sqeManager->sqInfos[sqId] = nullptr;
    ret = sqeManager->Add(sqId, notifyWaitSqe);
    EXPECT_EQ(ret, HcclResult::HCCL_E_NOT_FOUND);
    delete notifyWaitSqe;
}

TEST_F(SqeMgrTest, test_QuerySqStatusByType_functions)
{
    MOCKER(halSqCqQuery).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(sqeManager->QuerySqStatusByType(sqId, DRV_SQCQ_PROP_SQ_STATUS), DrvApiException);
}

TEST_F(SqeMgrTest, test_QuerySqBaseAdd)
{
    MOCKER(halSqCqQuery).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(sqeManager->QuerySqBaseAddr(sqId), DrvApiException);
}
