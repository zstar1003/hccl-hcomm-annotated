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
#include "aicpu/aicpu_hccl_sqcq.h"
#include "hccl_common.h"

using namespace std;
using namespace hccl;

namespace {
drvError_t g_halSqCqQueryResult = DRV_ERROR_NONE;
drvError_t g_halSqCqConfigResult = DRV_ERROR_NONE;
drvError_t g_halCqReportRecvResult = DRV_ERROR_NONE;
drvError_t g_halTsdrvCtlResult = DRV_ERROR_NONE;
uint32_t g_queryAbortStatus = 0;
} // namespace

extern "C" {
drvError_t halCqReportRecv(uint32_t devId, struct halReportRecvInfo* info);
drvError_t halSqCqQuery(uint32_t devId, struct halSqCqQueryInfo* info);
drvError_t halSqCqConfig(uint32_t devId, struct halSqCqConfigInfo* info);
drvError_t halTsdrvCtl(uint32_t devId, int cmd, void* param, size_t paramSize, void* out, size_t* outSize);
drvError_t halEschedSubmitEvent(uint32_t devId, struct event_summary* event);
}

drvError_t halSqCqQuery(uint32_t devId, struct halSqCqQueryInfo* info)
{
    (void)devId;
    if (info->prop == DRV_SQCQ_PROP_SQ_BASE) {
        info->value[0] = 0x1000U;
        info->value[1] = 0x0U;
    } else if (info->prop == DRV_SQCQ_PROP_SQ_TAIL) {
        info->value[0] = 0x10U;
    } else if (info->prop == DRV_SQCQ_PROP_SQ_HEAD) {
        info->value[0] = 0x8U;
    }
    return g_halSqCqQueryResult;
}

drvError_t halSqCqConfig(uint32_t devId, struct halSqCqConfigInfo* info)
{
    (void)devId;
    (void)info;
    return g_halSqCqConfigResult;
}

drvError_t halCqReportRecv(uint32_t devId, struct halReportRecvInfo* info)
{
    (void)devId;
    (void)info;
    return g_halCqReportRecvResult;
}

drvError_t halTsdrvCtl(uint32_t devId, int cmd, void* param, size_t paramSize, void* out, size_t* outSize)
{
    (void)devId;
    (void)param;
    (void)paramSize;
    if (out != nullptr && outSize != nullptr && *outSize >= 44) {
        uint32_t* statusPtr = static_cast<uint32_t*>(out);
        *statusPtr = g_queryAbortStatus;
    }
    return g_halTsdrvCtlResult;
}

drvError_t halEschedSubmitEvent(uint32_t devId, struct event_summary* event)
{
    (void)devId;
    (void)event;
    return DRV_ERROR_NONE;
}

class RtsqInteract_Sqcq_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RtsqInteract_Sqcq_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "RtsqInteract_Sqcq_UT TearDown" << std::endl; }
    virtual void SetUp()
    {
        g_halSqCqQueryResult = DRV_ERROR_NONE;
        g_halSqCqConfigResult = DRV_ERROR_NONE;
        g_halCqReportRecvResult = DRV_ERROR_NONE;
        g_halTsdrvCtlResult = DRV_ERROR_NONE;
        g_queryAbortStatus = ts::APP_ABORT_KILL_FINISH;
        std::cout << "RtsqInteract_Sqcq_UT Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "RtsqInteract_Sqcq_UT Test TearDown" << std::endl;
    }
};

TEST_F(RtsqInteract_Sqcq_UT, Ut_QuerySqBaseAddr_Success)
{
    uint32_t devId = 0;
    uint32_t sqId = 1;
    u64 outVal = 0;
    HcclResult ret = QuerySqBaseAddr(devId, sqId, outVal);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(outVal, 0x1000U);
}

TEST_F(RtsqInteract_Sqcq_UT, Ut_QuerySqBaseAddr_When_QueryFailed_Failed)
{
    uint32_t devId = 0;
    uint32_t sqId = 1;
    u64 outVal = 0;
    g_halSqCqQueryResult = DRV_ERROR_INVALID_VALUE;
    HcclResult ret = QuerySqBaseAddr(devId, sqId, outVal);
    EXPECT_EQ(ret, HCCL_E_DRV);
}

TEST_F(RtsqInteract_Sqcq_UT, Ut_QuerySqStatusByType_Success)
{
    uint32_t devId = 0;
    uint32_t sqId = 1;
    drvSqCqPropType_t type = DRV_SQCQ_PROP_SQ_TAIL;
    uint32_t outVal = 0;
    HcclResult ret = QuerySqStatusByType(devId, sqId, type, outVal);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(outVal, 0x10U);
}

TEST_F(RtsqInteract_Sqcq_UT, Ut_QuerySqStatusByType_When_QueryFailed_Failed)
{
    uint32_t devId = 0;
    uint32_t sqId = 1;
    drvSqCqPropType_t type = DRV_SQCQ_PROP_SQ_TAIL;
    uint32_t outVal = 0;
    g_halSqCqQueryResult = DRV_ERROR_INVALID_VALUE;
    HcclResult ret = QuerySqStatusByType(devId, sqId, type, outVal);
    EXPECT_EQ(ret, HCCL_E_DRV);
}

TEST_F(RtsqInteract_Sqcq_UT, Ut_QuerySqStatus_Success)
{
    uint32_t devId = 0;
    uint32_t sqId = 1;
    uint32_t sqHead = 0;
    uint32_t sqTail = 0;
    HcclResult ret = QuerySqStatus(devId, sqId, sqHead, sqTail);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(sqHead, 0x8U);
    EXPECT_EQ(sqTail, 0x10U);
}

TEST_F(RtsqInteract_Sqcq_UT, Ut_QuerySqStatus_When_TailQueryFailed_Failed)
{
    uint32_t devId = 0;
    uint32_t sqId = 1;
    uint32_t sqHead = 0;
    uint32_t sqTail = 0;
    g_halSqCqQueryResult = DRV_ERROR_INVALID_VALUE;
    HcclResult ret = QuerySqStatus(devId, sqId, sqHead, sqTail);
    EXPECT_EQ(ret, HCCL_E_DRV);
}

TEST_F(RtsqInteract_Sqcq_UT, Ut_ConfigSqStatusByType_Success)
{
    uint32_t devId = 0;
    uint32_t sqId = 1;
    drvSqCqPropType_t type = DRV_SQCQ_PROP_SQ_HEAD;
    uint32_t value = 0x20U;
    HcclResult ret = ConfigSqStatusByType(devId, sqId, type, value);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RtsqInteract_Sqcq_UT, Ut_ConfigSqStatusByType_When_ConfigFailed_Failed)
{
    uint32_t devId = 0;
    uint32_t sqId = 1;
    drvSqCqPropType_t type = DRV_SQCQ_PROP_SQ_HEAD;
    uint32_t value = 0x20U;
    g_halSqCqConfigResult = DRV_ERROR_INVALID_VALUE;
    HcclResult ret = ConfigSqStatusByType(devId, sqId, type, value);
    EXPECT_EQ(ret, HCCL_E_DRV);
}

TEST_F(RtsqInteract_Sqcq_UT, Ut_AddOneWaitStartSqe_When_LastIsTrue)
{
    rtStarsCcoreWaitStartSqe_t sqe = {};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    u64 waitAddr = 0x1000UL;
    u64 curTurnCntAddr = 0x2000UL;
    bool last = true;

    AddOneWaitStartSqe(streamId, taskId, waitAddr, curTurnCntAddr, last, &sqe, &sqeType);

    EXPECT_EQ(sqeType, CCORE_WAIT_START_SQE);
    EXPECT_EQ(sqe.sqeHeader.type, RT_STARS_SQE_TYPE_COND);
    EXPECT_EQ(sqe.sqeHeader.rtStreamId, streamId);
    EXPECT_EQ(sqe.sqeHeader.taskId, taskId);
    EXPECT_EQ(sqe.kernel_credit, RT_STARS_DEFAULT_KERNEL_CREDIT);
    EXPECT_EQ(sqe.csc, 1U);
}

TEST_F(RtsqInteract_Sqcq_UT, Ut_AddOneWaitStartSqe_When_LastIsFalse)
{
    rtStarsCcoreWaitStartSqe_t sqe = {};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    u64 waitAddr = 0x1000UL;
    u64 curTurnCntAddr = 0x2000UL;
    bool last = false;

    AddOneWaitStartSqe(streamId, taskId, waitAddr, curTurnCntAddr, last, &sqe, &sqeType);

    EXPECT_EQ(sqeType, CCORE_WAIT_START_SQE);
    EXPECT_EQ(sqe.sqeHeader.type, RT_STARS_SQE_TYPE_COND);
}

TEST_F(RtsqInteract_Sqcq_UT, Ut_AddOneWriteValueStartSqe)
{
    rtStarsCcoreWriteValueSqe_t sqe = {};
    uint8_t sqeType = 0;
    uint16_t streamId = 1;
    uint16_t taskId = 2;
    u64 writeAddr = 0x1000UL;
    u64 valueAddr = 0x2000UL;

    AddOneWriteValueStartSqe(streamId, taskId, writeAddr, valueAddr, &sqe, &sqeType);

    EXPECT_EQ(sqeType, CCORE_WRITE_VALUE_SQE);
    EXPECT_EQ(sqe.sqeHeader.type, RT_STARS_SQE_TYPE_COND);
    EXPECT_EQ(sqe.sqeHeader.rtStreamId, streamId);
    EXPECT_EQ(sqe.sqeHeader.taskId, taskId);
    EXPECT_EQ(sqe.kernel_credit, RT_STARS_DEFAULT_KERNEL_CREDIT);
    EXPECT_EQ(sqe.csc, 1U);
}

TEST_F(RtsqInteract_Sqcq_UT, Ut_CqReportRecv_When_Timeout)
{
    CqeQueryInput cqeQueryInput = {};
    cqeQueryInput.devId = 0;
    cqeQueryInput.streamId = 1;
    cqeQueryInput.cqId = 0;
    cqeQueryInput.type = 0;
    rtLogicCqReport_t cqeException = {};

    g_halCqReportRecvResult = DRV_ERROR_WAIT_TIMEOUT;
    CqeStatus status = CqReportRecv(cqeQueryInput, cqeException);
    EXPECT_EQ(status, CqeStatus::kCqeTimeOut);
}

TEST_F(RtsqInteract_Sqcq_UT, Ut_CqReportRecv_When_OtherError)
{
    CqeQueryInput cqeQueryInput = {};
    cqeQueryInput.devId = 0;
    cqeQueryInput.streamId = 1;
    cqeQueryInput.cqId = 0;
    cqeQueryInput.type = 0;
    rtLogicCqReport_t cqeException = {};

    g_halCqReportRecvResult = DRV_ERROR_INVALID_VALUE;
    CqeStatus status = CqReportRecv(cqeQueryInput, cqeException);
    EXPECT_EQ(status, CqeStatus::kCqeInnerError);
}

TEST_F(RtsqInteract_Sqcq_UT, Ut_StreamsKill_When_Success)
{
    uint32_t devId = 0;
    g_halTsdrvCtlResult = DRV_ERROR_NONE;
    HcclResult ret = StreamsKill(devId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RtsqInteract_Sqcq_UT, Ut_StreamsKill_When_Failed)
{
    uint32_t devId = 0;
    g_halTsdrvCtlResult = DRV_ERROR_INVALID_VALUE;
    HcclResult ret = StreamsKill(devId);
    EXPECT_EQ(ret, HCCL_E_DRV);
}
