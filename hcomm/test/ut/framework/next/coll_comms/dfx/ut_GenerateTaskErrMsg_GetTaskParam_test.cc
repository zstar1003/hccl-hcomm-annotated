/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "hccl_comm_pub.h"
#include "hccl_api_base_test.h"

#define private public
#define protected public
#include "hcclCommTaskExceptionLite.h"
#include "hcclCommTaskException.h"
#include "aicpu_ts_thread.h"
#undef private
#undef protected

#include "hcomm_task_scheduler_error.h"
#include "aicpu_indop_env.h"
#include "adapter_hal_pub.h"
#include "dlhal_function_v2.h"
#include "rtsq_base.h"
#include "task_param.h"
#include "error_message_v2.h"
#include "sqe_a5.h"

using namespace hccl;
using namespace hcomm;

namespace hcomm {
void GetTaskParam(Hccl::TaskParam& taskParam, const Hccl::ErrorMessageReport& errMsgInfo);
}

class TaskErrMsgTest : public testing::Test {
protected:
    void SetUp() override
    {
        MOCKER(::getpid).stubs().will(returnValue(12345));
        MOCKER(HrtHalDrvQueryProcessHostPid).stubs().will(returnValue(HCCL_SUCCESS));
        Hccl::DlHalFunctionV2::GetInstance().dlHalEschedSubmitEvent
            = [](unsigned int, struct event_summary*) -> drvError_t {
            return DRV_ERROR_NONE;
        };
        HcclCommTaskExceptionLite::GetInstance().Init(0);
    }
    void TearDown() override { GlobalMockObject::verify(); }
};

static Hccl::DfxTaskInfo MakeDfxTaskInfo(u8 taskType)
{
    Hccl::DfxTaskInfo taskInfo = {};
    taskInfo.taskType = taskType;
    return taskInfo;
}

static Hccl::Rt91095StarsNotifySqe
MakeNotifySqe(uint32_t notifyId, uint32_t cntValue, Hccl::Rt91095StarsSqeType sqeType)
{
    Hccl::Rt91095StarsNotifySqe sqe = {};
    sqe.header.type = static_cast<uint8_t>(sqeType);
    sqe.notifyId = notifyId;
    sqe.cntValue = cntValue;
    return sqe;
}

static rtLogicCqReport_t MakeExceptionInfo(uint32_t errorCode = 0xAB, uint8_t errorType = 1)
{
    rtLogicCqReport_t info{};
    info.errorCode = errorCode;
    info.errorType = errorType;
    return info;
}

TEST_F(TaskErrMsgTest, Ut_GenerateTaskErrMsg_When_NotifyWait_Expect_NotifyFieldsSet)
{
    auto sqe = MakeNotifySqe(100, 200, Hccl::Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    auto taskInfo = MakeDfxTaskInfo(static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT));
    taskInfo.taskPara.Notify.sqeAddr = reinterpret_cast<u64>(&sqe);
    auto exceptionInfo = MakeExceptionInfo();
    Hccl::ErrorMessageReport errMsgInfo{};

    HcclCommTaskExceptionLite::GetInstance().GenerateTaskErrMsg(taskInfo, errMsgInfo, exceptionInfo);

    EXPECT_EQ(errMsgInfo.notifyId, 100u);
    EXPECT_EQ(errMsgInfo.reduceType, 255u);
}

TEST_F(TaskErrMsgTest, Ut_GenerateTaskErrMsg_When_NotifyRecord_Expect_NotifyFieldsSet)
{
    auto sqe = MakeNotifySqe(300, 400, Hccl::Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
    auto taskInfo = MakeDfxTaskInfo(static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_RECORD));
    taskInfo.taskPara.Notify.sqeAddr = reinterpret_cast<u64>(&sqe);
    auto exceptionInfo = MakeExceptionInfo();
    Hccl::ErrorMessageReport errMsgInfo{};

    HcclCommTaskExceptionLite::GetInstance().GenerateTaskErrMsg(taskInfo, errMsgInfo, exceptionInfo);

    EXPECT_EQ(errMsgInfo.notifyId, 300u);
    EXPECT_EQ(errMsgInfo.reduceType, 255u);
}

TEST_F(TaskErrMsgTest, Ut_GenerateTaskErrMsg_When_UbReduceInline_Expect_ReduceFieldsSet)
{
    auto taskInfo = MakeDfxTaskInfo(static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB_REDUCE_INLINE));
    taskInfo.taskPara.Reduce.notifyId = 500;
    taskInfo.taskPara.Reduce.reduceOp = static_cast<u8>(HcclReduceOp::HCCL_REDUCE_SUM);
    taskInfo.taskPara.Reduce.srcAddr = 0x1000;
    taskInfo.taskPara.Reduce.dstAddr = 0x2000;
    taskInfo.taskPara.Reduce.size = 1024;
    taskInfo.linkType = static_cast<u8>(Hccl::DfxLinkType::HCCS);
    auto exceptionInfo = MakeExceptionInfo(0x1234);
    Hccl::ErrorMessageReport errMsgInfo{};

    HcclCommTaskExceptionLite::GetInstance().GenerateTaskErrMsg(taskInfo, errMsgInfo, exceptionInfo);

    EXPECT_EQ(errMsgInfo.notifyId, 500u);
    EXPECT_EQ(errMsgInfo.reduceType, static_cast<uint32_t>(HcclReduceOp::HCCL_REDUCE_SUM));
    EXPECT_EQ(errMsgInfo.ubCqeStatus, 0x34u);
    EXPECT_EQ(errMsgInfo.linkType, Hccl::DfxLinkType::HCCS);
    EXPECT_EQ(errMsgInfo.size, 1024u);
    EXPECT_EQ(errMsgInfo.taskSrcAddr, 0x1000u);
    EXPECT_EQ(errMsgInfo.taskDstAddr, 0x2000u);
}

TEST_F(TaskErrMsgTest, Ut_GenerateTaskErrMsg_When_WriteReduceWithNotify_Expect_ReduceFieldsSet)
{
    auto taskInfo = MakeDfxTaskInfo(static_cast<u8>(Hccl::TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY));
    taskInfo.taskPara.Reduce.notifyId = 700;
    taskInfo.taskPara.Reduce.reduceOp = static_cast<u8>(HcclReduceOp::HCCL_REDUCE_MAX);
    taskInfo.taskPara.Reduce.size = 2048;
    taskInfo.linkType = static_cast<u8>(Hccl::DfxLinkType::PCIE);
    auto exceptionInfo = MakeExceptionInfo(0x5678);
    Hccl::ErrorMessageReport errMsgInfo{};

    HcclCommTaskExceptionLite::GetInstance().GenerateTaskErrMsg(taskInfo, errMsgInfo, exceptionInfo);

    EXPECT_EQ(errMsgInfo.notifyId, 700u);
    EXPECT_EQ(errMsgInfo.reduceType, static_cast<uint32_t>(HcclReduceOp::HCCL_REDUCE_MAX));
    EXPECT_EQ(errMsgInfo.ubCqeStatus, 0x78u);
    EXPECT_EQ(errMsgInfo.linkType, Hccl::DfxLinkType::PCIE);
    EXPECT_EQ(errMsgInfo.size, 2048u);
}

TEST_F(TaskErrMsgTest, Ut_GenerateTaskErrMsg_When_ReduceInline_Expect_OnlyReduceTypeSet)
{
    auto taskInfo = MakeDfxTaskInfo(static_cast<u8>(Hccl::TaskParamTypeVal::TASK_REDUCE_INLINE));
    taskInfo.taskPara.Reduce.reduceOp = static_cast<u8>(HcclReduceOp::HCCL_REDUCE_PROD);
    auto exceptionInfo = MakeExceptionInfo();
    Hccl::ErrorMessageReport errMsgInfo{};

    HcclCommTaskExceptionLite::GetInstance().GenerateTaskErrMsg(taskInfo, errMsgInfo, exceptionInfo);

    EXPECT_EQ(errMsgInfo.reduceType, static_cast<uint32_t>(HcclReduceOp::HCCL_REDUCE_PROD));
    EXPECT_EQ(errMsgInfo.notifyId, 0u);
}

TEST_F(TaskErrMsgTest, Ut_GenerateTaskErrMsg_When_UbInlineWrite_Expect_DmaFieldsSet)
{
    auto taskInfo = MakeDfxTaskInfo(static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB_INLINE_WRITE));
    taskInfo.taskPara.ubDma.notifyId = 1100;
    taskInfo.taskPara.ubDma.srcAddr = 0x3000;
    taskInfo.taskPara.ubDma.dstAddr = 0x4000;
    taskInfo.taskPara.ubDma.size = 4096;
    taskInfo.linkType = static_cast<u8>(Hccl::DfxLinkType::ROCE);
    auto exceptionInfo = MakeExceptionInfo(0x9ABC);
    Hccl::ErrorMessageReport errMsgInfo{};

    HcclCommTaskExceptionLite::GetInstance().GenerateTaskErrMsg(taskInfo, errMsgInfo, exceptionInfo);

    EXPECT_EQ(errMsgInfo.notifyId, 1100u);
    EXPECT_EQ(errMsgInfo.reduceType, 255u);
    EXPECT_EQ(errMsgInfo.ubCqeStatus, 0xBCu);
    EXPECT_EQ(errMsgInfo.linkType, Hccl::DfxLinkType::ROCE);
    EXPECT_EQ(errMsgInfo.size, 4096u);
    EXPECT_EQ(errMsgInfo.taskSrcAddr, 0x3000u);
    EXPECT_EQ(errMsgInfo.taskDstAddr, 0x4000u);
}

TEST_F(TaskErrMsgTest, Ut_GenerateTaskErrMsg_When_WriteWithNotify_Expect_DmaFieldsSet)
{
    auto taskInfo = MakeDfxTaskInfo(static_cast<u8>(Hccl::TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY));
    taskInfo.taskPara.ubDma.notifyId = 1300;
    auto exceptionInfo = MakeExceptionInfo();
    Hccl::ErrorMessageReport errMsgInfo{};

    HcclCommTaskExceptionLite::GetInstance().GenerateTaskErrMsg(taskInfo, errMsgInfo, exceptionInfo);

    EXPECT_EQ(errMsgInfo.notifyId, 1300u);
    EXPECT_EQ(errMsgInfo.reduceType, 255u);
}

TEST_F(TaskErrMsgTest, Ut_GenerateTaskErrMsg_When_Ub_Expect_UbFieldsSet)
{
    auto taskInfo = MakeDfxTaskInfo(static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB));
    taskInfo.taskPara.ubDma.srcAddr = 0x1000;
    taskInfo.taskPara.ubDma.dstAddr = 0x2000;
    taskInfo.taskPara.ubDma.size = 2048;
    taskInfo.linkType = static_cast<u8>(Hccl::DfxLinkType::UB);
    auto exceptionInfo = MakeExceptionInfo(0x1234);
    Hccl::ErrorMessageReport errMsgInfo{};

    HcclCommTaskExceptionLite::GetInstance().GenerateTaskErrMsg(taskInfo, errMsgInfo, exceptionInfo);

    EXPECT_EQ(errMsgInfo.linkType, Hccl::DfxLinkType::UB);
    EXPECT_EQ(errMsgInfo.size, 2048u);
    EXPECT_EQ(errMsgInfo.ubCqeStatus, 0x34u);
    EXPECT_EQ(errMsgInfo.reduceType, 255u);
    EXPECT_EQ(errMsgInfo.taskSrcAddr, 0x1000u);
    EXPECT_EQ(errMsgInfo.taskDstAddr, 0x2000u);
}

TEST_F(TaskErrMsgTest, Ut_GenerateTaskErrMsg_When_Sdma_Expect_BasicDmaFieldsSet)
{
    auto taskInfo = MakeDfxTaskInfo(static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA));
    taskInfo.linkType = static_cast<u8>(Hccl::DfxLinkType::ROCE);
    auto exceptionInfo = MakeExceptionInfo();
    Hccl::ErrorMessageReport errMsgInfo{};

    HcclCommTaskExceptionLite::GetInstance().GenerateTaskErrMsg(taskInfo, errMsgInfo, exceptionInfo);

    EXPECT_EQ(errMsgInfo.linkType, Hccl::DfxLinkType::ROCE);
    EXPECT_EQ(errMsgInfo.reduceType, 255u);
}

TEST_F(TaskErrMsgTest, Ut_GetTaskParam_When_NotifyWait_Expect_NotifyFieldsSet)
{
    Hccl::ErrorMessageReport errMsgInfo{};
    errMsgInfo.taskType = Hccl::TaskParamType::TASK_NOTIFY_WAIT;
    errMsgInfo.notifyId = 100;
    errMsgInfo.notifyValue = 200;

    Hccl::TaskParam taskParam{};
    taskParam.taskType = errMsgInfo.taskType;
    GetTaskParam(taskParam, errMsgInfo);

    EXPECT_EQ(taskParam.taskPara.Notify.notifyID, 100u);
    EXPECT_EQ(taskParam.taskPara.Notify.value, 200u);
}

TEST_F(TaskErrMsgTest, Ut_GetTaskParam_When_NotifyRecord_Expect_NotifyFieldsSet)
{
    Hccl::ErrorMessageReport errMsgInfo{};
    errMsgInfo.taskType = Hccl::TaskParamType::TASK_NOTIFY_RECORD;
    errMsgInfo.notifyId = 300;
    errMsgInfo.notifyValue = 400;

    Hccl::TaskParam taskParam{};
    taskParam.taskType = errMsgInfo.taskType;
    GetTaskParam(taskParam, errMsgInfo);

    EXPECT_EQ(taskParam.taskPara.Notify.notifyID, 300u);
    EXPECT_EQ(taskParam.taskPara.Notify.value, 400u);
}

TEST_F(TaskErrMsgTest, Ut_GetTaskParam_When_UbReduceInline_Expect_ReduceFieldsSet)
{
    Hccl::ErrorMessageReport errMsgInfo{};
    errMsgInfo.taskType = Hccl::TaskParamType::TASK_UB_REDUCE_INLINE;
    errMsgInfo.notifyId = 500;
    errMsgInfo.notifyValue = 600;
    errMsgInfo.linkType = Hccl::DfxLinkType::HCCS;
    errMsgInfo.size = 1024;
    errMsgInfo.taskSrcAddr = 0x1000;
    errMsgInfo.taskDstAddr = 0x2000;

    Hccl::TaskParam taskParam{};
    taskParam.taskType = errMsgInfo.taskType;
    GetTaskParam(taskParam, errMsgInfo);

    EXPECT_EQ(taskParam.taskPara.Reduce.notifyID, 500u);
    EXPECT_EQ(taskParam.taskPara.Reduce.notifyValue, 600u);
    EXPECT_EQ(taskParam.taskPara.Reduce.linkType, Hccl::DfxLinkType::HCCS);
    EXPECT_EQ(taskParam.taskPara.Reduce.size, 1024u);
    EXPECT_EQ(taskParam.taskPara.Reduce.src, reinterpret_cast<void*>(0x1000));
    EXPECT_EQ(taskParam.taskPara.Reduce.dst, reinterpret_cast<void*>(0x2000));
}

TEST_F(TaskErrMsgTest, Ut_GetTaskParam_When_WriteReduceWithNotify_Expect_ReduceFieldsSet)
{
    Hccl::ErrorMessageReport errMsgInfo{};
    errMsgInfo.taskType = Hccl::TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY;
    errMsgInfo.notifyId = 700;
    errMsgInfo.notifyValue = 800;

    Hccl::TaskParam taskParam{};
    taskParam.taskType = errMsgInfo.taskType;
    GetTaskParam(taskParam, errMsgInfo);

    EXPECT_EQ(taskParam.taskPara.Reduce.notifyID, 700u);
    EXPECT_EQ(taskParam.taskPara.Reduce.notifyValue, 800u);
}

TEST_F(TaskErrMsgTest, Ut_GetTaskParam_When_ReduceInline_Expect_OnlyReduceTypeSet)
{
    Hccl::ErrorMessageReport errMsgInfo{};
    errMsgInfo.taskType = Hccl::TaskParamType::TASK_REDUCE_INLINE;
    errMsgInfo.reduceType = static_cast<uint32_t>(HcclReduceOp::HCCL_REDUCE_PROD);

    Hccl::TaskParam taskParam{};
    taskParam.taskType = errMsgInfo.taskType;
    GetTaskParam(taskParam, errMsgInfo);

    EXPECT_EQ(taskParam.taskPara.Reduce.reduceOp, HcclReduceOp::HCCL_REDUCE_PROD);
    EXPECT_EQ(taskParam.taskPara.Reduce.notifyID, 0u);
    EXPECT_EQ(taskParam.taskPara.Reduce.notifyValue, 0u);
}

TEST_F(TaskErrMsgTest, Ut_GetTaskParam_When_UbInlineWrite_Expect_DmaFieldsSet)
{
    Hccl::ErrorMessageReport errMsgInfo{};
    errMsgInfo.taskType = Hccl::TaskParamType::TASK_UB_INLINE_WRITE;
    errMsgInfo.notifyId = 1100;
    errMsgInfo.notifyValue = 1200;
    errMsgInfo.linkType = Hccl::DfxLinkType::ROCE;
    errMsgInfo.size = 4096;
    errMsgInfo.taskSrcAddr = 0x3000;
    errMsgInfo.taskDstAddr = 0x4000;

    Hccl::TaskParam taskParam{};
    taskParam.taskType = errMsgInfo.taskType;
    GetTaskParam(taskParam, errMsgInfo);

    EXPECT_EQ(taskParam.taskPara.DMA.notifyID, 1100u);
    EXPECT_EQ(taskParam.taskPara.DMA.notifyValue, 1200u);
    EXPECT_EQ(taskParam.taskPara.DMA.linkType, Hccl::DfxLinkType::ROCE);
    EXPECT_EQ(taskParam.taskPara.DMA.size, 4096u);
    EXPECT_EQ(taskParam.taskPara.DMA.src, reinterpret_cast<void*>(0x3000));
    EXPECT_EQ(taskParam.taskPara.DMA.dst, reinterpret_cast<void*>(0x4000));
}

TEST_F(TaskErrMsgTest, Ut_GetTaskParam_When_WriteWithNotify_Expect_DmaFieldsSet)
{
    Hccl::ErrorMessageReport errMsgInfo{};
    errMsgInfo.taskType = Hccl::TaskParamType::TASK_WRITE_WITH_NOTIFY;
    errMsgInfo.notifyId = 1300;
    errMsgInfo.notifyValue = 1400;

    Hccl::TaskParam taskParam{};
    taskParam.taskType = errMsgInfo.taskType;
    GetTaskParam(taskParam, errMsgInfo);

    EXPECT_EQ(taskParam.taskPara.DMA.notifyID, 1300u);
    EXPECT_EQ(taskParam.taskPara.DMA.notifyValue, 1400u);
}

TEST_F(TaskErrMsgTest, Ut_GetTaskParam_When_Ub_Expect_UbFieldsSet)
{
    Hccl::ErrorMessageReport errMsgInfo{};
    errMsgInfo.taskType = Hccl::TaskParamType::TASK_UB;
    errMsgInfo.linkType = Hccl::DfxLinkType::UB;
    errMsgInfo.size = 2048;
    errMsgInfo.taskSrcAddr = 0x1000;
    errMsgInfo.taskDstAddr = 0x2000;

    Hccl::TaskParam taskParam{};
    taskParam.taskType = errMsgInfo.taskType;
    GetTaskParam(taskParam, errMsgInfo);

    EXPECT_EQ(taskParam.taskPara.DMA.notifyID, 0u);
    EXPECT_EQ(taskParam.taskPara.DMA.notifyValue, 0u);
    EXPECT_EQ(taskParam.taskPara.DMA.linkType, Hccl::DfxLinkType::UB);
    EXPECT_EQ(taskParam.taskPara.DMA.size, 2048u);
    EXPECT_EQ(taskParam.taskPara.DMA.src, reinterpret_cast<void*>(0x1000));
    EXPECT_EQ(taskParam.taskPara.DMA.dst, reinterpret_cast<void*>(0x2000));
}

TEST_F(TaskErrMsgTest, Ut_GetTaskParam_When_Sdma_Expect_BasicDmaFieldsSet)
{
    Hccl::ErrorMessageReport errMsgInfo{};
    errMsgInfo.taskType = Hccl::TaskParamType::TASK_SDMA;
    errMsgInfo.linkType = Hccl::DfxLinkType::ROCE;
    errMsgInfo.size = 8192;
    errMsgInfo.taskSrcAddr = 0x5000;
    errMsgInfo.taskDstAddr = 0x6000;

    Hccl::TaskParam taskParam{};
    taskParam.taskType = errMsgInfo.taskType;
    GetTaskParam(taskParam, errMsgInfo);

    EXPECT_EQ(taskParam.taskPara.DMA.notifyID, 0u);
    EXPECT_EQ(taskParam.taskPara.DMA.notifyValue, 0u);
    EXPECT_EQ(taskParam.taskPara.DMA.linkType, Hccl::DfxLinkType::ROCE);
    EXPECT_EQ(taskParam.taskPara.DMA.size, 8192u);
    EXPECT_EQ(taskParam.taskPara.DMA.src, reinterpret_cast<void*>(0x5000));
    EXPECT_EQ(taskParam.taskPara.DMA.dst, reinterpret_cast<void*>(0x6000));
}
