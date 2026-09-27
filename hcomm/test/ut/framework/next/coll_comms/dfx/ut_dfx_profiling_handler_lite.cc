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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include <string>
#define private public
#define protected public
#include "dfx_profiling_handler_lite.h"
#include "res_pub.h"
#include "sqe_a5.h"
#undef private
#undef protected

using namespace Hccl;

namespace aicpu {
enum DfxStatusT : uint8_t { AICPU_ERROR_NONE = 0, AICPU_ERROR_FAILED = 1 };
DfxStatusT __attribute__((weak)) GetTaskAndStreamId(uint64_t& taskId, uint32_t& streamId) { return AICPU_ERROR_NONE; }
} // namespace aicpu

class DfxProfilingHandlerLiteTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DfxProfilingHandlerLiteTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "DfxProfilingHandlerLiteTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in DfxProfilingHandlerLiteTest SetUP" << std::endl;
        handler_.enableHcclL0_ = false;
        handler_.enableHcclL1_ = false;
        handler_.initializedFlag_ = false;
        handler_.cachedAlgTypeHashId_ = 0;
        handler_.taskTypeHashCache_.clear();
        handler_.opTypeHashCache_.clear();
        handler_.algTypeHashCache_.clear();
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in DfxProfilingHandlerLiteTest TearDown" << std::endl;
    }

    DfxProfilingHandlerLite& handler_ = DfxProfilingHandlerLite::GetInstance();
};

static void PrepareHandlerInit(DfxProfilingHandlerLite& handler)
{
    handler.initializedFlag_ = false;
    handler.Init();
}

static DfxCommContext MakeDefaultCtx() { return {nullptr, DFX_INVALID_U64, INVALID_U32, 0}; }

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetInstance_Expect_ReturnSameInstance)
{
    DfxProfilingHandlerLite& inst1 = DfxProfilingHandlerLite::GetInstance();
    DfxProfilingHandlerLite& inst2 = DfxProfilingHandlerLite::GetInstance();
    EXPECT_EQ(&inst1, &inst2);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_Init_When_NotInit_Expect_ReturnSuccess)
{
    handler_.initializedFlag_ = false;
    HcclResult ret = handler_.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(handler_.initializedFlag_, true);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_Init_When_AlreadyInit_Expect_ReturnSuccessDirectly)
{
    handler_.initializedFlag_ = true;
    HcclResult ret = handler_.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportHcclOpInfo_When_L0Off_Expect_EarlyReturn)
{
    handler_.enableHcclL0_ = false;
    DfxDfxOpInfo opInfo{};
    EXPECT_NO_THROW(handler_.ReportHcclOpInfo(opInfo, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportHcclOpInfo_When_L0On_Expect_RunToEnd)
{
    PrepareHandlerInit(handler_);
    handler_.enableHcclL0_ = true;
    DfxCommContext ctx = MakeDefaultCtx();
    ctx.groupName = 999;
    ctx.rankSize = 4;
    DfxDfxOpInfo opInfo{};
    opInfo.count = 100;
    opInfo.dataType = 1;
    EXPECT_NO_THROW(handler_.ReportHcclOpInfo(opInfo, ctx));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportMainStreamTask_When_L0Off_Expect_EarlyReturn)
{
    handler_.enableHcclL0_ = false;
    DfxFlagTaskInfo flagTaskInfo{};
    EXPECT_NO_THROW(handler_.ReportMainStreamTask(flagTaskInfo));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportMainStreamTask_When_L0On_Expect_RunToEnd)
{
    PrepareHandlerInit(handler_);
    handler_.enableHcclL0_ = true;
    DfxFlagTaskInfo flagTaskInfo{};
    flagTaskInfo.taskId = 0x00010002;
    flagTaskInfo.type = DfxMainStreamTaskType::TAIL;
    EXPECT_NO_THROW(handler_.ReportMainStreamTask(flagTaskInfo));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportAdditionInfo_When_Normal_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    MsprofAdditionalInfo reporterData{};
    EXPECT_NO_THROW(handler_.ReportAdditionInfo(reporterData));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillBatchReporterData_When_Normal_Expect_ReturnTrue)
{
    PrepareHandlerInit(handler_);
    MsprofAicpuHcclTaskInfo taskInfos[2] = {};
    MsprofAdditionalInfo addInfo{};
    bool ret = handler_.FillBatchReporterData(1, taskInfos, addInfo);
    EXPECT_EQ(ret, true);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillBatchReporterData_When_ZeroBatch_Expect_ReturnTrue)
{
    MsprofAicpuHcclTaskInfo taskInfos[1] = {};
    MsprofAdditionalInfo addInfo{};
    bool ret = handler_.FillBatchReporterData(0, taskInfos, addInfo);
    (void)ret;
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportBatchAddInfo_When_NotLast_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    MsprofAicpuHcclTaskInfo taskInfos[2] = {};
    MsprofAdditionalInfo addInfoVec[4] = {};
    uint32_t addInfoIndx = 0;
    bool ret = handler_.ReportBatchAddInfo(1, taskInfos, addInfoVec, addInfoIndx, 4, false);
    (void)ret;
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportBatchAddInfo_When_IsLast_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    MsprofAicpuHcclTaskInfo taskInfos[2] = {};
    MsprofAdditionalInfo addInfoVec[4] = {};
    uint32_t addInfoIndx = 0;
    bool ret = handler_.ReportBatchAddInfo(1, taskInfos, addInfoVec, addInfoIndx, 4, true);
    (void)ret;
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_UpdateProfSwitch_Expect_NoThrow)
{
    EXPECT_NO_THROW(handler_.UpdateProfSwitch());
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_IsProfOn_When_UnknownFeature_Expect_False)
{
    EXPECT_EQ(handler_.IsProfOn(0xFFFF), false);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_IsProfSwitchOn_When_L1Off_Expect_FalseAndFlagSet)
{
    handler_.enableHcclL1_ = true;
    bool ret = handler_.IsProfSwitchOn(DfxProfilingLevel::L1);
    EXPECT_EQ(ret, false);
    EXPECT_EQ(handler_.enableHcclL1_, false);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_SetProL0On_Expect_FlagSet)
{
    handler_.SetProL0On(true);
    EXPECT_EQ(handler_.enableHcclL0_, true);
    handler_.SetProL0On(false);
    EXPECT_EQ(handler_.enableHcclL0_, false);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_SetProL1On_Expect_FlagSet)
{
    handler_.SetProL1On(true);
    EXPECT_EQ(handler_.enableHcclL1_, true);
    handler_.SetProL1On(false);
    EXPECT_EQ(handler_.enableHcclL1_, false);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetProfHashId_When_NullName_Expect_ReturnInvalid)
{
    uint64_t hashId = handler_.GetProfHashId(nullptr, 10);
    EXPECT_EQ(hashId, DFX_INVALID_U64);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetProfHashId_When_ZeroLen_Expect_ReturnInvalid)
{
    uint64_t hashId = handler_.GetProfHashId("test", 0);
    EXPECT_EQ(hashId, DFX_INVALID_U64);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetCachedAlgTypeHashId_When_CacheEmpty_Expect_ReturnInvalid)
{
    handler_.cachedAlgTypeHashId_ = DFX_INVALID_U64;
    EXPECT_EQ(handler_.GetCachedAlgTypeHashId(), DFX_INVALID_U64);
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetCachedAlgTypeHashId_When_CacheHasValue_Expect_ReturnValue)
{
    handler_.cachedAlgTypeHashId_ = 55555;
    EXPECT_EQ(handler_.GetCachedAlgTypeHashId(), 55555u);
}

static void FillDfxTaskInfoForType(Hccl::DfxTaskInfo& taskInfo, u8 taskType)
{
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    taskInfo.channelHandle = DFX_INVALID_U64;
    taskInfo.taskType = taskType;
    taskInfo.linkType = 0;
    taskInfo.sqId = 0;
    taskInfo.taskId = 0;
    taskInfo.transportType = 0;
    taskInfo.taskPara.ubDma.notifyId = INVALID_U32;
    taskInfo.taskPara.ubDma.srcAddr = 0;
    taskInfo.taskPara.ubDma.dstAddr = 0;
    taskInfo.taskPara.ubDma.size = 0;
    taskInfo.taskPara.Reduce.notifyId = INVALID_U32;
    taskInfo.taskPara.Reduce.srcAddr = 0;
    taskInfo.taskPara.Reduce.dstAddr = 0;
    taskInfo.taskPara.Reduce.size = 0;
    taskInfo.taskPara.Reduce.reduceOp = 0;
    taskInfo.taskPara.Dma.sqeAddr = 0;
    taskInfo.taskPara.Notify.sqeAddr = 0;
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_Sdma_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    DfxCommContext ctx = MakeDefaultCtx();
    ctx.groupName = 100;
    ctx.localRank = 0;
    ctx.rankSize = 8;
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA));
    taskInfo.linkType = 2;
    taskInfo.sqId = 1;
    taskInfo.taskId = 10;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, ctx));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_Rdma_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_RDMA));
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_ReduceInline_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_REDUCE_INLINE));
    taskInfo.linkType = 1;
    taskInfo.taskPara.Reduce.notifyId = 5;
    taskInfo.taskPara.Reduce.srcAddr = 0x1000;
    taskInfo.taskPara.Reduce.dstAddr = 0x2000;
    taskInfo.taskPara.Reduce.size = 256;
    taskInfo.taskPara.Reduce.reduceOp = 2;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_UbDma_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB));
    taskInfo.linkType = 3;
    taskInfo.taskPara.ubDma.notifyId = 20;
    taskInfo.taskPara.ubDma.srcAddr = 0x3000;
    taskInfo.taskPara.ubDma.dstAddr = 0x4000;
    taskInfo.taskPara.ubDma.size = 1024;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_UbInlineWrite_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB_INLINE_WRITE));
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_UbReduceInline_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB_REDUCE_INLINE));
    taskInfo.taskPara.Reduce.reduceOp = 3;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_WriteWithNotify_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY));
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_WriteReduceWithNotify_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY));
    taskInfo.taskPara.Reduce.reduceOp = 1;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_NotifyRecord_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_RECORD));
    taskInfo.linkType = 0;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_NotifyWait_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT));
    taskInfo.linkType = 0;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_InvalidType_Expect_EarlyReturn)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(200));
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_GetTaskDetailInfosFromDfxTaskInfo_When_WithDfxOpInfo_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA));
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.opType = static_cast<u8>(OpTypeVal::OP_TYPE_ALLREDUCE);
    opInfo.dataType = 1;
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    std::unordered_map<u64, u32> rankMap;
    rankMap[0x5678] = 3;
    DfxCommContext ctx = MakeDefaultCtx();
    ctx.channelRemoteRankIdMap = &rankMap;
    taskInfo.channelHandle = 0x5678;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, ctx));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportStreamTaskDetailsLog_When_EmptyQueue_Expect_NoThrow)
{
    TaskInfoCircularQueue queue;
    EXPECT_NO_THROW(handler_.ReportStreamTaskDetailsLog(queue));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportStreamTaskDetailsLog_When_NonEmptyQueue_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    TaskInfoCircularQueue queue;
    Hccl::DfxTaskInfo* slot = static_cast<Hccl::DfxTaskInfo*>(queue.NextSlot());
    if (slot != nullptr) {
        FillDfxTaskInfoForType(*slot, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA));
    }
    EXPECT_NO_THROW(handler_.ReportStreamTaskDetailsLog(queue));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportStreamTaskDetails_When_EmptyQueue_Expect_EarlyReturn)
{
    TaskInfoCircularQueue queue;
    EXPECT_NO_THROW(handler_.ReportStreamTaskDetails(queue, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportStreamTaskDetails_When_NonEmptyQueue_Expect_RunToEnd)
{
    PrepareHandlerInit(handler_);
    DfxCommContext ctx = MakeDefaultCtx();
    ctx.groupName = 100;
    ctx.localRank = 0;
    ctx.rankSize = 8;
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA));
    taskInfo.linkType = 1;
    taskInfo.sqId = 0;
    taskInfo.taskId = 1;
    Hccl::TaskInfoCircularQueue queue;
    Hccl::DfxTaskInfo* slot = static_cast<Hccl::DfxTaskInfo*>(queue.NextSlot());
    if (slot != nullptr) {
        *slot = taskInfo;
    }
    EXPECT_NO_THROW(handler_.ReportStreamTaskDetails(queue, ctx));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_ReportStreamTaskDetails_When_BatchReport_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    DfxCommContext ctx = MakeDefaultCtx();
    ctx.groupName = 100;
    ctx.localRank = 0;
    ctx.rankSize = 8;
    Hccl::TaskInfoCircularQueue queue;
    for (int i = 0; i < 3; i++) {
        Hccl::DfxTaskInfo* slot = static_cast<Hccl::DfxTaskInfo*>(queue.NextSlot());
        if (slot != nullptr) {
            FillDfxTaskInfoForType(*slot, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA));
            slot->linkType = 1;
            slot->sqId = static_cast<u32>(i);
            slot->taskId = static_cast<u32>(i);
        }
    }
    EXPECT_NO_THROW(handler_.ReportStreamTaskDetails(queue, ctx));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillReduceInlineDetail_When_CalledViaGetDetail_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_REDUCE_INLINE));
    taskInfo.linkType = 1;
    taskInfo.taskPara.Reduce.notifyId = 10;
    taskInfo.taskPara.Reduce.srcAddr = 0x1000;
    taskInfo.taskPara.Reduce.dstAddr = 0x2000;
    taskInfo.taskPara.Reduce.size = 512;
    taskInfo.taskPara.Reduce.reduceOp = 1;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillSdmaRdmaDetail_When_CalledViaGetDetail_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA));
    taskInfo.linkType = 2;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillUbDmaDetail_When_ReduceInline_CalledViaGetDetail_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_UB_REDUCE_INLINE));
    taskInfo.linkType = 3;
    taskInfo.taskPara.ubDma.notifyId = 20;
    taskInfo.taskPara.ubDma.srcAddr = 0x3000;
    taskInfo.taskPara.ubDma.dstAddr = 0x4000;
    taskInfo.taskPara.ubDma.size = 1024;
    taskInfo.taskPara.Reduce.reduceOp = 5;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillNotifyDetail_When_CalledViaGetDetail_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_RECORD));
    taskInfo.linkType = 4;
    taskInfo.taskPara.Notify.sqeAddr = 0;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillDefaultDetail_When_CalledViaGetDetail_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    FillDfxTaskInfoForType(taskInfo, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_CCU));
    taskInfo.linkType = 0;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.GetTaskDetailInfosFromDfxTaskInfo(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillCclTagAndRemoteRank_When_DfxOpInfoInvalid_Expect_NoThrow)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    taskInfo.channelHandle = DFX_INVALID_U64;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.FillCclTagAndRemoteRank(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillCclTagAndRemoteRank_When_MapNullptr_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.opType = 0;
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    taskInfo.channelHandle = 0x1234;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.FillCclTagAndRemoteRank(&taskInfo, taskDetailsInfos, MakeDefaultCtx()));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillCclTagAndRemoteRank_When_ValidMap_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.opType = static_cast<u8>(OpTypeVal::OP_TYPE_ALLREDUCE);
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    std::unordered_map<u64, u32> rankMap;
    rankMap[0x5678] = 3;
    DfxCommContext ctx = MakeDefaultCtx();
    ctx.channelRemoteRankIdMap = &rankMap;
    taskInfo.channelHandle = 0x5678;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.FillCclTagAndRemoteRank(&taskInfo, taskDetailsInfos, ctx));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillCommonTailFields_When_DfxOpInfoInvalid_Expect_NoThrow)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    taskInfo.sqId = 1;
    taskInfo.taskId = 2;
    taskInfo.channelHandle = DFX_INVALID_U64;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.FillCommonTailFields(&taskInfo, taskDetailsInfos));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillCommonTailFields_When_RemoteRankInvalid_Expect_NoThrow)
{
    Hccl::DfxTaskInfo taskInfo{};
    taskInfo.dfxOpInfo = DFX_INVALID_U64;
    taskInfo.sqId = 1;
    taskInfo.taskId = 2;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    taskDetailsInfos.remoteRank = Hccl::DFX_INVALID_RANKID;
    EXPECT_NO_THROW(handler_.FillCommonTailFields(&taskInfo, taskDetailsInfos));
}

TEST_F(DfxProfilingHandlerLiteTest, Ut_FillCommonTailFields_When_WithDfxOpInfo_Expect_NoThrow)
{
    PrepareHandlerInit(handler_);
    Hccl::DfxTaskInfo taskInfo{};
    Hccl::DfxDfxOpInfo opInfo{};
    opInfo.dataType = 2;
    taskInfo.dfxOpInfo = reinterpret_cast<u64>(&opInfo);
    taskInfo.sqId = 5;
    taskInfo.taskId = 10;
    MsprofAicpuHcclTaskInfo taskDetailsInfos{};
    EXPECT_NO_THROW(handler_.FillCommonTailFields(&taskInfo, taskDetailsInfos));
}
