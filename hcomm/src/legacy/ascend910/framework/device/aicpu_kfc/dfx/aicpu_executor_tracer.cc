/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_executor_tracer.h"

#include "hccl_msg.h"
#include "common/aicpu_hccl_common.h"
#include "mc2_trace_utils.h"
#include "utils/aicpu_hdc_utils.h"
#include "framework/aicpu_kfc_process.h"

using HcclApi::HCCL_MSG_CNT;
using HcclApi::HcclMsgArea;
namespace dfx_tracer {

// recv host stop command
void AicpuExecutorTracer::HandleBackGround(AicpuComContext* const ctx)
{
    BackgroundCommand bgCmd;
    if (ctx->commOpenStatus) {
        (void)AicpuHdcUtils::GetBackGroundCommand(ctx->kfcControlTransferH2D, bgCmd);
        if (bgCmd == BackgroundCommand::kStop) {
            HCCL_RUN_INFO("ctx stop back ground");
            StopKfcThread(ctx, {});
            KfcExecStatus bgresponse;
            bgresponse.execStatus.backgroundStatus = BackgroundStatus::kStop;
            ctx->commOpenStatus = false;
            AicpuUpdatComContextMumber(offsetof(AicpuComContext, alreadyInit), false);
            (void)MC2TraceUtils::DestoryHandles();
            (void)AicpuHdcUtils::ResponseBackGroundStatus(ctx->kfcStatusTransferD2H, bgresponse);
            bgCmd = BackgroundCommand::kNone;
        }
    }
}

// handle StopLaunch Command
void AicpuExecutorTracer::StopLaunchCommandHandle(AicpuComContext* const ctx)
{
    if (ctx->commOpenStatus) {
        if (!ctx->endStopLaunch) {
            KfcCommand cmd = KfcCommand::kNone;
            (void)AicpuHdcUtils::GetOpExecCtrlCmd(ctx->kfcControlTransferH2D, cmd);
            if (cmd == KfcCommand::NsStopLaunch) {
                if (!ctx->isOpLaunch) {
                    (void)AicpuHdcUtils::SetOpExecStatus(
                        ctx->kfcStatusTransferD2H, KfcStatus::kStoplaunch, KfcError::kNone, 0);
                    AicpuUpdatComContextMumber(offsetof(AicpuComContext, endStopLaunch), true);
                    HCCL_INFO("[NsRecovery][backGround]send in mc2 environment");
                }
            }
        }
    }
}

// handle StopExec and Clean Command
void AicpuExecutorTracer::KfcCommandHandle(AicpuComContext* const ctx)
{
    if (ctx->commOpenStatus) {
        using CommandCall = std::function<void(AicpuComContext* const ctx)>;
        static std::map<KfcCommand, CommandCall> commandHandles = {
            {KfcCommand::NsStopExec, KfcCommandHandles::StopFunc}, {KfcCommand::NsClear, KfcCommandHandles::ClearFunc}};

        KfcCommand cmd = KfcCommand::kNone;
        (void)AicpuHdcUtils::GetOpExecCtrlCmd(ctx->kfcControlTransferH2D, cmd);
        auto iter = commandHandles.find(cmd);
        if (iter == commandHandles.cend()) {
            return;
        }
        HCCL_INFO("Start to run command %ld", cmd);
        iter->second(ctx);
    }
}

void AicpuExecutorTracer::HandleCqeStatus(AicpuComContext* const ctx)
{
    if (ctx == nullptr || ctx->alreadyInit == false) {
        return;
    }

    if (ctx->dfxExtendInfo.cqeStatus != dfx::CqeStatus::kDefault) {
        return;
    }

    if (ctx->dfxExtendInfo.kfcStatus != DfxKfcStatus::kOneStart) {
        return;
    }

    // 遍历每个卡的cqe状态, 如果一个卡有异常，所有的卡的ctx里的轮询状态都会被标记为异常
    // rankNum前面的逻辑保证了是大于0的, 多机场景取1，非多机场景取rankNum
    const u32 streamNum = ctx->multiServerFlag ? 1U : ctx->rankNum;
    for (uint32_t rank = 0U; rank <= streamNum - 1; ++rank) {
        HandleCqeStatusByRank(ctx, rank);
    }
}

// stop 主线程
void AicpuExecutorTracer::StopKfcThread(
    AicpuComContext* const ctx, std::vector<std::pair<std::string, hccl::HcclCommAicpu*>> aicpuCommInfo)
{
    const uint64_t waitTime = static_cast<uint64_t>(NSEC_PER_SEC) * 10U; // 10s
    uint64_t startTime = GetCurCpuTimestamp();
    while (ctx->isRunning) {
        if (!aicpuCommInfo.empty()) {
            for (auto& commInfo : aicpuCommInfo) {
                hccl::HcclCommAicpu* hcclAicpu = commInfo.second;
                DfxExtendInfo* dfxInfo = hcclAicpu->GetDfxExtendInfo();
                dfxInfo->pollStatus = PollStatus::kStopAsException;
            }
        } else {
            AicpuUpdatComContextMumber(
                offsetof(AicpuComContext, dfxExtendInfo.pollStatus), PollStatus::kStopAsException);
        }

        if ((GetCurCpuTimestamp() - startTime) >= waitTime) {
            HCCL_ERROR("stop kfc thread timeout [10s]");
            break;
        }
    }
}

void AicpuExecutorTracer::HandleCqeStatusByRank(AicpuComContext* const ctx, uint32_t rank)
{
    const HcclComStreamInfo& streamInfo = ctx->streamInfo[rank];
    CqeQueryInput cqeQueryInput;
    SetCqeQueryInput(ctx->devId, streamInfo, cqeQueryInput);
    rtLogicCqReport_t report[AC_SQE_REV_MAX_CNT];
    cqeQueryInput.cqeAddr = reinterpret_cast<uint8_t*>(report); // 用于存放接收到的cq
    rtLogicCqReport_t cqeException;
    CqeStatus cqeStatus = CqReportRecv(cqeQueryInput, cqeException);
    if (cqeStatus != dfx::CqeStatus::kDefault) {
        (void)MC2TraceUtils::Save();
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.cqeStatus), cqeStatus);
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.pollStatus), PollStatus::kStopAsException);
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.cqeException.sqeType), cqeException.sqeType);
        HCCL_ERROR(
            "After send sqe:%d, exception happened on rank %u, cqeStatus[%d], sqetype[%u]", streamInfo.sqId, rank,
            cqeStatus, cqeException.sqeType);
    }

    if (cqeStatus == dfx::CqeStatus::kCqeException) {
        PrintTaskException(cqeException);
    }
}

void AicpuExecutorTracer::PrintTaskException(const rtLogicCqReport_t& reportOfOne)
{
    const std::vector<std::string> StarsCqeErrorDesc
        = {"task exception",          "task trap",          "task timeout", "sqe error",
           "resource conflict error", "sq sw status error", "warning"};
    uint32_t errBit = static_cast<uint32_t>(getTrailingZeros(reportOfOne.errorType));
    const char* const errMsg = errBit < StarsCqeErrorDesc.size() ? StarsCqeErrorDesc[errBit].c_str() : "unknown";
    uint32_t idx = AicpuKfcProcess::GetStreamRankIdx(reportOfOne.streamId);
    SqeInfo sqeInfo;
    (void)AicpuSqeContext::QuerySqeInfoByTaskId(idx, reportOfOne.taskId, &sqeInfo);
    HCCL_ERROR(
        "Task run failed of exception, errorType [%u] error msg:[%s] sqe info:[%s]", errBit, errMsg,
        AicpuSqeContext::GetString(sqeInfo).c_str());
}

uint8_t AicpuExecutorTracer::getTrailingZeros(uint8_t num)
{
    uint8_t count = 0;
    while ((num & 1U) == 0) {
        count++;
        num >>= 1;
        if (num == 1U) {
            break;
        }
    }
    return count;
}

void AicpuExecutorTracer::SetCqeQueryInput(
    const uint32_t devId, const HcclComStreamInfo& streamInfo, CqeQueryInput& cqeQueryInput)
{
    cqeQueryInput.devId = devId;
    cqeQueryInput.streamId = streamInfo.actualStreamId;
    cqeQueryInput.sqId = streamInfo.sqId;
    cqeQueryInput.cqId = streamInfo.logicCqId;
    cqeQueryInput.type = static_cast<uint32_t>(DRV_LOGIC_TYPE);
}

void KfcCommandHandles::StopFunc(AicpuComContext* const ctx)
{
    HCCL_INFO("StopFunc, current clusterId:%d", ctx->clusterId);
    if (ctx->isStopLaunch) {
        // kill 流
        if ((StreamsKill(ctx->devId) != HCCL_SUCCESS)
            || (DeviceQuery(ctx->devId, ts::APP_ABORT_KILL_FINISH, 0U) != HCCL_SUCCESS)) {
            (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kError, KfcError::kExec, 0);
            AicpuUpdatComContextMumber(offsetof(AicpuComContext, isStopLaunch), false);
            return;
        }

        // 停止条件算子
        HcclMsgArea* hcclMsgArea = reinterpret_cast<HcclMsgArea*>(ctx->workSpaceAddr);
        for (uint32_t i = 0; i < HCCL_MSG_CNT; i++) {
            hcclMsgArea->commMsg.singleMsg.commitTurnCnt[i].cnt = 0xFF;
        }
        (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kStopExec, KfcError::kNone, 0);
    } else {
        (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kEnd, KfcError::kNone, 0);
    }
    HCCL_INFO("StopFunc Finish");
}

void KfcCommandHandles::ClearCq(AicpuComContext* const ctx)
{
    const u32 streamNum = ctx->multiServerFlag ? 1U : ctx->rankNum;
    for (u32 i = 0; i < streamNum; i++) {
        const HcclComStreamInfo& streamInfo = ctx->streamInfo[i];
        HCCL_INFO("ClearFunc, sqid:%d", streamInfo.sqId);
        if (ConfigSqStatusByType(ctx->devId, streamInfo.sqId, DRV_SQCQ_PROP_SQ_DISABLE_TO_ENABLE, 1) != HCCL_SUCCESS) {
            (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kError, KfcError::kExec, 0);
            return;
        }
        CqeQueryInput cqeQueryInput;
        AicpuExecutorTracer::SetCqeQueryInput(ctx->devId, streamInfo, cqeQueryInput);
        rtLogicCqReport_t report[AC_SQE_REV_MAX_CNT];
        cqeQueryInput.cqeAddr = reinterpret_cast<uint8_t*>(report); // 用于存放接收到的cq
        rtLogicCqReport_t cqeException;
        (void)CqReportRecv(cqeQueryInput, cqeException);
    }
}

void KfcCommandHandles::ClearFunc(AicpuComContext* const ctx)
{
    if (ctx->isStopLaunch) {
        // 等待drv任务停止
        if (DeviceQuery(ctx->devId, ts::APP_ABORT_TERMINATE_FINISH, 0U) != HCCL_SUCCESS) {
            (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kError, KfcError::kExec, 0);
            return;
        }
        HCCL_INFO("ClearFunc, after APP_ABORT_TERMINATE_FINISH");
        // 使能sq,读清cq
        ClearCq(ctx);

        // 清理dfxExtendInfo状态
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.kfcStatus), DfxKfcStatus::kDefault);
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.cqeStatus), dfx::CqeStatus::kDefault);
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.pollStatus), PollStatus::kDefault);

        // 清理SqeContext
        AicpuSqeContext::SyncVariable();
        AicpuSqeContext::SaveVariable();
        if (AicpuSqeContext::ClearLocalBuff() != HCCL_SUCCESS) {
            (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kError, KfcError::kExec, 0);
            return;
        }
        SqeContext* sqeContext = GetSqeContext();
        const u32 streamNum = ctx->multiServerFlag ? 1U : ctx->rankNum;
        for (u32 i = 0; i < streamNum; i++) {
            auto& buff = sqeContext->buffPtr[i];
            if ((QuerySqStatusByType(ctx->devId, ctx->streamInfo[i].sqId, DRV_SQCQ_PROP_SQ_TAIL, buff.sqTail)
                 != HCCL_SUCCESS)
                || (QuerySqStatusByType(ctx->devId, ctx->streamInfo[i].sqId, DRV_SQCQ_PROP_SQ_HEAD, buff.sqHead)
                    != HCCL_SUCCESS)) {
                (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kError, KfcError::kExec, 0);
                return;
            }
            HCCL_INFO(
                "hccl aicpu reset stream buffer, sqid:%d head:%u tail:%u.", ctx->streamInfo[i].sqId, buff.sqHead,
                buff.sqTail);
        }
    }
    if (!ctx->ibversData.empty()) {
        (void)memset_s(
            reinterpret_cast<void*>(ctx->ibversData[ctx->rankId].localInputMem.addr),
            ctx->ibversData[ctx->rankId].localInputMem.size, 0, ctx->ibversData[ctx->rankId].localInputMem.size);
        (void)memset_s(
            reinterpret_cast<void*>(ctx->ibversData[ctx->rankId].localOutputMem.addr),
            ctx->ibversData[ctx->rankId].localOutputMem.size, 0, ctx->ibversData[ctx->rankId].localOutputMem.size);
    }
    // 清理共享内存
    (void)memset_s(reinterpret_cast<void*>(ctx->workSpaceAddr), sizeof(HcclMsgArea), 0, sizeof(HcclMsgArea));

    AicpuUpdatComContextMumber(offsetof(AicpuComContext, isOpLaunch), false);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, endStopLaunch), false);
    if (ctx->isStopLaunch) {
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, isStopLaunch), false);
        (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kClear, KfcError::kNone, 0);
    } else {
        (void)AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kEnd, KfcError::kNone, 0);
    }
    HCCL_INFO("ClearFunc Finish");
}
} // namespace dfx_tracer
