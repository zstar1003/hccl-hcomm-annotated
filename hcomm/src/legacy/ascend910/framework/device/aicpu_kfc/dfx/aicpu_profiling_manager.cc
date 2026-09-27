/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_profiling_manager.h"

#include "aicpu_schedule/aicpu_context.h"
#include "common/aicpu_sqe_context.h"
#include "common/aicpu_hccl_common.h"
#include "profiling_manager_device.h"

namespace dfx {
constexpr std::uint32_t HCCLINFO_REPORT_BATCH_NUM = 2;
uint64_t g_groupNameHashId{0U};

HcclResult AicpuProfilingManager::ReportTaskInfo()
{
    SqeContext* context = GetSqeContext();
    auto ctx = AicpuGetComContext();
    CHK_PTR_NULL(context->buffPtr);
    MsprofAicpuHcclTaskInfo taskInfos[HCCLINFO_REPORT_BATCH_NUM] = {};
    for (uint32_t streamId = 0; streamId < AC_MAX_RANK_NUM; streamId++) {
        auto& buff = context->buffPtr[streamId];
        uint16_t& lastSqeIdx = ctx->profilingExtendInfo.lastSqeIdxs[streamId];
        HCCL_INFO(
            "Rank %u stream %u has %u sqes, sqeCnt: %u, lastSqeIdx: %u", ctx->rankId, streamId, buff.tailSqeIdx,
            buff.sqeCnt, lastSqeIdx);
        auto endIdx = static_cast<uint32_t>(buff.tailSqeIdx);
        for (uint32_t idx = lastSqeIdx, batchId = 0; idx < endIdx; ++idx) {
            auto& taskInfo = taskInfos[batchId++];
            Ctx2MsprofAicpuMC2HcclInfo(ctx, taskInfo);
            taskInfo.planeID = streamId;
            SqeInfo sqeInfo;
            SqeContextUtils::QuerySqeInfo(
                buff.localBuff + idx * AC_SQE_SIZE, buff.sqeType[idx], buff.addInfo[idx], &sqeInfo);
            ProfilingExtendInfoHelper::SqeInfo2MsprofAicpuMC2HcclInfo(sqeInfo, taskInfo);
            taskInfo.timeStamp = buff.profTimestap[idx]; // 时间戳
            ProfilingManager::DumpHcclInfo(taskInfo, batchId, idx);
            if (batchId == HCCLINFO_REPORT_BATCH_NUM || idx == (endIdx - 1)) {
                CHK_PRT(ProfilingManager::CallMsprofReportAdditionInfo(
                    MSPROF_REPORT_AICPU_MC2_BATCH_HCCL_INFO, 0, taskInfos, sizeof(MsprofAicpuHcclTaskInfo) * batchId));
                batchId = 0;
                CHK_SAFETY_FUNC_RET(memset_s(taskInfos, sizeof(taskInfos), 0, sizeof(taskInfos)));
            }
        }
        lastSqeIdx = buff.tailSqeIdx;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuProfilingManager::ReportTaskExecTimeLine(AicpuComProf* acprof, u32 turnOffset)
{
    if (!ProfilingManager::GetProfL1State()) {
        return HCCL_SUCCESS;
    }
    uint64_t taskId = 0U;
    uint32_t streamId = 0;
    if (AicpuGetStreamId == nullptr || AicpuGetTaskId == nullptr) {
        CHK_PRT_RET(
            aicpu::GetTaskAndStreamId(taskId, streamId) != aicpu::status_t::AICPU_ERROR_NONE,
            HCCL_ERROR("Failed to get task id and stream id."), HCCL_E_PARA);
    } else {
        streamId = AicpuGetStreamId();
        taskId = AicpuGetTaskId();
    }
    HCCL_INFO("[AicpuProfilingManager] [ReportTaskExecTimeLine] streamId = %u, taskId = %u", streamId, taskId);

    // 正常一个kfc算子的展开任务不会超过`AC_MAX_PROF_COMM_CNT`轮
    u32 workRcdCnt = acprof->workCnt;
    CHK_PRT_RET(
        workRcdCnt > AC_MAX_PROF_COMM_CNT,
        HCCL_ERROR("WorkRcdCnt %u should not bigger than %u,", workRcdCnt, AC_MAX_PROF_COMM_CNT), HCCL_E_PARA);

    for (u32 j = 0; j < workRcdCnt; j++) {
        AicpuComProfCommLoop* commRcd = &acprof->commLoop[j];
        AicpuKfcProfCommTurn commTurnProf;
        commTurnProf.serverStartTime = acprof->launchEntryTime;
        commTurnProf.waitMsgStartTime = acprof->commInitEndTime;
        commTurnProf.kfcAlgExeStartTime = commRcd->hccExecStartTime;
        commTurnProf.sendTaskStartTime = commRcd->sendTaskStartTime;
        commTurnProf.sendSqeFinishTime = commRcd->sendSqeFinishTime;
        commTurnProf.rtsqExeEndTime = acprof->receiveFinalizeTime;
        commTurnProf.serverEndTime = acprof->endTime;
        commTurnProf.dataLen = commRcd->dataLen;

        commTurnProf.deviceId = acprof->rankId;
        commTurnProf.streamId = streamId;
        commTurnProf.taskId = taskId;
        commTurnProf.commTurn = workRcdCnt;
        commTurnProf.currentTurn = j + turnOffset;
        commTurnProf.version = 0U; // default
        HCCL_INFO("Get one comm prof with details:[%s]", AicpuKfcProfCommTurnToString(commTurnProf).c_str());
        CHK_PRT(ProfilingManager::CallMsprofReportAdditionInfo(
            MSPROF_REPORT_AICPU_MC2_EXECUTE_COMM_TIME, GetCurCpuTimestamp(true), &commTurnProf, sizeof(commTurnProf)));
    }
    return HCCL_SUCCESS;
}

void AicpuProfilingManager::Init(const AicpuComContext* ctx)
{
    if (AdprofGetHashId == nullptr) {
        HCCL_INFO("AdprofGetHashId is null, just return");
        return;
    }
    ProfilingExtendInfoHelper::InitProfItemId();
    // ctx保证了非空
    u32 len = 0;                       // 初始化长度为0
    while (ctx->hcomId[len] != '\0') { // 遍历数组，直到遇到'\0'
        len++;                         // 每遍历一个字符，长度加1
    }
    g_groupNameHashId = AdprofGetHashId(ctx->hcomId, len);
    HCCL_DEBUG("Using %s and len %u to get hash %lu", ctx->hcomId, len, g_groupNameHashId);
}

void AicpuProfilingManager::Ctx2MsprofAicpuMC2HcclInfo(
    const AicpuComContext* ctx, MsprofAicpuHcclTaskInfo& msprofAicpuMC2HcclInfo)
{
    ProfilingExtendInfoHelper::InitHcclInfo(msprofAicpuMC2HcclInfo);
    msprofAicpuMC2HcclInfo.groupName = g_groupNameHashId;
    msprofAicpuMC2HcclInfo.localRank = ctx->rankId;
    msprofAicpuMC2HcclInfo.remoteRank = ctx->rankId;
    msprofAicpuMC2HcclInfo.rankSize = ctx->rankNum;
    msprofAicpuMC2HcclInfo.opType = static_cast<uint32_t>(ctx->reducekind);
}

std::string AicpuProfilingManager::AicpuKfcProfCommTurnToString(const AicpuKfcProfCommTurn& aicpuKfcProfCommTurn)
{
    std::stringstream ss;
    ss << "serverStartTime: " << aicpuKfcProfCommTurn.serverStartTime;
    ss << " waitMsgStartTime: " << aicpuKfcProfCommTurn.waitMsgStartTime;
    ss << " kfcAlgExeStartTime: " << aicpuKfcProfCommTurn.kfcAlgExeStartTime;
    ss << " sendTaskStartTime: " << aicpuKfcProfCommTurn.sendTaskStartTime;
    ss << " sendSqeFinishTime: " << aicpuKfcProfCommTurn.sendSqeFinishTime;
    ss << " rtsqExeEndTime: " << aicpuKfcProfCommTurn.rtsqExeEndTime;
    ss << " serverEndTime: " << aicpuKfcProfCommTurn.serverEndTime;
    ss << " dataLen: " << aicpuKfcProfCommTurn.dataLen;
    ss << " deviceId: " << aicpuKfcProfCommTurn.deviceId;
    ss << " streamId: " << aicpuKfcProfCommTurn.streamId;
    ss << " taskId: " << aicpuKfcProfCommTurn.taskId;
    ss << " version: " << aicpuKfcProfCommTurn.version;
    ss << " total commTurn: " << aicpuKfcProfCommTurn.commTurn;
    ss << " currentTurn: " << aicpuKfcProfCommTurn.currentTurn;
    return ss.str();
}

} // namespace dfx
