/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_kfc_prof.h"

#include "common/aicpu_hccl_common.h"
#include "common/aicpu_kfc_def.h"
#include "profiling_manager_device.h"

thread_local static uint32_t g_curLoopCnt = 0;
thread_local static AicpuComProf g_acprof[AC_MAX_PROF_LOOP];
uint8_t AicpuKfcProf::debugMode_ = 0;

AicpuComProf* AicpuKfcProf::GetCurrentAicpuProf() { return &g_acprof[g_curLoopCnt]; }

void AicpuKfcProf::SetCurrentProf(uint64_t launchTime)
{
    g_acprof[g_curLoopCnt].tid = syscall(__NR_gettid);
    g_acprof[g_curLoopCnt].launchEntryTime = launchTime;
}

void AicpuKfcProf::SetKfcTimeLine(KfcTimeLine kfcTimeLine)
{
    if (!AicpuKfcProf::NeedRecordTimeTaken()) {
        return;
    }
    AicpuComProf* acprof = AicpuKfcProf::GetCurrentAicpuProf();
    uint32_t recordIndex = acprof->workCnt;
    recordIndex = (recordIndex >= AC_MAX_PROF_COMM_CNT) ? (AC_MAX_PROF_COMM_CNT - 1) : recordIndex;
    switch (kfcTimeLine) {
        case KfcTimeLine::HCC_EXEC_START_TIME:
            acprof->commLoop[recordIndex].hccExecStartTime = GetCurCpuTimestamp(true);
            break;
        case KfcTimeLine::SEND_TASK_START_TIME:
            acprof->commLoop[recordIndex].sendTaskStartTime = GetCurCpuTimestamp(true);
            break;
        case KfcTimeLine::SEND_SQE_FINISH_TIME:
            acprof->commLoop[recordIndex].sendSqeFinishTime = GetCurCpuTimestamp(true);
            break;
        default:
            break;
    }
    return;
}

void AicpuKfcProf::SetDebugMode(uint8_t debugMode) { debugMode_ = debugMode; }

bool AicpuKfcProf::IsDebugModeEquals(const uint8_t mode) { return debugMode_ == mode; }

bool AicpuKfcProf::NeedRecordTimeTaken()
{
    return IsDebugModeEquals(MC2_DEBUG_TIME_TAKEN) || dfx::ProfilingManager::GetProfL1State();
}

AicpuComProf* AicpuKfcProf::GetaicpuProfInst() { return &g_acprof[0]; }

void AicpuKfcProf::AddProfLoopCnt(u32 addCnt)
{
    g_curLoopCnt = (g_curLoopCnt + addCnt) % AC_MAX_PROF_LOOP;
    HCCL_INFO("g_curLoopCnt set to %u", g_curLoopCnt);
}

void AicpuKfcProf::OutputProfLog(bool debugFlag, AicpuComProf* profInfo, AicpuComProf* backupProfInfo)
{
    if (!debugFlag || g_curLoopCnt < AC_MAX_PROF_LOOP - 1) {
        return;
    }
    thread_local static u32 profIdx = 0U;
    for (u32 i = 0; i < AC_MAX_PROF_LOOP - 1; i++) {
        AicpuComProf* prof = &(profInfo[i]);
        if (prof->workCnt <= 0 && backupProfInfo != nullptr) {
            prof = &(backupProfInfo[i]);
        }
        u32 workRcdCnt = prof->workCnt > AC_MAX_PROF_COMM_CNT ? AC_MAX_PROF_COMM_CNT : prof->workCnt;
        HCCL_RUN_INFO(
            "OP %u: clusterID %u, tid %lu, rankId %u, workCnt %u, serverStartTime %lu, waitMsgStartTime "
            "%lu, rtsqExeEndTime %lu, serverEndTime %lu, StartServer %lu, Finalize %lu, E2E %lu",
            profIdx + i, prof->clusterId, prof->tid, prof->rankId, prof->workCnt, prof->launchEntryTime,
            prof->commInitEndTime, prof->receiveFinalizeTime, prof->endTime,
            prof->commInitEndTime - prof->launchEntryTime, prof->endTime - prof->receiveFinalizeTime,
            prof->endTime - prof->launchEntryTime);

        for (u32 j = 0; j < workRcdCnt; j++) {
            AicpuComProfCommLoop* commRcd = &prof->commLoop[j];
            HCCL_RUN_INFO(
                "Turn %u: kfcAlgExeStartTime %lu, sendTaskStartTime %lu, sendSqeFinishTime %lu, "
                "TaskWaitRequest %lu, TaskOrchestration %lu, TaskLaunch %lu, TaskExecute %lu",
                j + 1, commRcd->hccExecStartTime, commRcd->sendTaskStartTime, commRcd->sendSqeFinishTime,
                commRcd->hccExecStartTime - prof->commInitEndTime,
                commRcd->sendTaskStartTime - commRcd->hccExecStartTime,
                commRcd->sendSqeFinishTime - commRcd->sendTaskStartTime,
                prof->receiveFinalizeTime - commRcd->sendSqeFinishTime);
        }

        prof->fillSqeCnt = 0;
        prof->fillSqeTimes = 0;
        prof->sendSqeBatch = 0;
        prof->sendSqeTimes = 0;
        prof->workCnt = 0;
        prof->traceSubmitTime = 0;
        prof->traceCtxTime = 0;
        prof->traceSqeTime = 0;
    }
    profIdx += AC_MAX_PROF_LOOP;
}

AicpuComProf& AicpuKfcProf::GetProInst(AicpuComContext& ctx) { return ctx.acprof[g_curLoopCnt]; }
