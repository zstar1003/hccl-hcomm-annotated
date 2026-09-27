/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_kfc_deprecated_process.h"

#include "dfx/dfx_extend_info.h"
#include "aicpu_kfc_process.h"
#include "algorithm/task_orchestrator.h"
#include "common/aicpu_kfc_utils.h"
#include "utils/aicpu_hdc_utils.h"
#include "framework/aicpu_hccl_process.h"

using namespace hccl;

ANONYMOUS_NAMESPACE_BEGIN
bool HcclOpCheckSupportRetry(HcclCMDType opType)
{
    const std::set<HcclCMDType> HcclSupportRetryOpSet
        = {HcclCMDType::HCCL_CMD_BROADCAST,  HcclCMDType::HCCL_CMD_ALLREDUCE,      HcclCMDType::HCCL_CMD_REDUCE,
           HcclCMDType::HCCL_CMD_ALLGATHER,  HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclCMDType::HCCL_CMD_ALLTOALLV,
           HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclCMDType::HCCL_CMD_ALLTOALL,       HcclCMDType::HCCL_CMD_GATHER,
           HcclCMDType::HCCL_CMD_SCATTER};
    return (HcclSupportRetryOpSet.find(opType) != HcclSupportRetryOpSet.end());
}

void HcclUpdateOpIndex(HcclCMDType opType, AicpuComContext* ctx)
{
    if (HcclOpCheckSupportRetry(opType)) {
        auto opIndex = ctx->opIndex + 1;
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, opIndex), opIndex);
    } else {
        // NOTE: send / recv / batchsendrecv 算子不是通信域内所有卡都参与，opIndex需要另行处理；重执行暂不支持该类算子
    }
    return;
}

HcclResult UpdateOpExecStatus(
    AicpuComContext* ctx, HcclOpExecFSM& fsmState, KfcStatus state, KfcError& errorCode, uint32_t retryCnt)
{
    auto ret = AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, state, errorCode, retryCnt);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("SetOpExecStatus failed, ret:%u", ret);
        errorCode = KfcError::kExec;
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    }
    return ret;
}

bool HcclOpCheckInplace(const AivAicpuOpParam& opParams)
{
    if (opParams.sendBuffer != opParams.recvBuffer) {
        return false;
    }

    const std::set<HcclCMDType> HcclInplaceOpSet
        = {HcclCMDType::HCCL_CMD_ALLREDUCE,      HcclCMDType::HCCL_CMD_REDUCE,    HcclCMDType::HCCL_CMD_ALLGATHER,
           HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclCMDType::HCCL_CMD_ALLTOALLV, HcclCMDType::HCCL_CMD_ALLTOALLVC,
           HcclCMDType::HCCL_CMD_ALLTOALL,       HcclCMDType::HCCL_CMD_GATHER,    HcclCMDType::HCCL_CMD_SCATTER};
    if (HcclInplaceOpSet.find(opParams.commType) != HcclInplaceOpSet.end()) {
        return true;
    }
    return false;
}

bool HcclOpSupportRetry(AicpuComContext* ctx, AivAicpuOpParam& opParams)
{
    if (!ctx->retryEnable) {
        HCCL_INFO("hccl aicpu can not retry, enable[%u].", ctx->retryEnable);
        return false;
    }

    // 不支持inplace的通信算子重执行
    if (HcclOpCheckInplace(opParams)) {
        HCCL_INFO(
            "hccl aicpu can not retry, opType[%u], sendBuffer[0x%016lx], recvBuffer[0x%016lx].", opParams.commType,
            opParams.sendBuffer, opParams.recvBuffer);
        return false;
    }

    if (HcclOpCheckSupportRetry(opParams.commType)) {
        return true;
    }
    return false;
}

#ifdef CCL_LLT
static constexpr u32 HCCL_AICPU_WAIT_HOST_BASE_TIME_MS = 200U;
#else
static constexpr u32 HCCL_AICPU_WAIT_HOST_BASE_TIME_MS = 200000U;
#endif
u32 HcclGetWaitRetryCmdTimeout(AicpuComContext* ctx, uint32_t retryCnt)
{
    if (retryCnt == 0) {
        return HCCL_AICPU_WAIT_HOST_BASE_TIME_MS + ctx->retryHoldTime;
    } else {
        return HCCL_AICPU_WAIT_HOST_BASE_TIME_MS + ctx->retryIntervalTime;
    }
}

HcclResult HcclOpExecFsmInitProcess(
    AicpuComContext* ctx, HcclOpExecFSM& state, KfcError& errorCode, AicpuKfcRpcServer& rpc, AivAicpuOpParam& opParams)
{
    rpc.CheckRcvAddrMsg(&opParams, 0);
    ctx->directlySendMainSteramSqe = true;

    HcclUpdateOpIndex(opParams.commType, ctx);
    opParams.opId.index = ctx->opIndex;
    if (ctx->endStopLaunch) {
        HCCL_WARNING("[NsRecovery] Suspending status should not launch task");
        state = HcclOpExecFSM::HCCL_OP_EXEC_STOP_LAUNCH;
        return HCCL_SUCCESS;
    }
    auto ret = AicpuHdcUtils::InitOpExecStatus(ctx->kfcStatusTransferD2H, opParams.opId);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, isOpLaunch), true);
    if (ret == HCCL_SUCCESS) {
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_LAUNCH;
    } else {
        HCCL_ERROR("InitOpExecStatus failed, ret:%u", ret);
        errorCode = KfcError::kInner;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    }
    return ret;
}

HcclResult HcclOpExecFsmStoppingProcess(AicpuComContext* ctx, HcclOpExecFSM& state, KfcError& errorCode)
{
    HCCL_DEBUG("hccl aicpu stopping.");
    if (TaskOrchestrator::IsTaskExceptionForHccs(ctx)) {
        HCCL_INFO("hccl aicpu recoverable task exception occurs.");
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPED;
        return HCCL_SUCCESS;
    }

    KfcCommand cmd = KfcCommand::kNone;
    auto ret = AicpuHdcUtils::GetOpExecCtrlCmd(ctx->kfcControlTransferH2D, cmd);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("GetOpExecCtrlCmd failed, ret:%u", ret);
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return ret;
    }
    if (cmd == KfcCommand::kExit) {
        HCCL_WARNING("hccl aicpu exec fsm stop by exit cmd.");
        errorCode = KfcError::kExit;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    } else if ((cmd == KfcCommand::kStopExec)) {
        HCCL_INFO("hccl aicpu get stop exec cmd.");
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPED;
    } else if ((cmd == KfcCommand::kNone) || (cmd == KfcCommand::kStopLaunch)) {
        HCCL_DEBUG("hccl aicpu wait for stop exec cmd.");
        // do nothing
    } else {
        HCCL_ERROR("GetOpExecCtrlCmd failed, invalid cmd[%u]", cmd);
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOpExecFsmStoppedProcess(
    AicpuComContext* ctx, HcclOpExecFSM& state, KfcError& errorCode, u32 retryCnt, AivAicpuOpParam& opParams,
    u32 beginSqePos, u32 endSqePos)
{
    HCCL_DEBUG("hccl aicpu stop exec.");
    KfcCommand cmd = KfcCommand::kNone;
    auto ret = AicpuHdcUtils::GetOpExecCtrlCmd(ctx->kfcControlTransferH2D, cmd);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("GetOpExecCtrlCmd failed, ret:%u", ret);
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return ret;
    }

    if (cmd == KfcCommand::kExit) {
        HCCL_ERROR("hccl aicpu exec fsm stop by exit cmd.");
        errorCode = KfcError::kExit;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return HCCL_SUCCESS;
    }

    if (!HcclOpSupportRetry(ctx, opParams)) {
        HCCL_ERROR("hccl aicpu not support retry, enable[%u], commType[%u].", ctx->retryEnable, opParams.commType);
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return HCCL_SUCCESS;
    }

    uint32_t sqHead = 0xFFFFFFFF;
    CHK_RET(QuerySqStatusByType(ctx->devId, ctx->streamInfo[ctx->rankId].sqId, DRV_SQCQ_PROP_SQ_HEAD, sqHead));
    if (sqHead == endSqePos) {
        HCCL_INFO(
            "hccl aicpu record complete task is complete, can not retry. params: sqHead %u, beginSqePos %u "
            "endSqePos %u",
            sqHead, beginSqePos, endSqePos);
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_END;
    } else if (sqHead == beginSqePos) {
        HCCL_ERROR(
            "hccl aicpu wait start task is not complete, can not retry. params: sqHead %u, beginSqePos %u "
            "endSqePos %u",
            sqHead, beginSqePos, endSqePos);
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    } else {
        HCCL_INFO(
            "hccl aicpu op is running, can retry. params: sqHead %u, beginSqePos %u endSqePos %u", sqHead, beginSqePos,
            endSqePos);
        if (TaskOrchestrator::IsTaskExceptionForHccs(ctx)) {
            HCCL_INFO("hccl aicpu stop by sdma/write task exception, can retry.");
            errorCode = KfcError::kSdma;
        }
        CHK_RET(UpdateOpExecStatus(ctx, state, KfcStatus::kStopExec, errorCode, retryCnt));
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOpExecFsmEndProcess(AicpuComContext* ctx, uint32_t retryCnt, AivAicpuOpParam& opParams)
{
    auto ret = AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kEnd, KfcError::kNone, retryCnt);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, isOpLaunch), false);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.kfcStatus), DfxKfcStatus::kOneFinished);
    AicpuKfcUtils::PrintBuffer(ctx, opParams);
    ctx->directlySendMainSteramSqe = false;
    return ret;
}
ANONYMOUS_NAMESPACE_END

HcclResult AicpuKfcDeprecatedProcess::LaunchHcclOp(
    AicpuComContext* ctx, AivAicpuOpParam* commParam, uint32_t& beginSqePos, uint32_t& endSqePos)
{
    // 获取通信stream上首次下发的notify wait
    // task的尾指针，已便重执行stop时判断是否已执行该task，如果该task已执行完成则可支持通信重执行
    CHK_RET(QuerySqStatusByType(ctx->devId, ctx->streamInfo[ctx->rankId].sqId, DRV_SQCQ_PROP_SQ_TAIL, beginSqePos));

    // STARS调度执行到该通信算子时，会触发一次本地notify record触发通信算子在AICPU上展开、执行
    CHK_RET(AicpuDispatcher::AicpuUnfoldSignalWait(ctx->rankId, 0, AicpuDispatcher::IPC));
    CHK_RET(AicpuKfcProcess::AicpuCcOpExe(commParam, nullptr, ctx));

    // AICPU上通信task下发完成后，在通信stream上紧跟着下发一个notify record，以通知通信主stream通信算子执行完成
    CHK_RET(AicpuDispatcher::AicpuUnfoldSignalRecord(ctx->rankId, 1, AicpuDispatcher::IPC));

    KfcCommand cmd = KfcCommand::kNone;
    if ((ctx->endStopLaunch == false) && (ctx->commOpenStatus == true)) {
        CHK_RET(AicpuHdcUtils::GetOpExecCtrlCmd(ctx->kfcControlTransferH2D, cmd));
        if (cmd == KfcCommand::NsStopLaunch) {
            AicpuUpdatComContextMumber(offsetof(AicpuComContext, endStopLaunch), true);
            AicpuUpdatComContextMumber(offsetof(AicpuComContext, isStopLaunch), true);
            return HCCL_E_SUSPENDING;
        }
    }
    // 启动通信task执行
    CHK_RET(TaskOrchestrator::ActiveRecordMain(AicpuKfcProcess::GetActiveSqId(ctx)));
    CHK_RET(QuerySqStatusByType(ctx->devId, ctx->streamInfo[ctx->rankId].sqId, DRV_SQCQ_PROP_SQ_TAIL, endSqePos));
    HCCL_INFO(
        "hccl aicpu launch hccl op task success. stream sqid:%d begin:%u end:%u", ctx->streamInfo[ctx->rankId].sqId,
        beginSqePos, endSqePos);
    return HCCL_SUCCESS;
}

HcclResult
AicpuKfcDeprecatedProcess::RetryLaunchHcclOp(AicpuComContext* ctx, AivAicpuOpParam* commParam, uint32_t& endSqePos)
{
    CHK_RET(AicpuKfcProcess::AicpuCcOpExe(commParam, nullptr, ctx));

    // AICPU上通信task下发完成后，在通信stream上紧跟着下发一个notify record，以通知通信主stream通信算子执行完成
    CHK_RET(AicpuDispatcher::AicpuUnfoldSignalRecord(ctx->rankId, 1, AicpuDispatcher::IPC));

    // 启动通信task执行
    CHK_RET(TaskOrchestrator::ActiveRecordMain(AicpuKfcProcess::GetActiveSqId(ctx)));

    CHK_RET(QuerySqStatusByType(ctx->devId, ctx->streamInfo[ctx->rankId].sqId, DRV_SQCQ_PROP_SQ_TAIL, endSqePos));

    HCCL_INFO(
        "hccl aicpu retry launch hccl op task success. stream sqid:%d end:%u", ctx->streamInfo[ctx->rankId].sqId,
        endSqePos);
    return HCCL_SUCCESS;
}

HcclResult AicpuKfcDeprecatedProcess::RunRpcServerOneStageWait(AicpuComContext* ctx, AicpuKfcRpcServer& rpc)
{
    AivAicpuOpParam g_msg[3];
    AivAicpuOpParam* msg = &g_msg[0];
    AivAicpuOpParam* preMsg = &g_msg[1];
    AivAicpuOpParam* nextMsg = &g_msg[2];
    AivAicpuOpParam* tmpptr = nullptr;
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.kfcStatus), DfxKfcStatus::kOneStart);
    AicpuHcclProcess::CallMC2MaintenanceThread(ctx);
    // 读取首轮任务，并准备，直接先下主流(直接激活)，后续还是先下从流，激活时下主流
    rpc.CheckRcvAddrMsg(msg, 0);
    if (!rpc.CheckAivIsEnd(0)) {
        rpc.ReadAddrMsg(nextMsg, 0);
        tmpptr = nextMsg;
    }
    HcclUpdateOpIndex(msg->commType, ctx);
    msg->opId.index = ctx->opIndex;
    if (ctx->endStopLaunch) {
        HCCL_WARNING("the op should not be launched in suspending status");
        return HCCL_E_SUSPENDING;
    }
    auto ret = AicpuHdcUtils::InitOpExecStatus(ctx->kfcStatusTransferD2H, msg->opId);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, isOpLaunch), true);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("InitOpExecStatus failed, ret:%u", ret);
        return ret;
    }
    ctx->directlySendMainSteramSqe = true;
    CHK_RET(AicpuKfcProcess::AicpuCcOpExe(msg, tmpptr, ctx));

    while (!rpc.CheckAivIsEnd(0)) {
        tmpptr = msg;
        msg = preMsg;
        preMsg = tmpptr; // msg <-> preMsg
        tmpptr = nullptr;

        // 读取下一次任务，并编排
        rpc.CheckRcvAddrMsg(msg, 0);
        if (!rpc.CheckAivIsEnd(0)) {
            rpc.ReadAddrMsg(nextMsg, 0);
            tmpptr = nextMsg;
        }
        CHK_RET(AicpuKfcProcess::AicpuCcOpExe(msg, tmpptr, ctx));

        // 激活下一次任务执行
        CHK_RET(TaskOrchestrator::ActiveRecordMain(AicpuKfcProcess::GetActiveSqId(ctx)));
    }

    // 激活下一次任务执行
    CHK_RET(TaskOrchestrator::ActiveRecordMain(AicpuKfcProcess::GetActiveSqId(ctx)));
    ctx->directlySendMainSteramSqe = false;
    CHK_RET(AicpuKfcProcess::WaitTaskFinish(ctx));
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.kfcStatus), DfxKfcStatus::kOneFinished);
    AicpuKfcUtils::PrintBuffer(ctx, *msg);
    return HCCL_SUCCESS;
}

HcclResult AicpuKfcDeprecatedProcess::HcclOpExecFsmWaitEndProcess(
    AicpuComContext* ctx, HcclOpExecFSM& state, KfcError& errorCode, u32 retryCnt)
{
    bool isWaitTask = (ctx->debugMode == MC2_DEBUG_WAIT_COMM);
    auto ret = AicpuKfcProcess::WaitTaskFinish(ctx, isWaitTask);
    if (ret == HCCL_SUCCESS) {
        HCCL_DEBUG("hccl aicpu exec complete.");
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_END;
    } else if (ret == HCCL_E_SUSPENDING) {
        HCCL_RUN_INFO("[NsRecovery][AICPU]hccl aicpu force stop in launch loop");
        if (ctx->isStopLaunch == true) {
            state = HcclOpExecFSM::HCCL_OP_EXEC_STOP_LAUNCH;
        } else {
            CHK_RET(UpdateOpExecStatus(ctx, state, KfcStatus::kStoplaunch, errorCode, retryCnt));
            state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING;
        }
    } else {
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    }
    return ret;
}

HcclResult AicpuKfcDeprecatedProcess::HcclOpExecFsmWaitRetryProcess(
    AicpuComContext* ctx, HcclOpExecFSM& state, KfcError& errorCode)
{
    KfcCommand cmd = KfcCommand::kNone;
    auto ret = AicpuHdcUtils::GetOpExecCtrlCmd(ctx->kfcControlTransferH2D, cmd);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("GetOpExecCtrlCmd failed, ret:%u", ret);
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return ret;
    }
    if (cmd == KfcCommand::kRetry) {
        HCCL_INFO("hccl aicpu recv retry cmd from host.");
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.pollStatus), PollStatus::kDefault);
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.cqeStatus), dfx::CqeStatus::kDefault);
        ret = AicpuKfcProcess::ResetSqBuff(ctx);
        if (ret != HCCL_SUCCESS) {
            errorCode = KfcError::kInner;
            state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
            return ret;
        }
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_RETRY;
    } else if (cmd == KfcCommand::kExit) {
        errorCode = KfcError::kExit;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    } else {
        // do nothing
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuKfcDeprecatedProcess::AICPU_RpcServerUnfoldStageWait(AicpuComContext* ctx, AicpuKfcRpcServer& rpc)
{
    AivAicpuOpParam opParams;
    auto waitStopExecCmdTimeout = std::chrono::milliseconds(HCCL_AICPU_WAIT_HOST_BASE_TIME_MS);
    auto startTime = std::chrono::steady_clock::now();

    KfcError errorCode = KfcError::kNone;
    uint32_t retryCnt = 0;
    uint32_t beginSqePos = INVALID_UINT;
    uint32_t endSqePos = INVALID_UINT;
    HcclOpExecFSM state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_INIT;
    HcclResult ret = HCCL_SUCCESS;
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.kfcStatus), DfxKfcStatus::kOneStart);
    AicpuHcclProcess::CallMC2MaintenanceThread(ctx);
    while (true) {
        switch (state) {
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_INIT:
                ret = HcclOpExecFsmInitProcess(ctx, state, errorCode, rpc, opParams);
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_LAUNCH:
                ret = HcclOpExecFsmLaunchProcess(ctx, state, errorCode, opParams, beginSqePos, endSqePos);
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END:
                ret = HcclOpExecFsmWaitEndProcess(ctx, state, errorCode, retryCnt);
                if (state == HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING) {
                    startTime = std::chrono::steady_clock::now();
                }
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING:
                if ((std::chrono::steady_clock::now() - startTime) >= waitStopExecCmdTimeout) {
                    HCCL_ERROR("hccl aicpu wait stop exec timeout[%u ms].", HCCL_AICPU_WAIT_HOST_BASE_TIME_MS);
                    errorCode = KfcError::kTimeout;
                    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
                } else {
                    ret = HcclOpExecFsmStoppingProcess(ctx, state, errorCode);
                }
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPED:
                ret = HcclOpExecFsmStoppedProcess(ctx, state, errorCode, retryCnt, opParams, beginSqePos, endSqePos);
                if (state == HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY) {
                    startTime = std::chrono::steady_clock::now();
                }
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY:
                if ((std::chrono::steady_clock::now() - startTime)
                    >= std::chrono::milliseconds(HcclGetWaitRetryCmdTimeout(ctx, retryCnt))) {
                    HCCL_ERROR("hccl aicpu wait retry timeout[%u ms].", HcclGetWaitRetryCmdTimeout(ctx, retryCnt));
                    errorCode = KfcError::kTimeout;
                    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
                } else {
                    ret = HcclOpExecFsmWaitRetryProcess(ctx, state, errorCode);
                }
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_RETRY:
                ret = HcclOpExecFsmRetryProcess(ctx, state, errorCode, retryCnt, opParams, endSqePos);
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_END:
                return HcclOpExecFsmEndProcess(ctx, retryCnt, opParams);
            case HcclOpExecFSM::HCCL_OP_EXEC_STOP_LAUNCH:
                HCCL_DEBUG("[NsTest][AICPU] stop the kernel");
                if (!ctx->isStopLaunch) {
                    return HCCL_E_SUSPENDING;
                } else {
                    HCCL_RUN_INFO("[NsTest][AICPU] stop the kernel for stop command");
                    AicpuHcclProcess::CopyCtxForBackGroundDfx(ctx);
                    if (UpdateOpExecStatus(ctx, state, KfcStatus::kStoplaunch, errorCode, 0) == HCCL_SUCCESS) {
                        return HCCL_E_SUSPENDING;
                    } else {
                        break;
                    }
                }
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR:
            default:
                UpdateOpExecStatus(ctx, state, KfcStatus::kError, errorCode, retryCnt);
                return (ret == HCCL_SUCCESS) ? HCCL_E_INTERNAL : ret;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuKfcDeprecatedProcess::RunRpcServerTwoStageWait(AicpuComContext* ctx, AicpuKfcRpcServer& rpc)
{
    AivAicpuOpParam gMsg[3];
    AivAicpuOpParam* msg = &gMsg[0];
    AivAicpuOpParam* msgWork = &gMsg[1];
    AivAicpuOpParam* nextMsg = &gMsg[2];
    AivAicpuOpParam* tmpptr = nullptr;

    // 读取首轮任务，并准备，直接先下主流(直接激活)，后续还是先下从流，激活时下主流
    // 1.1、首轮读地址（需要自动产生）
    rpc.CheckRcvAddrMsg(msg, 0);
    if (!rpc.CheckAivIsEnd(0)) {
        // 读取下一轮地址
        rpc.ReadAddrMsg(nextMsg, 0);
        tmpptr = nextMsg;
    }

    // 1.2 首轮提前读看是否需要提前下主流，即判断sendcnt是否大于等于当前轮次
    if (rpc.ReadWorkMsg(msgWork, 0, (ctx->curTurnCnt + 1)) && rpc.GetWaitPolicy() != 0) {
        ctx->directlySendMainSteramSqe = true;
    }

    // 1.3 首轮编排开始
    CHK_RET(AicpuKfcProcess::AicpuCcOpExe(msg, tmpptr, ctx));
    ctx->directlySendMainSteramSqe = false;
    // 2、等待激活任务执行，如果前面已经激活，则ActiveRecordMain会空转一圈
    if (rpc.GetWaitPolicy() != 0) {
        rpc.CheckRcvWorkMsg(msgWork, 0, ctx->curTurnCnt);
    }

    AicpuKfcUtils::PrintBuffer(ctx, *msg);
    CHK_RET(TaskOrchestrator::ActiveRecordMain(AicpuKfcProcess::GetActiveSqId(ctx)));
    while (!rpc.CheckAivIsEnd(0)) {
        tmpptr = nullptr;
        // 3.1、读取下一轮任务
        rpc.CheckRcvAddrMsg(msg, 0);
        if (!rpc.CheckAivIsEnd(0)) {
            rpc.ReadAddrMsg(nextMsg, 0);
            tmpptr = nextMsg;
        }

        // 3.2 开始编排下一轮
        CHK_RET(AicpuKfcProcess::AicpuCcOpExe(msg, tmpptr, ctx));

        // 5.1 等待上一轮执行结束
        TaskOrchestrator::WaitMainStreamFinish(ctx);

        // 6.1 激活下一轮
        rpc.CheckRcvWorkMsg(msgWork, 0, ctx->curTurnCnt);
        CHK_RET(TaskOrchestrator::ActiveRecordMain(AicpuKfcProcess::GetActiveSqId(ctx)));

        // 7.1 发送上一轮消息
        rpc.PostMsg(ctx->curTurnCnt - 1);
    }

    if (rpc.GetRspPolicy() != 0) {
        // 8.1 等待执行结束
        TaskOrchestrator::WaitMainStreamFinish(ctx);
        rpc.ClearWorkMsg();
        HCCL_INFO(
            "[commType:%d, opType:%s, sendBuffer:%p, recvBuffer:%p, count:%d, data_type:%s, "
            "sendCnt:%d, rcvCnt:%d, funID:%d, valid:%d, everyTurnRsp:%d, strideLen:%d, isLast:%d",
            msg->commType, GetReduceOpEnumStr(msg->opType).c_str(), msg->sendBuffer, msg->recvBuffer, msg->count,
            GetDataTypeEnumStr(msg->hcclDataType).c_str(), msg->sendCnt, msg->rcvCnt, msg->funID, msg->valid,
            msg->everyTurnRsp, msg->strideLen, msg->isLast);
        // 9.1 发送最后一轮消息
        rpc.PostMsg(ctx->curTurnCnt);
    }
    AicpuKfcUtils::PrintBuffer(ctx, *msg);
    return HCCL_SUCCESS;
}

HcclResult AicpuKfcDeprecatedProcess::TryRunRpcServerOneStageWait(AicpuComContext* ctx, AicpuKfcRpcServer& rpc)
{
    HCCL_INFO("Start to run, round %u", ctx->dfxExtendInfo.kfcRestartConfig.tryRestartTimes);
    if (dfx::DfxExtendInfoHelper::TryRestartTooManyTimes(ctx->dfxExtendInfo)) {
        HCCL_ERROR("Restart too many times, max try count is %u", ctx->dfxExtendInfo.kfcRestartConfig.maxRestartTimes);
        return HCCL_E_INTERNAL;
    }
    const auto ret = RunRpcServerOneStageWait(ctx, rpc);
    if (ret == HCCL_SUCCESS) {
        dfx::DfxExtendInfoHelper::ResetTryRestartTimes(ctx->dfxExtendInfo);
        CHK_RET(AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kEnd, KfcError::kNone, 0));
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, isOpLaunch), false);
        return HCCL_SUCCESS;
    }
    if (ctx->dfxExtendInfo.commandToKfc == CommandToKfc::kRestart) {
        dfx::DfxExtendInfoHelper::TryRestartOnceMore(ctx->dfxExtendInfo);
        return TryRunRpcServerOneStageWait(ctx, rpc);
    }
    dfx::DfxExtendInfoHelper::ResetTryRestartTimes(ctx->dfxExtendInfo);
    if (ctx->isStopLaunch) {
        AicpuHcclProcess::CopyCtxForBackGroundDfx(ctx);
        CHK_RET(AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kStoplaunch, KfcError::kNone, 0));
    } else {
        CHK_RET(AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kError, KfcError::kInner, 0));
    }
    return ret;
}

HcclResult AicpuKfcDeprecatedProcess::HcclOpExecFsmLaunchProcess(
    AicpuComContext* ctx, HcclOpExecFSM& state, KfcError& errorCode, AivAicpuOpParam& opParams, uint32_t& beginSqePos,
    uint32_t& endSqePos)
{
    HCCL_DEBUG("hccl aicpu start launch task");
    auto ret = AicpuKfcDeprecatedProcess::LaunchHcclOp(ctx, &opParams, beginSqePos, endSqePos);
    if (ret == HCCL_SUCCESS) {
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END;
    } else if (ret == HCCL_E_SUSPENDING) {
        HCCL_RUN_INFO("[NsRecovery][AICPU]hccl aicpu force stop in launch process");
        state = HcclOpExecFSM::HCCL_OP_EXEC_STOP_LAUNCH;
    } else {
        HCCL_ERROR("Failed to launch hccl op, ret:%u", ret);
        errorCode = KfcError::kInner;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    }
    return ret;
}

HcclResult AicpuKfcDeprecatedProcess::HcclOpExecFsmRetryProcess(
    AicpuComContext* ctx, HcclOpExecFSM& state, KfcError& errorCode, uint32_t& retryCnt, AivAicpuOpParam& opParams,
    uint32_t& endSqePos)
{
    HCCL_DEBUG("hccl retry launch task");
    retryCnt++;
    auto ret = AicpuKfcDeprecatedProcess::RetryLaunchHcclOp(ctx, &opParams, endSqePos);
    if (ret != HCCL_SUCCESS) {
        errorCode = KfcError::kInner;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return ret;
    }
    errorCode = KfcError::kNone;
    CHK_RET(UpdateOpExecStatus(ctx, state, KfcStatus::kRuning, errorCode, retryCnt));
    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END;
    return HCCL_SUCCESS;
}
