/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_kfc_retry_process.h"

#include "aicpu_hccl_def.h"
#include "framework/aicpu_communicator.h"

using namespace hccl;

ANONYMOUS_NAMESPACE_BEGIN
HcclResult MC2OpExecFsmStoppingProcess(HcclCommAicpu& comm, HcclOpExecFSM& state, KfcError& errorCode)
{
    KfcCommand cmd = KfcCommand::kNone;
    auto ret = comm.BackGroundGetCmd(cmd);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("MC2 restart GetOpExecCtrlCmd failed, ret:%u", ret);
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return ret;
    }

    if (cmd == KfcCommand::kExit) {
        HCCL_ERROR("MC2 restart aicpu exec fsm stop by exit cmd.");
        errorCode = KfcError::kExit;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    } else if (cmd == KfcCommand::kStopExec) {
        HCCL_DEBUG("MC2 restart MC2 aicpu get stop exec cmd.");
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPED;
    } else if (cmd == KfcCommand::kStopLaunch) {
        // do nothing
    } else if ((cmd == KfcCommand::kNone) || (cmd == KfcCommand::kRetry)) {
        // do nothing
    } else {
        HCCL_ERROR("MC2 restart GetOpExecCtrlCmd failed, invalid cmd[%u]", cmd);
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2OpExecFsmStoppedProcess(HcclCommAicpu& comm, HcclOpExecFSM& state, KfcError& errorCode)
{
    KfcCommand cmd = KfcCommand::kNone;
    auto ret = comm.BackGroundGetCmd(cmd);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("MC2 restart GetOpExecCtrlCmd failed, ret:%u", ret);
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return ret;
    }

    if (cmd == KfcCommand::kExit) {
        HCCL_ERROR("MC2 restart hccl aicpu exec fsm stop by exit cmd.");
        errorCode = KfcError::kExit;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    } else {
        errorCode = KfcError::kNone;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY;
    }
    return HCCL_SUCCESS;
}

HcclResult
MC2OpExecFsmWaitRetryProcess(HcclCommAicpu& comm, HcclOpExecFSM& state, KfcError& errorCode, bool linkChanged)
{
    KfcCommand cmd = KfcCommand::kNone;
    auto ret = comm.BackGroundGetCmd(cmd);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("MC2 restart GetOpExecCtrlCmd failed, ret:%u", ret);
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return ret;
    }

    if (cmd == KfcCommand::kRetry) {
        HCCL_DEBUG("MC2 restart aicpu recv retry cmd from host.");
        comm.GetDfxExtendInfo()->pollStatus = PollStatus::kDefault;
        comm.GetDfxExtendInfo()->cqeStatus = dfx::CqeStatus::kDefault;
        comm.ResetOpRetryException(HcclCMDType::HCCL_CMD_INVALID);
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_RETRY;
    } else if (cmd == KfcCommand::kChangeLink && !linkChanged) {
        HCCL_DEBUG("MC2 restart aicpu recv change link cmd");
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_CHANGE_LINK;
    } else if (cmd == KfcCommand::kExit) {
        HCCL_ERROR("MC2 restart aicpu recv exit cmd from host.");
        errorCode = KfcError::kExit;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    }
    return HCCL_SUCCESS;
}
ANONYMOUS_NAMESPACE_END

HcclResult AicpuKfcRetryProcess::RetryProcess(HcclCommAicpu& comm, RestartParam& restartParam, uint32_t idx)
{
    HcclResult ret = HCCL_SUCCESS;
    auto waitStopExecCmdTimeoutMs = comm.HcclGetWaitStopExecCmdTimeout();
    auto waitStopExecCmdTimeout = std::chrono::milliseconds(waitStopExecCmdTimeoutMs);
    auto waitRetryCmdTimeoutMs = comm.HcclGetWaitRetryCmdTimeout(restartParam.restartCnt);
    auto waitRetryCmdTimeout = std::chrono::milliseconds(waitRetryCmdTimeoutMs);

    switch (restartParam.fsmState[idx]) {
        case HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END:
            HCCL_INFO("MC2 restart state HCCL_OP_EXEC_FSM_WAIT_END");
            restartParam.errorCode[idx] = KfcError::kSdma;
            restartParam.fsmState[idx] = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING;
            ret = comm.UpdateOpExecStatus(
                restartParam.fsmState[idx], KfcStatus::kStoplaunch, restartParam.errorCode[idx],
                restartParam.restartCnt); // 上报sdma异常
            if (restartParam.fsmState[idx] == HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING) {
                restartParam.startTime[idx] = std::chrono::steady_clock::now();
            }
            break;
        case HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING:
            if ((std::chrono::steady_clock::now() - restartParam.startTime[idx]) >= waitStopExecCmdTimeout) {
                HCCL_ERROR("MC2 restart aicpu wait stop exec timeout[%u ms].", comm.HcclGetWaitStopExecCmdTimeout());
                restartParam.errorCode[idx] = KfcError::kTimeout;
                restartParam.fsmState[idx] = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
            } else {
                ret = MC2OpExecFsmStoppingProcess(comm, restartParam.fsmState[idx], restartParam.errorCode[idx]);
            }
            break;
        case HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPED:
            HCCL_INFO("MC2 restart state HCCL_OP_EXEC_FSM_STOPPED");
            ret = MC2OpExecFsmStoppedProcess(comm, restartParam.fsmState[idx], restartParam.errorCode[idx]);
            if (restartParam.fsmState[idx] == HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY) {
                CHK_RET(comm.UpdateOpExecStatus(
                    restartParam.fsmState[idx], KfcStatus::kStopExec, restartParam.errorCode[idx],
                    restartParam.restartCnt));
                restartParam.startTime[idx] = std::chrono::steady_clock::now();
            }
            break;
        case HcclOpExecFSM::HCCL_OP_EXEC_FSM_CHANGE_LINK:
            HCCL_INFO("MC2 restart state HCCL_OP_EXEC_FSM_CHANGE_LINK");
            // MC2重执行，清空所有rdma链接
            comm.CleanAllRoceResource();
            restartParam.errorCode[idx] = KfcError::kNone;
            ret = comm.UpdateOpExecStatus(
                restartParam.fsmState[idx], KfcStatus::kChanged, restartParam.errorCode[idx], restartParam.restartCnt);
            restartParam.linkChanged[idx] = true;
            restartParam.fsmState[idx] = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY;
            break;
        case HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY:
            if ((std::chrono::steady_clock::now() - restartParam.startTime[idx]) >= waitRetryCmdTimeout) {
                HCCL_ERROR("MC2 restart aicpu wait retry timeout[%u ms].", waitRetryCmdTimeoutMs);
                restartParam.errorCode[idx] = KfcError::kTimeout;
                restartParam.fsmState[idx] = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
            } else {
                ret = MC2OpExecFsmWaitRetryProcess(
                    comm, restartParam.fsmState[idx], restartParam.errorCode[idx], restartParam.linkChanged[idx]);
            }
            break;
        case HcclOpExecFSM::HCCL_OP_EXEC_FSM_RETRY:
            HCCL_INFO("MC2 restart state HCCL_OP_EXEC_FSM_RETRY");
            restartParam.consultationResult[idx] = true;
            restartParam.errorCode[idx] = KfcError::kNone;
            comm.UpdateOpExecStatus(
                restartParam.fsmState[idx], KfcStatus::kEnd, restartParam.errorCode[idx], restartParam.restartCnt);
            return HCCL_SUCCESS;
        case HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR:
        default: {
            HCCL_ERROR("MC2 restart aicpu restart process error.");
            comm.UpdateOpExecStatus(
                restartParam.fsmState[idx], KfcStatus::kError, restartParam.errorCode[idx], restartParam.restartCnt);
            comm.GetDfxExtendInfo()->kfcStatus = DfxKfcStatus::kOneFinished;
            return (ret == HCCL_SUCCESS) ? HCCL_E_INTERNAL : ret;
        }
    }
    return ret;
}
