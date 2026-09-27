/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <chrono>
#include "externalinput_pub.h"
#include "sal_pub.h"
#include "heartbeat.h"
#include "comm_configer.h"
#include "opretry_agent.h"

namespace hccl {
HcclResult CreateOpRetryAgentByState(RetryState state, RetryContext* retryCtx)
{
    HCCL_INFO("[OpRetry][Agent]CreateOpRetryAgentByState state[%s]", GetReadableState(state));
    std::shared_ptr<OpRetryBase> retryPtr = nullptr;
    switch (state) {
        case RETRY_STATE_AGENT_RUNNING: {
            EXCEPTION_CATCH(retryPtr = std::make_shared<OpRetryAgentRunning>(), return HCCL_E_PTR);
            break;
        }
        case RETRY_STATE_AGENT_RETRY_FAIL: {
            EXCEPTION_CATCH(retryPtr = std::make_shared<OpRetryAgentRetryFail>(), return HCCL_E_PTR);
            break;
        }
        case RETRY_STATE_RESP_AICPU_ERR:
        case RETRY_STATE_RESP_AICPU_STOPED:
        case RETRY_STATE_RESP_STREAM_STOPED:
        case RETRY_STATE_RESP_STREAM_CLEARED:
        case RETRY_STATE_RESP_LINK_CHANGED:
        case RETRY_STATE_RESP_STOP_TRANSPORT:
        case RETRY_STATE_RESP_NOTIFY_RESETED:
        case RETRY_STATE_RESP_RESUME_TRANSPORT:
        case RETRY_STATE_RESP_CHECK_INFO:
        case RETRY_STATE_RESP_AICPU_RETRYEND:
        case RETRY_STATE_RESP_RUNNING_ERR: {
            EXCEPTION_CATCH(retryPtr = std::make_shared<OpRetryAgentResponse>(), return HCCL_E_PTR);
            break;
        }
        case RETRY_STATE_RESP_LINK_CHECKED: {
            EXCEPTION_CATCH(retryPtr = std::make_shared<OpRetryAgentResponseLinkInfo>(), return HCCL_E_PTR);
            break;
        }
        case RETRY_STATE_WAIT_CHANGE_LINK_INFO: {
            EXCEPTION_CATCH(retryPtr = std::make_shared<OpRetryAgentWaitChangeLinkInfo>(), return HCCL_E_PTR);
            break;
        }
        case RETRY_STATE_WAIT_CMD_STOP_AICPU:
        case RETRY_STATE_WAIT_CMD_STOP_STREAM:
        case RETRY_STATE_WAIT_CMD_CLEAR_STREAM:
        case RETRY_STATE_WAIT_CMD_STOP_TRANSPORT:
        case RETRY_STATE_WAIT_CMD_RESET_NOTIFY:
        case RETRY_STATE_WAIT_CMD_CHECK_LINK:
        case RETRY_STATE_WAIT_CMD_RESUME_TRANSPORT:
        case RETRY_STATE_WAIT_CMD_CHECK:
        case RETRY_STATE_WAIT_CMD_CAN_RETRY:
        case RETRY_STATE_WAIT_CMD_RETRY_FAIL: {
            EXCEPTION_CATCH(retryPtr = std::make_shared<OpRetryAgentWaitCmd>(), return HCCL_E_PTR);
            break;
        }
        case RETRY_STATE_POLL_AICPU_STOPED:
        case RETRY_STATE_POLL_AICPU_CHANGED:
        case RETRY_STATE_POLL_AICPU_RETRYEND:
        case RETRY_STATE_POLL_STREAM_STOPED: {
            EXCEPTION_CATCH(retryPtr = std::make_shared<OpRetryAgentPollAicpuStop>(), return HCCL_E_PTR);
            break;
        }
        // 发送主动借轨信息
        case RETRY_STATE_SEND_SWITCH_INFO:
            EXCEPTION_CATCH(retryPtr = std::make_shared<SwitchNicAgentSendSwitchInfo>(), return HCCL_E_PTR);
            break;
        // 等待server RetryCommand命令
        case RETRY_STATE_WAIT_CMD_SEND_AICPU:
            EXCEPTION_CATCH(retryPtr = std::make_shared<SwitchNicAgentWaitCmd>(), return HCCL_E_PTR);
            break;
        // Resume 过程中检查网口和链路状态(接收来自Server的命令，检查网口和链路状态，将信息回复发送给Server)
        case RETRY_RESUME_STATE_AGENT_CHECK_LINK:
            EXCEPTION_CATCH(retryPtr = std::make_shared<ResumeAgentCheckLink>(), return HCCL_E_PTR);
            break;
        // Resume 过程中接收Server借轨命令，下发给Aicpu背景线程，借轨完成通知Server
        case RETRY_RESUME_STATE_AGENT_CHANGE_LINK:
            EXCEPTION_CATCH(retryPtr = std::make_shared<ResumeAgentChangeLink>(), return HCCL_E_PTR);
            break;
        default: {
            HCCL_ERROR(
                "[OpRetry][Agent]CreateOpRetryAgentByState failed, state[%s] is invalid", GetReadableState(state));
            return HCCL_E_NOT_SUPPORT;
        }
    }
    retryCtx->SetRetryState(state, retryPtr);
    return HCCL_SUCCESS;
}

HcclResult OpRetryAgentBase::ProcessError(RetryContext* retryCtx)
{
    HCCL_ERROR(
        "[%s]OpRetryAgent run fail, rankId[%u], state[%s], IpInfo[%s]", __func__, retryCtx->rankId_,
        retryCtx->GetReadableCtxState(), retryCtx->GetDfxIpInfo());
    // 状态切换至RETRY_STATE_RESP_RUNNING_ERR（上报Server）
    CHK_RET(CreateOpRetryAgentByState(RETRY_STATE_RESP_RUNNING_ERR, retryCtx));
    return HCCL_SUCCESS;
}

OpRetryAgentRunning::OpRetryAgentRunning()
{
    lastRecvCmdTime_ = std::chrono::steady_clock::now();
    lastPollAicpuTime_ = lastRecvCmdTime_;
    pollTimeout_ = std::chrono::seconds(OP_RETRY_POLL_AICPU_ERROR_INTERVAL); // 轮询aicpu间隔
    keepTimeout_ = std::chrono::seconds(OP_RETRY_KEEP_INTERVAL);             // 发送保活数据间隔

    lastPollRcTime_ = lastRecvCmdTime_;
    lastKeepTime_ = lastRecvCmdTime_;
    pollRcTimeout_ = std::chrono::seconds(OP_RETRY_POLL_RDMA_ERROR_INTERVAL); // 轮询rdma cqe间隔
}

// RETRY_STATE_AGENT_RUNNING
HcclResult OpRetryAgentRunning::ProcessEvent(RetryContext* retryCtx)
{
    HcclResult ret = HCCL_SUCCESS;
    std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();

    // 定期轮询aicpu状态
    const auto pollTime = std::chrono::duration_cast<std::chrono::seconds>(curTime - lastPollAicpuTime_);
    if (pollTime > pollTimeout_) {
        if (retryCtx->isEnableSdmaRetry_) {
            RetryState nextState = RETRY_STATE_RESERVED;
            CHK_RET(ParseKfcErr(retryCtx, nextState));
            if (nextState != RETRY_STATE_RESERVED) {
                CHK_RET(CreateOpRetryAgentByState(RETRY_STATE_RESP_AICPU_ERR, retryCtx));
                return HCCL_SUCCESS;
            }
        }
        lastPollAicpuTime_ = curTime;
    }

    // 定期轮询 Rdma Cqe 状态
    const auto pollRcTime = std::chrono::duration_cast<std::chrono::seconds>(curTime - lastPollRcTime_);
    if (pollRcTime > pollRcTimeout_) {
        HCCL_DEBUG("[OpRetry][Agent] OpRetryAgentRunning poll rdma err");
        if (retryCtx->isEnableBackupLink_) {
            RetryState nextState = RETRY_STATE_RESERVED;
            // 遍历 RDMA CQE Error 状态
            CHK_RET(ParseRdmaErr(retryCtx, nextState));
            if (nextState != RETRY_STATE_RESERVED) {
                CHK_RET(CreateOpRetryAgentByState(nextState, retryCtx));
                return HCCL_SUCCESS;
            }
        }
        lastPollRcTime_ = curTime;
    }

    // OpRetryAgent Running状态下,读取到KfcStatus:kPlanSwitch或kSwitchError,切换状态机状态。
    const auto activeTime = std::chrono::duration_cast<std::chrono::seconds>(curTime - lastPollAicpuTime_);
    KfcExecStatus& opInfo = retryCtx->localRetryInfo_.opInfo;
    CHK_RET(GetOpExecInfo(retryCtx->GetD2hPtr(), opInfo));
    const KfcStatus& aicpuState = opInfo.execStatus.kfcStatus;
    if (aicpuState == KfcStatus::kPlanSwitch || aicpuState == KfcStatus::kSwitchError) {
        CHK_RET(CreateOpRetryAgentByState(RETRY_STATE_SEND_SWITCH_INFO, retryCtx));
        return HCCL_SUCCESS;
    }

    // 查看是否收到server的command
    RetryCommandInfo commandinfo;
    ret = WaitCommandWithOpId(retryCtx->agentSocket_, commandinfo);
    if (ret == HCCL_SUCCESS) {
        if (commandinfo.command == RETRY_CMD_STOP_AICPU) { // 接收到有效command信息
            HCCL_RUN_INFO(
                "[OpRetry][Agent]OpRetryAgentRunning recv command[%s] success, "
                "tag[%s], index[%u], srcRank[%u], detRank[%u], isSendRecv[%d], streamId[%u]",
                GetReadableCmd(commandinfo.command), commandinfo.opId.tag, commandinfo.opId.index,
                commandinfo.opId.srcRank, commandinfo.opId.detRank, commandinfo.opId.isSendRecv,
                commandinfo.opId.streamId);
            CHK_RET(SetOpExecCmdWithOpId(retryCtx->GetH2dPtr(), KfcCommand::kStopLaunch, commandinfo.opId));
            retryCtx->curFaultOpId = commandinfo.opId;
            CHK_RET(CreateOpRetryAgentByState(RETRY_STATE_POLL_AICPU_STOPED, retryCtx));
            return HCCL_SUCCESS;
        } else if (commandinfo.command == RETRY_CMD_RUNNING) {
            // 接收到RUN命令时发送保活数据
            const auto keepTime = std::chrono::duration_cast<std::chrono::seconds>(curTime - lastKeepTime_);
            if (keepTime > keepTimeout_) {
                CHK_RET(GetRetryInfo(retryCtx, retryCtx->localRetryInfo_));
                HcclResult ret = IssueResponse(retryCtx->agentSocket_, retryCtx->localRetryInfo_);
                if (ret != HCCL_SUCCESS) { // 发送保活数据失败, 打印warning
                    HCCL_WARNING("[OpRetry][Agent]OpRetryAgentRunning issue response fail, ret[%d]", ret);
                }
                HCCL_RUN_INFO("[OpRetry][Agent]upload tag[%s]", retryCtx->localRetryInfo_.opInfo.opId.tag);
                lastKeepTime_ = curTime;
            }
        }
    }

    // 轮询间隔
    SaluSleep(OP_RETRY_RUNNING_POLL_INTERVAL);
    return HCCL_SUCCESS;
}

HcclResult OpRetryAgentRunning::ParseRdmaErr(RetryContext* retryCtx, RetryState& nextState)
{
    nextState = RETRY_STATE_RESERVED;
    CHK_RET(GetRetryInfo(retryCtx, retryCtx->localRetryInfo_));
    auto& opId = retryCtx->localRetryInfo_.opInfo.opId;

    // 先判断是否有遗留的cqe err需要处理，没有再去心跳获取
    bool isBatchSendRecv = (opId.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV);
    bool isSendRecv = (opId.opType == HcclCMDType::HCCL_CMD_SEND) || (opId.opType == HcclCMDType::HCCL_CMD_RECEIVE);
    bool IsSupportRdmaRetry = false;
    if (isBatchSendRecv) {
        if (retryCtx->isBSRRdmaSendError_) {
            CHK_RET(GetBsrOpId(retryCtx, HcclSendRecvType::HCCL_SEND));
            nextState = RETRY_STATE_RESP_AICPU_ERR;
            retryCtx->localRetryInfo_.opInfo.execStatus.kfcError = KfcError::kRdma;
            retryCtx->localRetryInfo_.opInfo.execStatus.kfcStatus = KfcStatus::kStoplaunch;
            HCCL_RUN_INFO(
                "[OpRetry][Agent]batchsendrecv rdma send op need retry, tag[%s] index[%u]", opId.tag, opId.index);
            return HCCL_SUCCESS;
        }
        if (retryCtx->isBSRRdmaRecvError_) {
            CHK_RET(GetBsrOpId(retryCtx, HcclSendRecvType::HCCL_RECV));
            nextState = RETRY_STATE_RESP_AICPU_ERR;
            retryCtx->localRetryInfo_.opInfo.execStatus.kfcError = KfcError::kRdma;
            retryCtx->localRetryInfo_.opInfo.execStatus.kfcStatus = KfcStatus::kStoplaunch;
            HCCL_RUN_INFO(
                "[OpRetry][Agent]batchsendrecv rdma recv op need retry, tag[%s] index[%u]", opId.tag, opId.index);
            return HCCL_SUCCESS;
        }
    }
    // 1. 获取 Rdma Err 信息
    std::set<std::tuple<u32, u32, u32>> infoSet;
    Heartbeat::GetInstance(retryCtx->deviceLogicId_).GetQpnErr(retryCtx->group_, infoSet);
    bool isExistQPErr = (infoSet.size() > 0);
    if (!isExistQPErr) {
        return HCCL_SUCCESS;
    }
    if (isBatchSendRecv) {
        for (auto& info : infoSet) {
            u32 qpn = std::get<2>(info);
            u32 qpnStatus = std::get<1>(info);
            if (qpn == opId.bsrInfo[HCCL_SEND].tpQpn && qpnStatus == RDMA_CQE_ERR_STATUS) { // SendQpn  QpnStatus
                retryCtx->isBSRRdmaSendError_ = true;
                CHK_RET(SetBsrOpId(retryCtx, HcclSendRecvType::HCCL_SEND));
            }
            if (qpn == opId.bsrInfo[HCCL_RECV].tpQpn && qpnStatus == RDMA_CQE_ERR_STATUS) { // RecvQpn QpnStatus
                retryCtx->isBSRRdmaRecvError_ = true;
                CHK_RET(SetBsrOpId(retryCtx, HcclSendRecvType::HCCL_RECV));
            }
            HCCL_RUN_INFO(
                "[OpRetry][Agent]pollcqeErr, ErrQpn [%u] SendQpn [%u] RecvQpn [%u]", qpn, opId.bsrInfo[HCCL_SEND].tpQpn,
                opId.bsrInfo[HCCL_RECV].tpQpn);
        }
        // 处理bsr重执行，若send/recv同时报错，优先处理send报错，重执行成功后再处理recv报错
        // 故障是从host侧识别出来的，然后上报故障的时候需要将aicpu侧上报的batchsendrecv刷成sendrecv
        if (retryCtx->isBSRRdmaSendError_) {
            opId.index = opId.bsrInfo[HCCL_SEND].index;
            CHK_SAFETY_FUNC_RET(memset_s(opId.tag, sizeof(opId.tag), 0, sizeof(opId.tag)));
            CHK_SAFETY_FUNC_RET(memcpy_s(
                opId.tag, sizeof(opId.tag), opId.bsrInfo[HCCL_SEND].bsrTag, sizeof(opId.bsrInfo[HCCL_SEND].bsrTag)));
            opId.srcRank = opId.bsrInfo[HCCL_SEND].srcRank;
            opId.detRank = opId.bsrInfo[HCCL_SEND].detRank;
            opId.streamId = opId.bsrInfo[HCCL_SEND].streamId;
            opId.isSendRecv = true;
            opId.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
            IsSupportRdmaRetry = true;
        } else if (retryCtx->isBSRRdmaRecvError_) {
            opId.index = opId.bsrInfo[HCCL_RECV].index;
            CHK_SAFETY_FUNC_RET(memset_s(opId.tag, sizeof(opId.tag), 0, sizeof(opId.tag)));
            CHK_SAFETY_FUNC_RET(memcpy_s(
                opId.tag, sizeof(opId.tag), opId.bsrInfo[HCCL_RECV].bsrTag, sizeof(opId.bsrInfo[HCCL_RECV].bsrTag)));
            opId.srcRank = opId.bsrInfo[HCCL_RECV].srcRank;
            opId.detRank = opId.bsrInfo[HCCL_RECV].detRank;
            opId.streamId = opId.bsrInfo[HCCL_RECV].streamId;
            opId.isSendRecv = true;
            opId.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
            IsSupportRdmaRetry = true;
        }
    } else if (isSendRecv) {
        auto detRank = retryCtx->localRetryInfo_.rankId == retryCtx->localRetryInfo_.opInfo.opId.detRank ?
                           retryCtx->localRetryInfo_.opInfo.opId.srcRank :
                           retryCtx->localRetryInfo_.opInfo.opId.detRank;
        HCCL_INFO("[OpRetry][Agent][Rdma]now in isSendRecv branch (isSendRecv[%d])", isSendRecv);
        bool isFindDstRank = false;
        for (auto& info : infoSet) {
            u32 remoteRank = std::get<0>(info);
            u32 qpnStatus = std::get<1>(info);
            if (remoteRank == detRank) {
                isFindDstRank = true;
                if (qpnStatus == RDMA_CQE_ERR_STATUS) {
                    HCCL_INFO("[OpRetry][Agent][Rdma]SendRecv can support Rdma Retry");
                    IsSupportRdmaRetry = true;
                }
                break;
            }
        }
        if (!isFindDstRank) {
            HCCL_ERROR("[OpRetry][Agent] dstRank[%u] is not in infolist, do nothing", detRank);
            nextState = RETRY_STATE_AGENT_RETRY_FAIL;
        }
        HCCL_INFO("[OpRetry][Agent][Rdma]SendRecv link IsSupportRdmaRetry[%d]", IsSupportRdmaRetry);
    } else {
        // 非点对点通信分支
        HCCL_INFO("[OpRetry][Agent][Rdma]now in Full link branch (isSendRecv[%d])", isSendRecv);
        for (auto& info : infoSet) {
            u32 remoteRank = std::get<0>(info);
            u32 qpnStatus = std::get<1>(info);
            HCCL_INFO("remoteRank = [%u] , status = [%u]", remoteRank, qpnStatus);
            IsSupportRdmaRetry = true; // 默认设置为支持Rdma重执行
            if (qpnStatus != RDMA_CQE_ERR_STATUS) {
                IsSupportRdmaRetry = false;
                nextState = RETRY_STATE_AGENT_RETRY_FAIL;
                HCCL_ERROR("[OpRetry][Agent] remoteRank[%u] status[%u] is not 12", remoteRank, qpnStatus);
                break;
            }
        }
        HCCL_INFO("[OpRetry][Agent][Rdma]Full link IsSupportRdmaRetry[%d]", IsSupportRdmaRetry);
    }
    if (IsSupportRdmaRetry) {
        nextState = RETRY_STATE_RESP_AICPU_ERR;
        retryCtx->localRetryInfo_.opInfo.execStatus.kfcError = KfcError::kRdma;
        retryCtx->localRetryInfo_.opInfo.execStatus.kfcStatus = KfcStatus::kStoplaunch;
    }
    HCCL_INFO("[OpRetry][Agent][Rdma]nextState is [%s]", GetReadableState(nextState));
    return HCCL_SUCCESS;
}

HcclResult OpRetryAgentRunning::ParseKfcErr(RetryContext* retryCtx, RetryState& nextState)
{
    nextState = RETRY_STATE_RESERVED;
    // 记录上一次轮询获取的错误码, 避免日志刷屏
    KfcError lastError = retryCtx->localRetryInfo_.opInfo.execStatus.kfcError;
    CHK_RET(GetRetryInfo(retryCtx, retryCtx->localRetryInfo_));
    KfcError kfcError = retryCtx->localRetryInfo_.opInfo.execStatus.kfcError;
    uint32_t retryCnt = retryCtx->localRetryInfo_.opInfo.execStatus.retryInfo.retryCount;
    switch (kfcError) {
        case KfcError::kNone: {
            break;
        }
        case KfcError::kSdma: {
            HCCL_RUN_INFO("[OpRetry][Agent]Get ErrorCode[%d] retryCnt[%u]", kfcError, retryCnt);
            nextState = RETRY_STATE_RESP_AICPU_ERR;
            break;
        }
        default: {
            if (lastError != kfcError) {
                HCCL_RUN_INFO("[OpRetry][Agent]KfcError[%d] is not support, do nothing", kfcError);
            }
            break;
        }
    }
    return HCCL_SUCCESS;
}

// 向server状态机发送信息
HcclResult OpRetryAgentResponse::ProcessEvent(RetryContext* retryCtx)
{
    // 获取预期的下一个状态
    RetryState nextState = RETRY_STATE_RESERVED;
    auto it = RETRY_AGENT_RESP_STATE_LABEL.find(retryCtx->localRetryInfo_.retryState);
    CHK_PRT_RET(
        it == RETRY_AGENT_RESP_STATE_LABEL.end(),
        HCCL_ERROR(
            "[OpRetry][Agent]OpRetryAgentResponse fail, state[%s] is not in RETRY_AGENT_RESP_STATE_LABEL",
            GetReadableState(retryCtx->localRetryInfo_.retryState)),
        HCCL_E_INTERNAL);
    nextState = it->second;
    auto& opInfo = retryCtx->localRetryInfo_.opInfo;
    HCCL_RUN_INFO(
        "[OpRetry][Agent]OpRetryAgentResponse tag[%s], index[%u], srcRank[%u], detRank[%u], isSendRecv[%d],"
        "opExeState[%d], errorCode[%d], retryCount[%u], streamId[%u], isNeedReportOpRetryErr[%d]",
        opInfo.opId.tag, opInfo.opId.index, opInfo.opId.srcRank, opInfo.opId.detRank, opInfo.opId.isSendRecv,
        opInfo.execStatus.kfcStatus, opInfo.execStatus.kfcError, opInfo.execStatus.retryInfo.retryCount,
        opInfo.opId.streamId, retryCtx->localRetryInfo_.isNeedReportOpRetryErr);

    // 发送数据
    HcclResult ret = IssueResponse(retryCtx->agentSocket_, retryCtx->localRetryInfo_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[OpRetry][Agent]OpRetryAgentResponse IssueResponse fail"), ret);
    CHK_RET(CreateOpRetryAgentByState(nextState, retryCtx));
    return HCCL_SUCCESS;
}

// 从server状态机接收命令
HcclResult OpRetryAgentWaitCmd::ProcessEvent(RetryContext* retryCtx)
{
    RetryState curState = retryCtx->localRetryInfo_.retryState;
    RetryState nextState = RETRY_STATE_RESERVED;
    RetryCommandInfo commandinfo;
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    const u32 timeoutValue = std::max(static_cast<u32>(GetExternalInputHcclLinkTimeOut()), OP_RETRY_SEND_RECV_TIMEOUT)
                             + OP_RETRY_WAIT_AICPU_TIMEOUT;
    const std::chrono::seconds timeout = std::chrono::seconds(timeoutValue);

    // 接收到命令和当前状态不匹配时, 不做处理, 等待下一个命令, 直到命令正确或者超时
    while (true) {
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        CHK_PRT_RET(elapsed > timeout, HCCL_ERROR("[OpRetry]WaitCommand timeout"), HCCL_E_TIMEOUT);
        CHK_PRT_RET(
            retryCtx->isAgentStateWaitResume_,
            HCCL_RUN_INFO("[OpRetry][Agent]switched state from wait cmd to wait resume"), HCCL_SUCCESS);

        HcclResult ret = WaitCommandWithOpId(retryCtx->agentSocket_, commandinfo);
        if (ret == HCCL_SUCCESS) {
            HCCL_RUN_INFO(
                "[OpRetry][Agent]OpRetryAgentGetCmd state[%s] command[%s]"
                "tag[%s], index[%u], srcRank[%u], detRank[%u], isSendRecv[%d], streamid[%u]",
                GetReadableState(curState), GetReadableCmd(commandinfo.command), commandinfo.opId.tag,
                commandinfo.opId.index, commandinfo.opId.srcRank, commandinfo.opId.detRank, commandinfo.opId.isSendRecv,
                commandinfo.opId.streamId);
            CHK_PRT(ParseCommandWithOpId(retryCtx, commandinfo, nextState));
            if (nextState != RETRY_STATE_RESERVED) { // 接收到的命令有效
                break;
            }
        }
    }
    CHK_RET(CreateOpRetryAgentByState(nextState, retryCtx));
    return HCCL_SUCCESS;
}

HcclResult
OpRetryAgentWaitCmd::ParseCommandWithOpId(RetryContext* retryCtx, RetryCommandInfo& commandinfo, RetryState& nextState)
{
    if (!retryCtx->isChangeLinkInfoInit_) {
        CHK_RET(InitChangeLinkInfo(retryCtx));
        retryCtx->isChangeLinkInfoInit_ = true;
    } else {
        // 增量建链场景
        CHK_RET(InitChangeLinkInfo(retryCtx, true));
    }
    RetryState curState = retryCtx->localRetryInfo_.retryState;
    switch (commandinfo.command) {
        case RETRY_CMD_STOP_AICPU:
            if (curState == RETRY_STATE_WAIT_CMD_STOP_AICPU) {
                CHK_RET(SetOpExecCmdWithOpId(retryCtx->GetH2dPtr(), KfcCommand::kStopLaunch, commandinfo.opId));
                retryCtx->curFaultOpId = commandinfo.opId;
                nextState = RETRY_STATE_POLL_AICPU_STOPED;
            }
            break;
        case RETRY_CMD_STOP_STREAM:
            if (curState == RETRY_STATE_WAIT_CMD_STOP_STREAM) {
                CHK_RET(ClearStreamWithOpId(
                    retryCtx->opStreamPtr_, HcclRtStreamClearStep::HCCL_STREAM_STOP, commandinfo.opId,
                    retryCtx->localRetryInfo_.opInfo.opId));
                CHK_RET(SetOpExecCmdWithOpId(retryCtx->GetH2dPtr(), KfcCommand::kStopExec, commandinfo.opId));
                nextState = RETRY_STATE_POLL_STREAM_STOPED;
            }
            break;
        case RETRY_CMD_CLEAR_STREAM:
            if (curState == RETRY_STATE_WAIT_CMD_CLEAR_STREAM) {
                CHK_RET(ClearStreamWithOpId(
                    retryCtx->opStreamPtr_, HcclRtStreamClearStep::HCCL_STREAM_CLEAR, commandinfo.opId,
                    retryCtx->localRetryInfo_.opInfo.opId));
                nextState = RETRY_STATE_RESP_STREAM_CLEARED;
            }
            break;
        case RETRY_CMD_STOP_TRANSPORT:
            if (curState == RETRY_STATE_WAIT_CMD_STOP_TRANSPORT) {
                CHK_RET(SetTransportStatusForStop(retryCtx));
                nextState = RETRY_STATE_RESP_STOP_TRANSPORT;
            }
            break;
        case RETRY_CMD_CHECK_LINK:
            if (curState == RETRY_STATE_WAIT_CMD_CHECK_LINK) {
                u32& retryCnt = retryCtx->localRetryInfo_.opInfo.execStatus.retryInfo.retryCount;
                CommConfiger& commConfiger = CommConfiger::GetInstance();
                u32 waitTime = (retryCnt == 0) ? commConfiger.GetCommConfigRetryHoldTime(retryCtx->group_) :
                                                 commConfiger.GetCommConfigRetryIntervalTime(retryCtx->group_);
                constexpr u32 TIME_MS_TO_US = 1000;
                SaluSleep(waitTime * TIME_MS_TO_US);
                HCCL_RUN_INFO("[OpRetry][Agent]wait for [%u]ms until the link recovers", waitTime);
                CHK_RET(GetLinkPortStatus(retryCtx, retryCtx->linkPortStatus_));
                nextState = RETRY_STATE_RESP_LINK_CHECKED;
            }
            break;
        case RETRY_CMD_RESUME_TRANSPORT:
            if (curState == RETRY_STATE_WAIT_CMD_RESUME_TRANSPORT) {
                CHK_RET(SetTransportStatusForResume(retryCtx));
                // 重新建链后给aicpu下发切换链路命令
                CHK_RET(SetOpChangeLinkInfo(
                    retryCtx->GetH2dPtr(), KfcCommand::kChangeLink, retryCtx->localChangeLinkInfo_));
                nextState = RETRY_STATE_POLL_AICPU_CHANGED;
            }
            break;
        case RETRY_CMD_RESET_NOTIFY:
            if (curState == RETRY_STATE_WAIT_CMD_RESET_NOTIFY) {
                CHK_RET(ResetNotify(retryCtx));
                nextState = RETRY_STATE_RESP_NOTIFY_RESETED;
            }
            break;
        case RETRY_CMD_CHECK_OPNAME:
            if (curState == RETRY_STATE_WAIT_CMD_CHECK) {
                CHK_RET(GetRetryInfo(retryCtx, retryCtx->localRetryInfo_));
                nextState = RETRY_STATE_RESP_CHECK_INFO;
            }
            break;
        case RETRY_CMD_CAN_RETRY:
            if (curState == RETRY_STATE_WAIT_CMD_CAN_RETRY) {
                bool isSendRecv = HcclCMDType::HCCL_CMD_SEND == retryCtx->localRetryInfo_.opInfo.opId.opType
                                  || HcclCMDType::HCCL_CMD_RECEIVE == retryCtx->localRetryInfo_.opInfo.opId.opType;
                u32 dstRank = retryCtx->localRetryInfo_.rankId == retryCtx->localRetryInfo_.opInfo.opId.detRank ?
                                  retryCtx->localRetryInfo_.opInfo.opId.srcRank :
                                  retryCtx->localRetryInfo_.opInfo.opId.detRank;
                if (isSendRecv) {
                    Heartbeat::GetInstance(retryCtx->deviceLogicId_).ClearCqeErr(retryCtx->group_, dstRank);
                } else if (HcclCMDType::HCCL_CMD_BATCH_SEND_RECV == retryCtx->localRetryInfo_.opInfo.opId.opType) {
                    ResetBatchSendRecvRdmaErr(retryCtx, dstRank);
                } else {
                    Heartbeat::GetInstance(retryCtx->deviceLogicId_).ClearAllCqeErr(retryCtx->group_);
                }
                CHK_RET(SetOpExecCmdWithOpId(retryCtx->GetH2dPtr(), KfcCommand::kRetry, commandinfo.opId));
                nextState = RETRY_STATE_POLL_AICPU_RETRYEND;
            }
            break;
        case RETRY_CMD_RETRY_CONSTRAINT_FAIL:
            nextState = RETRY_STATE_AGENT_RETRY_FAIL;
            retryCtx->localRetryInfo_.isNeedReportOpRetryErr = true;
            HCCL_RUN_INFO("[OpRetry][Agent]Retry is constraint(OpName is inconsistent or Inplace Error)");
            break;
        case RETRY_CMD_RETRY_FAIL:
            nextState = RETRY_STATE_AGENT_RETRY_FAIL;
            break;
        default: { // 命令非当前状态预期, 不处理
            break;
        }
    }
    return HCCL_SUCCESS;
}

void OpRetryAgentWaitCmd::ResetBatchSendRecvRdmaErr(RetryContext* retryCtx, u32 dstRank)
{
    bool isBatchSendRecv = (retryCtx->localRetryInfo_.opInfo.opId.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV);
    if (isBatchSendRecv) {
        auto curOpIdTag = std::string(reinterpret_cast<const char*>(retryCtx->localRetryInfo_.opInfo.opId.tag));
        auto curOpIdindex = retryCtx->localRetryInfo_.opInfo.opId.index;
        auto remainSendOpIdTag = std::string(reinterpret_cast<const char*>(retryCtx->RemainSendOpId_.tag));
        auto remainRecvOpIdTag = std::string(reinterpret_cast<const char*>(retryCtx->RemainRecvOpId_.tag));
        if (curOpIdTag == remainSendOpIdTag && curOpIdindex == retryCtx->RemainSendOpId_.index) {
            retryCtx->isBSRRdmaSendError_ = false;
            HCCL_INFO(
                "[OpRetry][Agent] bsr send clear cqe err, remoterank[%u], qpn[%u]", dstRank,
                retryCtx->RemainSendOpId_.bsrInfo[HCCL_SEND].tpQpn);
            Heartbeat::GetInstance(retryCtx->deviceLogicId_)
                .ClearCqeErr(retryCtx->group_, dstRank, retryCtx->RemainSendOpId_.bsrInfo[HCCL_SEND].tpQpn);
        }
        if (curOpIdTag == remainRecvOpIdTag && curOpIdindex == retryCtx->RemainRecvOpId_.index) {
            retryCtx->isBSRRdmaRecvError_ = false;
            HCCL_INFO(
                "[OpRetry][Agent] bsr recv clear cqe err, remoterank[%u], qpn[%u]", dstRank,
                retryCtx->RemainRecvOpId_.bsrInfo[HCCL_RECV].tpQpn);
            Heartbeat::GetInstance(retryCtx->deviceLogicId_)
                .ClearCqeErr(retryCtx->group_, dstRank, retryCtx->RemainRecvOpId_.bsrInfo[HCCL_RECV].tpQpn);
        }
    }
    return;
}

HcclResult OpRetryAgentPollAicpuStop::ProcessEvent(RetryContext* retryCtx)
{
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    const std::chrono::seconds timeout = std::chrono::seconds(OP_RETRY_WAIT_AICPU_TIMEOUT);
    RetryState curState = retryCtx->GetRetryState();
    RetryState nextState = RETRY_STATE_RESERVED;
    while (true) {
        CHK_PRT_RET(
            retryCtx->isAgentStateWaitResume_,
            HCCL_RUN_INFO("[OpRetry][Agent]switched state from poll aicpu to wait resume"), HCCL_SUCCESS);
        // 读取aicpuCtx中的状态
        KfcExecStatus& opInfo = retryCtx->localRetryInfo_.opInfo;
        CHK_RET(GetOpExecInfo(retryCtx->GetD2hPtr(), opInfo));
        const KfcStatus& aicpuState = opInfo.execStatus.kfcStatus;
        const char* tag = reinterpret_cast<const char*>(opInfo.opId.tag);
        u32 index = opInfo.opId.index;
        KfcError errorCode = opInfo.execStatus.kfcError;

        std::string curFaultTag = std::string(reinterpret_cast<const char*>(retryCtx->curFaultOpId.tag));
        std::string curd2hTag = std::string(reinterpret_cast<const char*>(opInfo.opId.tag));
        switch (curState) {
            case RETRY_STATE_POLL_AICPU_STOPED:
                if (aicpuState == KfcStatus::kStoplaunch || aicpuState == KfcStatus::kStopExec) {
                    if ((retryCtx->curFaultOpId.isSendRecv && curFaultTag == curd2hTag) || !opInfo.opId.isSendRecv) {
                        HCCL_RUN_INFO(
                            "[OpRetry][Agent]curFaultTag[%s] curd2hTag[%s], isSendRecv[%u]", curFaultTag.c_str(),
                            curd2hTag.c_str(), retryCtx->curFaultOpId.isSendRecv);
                        nextState = RETRY_STATE_RESP_AICPU_STOPED;
                    }
                } else if (aicpuState == KfcStatus::kRetryError && errorCode == KfcError::kExecConstraint) {
                    retryCtx->localRetryInfo_.isNeedReportOpRetryErr = true;
                    HCCL_ERROR("[OpRetry][Agent]can not retry for constraint");
                    nextState = RETRY_STATE_RESP_RUNNING_ERR;
                    break;
                }
                break;
            case RETRY_STATE_POLL_STREAM_STOPED:
                if (aicpuState == KfcStatus::kStopExec) {
                    nextState = RETRY_STATE_RESP_STREAM_STOPED;
                } else if (aicpuState == KfcStatus::kRetryError && errorCode == KfcError::kExecConstraint) {
                    retryCtx->localRetryInfo_.isNeedReportOpRetryErr = true;
                    HCCL_ERROR("[OpRetry][Agent]can not retry for constraint");
                    nextState = RETRY_STATE_RESP_RUNNING_ERR;
                    break;
                }
                break;
            case RETRY_STATE_POLL_AICPU_CHANGED:
                HCCL_RUN_INFO(
                    "[OpRetry][Agent]OpRetryAgentPollAicpuStop hostState[%s], aicpuState[%d]",
                    GetReadableState(curState), aicpuState);
                if (aicpuState == KfcStatus::kChanged) {
                    nextState = RETRY_STATE_RESP_RESUME_TRANSPORT;
                }
                break;
            case RETRY_STATE_POLL_AICPU_RETRYEND:
                if (aicpuState == KfcStatus::kStoplaunch || aicpuState == KfcStatus::kRuning
                    || aicpuState == KfcStatus::kEnd) {
                    nextState = RETRY_STATE_RESP_AICPU_RETRYEND;
                }
                break;
            default: {
                HCCL_ERROR(
                    "[OpRetry][Agent]OpRetryAgentPollAicpuStop state[%s] is invalid", GetReadableState(curState));
                return HCCL_E_INTERNAL;
            }
        }

        // 执行成功, 跳出循环进入下一个状态
        if (nextState != RETRY_STATE_RESERVED) {
            HCCL_RUN_INFO(
                "[OpRetry][Agent]OpRetryAgentPollAicpuStop success, retryState[%s], aicpuState[%d], "
                "tag[%s], index[%u]",
                GetReadableState(curState), aicpuState, tag, index);
            HCCL_RUN_INFO(
                "[OpRetry][agent pollaicpu OpId]tag[%s], index[%u], srcRank[%u], detRank[%u], isSendRecv[%d],"
                "streamid[%u], retryCnt[%u]",
                opInfo.opId.tag, opInfo.opId.index, opInfo.opId.srcRank, opInfo.opId.detRank, opInfo.opId.isSendRecv,
                opInfo.opId.streamId, opInfo.execStatus.retryInfo.retryCount);
            break;
        }

        // 超时机制
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        CHK_PRT_BREAK(
            elapsed >= timeout,
            HCCL_ERROR(
                "[OpRetry][Agent]OpRetryAgentPollAicpuStop timeout, retryState[%s], aicpuState[%d], "
                "tag[%s], index[%u]",
                GetReadableState(curState), aicpuState, tag, index),
            nextState = RETRY_STATE_RESP_RUNNING_ERR);

        // 轮询间隔
        SaluSleep(OP_RETRY_POLL_AICPU_STATE_INTERVAL);
    }

    CHK_RET(CreateOpRetryAgentByState(nextState, retryCtx));
    return HCCL_SUCCESS;
}

// 向server状态机发送主备链路信息
HcclResult OpRetryAgentResponseLinkInfo::ProcessEvent(RetryContext* retryCtx)
{
    HCCL_INFO("[OpRetry][Agent]OpRetryAgentResponseLinkInfo begin");
    // 获取预期的下一个状态
    RetryState nextState = RETRY_STATE_WAIT_CHANGE_LINK_INFO;

    // 发送数据
    HcclResult ret = IssueLinkPortCheckResult(retryCtx->agentSocket_, retryCtx->linkPortStatus_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[OpRetry][Agent]OpRetryAgentResponseLinkInfo IssueResponse fail"), ret);
    CHK_RET(CreateOpRetryAgentByState(nextState, retryCtx));
    HCCL_INFO("[OpRetry][Agent]OpRetryAgentResponseLinkInfo success");
    return HCCL_SUCCESS;
}

void OpRetryAgentWaitChangeLinkInfo::UpdateChangeLinkInfo(
    ChangeLinkInfo& localChangeLinkInfo, ChangeLinkInfo& recvChangeLinkInfo)
{
    // 记录localChangeLinkInfo_中已有的数据
    std::unordered_map<u32, u32> remoteRankPosition; // {remoteRank: position}
    for (u32 i = 0; i < localChangeLinkInfo.remoteRankNum; i++) {
        remoteRankPosition.insert({localChangeLinkInfo.remoteRankList[i], i});
    }
    // 将接收到的recvChangeLinkInfo更新到localChangeLinkInfo_中
    for (u32 i = 0; i < recvChangeLinkInfo.remoteRankNum; i++) {
        u32 remoteRank = recvChangeLinkInfo.remoteRankList[i];
        bool isUseDefaultPort = recvChangeLinkInfo.isUseDefaultPort[i];
        if (remoteRankPosition.find(remoteRank) != remoteRankPosition.end()) {
            // 若remoteRank在localChangeLinkInfo_中，更新其端口使用情况
            localChangeLinkInfo.isUseDefaultPort[remoteRankPosition[remoteRank]] = isUseDefaultPort;
            HCCL_RUN_INFO(
                "[OpRetry][Agent]update remoteRank[%u] to isUseDefaultPort[%d]", remoteRank, isUseDefaultPort);
        } else {
            // 若remoteRank不在localChangeLinkInfo_中，则添加到localChangeLinkInfo_中
            u32 position = localChangeLinkInfo.remoteRankNum;
            localChangeLinkInfo.remoteRankList[position] = remoteRank;
            localChangeLinkInfo.isUseDefaultPort[position] = isUseDefaultPort;
            localChangeLinkInfo.remoteRankNum += 1;
            HCCL_RUN_INFO(
                "[OpRetry][Agent]insert remoteRank[%u] to isUseDefaultPort[%d]", remoteRank, isUseDefaultPort);
        }
    }
    return;
}

// 从server状态机接收主备借轨命令
HcclResult OpRetryAgentWaitChangeLinkInfo::ProcessEvent(RetryContext* retryCtx)
{
    HCCL_INFO("[OpRetry][Agent]OpRetryAgentWaitChangeLinkInfo begin");
    RetryState nextState = RETRY_STATE_RESERVED;

    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    const u32 timeoutValue = std::max(static_cast<u32>(GetExternalInputHcclLinkTimeOut()), OP_RETRY_SEND_RECV_TIMEOUT)
                             + OP_RETRY_WAIT_AICPU_TIMEOUT;
    const std::chrono::seconds timeout = std::chrono::seconds(timeoutValue);
    ChangeLinkInfo tmpRecvChangeLinkInfo;
    // 接收到命令和当前状态不匹配时, 不做处理, 等待下一个命令, 直到命令正确或者超时
    while (true) {
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        CHK_PRT_RET(elapsed > timeout, HCCL_ERROR("[OpRetry]WaitChangeLink timeout"), HCCL_E_TIMEOUT);
        CHK_PRT_RET(
            retryCtx->isAgentStateWaitResume_,
            HCCL_RUN_INFO("[OpRetry][Agent]switched state from wait change link to wait resume"), HCCL_SUCCESS);

        HcclResult ret = WaitChangeLink(retryCtx->agentSocket_, tmpRecvChangeLinkInfo);
        if (ret == HCCL_SUCCESS) {
            HCCL_INFO("[OpRetry][Agent]WaitChangeLink success");
            // 将接收到的ChangeLinkInfo更新到已有的changeLinkInfo中
            UpdateChangeLinkInfo(retryCtx->localChangeLinkInfo_, tmpRecvChangeLinkInfo);
            // agent接收的changeLinkInfo信息
            std::string changeLinkInfoStr = "agent:";
            for (u32 i = 0; i < retryCtx->localChangeLinkInfo_.remoteRankNum; i++) {
                changeLinkInfoStr
                    += (std::to_string(retryCtx->localChangeLinkInfo_.remoteRankList[i]) + ":"
                        + std::to_string(retryCtx->localChangeLinkInfo_.isUseDefaultPort[i]) + "; ");
            }
            HCCL_RUN_INFO("[OpRetry][Agent]changeLinkInfoStr:%s", changeLinkInfoStr.c_str());

            // 收到changelinkinfo后切换到RETRY_STATE_WAIT_CMD_RESUME_TRANSPORT状态等待接收resume transport命令
            nextState = RETRY_STATE_WAIT_CMD_RESUME_TRANSPORT;
            break;
        }
    }
    CHK_RET(CreateOpRetryAgentByState(nextState, retryCtx));
    HCCL_INFO("[OpRetry][Agent]OpRetryAgentWaitChangeLinkInfo success");
    return HCCL_SUCCESS;
}

// RETRY_STATE_AGENT_RETRY_FAIL
HcclResult OpRetryAgentRetryFail::ProcessEvent(RetryContext* retryCtx)
{
    HCCL_INFO("[OpRetry][Agent]OpRetryAgentRetryFail, set state to running");
    if (retryCtx->localRetryInfo_.isNeedReportOpRetryErr) {
        CHK_RET(SetOpExecCmd(retryCtx->GetH2dPtr(), KfcCommand::kReportRetryErr));
        HCCL_RUN_INFO(
            "[OpRetry][Agent]OpRetryAgentRetryFail, isNeeReportOpRetryErr[%d]",
            retryCtx->localRetryInfo_.isNeedReportOpRetryErr);
    } else {
        CHK_RET(SetOpExecCmd(retryCtx->GetH2dPtr(), KfcCommand::kExit));
    }
    CHK_RET(CreateOpRetryAgentByState(RETRY_STATE_AGENT_RUNNING, retryCtx));
    Heartbeat::GetInstance(retryCtx->deviceLogicId_).BroadcastCqeErr(retryCtx->group_);
    return HCCL_SUCCESS;
}

HcclResult OpRetryAgentWaitResume::ProcessEvent(RetryContext* retryCtx)
{
    if (!retryCtx->isAgentStateWaitResume_ && !retryCtx->haveCommEnableBackupLink_) {
        CHK_RET(CreateOpRetryAgentByState(RETRY_STATE_AGENT_RUNNING, retryCtx));
        HCCL_RUN_INFO(
            "[OpRetry][Agent]OpRetryAgentWaitResume, group[%s], no comm enable backup link, set state to running",
            retryCtx->group_.c_str());
        return HCCL_SUCCESS;
    }
    RetryCommandInfo commandInfo;
    HcclResult ret = WaitCommandWithOpId(retryCtx->agentSocket_, commandInfo);
    if (ret == HCCL_SUCCESS) {
        HCCL_INFO(
            "[OpRetry][Agent]OpRetryAgentWaitResume, rankId[%u], command[%s], group[%s], isAgentStateWaitResume_[%d]",
            retryCtx->localRetryInfo_.rankId, GetReadableCmd(commandInfo.command), retryCtx->group_.c_str(),
            retryCtx->isAgentStateWaitResume_);
    }
    if (commandInfo.command == RESUME_CMD_RUNNING) {
        retryCtx->isRecivedCmdToRunning = true;
    } else if (commandInfo.command == RESUME_CMD_CHECK_LINK) {
        retryCtx->isRecivedCmdToCheckLink = true;
    }
    if (!retryCtx->isAgentStateWaitResume_ && retryCtx->isRecivedCmdToRunning) {
        CHK_RET(CreateOpRetryAgentByState(RETRY_STATE_AGENT_RUNNING, retryCtx));
        retryCtx->isRecivedCmdToRunning = false;
        HCCL_RUN_INFO("[OpRetry][Agent]OpRetryAgentWaitResume, set state to running");
    } else if (!retryCtx->isAgentStateWaitResume_ && retryCtx->isRecivedCmdToCheckLink) {
        CHK_RET(CreateOpRetryAgentByState(RETRY_RESUME_STATE_AGENT_CHECK_LINK, retryCtx));
        HCCL_RUN_INFO("[OpRetry][Agent]OpRetryAgentWaitResume, set state to check link");
    }
    return HCCL_SUCCESS;
}

// RETRY_STATE_WAIT_CMD_SEND_AICPU
HcclResult SwitchNicAgentWaitCmd::ParseCommand(RetryContext* retryCtx, RetryCommand& command, RetryState& nextState)
{
    switch (command) {
        case RETRY_CMD_NOTIFY_SWITCH_SUC:
            HCCL_INFO("[SwitchNic][Agent][WaitCmd] switch nic success, rank[%u]", retryCtx->rankId_);
            // OpRetryAgent接收到全局通信域的成功/错误信息后，刷新lastLinkPortStatus_信息
            for (u32 i = 0; i < retryCtx->switchInfo_.switchRankNum; i++) {
                if (retryCtx->switchInfo_.switchRankList[i] == retryCtx->rankId_) {
                    retryCtx->isUseDefaultPort_ = !retryCtx->switchInfo_.switchUseBackup[i];
                }
            }
            for (u32 i = 0; i < retryCtx->switchInfo_.remoteRankNum; i++) {
                if (retryCtx->switchInfo_.remoteRankNicStatus[i] == CONNECT_REMOTE_DEFAULT) {
                    retryCtx->lastLinkPortStatus_[i] = true;
                }
                if (retryCtx->switchInfo_.remoteRankNicStatus[i] == CONNECT_REMOTE_BACKUP) {
                    retryCtx->lastLinkPortStatus_[i] = false;
                }
            }
            CHK_RET(SetOpExecCmd(retryCtx->GetH2dPtr(), KfcCommand::kAllSwitched));
            nextState = RETRY_STATE_AGENT_RUNNING;
            break;
        case RETRY_CMD_NOTIFY_SWITCH_FAIL:
            HCCL_ERROR("[SwitchNic][Agent][WaitCmd] switch nic failed, rank[%u]", retryCtx->rankId_);
            CHK_RET(SetOpExecCmd(retryCtx->GetH2dPtr(), KfcCommand::kSwitchFail));
            nextState = RETRY_STATE_AGENT_RUNNING;
            break;
        case RETRY_CMD_RUNNING:
            HCCL_DEBUG(
                "[SwitchNic][Agent][WaitCmd] rank[%u] recv running command from server, ignored", retryCtx->rankId_);
            break;
        default: { // 命令非当前状态预期
            HCCL_ERROR(
                "[SwitchNic][Agent][WaitCmd] rank[%u], recv unexpected parse command[%s:%u].", retryCtx->rankId_,
                GetReadableCmd(command), command);
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult SwitchNicAgentWaitCmd::ProcessEvent(RetryContext* retryCtx)
{
    HCCL_RUN_INFO("[SwitchNic][Agent] rank[%u] begin to wait server cmd, set state to wait", retryCtx->rankId_);
    RetryState curState = retryCtx->localRetryInfo_.retryState;
    RetryState nextState = RETRY_STATE_RESERVED;
    RetryCommand command;
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut() * ACTIVE_SWITCH_TIMES);

    // 接收到命令和当前状态不匹配时, 不做处理, 等待下一个命令, 直到命令正确或者超时
    while (true) {
        CHK_PRT_RET(
            retryCtx->isAgentStateWaitResume_,
            HCCL_RUN_INFO("[OpRetry][Agent]switched state from switch nic to wait resume"), HCCL_SUCCESS);
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        CHK_PRT_RET(
            elapsed > timeout,
            HCCL_ERROR("[SwitchNic][Agent] timeout in getting cmd from server, waittime[%u s>%u s]", elapsed, timeout),
            HCCL_E_TIMEOUT);

        HcclResult ret = WaitCommand(retryCtx->agentSocket_, command);
        if (ret == HCCL_SUCCESS) {
            HCCL_DEBUG(
                "[SwitchNic][Agent]SwitchNicAgentWaitCmd state[%s] command[%s:%u]", GetReadableState(curState),
                GetReadableCmd(command), command);
            CHK_PRT(ParseCommand(retryCtx, command, nextState));
            if (nextState != RETRY_STATE_RESERVED) { // 接收到的命令有效
                break;
            }
        }
    }
    CHK_RET(CreateOpRetryAgentByState(nextState, retryCtx));
    return HCCL_SUCCESS;
}

HcclResult SwitchNicAgentSendSwitchInfo::ChangeAicpuStatus(RetryContext* retryCtx)
{
    HcclResult ret = SetOpExecCmd(retryCtx->GetH2dPtr(), KfcCommand::kWaitSwitchNic);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[SwitchNic][Agent] rank[%u], ChangeAicpuStatus, SetOpExecCmd to waitSwitchNic fail, ", retryCtx->rankId_);
    }
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    const std::chrono::seconds timeout = std::chrono::seconds(OP_RETRY_WAIT_AICPU_TIMEOUT);
    while (true) {
        CHK_PRT_RET(
            retryCtx->isAgentStateWaitResume_,
            HCCL_RUN_INFO("[OpRetry][Agent]switched state from send switch nic info to wait resume"), HCCL_SUCCESS);
        KfcExecStatus& opInfo = retryCtx->localRetryInfo_.opInfo;
        CHK_RET(GetOpExecInfo(retryCtx->GetD2hPtr(), opInfo));
        const KfcStatus& aicpuState = opInfo.execStatus.kfcStatus;
        if (aicpuState == KfcStatus::kWaitSwitchRes) {
            break;
        }

        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        CHK_PRT_RET(
            elapsed >= timeout,
            HCCL_ERROR(
                "[SwitchNic][Agent]SwitchNicAgentSendSwitchInfo, WaitAicpuResponse timeout, "
                "rank[%u], aicpuState[%d]",
                retryCtx->rankId_, aicpuState),
            HCCL_E_TIMEOUT);
    }
    return HCCL_SUCCESS;
}

HcclResult SwitchNicAgentSendSwitchInfo::CheckLocalPortStatus(RetryContext* retryCtx)
{
    HcclResult ret = HcclNetDevGetPortStatus(retryCtx->netDevCtx_, retryCtx->linkPortStatus_.defaultPort);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[SwitchNic][Agent] rank[%u], get default port status fail.", retryCtx->rankId_), HCCL_E_INTERNAL);
    ret = HcclNetDevGetPortStatus(retryCtx->backUpNetDevCtx_, retryCtx->linkPortStatus_.backupPort);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[SwitchNic][Agent] rank[%u], get backup port status fail.", retryCtx->rankId_),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

// RETRY_STATE_SEND_SWITCH_INFO
HcclResult SwitchNicAgentSendSwitchInfo::ProcessEvent(RetryContext* retryCtx)
{
    HCCL_RUN_INFO("[SwitchNic][Agent] rank[%u] begin to send switch info, set state to switch", retryCtx->rankId_);

    ActiveSwitchInfo& switchInfo = retryCtx->switchInfo_;
    const KfcStatus transportStatus = retryCtx->localRetryInfo_.opInfo.execStatus.kfcStatus;
    switchInfo.refreshTransportFin = (transportStatus == KfcStatus::kPlanSwitch) ? true : false;
    if (!switchInfo.refreshTransportFin) {
        HCCL_ERROR(
            "[SwitchNic][Agent] rank[%u], recv refresh transport kfcStatus[%u], "
            "refreshTransportFin is set to false, ",
            retryCtx->rankId_, transportStatus);
    }

    HcclResult ret = ChangeAicpuStatus(retryCtx);
    if (ret != HCCL_SUCCESS) {
        switchInfo.refreshTransportFin = false;
        HCCL_ERROR(
            "[SwitchNic][Agent] rank[%u], ChangeAicpuStatus fail, refreshTransportFin is set to false.",
            retryCtx->rankId_);
    }

    // 校验本端准备使用的主/备网卡状态
    ret = CheckLocalPortStatus(retryCtx);
    if (ret != HCCL_SUCCESS) {
        switchInfo.localPortsCheckRet = false;
        HCCL_ERROR(
            "[SwitchNic][Agent] rank[%u], get local port status fail, "
            "localPortsCheckRet is set to false.",
            retryCtx->rankId_);
    }

    // 并将本端完成借轨的flag, 以及之前登记的switchRanks信息发送给OpRetryServer
    bool needCheckDefaultNic = false;
    bool needCheckBackupNic = false;
    ret = GetSwitchRanks(retryCtx, needCheckDefaultNic, needCheckBackupNic);
    if (ret != HCCL_SUCCESS) {
        switchInfo.refreshTransportFin = false;
        HCCL_ERROR(
            "[SwitchNic][Agent] rank[%u], get switch rank info fail, "
            "refreshTransportFin is set to false.",
            retryCtx->rankId_);
    }

    switchInfo.defaultPortStatus = retryCtx->linkPortStatus_.defaultPort;
    switchInfo.backupPortStatus = retryCtx->linkPortStatus_.backupPort;
    if ((needCheckDefaultNic && !switchInfo.defaultPortStatus)
        || (needCheckBackupNic && !switchInfo.backupPortStatus)) {
        switchInfo.localPortsCheckRet = false;
        HCCL_ERROR(
            "[SwitchNic][Agent] localPortsCheckRet is false, needCheckDefaultNic[%u], needCheckBackupNic[%u],"
            "localPortsCheckRet[%u], defaultPortStatus[%u], backupPortStatus[%u], rank[%u]",
            needCheckDefaultNic, needCheckBackupNic, switchInfo.localPortsCheckRet, switchInfo.defaultPortStatus,
            switchInfo.backupPortStatus, retryCtx->rankId_);
    } else {
        switchInfo.localPortsCheckRet = true;
    }
    HCCL_RUN_INFO(
        "[SwitchNic][Agent] switch info, refresh transport kfcStatus[%u], needCheckDefaultNic[%u], "
        "needCheckBackupNic[%u], localPortsCheckRet[%u], defaultPortStatus[%u], backupPortStatus[%u], "
        "refreshTransportFin[%u], switchRankNum[%u], remoteRankNum[%u], rank[%u]",
        transportStatus, needCheckDefaultNic, needCheckBackupNic, switchInfo.localPortsCheckRet,
        switchInfo.defaultPortStatus, switchInfo.backupPortStatus, switchInfo.refreshTransportFin,
        switchInfo.switchRankNum, switchInfo.remoteRankNum, retryCtx->rankId_);
    //  第一次发送retryInfo，目的是为了server确认为主动接轨场景再接收switchInfo
    RetryInfo retryInfo = {};
    retryInfo.retryState = RETRY_STATE_SEND_SWITCH_INFO;
    retryInfo.rankId = retryCtx->rankId_;
    ret = IssueResponse(retryCtx->agentSocket_, retryInfo);
    if (ret == HCCL_SUCCESS) {
        ret = IssueActiveSwitchInfo(retryCtx->agentSocket_, switchInfo);
    } else {
        HCCL_ERROR("[SwitchNic][Agent] rank[%u], issue response to server fail.", retryCtx->rankId_);
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[SwitchNic][Agent] rank[%u], issue response or active switch info to server fail, "
            "send result to aicpu.",
            retryCtx->rankId_);
        CHK_RET(SetOpExecCmd(retryCtx->GetH2dPtr(), KfcCommand::kSwitchFail));
        CHK_RET(CreateOpRetryAgentByState(RETRY_STATE_AGENT_RUNNING, retryCtx));
    } else {
        CHK_RET(CreateOpRetryAgentByState(RETRY_STATE_WAIT_CMD_SEND_AICPU, retryCtx));
    }
    return HCCL_SUCCESS;
}

HcclResult ResumeAgentCheckLink::ProcessEvent(RetryContext* retryCtx)
{
    // 快恢阶段需要获取通信域内所有使用RDMA链路的rank列表, isGetGroupAllRemoteRank为true
    if (!retryCtx->isChangeLinkInfoInit_) {
        CHK_RET(InitChangeLinkInfo(retryCtx, false, true));
        retryCtx->isChangeLinkInfoInit_ = true;
    } else {
        // BatchSendRecv算子增量建链场景
        CHK_RET(InitChangeLinkInfo(retryCtx, true, true));
    }

    CHK_RET(SetTransportStatusForStop(retryCtx));
    RetryState nextState = RETRY_RESUME_STATE_AGENT_CHANGE_LINK;
    HCCL_RUN_INFO("[OpRetry][Agent]OpRetryAgentWaitResume, start to check link");
    // 获取当前主备网口状态，并且回复Server
    CHK_RET(GetLinkPortStatus(retryCtx, retryCtx->linkPortStatus_, true));
    HcclResult ret = IssueLinkPortCheckResult(retryCtx->agentSocket_, retryCtx->linkPortStatus_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[OpRetry][Agent]ResumeAgentCheckLink IssueResponse fail"), ret);
    CHK_RET(CreateOpRetryAgentByState(nextState, retryCtx));
    return HCCL_SUCCESS;
}

HcclResult ResumeAgentChangeLink::ProcessEvent(RetryContext* retryCtx)
{
    HCCL_RUN_INFO(
        "[OpRetry][Agent][Resume]ResumeAgentChangeLink group[%s] rank[%d] start", retryCtx->group_.c_str(),
        retryCtx->rankId_);
    RetryState nextState = RETRY_STATE_RESERVED;
    // 接收到Server下发的切换链路信息和重建transport命令
    CHK_RET(WaitResumeCmdResumeTransport(retryCtx));
    // 重建当前选择的链路并拷贝到device侧，给aicpu下发切换链路命令
    CHK_RET(SetTransportStatusForResume(retryCtx));
    HCCL_INFO(
        "[OpRetry][Agent][Resume]ResumeAgentChangeLink group[%s] rank[%d] SetTransportStatusForResume finish",
        retryCtx->group_.c_str(), retryCtx->rankId_);
    // 等待aicpu切换链路完成
    CHK_RET(SetOpChangeLinkInfo(retryCtx->GetH2dPtr(), KfcCommand::NsChangeLink, retryCtx->localChangeLinkInfo_));
    HCCL_INFO(
        "[OpRetry][Agent][Resume]ResumeAgentChangeLink group[%s] rank[%d] SetOpChangeLinkInfo finish",
        retryCtx->group_.c_str(), retryCtx->rankId_);
    CHK_RET(WaitAndRespLinkChanged(retryCtx, nextState));
    if (nextState != RETRY_STATE_AGENT_RUNNING) {
        HCCL_ERROR("[OpRetry][Agent]ResumeAgentChangeLink WaitAndRespLinkChanged fail, nextState[%s]", nextState);
    }
    CHK_RET(CreateOpRetryAgentByState(nextState, retryCtx));
    return HCCL_SUCCESS;
}

HcclResult ResumeAgentChangeLink::WaitResumeCmdResumeTransport(RetryContext* retryCtx)
{
    ChangeLinkInfo tmpRecvChangeLinkInfo;
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    const u32 timeoutValue = std::max(static_cast<u32>(GetExternalInputHcclLinkTimeOut()), OP_RETRY_SEND_RECV_TIMEOUT)
                             + OP_RETRY_WAIT_AICPU_TIMEOUT;
    const std::chrono::seconds timeout = std::chrono::seconds(timeoutValue);
    // 接收到命令和当前状态不匹配时, 不做处理, 等待下一个命令, 直到命令正确或者超时
    while (true) {
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        CHK_PRT_RET(elapsed > timeout, HCCL_ERROR("[OpRetry][Agent]ResumeAgentChangeLink timeout"), HCCL_E_TIMEOUT);

        HcclResult ret = WaitChangeLink(retryCtx->agentSocket_, tmpRecvChangeLinkInfo);
        if (ret == HCCL_SUCCESS) {
            HCCL_INFO(
                "[OpRetry][Agent][Resume]WaitResumeCmdResumeTransport receive changelinkinfo success, rank[%d], "
                "group[%s]",
                retryCtx->rankId_, retryCtx->group_.c_str());
            UpdateChangeLinkInfo(retryCtx->localChangeLinkInfo_, tmpRecvChangeLinkInfo);
            // agent接收到的changeLinkInfo信息
            std::string changeLinkInfoStr = "agent:";
            for (u32 i = 0; i < retryCtx->localChangeLinkInfo_.remoteRankNum; i++) {
                changeLinkInfoStr
                    += (std::to_string(retryCtx->localChangeLinkInfo_.remoteRankList[i]) + ":"
                        + std::to_string(retryCtx->localChangeLinkInfo_.isUseDefaultPort[i]) + "; ");
            }
            HCCL_RUN_INFO("[OpRetry][Agent][Resume]changeLinkInfoStr:%s", changeLinkInfoStr.c_str());
            break;
        }
    }
    HCCL_INFO(
        "[OpRetry][Agent][Resume]WaitResumeCmdResumeTransport begin to wait command to changelink, rank[%d], group[%s]",
        retryCtx->rankId_, retryCtx->group_.c_str());
    // 轮询等待接收借轨命令
    while (true) {
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        CHK_PRT_RET(elapsed > timeout, HCCL_ERROR("[OpRetry]WaitResumeCmdChangeLink timeout"), HCCL_E_TIMEOUT);
        RetryCommandInfo commandInfo;
        HcclResult ret = WaitCommandWithOpId(retryCtx->agentSocket_, commandInfo);
        if (ret == HCCL_SUCCESS) {
            if (commandInfo.command == RETRY_CMD_RESUME_TRANSPORT) {
                HCCL_RUN_INFO(
                    "[OpRetry][Agent][Resume]WaitResumeCmdChangeLink, recv command[%s], group[%s], rankId[%u]",
                    GetReadableCmd(commandInfo.command), retryCtx->group_.c_str(), retryCtx->rankId_);
                break;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ResumeAgentChangeLink::WaitAndRespLinkChanged(RetryContext* retryCtx, RetryState& nextState)
{
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    const std::chrono::seconds timeout = std::chrono::seconds(OP_RETRY_WAIT_AICPU_TIMEOUT);
    HCCL_INFO(
        "[OpRetry][Agent][Resume]WaitAndRespLinkChanged start, rank[%d], group[%s]", retryCtx->rankId_,
        retryCtx->group_.c_str());
    while (true) {
        KfcExecStatus& opInfo = retryCtx->localRetryInfo_.opInfo;
        CHK_RET(GetOpExecInfo(retryCtx->GetD2hPtr(), opInfo));
        const KfcStatus& aicpuState = opInfo.execStatus.kfcStatus;
        if (aicpuState == KfcStatus::kResumeChanged) {
            nextState = RETRY_STATE_AGENT_RUNNING;
            CHK_RET(SetOpExecCmd(retryCtx->GetH2dPtr(), KfcCommand::kNone));
            HCCL_INFO(
                "[OpRetry][Agent][Resume]WaitAndRespLinkChanged, aicpuState[%d], rank[%d], group[%s]", aicpuState,
                retryCtx->rankId_, retryCtx->group_.c_str());
            break;
        }
        // 超时机制
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        CHK_PRT_BREAK(
            elapsed >= timeout,
            HCCL_ERROR("[OpRetry][Agent][Resume]WaitAndRespLinkChanged timeout, aicpuState[%d]", aicpuState),
            nextState = RETRY_STATE_RESP_RUNNING_ERR);
        // 轮询间隔
        SaluSleep(OP_RETRY_POLL_AICPU_STATE_INTERVAL);
    }
    if (nextState == RETRY_STATE_AGENT_RUNNING) {
        // 回复成功
        retryCtx->localRetryInfo_.retryState = RETRY_STATE_AGENT_RUNNING;
        HCCL_INFO(
            "[OpRetry][Agent][Resume]WaitAndRespLinkChanged success, rank[%d], group[%s]", retryCtx->rankId_,
            retryCtx->group_.c_str());
        retryCtx->localRetryInfo_.opInfo.execStatus.kfcStatus = KfcStatus::kResumeChanged;
        HcclResult ret = IssueResponse(retryCtx->agentSocket_, retryCtx->localRetryInfo_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[OpRetry][Agent]ResumeAgentChangeLink IssueResponse fail"), ret);
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
