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
#include "comm_configer.h"
#include "externalinput_pub.h"
#include "opretry_manager.h"
#include "adapter_pub.h"
#include "rt_external.h"
#include "opretry_agent.h"
#include "opretry_server.h"
#include "opretry_base.h"
#include "snapshot_control.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef enum tagRtClearStep {
    RT_STREAM_STOP = 0,
    RT_STREAM_CLEAR,
} rtClearStep_t;

extern rtError_t rtStreamClear(rtStream_t stm, rtClearStep_t step);
#ifdef __cplusplus
}
#endif // __cplusplus

namespace {
HcclResult StreamClear(HcclRtStream stream, HcclRtStreamClearStep step)
{
    CHK_PTR_NULL(stream);

    aclError ret;
    if (step == HcclRtStreamClearStep::HCCL_STREAM_STOP) {
        ret = rtStreamClear(stream, rtClearStep_t::RT_STREAM_STOP);
        HCCL_INFO("[StreamClear]Call rtStreamClear, ret[%d], param: stream[%p], step[%d]", ret, stream, step);
    } else {
        ret = rtStreamClear(stream, rtClearStep_t::RT_STREAM_CLEAR);
        HCCL_INFO("[StreamClear]Call rtStreamClear, ret[%d], param: stream[%p], step[%d]", ret, stream, step);
    }
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[StreamClear]errNo[0x%016llx]Failed to clear stream. "
            "ret[%d], param: stream[%p], step[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, stream, step),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}
} // namespace

namespace hccl {
HcclResult OpRetryBase::Handle(RetryContext* retryCtx)
{
    CheckSnapshotStatus(retryCtx);
    if (retryCtx->IsPaused()) {
        SaluSleep(OP_RETRY_RUNNING_POLL_INTERVAL);
        return HCCL_SUCCESS;
    }

    if (!retryCtx->IsRootRetryCtx() && retryCtx->isAgentStateWaitResume_
        && retryCtx->GetRetryState() != RETRY_STATE_AGENT_WAIT_RESUME) {
        std::shared_ptr<OpRetryBase> retryPtr = nullptr;
        EXCEPTION_CATCH(retryPtr = std::make_shared<OpRetryAgentWaitResume>(), return HCCL_E_PTR);
        retryCtx->SetRetryState(RETRY_STATE_AGENT_WAIT_RESUME, retryPtr);
        retryCtx->ResetAgentState();
        HCCL_INFO("[OpRetry][Agent]switch to wait resume.");
        return HCCL_SUCCESS;
    }

    if (retryCtx->IsRootRetryCtx() && retryCtx->isServerStateWaitResume_
        && retryCtx->GetRetryState() != RETRY_STATE_SERVER_WAIT_RESUME) {
        std::shared_ptr<OpRetryBase> retryPtr = nullptr;
        EXCEPTION_CATCH(retryPtr = std::make_shared<OpRetryServerWaitResume>(), return HCCL_E_PTR);
        retryCtx->SetRetryState(RETRY_STATE_SERVER_WAIT_RESUME, retryPtr);
        retryCtx->ResetServerState();
        HCCL_INFO("[OpRetry][Server]switch to wait resume.");
        return HCCL_SUCCESS;
    }

    HcclResult ret = ProcessEvent(retryCtx);
    if (ret != HCCL_SUCCESS) {
        CHK_RET(ProcessError(retryCtx));
        retryCtx->isOpRetryQuit = true;
    }
    return ret;
}

/* root-host 交互 */
HcclResult OpRetryBase::IssueResponse(std::shared_ptr<HcclSocket> socket, RetryInfo& retryInfo)
{
    return Send(socket, &retryInfo, sizeof(RetryInfo));
}

// 非阻塞接收, 若已经收到部分数据, 则变为阻塞接收, 直到收到完整数据或超时
HcclResult OpRetryBase::WaitResponse(std::shared_ptr<HcclSocket> socket, RetryInfo& retryInfo)
{
    HcclResult ret = Recv(socket, &retryInfo, sizeof(RetryInfo));
    if (ret == HCCL_SUCCESS) {
        HCCL_DEBUG("[OpRetry]WaitResponse success, cmd[%u]", retryInfo.cmd);
    }
    return ret;
}

HcclResult OpRetryBase::IssueCommand(std::shared_ptr<HcclSocket> socket, RetryCommand command)
{
    HcclResult ret = Send(socket, &command, sizeof(RetryCommand));
    if (ret == HCCL_SUCCESS) {
        HCCL_DEBUG("[OpRetry]IssueCommand success, command[%s]", GetReadableCmd(command));
    }
    return ret;
}

HcclResult OpRetryBase::WaitCommand(std::shared_ptr<HcclSocket> socket, RetryCommand& command)
{
    HcclResult ret = Recv(socket, &command, sizeof(RetryCommand));
    if (ret == HCCL_SUCCESS) {
        HCCL_DEBUG("[OpRetry]WaitCommand success, command[%s]", GetReadableCmd(command));
    }
    return ret;
}

HcclResult OpRetryBase::IssueCommandWithOpId(std::shared_ptr<HcclSocket> socket, RetryCommandInfo& commandInfo)
{
    HcclResult ret = Send(socket, &commandInfo, sizeof(RetryCommandInfo));
    if (ret == HCCL_SUCCESS) {
        HCCL_DEBUG(
            "[OpRetry]IssueCommand success, command[%s], cmd[%u]", GetReadableCmd(commandInfo.command),
            commandInfo.cmd);
    }
    return ret;
}

HcclResult OpRetryBase::WaitCommandWithOpId(std::shared_ptr<HcclSocket> socket, RetryCommandInfo& commandInfo)
{
    HcclResult ret = Recv(socket, &commandInfo, sizeof(commandInfo));
    if (ret == HCCL_SUCCESS) {
        HCCL_DEBUG(
            "[OpRetry]WaitCommand success, command[%s], cmd[%u]", GetReadableCmd(commandInfo.command), commandInfo.cmd);
    }
    return ret;
}

HcclResult OpRetryBase::IssueLinkPortCheckResult(std::shared_ptr<HcclSocket> socket, LinkPortStatus& linkPortStatus)
{
    HcclResult ret = Send(socket, &linkPortStatus, sizeof(LinkPortStatus));
    if (ret == HCCL_SUCCESS) {
        HCCL_DEBUG("[OpRetry]IssueLinkPortCheckResult success");
    }
    return ret;
}

HcclResult OpRetryBase::WaitLinkPortCheckResult(std::shared_ptr<HcclSocket> socket, LinkPortStatus& linkPortStatus)
{
    HcclResult ret = Recv(socket, &linkPortStatus, sizeof(LinkPortStatus));
    if (ret == HCCL_SUCCESS) {
        HCCL_DEBUG("[OpRetry]WaitLinkPortCheckResult success, cmd[%u]", linkPortStatus.cmd);
    }
    return ret;
}

HcclResult OpRetryBase::IssueChangeLink(std::shared_ptr<HcclSocket> socket, ChangeLinkInfo& changeLinkInfo)
{
    HcclResult ret = Send(socket, &changeLinkInfo, sizeof(ChangeLinkInfo));
    if (ret == HCCL_SUCCESS) {
        HCCL_DEBUG("[OpRetry]IssueChangeLink success");
    }
    return ret;
}

HcclResult OpRetryBase::WaitChangeLink(std::shared_ptr<HcclSocket> socket, ChangeLinkInfo& changeLinkInfo)
{
    HcclResult ret = Recv(socket, &changeLinkInfo, sizeof(ChangeLinkInfo));
    if (ret == HCCL_SUCCESS) {
        HCCL_DEBUG("[OpRetry]WaitChangeLink success");
    }
    return ret;
}

/* 校验 */
HcclResult OpRetryBase::CheckRetryInfo(RetryContext& retryCtx)
{
    for (auto rank : retryCtx.needRetryServerRanks_) {
        // 校验opName一致性
        auto& retryInfoStand = retryCtx.serverSockets_[*(retryCtx.needRetryServerRanks_.begin())].retryInfo;
        auto& retryInfo = retryCtx.serverSockets_[rank].retryInfo;
        u32 retryCnt = retryInfo.opInfo.execStatus.retryInfo.retryCount;
        HCCL_RUN_INFO(
            "[OpRetry][Server][CheckRetryInfo]rankId[%u], opName[%s], index[%u], retryCnt[%u], linkState[%d]",
            retryInfo.rankId, retryInfo.opInfo.opId.tag, retryInfo.opInfo.opId.index, retryCnt, retryInfo.linkState);

        CHK_RET(CheckOpName(retryInfo, retryInfoStand));
        // 校验重传次数
        CHK_RET(CheckMaxRetryCnt(retryInfo, retryCtx.group_));
        // 校验链路状态
        CHK_RET(CheckLinkStates(retryInfo));
    }
    return HCCL_SUCCESS;
}

HcclResult OpRetryBase::CheckOpName(const RetryInfo& retryInfo1, const RetryInfo& retryInfo2)
{
    if (retryInfo1.opInfo.opId.isSendRecv == true && retryInfo2.opInfo.opId.isSendRecv == true) {
        const char* tag1 = reinterpret_cast<const char*>(retryInfo1.opInfo.opId.tag);
        const char* tag2 = reinterpret_cast<const char*>(retryInfo2.opInfo.opId.tag);
        const char* tag1Send = reinterpret_cast<const char*>(retryInfo1.opInfo.opId.bsrInfo[HCCL_SEND].bsrTag);
        const char* tag1Recv = reinterpret_cast<const char*>(retryInfo1.opInfo.opId.bsrInfo[HCCL_RECV].bsrTag);
        const char* tag2Send = reinterpret_cast<const char*>(retryInfo2.opInfo.opId.bsrInfo[HCCL_SEND].bsrTag);
        const char* tag2Recv = reinterpret_cast<const char*>(retryInfo2.opInfo.opId.bsrInfo[HCCL_RECV].bsrTag);
        u32 index1 = retryInfo1.opInfo.opId.index;
        u32 index2 = retryInfo2.opInfo.opId.index;
        u32 index1Send = retryInfo1.opInfo.opId.bsrInfo[HCCL_SEND].index;
        u32 index1Recv = retryInfo1.opInfo.opId.bsrInfo[HCCL_RECV].index;
        u32 index2Send = retryInfo2.opInfo.opId.bsrInfo[HCCL_SEND].index;
        u32 index2Recv = retryInfo2.opInfo.opId.bsrInfo[HCCL_RECV].index;
        bool isEqual
            = (((strcmp(tag1Send, tag2Recv) == 0) && (index1Send == index2Recv))
               || ((strcmp(tag1Recv, tag2Send) == 0) && (index1Recv == index2Send))
               || ((strcmp(tag1, tag2) == 0) && (index1 == index2)));
        CHK_PRT_RET(
            isEqual == false,
            HCCL_ERROR(
                "[OpRetry][CheckOpName]hccl aicpu can not retry, opName is inconsistent: "
                "rank[%u] tag1[%s] index1[%u] Stag1[%s] Sindex1[%u], Rtag1[%s] Rindex1[%u], IpInfo1[%s], "
                "rank[%u] tag2[%s] index2[%u] Stag2[%s] Sindex2[%u], Rtag2[%s] Rindex2[%u], IpInfo2[%s]",
                retryInfo1.rankId, tag1, index1, tag1Send, index1Send, tag1Recv, index1Recv, retryInfo1.dfxIpInfo,
                retryInfo2.rankId, tag1, index2, tag2Send, index2Send, tag2Recv, index2Recv, retryInfo2.dfxIpInfo),
            HCCL_E_OPRETRY_FAIL);
        return HCCL_SUCCESS;
    }
    const char* tag1 = reinterpret_cast<const char*>(retryInfo1.opInfo.opId.tag);
    const char* tag2 = reinterpret_cast<const char*>(retryInfo2.opInfo.opId.tag);
    u32 index1 = retryInfo1.opInfo.opId.index;
    u32 index2 = retryInfo2.opInfo.opId.index;
    bool isEqual = (strcmp(tag1, tag2) == 0) && (index1 == index2);
    CHK_PRT_RET(
        isEqual == false,
        HCCL_ERROR(
            "[OpRetry][CheckOpName]hccl aicpu can not retry, opName is inconsistent: "
            "rank[%u] tag1[%s] index1[%u] IpInfo1[%s], rank[%u] tag2[%s] index2[%u], IpInfo2[%s]",
            retryInfo1.rankId, tag1, index1, retryInfo1.dfxIpInfo, retryInfo2.rankId, tag2, index2,
            retryInfo2.dfxIpInfo),
        HCCL_E_OPRETRY_FAIL);
    return HCCL_SUCCESS;
}

HcclResult OpRetryBase::CheckMaxRetryCnt(const RetryInfo& retryInfo, const std::string& identifier)
{
    u32 retryCount = retryInfo.opInfo.execStatus.retryInfo.retryCount;
    u32 retryMaxCnt = CommConfiger::GetInstance().GetCommConfigRetryMaxCnt(identifier);
    if (retryCount >= retryMaxCnt) {
        HCCL_ERROR(
            "[OpRetry][CheckMaxRetryCnt]hccl aicpu can not retry, the retryCnt[%u] of rank[%u] with IpInfo[%s] "
            "exceeds the MaxCnt[%u]",
            retryCount, retryInfo.rankId, retryInfo.dfxIpInfo, retryMaxCnt);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult OpRetryBase::CheckLinkStates(const RetryInfo& retryInfo)
{
    if (retryInfo.linkState == false) {
        HCCL_ERROR(
            "[OpRetry][CheckLinkStates]hccl aicpu can not retry, the linkState[%u] of rank[%u] with IpInfo[%s] "
            "should be %u",
            retryInfo.linkState, retryInfo.rankId, retryInfo.dfxIpInfo, true);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

/* host-device 交互 */
HcclResult OpRetryBase::GetRetryInfo(RetryContext* retryCtx, RetryInfo& retryInfo)
{
    CHK_PTR_NULL(retryCtx);

    retryInfo.rankId = retryCtx->GetRankId();
    retryInfo.retryState = retryCtx->GetRetryState();
    retryInfo.linkState = true;
    CHK_RET(GetOpExecInfo(retryCtx->GetD2hPtr(), retryInfo.opInfo));

    HCCL_DEBUG(
        "[OpRetry][GetRetryInfo]rankId[%u], retryState[%d], linkState[%d]", retryInfo.rankId, retryInfo.retryState,
        retryInfo.linkState);

    KfcExecStatus opInfo = retryInfo.opInfo;
    HCCL_DEBUG(
        "[OpRetry][GetRetryInfo]tag[%s], index[%u], srcRank[%u], detRank[%u], isSendRecv[%d], opExeState[%d], "
        "errorCode[%d], retryCount[%u], streamid[%u]",
        opInfo.opId.tag, opInfo.opId.index, opInfo.opId.srcRank, opInfo.opId.detRank, opInfo.opId.isSendRecv,
        opInfo.execStatus.kfcStatus, opInfo.execStatus.kfcError, opInfo.execStatus.retryInfo.retryCount,
        opInfo.opId.streamId);

    return HCCL_SUCCESS;
}

HcclResult OpRetryBase::GetOpExecInfo(std::shared_ptr<HDCommunicate> hdcPtr, KfcExecStatus& opInfo)
{
    CHK_SMART_PTR_NULL(hdcPtr);
    CHK_RET(hdcPtr->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t*>(&opInfo)));
    HCCL_DEBUG(
        "[OpRetry][GetOpExecInfo]tag[%s], index[%u], srcRank[%u], detRank[%u], isSendRecv[%d], opExeState[%d], "
        "errorCode[%d], retryCount[%u]",
        opInfo.opId.tag, opInfo.opId.index, opInfo.opId.srcRank, opInfo.opId.detRank, opInfo.opId.isSendRecv,
        opInfo.execStatus.kfcStatus, opInfo.execStatus.kfcError, opInfo.execStatus.retryInfo.retryCount);
    return HCCL_SUCCESS;
}

HcclResult OpRetryBase::SetOpExecCmd(std::shared_ptr<HDCommunicate> hdcPtr, KfcCommand opCmd)
{
    HCCL_RUN_INFO("[OpRetry][SetOpExecCmd]set KfcCommand[%d]", opCmd);
    CHK_SMART_PTR_NULL(hdcPtr);
    CHK_RET(hdcPtr->Put(0, sizeof(KfcCommand), reinterpret_cast<uint8_t*>(&opCmd)));
    return HCCL_SUCCESS;
}

HcclResult
OpRetryBase::SetOpExecCmdWithOpId(std::shared_ptr<HDCommunicate> hdcPtr, KfcCommand opCmd, HcclOpIdentifier& opId)
{
    HCCL_RUN_INFO("[OpRetry][SetOpExecCmd]set KfcCommand[%d]", opCmd);
    CHK_SMART_PTR_NULL(hdcPtr);

    HCCL_RUN_INFO(
        "[OpRetry][SetOpExecCmdWithOpId]tag[%s], index[%u], srcRank[%u], detRank[%u], isSendRecv[%d], "
        "streamid[%u]",
        opId.tag, opId.index, opId.srcRank, opId.detRank, opId.isSendRecv, opId.streamId);
    // 发送KfcCommand命令到hdc buffer中
    CHK_RET(hdcPtr->Put(0, sizeof(KfcCommand), reinterpret_cast<uint8_t*>(&opCmd)));
    // 计算targetOp偏移，发送HcclOpIdentifier数据到hdc buffer中
    u32 targetOpStart = sizeof(KfcCommand) + sizeof(BackgroundCommand) + sizeof(HcclComSuspendingFlag);
    CHK_RET(hdcPtr->Put(targetOpStart, sizeof(HcclOpIdentifier), reinterpret_cast<uint8_t*>(&opId)));
    return HCCL_SUCCESS;
}

HcclResult OpRetryBase::SetOpChangeLinkInfo(
    std::shared_ptr<HDCommunicate> hdcPtr, KfcCommand opCmd, ChangeLinkInfo& changeLinkInfo)
{
    CHK_SMART_PTR_NULL(hdcPtr);

    // 发送KfcCommand命令到hdc buffer中
    CHK_RET(hdcPtr->Put(0, sizeof(KfcCommand), reinterpret_cast<uint8_t*>(&opCmd)));
    // 计算changeLinkInfo偏移，发送HcclOpIdentifier数据到hdc buffer中
    u32 changeLinkInfoStart
        = sizeof(KfcCommand) + sizeof(BackgroundCommand) + sizeof(HcclComSuspendingFlag) + sizeof(HcclOpIdentifier);
    CHK_RET(hdcPtr->Put(changeLinkInfoStart, sizeof(ChangeLinkInfo), reinterpret_cast<uint8_t*>(&changeLinkInfo)));
    return HCCL_SUCCESS;
}

HcclResult OpRetryBase::ClearStream(std::shared_ptr<HcclOpStreamRes> opStreamPtr_, HcclRtStreamClearStep clearStep)
{
    HCCL_INFO("[OpRetry][ClearStream]start");
    CHK_SMART_PTR_NULL(opStreamPtr_);
    CHK_PRT_RET(opStreamPtr_->empty(), HCCL_ERROR("[OpRetry][ClearStream]fail, stream is empty"), HCCL_E_PARA);
    for (auto it = opStreamPtr_->begin(); it != opStreamPtr_->end(); it++) {
        const std::string& tag = it->first;
        std::vector<Stream>& streams = it->second;
        HCCL_RUN_INFO("[OpRetry][Agent]ClearStream clearStep:%u, tag:%u", clearStep, tag.c_str());
        for (auto& stream : streams) {
            HCCL_RUN_INFO("[OpRetry][Agent]ClearStream streamId:%u", stream.id());
            HcclResult ret = StreamClear(stream.ptr(), clearStep);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[OpRetry][ClearStream]rtStream clear failed, stm:%p, "
                    "step:%d, ret[%d].",
                    stream.ptr(), clearStep, ret),
                HCCL_E_RUNTIME);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult OpRetryBase::ClearStreamWithOpId(
    std::shared_ptr<HcclOpStreamRes> opStreamPtr_, HcclRtStreamClearStep clearStep, HcclOpIdentifier& opId,
    HcclOpIdentifier& curOpId)
{
    HCCL_INFO("[OpRetry][ClearStream]start");
    CHK_SMART_PTR_NULL(opStreamPtr_);
    CHK_PRT_RET(opStreamPtr_->empty(), HCCL_ERROR("[OpRetry][ClearStream]fail, stream is empty"), HCCL_E_PARA);

    if (curOpId.isSendRecv && curOpId.streamId != ~0u) {
        if (std::string(reinterpret_cast<const char*>(curOpId.tag))
                != std::string(reinterpret_cast<const char*>(opId.tag))
            || curOpId.index != opId.index) {
            HCCL_ERROR(
                "[OpRetry][Agent]ClearStream clearStep tag is inconsistent, curtag[%s],cmdtag[%s], curindex[%u], "
                "cmdindex[%u]",
                curOpId.tag, opId.tag, curOpId.index, opId.index);
        }
        for (auto it = opStreamPtr_->begin(); it != opStreamPtr_->end(); it++) {
            const std::string& tag = it->first;
            std::vector<Stream>& streams = it->second;
            HCCL_RUN_INFO("[OpRetry][Agent]ClearStream clearStep:%u, tag:%s", clearStep, tag.c_str());
            for (auto& stream : streams) {
                if (static_cast<u32>(stream.id()) == curOpId.streamId) {
                    HCCL_RUN_INFO("[OpRetry][Agent]ClearStream streamId:%u", curOpId.streamId);
                    HcclResult ret = StreamClear(stream.ptr(), clearStep);
                    CHK_PRT_RET(
                        ret != HCCL_SUCCESS,
                        HCCL_ERROR(
                            "[OpRetry][ClearStream]rtStream clear failed, stm:%p, "
                            "step:%d, ret[%d].",
                            stream.ptr(), clearStep, ret),
                        HCCL_E_RUNTIME);
                }
            }
        }
    } else {
        for (auto it = opStreamPtr_->begin(); it != opStreamPtr_->end(); it++) {
            const std::string& tag = it->first;
            std::vector<Stream>& streams = it->second;
            HCCL_RUN_INFO("[OpRetry][Agent]ClearStream clearStep:%u, tag:%u", clearStep, tag.c_str());
            for (auto& stream : streams) {
                HCCL_RUN_INFO("[OpRetry][Agent]ClearStream streamId:%u", stream.id());
                HcclResult ret = StreamClear(stream.ptr(), clearStep);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[OpRetry][ClearStream]rtStream clear failed, stm:%p, "
                        "step:%d, ret[%d].",
                        stream.ptr(), clearStep, ret),
                    HCCL_E_RUNTIME);
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult OpRetryBase::ResetNotify(RetryContext* retryCtx)
{
    CHK_PTR_NULL(retryCtx);

    auto remoteRank = retryCtx->localRetryInfo_.rankId == retryCtx->localRetryInfo_.opInfo.opId.detRank ?
                          retryCtx->localRetryInfo_.opInfo.opId.srcRank :
                          retryCtx->localRetryInfo_.opInfo.opId.detRank;

    // send/recv场景下只对对端的notify重置，其他场景下需要重置全部notify
    return retryCtx->notifyResetCallback_(
        retryCtx->localRetryInfo_.opInfo.opId.isSendRecv, static_cast<s64>(remoteRank));
}

HcclResult OpRetryBase::SetTransportStatusForStop(RetryContext* retryCtx)
{
    CHK_PTR_NULL(retryCtx);
    // 用于表示当前rank与对端是否走借轨
    std::map<u32, bool> isChangeLinkMap;
    bool isChangeLinkFlag = false;
    if (retryCtx->isRecivedCmdToCheckLink) {
        return retryCtx->setTransportReseumeStatusCallback_(
            retryCtx->lastLinkPortStatus_, isChangeLinkMap, isChangeLinkFlag, true);
    }
    // stop阶段对当前正在使用的link执行，使用lastLinkPortStatus_表示当前正在使用的网口情况，默认为true，使用主网口
    return retryCtx->setTransportStatusCallback_(
        retryCtx->localRetryInfo_.opInfo.opId, true, retryCtx->lastLinkPortStatus_, isChangeLinkMap, isChangeLinkFlag);
}

HcclResult OpRetryBase::SetTransportStatusForResume(RetryContext* retryCtx)
{
    CHK_PTR_NULL(retryCtx);
    // 用于表示当前rank与对端是否走借轨
    std::map<u32, bool> isChangeLinkMap;
    bool isChangeLinkFlag = false;
    // resume阶段
    std::map<u32, bool> remoteRankPortMap;
    for (u32 i = 0; i < retryCtx->localChangeLinkInfo_.remoteRankNum; i++) {
        auto remoteRank = retryCtx->localChangeLinkInfo_.remoteRankList[i];
        auto isRemoteUseDefaultPort = retryCtx->localChangeLinkInfo_.isUseDefaultPort[i];
        // remoteRankPortMap用于表示链路切换后的网口使用情况
        remoteRankPortMap.insert({remoteRank, isRemoteUseDefaultPort});
        // 和上一次链路使用情况对比判断当前是否为借轨场景，同时更新lastLinkPortStatus_
        if (retryCtx->lastLinkPortStatus_.find(remoteRank) == retryCtx->lastLinkPortStatus_.end()) {
            // 若remoteRank不在lastLinkPortStatus_中，默认上一次使用的是主链路
            isChangeLinkMap.insert({remoteRank, !isRemoteUseDefaultPort});
            isChangeLinkFlag = isRemoteUseDefaultPort ? isChangeLinkFlag : true;
            retryCtx->lastLinkPortStatus_.insert({remoteRank, isRemoteUseDefaultPort});
        } else if (isRemoteUseDefaultPort == retryCtx->lastLinkPortStatus_[remoteRank]) {
            // 若本次和上一次使用同一个port，则属于原地重执行场景
            isChangeLinkMap.insert({remoteRank, false});
        } else {
            // 若本次和上一次使用不同port，则属于借轨场景，并更新到lastLinkPortStatus_中
            isChangeLinkMap.insert({remoteRank, true});
            isChangeLinkFlag = true;
            retryCtx->lastLinkPortStatus_[remoteRank] = isRemoteUseDefaultPort;
        }
    }

    std::string isChangeLinkMapStr = "";
    for (auto changeIt : isChangeLinkMap) {
        isChangeLinkMapStr += (std::to_string(changeIt.first) + ":" + std::to_string(changeIt.second) + ";");
    }
    HCCL_RUN_INFO("[OpRetry][Agent]isChangeLinkFlag[%d]:[%s]", isChangeLinkFlag, isChangeLinkMapStr.c_str());

    retryCtx->localRetryInfo_.isChangeLinkFlag = isChangeLinkFlag;      // 向server上报
    retryCtx->localChangeLinkInfo_.isChangeLinkFlag = isChangeLinkFlag; // 向aicpu下发

    if (retryCtx->isRecivedCmdToCheckLink) {
        retryCtx->isRecivedCmdToCheckLink = false;
        return retryCtx->setTransportReseumeStatusCallback_(
            remoteRankPortMap, isChangeLinkMap, isChangeLinkFlag, false);
    }
    return retryCtx->setTransportStatusCallback_(
        retryCtx->localRetryInfo_.opInfo.opId, false, remoteRankPortMap, isChangeLinkMap, isChangeLinkFlag);
}

HcclResult OpRetryBase::Send(std::shared_ptr<HcclSocket> socket, void* data, u64 size)
{
    HCCL_DEBUG("[OpRetry][Send]start, para: data[%p], size[%llu Byte]", data, size);
    const auto start = std::chrono::steady_clock::now();
    const u32 timeoutValue = std::max(static_cast<u32>(GetExternalInputHcclLinkTimeOut()), OP_RETRY_SEND_RECV_TIMEOUT)
                             + OP_RETRY_WAIT_AICPU_TIMEOUT;
    const std::chrono::seconds timeout = std::chrono::seconds(timeoutValue);

    u64 restSize = size; // 待发送数据长度
    while (true) {
        CHK_PRT_RET(!enableSendRecv, HCCL_DEBUG("[OpRetry][Send]Exit send."), HCCL_SUCCESS);
        u64 sendDis = size - restSize;
        void* dataPtr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(data) + sendDis);
        /* 获取当前时间，如果耗时超过timeout，则返回错误 */
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
        CHK_PRT_RET(
            elapsed > timeout,
            HCCL_WARNING(
                "Send fail, Wait timeout for sockets send, dataPtr[%p], restSize[%llu Byte]", dataPtr, restSize),
            HCCL_E_TIMEOUT);

        u64 compSize = 0; // 本次发送数据长度
        HcclResult ret = socket->ISend(dataPtr, restSize, compSize);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_WARNING("Send fail, dataPtr[%p], restSize[%llu Byte], compSize[%llu]", dataPtr, restSize, compSize),
            ret);

        if (restSize == compSize) { // 数据发送完成
            HCCL_DEBUG("OpRetryBase send end");
            return HCCL_SUCCESS;
        } else if (restSize < compSize) {
            HCCL_ERROR("Send fail, restSize[%llu Byte], compSize[%llu Byte]", restSize, compSize);
            return HCCL_E_TCP_TRANSFER;
        }
        restSize -= compSize;
    }
    return HCCL_SUCCESS;
}

HcclResult OpRetryBase::Recv(std::shared_ptr<HcclSocket> socket, void* data, u64 totalSize)
{
    const auto start = std::chrono::steady_clock::now();
    const u32 timeoutValue = std::max(static_cast<u32>(GetExternalInputHcclLinkTimeOut()), OP_RETRY_SEND_RECV_TIMEOUT)
                             + OP_RETRY_WAIT_AICPU_TIMEOUT;
    const std::chrono::seconds timeout = std::chrono::seconds(timeoutValue);

    u64 recvSize = 0;
    while (true) {
        CHK_PRT_RET(!enableSendRecv, HCCL_DEBUG("[OpRetry][Recv]Exit recv."), HCCL_SUCCESS);
        // 超时判断
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
        CHK_PRT_RET(
            elapsed > timeout,
            HCCL_ERROR(
                "[OpRetry]Recv timeout, data[%p], recvSize[%llu Byte], totalSize[%llu Byte]", data, recvSize,
                totalSize),
            HCCL_E_TIMEOUT);

        u64 compSize = 0;                     // 本次接收到的长度
        u64 resetSize = totalSize - recvSize; // 待接收长度
        void* recvPtr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(data) + recvSize);
        HcclResult ret = socket->IRecv(recvPtr, resetSize, compSize);
        CHK_PRT_RET(ret != HCCL_SUCCESS, , ret);

        recvSize += compSize;
        if (recvSize == 0) { // 未收到数据
            return HCCL_E_AGAIN;
        } else if (recvSize < totalSize) { // 数据未接收完, 继续等待该对端
            HCCL_DEBUG("[OpRetry]Recv not complete, recvSize[%llu Byte], totalSize[%llu Byte]", recvSize, totalSize);
            SaluSleep(OP_RETRY_SEND_RECV_INTERVAL);
            continue;
        } else { // 数据接收完成
            return HCCL_SUCCESS;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult OpRetryBase::InitChangeLinkInfo(RetryContext* retryCtx, bool incre, bool isGetGroupAllRemoteRank)
{
    std::string newTag = std::string(reinterpret_cast<const char*>(retryCtx->localRetryInfo_.opInfo.opId.newTag));
    std::vector<u32> rankList;
    auto ret = OpRetryManager::GetLinkInfoByIdentifier(
        retryCtx->deviceLogicId_, retryCtx->group_, newTag, rankList, isGetGroupAllRemoteRank);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[OpRetry][Agent][InitChangeLinkInfo] GetLinkInfoByIdentifier failed: deviceLogicId[%d], "
            "identify[%s], tag[%s], isGetGroupAllRemoteRank[%d]",
            retryCtx->deviceLogicId_, retryCtx->group_.c_str(), newTag.c_str(), isGetGroupAllRemoteRank),
        ret);
    if (retryCtx->localRetryInfo_.opInfo.opId.isSendRecv) {
        // send/recv场景下仅需校验对端
        auto remoteRank = retryCtx->localRetryInfo_.rankId == retryCtx->localRetryInfo_.opInfo.opId.detRank ?
                              retryCtx->localRetryInfo_.opInfo.opId.srcRank :
                              retryCtx->localRetryInfo_.opInfo.opId.detRank;
        bool isFind = std::count(rankList.begin(), rankList.end(), remoteRank) > 0;
        rankList.clear();
        if (isFind) {
            rankList.push_back(remoteRank);
        }
    }

    if (incre) {
        // 增量场景
        for (u32 remoteRank : rankList) {
            // 若对端不在已有的链路切换列表中，则加入，且默认为true
            if (retryCtx->lastLinkPortStatus_.find(remoteRank) == retryCtx->lastLinkPortStatus_.end()) {
                u32 index = retryCtx->localChangeLinkInfo_.remoteRankNum;
                retryCtx->localChangeLinkInfo_.remoteRankNum++;
                retryCtx->localChangeLinkInfo_.remoteRankList[index] = remoteRank;
                retryCtx->localChangeLinkInfo_.isUseDefaultPort[index] = true;

                retryCtx->lastLinkPortStatus_.insert({remoteRank, true});

                HCCL_RUN_INFO("[OpRetry][Agent]init changeLinkInfoStr add remoteRank[%u]", remoteRank);
            }
        }
    } else {
        // 首次初始化场景
        retryCtx->localChangeLinkInfo_.remoteRankNum = rankList.size();
        std::copy(rankList.begin(), rankList.end(), retryCtx->localChangeLinkInfo_.remoteRankList);
        CHK_SAFETY_FUNC_RET(
            memset_s(retryCtx->localChangeLinkInfo_.isUseDefaultPort, rankList.size(), true, rankList.size()));

        // agent的初始化changeLinkInfo信息
        std::string changeLinkInfoStr = "";
        for (u32 i = 0; i < retryCtx->localChangeLinkInfo_.remoteRankNum; i++) {
            changeLinkInfoStr
                += (std::to_string(retryCtx->localChangeLinkInfo_.remoteRankList[i]) + ":"
                    + std::to_string(retryCtx->localChangeLinkInfo_.isUseDefaultPort[i]) + "; ");
            retryCtx->lastLinkPortStatus_.insert({retryCtx->localChangeLinkInfo_.remoteRankList[i], true});
        }
        HCCL_RUN_INFO("[OpRetry][Agent]init changeLinkInfoStr:%s", changeLinkInfoStr.c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult OpRetryBase::GetLinkPortStatus(
    RetryContext* retryCtx, LinkPortStatus& linkPortStatus, bool isGetGroupAllRemoteRank) const
{
    std::string newTag = std::string(reinterpret_cast<const char*>(retryCtx->localRetryInfo_.opInfo.opId.newTag));
    HCCL_RUN_INFO(
        "[OpRetry][Agent]begin to GetLinkPortStatus from: deviceLogicId[%d], identifier[%s] tag[%s]",
        retryCtx->deviceLogicId_, retryCtx->group_.c_str(), newTag.c_str());

    std::vector<u32> rankList;
    auto ret = OpRetryManager::GetLinkInfoByIdentifier(
        retryCtx->deviceLogicId_, retryCtx->group_, newTag, rankList, isGetGroupAllRemoteRank);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[OpRetry][Agent][GetLinkPortStatus] GetLinkInfoByIdentifier failed: deviceLogicId[%d], "
            "identify[%s], tag[%s], isGetGroupAllRemoteRank[%d]",
            retryCtx->deviceLogicId_, retryCtx->group_.c_str(), newTag.c_str(), isGetGroupAllRemoteRank),
        ret);
    if (retryCtx->localRetryInfo_.opInfo.opId.isSendRecv) {
        // send/recv场景下仅需校验对端
        auto remoteRank = retryCtx->localRetryInfo_.rankId == retryCtx->localRetryInfo_.opInfo.opId.detRank ?
                              retryCtx->localRetryInfo_.opInfo.opId.srcRank :
                              retryCtx->localRetryInfo_.opInfo.opId.detRank;
        bool isFind = std::count(rankList.begin(), rankList.end(), remoteRank) > 0;
        rankList.clear();
        if (isFind) {
            rankList.push_back(remoteRank);
        }
    }
    std::copy(rankList.begin(), rankList.end(), linkPortStatus.rankList);
    linkPortStatus.rankSize = rankList.size();

    if (rankList.size() == 0) {
        // 若rankList为空，则说明当前无roce连接，无需获取主备网口link状态
        HCCL_RUN_INFO(
            "[OpRetry][Agent]deviceLogicId[%d], rankSize[%d], not need to get link port, identifier[%s], tag[%s]",
            retryCtx->deviceLogicId_, linkPortStatus.rankSize, retryCtx->group_.c_str(), newTag.c_str());
        return HCCL_SUCCESS;
    }

    if (retryCtx->isUseDefaultPort_) {
        CHK_RET(HcclNetDevGetPortStatus(retryCtx->netDevCtx_, linkPortStatus.defaultPort));
        if (!linkPortStatus.defaultPort) {
            HCCL_RUN_INFO("[OpRetry][Agent]defaultPort is down, set isUseDefaultPort_ to false");
            retryCtx->isUseDefaultPort_ = false;
        }
    } else {
        // 发生借轨后不检测主网口，暂不支持回切
        linkPortStatus.defaultPort = false;
        HCCL_RUN_INFO("[OpRetry][Agent]defaultPort is not enable after last check, set defaultPort to false");
    }

    if (retryCtx->isEnableBackupLink_) {
        // 使能借轨场景下才需要获取backupPort状态
        CHK_RET(HcclNetDevGetPortStatus(retryCtx->backUpNetDevCtx_, linkPortStatus.backupPort));
    } else {
        // 默认场景下backupPort状态为false
        linkPortStatus.backupPort = false;
        HCCL_RUN_INFO("[OpRetry][Agent]backUpLink is not enable, set backupPort to false");
    }

    HCCL_RUN_INFO(
        "[OpRetry][Agent]GetLinkPortStatus success: deviceLogicId[%d], rankSize[%d], identifier[%s], tag[%s]",
        retryCtx->deviceLogicId_, linkPortStatus.rankSize, retryCtx->group_.c_str(), newTag.c_str());
    return HCCL_SUCCESS;
}
HcclResult OpRetryBase::SetBsrOpId(RetryContext* retryCtx, HcclSendRecvType type)
{
    auto& opId = retryCtx->localRetryInfo_.opInfo.opId;
    auto& bsrOpId = (HcclSendRecvType::HCCL_SEND == type) ? retryCtx->RemainSendOpId_ : retryCtx->RemainRecvOpId_;

    // 重执行需要的信息
    bsrOpId.index = opId.bsrInfo[type].index;
    CHK_SAFETY_FUNC_RET(memset_s(bsrOpId.tag, sizeof(bsrOpId.tag), 0, sizeof(bsrOpId.tag)));
    CHK_SAFETY_FUNC_RET(
        memcpy_s(bsrOpId.tag, sizeof(bsrOpId.tag), opId.bsrInfo[type].bsrTag, sizeof(opId.bsrInfo[type].bsrTag)));
    bsrOpId.srcRank = opId.bsrInfo[type].srcRank;
    bsrOpId.detRank = opId.bsrInfo[type].detRank;
    bsrOpId.streamId = opId.bsrInfo[type].streamId;
    bsrOpId.bsrInfo[type].tpQpn = opId.bsrInfo[type].tpQpn;
    bsrOpId.isSendRecv = true;
    bsrOpId.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    return HCCL_SUCCESS;
}

HcclResult OpRetryBase::GetBsrOpId(RetryContext* retryCtx, HcclSendRecvType type)
{
    auto& opId = retryCtx->localRetryInfo_.opInfo.opId;
    auto& bsrOpId = (HcclSendRecvType::HCCL_SEND == type) ? retryCtx->RemainSendOpId_ : retryCtx->RemainRecvOpId_;
    // 重执行需要的信息
    opId.index = bsrOpId.index;
    CHK_SAFETY_FUNC_RET(memset_s(opId.tag, sizeof(opId.tag), 0, sizeof(opId.tag)));
    CHK_SAFETY_FUNC_RET(memcpy_s(opId.tag, sizeof(opId.tag), bsrOpId.tag, sizeof(bsrOpId.tag)));
    opId.srcRank = bsrOpId.srcRank;
    opId.detRank = bsrOpId.detRank;
    opId.streamId = bsrOpId.streamId;
    opId.isSendRecv = true;
    opId.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    return HCCL_SUCCESS;
}

HcclResult OpRetryBase::IssueActiveSwitchInfo(std::shared_ptr<HcclSocket> socket, ActiveSwitchInfo& switchInfo)
{
    HcclResult ret = Send(socket, &switchInfo, sizeof(ActiveSwitchInfo));
    if (ret == HCCL_SUCCESS) {
        HCCL_INFO(
            "[SwitchNic][send] send success fin[%u], switchRankNm[%u], remoteRankNum[%u]",
            switchInfo.refreshTransportFin, switchInfo.switchRankNum, switchInfo.remoteRankNum);
    } else {
        HCCL_ERROR("[SwitchNic] send active switch info fail.");
    }
    return ret;
}

HcclResult OpRetryBase::WaitActiveSwitchInfo(std::shared_ptr<HcclSocket> socket, ActiveSwitchInfo& switchInfo)
{
    HcclResult ret = Recv(socket, &switchInfo, sizeof(ActiveSwitchInfo));
    if (ret == HCCL_SUCCESS) {
        HCCL_INFO(
            "[SwitchNic] recv success fin[%u], switchRankNm[%u], remoteRankNum[%u], cmd[%u]",
            switchInfo.refreshTransportFin, switchInfo.switchRankNum, switchInfo.remoteRankNum, switchInfo.cmd);
    }
    return ret;
}

HcclResult
OpRetryBase::RecvActiveSwitchInfo(std::shared_ptr<HcclSocket> socket, const u32 rankId, ActiveSwitchInfo& switchInfo)
{
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut() * ACTIVE_SWITCH_TIMES);
    HcclResult ret = HCCL_SUCCESS;
    while (true) {
        // 判断是否超时
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        CHK_PRT_RET(
            elapsed > timeout,
            HCCL_ERROR(
                "[SwitchNic][Server] timeout in recv agent ActiveSwitchInfo, waittime[%u s>%u s]", elapsed, timeout),
            HCCL_E_TIMEOUT);
        ret = WaitActiveSwitchInfo(socket, switchInfo);
        if (ret == HCCL_SUCCESS) {
            HCCL_INFO("[SwitchNic][Server] recv ActiveSwitchInfo from rank[%u]", rankId);
            break;
        } else if (ret == HCCL_E_AGAIN) {
            // 未收到数据，发送一个保活数据给agent
            RetryCommand command = RETRY_CMD_RUNNING;
            CHK_RET(IssueCommand(socket, command));
            HCCL_DEBUG("[SwitchNic][Server] send keeping active info to rank[%u]", rankId);
        } else {
            HCCL_ERROR("[SwitchNic][Server] failed to recv ActiveSwitchInfo, rank[%u], ret[%u]", rankId, ret);
            break;
        }
        SaluSleep(OP_RETRY_POLL_AICPU_STATE_INTERVAL);
    }
    return ret;
}

HcclResult OpRetryBase::GetSwitchRanks(RetryContext* retryCtx, bool& needCheckDefaultNic, bool& needCheckBackupNic)
{
    CHK_PTR_NULL(retryCtx);
    return retryCtx->getSwitchRanksCallback_(
        retryCtx->switchInfo_.switchRankList, retryCtx->switchInfo_.switchUseBackup,
        retryCtx->switchInfo_.switchRankNum, retryCtx->switchInfo_.remoteRankNicStatus,
        retryCtx->switchInfo_.remoteRankNum, needCheckDefaultNic, needCheckBackupNic);
}

void OpRetryBase::SetEnableSendRecv(bool enable) { enableSendRecv = enable; }

void OpRetryBase::CheckSnapshotStatus(RetryContext* retryCtx) const
{
    auto snapshotStatus = SnapshotControl::GetInstance(retryCtx->deviceLogicId_).GetStatus();
    if (retryCtx->isPaused_ && snapshotStatus == SnapshotStatus::POST_SNAPSHOT) {
        retryCtx->isPaused_ = false;
        HCCL_RUN_INFO(
            "[OpRetryBase][CheckSnapshotStatus] detect snapshot post-processing, "
            "opretry is resumed, curState[%s], deviceLogicId[%d].",
            retryCtx->GetReadableCtxState(), retryCtx->deviceLogicId_);
    } else if (!retryCtx->isPaused_ && snapshotStatus == SnapshotStatus::PRE_SNAPSHOT) {
        retryCtx->isPaused_ = true;
        HCCL_RUN_INFO(
            "[OpRetryBase][CheckSnapshotStatus] detect snapshot pre-processing, "
            "opretry is paused, curState[%s], deviceLogicId[%d].",
            retryCtx->GetReadableCtxState(), retryCtx->deviceLogicId_);
    }
}
} // namespace hccl
