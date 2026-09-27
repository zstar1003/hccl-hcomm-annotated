/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RETRY_BASE_H
#define HCCL_RETRY_BASE_H

#include <memory>
#include "hccl_socket.h"
#include "notify_pool.h"
#include "hccl_op_retry_pub.h"
#include "hdc_pub.h"
#include "exception_handler.h"
#include "hccl_common.h"

namespace hccl {
constexpr u32 OP_RETRY_MAX_CNT = 3;
constexpr u32 OP_RETRY_WAIT_AICPU_TIMEOUT = 5;            // 等待Aicpu的时长, 单位s
constexpr u32 OP_RETRY_WAIT_AGENT_AICPU_TIMEOUT = 10;     // 等待Agent+Aicpu的时长, 单位s
constexpr u32 OP_RETRY_POLL_AICPU_ERROR_INTERVAL = 1;     // 正常状态轮询Aicpu错误码的间隔, 单位s
constexpr u32 OP_RETRY_POLL_RDMA_ERROR_INTERVAL = 1;      // 正常状态轮询RDMA错误码的间隔, 单位s
constexpr u32 OP_RETRY_POLL_AICPU_STATE_INTERVAL = 10000; // 重执行状态轮询Aicpu状态的间隔, 单位us
constexpr u32 OP_RETRY_SEND_RECV_TIMEOUT = 200;           // 发送和接收的超时时间, 单位s
constexpr u32 OP_RETRY_SEND_RECV_INTERVAL = 10000;        // 发送和接收的间隔时间, 单位us
constexpr u32 OP_RETRY_KEEP_INTERVAL = 1;                 // 保活时间间隔, 单位s
constexpr u32 OP_RETRY_RUNNING_POLL_INTERVAL = 100000;    // 重执行状态轮询状态的间隔, 单位us
constexpr u32 TIME_MS_TO_US = 1000;

// 重执行初始化需要用到的参数
struct OpRetryAgentParam {
    std::string group;
    std::shared_ptr<HcclSocket> agentConnection;
    std::shared_ptr<HDCommunicate> h2dPtr;
    std::shared_ptr<HDCommunicate> d2hPtr;
    std::shared_ptr<HcclOpStreamRes> opStreamPtr;
    OpRetryResetNotifyCallback notifyResetCallback;
    OpRetrySetTransportStatusCallback setTransportStatusCallback;
    OpRetryGetSwitchRanksCallback getSwitchRanksCallback;
    OpRetrySetTransportResumeStatusCallBack setTransportResumeStatusCallback;
    bool isEnableBackupLink;
    bool isEnableSdmaRetry;
    OpRetryAgentInfo agentInfo;
};

struct LinkPortStatus {
    u32 cmd = LINK_PORT_STATUS_CMD;
    bool defaultPort = false;
    bool backupPort = false;
    u32 rankSize = 0;
    u32 rankList[AICPU_MAX_RANK_NUM] = {};
};

struct ActiveSwitchInfo {
    u32 cmd = ACTIVE_SWITCH_INFO_CMD;
    u32 switchRankNum;
    u32 remoteRankNum;
    bool refreshTransportFin = false;
    bool defaultPortStatus = false;
    bool backupPortStatus = false;
    bool localPortsCheckRet = false;
    u32 switchRankList[AICPU_MAX_RANK_NUM] = {};
    bool switchUseBackup[AICPU_MAX_RANK_NUM] = {};
    u8 remoteRankNicStatus[AICPU_MAX_RANK_NUM] = {};
};

using HcclAgentRetryInfo = struct HcclAgentRetryInfoDef {
    std::shared_ptr<HcclSocket> socket{nullptr};
    RetryInfo retryInfo;
    ChangeLinkInfo changeLinkInfo;
    LinkPortStatus linkPortStatus;
    ActiveSwitchInfo switchInfo;
};

inline const char* GetReadableState(RetryState retryState)
{
    auto it = RETRY_STATE_STR_MAP.find(retryState);
    return (it != RETRY_STATE_STR_MAP.end()) ? it->second.c_str() : "unknown state";
}

inline const char* GetReadableCmd(RetryCommand retryCommand)
{
    auto it = RETRY_COMMAND_STR_MAP.find(retryCommand);
    return (it != RETRY_COMMAND_STR_MAP.end()) ? it->second.c_str() : "unknown cmd";
}

class RetryContext;

// 状态基类
class OpRetryBase {
public:
    virtual HcclResult Handle(RetryContext* retryCtx);
    virtual HcclResult ProcessEvent(RetryContext* retryCtx) = 0;
    virtual HcclResult ProcessError(RetryContext* retryCtx) = 0;

    OpRetryBase() {};
    virtual ~OpRetryBase() {};

    // 设置是否直接退出send/recv循环状态，规避长时间阻塞，无法切换状态
    void SetEnableSendRecv(bool enable);

protected:
    /* server-agent 交互 */
    HcclResult IssueResponse(std::shared_ptr<HcclSocket> socket, RetryInfo& retryInfo); // agent向server发送数据
    HcclResult WaitResponse(std::shared_ptr<HcclSocket> socket, RetryInfo& retryInfo);  // server等待agent回复

    HcclResult IssueCommand(std::shared_ptr<HcclSocket> socket, RetryCommand command); // server向agent发送命令
    HcclResult WaitCommand(std::shared_ptr<HcclSocket> socket, RetryCommand& command); // agent轮询命令

    // server向agent发送命令,携带opid
    HcclResult IssueCommandWithOpId(std::shared_ptr<HcclSocket> socket, RetryCommandInfo& commandInfo);
    // agent轮询命令,携带opid
    HcclResult WaitCommandWithOpId(std::shared_ptr<HcclSocket> socket, RetryCommandInfo& commandInfo);

    // server向agent发送借轨信息
    HcclResult IssueChangeLink(std::shared_ptr<HcclSocket> socket, ChangeLinkInfo& changeLinkInfo);
    // agent轮询借轨信息
    HcclResult WaitChangeLink(std::shared_ptr<HcclSocket> socket, ChangeLinkInfo& changeLinkInfo);
    // agent向server发送当前网口情况
    HcclResult IssueLinkPortCheckResult(std::shared_ptr<HcclSocket> socket, LinkPortStatus& linkPortStatus);
    // server接收当前网口情况
    HcclResult WaitLinkPortCheckResult(std::shared_ptr<HcclSocket> socket, LinkPortStatus& linkPortStatus);
    // agent向device发送借轨信息
    HcclResult
    SetOpChangeLinkInfo(std::shared_ptr<HDCommunicate> hdcPtr, KfcCommand opCmd, ChangeLinkInfo& changeLinkInfo);
    // agent向server发送主动借轨信息
    HcclResult IssueActiveSwitchInfo(std::shared_ptr<HcclSocket> socket, ActiveSwitchInfo& switchInfo);
    // server收到主动借轨信息
    HcclResult WaitActiveSwitchInfo(std::shared_ptr<HcclSocket> socket, ActiveSwitchInfo& switchInfo);
    // server处理收到主动借轨信息函数
    HcclResult RecvActiveSwitchInfo(std::shared_ptr<HcclSocket> socket, const u32 rankId, ActiveSwitchInfo& switchInfo);

    // 获取SwitchRanks等信息
    HcclResult GetSwitchRanks(RetryContext* retryCtx, bool& needCheckDefaultNic, bool& needCheckBackupNic);

    /* 校验 */
    HcclResult CheckRetryInfo(RetryContext& retryCtx); // 校验收到的N个RetryInfo
    HcclResult GetRetryInfo(RetryContext* retryCtx, RetryInfo& retryInfo);

    /* agent-device 交互 */
    HcclResult GetOpExecInfo(std::shared_ptr<HDCommunicate> hdcPtr, KfcExecStatus& opInfo);
    HcclResult SetOpExecCmd(std::shared_ptr<HDCommunicate> hdcPtr, KfcCommand opCmd);
    HcclResult ClearStream(std::shared_ptr<HcclOpStreamRes> opStreamPtr_, HcclRtStreamClearStep clearStep);
    HcclResult SetOpExecCmdWithOpId(std::shared_ptr<HDCommunicate> hdcPtr, KfcCommand opCmd, HcclOpIdentifier& opId);
    HcclResult ClearStreamWithOpId(
        std::shared_ptr<HcclOpStreamRes> opStreamPtr_, HcclRtStreamClearStep clearStep, HcclOpIdentifier& opId,
        HcclOpIdentifier& curOpId);
    HcclResult ResetNotify(RetryContext* retryCtx);
    HcclResult SetTransportStatusForStop(RetryContext* retryCtx);
    HcclResult SetTransportStatusForResume(RetryContext* retryCtx);
    HcclResult GetLinkPortStatus(
        RetryContext* retryCtx, LinkPortStatus& linkPortStatus, bool isGetGroupAllRemoteRank = false) const;
    HcclResult InitChangeLinkInfo(RetryContext* retryCtx, bool incre = false, bool isGetGroupAllRemoteRank = false);
    /*获取batchsendrecv rdma重执行时的故障信息*/
    HcclResult SetBsrOpId(RetryContext* retryCtx, HcclSendRecvType type);
    HcclResult GetBsrOpId(RetryContext* retryCtx, HcclSendRecvType type);

private:
    // 阻塞式发送 && 非阻塞式接收, 接口内部不报错, 返回值在上层判断并打印日志, 避免未进入重执行时出现ERROR日志
    HcclResult Send(std::shared_ptr<HcclSocket> socket, void* data, u64 size);
    HcclResult Recv(std::shared_ptr<HcclSocket> socket, void* data, u64 size);

    HcclResult CheckOpName(const RetryInfo& opInfo1, const RetryInfo& opInfo2); // 校验算子一致
    HcclResult
    CheckMaxRetryCnt(const RetryInfo& retryInfo, const std::string& identifier = HCCL_WORLD_GROUP); // 校验重执行次数
    HcclResult CheckLinkStates(const RetryInfo& retryInfo);                                         // 校验link状态
    void CheckSnapshotStatus(RetryContext* retryCtx) const;
    bool enableSendRecv = true;
};

class RetryContext {
public:
    // agent状态机初始化
    RetryContext(OpRetryAgentParam& param, std::shared_ptr<OpRetryBase> retryBase)
    {
        group_ = param.group;
        agentSocket_ = param.agentConnection;
        h2dPtr_ = param.h2dPtr;
        d2hPtr_ = param.d2hPtr;
        opStreamPtr_ = param.opStreamPtr;
        notifyResetCallback_ = param.notifyResetCallback;
        setTransportStatusCallback_ = param.setTransportStatusCallback;
        getSwitchRanksCallback_ = param.getSwitchRanksCallback;
        setTransportReseumeStatusCallback_ = param.setTransportResumeStatusCallback;
        isEnableBackupLink_ = param.isEnableBackupLink;
        isEnableSdmaRetry_ = param.isEnableSdmaRetry;
        retryBase_ = retryBase;
        isRootRetryCtx_ = false;

        rankId_ = param.agentInfo.userRank;
        deviceLogicId_ = param.agentInfo.deviceLogicId;
        netDevCtx_ = param.agentInfo.netDevCtx;
        backUpNetDevCtx_ = param.agentInfo.backUpNetDevCtx;
        std::string dfxInfo = "deviceIP:" + std::string(param.agentInfo.deviceIP.GetReadableIP())
                              + ";hostIP:" + std::string(param.agentInfo.hostIP.GetReadableIP());
        EXCEPTION_THROW_IF_COND_ERR(
            memcpy_s(localRetryInfo_.dfxIpInfo, sizeof(localRetryInfo_.dfxIpInfo), dfxInfo.c_str(), dfxInfo.size())
                != EOK,
            "memcpy_s dfxIpInfo failed.");
        localRetryInfo_.dfxIpInfo[dfxInfo.size()] = '\0';
    }

    // server状态机初始化
    RetryContext(
        std::map<u32, std::shared_ptr<HcclSocket>>& sockets, std::shared_ptr<OpRetryBase> retryBase,
        const OpRetryAgentInfo& agentInfo)
    {
        retryBase_ = retryBase;
        isRootRetryCtx_ = true;
        for (auto it = sockets.begin(); it != sockets.end(); ++it) {
            HcclAgentRetryInfo tempAgentInfo;
            tempAgentInfo.socket = it->second;
            serverSockets_.insert(std::make_pair(it->first, std::move(tempAgentInfo)));
        }
        rankId_ = agentInfo.userRank;
        deviceLogicId_ = agentInfo.deviceLogicId;
        std::string dfxInfo = "deviceIP:" + std::string(agentInfo.deviceIP.GetReadableIP())
                              + ",hostIP:" + std::string(agentInfo.hostIP.GetReadableIP());
        EXCEPTION_THROW_IF_COND_ERR(
            memcpy_s(localRetryInfo_.dfxIpInfo, sizeof(localRetryInfo_.dfxIpInfo), dfxInfo.c_str(), dfxInfo.size())
                != EOK,
            "memcpy_s dfxIpInfo failed.");
        localRetryInfo_.dfxIpInfo[dfxInfo.size()] = '\0';
    }

    RetryState GetRetryState() const { return state_; }
    const char* GetReadableCtxState() const { return GetReadableState(state_); }

    void SetRetryState(RetryState nextState, std::shared_ptr<OpRetryBase> retryBase)
    {
        HCCL_RUN_INFO(
            "[OpRetry][%s]State Transfer, cur state %s, next state %s", GetOpRetryMachineType(),
            GetReadableState(state_), GetReadableState(nextState));
        state_ = nextState;
        retryBase_ = retryBase;
        localRetryInfo_.retryState = state_;
    }

    void SetEnableSendRecv(bool enable) { retryBase_->SetEnableSendRecv(enable); };

    // 外部接口调用Request()
    HcclResult Request()
    {
        CHK_SMART_PTR_NULL(retryBase_);
        return retryBase_->Handle(this);
    }

    u32 GetRankId() const { return rankId_; }

    const char* GetOpRetryMachineType() const
    {
        std::string ctxType = isRootRetryCtx_ ? "Server" : "Agent";
        return ctxType.c_str();
    }

    bool IsRootRetryCtx() const { return isRootRetryCtx_; }

    const char* GetDfxIpInfo() const { return localRetryInfo_.dfxIpInfo; }

    void ResetAgentState()
    {
        localRetryInfo_.opInfo.execStatus.kfcError = KfcError::kNone;
        localRetryInfo_.isNeedReportOpRetryErr = false;
        isBSRRdmaRecvError_ = false;
        isBSRRdmaSendError_ = false;
    }

    void ResetServerState()
    {
        errorRankList_.clear();
        needRetryServerRanks_.clear();
        isNeedReportOpRetryErr = false;
    }

    std::shared_ptr<HDCommunicate> GetH2dPtr() { return h2dPtr_; }

    std::shared_ptr<HDCommunicate> GetD2hPtr() { return d2hPtr_; }

    bool IsPaused() const { return isPaused_; }

    std::string group_ = "";
    s32 deviceLogicId_ = INVALID_INT;
    u32 rankId_ = INVALID_UINT;
    bool haveCommEnableBackupLink_ = false;

    // agent状态机储存信息
    std::shared_ptr<HcclSocket> agentSocket_ = nullptr;
    std::shared_ptr<HcclOpStreamRes> opStreamPtr_ = nullptr;
    OpRetryResetNotifyCallback notifyResetCallback_ = nullptr;
    OpRetrySetTransportStatusCallback setTransportStatusCallback_ = nullptr;
    OpRetryGetSwitchRanksCallback getSwitchRanksCallback_ = nullptr;
    OpRetrySetTransportResumeStatusCallBack setTransportReseumeStatusCallback_ = nullptr;
    bool isEnableBackupLink_ = false;
    bool isEnableSdmaRetry_ = false;
    RetryInfo localRetryInfo_;
    ChangeLinkInfo localChangeLinkInfo_;
    LinkPortStatus linkPortStatus_;
    bool isChangeLinkInfoInit_ = false;
    std::map<u32, bool> lastLinkPortStatus_;
    bool isUseDefaultPort_ = true;
    HcclNetDevCtx netDevCtx_ = nullptr;
    HcclNetDevCtx backUpNetDevCtx_ = nullptr;
    bool isBSRRdmaRecvError_ = false;
    bool isBSRRdmaSendError_ = false;
    HcclOpIdentifier RemainSendOpId_;
    HcclOpIdentifier RemainRecvOpId_;
    ActiveSwitchInfo switchInfo_;
    bool isAgentStateWaitResume_ = false;

    bool isRecivedCmdToRunning = false;
    bool isRecivedCmdToCheckLink = false;
    // server状态机储存信息
    std::map<u32, HcclAgentRetryInfo> serverSockets_;
    std::vector<u32> needRetryServerRanks_;
    HcclOpIdentifier curFaultOpId;
    std::map<u32, HcclOpIdentifier> errorRankList_;
    bool isRdmaError = false;
    bool isAlreadyChangeLink = false;
    std::map<u32, ActiveSwitchInfo> switchInfoMap_;
    bool isServerStateWaitResume_ = false;
    bool isNeedReportOpRetryErr = false; // 针对重执行算子不一致和inplace场景，上报故障

    bool isOpRetryQuit = false;
    bool isPaused_ = false;

private:
    std::shared_ptr<OpRetryBase> retryBase_ = nullptr;
    RetryState state_ = RETRY_STATE_RESERVED;
    bool isRootRetryCtx_ = false;

    std::shared_ptr<HDCommunicate> h2dPtr_ = nullptr;
    std::shared_ptr<HDCommunicate> d2hPtr_ = nullptr;
};
} // namespace hccl
#endif
