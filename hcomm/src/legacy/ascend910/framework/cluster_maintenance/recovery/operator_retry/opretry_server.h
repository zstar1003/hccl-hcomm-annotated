/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RETRY_SERVER_H
#define HCCL_RETRY_SERVER_H
#include <unordered_set>
#include "opretry_base.h"

namespace hccl {

HcclResult CreateOpRetryServerByState(RetryState state, RetryContext* retryCtx);

// server状态机 正常运行状态转移表
const std::map<RetryState, RetryState> RETRY_SERVER_STATE_TRANSFER_LABEL{
    {RETRY_STATE_SERVER_RUNNING, RETRY_STATE_CMD_STOP_AICPU},
    {RETRY_STATE_CMD_STOP_AICPU, RETRY_STATE_WAIT_AICPU_STOPED},
    {RETRY_STATE_WAIT_AICPU_STOPED, RETRY_STATE_CMD_STOP_STREAM},
    {RETRY_STATE_CMD_STOP_STREAM, RETRY_STATE_WAIT_STREAM_STOPED},
    {RETRY_STATE_WAIT_STREAM_STOPED, RETRY_STATE_CMD_CLEAR_STREAM},
    {RETRY_STATE_CMD_CLEAR_STREAM, RETRY_STATE_WAIT_STREAM_CLEARED},
    {RETRY_STATE_WAIT_STREAM_CLEARED, RETRY_STATE_CMD_STOP_TRANSPORT},
    {RETRY_STATE_CMD_STOP_TRANSPORT, RETRY_STATE_WAIT_STOP_TRANSPORT},
    {RETRY_STATE_WAIT_STOP_TRANSPORT, RETRY_STATE_CMD_CHECK},
    {RETRY_STATE_CMD_CHECK, RETRY_STATE_WAIT_CHECK_INFO},
    {RETRY_STATE_WAIT_CHECK_INFO, RETRY_STATE_CHECK_OP},
    {RETRY_STATE_CHECK_OP, RETRY_STATE_CMD_CHECK_LINK},
    {RETRY_STATE_CMD_CHECK_LINK, RETRY_STATE_WAIT_LINK_CHECKED},
    {RETRY_STATE_WAIT_LINK_CHECKED, RETRY_STATE_CHECK_ALL_LINK},
    {RETRY_STATE_CHECK_ALL_LINK, RETRY_STATE_CMD_RESUME_TRANSPORT},
    {RETRY_STATE_CMD_RESUME_TRANSPORT, RETRY_STATE_WAIT_RESUME_TRANSPORT},
    {RETRY_STATE_WAIT_RESUME_TRANSPORT, RETRY_STATE_CMD_RESET_NOTIFY},
    {RETRY_STATE_CMD_RESET_NOTIFY, RETRY_STATE_WAIT_NOTIFY_RESETED},
    {RETRY_STATE_WAIT_NOTIFY_RESETED, RETRY_STATE_CMD_CAN_RETRY},
    {RETRY_STATE_CMD_CAN_RETRY, RETRY_STATE_WAIT_CAN_RETRY},
    {RETRY_STATE_WAIT_CAN_RETRY, RETRY_STATE_SERVER_RUNNING},
    {RETRY_STATE_SERVER_RETRY_FAIL, RETRY_STATE_SERVER_RUNNING}};

// server状态机 IssueCmd状态对应的command
const std::map<RetryState, RetryCommand> RETRY_SERVER_STATE_TO_CMD_LABEL{
    {RETRY_STATE_CMD_CHECK_LINK, RETRY_CMD_CHECK_LINK},
    {RETRY_STATE_CMD_STOP_AICPU, RETRY_CMD_STOP_AICPU},
    {RETRY_STATE_CMD_STOP_STREAM, RETRY_CMD_STOP_STREAM},
    {RETRY_STATE_CMD_CLEAR_STREAM, RETRY_CMD_CLEAR_STREAM},
    {RETRY_STATE_CMD_STOP_TRANSPORT, RETRY_CMD_STOP_TRANSPORT},
    {RETRY_STATE_CMD_RESET_NOTIFY, RETRY_CMD_RESET_NOTIFY},
    {RETRY_STATE_CMD_RESUME_TRANSPORT, RETRY_CMD_RESUME_TRANSPORT},
    {RETRY_STATE_CMD_CHECK, RETRY_CMD_CHECK_OPNAME},
    {RETRY_STATE_CMD_CAN_RETRY, RETRY_CMD_CAN_RETRY}};

// RETRY_STATE_SERVER_RETRY_FAIL 重执行异常状态处理
class OpRetryServerBase : public OpRetryBase {
public:
    HcclResult ProcessError(RetryContext* retryCtx) override;
};

// RETRY_STATE_SERVER_RUNNING
class OpRetryServerRunning : public OpRetryServerBase {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;
    HcclResult ParaseErrorCode(RetryContext* retryCtx, HcclAgentRetryInfo& agentInfo, RetryState& nextState);

protected:
    std::map<u32, std::chrono::steady_clock::time_point> lastRecvTimes_;
    std::unordered_set<u32> disableAgent_; // 记录已经关闭的对端, 不再轮询, 避免刷屏
};

// server处理错误rank状态机
class OpRetryServerHandleError : public OpRetryServerRunning {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;

private:
    HcclResult SetNeedRetryServerRank(RetryContext* retryCtx, const HcclOpIdentifier& opId);
};

// 公共状态-向agent状态机发送命令
class OpRetryServerIssueCmd : public OpRetryServerBase {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;
};

// 公共状态-等待agent状态机回复
class OpRetryServerWaitResp : public OpRetryServerBase {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;

private:
    // 接收到对端重执行失败的信息后，打印当前接收到的Agent节点信息
    void PrintAgentInfoAfterFail(
        std::map<u32, HcclAgentRetryInfo>& serverSockets, std::set<u32>& recvVaild,
        HcclAgentRetryInfo& agentRetryInfo) const;
};

class OpRetryServerCheckOp : public OpRetryServerBase {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;
};

class OpRetryServerCheckAllLink : public OpRetryServerBase {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;
};

class OpRetryServerIssueChangeLinkAndResume : public OpRetryServerBase {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;
};

class OpRetryServerWaitLinkInfo : public OpRetryServerBase {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;
};

class OpRetryServerRetryFail : public OpRetryServerBase {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;
};

// 强制终止NPU后的状态，等待通信域恢复
class OpRetryServerWaitResume : public OpRetryServerRunning {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;
};

class SwitchNicServerCheckAllSwitchRanks : public OpRetryServerBase {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;

private:
    bool
    CompareSwitchRankList(const u32* firstSwitchRankList, const u32* switchRankList, const u32 switchRankNum) const;
    bool CompareUseBackupLists(const bool* firstArray, const bool* secondArray, const u32 switchRankNum) const;
    bool CheckRemotePorts(const u32 rankId, const ActiveSwitchInfo& switchRankInfo);
    HcclResult
    CollectSingleAgentActiveSwitchInfo(RetryContext* retryCtx, const u32 rankId, HcclAgentRetryInfo& agentInfo);
    HcclResult CollectAgentActiveSwitchInfo(RetryContext* retryCtx);
    HcclResult CheckAgentActiveSwitchInfo(RetryContext* retryCtx);
};

class ResumeServerCheckAllLink : public OpRetryServerBase {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;

private:
    HcclResult WaitAgentCheckLinkResult(RetryContext* retryCtx);
    HcclResult CheckAllLink(RetryContext* retryCtx, RetryState& nextState);
};

class ResumeServerChangeLink : public OpRetryServerBase {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;

private:
    HcclResult CmdAgentChangeLink(RetryContext* retryCtx);
    HcclResult WaitAllChangeLinkResult(RetryContext* retryCtx, RetryState& nextState);
};
} // namespace hccl
#endif
