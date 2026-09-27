/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_OPRETRY_MANAGER_H
#define HCCL_OPRETRY_MANAGER_H
#include <thread>
#include <mutex>
#include "opretry_base.h"

namespace hccl {
struct RetryCtrl {
    std::unique_ptr<std::thread> thread;
    std::shared_ptr<RetryContext> retryCtx = nullptr;
    bool startExec = false;
};

class OpRetryManager {
public:
    OpRetryManager() = default;
    ~OpRetryManager()
    {
        HCCL_DEBUG("Destroy OpRetryManager");
        (void)DeInit();
    }
    HcclResult RegisterOpRetryMachine(
        OpRetryAgentParam& agentParam, u32 rankSize, bool isRoot,
        std::map<u32, std::shared_ptr<HcclSocket>>& serverConnections, const OpRetryServerInfo& serverInfo);
    HcclResult UnRegisterOpRetryManager(const std::string& group);

    static HcclResult AddLinkInfoByIdentifier(
        s32 deviceLogicID, const std::string& identifier, const std::string& newTag, std::vector<u32>& remoteRankList,
        bool incre = false);
    static HcclResult GetLinkInfoByIdentifier(
        s32 deviceLogicID, const std::string& identifier, const std::string& newTag, std::vector<u32>& remoteRankList,
        bool isGetGroupAllRemoteRank = false);
    static HcclResult DeleteLinkInfoByIdentifier(s32 deviceLogicID, const std::string& identifier);
    HcclResult SetRetryStateToWaitResume(const std::string& group, bool isRoot);
    HcclResult
    ExitWaitResumeState(const std::string& group, bool isRoot, bool haveCommEnableBackupLink, bool& isChangedLink);
    bool IsPaused(const std::string& group);
    bool IsResumed(const std::string& group);

private:
    HcclResult Init();
    HcclResult DeInit()
    {
        std::unique_lock<std::mutex> lock(ProcessLock_);
        if (initialized_) {
            initialized_ = false;
            for (auto it = agentOpRetry_.begin(); it != agentOpRetry_.end(); ++it) {
                if (it->second.thread != nullptr && it->second.thread->joinable()) {
                    it->second.thread->join();
                }
            }
            agentOpRetry_.clear();

            for (auto it = serverOpRetry.begin(); it != serverOpRetry.end(); ++it) {
                if (it->second.thread != nullptr && it->second.thread->joinable()) {
                    it->second.thread->join();
                }
            }
            serverOpRetry.clear();
            HCCL_INFO("OpRetryManager DeInit success");
        }
        return HCCL_SUCCESS;
    }
    HcclResult RegisterAgentRetryMachine(OpRetryAgentParam& agentParam);
    HcclResult RegisterServerRetryMachine(
        const std::string& group, std::map<u32, std::shared_ptr<HcclSocket>>& serverConnections,
        const OpRetryAgentInfo& agentInfo);
    void RetryStateMonitor(
        const std::string& group, std::shared_ptr<RetryContext> retryCtx, const bool& startExec, HcclRtContext rtCtx);

private:
    std::map<std::string, RetryCtrl> serverOpRetry;
    std::map<std::string, RetryCtrl> agentOpRetry_;
    bool initialized_ = false;
    std::mutex ProcessLock_;
};
} // namespace hccl
#endif
