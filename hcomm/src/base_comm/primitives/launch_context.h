/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LAUNCH_CONTEXT_H
#define LAUNCH_CONTEXT_H

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <mutex>
#include <algorithm>
#include "hccl_api_data.h"
#include "log.h"
#include "rtsq_base.h"

constexpr uint32_t NOTIFY_WAIT_TIMEOUT_DEFAULT = 1836;

class LaunchContext {
public:
    LaunchContext();

    HcclResult SetLaunchMode(const char* launchTag, HcommLaunchMode mode);
    inline void AddThreadWithTag(ThreadHandle thread) // ffts场景使用，支持储存存多个子图对应的thread信息
    {
        if (mode_ != HCOMM_LAUNCH_MODE_BATCH) {
            return;
        }
        auto& threadSet = launchModeMap_[launchTag_];
        threadSet.insert(thread);
    }

    inline void AddThread(ThreadHandle thread) // 储存当前线程使用的thread
    {
        if (UNLIKELY(mode_ != HCOMM_LAUNCH_MODE_BATCH)) {
            return;
        }
        if (std::find(threadVec_.begin(), threadVec_.end(), thread) == threadVec_.end()) {
            threadVec_.push_back(thread);
        }
    }

    HcclResult SetNotifyWaitTimeOut(uint32_t timeout);
    HcclResult GetNotifyWaitTimeOut(uint32_t& timeout);
    HcclResult SetSqFullTimeOut(uint32_t timeout);
    uint32_t GetSqFullTimeOut();
    inline bool IsBatchLaunchMode() const { return mode_ == HCOMM_LAUNCH_MODE_BATCH; }
    HcclResult HandleDispatchAllStreams();

private:
    HcclResult HandleBatchMode();
    HcclResult HandleEagerMode();
    HcclResult HandleClear();

    std::string launchTag_; // 当前tag
    std::unordered_map<std::string, std::unordered_set<ThreadHandle>>
        launchModeMap_;                   // 按tag粒度记录当前线程使用的thread
    std::vector<ThreadHandle> threadVec_; // 不区分tag，记录当前线程使用的thread

    struct NotifyWaitTimeoutConfig {
        uint32_t notifyWaitTimeout = NOTIFY_WAIT_TIMEOUT_DEFAULT;
        bool isSet = false;
    } notifyWaitTimeoutConfig_;

    struct SqFullTimeoutConfig {
        uint32_t sqFullTimeout = Hccl::RTSQ_FULL_TIMEOUT_DEFAULT;
        bool isSet = false;
    } sqFullTimeoutConfig_;

    HcommLaunchMode mode_ = HCOMM_LAUNCH_MODE_EAGER;
};

#endif
