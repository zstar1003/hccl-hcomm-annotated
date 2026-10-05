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
    // [中文导读] [AllReduce逐行 S31] AddThreadWithTag的接口声明：当前执行Thread句柄；这些参数属于本函数调用边界。
    inline void AddThreadWithTag(ThreadHandle thread) // ffts场景使用，支持储存存多个子图对应的thread信息
    // [中文导读] [AllReduce逐行 S32] 进入AddThreadWithTag函数体：仅BATCH模式按当前tag保存Thread集合。
    {
        // [中文导读] [AllReduce逐行 S33] 仅当`(mode_ != HCOMM_LAUNCH_MODE_BATCH)`（当前提交模式）成立时进入此分支。
        if (mode_ != HCOMM_LAUNCH_MODE_BATCH) {
            // [中文导读] [AllReduce逐行 S34] 结束本次void调用，当前路径不再继续下发后续操作。
            return;
        // [中文导读] [AllReduce逐行 S35] 结束`if (mode_ != HCOMM_LAUNCH_MODE_BATCH)`（当前提交模式）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S36] 设置auto& threadSet为/按`launchModeMap_[launchTag_]`（按提交标签登记的Thread集合、当前提交标签）。
        auto& threadSet = launchModeMap_[launchTag_];
        // [中文导读] [AllReduce逐行 S37] 把键/对象写入对应映射或集合；传入/处理当前执行Thread句柄。
        threadSet.insert(thread);
    // [中文导读] [AllReduce逐行 S38] 结束AddThreadWithTag函数体；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S40] AddThread的接口声明：当前执行Thread句柄；这些参数属于本函数调用边界。
    inline void AddThread(ThreadHandle thread) // 储存当前线程使用的thread
    // [中文导读] [AllReduce逐行 S41] 进入AddThread函数体：仅BATCH模式登记参与Thread并按句柄去重；批结束提交列表。
    {
        // [中文导读] [AllReduce逐行 S42] 仅当`(UNLIKELY(mode_ != HCOMM_LAUNCH_MODE_BATCH))`（当前提交模式）成立时进入此分支。
        if (UNLIKELY(mode_ != HCOMM_LAUNCH_MODE_BATCH)) {
            // [中文导读] [AllReduce逐行 S43] 结束本次void调用，当前路径不再继续下发后续操作。
            return;
        // [中文导读] [AllReduce逐行 S44] 结束`if (UNLIKELY(mode_ != HCOMM_LAUNCH_MODE_BATCH))`（当前提交模式）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S45] 仅当当前Thread句柄尚未存在于threadVec_时追加，避免同一批重复提交同一Thread。
        if (std::find(threadVec_.begin(), threadVec_.end(), thread) == threadVec_.end()) {
            // [中文导读] [AllReduce逐行 S46] 把当前Thread加入本轮BATCH参与列表；批结束HandleEagerMode将把该列表交CommTaskLaunch。
            threadVec_.push_back(thread);
        // [中文导读] [AllReduce逐行 S47] 结束`if (std::find(threadVec_.begin(), threadVec_.end(), thread) == threadVec_.end())`（本轮BATCH参与Thread列表的begin字段、本轮BATCH参与Thread列表的end字段、当前执行Thread句柄）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S48] 结束AddThread函数体；控制流返回外层。
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
