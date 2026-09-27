/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "launch_context.h"
#include "new/hccl_primitive_local.h"

constexpr u32 THREAD_VECTOR_DEFAULT_SIZE = 128; // 设置vector初始长度，避免频繁扩容
constexpr u32 NOTIFY_WAIT_TIMEOUT_OFFSET = 27;  // AICPU device侧notify等待超时偏移量

extern HcclResult CommTaskLaunch(ThreadHandle* threads, uint32_t threadNum); // host ffts+或aicpu stars使用"
extern HcclResult CommTaskPrepare(char* key, uint32_t keyLen);               // host ffts+使用
extern HcclResult DispatchAllStreams(const ThreadHandle* threads, uint32_t threadNum);

LaunchContext::LaunchContext() { threadVec_.reserve(THREAD_VECTOR_DEFAULT_SIZE); }

HcclResult LaunchContext::HandleEagerMode()
{
    // 带launchTag部分
    if (!launchModeMap_.empty()) {
        auto it = launchModeMap_.find(launchTag_);
        if (it != launchModeMap_.end()) {
            std::vector<ThreadHandle> threadVec(it->second.begin(), it->second.end());
            CHK_RET(CommTaskLaunch(threadVec.data(), threadVec.size()));
            HCCL_INFO("[%s] success, launchTag[%s], size[%zu]", __func__, launchTag_.c_str(), threadVec.size());
        }
    }

    // 不带launchTag部分
    if (!threadVec_.empty()) {
        CHK_RET(CommTaskLaunch(threadVec_.data(), threadVec_.size()));
        HCCL_INFO("[%s] success, size[%zu]", __func__, threadVec_.size());
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchContext::HandleDispatchAllStreams()
{
    // 带launchTag部分
    if (!launchModeMap_.empty()) {
        auto it = launchModeMap_.find(launchTag_);
        if (it != launchModeMap_.end()) {
            std::vector<ThreadHandle> threadVec(it->second.begin(), it->second.end());
            CHK_RET(DispatchAllStreams(threadVec.data(), threadVec.size()));
        }
    }

    // 不带launchTag部分
    if (!threadVec_.empty()) {
        CHK_RET(DispatchAllStreams(threadVec_.data(), threadVec_.size()));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchContext::HandleClear()
{
    threadVec_.clear();
    if (!launchModeMap_.empty()) {
        launchModeMap_.erase(launchTag_);
    }
    HCCL_INFO(
        "[%s] begin clear, launchTag[%s], launchMode[%d].", __func__, launchTag_.c_str(), static_cast<int32_t>(mode_));

    DevType devType = DevType::DEV_TYPE_COUNT;
    hrtGetDeviceType(devType);
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        HCCL_INFO("[%s] Running on A5/A6, HcclTaskClear skipped.", __func__);
        return HCCL_SUCCESS;
    }
    return HcclTaskClear(launchTag_);
}

HcclResult LaunchContext::SetNotifyWaitTimeOut(uint32_t timeout)
{
    notifyWaitTimeoutConfig_.notifyWaitTimeout = timeout;
    notifyWaitTimeoutConfig_.isSet = true;
    return HCCL_SUCCESS;
}

HcclResult LaunchContext::GetNotifyWaitTimeOut(uint32_t& timeout)
{
    timeout = notifyWaitTimeoutConfig_.notifyWaitTimeout;
#ifndef CCL_KERNEL_AICPU
    if (!notifyWaitTimeoutConfig_.isSet) {
        timeout = timeout + NOTIFY_WAIT_TIMEOUT_OFFSET;
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult LaunchContext::SetSqFullTimeOut(uint32_t timeout)
{
    sqFullTimeoutConfig_.sqFullTimeout = timeout;
    sqFullTimeoutConfig_.isSet = true;
    return HCCL_SUCCESS;
}

uint32_t LaunchContext::GetSqFullTimeOut() { return sqFullTimeoutConfig_.sqFullTimeout; }

/*
    1 AICPU_TS模式
    AICPU上执行
    告知后面的CommWrite等任务进入批量模式，（只写任务的SQE，但是不触发执行）
    举例：
    HcommSetLaunchMode("abc", HCOMM_LAUNCH_MODE_BATCH);
    HcommAclrtNotifyWaitOnThread(thread, notifyId, 0);
    HcommAclrtNotifyRecordOnThread(thread, notifyId);
    HcommSetLaunchMode("abc", HCOMM_LAUNCH_MODE_EAGER);

    2 CPU_TS模式
    FFTS+子图，最后批量提交。在HOST CPU上执行
    告知后面的CommWrite等任务进入批量模式（开始ffts+子图）

    1）复用task子图缓存
    增加 launchTag 的原因，进入批量模式之后，缓存要执行的一些task，最后提交。缓
    存的标识采用launchTag。在第二次执行想要复用子图执行时，只需要拿着相同的
    launchTag，调用 HcommSetLaunchMode接口，传入HCOMM_LAUNCH_MODE_EAGER参数，即可复用执行。
    比如下面的： HcommSetLaunchMode ("abc", HCOMM_LAUNCH_MODE_EAGER);
    执行之前缓存到"abc"下的几个数据面操作。

    2）清理
    如果不需要"abc"标识的这个子图的task 缓存了，可以采用如下方式清理该子图内容：
    HcommSetLaunchMode ("abc", HCOMM_LAUNCH_MODE_RESERVED)

    3）缺省 launchTag
    launchTag 如果为 nullptr，表示缺省值，标识不需要缓存到 FFTS+子图。
 */
HcclResult LaunchContext::SetLaunchMode(const char* launchTag, HcommLaunchMode mode)
{
    mode_ = mode;
    // 统一处理 launchTag
    bool defaultTag = (launchTag == nullptr);
    launchTag_ = defaultTag ? "" : std::string(launchTag);
    HCCL_INFO(
        "[%s] SetLaunchMode begin, launchTag[%s], launchMode[%d].", __func__, launchTag_.c_str(),
        static_cast<int32_t>(mode));

#ifndef CCL_KERNEL_AICPU
    DevType devType = DevType::DEV_TYPE_COUNT;
#endif
    switch (mode_) {
        case HCOMM_LAUNCH_MODE_BATCH:
#ifndef CCL_KERNEL_AICPU
            hrtGetDeviceType(devType);
            if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
                HCCL_INFO("[%s] Running on A5, CommTaskPrepare skipped.", __func__);
                return HCCL_SUCCESS;
            }
            HCCL_INFO("[%s] host mode, need CommTaskPrepare", __func__);
            if (!defaultTag) {
                // 仅非缺省 tag 需要准备任务缓存
                return CommTaskPrepare(const_cast<char*>(launchTag_.c_str()), launchTag_.length());
            }
#endif
            return HCCL_SUCCESS;
        case HCOMM_LAUNCH_MODE_EAGER:
            CHK_RET(HandleEagerMode());
            // 缺省 tag 模式下清理缓存
            return HandleClear();
        case HCOMM_LAUNCH_MODE_RESERVED:
            if (!defaultTag) {
                return HandleClear();
            }
            return HCCL_SUCCESS;
        default:
            return HCCL_SUCCESS;
    }
}
