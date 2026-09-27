/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_abort_handler_v1.h"
#include "sal_pub.h"
#include "hccl_communicator.h"
#include "adapter_rts_common.h"

using namespace hccl;
using namespace std;

static std::vector<HcclCommunicator*> commVector;
static Referenced ref_;
static std::mutex mutex_;
struct TaskAbortCbArgs {
    u64 commVectorAddr;
};

TaskAbortHandler::TaskAbortHandler() {}
TaskAbortHandler::~TaskAbortHandler() {}

int32_t
ProcessTaskAbortHandleCallback(int32_t deviceLogicId, aclrtDeviceTaskAbortStage stage, uint32_t timeout, void* args)
{
    HcclUs startut = TIME_NOW();
    CHK_PTR_NULL(args);
    std::lock_guard<std::mutex> lock(mutex_);
    HCCL_INFO(
        "ProcessTaskAbortHandleCallback begin, deviceLogicId [%d], stage [%d], args [%p], commVector v1 size [%u], "
        "ref_ count is [%d]",
        deviceLogicId, stage, args, commVector.size(), ref_.Count());
    const std::chrono::seconds localtimeout = std::chrono::seconds(timeout);
    HcclResult ret = HCCL_SUCCESS;
    if (localtimeout != std::chrono::seconds(0)) {
        if (stage == ACL_RT_DEVICE_TASK_ABORT_PRE) {
            for (size_t i = 0; i < commVector.size(); i++) {
                std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
                ret = commVector[i]->Suspend();
                std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
                if (ret != HCCL_SUCCESS && ret != HCCL_E_SUSPENDING) {
                    HCCL_ERROR("[NsRecovery] finish suspend failed");
                    return static_cast<int>(TaskAbortResult::TaskAbort_Fail);
                }
                HCCL_DEBUG("[NsRecovery]finish suspend success");
                const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
                CHK_PRT_RET(
                    elapsed > localtimeout, HCCL_ERROR("[NsRecovery][suspend] NsRecovery suspend timeOut"),
                    static_cast<int>(TaskAbortResult::TaskAbort_TimeOut));
            }
        } else if (stage == ACL_RT_DEVICE_TASK_ABORT_POST) {
            for (size_t i = 0; i < commVector.size(); i++) {
                std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
                ret = commVector[i]->StopExec();
                std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
                if (ret != HCCL_SUCCESS && ret != HCCL_E_SUSPENDING) {
                    HCCL_ERROR("[NsRecovery] finish stopExec failed");
                    return static_cast<int>(TaskAbortResult::TaskAbort_Fail);
                }
                HCCL_DEBUG("[NsRecovery]finish stopExec success");
                const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
                CHK_PRT_RET(
                    elapsed > localtimeout, HCCL_ERROR("[NsRecovery][StopExec] NsRecovery StopExec timeOut"),
                    static_cast<int>(TaskAbortResult::TaskAbort_TimeOut));
            }
            HcclUs stopExecUt = TIME_NOW();
            HCCL_RUN_INFO(
                "TaskAbortHandler:ProcessTaskAbortHandleCallback, stopExec take time:[%lld]us",
                DURATION_US(stopExecUt - startut).count());

            for (size_t i = 0; i < commVector.size(); i++) {
                std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
                ret = commVector[i]->Clean();
                std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
                if (ret != HCCL_SUCCESS && ret != HCCL_E_SUSPENDING) {
                    HCCL_ERROR("[NsRecovery] finish clean failed");
                    return static_cast<int>(TaskAbortResult::TaskAbort_Fail);
                }
                HCCL_DEBUG("[NsRecovery]finish clean success");
                const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
                CHK_PRT_RET(
                    elapsed > localtimeout, HCCL_ERROR("[NsRecovery][clean] NsRecovery Clean timeOut"),
                    static_cast<int>(TaskAbortResult::TaskAbort_TimeOut));
                CHK_RET(commVector[i]->Stop());
            }
        }
    } else {
        if (stage == ACL_RT_DEVICE_TASK_ABORT_PRE) {
            for (size_t i = 0; i < commVector.size(); i++) {
                ret = commVector[i]->Suspend();
                if (ret != HCCL_SUCCESS && ret != HCCL_E_SUSPENDING) {
                    HCCL_ERROR("[NsRecovery] finish suspend failed");
                    return static_cast<int>(TaskAbortResult::TaskAbort_Fail);
                }
                HCCL_DEBUG("[NsRecovery]finish suspend success");
            }
        } else if (stage == ACL_RT_DEVICE_TASK_ABORT_POST) {
            for (size_t i = 0; i < commVector.size(); i++) {
                ret = commVector[i]->StopExec();
                if (ret != HCCL_SUCCESS && ret != HCCL_E_SUSPENDING) {
                    HCCL_ERROR("[NsRecovery] finish stopExec failed");
                    return static_cast<int>(TaskAbortResult::TaskAbort_Fail);
                }
                HCCL_DEBUG("[NsRecovery]finish stopExec success");
            }
            HcclUs stopExecUt = TIME_NOW();
            HCCL_RUN_INFO(
                "TaskAbortHandler:ProcessTaskAbortHandleCallback, stopExec take time:[%lld]us",
                DURATION_US(stopExecUt - startut).count());
            for (size_t i = 0; i < commVector.size(); i++) {
                ret = commVector[i]->Clean();
                if (ret != HCCL_SUCCESS && ret != HCCL_E_SUSPENDING) {
                    HCCL_ERROR("[NsRecovery] finish clean failed");
                    return static_cast<int>(TaskAbortResult::TaskAbort_Fail);
                }
                HCCL_DEBUG("[NsRecovery]finish clean success");
                CHK_RET(commVector[i]->Stop());
            }
        }
    }

    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO(
        "TaskAbortHandler:ProcessTaskAbortHandleCallback, deviceLogicId [%d], stage [%d], total take time:[%lld]us",
        deviceLogicId, stage, DURATION_US(endut - startut).count());
    return static_cast<int>(TaskAbortResult::TaskAbort_Success);
}

HcclResult TaskAbortHandler::Init(HcclCommunicator* communicator)
{
    std::unique_lock<std::mutex> lock(mutex_);
    HCCL_INFO("TaskAbortHandler::Init commVector size is [%d], ref_ count is [%d]", commVector.size(), ref_.Count());
    if (ref_.Count() == 0) {
        CHK_RET(hrtTaskAbortHandleCallback(ProcessTaskAbortHandleCallback, static_cast<void*>(&commVector)));
    }
    ref_.Ref();
    commVector.push_back(communicator);

    return HCCL_SUCCESS;
}

HcclResult TaskAbortHandler::DeInit(HcclCommunicator* communicator)
{
    std::unique_lock<std::mutex> lock(mutex_);
    HCCL_INFO("TaskAbortHandler::DeInit commVector size is [%d], ref_ count is [%d]", commVector.size(), ref_.Count());
    for (auto it = commVector.begin(); it != commVector.end();) {
        if (*it == communicator) {
            it = commVector.erase(it);
        } else {
            ++it;
        }
    }
    ref_.Unref();
    if (ref_.Count() == 0) {
        commVector.clear();
        CHK_RET(hrtTaskAbortHandleCallback(nullptr, nullptr));
    }
    return HCCL_SUCCESS;
}
