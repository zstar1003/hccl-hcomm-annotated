/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <mutex>
#include "task_abort_handler_v2.h"
#include "ccu_dev_mgr.h"
#include "log.h"

namespace Hccl {
using HcclUs = std::chrono::steady_clock::time_point;
static int32_t TaskAbortPre(const std::vector<HcclCommunicator*>& commVector, const std::chrono::seconds& localtimeout)
{
    HcclResult ret = HCCL_SUCCESS;
    bool isUseTimeOut = localtimeout != std::chrono::seconds(0);
    std::chrono::seconds elapsed{};
    for (const auto& comm : commVector) {
        if (isUseTimeOut) {
            std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
            ret = comm->Suspend();
            elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - startTime);
        } else {
            ret = comm->Suspend();
        }
        if (ret != HCCL_SUCCESS && ret != HCCL_E_SUSPENDING) {
            HCCL_ERROR("[NsRecovery] finish suspend failed");
            return static_cast<int>(TaskAbortResult::TASK_ABORT_FAIL);
        }
        HCCL_DEBUG("[NsRecovery]finish suspend success");
        if (isUseTimeOut) {
            CHK_PRT_RET(
                elapsed > localtimeout, HCCL_ERROR("[NsRecovery][suspend] NsRecovery suspend timeOut"),
                static_cast<int>(TaskAbortResult::TASK_ABORT_TIMEOUT));
        }
    }
    return static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS);
}

static int32_t TaskAbortPost(
    const std::vector<HcclCommunicator*>& commVector, int32_t deviceLogicId, const std::chrono::seconds& localtimeout)
{
    HcclResult ret = HCCL_SUCCESS;
    bool isUseTimeOut = localtimeout != std::chrono::seconds(0);
    std::chrono::seconds elapsed{};
    if (CcuIsInited(deviceLogicId)) {
        CHK_RET(HcclCcuTaskKillPreProcess(deviceLogicId));
    } else {
        HCCL_INFO("[NsRecovery][Callback] CCU not inited, skip TaskKillPreProcess, deviceLogicId[%d]", deviceLogicId);
    }
    for (const auto& comm : commVector) {
        if (isUseTimeOut) {
            std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
            ret = comm->Clean();
            elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - startTime);
        } else {
            ret = comm->Clean();
        }
        if (ret != HCCL_SUCCESS && ret != HCCL_E_SUSPENDING) {
            HCCL_ERROR("[NsRecovery][Callback] finish clean failed");
            return static_cast<int>(TaskAbortResult::TASK_ABORT_FAIL);
        }
        HCCL_INFO("[NsRecovery][Callback] finish clean success");
        if (isUseTimeOut) {
            CHK_PRT_RET(
                elapsed > localtimeout, HCCL_ERROR("[NsRecovery][Callback] NsRecovery Clean timeout"),
                static_cast<int>(TaskAbortResult::TASK_ABORT_TIMEOUT));
        }
    }
    if (CcuIsInited(deviceLogicId)) {
        CHK_RET(HcclCcuTaskKillPostProcess(deviceLogicId));
        // Align with TaskException / HCOMM POST: clear residual CCU CKE after TaskKillDone.
        for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; ++dieId) {
            CHK_RET(CcuCleanDieCkes(deviceLogicId, dieId));
        }
    } else {
        HCCL_INFO(
            "[NsRecovery][Callback] CCU not inited, skip TaskKillPostProcess/CcuCleanDieCkes, deviceLogicId[%d]",
            deviceLogicId);
    }
    return static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS);
}

int32_t
ProcessTaskAbortHandleCallback(int32_t deviceLogicId, aclrtDeviceTaskAbortStage stage, uint32_t timeout, void* args)
{
    HcclUs startut = std::chrono::steady_clock::now();
    CHK_PTR_NULL(args);
    auto& commVector = *(static_cast<std::vector<HcclCommunicator*>*>(args));
    HCCL_INFO(
        "[NsRecovery][Callback] ProcessTaskAbortHandleCallback begin, deviceLogicId [%d], stage [%d], commVector "
        "size [%lu]",
        deviceLogicId, stage, commVector.size());
    const std::chrono::seconds localtimeout = std::chrono::seconds(timeout);

    if (stage == aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_PRE) {
        auto result = TaskAbortPre(commVector, localtimeout);
        if (result != static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS)) {
            return result;
        }
    } else if (stage == aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_POST) {
        auto result = TaskAbortPost(commVector, deviceLogicId, localtimeout);
        if (result != static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS)) {
            return result;
        }
    }
    HcclUs endut = std::chrono::steady_clock::now();
    HCCL_INFO(
        "[NsRecovery][Callback] ProcessTaskAbortHandleCallback success, take time:[%lld]us",
        std::chrono::duration_cast<std::chrono::microseconds>(endut - startut).count());
    return static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS);
}

TaskAbortHandler::TaskAbortHandler()
{
    std::string name = "HCCL";
    HrtDeviceAbortRegCallBack(ProcessTaskAbortHandleCallback, static_cast<void*>(&commVector), name);
}

TaskAbortHandler::~TaskAbortHandler()
{
    std::string name = "HCCL";
    DECTOR_TRY_CATCH("TaskAbortHandler", HrtDeviceAbortRegCallBack(nullptr, nullptr, name));
}

TaskAbortHandler& TaskAbortHandler::GetInstance()
{
    // Leaky Singleton: 故意不释放，规避 Static Destruction Order Fiasco，
    // 确保析构回调访问 commVector 时单例仍有效
    static TaskAbortHandler& handler = *new TaskAbortHandler();
    return handler;
}

HcclResult TaskAbortHandler::Register(HcclCommunicator* communicator)
{
    std::lock_guard<std::mutex> lock(vecMutex);
    commVector.push_back(communicator);
    HCCL_INFO("TaskAbortHandler::Register success, commVector size is [%lu]", commVector.size());

    return HCCL_SUCCESS;
}

HcclResult TaskAbortHandler::UnRegister(HcclCommunicator* communicator)
{
    std::lock_guard<std::mutex> lock(vecMutex);
    HCCL_INFO("TaskAbortHandler::UnRegister Begin, commVector size is [%lu]", commVector.size());
    auto it = std::find(commVector.begin(), commVector.end(), communicator);
    if (it != commVector.end()) {
        commVector.erase(it);
    } else {
        HCCL_WARNING("TaskAbortHandler::UnRegister, comm not found.");
    }
    HCCL_INFO("TaskAbortHandler::UnRegister finish, commVector size is [%lu]", commVector.size());
    return HCCL_SUCCESS;
}
} // namespace Hccl
