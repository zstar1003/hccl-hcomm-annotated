/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_abort_handler.h"
#include <algorithm>
#include <mutex>
#include "log.h"
#include "coll_comm.h"
#include "ccu_device_pub.h"

namespace hccl {
using HcclUs = std::chrono::steady_clock::time_point;

int32_t ProcessTaskAbortPre(const std::vector<CollComm*>& commVector, const std::chrono::seconds& localtimeout)
{
    HcclResult ret = HCCL_SUCCESS;
    bool isUseTimeOut = localtimeout != std::chrono::seconds(0);
    std::chrono::seconds elapsed{};
    for (auto& comm : commVector) {
        if (isUseTimeOut) {
            std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
            ret = comm->Suspend();
            elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - startTime);
        } else {
            ret = comm->Suspend();
        }
        if (ret != HCCL_SUCCESS && ret != HCCL_E_SUSPENDING) {
            HCCL_ERROR("[NsRecovery] finish suspend failed, ret = 0x%016llx", HCCL_ERROR_CODE(ret));
            return static_cast<int>(TaskAbortResult::TASK_ABORT_FAIL);
        }
        HCCL_INFO("[NsRecovery]finish suspend success");
        if (isUseTimeOut) {
            CHK_PRT_RET(
                elapsed > localtimeout, HCCL_ERROR("[NsRecovery][suspend] NsRecovery suspend timeOut"),
                static_cast<int>(TaskAbortResult::TASK_ABORT_TIMEOUT));
        }
    }
    return static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS);
}

int32_t ProcessTaskAbortPost(
    const std::vector<CollComm*>& commVector, int32_t deviceLogicId, const std::chrono::seconds& localtimeout)
{
    HcclResult ret = HCCL_SUCCESS;
    bool isUseTimeOut = localtimeout != std::chrono::seconds(0);
    std::chrono::seconds elapsed{};
    if (hcomm::CcuIsInited(deviceLogicId)) {
        CHK_RET(hcomm::CcuSetTaskKill(deviceLogicId));
    } else {
        HCCL_INFO("[NsRecovery][Callback] CCU not inited, skip CcuSetTaskKill, deviceLogicId[%d]", deviceLogicId);
    }
    for (auto& comm : commVector) {
        if (isUseTimeOut) {
            std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
            ret = comm->Clean();
            elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - startTime);
        } else {
            ret = comm->Clean();
        }
        if (ret != HCCL_SUCCESS && ret != HCCL_E_SUSPENDING) {
            HCCL_ERROR("[NsRecovery][Callback] finish clean failed, ret = 0x%016llx", HCCL_ERROR_CODE(ret));
            return static_cast<int>(TaskAbortResult::TASK_ABORT_FAIL);
        }
        HCCL_INFO("[NsRecovery][Callback] finish clean success");
        if (isUseTimeOut) {
            CHK_PRT_RET(
                elapsed > localtimeout, HCCL_ERROR("[NsRecovery][Callback] NsRecovery Clean timeout"),
                static_cast<int>(TaskAbortResult::TASK_ABORT_TIMEOUT));
        }
    }
    if (hcomm::CcuIsInited(deviceLogicId)) {
        CHK_RET(hcomm::CcuSetTaskKillDone(deviceLogicId));
        // Align with TaskException: clear residual CCU CKE after TaskKillDone (device-level, all IO dies).
        for (uint8_t dieId = 0; dieId < hcomm::CCU_MAX_IODIE_NUM; ++dieId) {
            CHK_RET(hcomm::CcuCleanDieCkes(deviceLogicId, dieId));
        }
    } else {
        HCCL_INFO(
            "[NsRecovery][Callback] CCU not inited, skip CcuSetTaskKillDone/CcuCleanDieCkes, deviceLogicId[%d]",
            deviceLogicId);
    }
    return static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS);
}

int32_t
ProcessTaskAbortHandleCallback(int32_t deviceLogicId, aclrtDeviceTaskAbortStage stage, uint32_t timeout, void* args)
{
    HcclUs startut = std::chrono::steady_clock::now();
    CHK_PTR_NULL(args);
    auto* handler = static_cast<HcclTaskAbortHandler*>(args);
    std::lock_guard<std::mutex> lock(handler->vecMutex_);
    auto& commVector = handler->commVector_;
    HCCL_INFO("[NsRecovery][Callback] ProcessTaskAbortHandleCallback start!");
    const std::chrono::seconds localtimeout = std::chrono::seconds(timeout);

    if (stage == aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_PRE) {
        auto result = ProcessTaskAbortPre(commVector, localtimeout);
        if (result != static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS)) {
            return result;
        }
    } else if (stage == aclrtDeviceTaskAbortStage::ACL_RT_DEVICE_TASK_ABORT_POST) {
        auto result = ProcessTaskAbortPost(commVector, deviceLogicId, localtimeout);
        if (result != static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS)) {
            return result;
        }
    }
    HcclUs endut = std::chrono::steady_clock::now();
    auto execTime = std::chrono::duration_cast<std::chrono::microseconds>(endut - startut).count();
    HCCL_RUN_INFO("[NsRecovery][Callback] ProcessTaskAbortHandleCallback success, take time:[%lld]us", execTime);
    return static_cast<int>(TaskAbortResult::TASK_ABORT_SUCCESS);
}

HcclTaskAbortHandler::HcclTaskAbortHandler()
{
    std::string name = "HCOMM";
    Hccl::HrtDeviceAbortRegCallBack(ProcessTaskAbortHandleCallback, static_cast<void*>(this), name);
}

HcclTaskAbortHandler::~HcclTaskAbortHandler()
{
    std::string name = "HCOMM";
    Hccl::HrtDeviceAbortRegCallBack(nullptr, nullptr, name);
}

HcclResult HcclTaskAbortHandler::Register(CollComm* communicator)
{
    std::lock_guard<std::mutex> lock(vecMutex_);
    commVector_.push_back(communicator);
    HCCL_INFO("HcclTaskAbortHandler::Register success, commVector_ size is [%zu]", commVector_.size());

    return HCCL_SUCCESS;
}

HcclResult HcclTaskAbortHandler::UnRegister(CollComm* communicator)
{
    std::lock_guard<std::mutex> lock(vecMutex_);
    HCCL_INFO("HcclTaskAbortHandler::UnRegister Begin, commVector_ size is [%zu]", commVector_.size());
    auto it = std::find(commVector_.begin(), commVector_.end(), communicator);
    if (it != commVector_.end()) {
        commVector_.erase(it);
    } else {
        HCCL_WARNING("HcclTaskAbortHandler::UnRegister, comm not found.");
    }
    HCCL_INFO("HcclTaskAbortHandler::UnRegister finish, commVector_ size is [%zu]", commVector_.size());
    return HCCL_SUCCESS;
}
} // namespace hccl
