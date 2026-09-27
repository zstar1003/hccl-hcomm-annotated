/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mc2_global_mirror_tasks.h"
#include "internal_exception.h"
#include "exception_util.h"

namespace Hccl {

using namespace std;

MC2GlobalMirrorTasks& MC2GlobalMirrorTasks::GetInstance()
{
    static MC2GlobalMirrorTasks instance;
    return instance;
}

void MC2GlobalMirrorTasks::Clear()
{
    for (uint32_t i = 0; i < DEVICE_MAX_NUM; ++i) {
        taskQueues_[i].clear();
    }
}

void MC2GlobalMirrorTasks::AddTaskInfo(u32 devLogicId, shared_ptr<TaskInfo> taskInfo)
{
    if (devLogicId >= DEVICE_MAX_NUM) {
        THROW<InternalException>(StringFormat("MC2GlobalMirrorTasks::AddTaskInfo devId[%u] out of range", devLogicId));
    }

    if (taskInfo == nullptr) {
        THROW<InternalException>(StringFormat("MC2GlobalMirrorTasks::AddTaskInfo taskInfo is nullptr"));
    }

    const auto& ccuPara = taskInfo->taskParam_.taskPara.Ccu;
    if (GetTaskInfo(devLogicId, ccuPara.dieId, ccuPara.missionId, ccuPara.instrId) != nullptr) {
        return;
    }

    taskQueues_[devLogicId].push_back(taskInfo);

    HCCL_INFO(
        "[MC2GlobalMirrorTasks][%s] deviceId[%u], dieId[%u], missionId[%u], instrId[%u], executeId[%llu]", __func__,
        devLogicId, static_cast<u32>(ccuPara.dieId), static_cast<u32>(ccuPara.missionId), ccuPara.instrId,
        ccuPara.executeId);
}

shared_ptr<TaskInfo> MC2GlobalMirrorTasks::GetTaskInfo(u32 devLogicId, u8 dieId, u8 missionId, u32 instrId) const
{
    if (devLogicId >= DEVICE_MAX_NUM) {
        HCCL_ERROR("MC2GlobalMirrorTasks::GetTaskInfo devId[%u] out of range", devLogicId);
        return nullptr;
    }
    for (const auto& taskInfoPtr : taskQueues_[devLogicId]) {
        const auto& ccuPara = taskInfoPtr->taskParam_.taskPara.Ccu;
        if (ccuPara.dieId == dieId && ccuPara.missionId == missionId && ccuPara.instrId == instrId) {
            return taskInfoPtr;
        }
    }
    return nullptr;
}

} // namespace Hccl
