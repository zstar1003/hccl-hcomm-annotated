/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_GLOBAL_MIRROR_TASKS_H
#define MC2_GLOBAL_MIRROR_TASKS_H

#include <array>
#include <vector>
#include <memory>
#include "dfx_common.h"
#include "task_info.h"

namespace Hccl {

class MC2GlobalMirrorTasks {
public:
    static MC2GlobalMirrorTasks& GetInstance();

    void Clear();
    void AddTaskInfo(u32 devLogicId, std::shared_ptr<TaskInfo> taskInfo);
    std::shared_ptr<TaskInfo> GetTaskInfo(u32 devLogicId, u8 dieId, u8 missionId, u32 instrId) const;

private:
    MC2GlobalMirrorTasks() = default;
    ~MC2GlobalMirrorTasks() = default;
    MC2GlobalMirrorTasks(const MC2GlobalMirrorTasks&) = delete;
    MC2GlobalMirrorTasks& operator=(const MC2GlobalMirrorTasks&) = delete;

private:
    std::array<std::vector<std::shared_ptr<TaskInfo>>, DEVICE_MAX_NUM> taskQueues_;
};

} // namespace Hccl

#endif // MC2_GLOBAL_MIRROR_TASKS_H
