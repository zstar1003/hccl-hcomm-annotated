/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "profiling_manager_device.h"
#include "profiling_command_handle.h"
namespace hccl {
int32_t DeviceCommandHandle([[maybe_unused]] uint32_t profType, void* data, uint32_t len)
{
    HCCL_INFO("[%s] start", __func__);
    (void)len;
    if (data == nullptr) {
        HCCL_ERROR("[%s] CommandHandle's data is NULL.", __func__);
        return PROF_FAILED;
    }
    MsprofCommandHandle* command = reinterpret_cast<MsprofCommandHandle*>(data);
    auto type = command->type;
    HCCL_INFO("[%s] type = [%u]. CommandHandle_switch = [%llu]", __func__, type, command->profSwitch);
    // 目前只会有两种状态 开启或者关闭
    if (type == PROF_COMMANDHANDLE_TYPE_START) {
        if ((ADPROF_TASK_TIME_L0 & command->profSwitch) != 0) {
            dfx::ProfilingManager::SetProL0On(true);
        }
        if ((ADPROF_TASK_TIME_L1 & command->profSwitch) != 0) {
            dfx::ProfilingManager::SetProL1On(true);
        }
    } else if (type == PROF_COMMANDHANDLE_TYPE_STOP) {
        dfx::ProfilingManager::SetProL0On(false);
        dfx::ProfilingManager::SetProL1On(false);
    }
    return PROF_SUCCESS;
}
} // namespace hccl
