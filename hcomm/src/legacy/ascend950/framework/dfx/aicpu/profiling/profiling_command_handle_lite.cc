/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "profiling_command_handle_lite.h"
#include "log.h"
#include "profiling_handler_lite.h"

namespace Hccl {
#ifdef CCL_KERNEL_AICPU

void RegisterProfCallBack()
{
    if (MsprofRegisterCallback != nullptr) {
        HCCL_INFO("RegisterProfCallBack not null");
        int32_t ret = MsprofRegisterCallback(AICPU, &DeviceCommandHandle);
        if (ret != 0) {
            THROW<InternalException>(
                StringFormat("CommunicatorImplLite::MsprofRegisterCallback failed, ret = %d", ret));
        }
    } else {
        HCCL_INFO("RegisterProfCallBack is null");
    }
}

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
            ProfilingHandlerLite::GetInstance().SetProL0On(true);
        }
        if ((ADPROF_TASK_TIME_L1 & command->profSwitch) != 0) {
            ProfilingHandlerLite::GetInstance().SetProL1On(true);
        }
    } else if (type == PROF_COMMANDHANDLE_TYPE_STOP) {
        ProfilingHandlerLite::GetInstance().SetProL0On(false);
        ProfilingHandlerLite::GetInstance().SetProL1On(false);
    }
    return PROF_SUCCESS;
}
#endif

} // namespace Hccl
