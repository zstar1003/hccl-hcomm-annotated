/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dfx_profiling_command_handle_lite.h"
#include "log.h"
#include "prof_common.h"
#include "dfx_profiling_handler_lite.h"

namespace Hccl {

HcclResult DfxRegisterProfCallBack()
{
    if (MsprofRegisterCallback != nullptr) {
        HCCL_INFO("RegisterProfCallBack not null");
        int32_t ret = MsprofRegisterCallback(AICPU, &DfxDeviceCommandHandle);
        CHK_PRT_RET((ret != 0), HCCL_ERROR("[%s] failed. ret = [%d]", __func__, ret), HCCL_E_PARA);
    } else {
        HCCL_INFO("RegisterProfCallBack is null");
    }
    return HCCL_SUCCESS;
}

int32_t DfxDeviceCommandHandle([[maybe_unused]] uint32_t profType, void* data, uint32_t len)
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
    if (type == PROF_COMMANDHANDLE_TYPE_START) {
        if ((ADPROF_TASK_TIME_L0 & command->profSwitch) != 0) {
            DfxProfilingHandlerLite::GetInstance().SetProL0On(true);
        }
        if ((ADPROF_TASK_TIME_L1 & command->profSwitch) != 0) {
            DfxProfilingHandlerLite::GetInstance().SetProL1On(true);
        }
    } else if (type == PROF_COMMANDHANDLE_TYPE_STOP) {
        DfxProfilingHandlerLite::GetInstance().SetProL0On(false);
        DfxProfilingHandlerLite::GetInstance().SetProL1On(false);
    }
    return PROF_SUCCESS;
}

} // namespace Hccl
