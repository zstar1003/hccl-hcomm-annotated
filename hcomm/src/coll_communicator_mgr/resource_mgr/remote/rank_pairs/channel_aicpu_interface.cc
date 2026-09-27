/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel_aicpu_interface.h"
#include "framework/aicpu_hccl_process.h"
#include "channel_param.h"

extern "C" {
__attribute__((visibility("default"))) uint32_t RunAicpuIndOpChannelInit(void* args)
{
    CHK_PRT_RET(args == nullptr, HCCL_ERROR("[%s]args is null.", __func__), HCCL_E_PARA);
    struct InitTask {
        u64 context;
        bool isCustom;
    };
    InitTask* ctxArgs = reinterpret_cast<InitTask*>(args);
    HcclIndOpChannelRemoteResV3* commParam = reinterpret_cast<HcclIndOpChannelRemoteResV3*>(ctxArgs->context);
    CHK_PRT_RET(commParam == nullptr, HCCL_ERROR("[%s]commParam is null.", __func__), HCCL_E_PARA);
    return AicpuHcclProcess::AicpuIndOpChannelInit(commParam);
}
}
