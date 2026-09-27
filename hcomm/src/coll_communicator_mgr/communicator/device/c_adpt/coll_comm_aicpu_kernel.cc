/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_aicpu_kernel.h"
#include "aicpu_init_param.h"
#include "coll_comm_aicpu_mgr.h"
#include "framework/aicpu_hccl_process.h"
#include "log.h"

extern "C" {
__attribute__((visibility("default"))) uint32_t RunAicpuCommInit(void* args)
{
    CHK_PRT_RET(args == nullptr, HCCL_ERROR("[%s]args is null.", __func__), HCCL_E_PARA);

    CommAicpuParam* commAicpuParam = static_cast<CommAicpuParam*>(args);
    DevType devType = static_cast<DevType>(commAicpuParam->deviceType);
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        HCCL_INFO(
            "[RunAicpuCommInit] group[%s], deviceLogicId[%u], devicePhyId[%u], deviceType[%u]", commAicpuParam->hcomId,
            commAicpuParam->deviceLogicId, commAicpuParam->devicePhyId, commAicpuParam->deviceType);
        return CollCommAicpuMgr::GetInstance().InitComm(commAicpuParam);
    }
    return AicpuHcclProcess::AicpuIndOpCommInit(commAicpuParam);
}

__attribute__((visibility("default"))) uint32_t RunAicpuDfxInitV2(void* args)
{
    HCCL_RUN_INFO("RunAicpuDfxInitV2 start.");
    CHK_PRT_RET(args == nullptr, HCCL_ERROR("[%s]args is null.", __func__), HCCL_E_PTR);
    struct InitTask {
        u64 context;
        char commTag[256];
    };
    InitTask* ctxArgs = static_cast<InitTask*>(args);
    CHK_PRT_RET(ctxArgs == nullptr, HCCL_ERROR("[%s]ctxArgs is null.", __func__), HCCL_E_PTR);
    HcclDfxOpInfo* dfxOpInfo = reinterpret_cast<HcclDfxOpInfo*>(ctxArgs->context);
    CollCommAicpu* currentComm = CollCommAicpuMgr::GetInstance().GetCurrentComm();
    CHK_PTR_NULL(currentComm);
    return currentComm->InitDfxOpInfo(dfxOpInfo);
}
}
