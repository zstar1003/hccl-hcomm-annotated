/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl/hccl_comm.h"
#include "coll_comm_aicpu_mgr.h"

using namespace hccl;

HcclResult HcclCommGetStatus(const char* commId, HcclCommStatus* status)
{
    CHK_PTR_NULL(commId);
    CHK_PTR_NULL(status);
    *status = HcclCommStatus::HCCL_COMM_STATUS_READY;
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType == DevType::DEV_TYPE_950 || deviceType == DevType::DEV_TYPE_960) {
        CollCommAicpu* collCommAicpu = CollCommAicpuMgr::GetInstance().GetCurrentComm(commId);
        CHK_PRT_RET(!collCommAicpu, HCCL_ERROR("%s AicpuGetComm is null, commId[%s]", __func__, commId), HCCL_E_PTR);
        *status = collCommAicpu->GetCommmStatus();
    } else if (deviceType == DevType::DEV_TYPE_910B) {
        HCCL_INFO("[%s] deviceType[%d] comm status ready", __func__, deviceType);
        *status = HcclCommStatus::HCCL_COMM_STATUS_READY;
    } else {
        HCCL_ERROR("[%s] deviceType[%d] is not support", __func__, deviceType);
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}
