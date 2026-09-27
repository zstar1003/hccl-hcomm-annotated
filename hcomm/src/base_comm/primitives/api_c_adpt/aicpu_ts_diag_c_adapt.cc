/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_diag.h"
#include "device/framework/aicpu_hccl_process.h"

HcclResult HcommRegOpInfo(const char* commId, void* opInfo, size_t size)
{
    CHK_PTR_NULL(commId);
    CHK_PTR_NULL(opInfo);
    CHK_RET(AicpuHcclProcess::AicpuRegOpInfo(opInfo, size));
    HCCL_INFO("%s success, commId[%s], opInfo[%p], size[%u]", __func__, commId, opInfo, size);
    return HCCL_SUCCESS;
}

HcclResult HcommRegOpTaskException(const char* commId, HcommGetOpInfoCallback callback)
{
    CHK_PTR_NULL(commId);
    CHK_PTR_NULL(callback);
    CHK_RET(AicpuHcclProcess::AicpuRegOpTaskException(callback));
    HCCL_INFO("%s success, commId[%s]", __func__, commId);
    return HCCL_SUCCESS;
}
