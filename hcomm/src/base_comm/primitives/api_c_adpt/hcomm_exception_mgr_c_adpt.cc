/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm/hcomm_exception.h"
#include "exception_callback_mgr.h"
#include "log.h"

int32_t HcommExceptionRegisterCallback(HcommExceptionCallback cb, void* userData)
{
    HcclResult ret = hcomm::ExceptionCallbackMgr::GetInstance().Register(cb, userData);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] register fail, ret[%d]", __func__, ret);
    }
    return static_cast<int32_t>(ret);
}

int32_t HcommExceptionUnregisterCallback(HcommExceptionCallback cb)
{
    HcclResult ret = hcomm::ExceptionCallbackMgr::GetInstance().Unregister(cb);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] unregister fail, ret[%d]", __func__, ret);
    }
    return static_cast<int32_t>(ret);
}
