/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "user_log.h"
#include "config_log.h"
#include "ra_async.h"
#include "ra_rs_comm.h"
#include "ra_hdc_async.h"
#include "hccp_async.h"

HCCP_ATTRI_VISI_DEF int RaGetAsyncReqResult(void *reqHandle, int *reqResult)
{
    struct RaRequestHandle *reqHandleTmp = NULL;

    CHK_PRT_RETURN(reqHandle == NULL || reqResult == NULL, hccp_err("[get][async]req_handle or req_result is NULL"),
        ConverReturnCode(OTHERS, -EINVAL));

    reqHandleTmp = (struct RaRequestHandle *)reqHandle;
    if (!reqHandleTmp->isDone) {
        return ConverReturnCode(OTHERS, -EAGAIN);
    }

    *reqResult = ConverReturnCode(reqHandleTmp->opHandle->opModule, reqHandleTmp->opRet);
    HdcAsyncDelResponse(reqHandleTmp);
    return 0;
}
