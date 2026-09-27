/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter_error_manager.h"
#include "adapter_error_manager_pub.h"

ErrContextPub hrtErrMGetErrorContextPub(void)
{
    ErrContext errorContext = hrtErrMGetErrorContext();
    ErrContextPub errorContextPub;
    errorContextPub.work_stream_id = errorContext.work_stream_id;
    (void)memcpy_s(
        errorContextPub.reserved, sizeof(errorContextPub.reserved), errorContext.reserved,
        sizeof(errorContext.reserved));
    return errorContextPub;
}

void hrtErrMSetErrorContextPub(ErrContextPub errorContextPub)
{
    ErrContext errorContext;
    errorContext.work_stream_id = errorContextPub.work_stream_id;
    (void)memcpy_s(
        errorContext.reserved, sizeof(errorContext.reserved), errorContextPub.reserved,
        sizeof(errorContextPub.reserved));
    hrtErrMSetErrorContext(errorContext);
}
