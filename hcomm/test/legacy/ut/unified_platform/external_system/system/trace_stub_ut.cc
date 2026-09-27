/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "atrace_pub.h"
/**
 * @brief       Create trace handle.
 * @param [in]  tracerType:    trace type
 * @param [in]  objName:       object name
 * @param [in]  attr:          object attribute
 * @return      atrace handle
 */
TraHandle AtraceCreateWithAttr(TracerType tracerType, const char* objName, const TraceAttr* attr) { return 0; }

/**
 * @brief       Submite trace info
 * @param [in]  handle:    trace handle
 * @param [in]  buffer:    trace info buffer
 * @param [in]  bufSize:   size of buffer
 * @return      TraStatus
 */
TraStatus AtraceSubmit(TraHandle handle, const void* buffer, uint32_t bufSize)
{
    if (handle == 0) {
        return 0;
    } else if (handle == 1) {
        return -1;
    }
    return 0;
}

/**
 * @brief       Destroy trace handle
 * @param [in]  handle:    trace handle
 * @return      NA
 */
void AtraceDestroy(TraHandle handle) { return; }
