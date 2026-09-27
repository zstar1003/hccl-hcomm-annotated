/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascend_hal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

DLLEXPORT hdcError_t drvHdcInit(enum drvHdcServiceType serviceType) { return DRV_ERROR_NONE; }

DLLEXPORT hdcError_t drvHdcUninit(void) {}

DLLEXPORT hdcError_t drvHdcSendFile(
    int peer_node, int peer_devid, const char* file, const char* dst_path,
    void (*progress_notifier)(struct drvHdcProgInfo*))
{
    return DRV_ERROR_NONE;
};
