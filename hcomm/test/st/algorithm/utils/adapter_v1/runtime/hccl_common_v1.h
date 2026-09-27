/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMMON_V1_H
#define HCCL_COMMON_V1_H

#include "dtype_common.h"

typedef struct {
    uint32_t addrOffset;
    uint32_t dataOffset;
} rtPlaceHolderInfo_t;

typedef enum tagRtStreamCaptureStatus {
    RT_STREAM_CAPTURE_STATUS_NONE = 0,
    RT_STREAM_CAPTURE_STATUS_ACTIVE = 1,
    RT_STREAM_CAPTURE_STATUS_INVALIDATED = 2,
    RT_STREAM_CAPTURE_STATUS_MAX
} rtStreamCaptureStatus;

#endif
