/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_DEVICE_PROFILING_H
#define HCOMM_DEVICE_PROFILING_H

#include <stdint.h>
#include <securec.h>
#include <arpa/inet.h>
#include "acl/acl_rt.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
constexpr u32 MAX_LENGTH = 128;

typedef struct HcomProInfo {
    uint8_t dataType;
    uint8_t cmdType;
    uint64_t dataCount;
    uint32_t rankSize;
    uint32_t userRank;
    uint32_t blockDim = 0;
    uint64_t beginTime;
    uint32_t root;
    uint32_t slaveThreadNum;
    uint64_t commNameLen;
    uint64_t algTypeLen;
    char tag[MAX_LENGTH];
    char commName[MAX_LENGTH];
    char algType[MAX_LENGTH];
    bool isCapture = false;
    bool isAiv = false;
    uint8_t reserved[MAX_LENGTH];
} HcomProInfo;

typedef uint64_t ThreadHandle;

extern HcclResult HcommProfilingReportMainStreamAndFirstTask(ThreadHandle thread);

extern HcclResult HcommProfilingReportMainStreamAndLastTask(ThreadHandle thread);

// device侧的OP
extern HcclResult HcommProfilingReportDeviceHcclOpInfo(HcomProInfo profInfo);

extern HcclResult HcommProfilingInit(ThreadHandle* threads, u32 threadNum);

extern HcclResult HcommProfilingEnd(ThreadHandle* threads, u32 threadNum);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif
