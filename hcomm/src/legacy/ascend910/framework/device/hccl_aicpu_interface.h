/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __MC2_AICPU_INTERFACE_H__
#define __MC2_AICPU_INTERFACE_H__

#include <cstdint>
#include "hcomm_res_defs.h"
#include "hccl_group_utils.h"

struct ThreadNotifyRecordParam {
    char commName[COMM_NAME_MAX_LENGTH];
    ThreadHandle thread;
    ThreadHandle dstThread;
    uint32_t dstNotifyIdx;
    uint32_t dataType;
};
struct ThreadNotifyWaitParam {
    char commName[COMM_NAME_MAX_LENGTH];
    ThreadHandle thread;
    uint32_t notifyIdx;
    uint32_t dataType;
};

extern "C" {
__attribute__((visibility("default"))) uint32_t RunAicpuKfcResInitV2(void* args);
__attribute__((visibility("default"))) uint32_t RunAicpuRpcSrvLaunchV2(void* args);
__attribute__((visibility("default"))) uint32_t RunAicpuNotifyRecordAicpuKernel(void* args);
__attribute__((visibility("default"))) uint32_t RunAicpuNotifyWaitAicpuKernel(void* args);
}

#endif // __MC2_AICPU_INTERFACE_HPP__
