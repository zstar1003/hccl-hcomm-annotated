/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "aicpu_ts_urma_channel_kernel.h"
#include "channel_param.h"
#include "coll_comm_aicpu_kernel_adpt.h"
#include "aicpu_channel_process.h"

extern "C" {
__attribute__((visibility("default"))) uint32_t RunAicpuIndOpChannelInitV2(void* args)
{
    HCCL_RUN_INFO("RunAicpuIndOpChannelInitV2 start.");
    CHK_PTR_NULL(args);
    uint64_t devAddr = *reinterpret_cast<uint64_t*>(args);
    HcclChannelUrmaRes* commParam = reinterpret_cast<HcclChannelUrmaRes*>(devAddr);
    CHK_PTR_NULL(commParam);
    return CollCommAicpuKernelAdptInitChannel(commParam);
}

__attribute__((visibility("default"))) uint32_t RunAicpuChannelInitV2(void* args)
{
    HCCL_RUN_INFO("RunAicpuIndOpChannelInitV2Internal start.");
    CHK_PTR_NULL(args);
    uint64_t devAddr = *reinterpret_cast<uint64_t*>(args);
    HcclChannelUrmaRes* commParam = reinterpret_cast<HcclChannelUrmaRes*>(devAddr);
    CHK_PTR_NULL(commParam);
    return AicpuChannelProcess::AicpuChannelInit(commParam);
}

__attribute__((visibility("default"))) uint32_t RunAicpuChannelInitV3(void* args)
{
    HCCL_RUN_INFO("RunAicpuChannelInitV3 start.");
    CHK_PTR_NULL(args);
    uint64_t devAddr = *reinterpret_cast<uint64_t*>(args);
    HcommChannelRes* commParam = reinterpret_cast<HcommChannelRes*>(devAddr);
    return AicpuChannelProcess::InitHcommChannelRes(commParam);
}

__attribute__((visibility("default"))) uint32_t RunAicpuChannelDestroyV2(void* args)
{
    HCCL_RUN_INFO("RunAicpuIndOpChannelDestroyV2Internal start.");
    CHK_PTR_NULL(args);
    uint64_t devAddr = *reinterpret_cast<uint64_t*>(args);
    HcclChannelUrmaRes* commParam = reinterpret_cast<HcclChannelUrmaRes*>(devAddr);
    CHK_PTR_NULL(commParam);
    return AicpuChannelProcess::AicpuChannelDestroy(commParam);
}

__attribute__((visibility("default"))) uint32_t RunAicpuIndOpChannelUpdateV2(void* args)
{
    HCCL_RUN_INFO("RunAicpuIndOpChannelUpdateV2 start.");
    CHK_PTR_NULL(args);
    uint64_t devAddr = *reinterpret_cast<uint64_t*>(args);
    HcclChannelUrmaRes* commParam = reinterpret_cast<HcclChannelUrmaRes*>(devAddr);
    CHK_PTR_NULL(commParam);
    return CollCommAicpuKernelAdptUpdateChannel(commParam);
}
}
