/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TASK_PARAM_H
#define HCCL_TASK_PARAM_H

#include <stdint.h>
#include <hccl/hccl_types.h>
// 参照 2.0 task_params.h 定义, 待补充 ccu
typedef enum {
    HCCL_TASK_TYPE_SDMA = 0,
    HCCL_TASK_TYPE_RDMA = 1,
    HCCL_TASK_TYPE_REDUCE_INLINE = 2,
    HCCL_TASK_TYPE_NOTIFY_RECORD = 3,
    HCCL_TASK_TYPE_NOTIFY_WAIT = 4,
    HCCL_TASK_RTYPE_RESERVED
} HcclTaskType;

typedef enum {
    HCCL_DMA_OP_READ = 0,
    HCCL_DMA_OP_WRITE = 1,
    HCCL_DMA_OP_NOTIFY_WAIT = 2,
    HCCL_DMA_OP_RESERVED
} HcclDmaOp;

typedef enum {
    HCCL_DFX_LINK_TYPE_ONCHIP = 0,
    HCCL_DFX_LINK_TYPE_HCCS = 1,
    HCCL_DFX_LINK_TYPE_PCIE = 2,
    HCCL_DFX_LINK_TYPE_ROCE = 3,
    HCCL_DFX_LINK_TYPE_SIO = 4,
    HCCL_DFX_LINK_TYPE_HCCS_SW = 5,
    HCCL_DFX_LINK_TYPE_STANDARD_ROCE = 6,
    HCCL_DFX_LINK_TYPE_RESERVED
} HcclDfxLinkType;

typedef struct {
    const void* src;
    const void* dst;
    uint64_t size;
    uint64_t notifyID;
    HcclDfxLinkType linkType;
    HcclDmaOp dmaOp;
} HcclDfxParaDMA;

typedef struct {
    const void* src;
    const void* dst;
    uint64_t size;
    uint64_t notifyID;
    HcclDfxLinkType linkType;
    HcclReduceOp reduceOp;
    HcclDataType dataType;
} HcclDfxParaReduce;

typedef struct {
    uint64_t notifyID;
    uint32_t value;
} HcclDfxParaNotify;

typedef struct {
    HcclTaskType taskType;
    uint64_t beginTime;
    uint64_t endTime;
    union {
        HcclDfxParaDMA DMA;
        HcclDfxParaReduce Reduce;
        HcclDfxParaNotify Notify;
        // CCU待补充
    } taskPara;
} HcclTaskParam;

typedef void (*HcclDfxCallback)(uint32_t streamId, uint32_t taskId, HcclTaskParam* taskParam, void* args);

#endif
