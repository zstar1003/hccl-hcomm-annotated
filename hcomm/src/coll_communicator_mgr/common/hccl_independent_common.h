/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INDEPENDENT_COMMON_H
#define HCCL_INDEPENDENT_COMMON_H

#include "hccl/hccl_res.h"
#include "hccl_common.h"
#include "hcomm_res_defs.h"
#include "hccl_mem_defs.h"
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef enum {
    RANK_GRAPH_RESERVED = -1,
    RANK_GRAPH_910_93 = 0,
    RANK_GRAPH_910_95 = 1,
} GraphType;

typedef enum {
    NOTIFY_TYPE_RESERVED = -1,
    NOTIFY_TYPE_RTS_NOTIFY = 0,
    NOTIFY_TYPE_RTS_EVENT = 1,
    NOTIFY_TYPE_DEVICE_MEM = 2,
} NotifyType;

typedef uint64_t NotifyHandle;

typedef union {
    struct {
        uint64_t requireShare : 1;
        uint64_t rsvd : 63;
    };
    uint64_t value;
} HcclRegMemAttr;

typedef struct {
    HcclMemType type;
    void* addr;
    u64 size;
} CommBuffer;

extern HcclResult HcclGetNotifyNumInThread(HcclComm comm, ThreadHandle thread, CommEngine engine, uint32_t* notifyNum);

constexpr u32 NOTIFY_MAX_NUM = 2048;
inline bool IsValidCommEngine(CommEngine engine)
{
    switch (engine) {
        case COMM_ENGINE_CPU:
        case COMM_ENGINE_CPU_TS:
        case COMM_ENGINE_AICPU:
        case COMM_ENGINE_AICPU_TS:
        case COMM_ENGINE_AIV:
        case COMM_ENGINE_CCU:
            return true;
        default:
            return false;
    }
}

inline bool IsValidNotify(NotifyType notifyType)
{
    switch (notifyType) {
        case NOTIFY_TYPE_RESERVED:
        case NOTIFY_TYPE_RTS_NOTIFY:
        case NOTIFY_TYPE_RTS_EVENT:
        case NOTIFY_TYPE_DEVICE_MEM:
            return true;
        default:
            return false;
    }
}

#ifdef __cplusplus
}
#endif // __cplusplus

#endif
