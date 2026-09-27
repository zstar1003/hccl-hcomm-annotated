/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PROFILER_BASE_PUB_EXTEND_H
#define PROFILER_BASE_PUB_EXTEND_H

#include <memory>
#include <mutex>
#include <string>
#include <map>
#include <thread>

#include <hccl/hccl_types.h>
#include "hccl_common.h"
#include "common.h"
#include "workflow_pub.h"
#include "dispatcher_task_types.h"
#include "alg_profiling.h"

#define HCCL_PROFILER_ADD_STREAM_BY_STREAMID(streamId, tag, planeID, algType)
#define HCCL_PROFILER_DEL_STREAM_BY_STREAMID(streamId)
#define HCCL_PROFILER_ADD_STREAM(stream, tag, planeID, algType)
#define HCCL_PROFILER_DEL_STREAM(stream)
#define HCCL_PROFILER_ADD_TAG(tag, group, workFlowMode)
#define HCCL_PROFILER_ADD_TAG_AIV(tag, group, workFlowMode)
#define HCCL_PROFILER_ADD_TAG_SENDRECV(tag, group, workFlowMode)
#define HCCL_PROFILER_DEL_TAG(tag)

// 兼容性考虑，需保留
#define HCCL_PROFILER_ADD_OPDATA(tag, count, src, dst, dataType, rootId, group)                    \
    do {                                                                                           \
        HcclResult __ret = ProfilerBase::AddOpData(tag, count, src, dst, dataType, rootId, group); \
        if (UNLIKELY(__ret != 0)) {                                                                \
            HCCL_ERROR("profiler add opData error[%d]", __ret);                                    \
            return HCCL_E_INTERNAL;                                                                \
        }                                                                                          \
    } while (0)

#define HCCL_PROFILER_ADD_OPDATA_OP(tag, count, src, dst, dataType, rootId, group, reduceType)
#define HCCL_PROFILER_DEL_OPDATA(tag)
#define HCCL_PROFILER_ADD_GROUPRANK(group, rankSize, rankId)
#define HCCL_PROFILER_ADD_GROUPRANK_SENDRECV(group, rankSize, rankId, remoteRankId)
#define HCCL_PROFILER_DEL_GROUPRANK(group)
#define HCCL_PROFILER_ADD_GROUP_UDI(group, udi)
#define HCCL_PROFILER_DEL_GROUP_UDI(group)

#endif /* PROFILER_BASE_PUB_EXTEND_H */
