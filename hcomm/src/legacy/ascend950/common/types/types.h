/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_TYPES_H
#define HCCLV2_TYPES_H

#include <cstdint>
#include <unordered_map>
#include "hccl/base.h"
#include "enum_factory.h"

namespace Hccl {
using char_t = char;

using RankId = s32;
using QId = u32;
using DevId = u32;
using Timeout = u32;

using SockPort = u16;

// 将用户配置的HcclCommConfig.hcclOpExpansionMode -- int32_t 转为相应枚举
MAKE_ENUM(HcclAccelerator, DEFAULT, HOSTCPU_TS, AICPU_TS, AIV, AIV_ONLY, CCU_MS, CCU_SCHED, AICPU);

// 通信域粒度加速模式
MAKE_ENUM(AcceleratorState, CCU_MS, CCU_SCHED, CCU_FALLBACK, AIV, AIV_ONLY, AICPU_TS, HOSTCPU_TS, AICPU);

// AcceleratorState是通信域内部加速模式
struct OpExecuteConfig {
    AcceleratorState accState = AcceleratorState::CCU_MS;
};

struct AcceleratorEnumHash {
    template <typename T>
    std::size_t operator()(T t) const
    {
        return static_cast<std::size_t>(t);
    }
};

const std::unordered_map<AcceleratorState, std::string, AcceleratorEnumHash> AcceleratorStateToString
    = {{AcceleratorState::CCU_MS, "CCU_MS"},
       {AcceleratorState::CCU_SCHED, "CCU_SCHED"},
       {AcceleratorState::CCU_FALLBACK, "CCU_FALLBACK"},
       {AcceleratorState::AIV, "AIV"},
       {AcceleratorState::AIV_ONLY, "AIV_ONLY"},
       {AcceleratorState::AICPU_TS, "AICPU_TS"},
       {AcceleratorState::HOSTCPU_TS, "HOSTCPU_TS"},
       {AcceleratorState::AICPU, "AICPU"}};

MAKE_ENUM(
    HcclOpType, HCCL_INVALID, HCCL_BROADCAST, HCCL_ALLREDUCE, HCCL_REDUCE, HCCL_SEND, HCCL_RECEIVE, HCCL_ALLGATHER,
    HCCL_REDUCE_SCATTER, HCCL_ALLTOALLV, HCCL_ALLTOALLVC, HCCL_ALLTOALL, HCCL_GATHER, HCCL_SCATTER,
    HCCL_BATCH_SEND_RECV, HCCL_BATCH_PUT, HCCL_BATCH_GET, HCCL_ALLGATHER_V, HCCL_REDUCE_SCATTER_V, HCCL_BATCH_WRITE,
    HCCL_HALF_ALLTOALLV, HCCL_ALL, HCCL_FINALIZE = 100, HCCL_INTER_GROUP_SYNC, HCCL_INIT, HCCL_BARRIER, HCCL_MAX);

} // namespace Hccl

#endif // HCCLV2_TYPES_H
