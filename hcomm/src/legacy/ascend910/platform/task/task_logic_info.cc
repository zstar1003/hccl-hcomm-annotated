/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_logic_info.h"

namespace hccl {
TaskLogicInfo::TaskLogicInfo(u32 index, TaskLogicType taskLogicType, TaskLogicFuncType funcType)
    : taskLogicCmd{taskLogicType, index},
      taskFuncType(funcType)
{}

TaskLogicInfo::TaskLogicInfo(
    u32 index, TaskLogicType taskLogicType, TaskLogicFuncType funcType, std::vector<TxMemoryInfo>& txMems)
    : taskLogicCmd{taskLogicType, index},
      taskFuncType(funcType),
      txAsync{txMems}
{}

TaskLogicInfo::TaskLogicInfo(
    u32 index, TaskLogicType taskLogicType, TaskLogicFuncType funcType, std::vector<RxMemoryInfo>& rxMems)
    : taskLogicCmd{taskLogicType, index},
      taskFuncType(funcType),
      rxAsync{rxMems}
{}

TaskLogicInfo::TaskLogicInfo(
    u32 index, TaskLogicType taskLogicType, TaskLogicFuncType funcType, void* signal, u32 userRank, u32 remoteUserRank,
    s32 stage)
    : taskLogicCmd{taskLogicType, index},
      taskFuncType(funcType)
{
    taskLogicPara.dispatcherTaskLogicPara.signalWait.signal = signal;
    taskLogicPara.dispatcherTaskLogicPara.signalWait.userRank = userRank;
    taskLogicPara.dispatcherTaskLogicPara.signalWait.remoteRank = remoteUserRank;
    taskLogicPara.dispatcherTaskLogicPara.signalWait.stage = stage;
}

TaskLogicInfo::TaskLogicInfo(
    u32 index, TaskLogicType taskLogicType, TaskLogicFuncType funcType, void* signal, u32 userRank, u64 offset,
    s32 stage)
    : taskLogicCmd{taskLogicType, index},
      taskFuncType(funcType)
{
    taskLogicPara.dispatcherTaskLogicPara.signalRecord.signal = signal;
    taskLogicPara.dispatcherTaskLogicPara.signalRecord.userRank = userRank;
    taskLogicPara.dispatcherTaskLogicPara.signalRecord.offset = offset;
    taskLogicPara.dispatcherTaskLogicPara.signalRecord.stage = stage;
}

TaskLogicInfo::TaskLogicInfo(
    u32 index, TaskLogicType taskLogicType, TaskLogicFuncType funcType, void* dst, uint64_t destMax, void* src,
    u64 count, HcclRtMemcpyKind kind)
    : taskLogicCmd{taskLogicType, index},
      taskFuncType(funcType)
{
    taskLogicPara.dispatcherTaskLogicPara.memAsync.dst = dst;
    taskLogicPara.dispatcherTaskLogicPara.memAsync.destMax = destMax;
    taskLogicPara.dispatcherTaskLogicPara.memAsync.src = src;
    taskLogicPara.dispatcherTaskLogicPara.memAsync.count = count;
    taskLogicPara.dispatcherTaskLogicPara.memAsync.kind = kind;
}
} // namespace hccl
