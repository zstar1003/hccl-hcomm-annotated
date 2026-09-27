/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_COMMON_H
#define HCCLV2_CCU_COMMON_H

#include <map>
#include <set>
#include <queue>
#include <vector>
#include <memory>
#include <unordered_map>
#include <hccl/hccl_types.h>

#include "base.h"
#include "log.h"
#include "checker_data_slice.h"
#include "task_ccu.h"
#include "ccu_error_handler.h"

using namespace checker;
namespace Hccl {
void PrintCcuGraph(TaskNodePtr dummyStart);
void PrintGraphOneNode(TaskNodePtr curNode);
void PrintGraphRevamp(TaskNodePtr head);
void PrintGraphRevampByTypes(TaskNodePtr head, std::set<TaskTypeStub> types);
void PrintGraphRevampByQueue(TaskNodePtr head, std::set<uint32_t> queList);
HcclResult GetPeerRankByTaskNode(TaskNodePtr currNode, RankId& peerRank);

HcclResult CollectBilateralWaitInfo(TaskStubCcuGraph* curCcuTask, uint32_t queId, TaskNodePtr node);
HcclResult
CollectBilateralPostInfo(TaskStubCcuGraph* curCcuTask, uint32_t queId, TaskNodePtr node, bool isLast = false);

HcclResult AppendTailNode(TaskStubCcuGraph* curCcuTask, uint32_t queId, TaskNodePtr node);

void AddLocalCopy(
    uint32_t rankId, uint32_t queId, TaskStubCcuGraph* curCcuTask, const checker::DataSlice& srcSlice,
    const checker::DataSlice& dstSlice);
void AddLocalReduce(
    uint32_t rankId, uint32_t queId, TaskStubCcuGraph* curCcuTask, const checker::DataSlice& srcSlice,
    const checker::DataSlice& dstSlice, CheckerDataType checkerDataType, CheckerReduceOp checkerReduceOp);
void AddLocalBatchReduce(
    uint32_t rankId, uint32_t queId, TaskStubCcuGraph* curCcuTask, const std::vector<checker::DataSlice>& srcSlices,
    const checker::DataSlice& dstSlice, DataType hcclDataType);
void AddWrite(
    uint32_t rankId, uint32_t rmtRankId, uint32_t queId, TaskStubCcuGraph* curCcuTask,
    const checker::DataSlice& srcSlice, const checker::DataSlice& dstSlice);
void AddRead(
    uint32_t rankId, uint32_t rmtRankId, uint32_t queId, TaskStubCcuGraph* curCcuTask,
    const checker::DataSlice& srcSlice, const checker::DataSlice& dstSlice);
void AddWriteReduce(
    uint32_t rankId, uint32_t rmtRankId, uint32_t queId, TaskStubCcuGraph* curCcuTask,
    const checker::DataSlice& srcSlice, const checker::DataSlice& dstSlice, CheckerDataType checkerDataType,
    CheckerReduceOp checkerReduceOp);
void AddReadReduce(
    uint32_t rankId, uint32_t rmtRankId, uint32_t queId, TaskStubCcuGraph* curCcuTask,
    const checker::DataSlice& srcSlice, const checker::DataSlice& dstSlice, CheckerDataType checkerDataType,
    CheckerReduceOp checkerReduceOp);
TaskNodePtr AddLoopStartTask(uint32_t queId, uint32_t loopIdx, uint32_t loopGroupIdx, TaskStubCcuGraph* curCcuTask);
TaskNodePtr AddLoopEndTask(uint32_t queId, uint32_t loopIdx, uint32_t loopGroupIdx, TaskStubCcuGraph* curCcuTask);
void AddPost(
    uint32_t rankId, uint32_t rmtRankId, uint32_t queId, TaskStubCcuGraph* curCcuTask, uint32_t rmtDieId,
    uint16_t rmtCKEId, uint16_t setRmtCKEMask);
TaskNodePtr AddWait(uint32_t rankId, uint32_t queId, TaskStubCcuGraph* curCcuTask, uint16_t remoteWaitMask);
void AddLocalPost(
    uint32_t rankId, uint32_t queId, uint32_t dieId, TaskStubCcuGraph* curCcuTask, uint16_t setCKEId,
    uint16_t setCKEMask, bool invalidPost = false);
TaskNodePtr AddLocalWait(uint32_t rankId, uint32_t queId, TaskStubCcuGraph* curCcuTask, uint16_t localWaitMask);
void AddCcuSubGraphEnd(TaskNodePtr node, TaskStubCcuGraph* curCcuTask);
void AddNodeRelation(TaskNodePtr parent, TaskNodePtr child);
} // namespace Hccl

#endif // HCCLV2_CCU_COMMON_H
