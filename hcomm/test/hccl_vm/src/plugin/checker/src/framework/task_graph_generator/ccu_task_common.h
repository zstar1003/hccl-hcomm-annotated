/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_COMMON_H
#define HCCLV2_CCU_COMMON_H

#include <hccl_types.h>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <unordered_map>
#include <vector>

#include "base.h"
#include "data_type.h"
#include "log.h"
#include "task_ccu.h"

namespace HcclSim {
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
    uint32_t rankId, uint32_t queId, TaskStubCcuGraph* curCcuTask, const DataSlice& srcSlice,
    const DataSlice& dstSlice);
void AddLocalReduce(
    uint32_t rankId, uint32_t queId, TaskStubCcuGraph* curCcuTask, const DataSlice& srcSlice, const DataSlice& dstSlice,
    HcclDataType checkerDataType, HcclReduceOp checkerReduceOp);
void AddLocalBatchReduce(
    uint32_t rankId, uint32_t queId, TaskStubCcuGraph* curCcuTask, const std::vector<DataSlice>& srcSlices,
    const DataSlice& dstSlice, DataType hcclDataType);
void AddWrite(
    uint32_t rankId, uint32_t rmtRankId, uint32_t queId, TaskStubCcuGraph* curCcuTask, const DataSlice& srcSlice,
    const DataSlice& dstSlice);
void AddRead(
    uint32_t rankId, uint32_t rmtRankId, uint32_t queId, TaskStubCcuGraph* curCcuTask, const DataSlice& srcSlice,
    const DataSlice& dstSlice);
void AddWriteReduce(
    uint32_t rankId, uint32_t rmtRankId, uint32_t queId, TaskStubCcuGraph* curCcuTask, const DataSlice& srcSlice,
    const DataSlice& dstSlice, HcclDataType checkerDataType, HcclReduceOp checkerReduceOp);
void AddReadReduce(
    uint32_t rankId, uint32_t rmtRankId, uint32_t queId, TaskStubCcuGraph* curCcuTask, const DataSlice& srcSlice,
    const DataSlice& dstSlice, HcclDataType checkerDataType, HcclReduceOp checkerReduceOp);
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

HcclResult ClearWaitMask(RankId rankId, uint32_t dieId, uint16_t waitCKEId, uint16_t waitCKEMask);
HcclResult ProcessSetMask(
    RankId rankId, uint32_t dieId, TaskStubCcuGraph* curCcuTask, uint32_t queId, uint16_t setCKEId, uint16_t setCKEMask,
    bool invalidPost = false);

uint16_t UpdateId(uint16_t CKEId, uint32_t curLoopIdx, uint32_t expandOffset, uint32_t ckOffset, uint32_t curExpandCnt);
HcclResult GenSliceFromMs(uint16_t msId, uint64_t len, DataSlice& slice);
} // namespace HcclSim

#endif // HCCLV2_CCU_COMMON_H
