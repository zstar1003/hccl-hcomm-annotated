/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV1_SINGLETASK_CHECK_H
#define HCCLV1_SINGLETASK_CHECK_H

#include <vector>

#include "llt_common.h"
#include "checker_data_slice.h"
#include "task_def.h"
#include <queue>

using namespace std;

namespace checker {

class SingleTaskCheck {
public:
    HcclResult CheckSlaveTaskQueue();
    HcclResult CheckTaskMem(TaskNodePtr dummyStart);

private:
    HcclResult CheckSingleSlice(RankId taskRank, u32 queueId, u32 taskId, const DataSlice& slice, RankId sliceRank);
    HcclResult
    CheckTwoSliceOverlap(RankId rank, u32 queueId, u32 taskId, const DataSlice& sliceA, const DataSlice& sliceB);

    HcclResult CheckSingleTaskMem(TaskNodePtr curTask);
    HcclResult CheckSingleCcuTaskMem(TaskNodePtr curTask);
    void AddChildrenToQueue(
        TaskNode* node, std::set<TaskNode*>& visitedNodes, std::queue<TaskNode*>& walkQue,
        std::set<TaskNode*>& simulatedNodes);
};
} // namespace checker

#endif
