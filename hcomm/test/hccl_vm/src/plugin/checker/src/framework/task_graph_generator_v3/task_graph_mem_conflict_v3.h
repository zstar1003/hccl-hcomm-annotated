/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_MEM_CONFLICT_V3_H
#define CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_MEM_CONFLICT_V3_H

#include <cstddef>

#include "hccl_types.h"
#include "task_def_v3.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    struct MemConflictCheckStats {
        size_t nodeCount{0};
        size_t dataTaskNodeCount{0};
        size_t processedDataTaskNodeCount{0};
        size_t accessIntervalCount{0};
        size_t memoryBucketCount{0};
        size_t overlapCandidatePairCount{0};
        size_t overlapCandidateTaskNodeCount{0};
        size_t orderedCandidatePairCount{0};
        size_t parallelCandidatePairCount{0};
    };

    HcclResult CheckDataMoveTaskMemConflict(const TaskNode* start, MemConflictCheckStats* stats = nullptr);
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

#endif // CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_MEM_CONFLICT_V3_H
