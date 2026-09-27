/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_SYNC_CONFLICT_V3_H
#define CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_SYNC_CONFLICT_V3_H

#include <cstddef>

#include "hccl_types.h"
#include "task_def_v3.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    struct SyncConflictCheckStats {
        size_t originalNodeCount{0};
        size_t copiedNodeCount{0};
        size_t copiedEdgeCount{0};
        size_t resourceBucketCount{0};
        size_t pairCount{0};
        size_t checkedBucketCount{0};
        size_t conflictCount{0};
    };

    HcclResult CheckSyncResourceConflict(const TaskNode* start, SyncConflictCheckStats* stats = nullptr);
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

#endif // CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_SYNC_CONFLICT_V3_H
