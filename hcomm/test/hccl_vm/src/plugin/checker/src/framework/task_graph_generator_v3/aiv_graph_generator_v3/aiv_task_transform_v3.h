/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_TASK_GRAPH_GENERATOR_V3_AIV_TASK_TRANSFORM_V3_H
#define CHECKER_TASK_GRAPH_GENERATOR_V3_AIV_TASK_TRANSFORM_V3_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../task_graph_generator_v3.h"
#include "storage_manager.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {

    struct AivGraphsGenerateInputV3 {
        TaskGraphGeneratorV3* graph{nullptr};
        std::vector<TaskAivGraph*> aivGraphs;
        StorageManager* storage{nullptr};
    };

    struct AivGraphGenerateOutputV3 {
        size_t internalNodeCount{0};
        size_t setWaitEdgeCount{0};
        size_t pipeBarrierMergeCount{0};
        size_t syncAllMergeCount{0};
        size_t sendRecvEdgeCount{0};
        size_t taskJsonTotalTaskCount{0};
        size_t dagNodeCountBeforeCpGmMerge{0};
        size_t dagNodeCountAfterCpGmMerge{0};
        size_t cpGmLoopMergeCount{0};
        size_t cpGmMergedIterationCount{0};
        size_t cpGmMergedOriginalNodeCount{0};
        size_t cpGmGeneratedNodeCount{0};
        size_t cpGmInactiveNodeCount{0};
        uint64_t expandNs{0};
    };

    HcclResult ExpandAivGraphsV3(const AivGraphsGenerateInputV3& input, AivGraphGenerateOutputV3& output);

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

#endif // CHECKER_TASK_GRAPH_GENERATOR_V3_AIV_TASK_TRANSFORM_V3_H
