/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_REACHABILITY_V3_H
#define CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_REACHABILITY_V3_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "hccl_types.h"
#include "task_def_v3.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    using BitRow = std::vector<uint64_t>;
    using ReachMatrix = std::vector<BitRow>;
    // using RoaringReachMatrix = std::vector<roaring::Roaring>;

    // The closure matrix only stores reachability between data-move task nodes.
    // Public callers should keep using raw nodeId through IsReachable().
    struct ReachabilityClosure {
        ReachMatrix matrix;
        std::map<NodeId, size_t> dataIndexByNodeId;
        std::vector<NodeId> dataNodeIdByIndex;
    };

    // struct RoaringReachabilityClosure {
    //     RoaringReachMatrix matrix;
    //     std::map<NodeId, size_t> dataIndexByNodeId;
    //     std::vector<NodeId> dataNodeIdByIndex;
    // };

    struct DataTaskReachabilityStats {
        size_t nodeCount{0};
        size_t edgeCount{0};
        size_t dataTaskNodeCount{0};
        size_t dataTaskPairCount{0};
        size_t reachablePairCount{0};
        size_t unreachablePairCount{0};
    };

    // topoOrder is the full graph topo order: mainStart(-1) first, followed by normal task nodes.
    HcclResult GenReachabilityClosure(
        const TaskNode* start, ReachabilityClosure& closure, std::vector<NodeId>* topoOrder = nullptr);
    HcclResult IsReachable(const ReachabilityClosure& closure, NodeId fromNodeId, NodeId toNodeId, bool& isReachable);

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

#endif // CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_REACHABILITY_V3_H
