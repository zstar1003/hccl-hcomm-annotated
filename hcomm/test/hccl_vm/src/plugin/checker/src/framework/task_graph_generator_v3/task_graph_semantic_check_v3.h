/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_SEMANTIC_CHECK_V3_H
#define CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_SEMANTIC_CHECK_V3_H

#include <cstddef>

#include "hccl_types.h"
#include "task_def_v3.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    struct SemanticCheckStats {
        size_t handledNodeCount{0};
        size_t rankCount{0};
        size_t normalSemanticCount{0};
        uint64_t normalSemanticBytes{0};
        size_t msBucketCount{0};
    };

    HcclResult CheckSemantics(const TaskNode* start, SemanticCheckStats* stats = nullptr);
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

#endif // CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_GRAPH_SEMANTIC_CHECK_V3_H
