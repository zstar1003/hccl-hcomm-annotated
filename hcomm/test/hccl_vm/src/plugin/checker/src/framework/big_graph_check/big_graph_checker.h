/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_BIG_GRAPH_CHECKER_H
#define CHECKER_BIG_GRAPH_CHECKER_H

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "big_graph_data_loader.h"
#include "storage_manager.h"
#include "task_graph_generator_v3/task_graph_generator_v3.h"

namespace HcclSim {
namespace BigGraphCheckV3 {

    class BigGraphCheckerV3 {
    public:
        BigGraphCheckerV3() = default;
        ~BigGraphCheckerV3() = default;

        HcclResult LoadOpData(loader::Loader& loader, uint32_t syncIter);
        const BigGraphData& GetData() const { return data_; }
        const std::vector<OpParam>& GetOpParams() const { return data_.operators; }
        const TaskGraphGeneratorV3::TaskGraphGeneratorV3* GetGraph() const { return graph_.get(); }

        // Future stages intentionally remain separate from the existing V3 path.
        HcclResult TranslateTask();
        HcclResult GenerateBigGraph();
        HcclResult SingleTaskCheck();
        HcclResult SyncCheck();
        HcclResult MemConflictCheck();
        HcclResult SemanticCheck();

    private:
        BigGraphData data_;
        BigGraphDataLoader dataLoader_;
        StorageManager storage_;
        std::vector<std::unique_ptr<TaskGraphGeneratorV3::TaskNode>> translatedNodes_;
        TaskGraphGeneratorV3::AllRankNodeQueues translatedTaskQueues_;
        std::unique_ptr<TaskGraphGeneratorV3::TaskGraphGeneratorV3> graph_;
    };

} // namespace BigGraphCheckV3
} // namespace HcclSim

#endif // CHECKER_BIG_GRAPH_CHECKER_H
