/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_META_TRANSLATOR_V3_H
#define CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_META_TRANSLATOR_V3_H

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "hccl_types.h"
#include "storage_manager.h"
#include "task_meta_defs.h"
#include "task_graph_generator_v3.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    class TaskMetaTranslatorV3 {
    public:
        TaskMetaTranslatorV3() = default;
        ~TaskMetaTranslatorV3() = default;

        TaskMetaTranslatorV3(TaskMetaTranslatorV3&&) = default;
        TaskMetaTranslatorV3& operator=(TaskMetaTranslatorV3&&) = default;
        TaskMetaTranslatorV3(const TaskMetaTranslatorV3&) = delete;
        TaskMetaTranslatorV3& operator=(const TaskMetaTranslatorV3&) = delete;

        HcclResult Translate(StorageManager& storage, OperatorId operatorId);
        void Reset();

        const std::vector<std::unique_ptr<TaskNode>>& GetNodes() const { return nodes_; }
        const AllRankNodeQueues& GetTaskQueues() const { return taskQueues_; }
        std::vector<std::unique_ptr<TaskNode>> TakeNodes();
        AllRankNodeQueues TakeTaskQueues();

    private:
        struct CcuMissionKey {
            RankId rankId{INVALID_RANK_ID};
            uint8_t dieId{0};
            uint8_t missionId{0};

            bool operator<(const CcuMissionKey& rhs) const
            {
                if (rankId != rhs.rankId) {
                    return rankId < rhs.rankId;
                }
                if (dieId != rhs.dieId) {
                    return dieId < rhs.dieId;
                }
                return missionId < rhs.missionId;
            }
        };

        HcclResult TranslateOneTaskMeta(
            const HcclTaskMetaData& taskMeta, StorageManager& storage, uint32_t taskIndex, OperatorId operatorId,
            NodeId& nodeId);
        HcclResult AddTaskNode(const TaskPosition& position, std::unique_ptr<TaskNode> node, NodeId& nodeId);

        std::vector<std::unique_ptr<TaskNode>> nodes_;
        AllRankNodeQueues taskQueues_;
        std::map<CcuMissionKey, NodeId> ccuMissionNodes_;
    };
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

#endif // CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_META_TRANSLATOR_V3_H
