/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VM_VALIDATION_ISSUE_RECORD_UTILS_H
#define HCCL_VM_VALIDATION_ISSUE_RECORD_UTILS_H

#include <algorithm>
#include <nlohmann_json/json.hpp>
#include <string>

#include "dump/dump_manager.h"
#include "dump/dump_json_utils.h"
#include "dump/validation_issue_recorder.h"

namespace HcclSim {
inline nlohmann::json MakeRelatedNodeRef(const std::string& nodeId)
{
    nlohmann::json node = nlohmann::json::object();
    if (!nodeId.empty()) {
        node["node_id"] = nodeId;
    }
    return node;
}

inline nlohmann::json MakeRelatedNodeRef(const TaskNode* node)
{
    if (node == nullptr) {
        return nlohmann::json::object();
    }
    return DumpIssueTaskNodeToJson(const_cast<TaskNode*>(node));
}

inline nlohmann::json MakeRelatedQueueRef(RankId rankId, u32 queueId)
{
    nlohmann::json queue = nlohmann::json::object();
    queue["rank_id"] = rankId;
    queue["queue_id"] = queueId;
    return queue;
}

inline nlohmann::json MakeRelatedQueueRef(const TaskNode* node)
{
    nlohmann::json queue = nlohmann::json::object();
    if (node == nullptr) {
        return queue;
    }
    queue["rank_id"] = node->rankIdx;
    queue["queue_id"] = node->queIdx;
    queue["pos"] = node->pos;
    return queue;
}

inline nlohmann::json MakeSnapshotStepRef(u32 snapshotStep)
{
    nlohmann::json step = nlohmann::json::object();
    step["snapshot_step"] = snapshotStep;
    return step;
}

inline void AppendRelatedRank(nlohmann::json& detail, RankId rankId)
{
    if (!detail.contains("related_rank") || !detail["related_rank"].is_array()) {
        detail["related_rank"] = nlohmann::json::array();
    }
    auto& relatedRank = detail["related_rank"];
    if (std::find(relatedRank.begin(), relatedRank.end(), rankId) == relatedRank.end()) {
        relatedRank.push_back(rankId);
    }
}

inline void AppendRelatedStep(nlohmann::json& detail, const nlohmann::json& step)
{
    if (step.is_null() || step.empty()) {
        return;
    }

    if (!detail.contains("related_step") || !detail["related_step"].is_array()) {
        detail["related_step"] = nlohmann::json::array();
    }
    auto& relatedStep = detail["related_step"];
    if (std::find(relatedStep.begin(), relatedStep.end(), step) == relatedStep.end()) {
        relatedStep.push_back(step);
    }
}

inline void AppendRelatedNode(nlohmann::json& detail, const nlohmann::json& node)
{
    if (node.is_null() || node.empty()) {
        return;
    }
    if (!detail.contains("related_node") || !detail["related_node"].is_array()) {
        detail["related_node"] = nlohmann::json::array();
    }
    auto& relatedNode = detail["related_node"];
    if (std::find(relatedNode.begin(), relatedNode.end(), node) == relatedNode.end()) {
        relatedNode.push_back(node);
    }
}

inline void AppendRelatedNodeId(nlohmann::json& detail, const std::string& nodeId)
{
    AppendRelatedNode(detail, MakeRelatedNodeRef(nodeId));
}

inline void AppendRelatedNodeId(nlohmann::json& detail, const TaskNode* node)
{
    AppendRelatedNode(detail, MakeRelatedNodeRef(node));
}

inline void AppendRelatedQueue(nlohmann::json& detail, RankId rankId, u32 queueId)
{
    nlohmann::json queue = MakeRelatedQueueRef(rankId, queueId);
    if (!detail.contains("related_queue") || !detail["related_queue"].is_array()) {
        detail["related_queue"] = nlohmann::json::array();
    }
    auto& relatedQueue = detail["related_queue"];
    if (std::find(relatedQueue.begin(), relatedQueue.end(), queue) == relatedQueue.end()) {
        relatedQueue.push_back(queue);
    }
}

inline void AppendRelatedQueue(nlohmann::json& detail, const TaskNode* node)
{
    if (node == nullptr) {
        return;
    }
    AppendRelatedQueue(detail, node->rankIdx, node->queIdx);
}

inline void AppendRelatedSnapshotStep(nlohmann::json& detail, u32 snapshotStep)
{
    AppendRelatedStep(detail, MakeSnapshotStepRef(snapshotStep));
}

inline void RecordValidationIssueOnRet(
    const std::string& stage, const std::string& code, HcclResult ret, nlohmann::json detail = nlohmann::json::object())
{
    if (!DumpManager::GetInstance().IsEnabled()) {
        return;
    }
    detail["ret_code"] = static_cast<u32>(ret);
    ValidationIssueRecorder::GetInstance().RecordIssue("ERROR", stage, code, detail);
}

template <typename DetailBuilder>
inline void RecordValidationIssueOnRetLazy(
    const std::string& stage, const std::string& code, HcclResult ret, DetailBuilder&& detailBuilder)
{
    if (!DumpManager::GetInstance().IsEnabled()) {
        return;
    }
    RecordValidationIssueOnRet(stage, code, ret, detailBuilder());
}

inline nlohmann::json MakeTaskNodeDetail(TaskNode* node)
{
    nlohmann::json detail = nlohmann::json::object();
    AppendRelatedNodeId(detail, node);
    return detail;
}
} // namespace HcclSim

#define HCCL_VM_RETURN_WITH_ISSUE(stage, code, ret, detail_expr)                \
    do {                                                                        \
        HcclSim::RecordValidationIssueOnRetLazy((stage), (code), (ret), [&]() { \
            return (detail_expr);                                               \
        });                                                                     \
        return (ret);                                                           \
    } while (0)

#define HCCL_VM_CHK_RET_WITH_ISSUE(call, stage, code, detail_expr)                          \
    do {                                                                                    \
        HcclResult validationRet = (call);                                                  \
        if (validationRet != HCCL_SUCCESS) {                                                \
            HcclSim::RecordValidationIssueOnRetLazy((stage), (code), validationRet, [&]() { \
                return (detail_expr);                                                       \
            });                                                                             \
            return validationRet;                                                           \
        }                                                                                   \
    } while (0)

#endif // HCCL_VM_VALIDATION_ISSUE_RECORD_UTILS_H
