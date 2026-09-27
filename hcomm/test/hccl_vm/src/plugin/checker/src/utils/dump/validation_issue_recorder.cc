/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dump/dump_graph.h"
#include "dump/dump_json_utils.h"
#include "dump/dump_manager.h"
#include "dump/dump_run_manifest.h"
#include "dump/validation_issue_recorder.h"

namespace HcclSim {
using Json = nlohmann::json;

static const std::string VALIDATION_ISSUE_DUMP_TYPE = "validation_issues";
static const std::string VALIDATION_ISSUE_DUMP_PATH = "validation/issues.msgpack";

static Json DumpIssueTaskNodeRefToJson(const TaskNode* node)
{
    Json nodeRefJson = Json::object();
    if (node == nullptr) {
        return nodeRefJson;
    }
    nodeRefJson["rank_id"] = node->rankIdx;
    nodeRefJson["queue_id"] = node->queIdx;
    nodeRefJson["pos"] = node->pos;
    return nodeRefJson;
}

Json DumpIssueTaskNodeToJson(TaskNode* node, const std::string& nodeId)
{
    Json nodeJson = Json::object();
    if (node == nullptr) {
        return nodeJson;
    }

    nodeJson = DumpIssueTaskNodeRefToJson(node);
    if (!nodeId.empty()) {
        nodeJson["node_id"] = nodeId;
    } else if (node->task != nullptr && !node->task->GetTaskId().empty()) {
        nodeJson["node_id"] = node->task->GetTaskId();
    } else if (node->rankIdx != static_cast<RankId>(-1)) {
        nodeJson["node_id"] = StringFormat("r%un%u", node->rankIdx, node->pos);
    } else {
        nodeJson["node_id"] = "dummy_start";
    }
    return nodeJson;
}

void ValidationIssueRecorder::Reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    nextIssueId_ = 1;
    issues_.clear();
}

void ValidationIssueRecorder::RecordIssue(
    const std::string& severity, const std::string& stage, const std::string& code, const nlohmann::json& detail)
{
    if (!DumpManager::GetInstance().IsEnabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    Json issue = Json::object();
    issue["issue_id"] = nextIssueId_++;
    issue["severity"] = severity;
    issue["stage"] = stage;
    issue["code"] = code;
    issue["detail"] = detail;
    issues_.push_back(issue);
}

HcclResult ValidationIssueRecorder::Flush() const
{
    DumpManager& dumpManager = DumpManager::GetInstance();
    if (!dumpManager.IsEnabled()) {
        return HcclResult::HCCL_SUCCESS;
    }

    Json issuesSnapshot = Json::array();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        issuesSnapshot = issues_;
    }

    Json issueDumpJson = Json::object();
    issueDumpJson["type"] = VALIDATION_ISSUE_DUMP_TYPE;
    issueDumpJson["issue_count"] = issuesSnapshot.size();
    issueDumpJson["issues"] = issuesSnapshot;
    DumpRunManifest::GetInstance().SetErrorCount(issuesSnapshot.size());
    return dumpManager.Write(VALIDATION_ISSUE_DUMP_PATH, issueDumpJson);
}
} // namespace HcclSim
