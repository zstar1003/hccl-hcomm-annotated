/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "big_graph_checker.h"

#include <cstring>
#include <limits>
#include <utility>

#include "sim_log.h"
#include "task_graph_generator_v3/task_graph_single_task_check_v3.h"
#include "task_graph_generator_v3/task_graph_sync_conflict_v3.h"
#include "task_graph_generator_v3/task_meta_translator_v3.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace BigGraphCheckV3 {

    namespace {
        using V3Graph = TaskGraphGeneratorV3::TaskGraphGeneratorV3;
        using V3Node = TaskGraphGeneratorV3::TaskNode;
        using V3NodeId = TaskGraphGeneratorV3::NodeId;
        using V3RankNodeQueues = TaskGraphGeneratorV3::RankNodeQueues;

        HcclResult DecodeOpDetails(const sim::OpDetailTab& detailTab, OpDetails& details)
        {
            if (detailTab.opDetail.size() < sizeof(OpDetails)) {
                HCCL_VM_ERROR(
                    "Operator detail payload is too small, rankId={}, actualSize={}, expectedSize={}", detailTab.rankId,
                    detailTab.opDetail.size(), sizeof(OpDetails));
                return HCCL_E_PARA;
            }
            std::memcpy(&details, detailTab.opDetail.data(), sizeof(OpDetails));
            return HCCL_SUCCESS;
        }

        HcclResult AppendTranslatedOperator(
            TaskGraphGeneratorV3::TaskMetaTranslatorV3& translator, std::vector<std::unique_ptr<V3Node>>& nodes,
            TaskGraphGeneratorV3::AllRankNodeQueues& queues)
        {
            std::vector<std::unique_ptr<V3Node>> localNodes = translator.TakeNodes();
            TaskGraphGeneratorV3::AllRankNodeQueues localQueues = translator.TakeTaskQueues();
            const size_t nodeOffset = nodes.size();
            if (nodeOffset > static_cast<size_t>(std::numeric_limits<V3NodeId>::max())
                || localNodes.size() > static_cast<size_t>(std::numeric_limits<V3NodeId>::max()) - nodeOffset) {
                return HCCL_E_MEMORY;
            }
            for (const auto& node : localNodes) {
                if (node == nullptr) {
                    return HCCL_E_PTR;
                }
            }

            for (auto& node : localNodes) {
                node->SetNodeId(static_cast<V3NodeId>(nodeOffset + static_cast<size_t>(node->GetNodeId())));
                nodes.emplace_back(std::move(node));
            }

            for (const auto& rankEntry : localQueues) {
                V3RankNodeQueues& target = queues[rankEntry.first];
                if (target.size() < rankEntry.second.size()) {
                    target.resize(rankEntry.second.size());
                }
                for (size_t streamIndex = 0; streamIndex < rankEntry.second.size(); ++streamIndex) {
                    auto& targetStream = target[streamIndex];
                    for (V3NodeId nodeId : rankEntry.second[streamIndex]) {
                        if (nodeId < 0 || static_cast<size_t>(nodeId) >= localNodes.size()) {
                            return HCCL_E_PARA;
                        }
                        targetStream.push_back(static_cast<V3NodeId>(nodeOffset + static_cast<size_t>(nodeId)));
                    }
                }
            }
            return HCCL_SUCCESS;
        }
    } // namespace

    HcclResult BigGraphCheckerV3::LoadOpData(loader::Loader& loader, uint32_t syncIter)
    {
        return dataLoader_.Load(loader, syncIter, data_);
    }

    HcclResult BigGraphCheckerV3::TranslateTask()
    {
        if (data_.operators.empty()) {
            return HCCL_E_PARA;
        }

        storage_.Reset(false);
        translatedNodes_.clear();
        translatedTaskQueues_.clear();
        graph_.reset();

        for (const OpParam& opParam : data_.operators) {
            for (const OperatorRankData& rankData : opParam.ranks) {
                HcclResult ret = storage_.LoadHcclVmSynthesisData(rankData.rankId, rankData.op.memInfo, data_.channels);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
        }
        HcclResult ret = storage_.LoadHcclVmInstrData(data_.instrRes);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }

        for (const OpParam& opParam : data_.operators) {
            storage_.BeginOpGroup();
            for (const OperatorRankData& rankData : opParam.ranks) {
                OpDetails details{};
                ret = DecodeOpDetails(rankData.op.detail, details);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                sim::OpDetailTab detailTab = rankData.op.detail;
                ret = storage_.Trans2CheckerParam(detailTab, details);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            ret = storage_.FinalizeOpGroup();
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            storage_.MergeAll2AllVSendCountMatrix();
            storage_.SaveCheckerParam(opParam.operatorId);

            std::vector<std::vector<HcclTaskMetaData>> allTaskMetas;
            allTaskMetas.reserve(opParam.ranks.size());
            for (const OperatorRankData& rankData : opParam.ranks) {
                allTaskMetas.push_back(rankData.taskMetas);
            }
            ret = storage_.LoadDecodedHcclVmTaskMetaData(allTaskMetas);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }

            TaskGraphGeneratorV3::TaskMetaTranslatorV3 translator;
            ret = translator.Translate(storage_, opParam.operatorId);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            ret = AppendTranslatedOperator(translator, translatedNodes_, translatedTaskQueues_);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
        }

        HCCL_VM_INFO(
            "Translated big graph tasks, operatorCount={}, nodeCount={}, rankCount={}", data_.operators.size(),
            translatedNodes_.size(), translatedTaskQueues_.size());
        return HCCL_SUCCESS;
    }

    HcclResult BigGraphCheckerV3::GenerateBigGraph()
    {
        if (translatedNodes_.empty() || translatedTaskQueues_.empty()) {
            HCCL_VM_ERROR(
                "{} Checker get empty task queue, please check if the HCCL-VM end normally, operatorCount={}, "
                "nodeCount={}, rankCount={}",
                MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), data_.operators.size(), translatedNodes_.size(),
                translatedTaskQueues_.size());
            return HCCL_E_PARA;
        }

        HcclResult ret = TaskGraphGeneratorV3::CheckSlaveTaskQueue(translatedNodes_, translatedTaskQueues_);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }

        auto graph = std::make_unique<V3Graph>();
        graph->SetStorageManager(&storage_);
        ret = graph->GenGraph(std::move(translatedNodes_), std::move(translatedTaskQueues_));
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        graph_ = std::move(graph);
        HCCL_VM_INFO(
            "Generated big graph, nodeCount={}, rankCount={}", graph_->GetNodes().size(),
            graph_->GetTaskQueues().size());
        return HCCL_SUCCESS;
    }

    HcclResult BigGraphCheckerV3::SingleTaskCheck() { return HCCL_E_NOT_SUPPORT; }

    HcclResult BigGraphCheckerV3::SyncCheck()
    {
        if (graph_ == nullptr || graph_->GetMainStartNode() == nullptr) {
            HCCL_VM_ERROR(
                "{} Cannot run big graph sync-conflict check before the graph is generated",
                MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR));
            return HCCL_E_PARA;
        }

        TaskGraphGeneratorV3::SyncConflictCheckStats stats;
        const HcclResult ret = TaskGraphGeneratorV3::CheckSyncResourceConflict(graph_->GetMainStartNode(), &stats);
        HCCL_VM_INFO(
            "Big graph sync-conflict check finished, status={}, originalNodeCount={}, copiedNodeCount={}, "
            "copiedEdgeCount={}, resourceBucketCount={}, pairCount={}, checkedBucketCount={}, conflictCount={}",
            ret == HCCL_SUCCESS ? "success" : "failed", stats.originalNodeCount, stats.copiedNodeCount,
            stats.copiedEdgeCount, stats.resourceBucketCount, stats.pairCount, stats.checkedBucketCount,
            stats.conflictCount);
        return ret;
    }

    HcclResult BigGraphCheckerV3::MemConflictCheck() { return HCCL_E_NOT_SUPPORT; }

    HcclResult BigGraphCheckerV3::SemanticCheck() { return HCCL_E_NOT_SUPPORT; }

} // namespace BigGraphCheckV3
} // namespace HcclSim
