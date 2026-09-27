/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mem_conflict_check_utils.h"

#include <cstdint>

#include "log.h"
#include "task_ccu.h"

namespace HcclSim {
std::unordered_map<TaskStubPtr, TaskStubPtr> g_ccuGraphTaskOri2New;
TaskNodePtr GetCcuTaskHead(TaskNodePtr node)
{
    TaskNode* retNode = node;
    if (node->task != nullptr && node->task->GetType() == TaskTypeStub::CCU_GRAPH) {
        // 首次进入子图
        TaskStubCcuGraph* curCcuTask = dynamic_cast<TaskStubCcuGraph*>(node->task);
        retNode = curCcuTask->ccuHeadTaskNode;
    } else if (node->task != nullptr && node->task->GetType() == TaskTypeStub::SUB_GRAPH_END) {
        // 走到子图的最后一个子节点了，就回到整图
        TaskStubSubGraphEnd* subGraphEnd = dynamic_cast<TaskStubSubGraphEnd*>(node->task);
        retNode = subGraphEnd->subGraphNode;
    }
    return retNode;
}

HcclResult
GetNewNode(const Ori2NewNodeMap& originNode2copyNode, TaskNodePtr oldNode, TaskNodePtr& newNode, bool retErr = true)
{
    if (oldNode == nullptr && !retErr) {
        return HcclResult::HCCL_SUCCESS;
    }
    if (oldNode == nullptr && retErr) {
        HCCL_ERROR("[GetNewNode] target node is nullptr.");
        return HCCL_E_INTERNAL;
    }

    auto iter = originNode2copyNode.find(oldNode);
    if (iter == originNode2copyNode.end() && retErr) {
        HCCL_ERROR("[GetNewNode] cannot find new node to pair with the targe node.");
        return HCCL_E_INTERNAL;
    }
    newNode = iter->second;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CopyCcuSubGraphNode(
    TaskStub* originCcu, TaskStub** newCcu, std::vector<std::unordered_map<TaskStubPtr, TaskStubPtr>>& ccuGraphs,
    std::vector<CcuOri2NewNodeMap>& AllOri2NewNodeMap)
{
    if (originCcu->GetType() != TaskTypeStub::CCU_GRAPH) {
        HCCL_ERROR("origin node is not ccu graph");
        return HcclResult::HCCL_E_INTERNAL;
    }

    TaskStubCcuGraph* oriCcuTask = dynamic_cast<TaskStubCcuGraph*>(originCcu);
    *newCcu = new TaskStubCcuGraph(oriCcuTask);
    TaskStubCcuGraph* newCcuTask = dynamic_cast<TaskStubCcuGraph*>(*newCcu);

    auto rankId = oriCcuTask->rankId;
    ccuGraphs[rankId].insert(std::make_pair(originCcu, *newCcu));

    std::map<TaskNodePtr, TaskNodePtr> originNode2copyNode; // 用来收录原节点到新节点的映射
    originNode2copyNode[oriCcuTask->ccuHeadTaskNode] = newCcuTask->ccuHeadTaskNode;
    // 拷贝节点
    for (auto& oriNode : oriCcuTask->toDeleteTaskNode_) {
        if (oriNode == oriCcuTask->ccuHeadTaskNode) {
            continue;
        }
        auto newNode = new TaskNode(oriNode->task, oriNode->rankIdx, oriNode->queIdx, oriNode->pos);
        originNode2copyNode[oriNode] = newNode;
        newCcuTask->toDeleteTaskNode_.push_back(newNode);
    }

    AllOri2NewNodeMap[rankId][originCcu] = originNode2copyNode;

    // 恢复内存冲突改造所需的成员变量
    // 拷贝loop节点 —— loop并行化改造
    for (const auto& loopGroupInfo : oriCcuTask->loopGroupInfo_) {
        std::vector<LoopInfo> loopGroup;
        for (const auto& loop : loopGroupInfo) {
            loopGroup.push_back(LoopInfo(originNode2copyNode[loop.loopStart], originNode2copyNode[loop.loopEnd]));
        }
        newCcuTask->loopGroupInfo_.push_back(loopGroup);
    }
    for (auto it = oriCcuTask->localPostWaitPairs_.begin(); it != oriCcuTask->localPostWaitPairs_.end(); ++it) {
        auto newLocPost = originNode2copyNode.at(it->first);
        newCcuTask->localPostWaitPairs_[newLocPost] = originNode2copyNode.at(it->second);
    }
    // 拷贝双边语义节点 —— 单边转双边改造
    newCcuTask->bilateralNodes_.resize(newCcuTask->queueNum_);
    for (const auto& part1 : oriCcuTask->bilateralPart1_) {
        std::map<TaskNodePtr, TaskNodePtr> mapTmp;
        for (auto iter = part1.begin(); iter != part1.end(); ++iter) {
            TaskNodePtr newPost = nullptr;
            TaskNodePtr newWait = nullptr;
            CHK_RET(GetNewNode(originNode2copyNode, iter->first, newPost, true));
            CHK_RET(GetNewNode(originNode2copyNode, iter->second, newWait, false));
            mapTmp[newPost] = newWait;
        }
        newCcuTask->bilateralPart1_.push_back(mapTmp);
    }
    for (const auto& part2 : oriCcuTask->bilateralPart2_) {
        std::map<TaskNodePtr, TaskNodePtr> mapTmp;
        for (auto iter = part2.begin(); iter != part2.end(); ++iter) {
            TaskNodePtr newPost = nullptr;
            TaskNodePtr newWait = nullptr;
            CHK_RET(GetNewNode(originNode2copyNode, iter->first, newPost, true));
            CHK_RET(GetNewNode(originNode2copyNode, iter->second, newWait, false));
            mapTmp[newPost] = newWait;
        }
        newCcuTask->bilateralPart2_.push_back(mapTmp);
    }

    // 拷贝异步节点 —— 并行化改造
    for (const auto& parNode : oriCcuTask->parallelNodes_) {
        newCcuTask->parallelNodes_.push_back(originNode2copyNode[parNode]);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult QueryCcuGraphNode(
    const Ori2NewNodeMap& ccuNodeMap, const std::vector<CcuOri2NewNodeMap>& AllOri2NewNodeMap, TaskNodePtr oriNode,
    TaskNodePtr& newNode)
{
    auto locRes = ccuNodeMap.find(oriNode);
    if (locRes == ccuNodeMap.end()) {
        if (oriNode->task->GetType() != TaskTypeStub::WAIT && oriNode->task->GetType() != TaskTypeStub::POST) {
            HCCL_ERROR("[CopyCcuSubGraph] can not find node in local ccu graph.");
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        uint32_t rmtRankId = 0;
        TaskStubPtr rmtCcuTaskPtr = nullptr;
        if (oriNode->task->GetType() == TaskTypeStub::POST) {
            TaskStubPost* candPost = dynamic_cast<TaskStubPost*>(oriNode->task);
            if (candPost->ccuTaskPtr_ == 0) {
                HCCL_ERROR("[QueryCcuGraphNode] cannot find ccu subgraph head node by post node");
                return HCCL_E_INTERNAL;
            }
            TaskStubCcuGraph* rmtCcuTask = reinterpret_cast<TaskStubCcuGraph*>(candPost->ccuTaskPtr_);
            rmtRankId = oriNode->rankIdx;
            rmtCcuTaskPtr = reinterpret_cast<TaskStubPtr>(candPost->ccuTaskPtr_);
        } else if (oriNode->task->GetType() == TaskTypeStub::WAIT) {
            TaskStubWait* candWait = dynamic_cast<TaskStubWait*>(oriNode->task);
            if (candWait->ccuTaskPtr_ == 0) {
                HCCL_ERROR("[QueryCcuGraphNode] cannot find ccu subgraph head node by wait node");
                return HCCL_E_INTERNAL;
            }
            TaskStubCcuGraph* rmtCcuTask = reinterpret_cast<TaskStubCcuGraph*>(candWait->ccuTaskPtr_);
            rmtRankId = oriNode->rankIdx;
            rmtCcuTaskPtr = reinterpret_cast<TaskStubPtr>(candWait->ccuTaskPtr_);
        }
        auto rmtCcuNodeMapIter = AllOri2NewNodeMap[rmtRankId].find(rmtCcuTaskPtr);
        if (rmtCcuNodeMapIter == AllOri2NewNodeMap[rmtRankId].end()) {
            HCCL_ERROR("[QueryCcuGraphNode] can not find node map of remote ccu graph.");
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        auto rmtCcuNodeMap = rmtCcuNodeMapIter->second;
        auto rmtRes = rmtCcuNodeMap.find(oriNode);
        if (rmtRes == rmtCcuNodeMap.end()) {
            HCCL_ERROR("[CopyCcuSubGraph] can not find node in remote ccu graph.");
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        newNode = rmtRes->second;
        return HcclResult::HCCL_SUCCESS;
    }
    newNode = locRes->second;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CopyCcuSubGraphConnection(
    std::vector<std::unordered_map<TaskStubPtr, TaskStubPtr>>& ccuGraphs,
    const std::vector<CcuOri2NewNodeMap>& AllOri2NewNodeMap)
{
    for (auto& ccuGraph : ccuGraphs) {
        for (auto& ccuPair : ccuGraph) {
            TaskStubCcuGraph* oriCcuTask = dynamic_cast<TaskStubCcuGraph*>(ccuPair.first);
            TaskStubCcuGraph* newCcuTask = dynamic_cast<TaskStubCcuGraph*>(ccuPair.second);

            auto rankId = oriCcuTask->rankId;
            auto ccuNodeMapIter = AllOri2NewNodeMap[rankId].find(ccuPair.first);
            if (ccuNodeMapIter == AllOri2NewNodeMap[rankId].end()) {
                HCCL_ERROR(
                    "[CopyCcuSubGraphConnection] can not find node map of ccu graph [%s].", oriCcuTask->des.c_str());
                return HcclResult::HCCL_E_NOT_FOUND;
            }
            auto ccuNodeMap = ccuNodeMapIter->second;

            // 按原节点，拷贝副本连接关系
            for (auto& oriNode : oriCcuTask->toDeleteTaskNode_) {
                for (auto& parent : oriNode->parents) {
                    TaskNodePtr newParent = nullptr;
                    CHK_RET(QueryCcuGraphNode(ccuNodeMap, AllOri2NewNodeMap, parent, newParent));
                    ccuNodeMap[oriNode]->parents.push_back(newParent);
                }
                for (auto& child : oriNode->children) {
                    TaskNodePtr newChild = nullptr;
                    CHK_RET(QueryCcuGraphNode(ccuNodeMap, AllOri2NewNodeMap, child, newChild));
                    ccuNodeMap[oriNode]->children.push_back(newChild);
                }
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
