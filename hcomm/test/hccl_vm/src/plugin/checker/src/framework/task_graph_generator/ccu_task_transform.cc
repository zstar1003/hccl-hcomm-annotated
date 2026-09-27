/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_task_transform.h"

#include <cstdint>

#include "ccu_all_rank_param_recorder.h"
#include "ccu_task_common.h"
#include "sim_log.h"
#include "sim_common.h"
#include "storage_manager.h"
#include "task_graph_generator.h"
#include "type_conversion.h"

namespace HcclSim {
std::unique_ptr<InstructMapBase> g_instrMap;

u32 GetTopicId(HcclSim::TaskNode* post)
{
    if (post->task->GetType() == TaskTypeStub::LOCAL_POST_TO) {
        TaskStubLocalPostTo* postTask = dynamic_cast<TaskStubLocalPostTo*>(post->task);
        return postTask->GetTopicId();
    }
    TaskStubPost* postTask = dynamic_cast<TaskStubPost*>(post->task);
    return postTask->GetTopicId();
}

void SetTopicId(HcclSim::TaskNode* post, u32 topicId)
{
    if (post->task->GetType() == TaskTypeStub::LOCAL_POST_TO) {
        TaskStubLocalPostTo* postTask = dynamic_cast<TaskStubLocalPostTo*>(post->task);
        postTask->SetTopicId(topicId);
        return;
    }
    TaskStubPost* postTask = dynamic_cast<TaskStubPost*>(post->task);
    postTask->SetTopicId(topicId);
    return;
}

void GenWaitNode(TaskStubCcuGraph* curCcuTask, uint32_t queId, uint16_t waitCKEId, uint16_t waitCKEMask)
{
    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    uint16_t localWaitMask = 0;
    uint16_t remoteWaitMask = 0;
    TaskNode* localWaitNode = nullptr;
    TaskNode* remoteWaitNode = nullptr;

    std::set<HcclSim::TaskNode*>& seenPosts = AllRankParamRecorder::Global()->seenPost[rankId][dieId][waitCKEId];
    std::vector<TaskNodePtr> localPosts;
    for (auto& post : seenPosts) {
        u32 topicId = GetTopicId(post);
        uint16_t curPostMask = topicId & waitCKEMask;
        if (curPostMask == 0) {
            continue;
        }

        if (post->task->GetType() == TaskTypeStub::LOCAL_POST_TO) {
            localWaitMask = localWaitMask | curPostMask;
            localPosts.push_back(post);
        } else {
            remoteWaitMask = remoteWaitMask | curPostMask;
        }
    }

    if (remoteWaitMask != 0) {
        remoteWaitNode = AddWait(rankId, queId, curCcuTask, remoteWaitMask);
    }

    if (localWaitMask != 0) {
        localWaitNode = AddLocalWait(rankId, queId, curCcuTask, localWaitMask);
        // ccu子图中，local post和local wait是多对一的关系
        for (auto& locPost : localPosts) {
            curCcuTask->localPostWaitPairs_[locPost] = localWaitNode;
        }
    }

    for (auto it = seenPosts.begin(); it != seenPosts.end();) {
        auto post = *it;
        u32 topicId = GetTopicId(post);
        uint16_t curPostMask = topicId & waitCKEMask;
        if (curPostMask == 0) {
            ++it;
            continue;
        }
        if (post->task->GetType() == TaskTypeStub::LOCAL_POST_TO) {
            AddNodeRelation(post, localWaitNode);
        } else {
            AddNodeRelation(post, remoteWaitNode);
            auto waitTask = dynamic_cast<TaskStubWait*>(remoteWaitNode->task);
            waitTask->SetRemoteRank(post->rankIdx);
            CollectBilateralWaitInfo(curCcuTask, queId, remoteWaitNode);
        }
        topicId = topicId & (~waitCKEMask);
        SetTopicId(post, topicId);
        if (topicId == 0) {
            it = seenPosts.erase(it);
        } else {
            ++it;
        }
    }
    return;
}

HcclResult ProcessWaitMask(
    RankId rankId, uint32_t dieId, TaskStubCcuGraph* curCcuTask, uint32_t queId, uint16_t waitCKEId,
    uint16_t waitCKEMask, bool& isContinue)
{
    HCCL_VM_DEBUG(
        "Enter...rank {:d}, die{:d}, que{:d}, waitCke{:d}, waitMask{:d}", rankId, static_cast<uint32_t>(dieId), queId,
        waitCKEId, waitCKEMask);
    if (waitCKEMask != 0x0000) {
        uint16_t ckeValue = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetCKE(rankId, dieId, waitCKEId, ckeValue));
        // 条件不满足，继续等待
        if ((ckeValue & waitCKEMask) != waitCKEMask) {
            isContinue = 0;
            HCCL_VM_DEBUG(
                "Does not meet the condition, continue: ckeId: {:d}, ckeValue: {:d}, mask: {:x}", waitCKEId, ckeValue,
                waitCKEMask);
            return HCCL_SUCCESS;
        } else {
            GenWaitNode(curCcuTask, queId, waitCKEId, waitCKEMask);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransformInstr(const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue)
{
    if (g_instrMap == nullptr) {
        HCCL_ERROR("g_instrMap error");
        return HCCL_E_INTERNAL;
    }
    if (!g_instrMap->IsSupported(instr->header.header)) {
        HCCL_ERROR("ins type not supported %hu", instr->header.header);
        return HCCL_E_INTERNAL;
    }
    return g_instrMap.get()->Transform(instr, curCcuTask, queId, isContinue, nullptr);
}

void addChildNode(std::deque<TaskNode*>& queue, std::set<TaskNode*>& isVisitedNode, TaskNode* curNode)
{
    for (auto node = curNode->children.begin(); node != curNode->children.end(); node++) {
        TaskNode* child = *node;
        if (isVisitedNode.count(child) == 0) {
            isVisitedNode.insert(child);
            queue.push_back(child);
        }
    }
    return;
}

bool AreParentsCompleted(TaskNode* node, const std::set<TaskNode*>& completedNodes)
{
    if (node == nullptr) {
        return false;
    }
    for (auto* parent : node->parents) {
        if (parent == nullptr || completedNodes.count(parent) == 0) {
            return false;
        }
    }
    return true;
}

CcuInstrVersion GetCcuInstrVersion()
{
    DevType devType = AllRankParamRecorder::Global()->GetDevType();
    if (devType == DevType::DEV_TYPE_950) {
        return CcuInstrVersion::VERSION_A5;
#ifdef BUILD_A6_CCU_INSTR
    } else if (devType == DevType::DEV_TYPE_960) {
        return CcuInstrVersion::VERSION_A6;
#endif
    } else {
        return CcuInstrVersion::VERSION_A5;
    }
}

HcclResult TransformInstrQue(TaskNodePtr node, TaskStubCcuGraph* curCcuTask, uint32_t queId)
{
    // 表示当前微码序列已经完成成图
    if (curCcuTask->instrQueStatus[queId]) {
        return HCCL_SUCCESS;
    }
    g_instrMap = InstructMapFactory::Create(GetCcuInstrVersion());

    bool isContinue = true;
    hcomm::CcuRep::CcuInstrInfo& microCodeQue = curCcuTask->instrInfo[queId];
    auto endInstrId = curCcuTask->GetMissionEndInstrId(queId);
    u32& pos = curCcuTask->microCodePosInQue[queId];
    while (pos < endInstrId) {
        u32 prePos = pos;
        HCCL_VM_DEBUG(
            "Current process ccu graph: {}, instruction id={}, header={}, queId={}, pointer={:x}",
            curCcuTask->Describe(), pos, microCodeQue.instrVec[pos].header.header, queId, (uint64_t)curCcuTask);
        CHK_RET(TransformInstr(&microCodeQue.instrVec[pos], curCcuTask, queId, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }
        // 相等表示当前指令未刷新pos，则进行递增
        if (pos == prePos) {
            pos++;
        }
    }

    curCcuTask->instrQueStatus[queId] = true;
    curCcuTask->isGenGraphed
        = std::all_of(curCcuTask->instrQueStatus.begin(), curCcuTask->instrQueStatus.end(), [](bool b) {
              return b;
          });

    // 如果整图匹配完成，添加一个结束节点
    if (curCcuTask->isGenGraphed) {
        AddCcuSubGraphEnd(node, curCcuTask);
    }

    HCCL_VM_DEBUG("end...");
    return HCCL_SUCCESS;
}

HcclResult ProcessCcuNode(TaskNodePtr node, TaskStubCcuGraph* curCcuTask)
{
    curCcuTask->queueNum_ = curCcuTask->instrInfo.size();
    uint32_t rankSize = HcclSim::StorageManager::GetInstance().GetRankSize();
    curCcuTask->bilateralPart1_.resize(curCcuTask->queueNum_);
    curCcuTask->bilateralPart2_.resize(curCcuTask->queueNum_);
    curCcuTask->bilateralNodes_.resize(curCcuTask->queueNum_);
    curCcuTask->waitInfoTmp_.resize(rankSize);
    curCcuTask->postInfoTmp_.resize(rankSize);
    for (uint32_t queId = 0; queId < curCcuTask->queueNum_; queId++) {
        CHK_RET(TransformInstrQue(node, curCcuTask, queId));
    }
    return HCCL_SUCCESS;
}

void PrintCcuSingleQue(TaskNodePtr head, u32 rankId, u32 queueIdx)
{
    std::deque<TaskNode*> candNode;
    std::set<TaskNode*> isVisitedNode;
    std::set<TaskNode*> alreadyPrintedNodes;

    // 打印首节点
    HCCL_VM_INFO("ccu single queue head[{:p}]{}", (void*)head, head->task->Describe().c_str());
    alreadyPrintedNodes.insert(head);

    HCCL_VM_INFO("children[");
    for (auto& child : head->children) {
        if (isVisitedNode.count(child) == 0 && child->rankIdx == rankId && child->queIdx == queueIdx) {
            candNode.push_back(child);
            isVisitedNode.insert(child);
        }
    }
    HCCL_VM_INFO("]");

    while (!candNode.empty()) {
        TaskNodePtr curNode = candNode.front();
        candNode.pop_front();
        for (auto& child : curNode->children) {
            if (isVisitedNode.count(child) == 0 && child->rankIdx == rankId && child->queIdx == queueIdx) {
                candNode.push_back(child);
                isVisitedNode.insert(child);
            }
        }

        bool parentsAllPrint = true;
        for (auto& parent : curNode->parents) {
            if (parent->rankIdx == rankId && parent->queIdx == queueIdx) {
                if (alreadyPrintedNodes.count(parent) == 0) {
                    parentsAllPrint = false;
                    break;
                }
            }
        }

        if (parentsAllPrint) {
            HCCL_VM_INFO("[{:p}]{}", (void*)curNode, curNode->task->Describe().c_str());
            alreadyPrintedNodes.insert(curNode);
        } else {
            candNode.push_back(curNode);
        }
    }
    return;
}

void PrintCcuGraph(TaskNodePtr dummyStart)
{
    std::deque<TaskNode*> candNode;
    std::set<TaskNode*> isVisitedNode;

    for (auto node = dummyStart->children.begin(); node != dummyStart->children.end(); node++) {
        TaskNode* child = *node;
        isVisitedNode.insert(child);
        candNode.push_back(child);
    }

    while (!candNode.empty()) {
        TaskNodePtr curNode = candNode.front();
        candNode.pop_front();
        addChildNode(candNode, isVisitedNode, curNode);

        if (curNode->task->GetType() != TaskTypeStub::CCU_GRAPH) {
            continue;
        }

        HCCL_VM_INFO("=======================================================");
        HCCL_VM_INFO("rank id is {:d}", curNode->rankIdx);

        TaskStubCcuGraph* curCcuTask = dynamic_cast<TaskStubCcuGraph*>(curNode->task);
        for (auto& child : curCcuTask->ccuHeadTaskNode->children) {
            u32 queueIdx = child->queIdx;
            HCCL_VM_INFO("-------------------------------------------------------");
            HCCL_VM_INFO("stream/queue id is {:d}", queueIdx);
            PrintCcuSingleQue(child, curNode->rankIdx, queueIdx);
        }
    }
    HCCL_VM_INFO("-------------------------------------------------------");
}

void PrintSQEGraph(TaskNodePtr dummyStart)
{
    std::deque<TaskNode*> candNode;
    std::set<TaskNode*> isVisitedNode;

    for (auto node = dummyStart->children.begin(); node != dummyStart->children.end(); node++) {
        TaskNode* child = *node;
        isVisitedNode.insert(child);
        candNode.push_back(child);
    }

    HCCL_VM_INFO("---------------------------YY----------------------------");
    while (!candNode.empty()) {
        TaskNodePtr curNode = candNode.front();
        candNode.pop_front();
        addChildNode(candNode, isVisitedNode, curNode);

        HCCL_VM_INFO("currNode: {}, {}", (void*)curNode, curNode->task->Describe().c_str());
        for (auto father : curNode->parents) {
            if (father->task == nullptr) {
                continue;
            }
            HCCL_VM_INFO("father: {}, {}", (void*)father, father->task->Describe().c_str());
        }
        for (auto child : curNode->children) {
            if (child->task == nullptr) {
                continue;
            }
            HCCL_VM_INFO("child: {}, {}", (void*)child, child->task->Describe().c_str());
        }
    }
    HCCL_VM_INFO("---------------------------YY----------------------------");
    return;
}

HcclResult GenCcuGraph(TaskNode* dummyStart)
{
    std::deque<TaskNode*> candNode;
    std::set<TaskNode*> isVisitedNode;
    std::set<TaskNode*> completedNodes;
    AllRankParamRecorder::Global()->InitParam();
    HcclSim::StorageManager::GetInstance().InitCcuInfo(
        AllRankParamRecorder::Global()->devType_, AllRankParamRecorder::Global()->ccu_resource_base_addr_);
    HCCL_VM_INFO("dummyStart children size: {}", dummyStart->children.size());

    bool isPrintCcuGraph = std::getenv("CCU_TASK_PRINT");
    completedNodes.insert(dummyStart);
    for (auto node = dummyStart->children.begin(); node != dummyStart->children.end(); node++) {
        TaskNode* child = *node;
        isVisitedNode.insert(child);
        candNode.push_back(child);
    }

    HCCL_VM_INFO("start......{}", candNode.size());
    u32 unmatchedCnt = 0;
    while (!candNode.empty()) {
        // 先判断是否存在死锁的情况
        if (unmatchedCnt >= candNode.size()) {
            for (auto& node : candNode) {
                node->unmatch = true;
            }
            HCCL_VM_ERROR("deadLocking occurs due to mismatch.");
            return HcclResult::HCCL_E_INTERNAL;
        }

        TaskNodePtr curNode = candNode.front();
        candNode.pop_front();
        if (completedNodes.count(curNode) != 0) {
            unmatchedCnt = 0;
            continue;
        }
        // A node can start scheduling only after all of its predecessors have completed.
        if (!AreParentsCompleted(curNode, completedNodes)) {
            candNode.push_back(curNode);
            unmatchedCnt++;
            continue;
        }
        if (curNode->task->GetType() != TaskTypeStub::CCU_GRAPH) {
            completedNodes.insert(curNode);
            addChildNode(candNode, isVisitedNode, curNode);
            unmatchedCnt = 0;
            continue;
        }
        TaskStubCcuGraph* curCcuTask = dynamic_cast<TaskStubCcuGraph*>(curNode->task);
        std::vector<u32> microCodePosInQuePre = curCcuTask->microCodePosInQue;
        HcclResult ret = ProcessCcuNode(curNode, curCcuTask);
        if (ret != HCCL_SUCCESS) {
            HCCL_VM_ERROR("ProcessCcuNode failed");
            if (isPrintCcuGraph) {
                PrintCcuGraph(dummyStart);
            }
            return ret;
        }

        if (microCodePosInQuePre != curCcuTask->microCodePosInQue) {
            // 表示当前子图有匹配到部分微码
            unmatchedCnt = 0;
        } else {
            unmatchedCnt++;
        }

        // 如果整个CCU子图已经匹配完成
        if (curCcuTask->isGenGraphed) {
            for (uint32_t queId = 0; queId < curCcuTask->queueNum_; queId++) {
                CollectBilateralPostInfo(curCcuTask, queId, nullptr, true);
            }
            completedNodes.insert(curNode);
            addChildNode(candNode, isVisitedNode, curNode);
            unmatchedCnt = 0;
            continue;
        }

        // CCU子图未完成匹配，需要放回到队列中，等待下次处理
        candNode.push_back(curNode);
    }

    // 检查是否有未匹配的post节点
    CHK_RET(AllRankParamRecorder::Global()->CheckAllPostMatch());

    if (isPrintCcuGraph) {
        PrintCcuGraph(dummyStart);
    }
    return HCCL_SUCCESS;
}

std::unique_ptr<InstructMapBase> InstructMapFactory::Create(CcuInstrVersion version)
{
    switch (version) {
        case CcuInstrVersion::VERSION_A5:
            return std::make_unique<InstructMapA5>();
#ifdef BUILD_A6_CCU_INSTR
        case CcuInstrVersion::VERSION_A6:
            return std::make_unique<InstructMapA6>();
#endif
        default:
            return nullptr;
    }
}
} // namespace HcclSim
