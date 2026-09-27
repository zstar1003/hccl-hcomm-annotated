/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "check_rank_mem.h"

#include <cstdint>
#include <queue>
#include <set>
#include <unordered_set>

#include "check_utils.h"
#include "sim_log.h"
#include "mem_conflict_check_utils.h"

namespace HcclSim {
bool IsGenFromSync(TaskStub* task)
{
    bool isGenFromSync = false;
    if (task->GetType() == TaskTypeStub::LOCAL_COPY) {
        TaskStubLocalCopy* candLocalCopy = dynamic_cast<TaskStubLocalCopy*>(task);
        isGenFromSync = candLocalCopy->IsGenFromSync();
    } else if (task->GetType() == TaskTypeStub::READ) {
        TaskStubRead* candRead = dynamic_cast<TaskStubRead*>(task);
        isGenFromSync = candRead->IsGenFromSync();
    }
    return isGenFromSync;
}

MemoryStatus operator|(MemoryStatus a, MemoryStatus b)
{
    return static_cast<MemoryStatus>(static_cast<u32>(a) | static_cast<u32>(b));
}

MemoryStatus operator&(MemoryStatus a, MemoryStatus b)
{
    return static_cast<MemoryStatus>(static_cast<u32>(a) & static_cast<u32>(b));
}

MemoryStatus& operator|=(MemoryStatus& a, MemoryStatus b) { return a = a | b; }

// 边界节点，用于将一个原语队列切分为多个碎片
bool IsBoardType(TaskTypeStub type)
{
    const std::set<TaskTypeStub> boardTypes
        = {TaskTypeStub::LOCAL_POST_TO,
           TaskTypeStub::LOCAL_WAIT_FROM,
           TaskTypeStub::LOCAL_POST_TO_SHADOW,
           TaskTypeStub::LOCAL_WAIT_FROM_SHADOW,
           TaskTypeStub::SET_FLAG,
           TaskTypeStub::WAIT_FLAG,
           TaskTypeStub::SET_FLAG_SHADOW,
           TaskTypeStub::WAIT_FLAG_SHADOW,
           TaskTypeStub::PIPE_BARRIER,
           TaskTypeStub::SEND_SYNC,
           TaskTypeStub::RECV_SYNC,
           TaskTypeStub::SEND_SYNC_REDUCE};
    return boardTypes.count(type) != 0;
}

std::string GenFragQueueMemDes(FragQueueMemStatus& fragQueMemStatus)
{
    std::stringstream ret;
    for (auto iter = fragQueMemStatus.begin(); iter != fragQueMemStatus.end(); iter++) {
        BufferType type = iter->first;
        ret << FOUR_INDENT_SPACE << FOUR_INDENT_SPACE << "BufferType is " << type << std::endl;
        for (auto& ele : iter->second) {
            ret << FOUR_INDENT_SPACE << FOUR_INDENT_SPACE << FOUR_INDENT_SPACE << ele.Describe();
        }
    }
    return ret.str();
}

bool CheckRankMem::CheckCcuInvalidNode(bool isCcuGraph, TaskNode* node)
{
    if (node == nullptr) {
        return false;
    }
    if (!isCcuGraph) {
        return false;
    }
    if (node->task->GetType() == TaskTypeStub::LOCAL_POST_TO) {
        auto locPost = dynamic_cast<TaskStubLocalPostTo*>(node->task);
        return locPost->IsInvalidPost();
    }
    if (node->parents.size() != 1) {
        return false;
    }
    return false;
}

void CheckRankMem::GenFragQueueInOneQueue(TaskNode* head, std::set<u32>& seenQueues)
{
    TaskNode* fragStart = nullptr;
    TaskNode* fragEnd = nullptr;

    std::set<TaskNode*> visitedNodes;
    visitedNodes.insert(head);
    std::queue<TaskNode*> walkQue;
    walkQue.push(head);

    // 出于灵活性考虑，一个queue的头节点不一定是Post/Wait类型
    if (IsBoardType(head->task->GetType())) {
        fragStart = head;
    }

    while (!walkQue.empty()) {
        TaskNode* curNode = walkQue.front();
        walkQue.pop();
        for (auto& child : curNode->children) {
            // 不是同一个rank上的不考虑
            if (child->rankIdx != head->rankIdx) {
                continue;
            }

            if ((child->queIdx != head->queIdx) && (seenQueues.count(child->queIdx) == 0)) {
                bool isHeadNode = true;
                for (int i = 0; i < child->parents.size(); i++) {
                    if (child->parents[i]->queIdx == child->queIdx) {
                        isHeadNode = false;
                        break;
                    }
                }
                if (isHeadNode) {
                    seenQueues.insert(child->queIdx);
                    GenFragQueueInOneQueue(child, seenQueues);
                }
            }

            if (child->queIdx != head->queIdx) {
                continue;
            }

            if (visitedNodes.count(child) == 0) {
                walkQue.push(child);
                visitedNodes.insert(child);
            }
        }

        if (curNode == fragStart) {
            continue;
        }

        // 1）遇到边界节点 2）待循环队列已经为空
        if (IsBoardType(curNode->task->GetType()) || walkQue.empty()) {
            if (IsBoardType(curNode->task->GetType())) {
                fragEnd = curNode;
            }
            FragmentQueue ele{head->queIdx, 0, 0, false, fragStart, fragEnd};
            rank2FragQueue_[head->rankIdx].insert(ele);
            fragStart = curNode;
            fragEnd = nullptr;
        }
    }
    return;
}

void CheckRankMem::GenFragQueueInOneRank(TaskNode* node)
{
    u32 queueId = node->queIdx;
    // 头结点链接的都应该是主流，queIdx=0
    std::set<u32> seenQueues;
    seenQueues.insert(queueId);
    GenFragQueueInOneQueue(node, seenQueues);
    return;
}

void CheckRankMem::GenFragQueue()
{
    // 头节点的每个child应该代表了一个rank
    for (auto& child : graphHead_->children) {
        GenFragQueueInOneRank(child);
    }
    return;
}

void CheckRankMem::FindPostWaitNode(
    TaskNode* node, std::set<TaskNode*>& postNodes, std::set<TaskNode*>& waitNodes) const
{
    if (node == nullptr) {
        return;
    }
    switch (GetNodeType(node)) {
        case TaskTypeStub::LOCAL_POST_TO:
        case TaskTypeStub::LOCAL_POST_TO_SHADOW:
            postNodes.insert(node);
            break;
        case TaskTypeStub::LOCAL_WAIT_FROM:
        case TaskTypeStub::LOCAL_WAIT_FROM_SHADOW:
            waitNodes.insert(node);
            break;
        default:
            return;
    }
    return;
}

HcclResult CheckRankMem::FindPostWaitPair(RankId rankId, bool isCcuGraph)
{
    std::set<TaskNode*> postNodes;
    std::set<TaskNode*> waitNodes;
    for (auto& ele : rank2FragQueue_[rankId]) {
        if (!CheckCcuInvalidNode(isCcuGraph, ele.head)) {
            FindPostWaitNode(ele.head, postNodes, waitNodes);
        }
        if (!CheckCcuInvalidNode(isCcuGraph, ele.tail)) {
            FindPostWaitNode(ele.tail, postNodes, waitNodes);
        }
    }

    for (auto& post : postNodes) {
        TaskNode* wait = nullptr;
        for (auto& child : post->children) {
            if (child->queIdx == post->queIdx) {
                continue;
            }
            if (waitNodes.count(child) == 1) {
                wait = child;
            }
        }
        if (wait == nullptr) {
            HCCL_ERROR("Can not find corresponding wait node for post node");
            return HcclResult::HCCL_E_PARA;
        }

        rank2PostWaitPairs_[rankId][post] = wait;
        rank2PostWaitPairs_[rankId][wait] = post;
    }
    return HcclResult::HCCL_SUCCESS;
}

void CheckRankMem::ProcessEqualToTargetStartAddr(
    u64& sliceStartAddr, u64 sliceEndAddr, std::vector<SliceMemoryStatus>& addedEles, MemoryStatus sliceStatus,
    std::set<SliceMemoryStatus>::iterator target) const
{
    u64 eleEndAddr = target->startAddr + target->size;
    // 已经打过相同的标记位，不需要重复打
    if ((static_cast<u32>(target->status) & static_cast<u32>(sliceStatus)) != 0) {
        sliceStartAddr = eleEndAddr;
        return;
    }

    if (sliceEndAddr < eleEndAddr) {
        SliceMemoryStatus sliceMemStatus{sliceEndAddr, eleEndAddr - sliceEndAddr, target->status};
        addedEles.push_back(sliceMemStatus);
        target->size = sliceEndAddr - target->startAddr;
        target->status |= sliceStatus;
        sliceStartAddr = sliceEndAddr;
    } else if (sliceEndAddr == eleEndAddr) {
        target->status |= sliceStatus;
        sliceStartAddr = sliceEndAddr;
    } else { // sliceEndAddr > eleEndAddr
        target->status |= sliceStatus;
        sliceStartAddr = eleEndAddr;
    }
}

void CheckRankMem::ProcessGreatThanTargetStartAddr(
    u64& sliceStartAddr, u64 sliceEndAddr, std::vector<SliceMemoryStatus>& addedEles, MemoryStatus sliceStatus,
    std::set<SliceMemoryStatus>::iterator target) const
{
    u64 eleEndAddr = target->startAddr + target->size;
    // 已经打过相同的标记位，不需要重复打
    if ((static_cast<u32>(target->status) & static_cast<u32>(sliceStatus)) != 0) {
        sliceStartAddr = eleEndAddr;
        return;
    }

    if (sliceEndAddr < eleEndAddr) {
        SliceMemoryStatus sliceMemStatus{sliceStartAddr, sliceEndAddr - sliceStartAddr, target->status | sliceStatus};
        addedEles.push_back(sliceMemStatus);

        SliceMemoryStatus tmp{sliceEndAddr, eleEndAddr - sliceEndAddr, target->status};
        addedEles.push_back(tmp);

        target->size = sliceStartAddr - target->startAddr;
        sliceStartAddr = sliceEndAddr;
    } else if (sliceEndAddr == eleEndAddr) {
        SliceMemoryStatus sliceMemStatus{sliceStartAddr, sliceEndAddr - sliceStartAddr, target->status | sliceStatus};
        addedEles.push_back(sliceMemStatus);

        target->size = sliceStartAddr - target->startAddr;
        sliceStartAddr = sliceEndAddr;
    } else { // sliceEndAddr > eleEndAddr
        SliceMemoryStatus sliceMemStatus{sliceStartAddr, eleEndAddr - sliceStartAddr, target->status | sliceStatus};
        addedEles.push_back(sliceMemStatus);

        target->size = sliceStartAddr - target->startAddr;
        sliceStartAddr = eleEndAddr;
    }
}

void CheckRankMem::GenSliceMemoryInfo(DataSlice& slice, MemoryStatus sliceStatus, FragQueueMemStatus& result)
{
    BufferType sliceBufferType = slice.GetType();
    u64 sliceStartAddr = slice.GetOffset(); // offset
    u64 sliceEndAddr = sliceStartAddr + slice.GetSize();

    std::vector<SliceMemoryStatus> addedEles;
    for (auto ele = result[sliceBufferType].begin(); ele != result[sliceBufferType].end(); ele++) {
        u64 eleEndAddr = ele->startAddr + ele->size;
        // 下面两个判断保证了slice和ele有交集部分
        if ((sliceStartAddr >= eleEndAddr) || (sliceEndAddr <= ele->startAddr)) {
            continue;
        }

        if (sliceStartAddr < ele->startAddr) {
            SliceMemoryStatus sliceMemStatus{sliceStartAddr, ele->startAddr - sliceStartAddr, sliceStatus};
            addedEles.push_back(sliceMemStatus);
            sliceStartAddr = ele->startAddr;
        } else if (sliceStartAddr == ele->startAddr) {
            ProcessEqualToTargetStartAddr(sliceStartAddr, sliceEndAddr, addedEles, sliceStatus, ele);
        } else { // sliceStartAddr > ele->startAddr
            ProcessGreatThanTargetStartAddr(sliceStartAddr, sliceEndAddr, addedEles, sliceStatus, ele);
        }
    }

    if (sliceEndAddr > sliceStartAddr) {
        SliceMemoryStatus sliceMemStatus{sliceStartAddr, sliceEndAddr - sliceStartAddr, sliceStatus};
        addedEles.push_back(sliceMemStatus);
    }

    // 将addedElem给刷新上去
    for (auto& ele : addedEles) {
        result[sliceBufferType].insert(ele);
    }
    return;
}

HcclResult CheckRankMem::GenPrimNodeMemoryInfo(TaskNode* node, FragQueueMemStatus& result)
{
    std::vector<DataSlice> readSlices;
    std::vector<DataSlice> writeSlices;
    GetReadSlice(node, readSlices);
    GetWriteSlice(node, writeSlices);

    for (auto& ele : readSlices) {
        GenSliceMemoryInfo(ele, MemoryStatus::READ, result);
    }

    for (auto& ele : writeSlices) {
        GenSliceMemoryInfo(ele, MemoryStatus::WRITE, result);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CheckRankMem::GenFragQueueMemoryInfo(FragmentQueue& fragQueue, FragQueueMemStatus& result)
{
    std::queue<TaskNode*> walkQueue;
    walkQueue.push(fragQueue.head);

    std::set<TaskNode*> visitedNodes;
    visitedNodes.insert(fragQueue.head);

    while (!walkQueue.empty()) {
        TaskNode* curNode = walkQueue.front();
        walkQueue.pop();

        // 只有主流最前面的原语碎片才会出现头节点为空的情况，主流最前面的原语碎片不会和其他的原语碎片冲突，不生成内存信息也没关系
        if (curNode == nullptr) {
            continue;
        }

        for (auto& child : curNode->children) {
            if (curNode->isAivNode) {
                if (child->rankIdx != curNode->rankIdx or child->blockIdx != curNode->blockIdx
                    or child->pipeIdx != curNode->pipeIdx) {
                    continue;
                }
            } else {
                if (child->rankIdx != curNode->rankIdx or child->queIdx != curNode->queIdx) {
                    continue;
                }
            }

            if (visitedNodes.count(child) == 0) {
                walkQueue.push(child);
                visitedNodes.insert(child);
            }
        }

        GenPrimNodeMemoryInfo(curNode, result);

        if (curNode == fragQueue.tail) {
            break;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CheckRankMem::CompareBufferTypeMemoryInfo(std::set<SliceMemoryStatus>& left, std::set<SliceMemoryStatus>& right)
{
    std::set<SliceMemoryStatus>::iterator leftIter = left.begin();
    std::set<SliceMemoryStatus>::iterator rightIter = right.begin();

    while (leftIter != left.end() && rightIter != right.end()) {
        if (leftIter->size == 0) {
            leftIter++;
            continue;
        }

        if (rightIter->size == 0) {
            rightIter++;
            continue;
        }

        if (leftIter->startAddr + leftIter->size <= rightIter->startAddr) {
            leftIter++;
            continue;
        }

        if (rightIter->startAddr + rightIter->size <= leftIter->startAddr) {
            rightIter++;
            continue;
        }

        if ((leftIter->status & MemoryStatus::WRITE) == MemoryStatus::WRITE
            or (rightIter->status & MemoryStatus::WRITE) == MemoryStatus::WRITE) {
            HCCL_ERROR("there is memory use confilict in two SliceMemoryStatus");
            HCCL_ERROR("one is %s", leftIter->Describe().c_str());
            HCCL_ERROR("another is %s", rightIter->Describe().c_str());
            return HcclResult::HCCL_E_MEMORY;
        }

        if (leftIter->startAddr + leftIter->size <= rightIter->startAddr + rightIter->size) {
            leftIter++;
        } else {
            rightIter++;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

// 仅内部dump使用，不需要对外提供
HcclResult CheckRankMem::CompareBufferTypeMemoryInfo(
    std::set<SliceMemoryStatus>& left, std::set<SliceMemoryStatus>& right, SliceMemoryStatus& conflictEleA,
    SliceMemoryStatus& conflictEleB)
{
    std::set<SliceMemoryStatus>::iterator leftIter = left.begin();
    std::set<SliceMemoryStatus>::iterator rightIter = right.begin();

    while (leftIter != left.end() && rightIter != right.end()) {
        if (leftIter->size == 0) {
            leftIter++;
            continue;
        }

        if (rightIter->size == 0) {
            rightIter++;
            continue;
        }

        if (leftIter->startAddr + leftIter->size <= rightIter->startAddr) {
            leftIter++;
            continue;
        }

        if (rightIter->startAddr + rightIter->size <= leftIter->startAddr) {
            rightIter++;
            continue;
        }

        if ((leftIter->status & MemoryStatus::WRITE) == MemoryStatus::WRITE
            or (rightIter->status & MemoryStatus::WRITE) == MemoryStatus::WRITE) {
            conflictEleA = *leftIter;
            conflictEleB = *rightIter;
            return HcclResult::HCCL_E_MEMORY;
        }

        if (leftIter->startAddr + leftIter->size <= rightIter->startAddr + rightIter->size) {
            leftIter++;
        } else {
            rightIter++;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CheckRankMem::CompareSliceMemoryInfo(FragQueueMemStatus& left, FragQueueMemStatus& right)
{
    for (auto iter = left.begin(); iter != left.end(); iter++) {
        BufferType type = iter->first;
        if (right.count(type) != 0) {
            auto ret = CompareBufferTypeMemoryInfo(iter->second, right[type]);
            if (ret != HcclResult::HCCL_SUCCESS) {
                HCCL_ERROR("failed to check memory type[%d]", static_cast<int>(type));
                return ret;
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

std::string GenConflictDetailInfo(TaskNode* node)
{
    if (node->realPeerNode) {
        return node->realPeerNode->GenPosInfo();
    }
    return node->GenPosInfo();
}

struct BoundaryAdj {
    int to[2] = {-1, -1};
    uint8_t deg = 0;
    u32 markFragIdx = 0;
};
using BitRow = std::vector<uint64_t>;
using ReachMatrix = std::vector<BitRow>;
static void SetBit(BitRow& row, uint32_t idx) { row[idx >> 6] |= (1ULL << (idx & 63)); }
static bool TestBit(const BitRow& row, uint32_t idx) { return ((row[idx >> 6] >> (idx & 63)) & 1ULL) != 0; }
static bool IsPostLike(TaskNode* node)
{
    if (node == nullptr || node->task == nullptr) {
        return false;
    }

    auto type = GetNodeType(node);
    return type == TaskTypeStub::LOCAL_POST_TO || type == TaskTypeStub::LOCAL_POST_TO_SHADOW
           || type == TaskTypeStub::SET_FLAG || type == TaskTypeStub::SET_FLAG_SHADOW || type == TaskTypeStub::SEND_SYNC
           || type == TaskTypeStub::SEND_SYNC_REDUCE;
}
static bool IsWaitLike(TaskNode* node)
{
    if (node == nullptr || node->task == nullptr) {
        return false;
    }

    auto type = GetNodeType(node);
    return type == TaskTypeStub::LOCAL_WAIT_FROM || type == TaskTypeStub::LOCAL_WAIT_FROM_SHADOW
           || type == TaskTypeStub::WAIT_FLAG || type == TaskTypeStub::WAIT_FLAG_SHADOW
           || type == TaskTypeStub::RECV_SYNC;
}
static void AddEdge(BoundaryAdj& adj, u32 u, u32 v, std::vector<u32>& indegree)
{
    if (u == v) {
        return;
    }

    for (uint8_t i = 0; i < adj.deg; ++i) {
        if (adj.to[i] == static_cast<int>(v)) {
            return;
        }
    }

    if (adj.deg >= 2) {
        throw std::logic_error("fragment out-degree > 2");
    }

    adj.to[adj.deg++] = static_cast<int>(v);
    ++indegree[v];
}
static bool CanParallel(const ReachMatrix& reach, u32 i, u32 j)
{
    if (i == j) {
        return false;
    }
    return !TestBit(reach[i], j) && !TestBit(reach[j], i);
}

HcclResult CheckRankMem::CompareFragQueStatus(
    u32 fragQueueSize, std::vector<FragmentQueue>& index2FragQueue, std::vector<std::vector<uint64_t>>& fragQueueMatrix)
{
    std::vector<FragQueueMemStatus> memStatusCache(fragQueueSize);
    std::vector<bool> memStatusValid(fragQueueSize, false);

    auto getOrGenMemInfo = [&](u32 idx) -> std::pair<FragQueueMemStatus*, HcclResult> {
        if (memStatusValid[idx]) {
            return {&memStatusCache[idx], HCCL_SUCCESS};
        }
        HcclResult ret = GenFragQueueMemoryInfo(index2FragQueue[idx], memStatusCache[idx]);
        if (ret != HCCL_SUCCESS) {
            return {nullptr, ret};
        }
        memStatusValid[idx] = true;
        return {&memStatusCache[idx], HCCL_SUCCESS};
    };

    for (u32 i = 0; i < fragQueueSize; i++) {
        for (u32 j = i + 1; j < fragQueueSize; j++) {
            if (!CanParallel(fragQueueMatrix, i, j)) {
                continue;
            }

            // 产生两条queue的内存状态
            auto resultI = getOrGenMemInfo(i);
            if (resultI.second != HCCL_SUCCESS) {
                return resultI.second;
            }

            auto resultJ = getOrGenMemInfo(j);
            if (resultJ.second != HCCL_SUCCESS) {
                return resultJ.second;
            }

            HcclResult ret{HCCL_SUCCESS};
            ret = CompareSliceMemoryInfo(*resultI.first, *resultJ.first);

            if (ret != HcclResult::HCCL_SUCCESS) {
                for (TaskNode* nodeA = index2FragQueue[i].head; nodeA != index2FragQueue[i].tail;) {
                    for (TaskNode* nodeB = index2FragQueue[j].head; nodeB != index2FragQueue[j].tail;) {
                        // 判断是否有冲突，如果有，就dump数据
                        SliceMemoryStatus conflictEleA;
                        SliceMemoryStatus conflictEleB;
                        if (IsConfilictBetweenTwoNodes(nodeA, nodeB, conflictEleA, conflictEleB)) {
                            HCCL_ERROR(
                                "memory conflict between node %s and node %s", nodeA->GenPosInfo().c_str(),
                                nodeB->GenPosInfo().c_str());
                            break;
                        }
                        auto nodeBOld = nodeB;
                        for (auto& child : nodeB->children) {
                            if (nodeB->isAivNode) {
                            } else {
                                if (child->rankIdx != nodeB->rankIdx || child->queIdx != nodeB->queIdx) {
                                    continue;
                                }
                                nodeB = child;
                                break;
                            }
                        }

                        if (nodeB->children.size() == 0 || nodeB == nodeBOld) {
                            break;
                        }
                    }

                    auto nodeAOld = nodeA;
                    for (auto& child : nodeA->children) {
                        if (nodeA->isAivNode) {
                        } else {
                            if (child->rankIdx != nodeA->rankIdx || child->queIdx != nodeA->queIdx) {
                                continue;
                            }
                            nodeA = child;
                            break;
                        }
                    }

                    if (nodeA->children.size() == 0 || nodeA == nodeAOld) {
                        break;
                    }
                }
                return ret;
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

bool CheckRankMem::IsConfilictBetweenTwoNodes(
    TaskNode* nodeA, TaskNode* nodeB, SliceMemoryStatus& conflictEleA, SliceMemoryStatus& conflictEleB)
{
    FragQueueMemStatus resultA;
    FragQueueMemStatus resultB;

    GenPrimNodeMemoryInfo(nodeA, resultA);

    GenPrimNodeMemoryInfo(nodeB, resultB);

    HcclResult ret;
    for (auto iter = resultA.begin(); iter != resultA.end(); iter++) {
        BufferType type = iter->first;
        if (resultB.count(type) != 0) {
            ret = CompareBufferTypeMemoryInfo(iter->second, resultB[type], conflictEleA, conflictEleB);
            if (ret != HcclResult::HCCL_SUCCESS) {
                break;
            }
        }
    }

    if (ret != HcclResult::HCCL_SUCCESS) {
        return true;
    }
    return false;
}

// 判断是否是连接AivEnd的最后一个TaskNode
bool CheckRankMem::IsLastTaskNode(TaskNode* node) { return false; }

HcclResult CheckRankMem::GenFragQueConcurrencyMatrixAndCompare(RankId rankId)
{
    u32 fragQueueSize = rank2FragQueue_[rankId].size();

    auto& postWaitPairs = rank2PostWaitPairs_[rankId];
    std::vector<FragmentQueue> index2FragQueue;
    std::unordered_map<TaskNode*, FragmentQueue> headNode2FragQueue;
    std::unordered_map<TaskNode*, u32> headNode2Index;
    index2FragQueue.reserve(fragQueueSize);

    u32 index = 0;
    for (auto& fragQueue : rank2FragQueue_[rankId]) {
        headNode2FragQueue[fragQueue.head] = fragQueue;
        index2FragQueue.push_back(fragQueue);
        headNode2Index[fragQueue.head] = index;
        index++;
    }

    std::vector<BoundaryAdj> dag(fragQueueSize);
    std::vector<u32> indegree(fragQueueSize, 0);
    for (u32 s = 0; s < fragQueueSize; ++s) {
        TaskNode* tail = index2FragQueue[s].tail;
        if (tail == nullptr || tail->task == nullptr) {
            continue;
        }
        auto nextFragIdxIt = headNode2Index.find(tail);
        if (nextFragIdxIt != headNode2Index.end()) {
            AddEdge(dag[s], s, nextFragIdxIt->second, indegree);
        }

        if (IsPostLike(tail)) {
            auto pairIt = postWaitPairs.find(tail);
            if (pairIt == postWaitPairs.end() || pairIt->second == nullptr) {
                continue;
            }
            TaskNode* waitNode = pairIt->second;
            auto waitFragIdxIt = headNode2Index.find(waitNode);
            if (waitFragIdxIt != headNode2Index.end()) {
                AddEdge(dag[s], s, waitFragIdxIt->second, indegree);
            }
        }
    }
    std::vector<u32> initIndegree = indegree;
    std::queue<u32> topoQue;
    for (u32 i = 0; i < fragQueueSize; ++i) {
        if (indegree[i] == 0) {
            topoQue.push(i);
        }
    }

    std::vector<u32> topoOrder;
    topoOrder.reserve(fragQueueSize);

    while (!topoQue.empty()) {
        u32 u = topoQue.front();
        topoQue.pop();
        topoOrder.push_back(u);

        for (uint8_t k = 0; k < dag[u].deg; ++k) {
            u32 v = static_cast<u32>(dag[u].to[k]);
            if (--indegree[v] == 0) {
                topoQue.push(v);
            }
        }
    }
    int topoSize = topoOrder.size();
    if (topoOrder.size() != fragQueueSize) {
        HCCL_ERROR(
            "[GenFragQueConcurrencyMatrixAndCompare] fragment graph is not a DAG, topo size = %d, ori dag size =  %d",
            topoSize, dag.size());
        return HcclResult::HCCL_E_INTERNAL;
    }
    const u32 wordCount = (fragQueueSize + 63U) >> 6U;
    ReachMatrix fragReach(fragQueueSize, BitRow(wordCount, 0));

    for (auto it = topoOrder.rbegin(); it != topoOrder.rend(); ++it) {
        u32 u = *it;
        SetBit(fragReach[u], dag[u].markFragIdx);
        for (uint8_t k = 0; k < dag[u].deg; ++k) {
            u32 v = static_cast<u32>(dag[u].to[k]);

            // 直接后继 v 必然不可并行
            SetBit(fragReach[u], v);

            // 把 v 的全部后继可达集合并到 u 上
            for (u32 w = 0; w < wordCount; ++w) {
                fragReach[u][w] |= fragReach[v][w];
            }
        }
    }
    return CompareFragQueStatus(fragQueueSize, index2FragQueue, fragReach);
}

// 被读的内存块
void CheckRankMem::GetReadSlice(TaskNode* node, std::vector<DataSlice>& slices)
{
    TaskTypeStub type = node->task->GetType();
    bool isGenFromSync = IsGenFromSync(node->task);
    if (type == TaskTypeStub::LOCAL_COPY) {
        auto task = dynamic_cast<TaskStubLocalCopy*>(node->task);
        if (task->GetSrcSlice().GetType() == BufferType::OUTPUT_AIV && isGenFromSync) {
            return;
        }
        slices.push_back(task->GetSrcSlice());
    } else if (type == TaskTypeStub::LOCAL_REDUCE) {
        auto task = dynamic_cast<TaskStubLocalReduce*>(node->task);
        slices.push_back(task->GetSrcSlice());
    } else if (type == TaskTypeStub::BEING_READ && !isGenFromSync) {
        auto task = dynamic_cast<TaskStubBeingRead*>(node->task);
        slices.push_back(task->GetLocalSlice());
    } else if (type == TaskTypeStub::WRITE) {
        auto task = dynamic_cast<TaskStubWrite*>(node->task);
        slices.push_back(task->GetLocalSlice());
    } else if (type == TaskTypeStub::BEING_READ_REDUCE && !isGenFromSync) {
        auto task = dynamic_cast<TaskStubBeingReadReduce*>(node->task);
        slices.push_back(task->GetLocalSlice());
    } else if (type == TaskTypeStub::WRITE_REDUCE) {
        auto task = dynamic_cast<TaskStubWriteReduce*>(node->task);
        slices.push_back(task->GetLocalSlice());
    }
    return;
}

void CheckRankMem::GetWriteSlice(TaskNode* node, std::vector<DataSlice>& slices)
{
    TaskTypeStub type = node->task->GetType();
    bool isGenFromSync = IsGenFromSync(node->task);
    if (type == TaskTypeStub::LOCAL_COPY) {
        auto task = dynamic_cast<TaskStubLocalCopy*>(node->task);
        if (task->GetDstSlice().GetType() == BufferType::OUTPUT_AIV && isGenFromSync) {
            return;
        }
        slices.push_back(task->GetDstSlice());
    } else if (type == TaskTypeStub::LOCAL_REDUCE && !isGenFromSync) {
        auto task = dynamic_cast<TaskStubLocalReduce*>(node->task);
        slices.push_back(task->GetDstSlice());
    } else if (type == TaskTypeStub::BEING_WRITTEN && !isGenFromSync) {
        auto task = dynamic_cast<TaskStubBeingWritten*>(node->task);
        slices.push_back(task->GetLocalSlice());
    } else if (type == TaskTypeStub::READ) {
        auto task = dynamic_cast<TaskStubRead*>(node->task);
        slices.push_back(task->GetLocalSlice());
    } else if (type == TaskTypeStub::BEING_WRITTEN_REDUCE && !isGenFromSync) {
        auto task = dynamic_cast<TaskStubBeingWrittenReduce*>(node->task);
        slices.push_back(task->GetLocalSlice());
    } else if (type == TaskTypeStub::READ_REDUCE) {
        auto task = dynamic_cast<TaskStubReadReduce*>(node->task);
        slices.push_back(task->GetLocalSlice());
    }
    return;
}

HcclResult CheckRankMem::Execute()
{
    // 如果有AivTask，不管是纯AIV算法还是混编算法，均先处理AIV子图内层的冲突校验
    if (graphHead_->hasCcuTask) {
        CHK_RET(CcuGraphMemCheck());
    }
    // 先从整图中提取信息
    GenFragQueue();

    // 从每个rank中提取post/wait队列
    for (auto& child : graphHead_->children) {
        RankId rankId = child->rankIdx;
        CHK_RET(FindPostWaitPair(rankId));
        auto ret = GenFragQueConcurrencyMatrixAndCompare(rankId);
        if (ret != HcclResult::HCCL_SUCCESS) {
            // DataDumper::Global()->SetResultStatus(gui::ResultStatus::MEMORY_CONFLICT);
            HCCL_ERROR("check rank memory conflict failed for rank %d", rankId);
            return ret;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CheckRankMem::CcuGraphMemCheck()
{
    for (auto& child : graphHead_->children) {
        uint32_t queueId = child->queIdx;
        RankId rankId = child->rankIdx;
        // // 头结点链接的都应该是主流，queIdx=0
        if (queueId != 0) {
            HCCL_ERROR("[CcuGraphMemCheck] The node connecting the head node should be the mainstream.");
            return HcclResult::HCCL_E_PARA;
        }

        std::queue<TaskNodePtr> graphNodeQue;
        std::set<TaskNodePtr> isVisited;
        graphNodeQue.push(child);
        isVisited.insert(child);
        int numCcuGraph{0};

        while (!graphNodeQue.empty()) {
            TaskNodePtr currNode = graphNodeQue.front();
            graphNodeQue.pop();
            CHK_RET(ProceedNode(currNode, graphNodeQue, isVisited));

            if (currNode->task->GetType() == TaskTypeStub::CCU_GRAPH) {
                auto ccuNode = GetCcuTaskHead(currNode);
                CHK_RET(CcuGraphMemCheckProc(rankId, queueId, ccuNode));
            }
            numCcuGraph++;
        }
        HCCL_VM_INFO("{} CcuGraphMemCheck proc success. ccu graph num: {}", rankId, numCcuGraph);
    }
    ClearCcuData();

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CheckRankMem::CollectCcuQueueGraph(TaskNode* ccuNode, RankId rankId, CcuQueueGraph& graph)
{
    std::queue<TaskNode*> walkQue;
    std::unordered_set<TaskNode*> visited;

    for (auto* child : ccuNode->children) {
        if (child == nullptr || child->rankIdx != rankId) {
            continue;
        }
        if (visited.insert(child).second) {
            walkQue.push(child);
        }
    }

    std::vector<TaskNode*> allNodes;
    while (!walkQue.empty()) {
        TaskNode* curNode = walkQue.front();
        walkQue.pop();
        allNodes.push_back(curNode);

        int sameQueueParentCnt = 0;
        for (auto* parent : curNode->parents) {
            if (parent != nullptr && parent->rankIdx == curNode->rankIdx && parent->queIdx == curNode->queIdx) {
                ++sameQueueParentCnt;
            }
        }

        const bool isHead = (sameQueueParentCnt == 0);
        graph.nodeIsQueueHead[curNode] = isHead ? 1 : 0;
        if (isHead) {
            auto it = graph.queueId2Head.find(curNode->queIdx);
            if (it == graph.queueId2Head.end()) {
                graph.queueId2Head[curNode->queIdx] = curNode;
            } else if (it->second != curNode) {
                HCCL_ERROR(
                    "[CollectCcuQueueGraph] queue %u has multiple head nodes in one ccu sub graph", curNode->queIdx);
                return HcclResult::HCCL_E_INTERNAL;
            }
        }

        for (auto* child : curNode->children) {
            if (child == nullptr || child->rankIdx != rankId) {
                continue;
            }
            if (visited.insert(child).second) {
                walkQue.push(child);
            }
        }
    }

    std::unordered_set<u32> entrySeen;
    for (auto* child : ccuNode->children) {
        if (child == nullptr || child->rankIdx != rankId) {
            continue;
        }
        if (graph.nodeIsQueueHead[child] != 0 && entrySeen.insert(child->queIdx).second) {
            graph.entryQueueIds.push_back(child->queIdx);
        }
    }

    for (auto* curNode : allNodes) {
        for (auto* child : curNode->children) {
            if (child == nullptr || child->rankIdx != rankId) {
                continue;
            }
            if (child->queIdx == curNode->queIdx) {
                continue;
            }
            if (graph.nodeIsQueueHead[child] == 0) {
                continue;
            }
            graph.queueAdj[curNode->queIdx].push_back(child->queIdx);
        }
    }

    for (auto& it : graph.queueAdj) {
        auto& vec = it.second;
        std::sort(vec.begin(), vec.end());
        vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
    }

    return HcclResult::HCCL_SUCCESS;
}
void CheckRankMem::GenFragQueueInOneQueueOnly(TaskNode* head)
{
    TaskNode* fragStart = nullptr;
    TaskNode* fragEnd = nullptr;

    std::queue<TaskNode*> walkQue;
    std::unordered_set<TaskNode*> visitedNodes;
    walkQue.push(head);
    visitedNodes.insert(head);

    if (IsBoardType(head->task->GetType())) {
        fragStart = head;
    }

    while (!walkQue.empty()) {
        TaskNode* curNode = walkQue.front();
        walkQue.pop();
        std::vector<TaskNode*> sameQueueChildren{};
        int boardChildNum = 0;

        for (auto& child : curNode->children) {
            if (child == nullptr) {
                continue;
            }
            if (child->rankIdx != head->rankIdx || child->queIdx != head->queIdx) {
                continue;
            }
            if (child->queIdx == head->queIdx) {
                sameQueueChildren.push_back(child);
                if (IsBoardType(child->task->GetType()) && IsBoardType(curNode->task->GetType())) {
                    boardChildNum++;
                }
            }
        }

        for (size_t i = 0; i < sameQueueChildren.size(); i++) {
            TaskNode* child = sameQueueChildren[i];
            if (visitedNodes.count(child) == 0) {
                if (boardChildNum > 1) {
                    auto type = child->task->GetType();
                    if (type == TaskTypeStub::LOCAL_WAIT_FROM || type == TaskTypeStub::LOCAL_WAIT_FROM_SHADOW) {
                        HCCL_VM_DEBUG(
                            "ccu multi samequeue node found, node: {} "
                            "ignore child: {} addr: {}",
                            curNode->task->Describe(), child->task->Describe(), static_cast<void*>(child));
                        boardChildNum--;
                        continue;
                    } else {
                        walkQue.push(child);
                        visitedNodes.insert(child);
                    }
                } else {
                    walkQue.push(child);
                    visitedNodes.insert(child);
                }
            }
        }

        if (curNode == fragStart) {
            continue;
        }

        if (IsBoardType(curNode->task->GetType()) || walkQue.empty()) {
            if (IsBoardType(curNode->task->GetType())) {
                fragEnd = curNode;
            }
            FragmentQueue ele{head->queIdx, 0, 0, false, fragStart, fragEnd};
            rank2FragQueue_[head->rankIdx].insert(ele);
            fragStart = curNode;
            fragEnd = nullptr;
        }
    }
}
static void CollectReachableQueues(
    u32 rootQueueId, const std::unordered_map<u32, std::vector<u32>>& queueAdj,
    std::unordered_set<u32>& processedQueues, std::vector<u32>& componentQueues)
{
    std::queue<u32> walkQue;
    walkQue.push(rootQueueId);
    processedQueues.insert(rootQueueId);

    while (!walkQue.empty()) {
        u32 curQueueId = walkQue.front();
        walkQue.pop();
        componentQueues.push_back(curQueueId);

        auto it = queueAdj.find(curQueueId);
        if (it == queueAdj.end()) {
            continue;
        }

        for (u32 nextQueueId : it->second) {
            if (processedQueues.insert(nextQueueId).second) {
                walkQue.push(nextQueueId);
            }
        }
    }
}

HcclResult CheckRankMem::CcuGraphMemCheckProc(RankId rankId, uint32_t queueId, TaskNodePtr ccuNode)
{
    CcuQueueGraph queueGraph;
    CHK_RET(CollectCcuQueueGraph(ccuNode, rankId, queueGraph));

    std::unordered_set<u32> processedQueues;
    std::vector<u32> componentQueues;

    int ccuHeadChildNum{0};
    for (u32 entryQueueId : queueGraph.entryQueueIds) {
        ccuHeadChildNum++;
        if (processedQueues.count(entryQueueId) != 0) {
            continue;
        }
        CollectReachableQueues(entryQueueId, queueGraph.queueAdj, processedQueues, componentQueues);
        processedQueues.clear();
    }
    std::sort(componentQueues.begin(), componentQueues.end());
    componentQueues.erase(std::unique(componentQueues.begin(), componentQueues.end()), componentQueues.end());

    for (u32 componentQueueId : componentQueues) {
        auto headIt = queueGraph.queueId2Head.find(componentQueueId);
        if (headIt == queueGraph.queueId2Head.end() || headIt->second == nullptr) {
            HCCL_ERROR("can not find queue head for queue %u", componentQueueId);
            return HcclResult::HCCL_E_INTERNAL;
        }
        GenFragQueueInOneQueueOnly(headIt->second);
    }
    CHK_RET(FindPostWaitPair(rankId, true));
    CHK_RET(GenFragQueConcurrencyMatrixAndCompare(rankId));
    ClearCcuData();

    HCCL_VM_DEBUG("ccuHeadChildNum= {}", ccuHeadChildNum);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CheckRankMem::ProceedNode(TaskNodePtr currNode, std::queue<TaskNodePtr>& graphNodeQue, std::set<TaskNodePtr>& isVisited)
{
    for (auto childIter = currNode->children.begin(); childIter != currNode->children.end(); childIter++) {
        if (!isVisited.count(*childIter) && (*childIter)->rankIdx == currNode->rankIdx) {
            graphNodeQue.push((*childIter));
            isVisited.insert((*childIter));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void CheckRankMem::ClearCcuData()
{
    for (auto& fragQueue : rank2FragQueue_) {
        fragQueue.second.clear();
    }
    rank2FragQueue_.clear();

    for (auto& postWaitPairs : rank2PostWaitPairs_) {
        postWaitPairs.second.clear();
    }
    rank2PostWaitPairs_.clear();
}
} // namespace HcclSim
