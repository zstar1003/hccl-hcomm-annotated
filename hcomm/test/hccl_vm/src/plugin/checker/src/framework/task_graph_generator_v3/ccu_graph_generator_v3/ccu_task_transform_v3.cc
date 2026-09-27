/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_task_transform_v3.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "ccu_all_rank_param_recorder_v3.h"
#include "sim_log.h"
#include "storage_manager.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {

    std::unique_ptr<InstructMapBase> g_instrMap;

    // 作用：读取 post 节点当前还未被 wait 消费掉的 CKE mask。
    u32 GetPostRemainingCkeMask(TaskNode* post)
    {
        return AllRankParamRecorder::Global()->GetPostRemainingCkeMask(post);
    }

    // 作用：更新 post 节点当前还未被 wait 消费掉的 CKE mask。
    void SetPostRemainingCkeMask(TaskNode* post, u32 remainingCkeMask)
    {
        AllRankParamRecorder::Global()->SetPostRemainingCkeMask(post, remainingCkeMask);
    }

    // 作用：按 wait mask 生成本地/远端 wait 节点，并补齐 post 到 wait 的依赖边。
    void GenWaitNode(CcuGraphStateV3* curCcuTask, uint32_t queId, uint16_t waitCKEId, uint16_t waitCKEMask)
    {
        RankId rankId = curCcuTask->GetRankId();
        uint32_t dieId;
        curCcuTask->GetDieId(queId, dieId);

        uint16_t localWaitMask = 0;
        uint16_t remoteWaitMask = 0;
        TaskNode* localWaitNode = nullptr;
        TaskNode* remoteWaitNode = nullptr;

        // 只有目标 CKE bit 全部就绪后才真正生成 wait 节点；此时会把待消费的 CKE mask bit
        // 反查回它实际依赖的那些 post 节点。
        std::set<TaskNode*>& seenPosts = AllRankParamRecorder::Global()->seenPost[rankId][dieId][waitCKEId];
        std::vector<TaskNode*> localPosts;
        for (auto* post : seenPosts) {
            u32 remainingCkeMask = GetPostRemainingCkeMask(post);
            uint16_t curPostMask = remainingCkeMask & waitCKEMask;
            if (curPostMask == 0) {
                continue;
            }

            auto postMeta = AllRankParamRecorder::Global()->GetPostNodeMeta(post);
            if (postMeta != nullptr && postMeta->isLocal) {
                localWaitMask = localWaitMask | curPostMask;
                localPosts.push_back(post);
            } else {
                remoteWaitMask = remoteWaitMask | curPostMask;
            }
        }

        if (remoteWaitMask != 0) {
            remoteWaitNode = AddWait(rankId, queId, curCcuTask, dieId, waitCKEId, remoteWaitMask);
        }

        if (localWaitMask != 0) {
            localWaitNode = AddLocalWait(rankId, queId, curCcuTask, dieId, waitCKEId, localWaitMask);
            for (auto* locPost : localPosts) {
                curCcuTask->localPostWaitPairs_[locPost] = localWaitNode;
            }
        }

        for (auto it = seenPosts.begin(); it != seenPosts.end();) {
            auto* post = *it;
            u32 remainingCkeMask = GetPostRemainingCkeMask(post);
            uint16_t curPostMask = remainingCkeMask & waitCKEMask;
            if (curPostMask == 0) {
                ++it;
                continue;
            }
            auto postMeta = AllRankParamRecorder::Global()->GetPostNodeMeta(post);
            if (postMeta != nullptr && postMeta->isLocal) {
                AddNodeRelation(post, localWaitNode);
            } else {
                AddNodeRelation(post, remoteWaitNode);
                if (postMeta != nullptr) {
                    curCcuTask->SetNodePeerRank(remoteWaitNode, postMeta->recordRankId);
                } else {
                    curCcuTask->SetNodePeerRank(remoteWaitNode, post->GetPosition().rankId);
                }
                (void)CollectBilateralWaitInfo(curCcuTask, queId, remoteWaitNode);
            }
            remainingCkeMask = remainingCkeMask & (~waitCKEMask);
            SetPostRemainingCkeMask(post, remainingCkeMask);
            if (remainingCkeMask == 0) {
                it = seenPosts.erase(it);
            } else {
                ++it;
            }
        }
    }

    // 作用：检查当前 wait 条件是否满足，满足时生成 wait 节点，不满足时暂停当前 queue。
    HcclResult ProcessWaitMask(
        RankId rankId, uint32_t dieId, CcuGraphStateV3* curCcuTask, uint32_t queId, uint16_t waitCKEId,
        uint16_t waitCKEMask, bool& isContinue)
    {
        HCCL_VM_DEBUG(
            "Evaluate CCU wait condition, rankId={}, dieId={}, queueId={}, waitCkeId={}, "
            "requiredMask=0x{:04x}",
            rankId, static_cast<uint32_t>(dieId), queId, waitCKEId, waitCKEMask);
        if (waitCKEMask != 0x0000) {
            uint16_t ckeValue = 0;
            CHK_RET(AllRankParamRecorder::Global()->GetCKE(rankId, dieId, waitCKEId, ckeValue));
            if ((ckeValue & waitCKEMask) != waitCKEMask) {
                // 当前 queue 在这里暂停，直到生产侧把所需的 wait bit 全部 post 完成。
                isContinue = false;
                HCCL_VM_DEBUG(
                    "Pause this CCU queue because the wait condition is not satisfied yet, "
                    "rankId={}, dieId={}, queueId={}, waitCkeId={}, currentCkeValue=0x{:04x}, requiredMask=0x{:04x}",
                    rankId, static_cast<uint32_t>(dieId), queId, waitCKEId, ckeValue, waitCKEMask);
                return HCCL_SUCCESS;
            }
            GenWaitNode(curCcuTask, queId, waitCKEId, waitCKEMask);
        }
        return HCCL_SUCCESS;
    }

    // 作用：调用具体指令映射，把一条 CCU 指令转换成任务图节点。
    HcclResult
    TransformInstr(const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue)
    {
        if (g_instrMap == nullptr) {
            HCCL_VM_ERROR(
                "{} Failed to create the CCU instruction translator, rankId={}, queueId={}",
                MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR).c_str(),
                curCcuTask == nullptr ? std::string("null") : std::to_string(curCcuTask->GetRankId()), queId);
            return HCCL_E_INTERNAL;
        }
        const uint16_t instructionHeader = static_cast<uint16_t>(instr->header.header);
        if (!g_instrMap->IsSupported(instructionHeader)) {
            HCCL_VM_ERROR(
                "{} This CCU instruction type is not supported by CheckerV3 graph expansion, "
                "rankId={}, queueId={}, instructionHeader=0x{:04x}",
                MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED).c_str(),
                curCcuTask == nullptr ? std::string("null") : std::to_string(curCcuTask->GetRankId()), queId,
                instructionHeader);
            return HCCL_E_INTERNAL;
        }
        return g_instrMap.get()->Transform(instr, curCcuTask, queId, isContinue, nullptr);
    }

    // 作用：根据设备类型选择当前使用的 CCU 指令版本。
    CcuInstrVersion GetCcuInstrVersion()
    {
        DevType devType = AllRankParamRecorder::Global()->GetDevType();
        if (devType == DevType::DEV_TYPE_950) {
            return CcuInstrVersion::VERSION_A5;
#ifdef BUILD_A6_CCU_INSTR
        } else if (devType == DevType::DEV_TYPE_960) {
            return CcuInstrVersion::VERSION_A6;
#endif
        }
        return CcuInstrVersion::VERSION_A5;
    }

    // 作用：按指令顺序展开单个 queue 的微码序列。
    HcclResult TransformInstrQue(TaskNode* node, CcuGraphStateV3* curCcuTask, uint32_t queId)
    {
        if (curCcuTask->instrQueStatus[queId]) {
            return HCCL_SUCCESS;
        }
        g_instrMap = InstructMapFactory::Create(GetCcuInstrVersion());

        bool isContinue = true;
        hcomm::CcuRep::CcuInstrInfo& microCodeQue = curCcuTask->instrInfo[queId];
        auto endInstrId = curCcuTask->GetMissionEndInstrId(queId);
        u32& pos = curCcuTask->microCodePosInQue[queId];
        // 单个 queue 严格按指令顺序推进；跨 queue 的协同只通过生成出来的 wait/post 边表达。
        while (pos < endInstrId) {
            u32 prePos = pos;
            const uint32_t instrId = curCcuTask->startInstrIdInQue[queId] + pos;
            const uint16_t instructionHeader = static_cast<uint16_t>(microCodeQue.instrVec[pos].header.header);
            HCCL_VM_DEBUG(
                "Translating one CCU instruction, graph={}, instructionId={}, "
                "instructionHeader=0x{:04x}",
                curCcuTask->Describe(), instrId, instructionHeader);
            HcclResult ret = TransformInstr(&microCodeQue.instrVec[pos], curCcuTask, queId, isContinue);
            if (ret != HCCL_SUCCESS) {
                HCCL_VM_ERROR(
                    "{} Failed to translate one CCU instruction, rankId={}, queueId={}, "
                    "instructionId={}, instructionHeader=0x{:04x}, ret={}, graph={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_TRANSLATE_FAILED).c_str(), curCcuTask->GetRankId(), queId,
                    instrId, instructionHeader, static_cast<uint32_t>(ret), curCcuTask->Describe().c_str());
                return ret;
            }
            if (!isContinue) {
                return HCCL_SUCCESS;
            }
            if (pos == prePos) {
                pos++;
            }
        }

        curCcuTask->instrQueStatus[queId] = true;
        curCcuTask->isGenGraphed
            = std::all_of(curCcuTask->instrQueStatus.begin(), curCcuTask->instrQueStatus.end(), [](bool b) {
                  return b;
              });

        HCCL_VM_DEBUG("Finished translating one CCU queue, rankId={}, queueId={}", curCcuTask->GetRankId(), queId);
        return HCCL_SUCCESS;
    }

    // 作用：轮流推进当前 CCU 图中的所有 queue，直到各自无法继续或完成展开。
    HcclResult ProcessCcuNode(TaskNode* node, CcuGraphStateV3* curCcuTask)
    {
        curCcuTask->queueNum_ = static_cast<uint32_t>(curCcuTask->instrInfo.size());
        uint32_t rankSize = curCcuTask->GetStorageManager().GetRankSize();
        curCcuTask->bilateralPart1_.resize(curCcuTask->queueNum_);
        curCcuTask->bilateralPart2_.resize(curCcuTask->queueNum_);
        curCcuTask->bilateralNodes_.resize(curCcuTask->queueNum_);
        curCcuTask->waitInfoTmp_.resize(rankSize);
        curCcuTask->postInfoTmp_.resize(rankSize);
        // raw 成图阶段会轮流推进每个 queue，这样被阻塞的 queue 可以等待，
        // 其他 queue 则继续前进并发布所需的 post bit。
        for (uint32_t queId = 0; queId < curCcuTask->queueNum_; queId++) {
            CHK_RET(TransformInstrQue(node, curCcuTask, queId));
        }
        return HCCL_SUCCESS;
    }

    namespace {

        struct CcuGraphRuntimeV3 {
            TaskCcuGraph* ccuGraph{nullptr};
            std::unique_ptr<CcuGraphStateV3> state;
            std::vector<TaskNode*> originalChildren;
            bool generated{false};
            bool finalized{false};
        };

        enum class QueueReuseUnitKindV3 : uint8_t {
            LOOP = 0,
            ASYNC = 1,
        };

        struct QueueReuseUnitV3 {
            QueueReuseUnitKindV3 kind{QueueReuseUnitKindV3::LOOP};
            TaskNode* head{nullptr};
            TaskNode* tail{nullptr};
            // 保存该并行单元改造前由 tail 直接连接的外部后继，出口重连只能使用这组节点。
            std::vector<TaskNode*> afterNodes;
            NodeId startNodeId{INVALID_NODE_ID};
            NodeId endNodeId{INVALID_NODE_ID};
            NodeId ownerLoopStartNodeId{INVALID_NODE_ID};
            QueueId queueId{INVALID_QUEUE_ID};
            std::set<TaskNode*> nodes;
        };

        struct LoopParallelGroupV3 {
            QueueId originalQueueId{INVALID_QUEUE_ID};
            std::vector<TaskNode*> beforeNodes;
            std::vector<TaskNode*> afterNodes;
            std::vector<QueueReuseUnitV3> units;
        };

        struct AsyncParallelGroupV3 {
            QueueId originalQueueId{INVALID_QUEUE_ID};
            std::vector<TaskNode*> beforeNodes;
            std::vector<TaskNode*> afterNodes;
            std::vector<QueueReuseUnitV3> units;
        };

        using CompressedLoopBoundaryV3 = std::map<TaskNode*, std::vector<TaskNode*>>;

        class QueueIdReuseAllocatorV3 {
        public:
            // 作用：初始化 queue 复用分配器的起始编号范围。
            explicit QueueIdReuseAllocatorV3(uint32_t baseQueueCount)
            {
                curMaxQueueId_ = (baseQueueCount == 0U) ? INVALID_QUEUE_ID : static_cast<QueueId>(baseQueueCount - 1U);
            }

            // 作用：申请一个可用的 queueId，优先复用已释放的编号。
            QueueId Acquire()
            {
                if (!freeQueueIds_.empty()) {
                    const auto iter = freeQueueIds_.begin();
                    const QueueId queueId = *iter;
                    freeQueueIds_.erase(iter);
                    return queueId;
                }
                curMaxQueueId_ = (curMaxQueueId_ == INVALID_QUEUE_ID) ? 0U : static_cast<QueueId>(curMaxQueueId_ + 1U);
                return curMaxQueueId_;
            }

            // 作用：释放一个 queueId，供后续并行单元复用。
            void Release(QueueId queueId)
            {
                if (queueId == INVALID_QUEUE_ID) {
                    return;
                }
                freeQueueIds_.insert(queueId);
            }

            // 作用：返回当前分配结束后下一可用的 queueId 上界。
            QueueId NextQueueId() const
            {
                return (curMaxQueueId_ == INVALID_QUEUE_ID) ? 0U : static_cast<QueueId>(curMaxQueueId_ + 1U);
            }

        private:
            QueueId curMaxQueueId_{INVALID_QUEUE_ID};
            std::set<QueueId> freeQueueIds_;
        };

        // 作用：把未访问过的子节点加入候选队列，供外层遍历继续推进。
        void AddReachableChildren(
            std::deque<TaskNode*>& queue, std::set<TaskNode*>& visited, const std::vector<TaskNode*>& children)
        {
            for (TaskNode* child : children) {
                if (child == nullptr) {
                    continue;
                }
                if (visited.insert(child).second) {
                    queue.push_back(child);
                }
            }
        }

        // 作用：判断一个节点的所有父节点是否都已经完成处理。
        bool AreParentsCompleted(const TaskNode* node, const std::set<TaskNode*>& completedNodes)
        {
            if (node == nullptr) {
                return false;
            }
            for (TaskNode* parent : node->GetParents()) {
                if (parent == nullptr || completedNodes.count(parent) == 0) {
                    return false;
                }
            }
            return true;
        }

        // 作用：同步更新节点本体和 trace 中记录的 queueId。
        void SetNodeQueueIdV3(TaskNode* node, QueueId queueId)
        {
            if (node == nullptr) {
                return;
            }

            TaskLocation taskLoc = node->GetLocation();
            taskLoc.queueId = queueId;
            node->SetLocation(taskLoc);

            if (!node->HasCcuTrace()) {
                return;
            }

            CcuTraceInfo trace = node->GetCcuTrace();
            trace.queueId = queueId;
            trace.position.queueId = queueId;
            trace.taskLoc.queueId = queueId;
            node->SetCcuTrace(std::move(trace));
        }

        // 作用：判断节点是否属于 async 并行化改造的候选节点。
        bool IsAsyncNodeV3(const CcuGraphStateV3& state, const TaskNode* node)
        {
            if (node == nullptr) {
                return false;
            }
            const CcuNodeRoleV3 role = state.GetNodeRole(node);
            return role == CcuNodeRoleV3::LOCAL_COPY || role == CcuNodeRoleV3::LOCAL_REDUCE
                   || role == CcuNodeRoleV3::LOCAL_BATCH_REDUCE || role == CcuNodeRoleV3::READ
                   || role == CcuNodeRoleV3::READ_REDUCE || role == CcuNodeRoleV3::WRITE
                   || role == CcuNodeRoleV3::WRITE_REDUCE;
        }

        // 作用：根据 CCU 通知参数判断 RECORD 是否对应 WAIT；这里只判断同步语义，不负责判断图上是否已有边。
        bool IsCcuRecordWaitPairV3(const TaskNode* recordNode, const TaskNode* waitNode)
        {
            if (recordNode == nullptr || waitNode == nullptr || recordNode->GetType() != TaskType::RECORD
                || waitNode->GetType() != TaskType::WAIT) {
                return false;
            }

            const auto* record = dynamic_cast<const TaskRecordCCU*>(recordNode);
            const auto* wait = dynamic_cast<const TaskWaitCCU*>(waitNode);
            if (record == nullptr || wait == nullptr) {
                return false;
            }

            const CcuNotify& recordNotify = record->GetNotify();
            const CcuNotify& waitNotify = wait->GetNotify();
            return recordNotify.waitRankId == waitNotify.waitRankId && recordNotify.dieId == waitNotify.dieId
                   && recordNotify.ckeId == waitNotify.ckeId && (recordNotify.ckeMask & waitNotify.ckeMask) != 0;
        }

        // 作用：确认原始图中确实存在 parent -> child，避免仅凭通知参数凭空恢复一条新边。
        bool HasNodeEdgeV3(const TaskNode* parentNode, const TaskNode* childNode)
        {
            if (parentNode == nullptr || childNode == nullptr) {
                return false;
            }
            const auto& children = parentNode->GetChildren();
            return std::find(children.begin(), children.end(), childNode) != children.end();
        }

        // 作用：在目标数组中去重追加一个节点。
        void AppendNodeIfAbsentV3(std::vector<TaskNode*>& nodes, TaskNode* node)
        {
            if (node == nullptr) {
                return;
            }
            if (std::find(nodes.begin(), nodes.end(), node) == nodes.end()) {
                nodes.push_back(node);
            }
        }

        // 作用：收集某节点在指定 queue 上的父节点。
        std::vector<TaskNode*> GetSameQueueParentsV3(const TaskNode* node, QueueId queueId)
        {
            std::vector<TaskNode*> parents;
            if (node == nullptr) {
                return parents;
            }
            for (TaskNode* parent : node->GetParents()) {
                if (parent == nullptr || parent->GetLocation().queueId != queueId) {
                    continue;
                }
                AppendNodeIfAbsentV3(parents, parent);
            }
            return parents;
        }

        // 作用：收集某节点所有不在排除集合中的父节点。
        std::vector<TaskNode*> GetParentsOutsideNodesV3(const TaskNode* node, const std::set<TaskNode*>& excludedNodes)
        {
            std::vector<TaskNode*> parents;
            if (node == nullptr) {
                return parents;
            }
            for (TaskNode* parent : node->GetParents()) {
                if (parent == nullptr || excludedNodes.count(parent) != 0) {
                    continue;
                }
                AppendNodeIfAbsentV3(parents, parent);
            }
            return parents;
        }

        // 作用：收集某节点在指定 queue 上的子节点。
        std::vector<TaskNode*> GetSameQueueChildrenV3(const TaskNode* node, QueueId queueId)
        {
            std::vector<TaskNode*> children;
            if (node == nullptr) {
                return children;
            }
            for (TaskNode* child : node->GetChildren()) {
                if (child == nullptr || child->GetLocation().queueId != queueId) {
                    continue;
                }
                AppendNodeIfAbsentV3(children, child);
            }
            return children;
        }

        // 作用：收集某节点所有不在排除集合中的子节点。
        std::vector<TaskNode*> GetChildrenOutsideNodesV3(const TaskNode* node, const std::set<TaskNode*>& excludedNodes)
        {
            std::vector<TaskNode*> children;
            if (node == nullptr) {
                return children;
            }
            for (TaskNode* child : node->GetChildren()) {
                if (child == nullptr || excludedNodes.count(child) != 0) {
                    continue;
                }
                AppendNodeIfAbsentV3(children, child);
            }
            return children;
        }

        // 作用：判断一个节点是否为 CCU 子图的结束节点。
        bool IsCcuSubGraphEndNodeV3(const TaskNode* node)
        {
            if (node == nullptr || node->GetType() != TaskType::END) {
                return false;
            }

            const auto* endNode = dynamic_cast<const TaskEnd*>(node);
            return endNode != nullptr && endNode->GetBoundaryType() == BoundaryType::CCU_SUB_GRAPH;
        }

        // 作用：找到同 queue 上按节点编号递增的下一个串行子节点。
        TaskNode* GetSerialSameQueueChildV3(const TaskNode* node, QueueId queueId)
        {
            if (node == nullptr) {
                return nullptr;
            }

            TaskNode* serialChild = nullptr;
            for (TaskNode* child : node->GetChildren()) {
                if (child == nullptr || child->GetLocation().queueId != queueId
                    || child->GetNodeId() <= node->GetNodeId()) {
                    continue;
                }
                if (serialChild == nullptr || child->GetNodeId() < serialChild->GetNodeId()) {
                    serialChild = child;
                }
            }
            return serialChild;
        }

        // 作用：计算一个并行单元何时可以释放其占用的 queueId。
        NodeId GetReleaseNodeIdV3(const std::vector<TaskNode*>& afterNodes)
        {
            if (afterNodes.empty()) {
                return std::numeric_limits<NodeId>::max();
            }

            NodeId releaseNodeId = INVALID_NODE_ID;
            for (TaskNode* node : afterNodes) {
                if (node == nullptr) {
                    continue;
                }
                releaseNodeId = std::max(releaseNodeId, node->GetNodeId());
            }
            return (releaseNodeId == INVALID_NODE_ID) ? std::numeric_limits<NodeId>::max() : releaseNodeId;
        }

        // 作用：收集同一 queue 内从 loopStart 到 loopEnd 的 loop body 节点集合。
        void CollectLoopBodyNodesSameQueueV3(
            TaskNode* loopStart, TaskNode* loopEnd, QueueId originalQueueId, std::set<TaskNode*>& nodes)
        {
            if (loopStart == nullptr || loopEnd == nullptr) {
                return;
            }

            std::deque<TaskNode*> candidateNodes;
            std::set<TaskNode*> visitedNodes;
            candidateNodes.push_back(loopStart);
            visitedNodes.insert(loopStart);

            while (!candidateNodes.empty()) {
                TaskNode* curNode = candidateNodes.front();
                candidateNodes.pop_front();
                if (curNode == nullptr || curNode->GetLocation().queueId != originalQueueId) {
                    continue;
                }

                nodes.insert(curNode);
                if (curNode == loopEnd) {
                    continue;
                }

                for (TaskNode* child : curNode->GetChildren()) {
                    if (child == nullptr || child->GetLocation().queueId != originalQueueId) {
                        continue;
                    }
                    if (visitedNodes.insert(child).second) {
                        candidateNodes.push_back(child);
                    }
                }
            }
        }

        // 作用：收集紧跟在 async 节点之后、仍属于同一串行链的 record 节点。
        void CollectAsyncBoundRecordNodesV3(
            TaskNode* asyncNode, QueueId originalQueueId, std::set<TaskNode*>& nodes, TaskNode*& tailNode)
        {
            if (asyncNode == nullptr) {
                return;
            }

            tailNode = asyncNode;
            TaskNode* nextNode = GetSerialSameQueueChildV3(asyncNode, originalQueueId);
            while (nextNode != nullptr && nextNode->GetType() == TaskType::RECORD) {
                nodes.insert(nextNode);
                tailNode = nextNode;
                nextNode = GetSerialSameQueueChildV3(nextNode, originalQueueId);
            }
        }

        // 作用：获取当前 loop group 的入口边界。若前序 group 已被压缩成边，则沿用压缩后的等价边界。
        std::vector<TaskNode*> GetLoopBeforeNodesV3(
            TaskNode* firstHead, QueueId originalQueueId, const CompressedLoopBoundaryV3& compressedBoundary)
        {
            auto iter = compressedBoundary.find(firstHead);
            if (iter != compressedBoundary.end()) {
                return iter->second;
            }
            return GetSameQueueParentsV3(firstHead, originalQueueId);
        }

        // 作用：从一个 loop 信息组中构造待并行化的 loop 分组。
        HcclResult BuildLoopParallelGroupV3(
            const std::vector<LoopInfoV3>& loopGroup, const CompressedLoopBoundaryV3& compressedBoundary,
            LoopParallelGroupV3& group)
        {
            group = LoopParallelGroupV3{};
            if (loopGroup.empty() || loopGroup.front().loopStart == nullptr || loopGroup.back().loopEnd == nullptr) {
                return HCCL_SUCCESS;
            }

            const QueueId originalQueueId = loopGroup.front().loopStart->GetLocation().queueId;
            if (originalQueueId == INVALID_QUEUE_ID) {
                return HCCL_SUCCESS;
            }

            group.originalQueueId = originalQueueId;
            // 老 checker 按 loopGroupInfo_ 顺序边改图边处理；这里用压缩边界承接前序 group
            // 本该通过 localWaitTail 表达的真实流前沿。
            group.beforeNodes = GetLoopBeforeNodesV3(loopGroup.front().loopStart, originalQueueId, compressedBoundary);
            group.afterNodes = GetSameQueueChildrenV3(loopGroup.back().loopEnd, originalQueueId);
            if (group.beforeNodes.empty() || group.afterNodes.empty()) {
                HCCL_VM_ERROR(
                    "{} Failed to split one CCU loop-parallel group because its entry or exit "
                    "boundary could not be determined.",
                    MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str());
                return HCCL_E_INTERNAL;
            }

            for (const auto& loopInfo : loopGroup) {
                if (loopInfo.loopStart == nullptr || loopInfo.loopEnd == nullptr) {
                    continue;
                }

                QueueReuseUnitV3 unit;
                unit.kind = QueueReuseUnitKindV3::LOOP;
                unit.head = loopInfo.loopStart;
                unit.tail = loopInfo.loopEnd;
                unit.startNodeId = loopInfo.loopStart->GetNodeId();
                CollectLoopBodyNodesSameQueueV3(loopInfo.loopStart, loopInfo.loopEnd, originalQueueId, unit.nodes);
                if (!unit.nodes.empty()) {
                    group.units.push_back(std::move(unit));
                }
            }

            if (group.units.empty()) {
                group = LoopParallelGroupV3{};
                return HCCL_SUCCESS;
            }

            const NodeId releaseNodeId = GetReleaseNodeIdV3(group.afterNodes);
            for (auto& unit : group.units) {
                unit.endNodeId = releaseNodeId;
            }
            return HCCL_SUCCESS;
        }

        // 作用：为 async 分组提取外部前驱和后继边界节点。
        void PopulateAsyncGroupBoundaryNodesV3(AsyncParallelGroupV3& group)
        {
            if (group.units.empty() || group.units.back().tail == nullptr) {
                return;
            }

            std::set<TaskNode*> groupNodes;
            for (const auto& unit : group.units) {
                groupNodes.insert(unit.nodes.begin(), unit.nodes.end());
            }

            // async 单元在 loop 并行化后可能已经被重新分配 queue，
            // 因此它们的边界节点必须基于原图真实边来取，不能再依赖“同 queue 邻接”。
            group.beforeNodes = GetParentsOutsideNodesV3(group.units.front().head, groupNodes);
            for (auto& unit : group.units) {
                // 每个单元单独记录出口。不同单元的 tail 可能分别连接到不同的 WAIT，
                // 不能使用整个 async 分组最后一个 tail 的 afterNodes 代替。
                unit.afterNodes = GetChildrenOutsideNodesV3(unit.tail, groupNodes);
                unit.afterNodes.erase(
                    std::remove_if(
                        unit.afterNodes.begin(), unit.afterNodes.end(),
                        [](TaskNode* node) {
                            return IsCcuSubGraphEndNodeV3(node);
                        }),
                    unit.afterNodes.end());
            }
            // 分组级 afterNodes 用于原 queue 前后节点之间的桥接边，以及 queue 释放时机计算。
            group.afterNodes = GetChildrenOutsideNodesV3(group.units.back().tail, groupNodes);
            group.afterNodes.erase(
                std::remove_if(
                    group.afterNodes.begin(), group.afterNodes.end(),
                    [](TaskNode* node) {
                        return IsCcuSubGraphEndNodeV3(node);
                    }),
                group.afterNodes.end());
        }

        // 作用：从候选 async 节点中构造可并行化的 async 分组。
        void BuildAsyncParallelGroupsV3(const CcuGraphStateV3& state, std::vector<AsyncParallelGroupV3>& groups)
        {
            std::vector<TaskNode*> asyncNodes;
            for (TaskNode* node : state.parallelNodes_) {
                if (IsAsyncNodeV3(state, node)) {
                    asyncNodes.push_back(node);
                }
            }
            std::sort(asyncNodes.begin(), asyncNodes.end(), [](const TaskNode* lhs, const TaskNode* rhs) {
                return lhs->GetNodeId() < rhs->GetNodeId();
            });

            std::set<TaskNode*> processedHeads;
            for (TaskNode* node : asyncNodes) {
                if (node == nullptr || processedHeads.count(node) != 0) {
                    continue;
                }

                const QueueId originalQueueId = node->GetLocation().queueId;
                if (originalQueueId == INVALID_QUEUE_ID) {
                    continue;
                }

                AsyncParallelGroupV3 group;
                group.originalQueueId = originalQueueId;
                TaskNode* curHead = node;
                while (curHead != nullptr && curHead->GetLocation().queueId == originalQueueId
                       && processedHeads.count(curHead) == 0 && IsAsyncNodeV3(state, curHead)) {
                    QueueReuseUnitV3 unit;
                    unit.kind = QueueReuseUnitKindV3::ASYNC;
                    unit.head = curHead;
                    unit.tail = curHead;
                    unit.startNodeId = curHead->GetNodeId();
                    unit.nodes.insert(curHead);
                    // 把 async 后面紧跟的 record 链保留在同一个单元里，
                    // 这样重新分配 queue 后仍能保持生产侧 async 和 record 的先后顺序。
                    CollectAsyncBoundRecordNodesV3(curHead, originalQueueId, unit.nodes, unit.tail);
                    group.units.push_back(std::move(unit));
                    processedHeads.insert(curHead);

                    TaskNode* nextHead = GetSerialSameQueueChildV3(group.units.back().tail, originalQueueId);
                    if (nextHead == nullptr || nextHead->GetLocation().queueId != originalQueueId
                        || processedHeads.count(nextHead) != 0 || !IsAsyncNodeV3(state, nextHead)) {
                        break;
                    }
                    curHead = nextHead;
                }

                if (group.units.empty()) {
                    continue;
                }
                PopulateAsyncGroupBoundaryNodesV3(group);
                const NodeId releaseNodeId = GetReleaseNodeIdV3(group.afterNodes);
                for (auto& unit : group.units) {
                    unit.endNodeId = releaseNodeId;
                }
                groups.push_back(std::move(group));
            }
        }

        // 作用：把 loop 分组中的并行单元统一收集出来，供后续 queue 复用分配使用。
        void CollectQueueUnitsV3(std::vector<LoopParallelGroupV3>& groups, std::vector<QueueReuseUnitV3*>& units)
        {
            for (auto& group : groups) {
                for (auto& unit : group.units) {
                    units.push_back(&unit);
                }
            }
        }

        // 作用：把 async 分组中的并行单元统一收集出来，供后续 queue 复用分配使用。
        void CollectQueueUnitsV3(std::vector<AsyncParallelGroupV3>& groups, std::vector<QueueReuseUnitV3*>& units)
        {
            for (auto& group : groups) {
                for (auto& unit : group.units) {
                    units.push_back(&unit);
                }
            }
        }

        // 作用：为 loop 并行单元按改造顺序分配最终 queueId。
        void AssignLoopQueueIdsV3(LoopParallelGroupV3& group, QueueId& nextQueueId)
        {
            for (auto& unit : group.units) {
                unit.queueId = nextQueueId++;
                for (TaskNode* node : unit.nodes) {
                    SetNodeQueueIdV3(node, unit.queueId);
                }
            }
        }

        // 作用：按生命周期复用规则为传入的并行单元分配最终 queueId。
        void AssignQueueIdsWithReuseV3(std::vector<QueueReuseUnitV3*>& units, QueueIdReuseAllocatorV3& allocator)
        {
            std::sort(units.begin(), units.end(), [](const QueueReuseUnitV3* lhs, const QueueReuseUnitV3* rhs) {
                if (lhs == nullptr || rhs == nullptr) {
                    return lhs < rhs;
                }
                if (lhs->startNodeId != rhs->startNodeId) {
                    return lhs->startNodeId < rhs->startNodeId;
                }
                if (lhs->endNodeId != rhs->endNodeId) {
                    return lhs->endNodeId < rhs->endNodeId;
                }
                return static_cast<uint8_t>(lhs->kind) < static_cast<uint8_t>(rhs->kind);
            });

            std::multimap<NodeId, QueueId> activeUnits;
            for (QueueReuseUnitV3* unit : units) {
                if (unit == nullptr) {
                    continue;
                }
                while (!activeUnits.empty() && activeUnits.begin()->first < unit->startNodeId) {
                    allocator.Release(activeUnits.begin()->second);
                    activeUnits.erase(activeUnits.begin());
                }
                unit->queueId = allocator.Acquire();
                activeUnits.insert(std::make_pair(unit->endNodeId, unit->queueId));
            }
        }

        // 作用：把最终分配好的 queueId 回写到各并行单元包含的节点上。
        void ApplyQueueReuseUnitsV3(std::vector<QueueReuseUnitV3*>& units)
        {
            std::sort(units.begin(), units.end(), [](const QueueReuseUnitV3* lhs, const QueueReuseUnitV3* rhs) {
                if (lhs == nullptr || rhs == nullptr) {
                    return lhs < rhs;
                }
                if (lhs->kind != rhs->kind) {
                    return lhs->kind < rhs->kind;
                }
                if (lhs->startNodeId != rhs->startNodeId) {
                    return lhs->startNodeId < rhs->startNodeId;
                }
                return lhs->endNodeId > rhs->endNodeId;
            });

            for (QueueReuseUnitV3* unit : units) {
                if (unit == nullptr || unit->queueId == INVALID_QUEUE_ID) {
                    continue;
                }
                for (TaskNode* node : unit->nodes) {
                    SetNodeQueueIdV3(node, unit->queueId);
                }
            }
        }

        // 作用：拆除 loop 单元原有串行边，并重建 loop 并行化后的扇出/汇合关系。
        HcclResult RewireLoopParallelGroupV3(TaskGraphGeneratorV3& graph, const LoopParallelGroupV3& group)
        {
            // 等价于老 checker 的 ProcLoopNode：先拆相邻 loop 的原串行边。
            for (size_t index = 1; index < group.units.size(); ++index) {
                if (group.units[index - 1].tail == nullptr || group.units[index].head == nullptr) {
                    continue;
                }
                CHK_RET(
                    graph.RemoveEdge(group.units[index - 1].tail->GetNodeId(), group.units[index].head->GetNodeId()));
            }

            if (!group.units.empty()) {
                TaskNode* firstHead = group.units.front().head;
                TaskNode* lastTail = group.units.back().tail;
                for (TaskNode* beforeNode : group.beforeNodes) {
                    if (beforeNode == nullptr || firstHead == nullptr) {
                        continue;
                    }
                    CHK_RET(graph.RemoveEdge(beforeNode->GetNodeId(), firstHead->GetNodeId()));
                }
                for (TaskNode* afterNode : group.afterNodes) {
                    if (afterNode == nullptr || lastTail == nullptr) {
                        continue;
                    }
                    CHK_RET(graph.RemoveEdge(lastTail->GetNodeId(), afterNode->GetNodeId()));
                }
            }

            for (TaskNode* beforeNode : group.beforeNodes) {
                if (beforeNode == nullptr) {
                    continue;
                }
                for (const auto& unit : group.units) {
                    if (unit.head == nullptr) {
                        continue;
                    }
                    CHK_RET(graph.AddEdge(beforeNode->GetNodeId(), unit.head->GetNodeId()));
                }
            }

            for (TaskNode* afterNode : group.afterNodes) {
                if (afterNode == nullptr) {
                    continue;
                }
                for (const auto& unit : group.units) {
                    if (unit.tail == nullptr) {
                        continue;
                    }
                    CHK_RET(graph.AddEdge(unit.tail->GetNodeId(), afterNode->GetNodeId()));
                }
            }

            for (TaskNode* beforeNode : group.beforeNodes) {
                if (beforeNode == nullptr) {
                    continue;
                }
                for (TaskNode* afterNode : group.afterNodes) {
                    if (afterNode == nullptr) {
                        continue;
                    }
                    CHK_RET(graph.AddEdge(beforeNode->GetNodeId(), afterNode->GetNodeId()));
                }
            }
            return HCCL_SUCCESS;
        }

        // 作用：记录当前 loop group 去掉 localPost/localWait 壳节点后的出口等价边界。
        void
        UpdateCompressedLoopBoundaryV3(const LoopParallelGroupV3& group, CompressedLoopBoundaryV3& compressedBoundary)
        {
            std::vector<TaskNode*> frontier;
            for (TaskNode* beforeNode : group.beforeNodes) {
                AppendNodeIfAbsentV3(frontier, beforeNode);
            }
            for (const auto& unit : group.units) {
                AppendNodeIfAbsentV3(frontier, unit.tail);
            }

            for (TaskNode* afterNode : group.afterNodes) {
                if (afterNode == nullptr) {
                    continue;
                }
                compressedBoundary[afterNode] = frontier;
            }
        }

        // 作用：重连 async 并行单元的入口、出口及原 queue 上的桥接边。
        HcclResult
        RewireAsyncParallelGroupsV3(TaskGraphGeneratorV3& graph, const std::vector<AsyncParallelGroupV3>& groups)
        {
            for (const auto& group : groups) {
                // async 单元原本共享同一条串行 queue；重新分配 queue 后，它们会从同一批外部前驱出发，
                // 再汇合回同一批外部后继。与此同时，原 queue 上被“挖空”后留下的前后节点之间
                // 也要补一条直连桥接边，保证原 queue 的串行流仍然连通。
                for (size_t index = 1; index < group.units.size(); ++index) {
                    if (group.units[index - 1].tail == nullptr || group.units[index].head == nullptr) {
                        continue;
                    }
                    CHK_RET(graph.RemoveEdge(
                        group.units[index - 1].tail->GetNodeId(), group.units[index].head->GetNodeId()));
                }

                for (TaskNode* beforeNode : group.beforeNodes) {
                    if (beforeNode == nullptr) {
                        continue;
                    }
                    for (const auto& unit : group.units) {
                        if (unit.head == nullptr) {
                            continue;
                        }
                        CHK_RET(graph.AddEdge(beforeNode->GetNodeId(), unit.head->GetNodeId()));
                    }
                }

                for (const auto& unit : group.units) {
                    TaskNode* tailNode = unit.tail;
                    if (tailNode == nullptr) {
                        continue;
                    }

                    bool hasMatchingWaitNode = false;
                    if (tailNode->GetType() == TaskType::RECORD) {
                        for (TaskNode* afterNode : unit.afterNodes) {
                            if (HasNodeEdgeV3(tailNode, afterNode) && IsCcuRecordWaitPairV3(tailNode, afterNode)) {
                                hasMatchingWaitNode = true;
                                break;
                            }
                        }
                        if (!hasMatchingWaitNode) {
                            HCCL_VM_ERROR(
                                "{} Async parallel tail RECORD has no matching wait node, tail={}",
                                MakeErrorCodeText(ErrorCode::GRAPH_UNMATCHED).c_str(), tailNode->Describe());
                            return HCCL_E_INTERNAL;
                        }
                    } else {
                        HCCL_VM_WARN(
                            "Skip async parallel exit edge because the tail node is not a RECORD, "
                            "tail={}",
                            tailNode->Describe());
                    }

                    for (TaskNode* afterNode : unit.afterNodes) {
                        if (afterNode == nullptr) {
                            continue;
                        }
                        // afterNodes 是改造前的快照；如果原边已被前面的重连步骤移除，不能再次处理。
                        if (!HasNodeEdgeV3(tailNode, afterNode)) {
                            continue;
                        }

                        CHK_RET(graph.RemoveEdge(tailNode->GetNodeId(), afterNode->GetNodeId()));
                        if (!IsCcuRecordWaitPairV3(tailNode, afterNode)) {
                            continue;
                        }

                        // 只有原本存在且确认属于 RECORD/WAIT 的边才恢复；普通 queue 顺序边保持删除。
                        CHK_RET(graph.AddEdge(tailNode->GetNodeId(), afterNode->GetNodeId()));
                    }
                }

                for (TaskNode* beforeNode : group.beforeNodes) {
                    if (beforeNode == nullptr) {
                        continue;
                    }
                    for (TaskNode* afterNode : group.afterNodes) {
                        if (afterNode == nullptr) {
                            continue;
                        }
                        CHK_RET(graph.AddEdge(beforeNode->GetNodeId(), afterNode->GetNodeId()));
                    }
                }
            }
            return HCCL_SUCCESS;
        }

        // 作用：按当前图结构重新计算每个 queue 的 tail 节点。
        void RebuildTailNodesByQueueV3(CcuGraphStateV3& state)
        {
            state.tailNodes.assign(state.queueNum_, nullptr);
            if (state.ccuHeadTaskNode == nullptr) {
                return;
            }

            std::deque<TaskNode*> candidateNodes{state.ccuHeadTaskNode};
            std::set<TaskNode*> visitedNodes{state.ccuHeadTaskNode};
            while (!candidateNodes.empty()) {
                TaskNode* curNode = candidateNodes.front();
                candidateNodes.pop_front();
                if (curNode == nullptr) {
                    continue;
                }

                for (TaskNode* child : curNode->GetChildren()) {
                    if (child != nullptr && visitedNodes.insert(child).second) {
                        candidateNodes.push_back(child);
                    }
                }

                const QueueId queueId = curNode->GetLocation().queueId;
                if (queueId == INVALID_QUEUE_ID) {
                    continue;
                }
                if (queueId >= state.tailNodes.size()) {
                    state.tailNodes.resize(queueId + 1U, nullptr);
                    state.queueNum_ = std::max(state.queueNum_, queueId + 1U);
                }

                bool hasSameQueueChild = false;
                for (TaskNode* child : curNode->GetChildren()) {
                    if (child != nullptr && child->GetLocation().queueId == queueId) {
                        hasSameQueueChild = true;
                        break;
                    }
                }
                if (hasSameQueueChild) {
                    continue;
                }
                // 某个 queue 中没有同 queue 子节点的最新节点，会被视为该 queue 的 tail。
                if (state.tailNodes[queueId] == nullptr
                    || state.tailNodes[queueId]->GetNodeId() < curNode->GetNodeId()) {
                    state.tailNodes[queueId] = curNode;
                }
            }
        }

        // 作用：执行 CCU 子图的 queue 并行化改造总流程。
        HcclResult ApplyCcuQueueParallelizationV3(CcuGraphStateV3& state)
        {
            const uint32_t baseQueueCount = state.queueNum_;
            QueueId nextTempQueueId = baseQueueCount;
            CompressedLoopBoundaryV3 compressedLoopBoundary;
            // loop 必须按 loopGroupInfo_ 顺序边改图边处理，才能复刻老 checker 的真实流壳链语义。
            for (const auto& loopGroupInfo : state.loopGroupInfo_) {
                LoopParallelGroupV3 group;
                CHK_RET(BuildLoopParallelGroupV3(loopGroupInfo, compressedLoopBoundary, group));
                if (group.units.empty()) {
                    continue;
                }
                AssignLoopQueueIdsV3(group, nextTempQueueId);
                CHK_RET(RewireLoopParallelGroupV3(state.graph, group));
                UpdateCompressedLoopBoundaryV3(group, compressedLoopBoundary);
            }

            std::vector<AsyncParallelGroupV3> asyncGroups;
            BuildAsyncParallelGroupsV3(state, asyncGroups);

            std::vector<QueueReuseUnitV3*> asyncUnits;
            CollectQueueUnitsV3(asyncGroups, asyncUnits);
            if (asyncUnits.empty()) {
                state.queueNum_ = nextTempQueueId;
                RebuildTailNodesByQueueV3(state);
                return HCCL_SUCCESS;
            }

            // loop queue 已经按改造顺序固定占用；async 只能从 loop 占用后的 queueId 继续复用分配，
            // 避免跨原始 queue 的并行 loop 被最终压缩到同一个 queue 上。
            QueueIdReuseAllocatorV3 allocator(nextTempQueueId);
            AssignQueueIdsWithReuseV3(asyncUnits, allocator);
            ApplyQueueReuseUnitsV3(asyncUnits);
            CHK_RET(RewireAsyncParallelGroupsV3(state.graph, asyncGroups));
            state.queueNum_ = allocator.NextQueueId();
            RebuildTailNodesByQueueV3(state);
            return HCCL_SUCCESS;
        }

        // 作用：初始化多图成图流程共享的全局上下文和存储快照。
        void InitCcuGeneratorContextV3(StorageManager& storage)
        {
            AllRankParamRecorder::Global()->InitParam();
            storage.InitCcuInfo(
                AllRankParamRecorder::Global()->devType_, AllRankParamRecorder::Global()->ccu_resource_base_addr_);
        }

        // 作用：准备单个 CCU 图 raw 成图所需的初始状态。
        HcclResult PrepareRawCcuGraphStateV3(StorageManager& storage, CcuGraphStateV3& state)
        {
            // 第一阶段从 CCU 图的原始单 queue 视角开始展开。
            CHK_RET(state.InitInstrInfo(storage));
            CHK_RET(state.CreateHeadNode());
            return HCCL_SUCCESS;
        }

        // 作用：初始化一个 TaskCcuGraph 对应的运行时状态对象。
        HcclResult InitCcuGraphRuntimeV3(
            TaskGraphGeneratorV3& graph, StorageManager& storage, TaskCcuGraph* ccuGraph, CcuGraphRuntimeV3& runtime)
        {
            if (ccuGraph == nullptr) {
                return HCCL_E_PTR;
            }

            runtime.ccuGraph = ccuGraph;
            runtime.originalChildren = ccuGraph->GetChildren();
            runtime.state = std::make_unique<CcuGraphStateV3>(graph, *ccuGraph);
            CHK_RET(PrepareRawCcuGraphStateV3(storage, *runtime.state));
            return HCCL_SUCCESS;
        }

        // 作用：按需获取或创建某个 CCU 图对应的运行时对象。
        HcclResult GetOrCreateCcuGraphRuntimeV3(
            TaskGraphGeneratorV3& graph, StorageManager& storage, TaskCcuGraph* ccuGraph,
            std::vector<std::unique_ptr<CcuGraphRuntimeV3>>& runtimes,
            std::map<NodeId, CcuGraphRuntimeV3*>& runtimeByNodeId, CcuGraphRuntimeV3*& runtime)
        {
            if (ccuGraph == nullptr) {
                return HCCL_E_PTR;
            }

            auto runtimeIter = runtimeByNodeId.find(ccuGraph->GetNodeId());
            if (runtimeIter != runtimeByNodeId.end()) {
                runtime = runtimeIter->second;
                return (runtime == nullptr) ? HCCL_E_PTR : HCCL_SUCCESS;
            }

            auto newRuntime = std::make_unique<CcuGraphRuntimeV3>();
            CHK_RET(InitCcuGraphRuntimeV3(graph, storage, ccuGraph, *newRuntime));
            runtime = newRuntime.get();
            runtimeByNodeId[ccuGraph->GetNodeId()] = runtime;
            runtimes.emplace_back(std::move(newRuntime));
            return HCCL_SUCCESS;
        }

        // 作用：完成 raw 图阶段的收尾，并生成并行化前固定下来的 subGraphEnd。
        HcclResult FinalizeRawCcuGraphStateV3(CcuGraphStateV3& state)
        {
            // 在 queue 并行化重连之前，先把末尾残留的双边 post 记账收尾。
            for (uint32_t queId = 0; queId < state.queueNum_; ++queId) {
                CHK_RET(CollectBilateralPostInfo(&state, queId, nullptr, true));
            }
            // 在并行化改写 queueId 之前，先用原始 queue 的 tail 封口 raw 图。
            if (state.ccuSubGraphEndNode == nullptr) {
                AddCcuSubGraphEnd(&state.ccuGraph, &state);
            }
            if (state.ccuSubGraphEndNode == nullptr) {
                return HCCL_E_INTERNAL;
            }
            return HCCL_SUCCESS;
        }

        // 作用：推进单个 CCU 图直到 raw 成图结束。
        HcclResult GenerateRawCcuGraphStateV3(TaskNode* node, CcuGraphStateV3& state)
        {
            uint32_t unmatchedCnt = 0;
            while (!state.isGenGraphed) {
                std::vector<u32> microCodePosInQuePre = state.microCodePosInQue;
                CHK_RET(ProcessCcuNode(node, &state));
                if (state.microCodePosInQue != microCodePosInQuePre) {
                    unmatchedCnt = 0;
                } else {
                    ++unmatchedCnt;
                }
                if (unmatchedCnt > state.queueNum_) {
                    HCCL_VM_ERROR(
                        "{} CCU raw graph generation is stuck inside one graph. All queues are "
                        "blocked and no instruction can advance, rankId={}, queueCount={}, nextInstructionByQueue={}, "
                        "blockedGraphSummary={}",
                        MakeErrorCodeText(ErrorCode::GRAPH_DEADLOCK).c_str(), state.GetRankId(), state.queueNum_,
                        state.DescribeNextInstructionByQueue().c_str(), state.Describe().c_str());
                    return HCCL_E_INTERNAL;
                }
            }
            return FinalizeRawCcuGraphStateV3(state);
        }

        // 作用：按固定顺序打印所有已经完成 raw 成图的 CCU 子图。
        void PrintGeneratedCcuGraphsV3(const std::vector<std::unique_ptr<CcuGraphRuntimeV3>>& runtimes)
        {
            std::vector<const CcuGraphRuntimeV3*> orderedRuntimes;
            orderedRuntimes.reserve(runtimes.size());
            for (const auto& runtime : runtimes) {
                if (runtime == nullptr || !runtime->generated || runtime->state == nullptr
                    || runtime->state->ccuHeadTaskNode == nullptr || runtime->ccuGraph == nullptr) {
                    continue;
                }
                orderedRuntimes.push_back(runtime.get());
            }

            std::sort(
                orderedRuntimes.begin(), orderedRuntimes.end(),
                [](const CcuGraphRuntimeV3* lhs, const CcuGraphRuntimeV3* rhs) {
                    if (lhs == nullptr || rhs == nullptr) {
                        return lhs < rhs;
                    }
                    const RankId lhsRankId = lhs->ccuGraph->GetRankId();
                    const RankId rhsRankId = rhs->ccuGraph->GetRankId();
                    if (lhsRankId != rhsRankId) {
                        return lhsRankId < rhsRankId;
                    }
                    return lhs->ccuGraph->GetNodeId() < rhs->ccuGraph->GetNodeId();
                });

            HCCL_VM_INFO(
                "Printing all generated CCU graphs before queue parallelization, graphCount={}",
                orderedRuntimes.size());
            for (const CcuGraphRuntimeV3* runtime : orderedRuntimes) {
                if (runtime == nullptr || runtime->state == nullptr) {
                    continue;
                }
                HCCL_VM_INFO(
                    "CCU graph detail, ccuGraphNodeId={}, rankId={}", runtime->ccuGraph->GetNodeId(),
                    runtime->ccuGraph->GetRankId());
                PrintCcuGraph(runtime->state->ccuHeadTaskNode);
            }
        }

        // 作用：把已经展开完成的内部 CCU 子图重新接回主图中的占位节点位置。
        HcclResult AttachExpandedCcuGraphV3(
            TaskGraphGeneratorV3& graph, TaskCcuGraph& ccuGraph, const std::vector<TaskNode*>& originalChildren,
            CcuGraphStateV3& state, CcuGraphGenerateOutputV3& output)
        {
            if (state.ccuSubGraphEndNode == nullptr) {
                return HCCL_E_INTERNAL;
            }
            if (state.ccuHeadTaskNode == nullptr) {
                return HCCL_E_INTERNAL;
            }

            // 用展开后的内部子图替换占位的 TaskCcuGraph 节点：
            // 父侧从 ccuHeadTaskNode 进入，子侧从 subGraphEnd 离开。
            CHK_RET(graph.AddEdge(ccuGraph.GetNodeId(), state.ccuHeadTaskNode->GetNodeId()));
            for (TaskNode* childNode : originalChildren) {
                if (childNode == nullptr) {
                    return HCCL_E_PTR;
                }
                CHK_RET(graph.RemoveEdge(ccuGraph.GetNodeId(), childNode->GetNodeId()));
                CHK_RET(graph.AddEdge(state.ccuSubGraphEndNode->GetNodeId(), childNode->GetNodeId()));
            }

            output.endNodeId = state.ccuSubGraphEndNode->GetNodeId();
            output.internalNodeCount += state.internalNodeCount;
            return HCCL_SUCCESS;
        }

        // 作用：完成单个 CCU 图的并行化改造，并把它重新挂回主图。
        HcclResult FinalizeGeneratedCcuGraphRuntimeV3(
            TaskGraphGeneratorV3& graph, CcuGraphRuntimeV3& runtime, CcuGraphGenerateOutputV3& output)
        {
            if (runtime.ccuGraph == nullptr || runtime.state == nullptr) {
                return HCCL_E_PTR;
            }
            if (runtime.finalized) {
                return HCCL_SUCCESS;
            }
            if (!runtime.generated) {
                return HCCL_E_INTERNAL;
            }

            CcuGraphStateV3& state = *runtime.state;
            // 第二阶段只在已经完成的 raw 图上做 queue 分配和图重连。
            CHK_RET(ApplyCcuQueueParallelizationV3(state));
            CHK_RET(AttachExpandedCcuGraphV3(graph, *runtime.ccuGraph, runtime.originalChildren, state, output));
            runtime.finalized = true;
            return HCCL_SUCCESS;
        }

        // 作用：联合推进多个 CCU 图的 raw 成图阶段，处理跨图依赖解锁。
        HcclResult GenerateRawCcuGraphsV3(
            const CcuGraphsGenerateInputV3& input, std::vector<std::unique_ptr<CcuGraphRuntimeV3>>& runtimes,
            CcuGraphGenerateOutputV3& output)
        {
            TaskGraphGeneratorV3& graph = *input.graph;
            StorageManager& storage = *input.storage;
            std::map<NodeId, CcuGraphRuntimeV3*> runtimeByNodeId;
            std::map<NodeId, TaskCcuGraph*> ccuGraphByNodeId;
            for (TaskCcuGraph* ccuGraph : input.ccuGraphs) {
                if (ccuGraph == nullptr) {
                    return HCCL_E_PTR;
                }
                ccuGraphByNodeId[ccuGraph->GetNodeId()] = ccuGraph;
            }
            runtimes.reserve(input.ccuGraphs.size());

            TaskNode* mainStart = graph.GetMainStartNode();
            if (mainStart == nullptr) {
                return HCCL_E_INTERNAL;
            }

            std::deque<TaskNode*> candNode;
            std::set<TaskNode*> visitedNodes;
            std::set<TaskNode*> completedNodes;
            completedNodes.insert(mainStart);
            AddReachableChildren(candNode, visitedNodes, mainStart->GetChildren());

            uint32_t unmatchedCnt = 0;
            size_t generatedCount = 0;
            while (!candNode.empty()) {
                if (unmatchedCnt >= candNode.size()) {
                    TaskNode* firstBlockedGraph = candNode.front();
                    const CcuGraphRuntimeV3* blockedRuntime = nullptr;
                    auto runtimeIter = runtimeByNodeId.find(
                        firstBlockedGraph == nullptr ? INVALID_NODE_ID : firstBlockedGraph->GetNodeId());
                    if (runtimeIter != runtimeByNodeId.end()) {
                        blockedRuntime = runtimeIter->second;
                    }
                    HCCL_VM_ERROR(
                        "{} CCU multi-graph raw generation is stuck. Some CCU graphs are still "
                        "blocked by dependencies, but no graph can advance, blockedGraphCount={}, rankId={}, "
                        "queueCount={}, "
                        "nextInstructionByQueue={}, firstBlockedGraph={}",
                        MakeErrorCodeText(ErrorCode::GRAPH_DEADLOCK).c_str(), candNode.size(),
                        blockedRuntime == nullptr || blockedRuntime->state == nullptr ?
                            std::string("null") :
                            std::to_string(blockedRuntime->state->GetRankId()),
                        blockedRuntime == nullptr || blockedRuntime->state == nullptr ?
                            0U :
                            blockedRuntime->state->queueNum_,
                        blockedRuntime == nullptr || blockedRuntime->state == nullptr ?
                            "[]" :
                            blockedRuntime->state->DescribeNextInstructionByQueue().c_str(),
                        firstBlockedGraph == nullptr ? "node=null" : firstBlockedGraph->Describe().c_str());
                    return HCCL_E_INTERNAL;
                }

                TaskNode* curNode = candNode.front();
                candNode.pop_front();
                if (curNode == nullptr) {
                    return HCCL_E_PTR;
                }

                if (completedNodes.count(curNode) != 0) {
                    unmatchedCnt = 0;
                    continue;
                }

                if (!AreParentsCompleted(curNode, completedNodes)) {
                    candNode.push_back(curNode);
                    ++unmatchedCnt;
                    continue;
                }

                auto ccuGraphIter = ccuGraphByNodeId.find(curNode->GetNodeId());
                if (ccuGraphIter == ccuGraphByNodeId.end()) {
                    completedNodes.insert(curNode);
                    AddReachableChildren(candNode, visitedNodes, curNode->GetChildren());
                    unmatchedCnt = 0;
                    continue;
                }

                CcuGraphRuntimeV3* runtime = nullptr;
                CHK_RET(GetOrCreateCcuGraphRuntimeV3(
                    graph, storage, ccuGraphIter->second, runtimes, runtimeByNodeId, runtime));
                if (runtime == nullptr || runtime->state == nullptr || runtime->ccuGraph == nullptr) {
                    return HCCL_E_PTR;
                }
                if (output.startNodeId == INVALID_NODE_ID && runtime->state->ccuHeadTaskNode != nullptr) {
                    output.startNodeId = runtime->state->ccuHeadTaskNode->GetNodeId();
                }
                if (runtime->generated) {
                    unmatchedCnt = 0;
                    continue;
                }

                CcuGraphStateV3& state = *runtime->state;
                // 这里会让多个 rank 一起推进，
                // 这样跨 rank 的 wait/post 依赖能在 raw 图封口前彼此解锁。
                std::vector<u32> microCodePosInQuePre = state.microCodePosInQue;
                CHK_RET(ProcessCcuNode(runtime->ccuGraph, &state));
                if (state.microCodePosInQue != microCodePosInQuePre) {
                    unmatchedCnt = 0;
                } else {
                    ++unmatchedCnt;
                }

                if (state.isGenGraphed) {
                    CHK_RET(FinalizeRawCcuGraphStateV3(state));
                    runtime->generated = true;
                    ++generatedCount;
                    completedNodes.insert(runtime->ccuGraph);
                    AddReachableChildren(candNode, visitedNodes, runtime->originalChildren);
                    unmatchedCnt = 0;
                    continue;
                }

                candNode.push_back(runtime->ccuGraph);
            }

            if (generatedCount != input.ccuGraphs.size()) {
                const CcuGraphRuntimeV3* firstUnfinishedRuntime = nullptr;
                for (const auto& runtime : runtimes) {
                    if (runtime != nullptr && !runtime->generated) {
                        firstUnfinishedRuntime = runtime.get();
                        break;
                    }
                }
                HCCL_VM_ERROR(
                    "{} CCU raw graph generation ended before all CCU graphs were finished, "
                    "generatedGraphCount={}, totalGraphCount={}, firstUnfinishedGraph={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), generatedCount,
                    input.ccuGraphs.size(),
                    firstUnfinishedRuntime == nullptr || firstUnfinishedRuntime->ccuGraph == nullptr ?
                        "node=null" :
                        firstUnfinishedRuntime->ccuGraph->Describe().c_str());
                return HCCL_E_INTERNAL;
            }
            return HCCL_SUCCESS;
        }

        // 作用：批量完成所有已生成 CCU 图的并行化改造和主图挂接。
        HcclResult FinalizeGeneratedCcuGraphsV3(
            TaskGraphGeneratorV3& graph, std::vector<std::unique_ptr<CcuGraphRuntimeV3>>& runtimes,
            CcuGraphGenerateOutputV3& output)
        {
            size_t finalizedCount = 0;
            for (auto& runtime : runtimes) {
                if (runtime == nullptr || !runtime->generated) {
                    continue;
                }
                CHK_RET(FinalizeGeneratedCcuGraphRuntimeV3(graph, *runtime, output));
                ++finalizedCount;
            }
            if (finalizedCount != runtimes.size()) {
                const CcuGraphRuntimeV3* firstUnfinishedRuntime = nullptr;
                for (const auto& runtime : runtimes) {
                    if (runtime != nullptr && !runtime->finalized) {
                        firstUnfinishedRuntime = runtime.get();
                        break;
                    }
                }
                HCCL_VM_ERROR(
                    "{} CCU graph finalization ended before all generated graphs were attached back "
                    "to the main graph, finalizedGraphCount={}, totalGraphCount={}, firstUnfinishedGraph={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), finalizedCount, runtimes.size(),
                    firstUnfinishedRuntime == nullptr || firstUnfinishedRuntime->ccuGraph == nullptr ?
                        "node=null" :
                        firstUnfinishedRuntime->ccuGraph->Describe().c_str());
                return HCCL_E_INTERNAL;
            }
            return HCCL_SUCCESS;
        }

    } // namespace

    // 作用：CCU V3 多图展开总入口，串联 raw 成图、并行化改造和主图回接。
    HcclResult ExpandCcuGraphsV3(const CcuGraphsGenerateInputV3& input, CcuGraphGenerateOutputV3& output)
    {
        if (input.graph == nullptr || input.storage == nullptr) {
            return HCCL_E_PTR;
        }

        output = CcuGraphGenerateOutputV3{};
        if (input.ccuGraphs.empty()) {
            return HCCL_SUCCESS;
        }

        StorageManager& storage = *input.storage;
        InitCcuGeneratorContextV3(storage);

        std::vector<std::unique_ptr<CcuGraphRuntimeV3>> runtimes;
        // 第一阶段先把所有 CCU raw 图都生成出来，
        // 这样在 queue 并行化改拓扑之前，跨 rank 依赖已经完整可见。
        CHK_RET(GenerateRawCcuGraphsV3(input, runtimes, output));
        // 如需排查并行化改造前的全 rank 状态，可放开这一行打印。
        PrintGeneratedCcuGraphsV3(runtimes);
        // 第二阶段负责重连 queue，并把每个展开后的子图重新挂回主图。
        CHK_RET(FinalizeGeneratedCcuGraphsV3(*input.graph, runtimes, output));
        CHK_RET(AllRankParamRecorder::Global()->CheckAllPostMatch());
        return HCCL_SUCCESS;
    }

    // 作用：按指令版本创建对应的指令转换映射表。
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

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
