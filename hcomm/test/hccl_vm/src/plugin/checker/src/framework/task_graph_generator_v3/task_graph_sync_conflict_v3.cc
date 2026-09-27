/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_graph_sync_conflict_v3.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "sim_log.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {
        enum class SyncResourceKind : uint8_t {
            AICPU_NOTIFY = 0,
            CCU_CKE,
            AIV_EVENT,
            AIV_FLAG,
        };

        struct SyncResourceKey {
            SyncResourceKind kind{SyncResourceKind::AICPU_NOTIFY};
            uint32_t notifyId{0};
            RankId rankId{INVALID_RANK_ID};
            uint32_t dieId{INVALID_DIE_ID};
            uint16_t ckeId{INVALID_CCU_CKE};
            uint8_t ckeBit{0};
            uint64_t launchIdx{0};
            uint32_t blockId{std::numeric_limits<uint32_t>::max()};
            uint32_t srcPipe{std::numeric_limits<uint32_t>::max()};
            uint32_t dstPipe{std::numeric_limits<uint32_t>::max()};
            int32_t eventId{0};
            uint64_t commInfoOffset{0};
            int32_t flagValue{0};

            bool operator<(const SyncResourceKey& rhs) const
            {
                return std::tie(
                           kind, notifyId, rankId, dieId, ckeId, ckeBit, launchIdx, blockId, srcPipe, dstPipe, eventId,
                           commInfoOffset, flagValue)
                       < std::tie(
                           rhs.kind, rhs.notifyId, rhs.rankId, rhs.dieId, rhs.ckeId, rhs.ckeBit, rhs.launchIdx,
                           rhs.blockId, rhs.srcPipe, rhs.dstPipe, rhs.eventId, rhs.commInfoOffset, rhs.flagValue);
            }

            bool operator==(const SyncResourceKey& rhs) const { return !(*this < rhs) && !(rhs < *this); }
        };

        enum class CopyNodeRole : uint8_t {
            NORMAL = 0,
            START_WAIT,
            RECEIVE_WAIT,
        };

        struct CopyNode {
            const TaskNode* origin{nullptr};
            CopyNodeRole role{CopyNodeRole::NORMAL};
            size_t index{0};
            std::vector<CopyNode*> parents;
            std::vector<CopyNode*> children;
        };

        struct CopiedGraph {
            std::vector<std::unique_ptr<CopyNode>> nodes;
            std::map<const TaskNode*, CopyNode*> normalNodes;
            std::map<const TaskNode*, CopyNode*> startWaitNodes;
            std::map<const TaskNode*, CopyNode*> receiveWaitNodes;
            CopyNode* start{nullptr};
        };

        struct SyncPair {
            SyncResourceKey resource;
            const TaskNode* post{nullptr};
            const TaskNode* wait{nullptr};
            CopyNode* postCopy{nullptr};
            CopyNode* startWaitCopy{nullptr};
            CopyNode* receiveWaitCopy{nullptr};
            size_t order{0};
        };

        struct SyncBucket {
            std::vector<SyncPair> pairs;
            std::set<const TaskNode*> originalPosts;
            std::set<const TaskNode*> originalWaits;
            std::set<CopyNode*> postCopies;
        };

        struct AivFlagCellKey {
            RankId rankId{INVALID_RANK_ID};
            uint64_t launchIdx{0};
            uint64_t commInfoOffset{0};

            bool operator<(const AivFlagCellKey& rhs) const
            {
                return std::tie(rankId, launchIdx, commInfoOffset)
                       < std::tie(rhs.rankId, rhs.launchIdx, rhs.commInfoOffset);
            }
        };

        struct AivFlagGroup {
            const TaskNode* post{nullptr};
            CopyNode* postCopy{nullptr};
            int32_t flagValue{0};
            std::vector<const SyncPair*> pairs;
        };

        using SyncBuckets = std::map<SyncResourceKey, SyncBucket>;
        using SyncParentMap = std::map<const TaskNode*, std::set<const TaskNode*>>;

        // 将 SyncResourceKind 枚举转换为可读名称，供日志/诊断输出使用。
        const char* SyncResourceKindName(SyncResourceKind kind)
        {
            switch (kind) {
                case SyncResourceKind::AICPU_NOTIFY:
                    return "AICPU_NOTIFY";
                case SyncResourceKind::CCU_CKE:
                    return "CCU_CKE";
                case SyncResourceKind::AIV_EVENT:
                    return "AIV_EVENT";
                case SyncResourceKind::AIV_FLAG:
                    return "AIV_FLAG";
                default:
                    return "INVALID";
            }
        }

        const char* SyncPostTaskTypeName(SyncResourceKind kind)
        {
            switch (kind) {
                case SyncResourceKind::AICPU_NOTIFY:
                case SyncResourceKind::CCU_CKE:
                    return "RECORD";
                case SyncResourceKind::AIV_EVENT:
                    return "AIV_SET_FLAG";
                case SyncResourceKind::AIV_FLAG:
                    return "AIV_SEND_FLAG";
                default:
                    return "POST";
            }
        }

        const char* SyncWaitTaskTypeName(SyncResourceKind kind)
        {
            switch (kind) {
                case SyncResourceKind::AICPU_NOTIFY:
                case SyncResourceKind::CCU_CKE:
                    return "WAIT";
                case SyncResourceKind::AIV_EVENT:
                    return "AIV_WAIT_FLAG";
                case SyncResourceKind::AIV_FLAG:
                    return "AIV_RECV_FLAG";
                default:
                    return "WAIT";
            }
        }

        // 生成同步资源的单行描述，只输出与该资源 kind 相关的字段。
        // 所有冲突/错误日志都通过它输出资源标识，保证可读性一致。
        std::string DescribeResource(const SyncResourceKey& resource)
        {
            std::ostringstream os;
            os << "kind=" << SyncResourceKindName(resource.kind);
            if (resource.kind == SyncResourceKind::AICPU_NOTIFY) {
                os << ", notifyId=" << resource.notifyId;
            } else if (resource.kind == SyncResourceKind::CCU_CKE) {
                os << ", waitRankId=" << resource.rankId << ", dieId=" << resource.dieId << ", ckeId=" << resource.ckeId
                   << ", bit=" << static_cast<uint32_t>(resource.ckeBit);
            } else if (resource.kind == SyncResourceKind::AIV_EVENT) {
                os << ", rankId=" << resource.rankId << ", launchIdx=" << resource.launchIdx
                   << ", blockId=" << resource.blockId << ", srcPipe=" << resource.srcPipe
                   << ", dstPipe=" << resource.dstPipe << ", eventId=" << resource.eventId;
            } else {
                os << ", flagOwnerRank=" << resource.rankId << ", launchIdx=" << resource.launchIdx
                   << ", commInfoOffset=0x" << std::hex << resource.commInfoOffset << std::dec
                   << ", value=" << resource.flagValue;
            }
            return os.str();
        }

        // 判断一个任务节点是否在 post/wait 同步对中扮演"消费者"角色
        // （AICPU Wait、AIV WaitFlag 或 AIV RecvFlag）。
        bool IsWaitNode(const TaskNode* node)
        {
            if (node == nullptr) {
                return false;
            }
            const TaskType type = node->GetType();
            return type == TaskType::WAIT || type == TaskType::AIV_WAIT_FLAG || type == TaskType::AIV_RECV_FLAG;
        }

        // 判断一个任务节点是否在 post/wait 同步对中扮演"生产者"角色
        // （AICPU Record、AIV SetFlag 或 AIV SendFlag）。
        bool IsPostNode(const TaskNode* node)
        {
            if (node == nullptr) {
                return false;
            }
            const TaskType type = node->GetType();
            return type == TaskType::RECORD || type == TaskType::AIV_SET_FLAG || type == TaskType::AIV_SEND_FLAG;
        }

        // 推导节点所生产（post）或消费（wait）的同步资源 key。
        // 一个节点可能映射到多个资源：带 16 位 ckeMask 的 CCU Record/Wait 会按每个置位 bit
        // 展开成一个资源；其余 kind 各映射到单个资源。
        // 对非同步或非法节点返回空 vector。
        std::vector<SyncResourceKey> GetSyncResources(const TaskNode* node)
        {
            std::vector<SyncResourceKey> resources;
            if (node == nullptr) {
                return resources;
            }

            if (const auto* record = dynamic_cast<const TaskRecordAICPU*>(node)) {
                if (record->GetNotify().notifyId != INVALID_NOTIFY_ID) {
                    SyncResourceKey resource;
                    resource.kind = SyncResourceKind::AICPU_NOTIFY;
                    resource.notifyId = record->GetNotify().notifyId;
                    resources.push_back(resource);
                }
                return resources;
            }
            if (const auto* wait = dynamic_cast<const TaskWaitAICPU*>(node)) {
                if (wait->GetNotify().notifyId != INVALID_NOTIFY_ID) {
                    SyncResourceKey resource;
                    resource.kind = SyncResourceKind::AICPU_NOTIFY;
                    resource.notifyId = wait->GetNotify().notifyId;
                    resources.push_back(resource);
                }
                return resources;
            }
            if (const auto* record = dynamic_cast<const TaskRecordCCU*>(node)) {
                const CcuNotify& notify = record->GetNotify();
                if (notify.ckeId == INVALID_CCU_CKE) {
                    return resources;
                }
                for (uint32_t bit = 0; bit < 16; ++bit) {
                    if ((notify.ckeMask & (static_cast<uint16_t>(1U << bit))) == 0) {
                        continue;
                    }
                    SyncResourceKey resource;
                    resource.kind = SyncResourceKind::CCU_CKE;
                    resource.rankId = notify.waitRankId;
                    resource.dieId = notify.dieId;
                    resource.ckeId = notify.ckeId;
                    resource.ckeBit = static_cast<uint8_t>(bit);
                    resources.push_back(resource);
                }
                return resources;
            }
            if (const auto* wait = dynamic_cast<const TaskWaitCCU*>(node)) {
                const CcuNotify& notify = wait->GetNotify();
                if (notify.ckeId == INVALID_CCU_CKE) {
                    return resources;
                }
                for (uint32_t bit = 0; bit < 16; ++bit) {
                    if ((notify.ckeMask & (static_cast<uint16_t>(1U << bit))) == 0) {
                        continue;
                    }
                    SyncResourceKey resource;
                    resource.kind = SyncResourceKind::CCU_CKE;
                    resource.rankId = notify.waitRankId;
                    resource.dieId = notify.dieId;
                    resource.ckeId = notify.ckeId;
                    resource.ckeBit = static_cast<uint8_t>(bit);
                    resources.push_back(resource);
                }
                return resources;
            }
            if (const auto* setFlag = dynamic_cast<const TaskAivSetFlag*>(node)) {
                const AivPipeEvent& event = setFlag->GetEvent();
                SyncResourceKey resource;
                resource.kind = SyncResourceKind::AIV_EVENT;
                resource.rankId = event.rankId;
                resource.launchIdx = event.launchIdx;
                resource.blockId = event.blockId;
                resource.srcPipe = event.srcPipe;
                resource.dstPipe = event.dstPipe;
                resource.eventId = event.eventId;
                resources.push_back(resource);
                return resources;
            }
            if (const auto* waitFlag = dynamic_cast<const TaskAivWaitFlag*>(node)) {
                const AivPipeEvent& event = waitFlag->GetEvent();
                SyncResourceKey resource;
                resource.kind = SyncResourceKind::AIV_EVENT;
                resource.rankId = event.rankId;
                resource.launchIdx = event.launchIdx;
                resource.blockId = event.blockId;
                resource.srcPipe = event.srcPipe;
                resource.dstPipe = event.dstPipe;
                resource.eventId = event.eventId;
                resources.push_back(resource);
                return resources;
            }
            if (const auto* sendFlag = dynamic_cast<const TaskAivSendFlag*>(node)) {
                const AivFlagSync& flag = sendFlag->GetFlag();
                SyncResourceKey resource;
                resource.kind = SyncResourceKind::AIV_FLAG;
                resource.rankId = flag.flagOwnerRank;
                resource.launchIdx = flag.launchIdx;
                resource.commInfoOffset = flag.commInfoOffset;
                resource.flagValue = flag.value;
                resources.push_back(resource);
                return resources;
            }
            if (const auto* recvFlag = dynamic_cast<const TaskAivRecvFlag*>(node)) {
                const AivFlagSync& flag = recvFlag->GetFlag();
                SyncResourceKey resource;
                resource.kind = SyncResourceKind::AIV_FLAG;
                resource.rankId = flag.flagOwnerRank;
                resource.launchIdx = flag.launchIdx;
                resource.commInfoOffset = flag.commInfoOffset;
                resource.flagValue = flag.value;
                resources.push_back(resource);
                return resources;
            }
            return resources;
        }

        // 线性成员判定，用于在为 wait 节点配对时检查其父节点投递的资源集合。
        bool ContainsResource(const std::vector<SyncResourceKey>& resources, const SyncResourceKey& resource)
        {
            return std::find(resources.begin(), resources.end(), resource) != resources.end();
        }

        // 广度优先收集从 `start` 可达的所有节点到 `nodes`，保持 BFS 顺序。
        // 同步冲突检查器基于这个扁平化遍历结果工作，而非直接操作原始 DAG。
        HcclResult CollectOriginalNodes(const TaskNode* start, std::vector<const TaskNode*>& nodes)
        {
            nodes.clear();
            if (start == nullptr) {
                return HCCL_E_PTR;
            }

            std::queue<const TaskNode*> pending;
            std::set<const TaskNode*> visited;
            pending.push(start);
            visited.insert(start);
            while (!pending.empty()) {
                const TaskNode* node = pending.front();
                pending.pop();
                nodes.push_back(node);
                for (const TaskNode* child : node->GetChildren()) {
                    if (child == nullptr) {
                        HCCL_VM_ERROR(
                            "{} Sync DAG is invalid because a child node is null, parent={}",
                            MakeErrorCodeText(ErrorCode::SYNC_DAG_INVALID), node->Describe());
                        return HCCL_E_PTR;
                    }
                    if (visited.insert(child).second) {
                        pending.push(child);
                    }
                }
            }
            return HCCL_SUCCESS;
        }

        // 基于遍历结果构建 post/wait 配对模型：
        //  1. 把每个 Post 节点按其投递的每个资源注册到 originalPosts。
        //  2. 对每个 Wait 节点，按其消费的每个资源注册到 originalWaits；并对每个 Post 类型
        //     且其资源与该 Wait 消费资源匹配的父节点，同时在 bucket 中记录 (post,wait) 的
        //     SyncPair，并把该父节点记入 syncParents[wait]。
        // 单个 Post 合法地可与多个 Wait 配对（如 AIV SendFlag 的扇出），因此配对本身是多对多
        // 的；顺序正确性留待后续校验。
        HcclResult
        BuildSyncPairs(const std::vector<const TaskNode*>& nodes, SyncBuckets& buckets, SyncParentMap& syncParents)
        {
            buckets.clear();
            syncParents.clear();

            // 保留每一次出现，包括图匹配器在 flag cell 被覆盖后可能遗留的未配对 post。
            for (const TaskNode* node : nodes) {
                if (IsPostNode(node)) {
                    for (const SyncResourceKey& resource : GetSyncResources(node)) {
                        buckets[resource].originalPosts.insert(node);
                    }
                }
            }

            for (const TaskNode* wait : nodes) {
                if (!IsWaitNode(wait)) {
                    continue;
                }
                const std::vector<SyncResourceKey> waitResources = GetSyncResources(wait);
                for (const SyncResourceKey& resource : waitResources) {
                    buckets[resource].originalWaits.insert(wait);
                }
                for (const TaskNode* parent : wait->GetParents()) {
                    if (parent == nullptr) {
                        HCCL_VM_ERROR(
                            "{} Sync DAG is invalid because a parent node is null, childWait={}",
                            MakeErrorCodeText(ErrorCode::SYNC_DAG_INVALID), wait->Describe());
                        return HCCL_E_PTR;
                    }
                    if (!IsPostNode(parent)) {
                        continue;
                    }
                    const std::vector<SyncResourceKey> postResources = GetSyncResources(parent);
                    for (const SyncResourceKey& resource : postResources) {
                        if (!ContainsResource(waitResources, resource)) {
                            continue;
                        }
                        syncParents[wait].insert(parent);
                        buckets[resource].pairs.push_back(SyncPair{resource, parent, wait});
                    }
                }
            }
            return HCCL_SUCCESS;
        }

        // 定位代表原始节点"完成点"的拷贝节点：
        // wait 节点返回 ReceiveWait 拷贝（被等待资源已到达的时刻），其余返回唯一的 normal 拷贝。
        CopyNode* FindCompletionCopy(const CopiedGraph& graph, const TaskNode* node)
        {
            if (IsWaitNode(node)) {
                const auto iter = graph.receiveWaitNodes.find(node);
                return iter == graph.receiveWaitNodes.end() ? nullptr : iter->second;
            }
            const auto iter = graph.normalNodes.find(node);
            return iter == graph.normalNodes.end() ? nullptr : iter->second;
        }

        // 定位代表原始节点"起始点"的拷贝节点：
        // wait 节点返回 StartWait 拷贝（wait 可以开始的时刻），其余返回 normal 拷贝。
        CopyNode* FindStartCopy(const CopiedGraph& graph, const TaskNode* node)
        {
            if (IsWaitNode(node)) {
                const auto iter = graph.startWaitNodes.find(node);
                return iter == graph.startWaitNodes.end() ? nullptr : iter->second;
            }
            const auto iter = graph.normalNodes.find(node);
            return iter == graph.normalNodes.end() ? nullptr : iter->second;
        }

        // 判断两个节点是否位于同一执行通道（rank + stream + queue）。
        // post 与其 wait 同通道时，表示 post 完成即可解锁 wait 的开始。
        bool IsSameRankStreamQueue(const TaskNode* lhs, const TaskNode* rhs)
        {
            if (lhs == nullptr || rhs == nullptr) {
                return false;
            }
            const TaskPosition& lhsPosition = lhs->GetPosition();
            const TaskPosition& rhsPosition = rhs->GetPosition();
            return lhsPosition.rankId == rhsPosition.rankId && lhsPosition.streamId == rhsPosition.streamId
                   && lhsPosition.queueId == rhsPosition.queueId;
        }

        // 向拷贝图追加 parent->child 边，两端均做去重。
        void AddCopyEdge(CopyNode* parent, CopyNode* child)
        {
            if (parent == nullptr || child == nullptr || parent == child) {
                return;
            }
            if (std::find(parent->children.begin(), parent->children.end(), child) == parent->children.end()) {
                parent->children.push_back(child);
            }
            if (std::find(child->parents.begin(), child->parents.end(), parent) == child->parents.end()) {
                child->parents.push_back(parent);
            }
        }

        // 构造可达性检查所用的"拷贝"DAG。
        // 每个 wait 节点被拆成两份拷贝：StartWait（wait 可开始的时刻）与 ReceiveWait（被等待
        // 资源已到达的时刻），并固定连一条 StartWait->ReceiveWait 边，使"经过 wait 可达"即意味
        // 着"等待已完成"；非 wait 节点只生成一份 normal 拷贝。
        // 随后按父节点是否为该 wait 的匹配生产者、以及是否与 wait 同通道，独立地为 parent 连
        // 到 StartWait 和/或 ReceiveWait 的边：
        //   - 匹配生产者且同通道 -> 同时连 StartWait 和 ReceiveWait（post 完成既解锁 wait 开始，
        //     又送达资源）；
        //   - 匹配生产者但不同通道 -> 只连 ReceiveWait（资源远端到达）；
        //   - 非匹配父节点 -> 只连 StartWait（纯控制序，回退到开始点）。
        HcclResult BuildCopiedGraph(
            const std::vector<const TaskNode*>& nodes, const SyncParentMap& syncParents, CopiedGraph& graph)
        {
            graph = CopiedGraph{};
            graph.nodes.reserve(nodes.size() + 1U);
            for (const TaskNode* origin : nodes) {
                if (origin == nullptr) {
                    return HCCL_E_PTR;
                }
                if (IsWaitNode(origin)) {
                    auto startWait = std::make_unique<CopyNode>();
                    startWait->origin = origin;
                    startWait->role = CopyNodeRole::START_WAIT;
                    startWait->index = graph.nodes.size();
                    CopyNode* startWaitPtr = startWait.get();
                    graph.nodes.emplace_back(std::move(startWait));
                    graph.startWaitNodes[origin] = startWaitPtr;

                    auto receiveWait = std::make_unique<CopyNode>();
                    receiveWait->origin = origin;
                    receiveWait->role = CopyNodeRole::RECEIVE_WAIT;
                    receiveWait->index = graph.nodes.size();
                    CopyNode* receiveWaitPtr = receiveWait.get();
                    graph.nodes.emplace_back(std::move(receiveWait));
                    graph.receiveWaitNodes[origin] = receiveWaitPtr;
                    AddCopyEdge(startWaitPtr, receiveWaitPtr);
                    continue;
                }

                auto copy = std::make_unique<CopyNode>();
                copy->origin = origin;
                copy->role = CopyNodeRole::NORMAL;
                copy->index = graph.nodes.size();
                CopyNode* copyPtr = copy.get();
                graph.nodes.emplace_back(std::move(copy));
                graph.normalNodes[origin] = copyPtr;
            }

            const auto startIter = graph.normalNodes.find(nodes.front());
            graph.start = startIter == graph.normalNodes.end() ? nullptr : startIter->second;
            if (graph.start == nullptr) {
                return HCCL_E_INTERNAL;
            }

            for (const TaskNode* parent : nodes) {
                for (const TaskNode* child : parent->GetChildren()) {
                    const auto syncIter = syncParents.find(child);
                    const bool childIsWait = IsWaitNode(child);
                    if (childIsWait) {
                        CopyNode* parentCopy = FindCompletionCopy(graph, parent);
                        if (parentCopy == nullptr) {
                            return HCCL_E_INTERNAL;
                        }
                        const bool isSyncParent = syncIter != syncParents.end() && syncIter->second.count(parent) != 0;
                        const bool isSameRankStreamQueue = IsSameRankStreamQueue(parent, child);

                        // 独立决定连向 StartWait 与 ReceiveWait 的边。配对 post 若与 wait 同通道，
                        // 则有意同时连到两份拷贝；非匹配父节点则只回退连到 StartWait。
                        if (isSameRankStreamQueue || !isSyncParent) {
                            AddCopyEdge(parentCopy, graph.startWaitNodes.at(child));
                        }
                        if (isSyncParent) {
                            AddCopyEdge(parentCopy, graph.receiveWaitNodes.at(child));
                        }
                        continue;
                    }

                    CopyNode* parentCopy = FindCompletionCopy(graph, parent);
                    CopyNode* childCopy = FindCompletionCopy(graph, child);
                    if (parentCopy == nullptr || childCopy == nullptr) {
                        return HCCL_E_INTERNAL;
                    }
                    AddCopyEdge(parentCopy, childCopy);
                }
            }
            return HCCL_SUCCESS;
        }

        // Kahn 算法配合最小索引优先队列，得到稳定的、索引从小到大的拓扑序。
        // 若出现环（拓扑结果数量少于节点数）则报 DAG 错误。
        HcclResult BuildTopologicalOrder(const CopiedGraph& graph, std::vector<CopyNode*>& topoOrder)
        {
            topoOrder.clear();
            std::vector<size_t> indegree(graph.nodes.size(), 0);
            for (const auto& node : graph.nodes) {
                if (node == nullptr) {
                    return HCCL_E_PTR;
                }
                for (const CopyNode* child : node->children) {
                    if (child == nullptr || child->index >= indegree.size()) {
                        return HCCL_E_INTERNAL;
                    }
                    ++indegree[child->index];
                }
            }

            std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> ready;
            for (size_t index = 0; index < indegree.size(); ++index) {
                if (indegree[index] == 0) {
                    ready.push(index);
                }
            }
            while (!ready.empty()) {
                const size_t index = ready.top();
                ready.pop();
                topoOrder.push_back(graph.nodes[index].get());
                for (const CopyNode* child : graph.nodes[index]->children) {
                    if (--indegree[child->index] == 0) {
                        ready.push(child->index);
                    }
                }
            }
            if (topoOrder.size() != graph.nodes.size()) {
                HCCL_VM_ERROR(
                    "{} Sync conflict check failed because the copied sync DAG contains a cycle, "
                    "copiedNodeCount={}, topoNodeCount={}",
                    MakeErrorCodeText(ErrorCode::SYNC_DAG_INVALID), graph.nodes.size(), topoOrder.size());
                return HCCL_E_INTERNAL;
            }
            return HCCL_SUCCESS;
        }

        // 拷贝 DAG 上的可达性 oracle。预计算的拓扑索引使每次查询都能剪掉拓扑位置 >= 目标
        // 的后继（它们不可能到达目标），把每次 IsReachable 退化为有界前向 DFS。单调递增的
        // query stamp 避免在查询之间反复清零 visited 数组（仅溢出时重置）。
        class ReachabilityChecker {
        public:
            ReachabilityChecker(size_t nodeCount, const std::vector<size_t>& topoIndex)
                : nodeCount_(nodeCount),
                  topoIndex_(topoIndex),
                  visitStamp_(nodeCount, 0)
            {}

            // 当 `to` 可经 child 边从 `from` 到达时返回 true，使用上述拓扑剪枝。
            bool IsReachable(const CopyNode* from, const CopyNode* to)
            {
                // 只回答被查询的那一对；拓扑剪枝避免探索目标之后的节点。
                if (from == nullptr || to == nullptr || from->index >= nodeCount_ || to->index >= nodeCount_) {
                    return false;
                }
                if (from == to) {
                    return true;
                }
                if (from->index >= topoIndex_.size() || to->index >= topoIndex_.size()
                    || topoIndex_[from->index] >= topoIndex_[to->index]) {
                    return false;
                }

                if (queryStamp_ == std::numeric_limits<size_t>::max()) {
                    std::fill(visitStamp_.begin(), visitStamp_.end(), 0);
                    queryStamp_ = 1;
                } else {
                    ++queryStamp_;
                }

                const size_t targetTopo = topoIndex_[to->index];
                std::vector<const CopyNode*> pending;
                pending.push_back(from);
                visitStamp_[from->index] = queryStamp_;
                while (!pending.empty()) {
                    const CopyNode* node = pending.back();
                    pending.pop_back();
                    for (const CopyNode* child : node->children) {
                        if (child == nullptr || child->index >= nodeCount_ || child->index >= topoIndex_.size()) {
                            continue;
                        }
                        if (child == to) {
                            return true;
                        }
                        if (topoIndex_[child->index] >= targetTopo || visitStamp_[child->index] == queryStamp_) {
                            continue;
                        }
                        visitStamp_[child->index] = queryStamp_;
                        pending.push_back(child);
                    }
                }
                return false;
            }

        private:
            size_t nodeCount_{0};
            const std::vector<size_t>& topoIndex_;
            std::vector<size_t> visitStamp_;
            size_t queryStamp_{0};
        };

        // 薄包装：使调用点写作 IsReachable(from, to, checker) 而非 checker.IsReachable(from, to)，
        // 仅为下方校验点的可读性保留。
        bool IsReachable(const CopyNode* from, const CopyNode* to, ReachabilityChecker& reachable)
        {
            return reachable.IsReachable(from, to);
        }

        // 拷贝图中 parent->child 边的总数（仅供统计/诊断使用）。
        size_t CountCopyEdges(const CopiedGraph& graph)
        {
            size_t count = 0;
            for (const auto& node : graph.nodes) {
                if (node != nullptr) {
                    count += node->children.size();
                }
            }
            return count;
        }

        // 检测同一 bucket 内是否有单个 Post（byPost=true）或 ReceiveWait（byPost=false）
        // 被配对了多于一个对端。对 CheckBuckets 处理的通用 1:1 资源而言，这种多对一配对是非法的
        // （AIV SendFlag 扇出已在别处单独处理，不进入此处）。
        bool HasMultiplePeer(const std::vector<SyncPair>& pairs, bool byPost)
        {
            std::map<const CopyNode*, std::set<const CopyNode*>> peers;
            for (const SyncPair& pair : pairs) {
                if (!byPost && pair.receiveWaitCopy == nullptr) {
                    continue;
                }
                const CopyNode* key = byPost ? pair.postCopy : pair.receiveWaitCopy;
                const CopyNode* peer = byPost ? pair.receiveWaitCopy : pair.postCopy;
                peers[key].insert(peer);
            }
            return std::any_of(peers.begin(), peers.end(), [](const auto& entry) {
                return entry.second.size() > 1;
            });
        }

        // 判断资源是否为 AIV SendFlag/RecvFlag（基于 flag cell 的）类型，该类型由专用扇出
        // 检查器校验，而非通用 bucket 检查器。
        bool IsAivFlagResource(SyncResourceKind kind) { return kind == SyncResourceKind::AIV_FLAG; }

        // 构造 flag cell 标识（rank + launchIdx + commInfoOffset），所有仅在 flagValue 上不同的
        // bucket 共享此标识。cell 把同一逻辑 flag 位置的不同取值聚合起来，用于跨取值的顺序校验。
        AivFlagCellKey MakeAivFlagCellKey(const SyncResourceKey& resource)
        {
            return AivFlagCellKey{resource.rankId, resource.launchIdx, resource.commInfoOffset};
        }

        // flag cell 的单行描述，供 AIV 专用错误日志使用。
        std::string DescribeAivFlagCell(const AivFlagCellKey& cell)
        {
            std::ostringstream os;
            os << "flagOwnerRank=" << cell.rankId << ", launchIdx=" << cell.launchIdx << ", commInfoOffset=0x"
               << std::hex << cell.commInfoOffset << std::dec;
            return os.str();
        }

        HcclResult CheckAivFlagProducerOrdering(
            const AivFlagCellKey& cell, const AivFlagGroup& previousGroup, const AivFlagGroup& nextGroup,
            ReachabilityChecker& reachable, SyncConflictCheckStats& stats)
        {
            for (const SyncPair* previousPair : previousGroup.pairs) {
                if (previousPair == nullptr || previousPair->wait == nullptr) {
                    return HCCL_E_INTERNAL;
                }
                if (IsReachable(previousPair->receiveWaitCopy, nextGroup.postCopy, reachable)) {
                    continue;
                }
                ++stats.conflictCount;
                HCCL_VM_ERROR(
                    "{} AIV flag resource has a many-to-one ordering conflict, "
                    "conflictType=many-to-one, producerTaskType=AIV_SEND_FLAG, "
                    "consumerTaskType=AIV_RECV_FLAG, cell={}, recvFlag={}, previousSendFlag={}, "
                    "nextSendFlag={}",
                    MakeErrorCodeText(ErrorCode::SYNC_RESOURCE_CONFLICT), DescribeAivFlagCell(cell),
                    previousPair->wait->Describe(), previousPair->post->Describe(), nextGroup.post->Describe());
                return HCCL_E_INTERNAL;
            }
            return HCCL_SUCCESS;
        }

        HcclResult CheckAivFlagFanOutOrdering(
            const AivFlagCellKey& cell, const AivFlagGroup& previousGroup, const AivFlagGroup& nextGroup,
            ReachabilityChecker& reachable, SyncConflictCheckStats& stats)
        {
            for (const SyncPair* previousPair : previousGroup.pairs) {
                if (previousPair == nullptr || previousPair->wait == nullptr) {
                    return HCCL_E_INTERNAL;
                }
                for (const SyncPair* nextPair : nextGroup.pairs) {
                    if (nextPair == nullptr || nextPair->wait == nullptr) {
                        return HCCL_E_INTERNAL;
                    }
                    if (IsReachable(previousPair->receiveWaitCopy, nextPair->startWaitCopy, reachable)) {
                        continue;
                    }
                    ++stats.conflictCount;
                    HCCL_VM_ERROR(
                        "{} AIV flag resource has a one-to-many ordering conflict, "
                        "conflictType=one-to-many, producerTaskType=AIV_SEND_FLAG, "
                        "consumerTaskType=AIV_RECV_FLAG, cell={}, flagValue={}, sendFlag={}, previousRecvFlag={}, "
                        "nextRecvFlag={}",
                        MakeErrorCodeText(ErrorCode::SYNC_RESOURCE_CONFLICT), DescribeAivFlagCell(cell),
                        nextGroup.flagValue, previousPair->post->Describe(), previousPair->wait->Describe(),
                        nextPair->wait->Describe());
                    return HCCL_E_INTERNAL;
                }
            }
            return HCCL_SUCCESS;
        }

        // AIV SendFlag/RecvFlag 资源的专用校验器。该类型允许一对多扇出（一个 SendFlag 可满足
        // 多个 RecvFlag），因此不能用通用 1:1 的 CheckBuckets 规则。flag cell = (flagOwnerRank,
        // launchIdx, commInfoOffset) 把所有仅在 flagValue 上不同的 bucket 聚合在一起。不同 value
        // 之间只检查上一轮消费者到下一轮生产者的覆盖顺序；一对多的消费者启动顺序只在同 value
        // 的跨组之间检查。
        //
        // 每个 cell 的检查分两阶段：
        //   1. 配对校验：在每个有等待方的 bucket 内，每个 RecvFlag 必须恰好匹配一个 SendFlag
        //      （无匹配或多匹配均判为冲突）。整个 cell 内没有任何 RecvFlag 的 SendFlag 是合法的
        //      仅生产方操作，跳过。
        //   2. 跨组顺序：把配对按 Post 折叠成带 value 的组，再按 post 的拓扑序排序后遍历。全局
        //      检查每个已匹配组的 ReceiveWait 是否能到达下一组的 Post；同一 value 内再检查其
        //      ReceiveWait 是否能到达下一组各 RecvFlag 的 StartWait。残留的（无 wait）post 也
        //      必须排在上一已匹配组之后。
        HcclResult CheckAivFlagBuckets(
            const SyncBuckets& buckets, const std::vector<size_t>& topoIndex, ReachabilityChecker& reachable,
            SyncConflictCheckStats& stats)
        {
            // 阶段 0：按 flag cell 收集 AIV_FLAG bucket（折叠 flagValue）。
            std::map<AivFlagCellKey, std::vector<const SyncBucket*>> cellBuckets;
            for (const auto& entry : buckets) {
                if (IsAivFlagResource(entry.first.kind)) {
                    cellBuckets[MakeAivFlagCellKey(entry.first)].push_back(&entry.second);
                }
            }

            for (const auto& cellEntry : cellBuckets) {
                const AivFlagCellKey& cell = cellEntry.first;
                std::map<const TaskNode*, AivFlagGroup> groupsByPost;
                bool cellHasWait = false;

                for (const SyncBucket* bucket : cellEntry.second) {
                    if (bucket == nullptr) {
                        return HCCL_E_INTERNAL;
                    }

                    if (!bucket->originalWaits.empty()) {
                        cellHasWait = true;
                        ++stats.checkedBucketCount;
                        std::map<const TaskNode*, std::set<const TaskNode*>> waitPosts;
                        for (const SyncPair& pair : bucket->pairs) {
                            if (pair.wait != nullptr) {
                                waitPosts[pair.wait].insert(pair.post);
                            }
                        }
                        for (const TaskNode* wait : bucket->originalWaits) {
                            const auto iter = waitPosts.find(wait);
                            if (iter == waitPosts.end() || iter->second.empty()) {
                                ++stats.conflictCount;
                                HCCL_VM_ERROR(
                                    "{} AIV flag resource has an unmatched consumer task, cell={}, "
                                    "consumerTaskType=AIV_RECV_FLAG, recvFlag={}",
                                    MakeErrorCodeText(ErrorCode::SYNC_RESOURCE_CONFLICT), DescribeAivFlagCell(cell),
                                    wait->Describe());
                                return HCCL_E_INTERNAL;
                            }
                            if (iter->second.size() > 1U) {
                                ++stats.conflictCount;
                                HCCL_VM_ERROR(
                                    "{} AIV flag resource has a many-to-one conflict, "
                                    "conflictType=many-to-one, producerTaskType=AIV_SEND_FLAG, "
                                    "consumerTaskType=AIV_RECV_FLAG, cell={}, recvFlag={}, matchingSendFlagCount={}",
                                    MakeErrorCodeText(ErrorCode::SYNC_RESOURCE_CONFLICT), DescribeAivFlagCell(cell),
                                    wait->Describe(), iter->second.size());
                                return HCCL_E_INTERNAL;
                            }
                        }
                    }
                }

                // 同一 flag cell 内没有任何 RecvFlag 的 SendFlag 是合法的仅生产方操作，
                // 不存在需要校验的顺序关系。
                if (!cellHasWait) {
                    continue;
                }

                // 阶段 2：跨组顺序。把该 cell 的所有配对按共享的 Post 折叠成 AivFlagGroup
                // （每个 Post 一组），再按 post 的拓扑序排序，以便按执行顺序遍历生产方组。
                for (const SyncBucket* bucket : cellEntry.second) {
                    for (const SyncPair& pair : bucket->pairs) {
                        auto iter = groupsByPost.find(pair.post);
                        if (iter == groupsByPost.end()) {
                            AivFlagGroup group;
                            group.post = pair.post;
                            group.postCopy = pair.postCopy;
                            group.flagValue = pair.resource.flagValue;
                            iter = groupsByPost.emplace(pair.post, std::move(group)).first;
                        }
                        iter->second.pairs.push_back(&pair);
                    }
                }

                std::vector<const AivFlagGroup*> groups;
                groups.reserve(groupsByPost.size());
                for (const auto& entry : groupsByPost) {
                    const AivFlagGroup& group = entry.second;
                    if (group.post == nullptr || group.postCopy == nullptr
                        || group.postCopy->index >= topoIndex.size()) {
                        return HCCL_E_INTERNAL;
                    }
                    groups.push_back(&group);
                }
                std::sort(groups.begin(), groups.end(), [&topoIndex](const AivFlagGroup* lhs, const AivFlagGroup* rhs) {
                    const size_t lhsTopo = topoIndex[lhs->postCopy->index];
                    const size_t rhsTopo = topoIndex[rhs->postCopy->index];
                    if (lhsTopo != rhsTopo) {
                        return lhsTopo < rhsTopo;
                    }
                    return lhs->post->GetNodeId() < rhs->post->GetNodeId();
                });

                // `lastMatchedGroup` 用于保留同一 cell 上全局的生产顺序检查；不同 value 的 RecvFlag
                // 可以并发启动，因此一对多顺序单独按 value 记录最近的 matched group。
                const AivFlagGroup* lastMatchedGroup = nullptr;
                std::map<int32_t, const AivFlagGroup*> lastMatchedGroupByValue;
                for (const AivFlagGroup* group : groups) {
                    // 一组是否"已匹配"取决于其配对是否携带 wait。由构造（groupsByPost）可知同组所有
                    // 配对共享同一 Post 且 wait 是否存在一致，故检查首条配对即可判定整组。
                    const bool hasWait = !group->pairs.empty() && group->pairs.front()->wait != nullptr;
                    if (!hasWait) {
                        if (lastMatchedGroup == nullptr) {
                            ++stats.conflictCount;
                            HCCL_VM_ERROR(
                                "{} AIV flag resource has an unmatched producer task before the first "
                                "matched pair, producerTaskType=AIV_SEND_FLAG, consumerTaskType=AIV_RECV_FLAG, "
                                "cell={}, sendFlag={}",
                                MakeErrorCodeText(ErrorCode::SYNC_RESOURCE_CONFLICT), DescribeAivFlagCell(cell),
                                group->post->Describe());
                            return HCCL_E_INTERNAL;
                        }
                        HcclResult ret
                            = CheckAivFlagProducerOrdering(cell, *lastMatchedGroup, *group, reachable, stats);
                        if (ret != HCCL_SUCCESS) {
                            return ret;
                        }
                        continue;
                    }

                    if (lastMatchedGroup != nullptr) {
                        HcclResult ret
                            = CheckAivFlagProducerOrdering(cell, *lastMatchedGroup, *group, reachable, stats);
                        if (ret != HCCL_SUCCESS) {
                            return ret;
                        }
                    }

                    const auto previousSameValue = lastMatchedGroupByValue.find(group->flagValue);
                    if (previousSameValue != lastMatchedGroupByValue.end()) {
                        HcclResult ret
                            = CheckAivFlagFanOutOrdering(cell, *previousSameValue->second, *group, reachable, stats);
                        if (ret != HCCL_SUCCESS) {
                            return ret;
                        }
                    }
                    lastMatchedGroup = group;
                    lastMatchedGroupByValue[group->flagValue] = group;
                }
            }
            return HCCL_SUCCESS;
        }

        // 针对 1:1 严格 post/wait 资源（AICPU Notify、CCU CKE、AIV pipe-event SetFlag/WaitFlag）
        // 的通用逐 bucket 校验器。对每个资源 bucket 强制：
        //   - Post 数不少于 Wait 数；
        //   - 没有任何 Post/Wait 参与多于一个对端配对（HasMultiplePeer）；
        //   - 顺序覆盖约束：每个已匹配对的 ReceiveWait 必须能到达下一对的 Post 以及下一对的
        //     StartWait，使后继的生产/消费方不能越过先前同资源的操作；残留（未匹配）的 Post 也
        //     必须排在上一已匹配对的 ReceiveWait 之后。
        // AIV SendFlag/RecvFlag 因允许一对多扇出而在此排除（由 CheckAivFlagBuckets 处理）。
        HcclResult
        CheckBuckets(const SyncBuckets& buckets, ReachabilityChecker& reachable, SyncConflictCheckStats& stats)
        {
            for (const auto& entry : buckets) {
                const SyncResourceKey& resource = entry.first;
                const SyncBucket& bucket = entry.second;
                const std::vector<SyncPair>& pairs = bucket.pairs;

                const size_t postCount = bucket.originalPosts.size();
                const size_t waitCount = bucket.originalWaits.size();

                // AIV SendFlag/RecvFlag 由上方的专用扇出检查器处理。
                if (IsAivFlagResource(resource.kind)) {
                    continue;
                }

                if (postCount < waitCount) {
                    ++stats.checkedBucketCount;
                    ++stats.conflictCount;
                    HCCL_VM_ERROR(
                        "{} Sync resource has fewer producer tasks than consumer tasks, "
                        "producerTaskType={}, consumerTaskType={}, resource={}, producerCount={}, consumerCount={}",
                        MakeErrorCodeText(ErrorCode::SYNC_RESOURCE_CONFLICT), SyncPostTaskTypeName(resource.kind),
                        SyncWaitTaskTypeName(resource.kind), DescribeResource(resource), postCount, waitCount);
                    return HCCL_E_INTERNAL;
                }

                if (waitCount == 0U || pairs.size() <= 1U) {
                    continue;
                }
                ++stats.checkedBucketCount;

                if (HasMultiplePeer(pairs, true) || HasMultiplePeer(pairs, false)) {
                    ++stats.conflictCount;
                    HCCL_VM_ERROR(
                        "{} Sync resource has multiple producer/consumer peers, "
                        "producerTaskType={}, consumerTaskType={}, resource={}, producerCount={}, consumerCount={}, "
                        "pairCount={}",
                        MakeErrorCodeText(ErrorCode::SYNC_RESOURCE_CONFLICT), SyncPostTaskTypeName(resource.kind),
                        SyncWaitTaskTypeName(resource.kind), DescribeResource(resource), postCount, waitCount,
                        pairs.size());
                    return HCCL_E_INTERNAL;
                }

                const SyncPair* lastPaired = nullptr;
                for (const SyncPair& pair : pairs) {
                    if (pair.wait == nullptr) {
                        if (lastPaired == nullptr) {
                            ++stats.conflictCount;
                            HCCL_VM_ERROR(
                                "{} Sync resource has an unmatched producer task before the first matched "
                                "pair, producerTaskType={}, consumerTaskType={}, resource={}, producer={}",
                                MakeErrorCodeText(ErrorCode::SYNC_RESOURCE_CONFLICT),
                                SyncPostTaskTypeName(resource.kind), SyncWaitTaskTypeName(resource.kind),
                                DescribeResource(resource), pair.post->Describe());
                            return HCCL_E_INTERNAL;
                        }
                        if (!IsReachable(lastPaired->receiveWaitCopy, pair.postCopy, reachable)) {
                            ++stats.conflictCount;
                            HCCL_VM_ERROR(
                                "{} Sync resource has a many-to-one ordering conflict, "
                                "conflictType=many-to-one, producerTaskType={}, consumerTaskType={}, resource={}, "
                                "consumer={}, previousProducer={}, nextProducer={}",
                                MakeErrorCodeText(ErrorCode::SYNC_RESOURCE_CONFLICT),
                                SyncPostTaskTypeName(resource.kind), SyncWaitTaskTypeName(resource.kind),
                                DescribeResource(resource), lastPaired->wait->Describe(), lastPaired->post->Describe(),
                                pair.post->Describe());
                            return HCCL_E_INTERNAL;
                        }
                        continue;
                    }

                    if (lastPaired == nullptr) {
                        lastPaired = &pair;
                        continue;
                    }

                    const SyncPair& current = *lastPaired;
                    const SyncPair& next = pair;
                    if (current.receiveWaitCopy == nullptr) {
                        ++stats.conflictCount;
                        HCCL_VM_ERROR(
                            "{} Sync conflict checker has an invalid pair without a consumer completion node, "
                            "producerTaskType={}, consumerTaskType={}, resource={}, producer={}, consumer={}",
                            MakeErrorCodeText(ErrorCode::SYNC_RESOURCE_CONFLICT), SyncPostTaskTypeName(resource.kind),
                            SyncWaitTaskTypeName(resource.kind), DescribeResource(resource), current.post->Describe(),
                            current.wait->Describe());
                        return HCCL_E_INTERNAL;
                    }
                    if (!IsReachable(current.receiveWaitCopy, next.postCopy, reachable)) {
                        ++stats.conflictCount;
                        HCCL_VM_ERROR(
                            "{} Sync resource has a many-to-one ordering conflict, "
                            "conflictType=many-to-one, producerTaskType={}, consumerTaskType={}, resource={}, "
                            "order={}, "
                            "consumer={}, previousProducer={}, nextProducer={}",
                            MakeErrorCodeText(ErrorCode::SYNC_RESOURCE_CONFLICT), SyncPostTaskTypeName(resource.kind),
                            SyncWaitTaskTypeName(resource.kind), DescribeResource(resource), next.order,
                            current.wait->Describe(), current.post->Describe(), next.post->Describe());
                        return HCCL_E_INTERNAL;
                    }
                    if (!IsReachable(current.receiveWaitCopy, next.startWaitCopy, reachable)) {
                        ++stats.conflictCount;
                        HCCL_VM_ERROR(
                            "{} Sync resource has a one-to-many ordering conflict, "
                            "conflictType=one-to-many, producerTaskType={}, consumerTaskType={}, resource={}, "
                            "order={}, "
                            "producer={}, previousConsumer={}, nextConsumer={}",
                            MakeErrorCodeText(ErrorCode::SYNC_RESOURCE_CONFLICT), SyncPostTaskTypeName(resource.kind),
                            SyncWaitTaskTypeName(resource.kind), DescribeResource(resource), next.order,
                            current.post->Describe(), current.wait->Describe(), next.wait->Describe());
                        return HCCL_E_INTERNAL;
                    }
                    lastPaired = &pair;
                }
            }
            return HCCL_SUCCESS;
        }
    } // namespace

    // 同步资源冲突检查的入口。
    // 流水线（从此处起最大调用深度 2）：收集遍历 -> 构建 post/wait bucket 与拷贝 DAG -> 对
    // DAG 做拓扑排序 -> 为每个配对绑定其拷贝节点 -> 先运行 AIV flag 扇出校验器，再运行通用
    // 1:1 校验器。统计经 `stats` 输出。
    HcclResult CheckSyncResourceConflict(const TaskNode* start, SyncConflictCheckStats* stats)
    {
        SyncConflictCheckStats localStats;
        if (start == nullptr) {
            return HCCL_E_PTR;
        }

        std::vector<const TaskNode*> originalNodes;
        HcclResult ret = CollectOriginalNodes(start, originalNodes);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        localStats.originalNodeCount = originalNodes.size();

        SyncBuckets buckets;
        SyncParentMap syncParents;
        ret = BuildSyncPairs(originalNodes, buckets, syncParents);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }

        CopiedGraph copiedGraph;
        ret = BuildCopiedGraph(originalNodes, syncParents, copiedGraph);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        localStats.copiedNodeCount = copiedGraph.nodes.size();
        localStats.copiedEdgeCount = CountCopyEdges(copiedGraph);
        localStats.resourceBucketCount = buckets.size();

        for (auto& entry : buckets) {
            SyncBucket& bucket = entry.second;
            for (const TaskNode* post : bucket.originalPosts) {
                CopyNode* postCopy = FindCompletionCopy(copiedGraph, post);
                if (postCopy == nullptr) {
                    return HCCL_E_INTERNAL;
                }
                bucket.postCopies.insert(postCopy);
            }
            for (SyncPair& pair : bucket.pairs) {
                pair.postCopy = FindCompletionCopy(copiedGraph, pair.post);
                if (pair.postCopy == nullptr) {
                    return HCCL_E_INTERNAL;
                }
                if (pair.wait != nullptr) {
                    pair.startWaitCopy = FindStartCopy(copiedGraph, pair.wait);
                    pair.receiveWaitCopy = FindCompletionCopy(copiedGraph, pair.wait);
                    if (pair.startWaitCopy == nullptr || pair.receiveWaitCopy == nullptr) {
                        return HCCL_E_INTERNAL;
                    }
                }
            }

            std::set<const TaskNode*> pairedPosts;
            for (const SyncPair& pair : bucket.pairs) {
                pairedPosts.insert(pair.post);
            }
            for (const TaskNode* post : bucket.originalPosts) {
                if (pairedPosts.count(post) == 0) {
                    SyncPair unmatchedPair;
                    unmatchedPair.resource = entry.first;
                    unmatchedPair.post = post;
                    unmatchedPair.postCopy = FindCompletionCopy(copiedGraph, post);
                    if (unmatchedPair.postCopy == nullptr) {
                        return HCCL_E_INTERNAL;
                    }
                    bucket.pairs.push_back(unmatchedPair);
                }
            }
            localStats.pairCount += bucket.pairs.size();
        }

        std::vector<CopyNode*> topoOrder;
        ret = BuildTopologicalOrder(copiedGraph, topoOrder);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        std::vector<size_t> topoIndex(copiedGraph.nodes.size(), 0);
        for (size_t index = 0; index < topoOrder.size(); ++index) {
            topoIndex[topoOrder[index]->index] = index;
        }

        for (auto& entry : buckets) {
            std::vector<SyncPair>& pairs = entry.second.pairs;
            std::sort(pairs.begin(), pairs.end(), [&topoIndex](const SyncPair& lhs, const SyncPair& rhs) {
                const size_t lhsPost = topoIndex[lhs.postCopy->index];
                const size_t rhsPost = topoIndex[rhs.postCopy->index];
                if (lhsPost != rhsPost) {
                    return lhsPost < rhsPost;
                }
                const size_t lhsWait = lhs.receiveWaitCopy == nullptr ? std::numeric_limits<size_t>::max() :
                                                                        topoIndex[lhs.receiveWaitCopy->index];
                const size_t rhsWait = rhs.receiveWaitCopy == nullptr ? std::numeric_limits<size_t>::max() :
                                                                        topoIndex[rhs.receiveWaitCopy->index];
                if (lhsWait != rhsWait) {
                    return lhsWait < rhsWait;
                }
                if (lhs.post->GetNodeId() != rhs.post->GetNodeId()) {
                    return lhs.post->GetNodeId() < rhs.post->GetNodeId();
                }
                const NodeId lhsWaitId = lhs.wait == nullptr ? INVALID_NODE_ID : lhs.wait->GetNodeId();
                const NodeId rhsWaitId = rhs.wait == nullptr ? INVALID_NODE_ID : rhs.wait->GetNodeId();
                return lhsWaitId < rhsWaitId;
            });
            for (size_t index = 0; index < pairs.size(); ++index) {
                pairs[index].order = index + 1U;
            }
        }

        ReachabilityChecker reachable(copiedGraph.nodes.size(), topoIndex);
        ret = CheckAivFlagBuckets(buckets, topoIndex, reachable, localStats);
        if (ret == HCCL_SUCCESS) {
            ret = CheckBuckets(buckets, reachable, localStats);
        }
        if (stats != nullptr) {
            *stats = localStats;
        }
        return ret;
    }
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
