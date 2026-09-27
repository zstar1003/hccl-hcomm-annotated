/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_graph_mem_conflict_v3.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "sim_log.h"
#include "task_graph_reachability_v3.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {
        enum class MemoryAccessKind : uint8_t {
            READ = 0,
            WRITE,
        };

        struct MemoryKey {
            RankId rankId{INVALID_RANK_ID};
            MemType memType{MemType::INVALID};
            uint64_t aivUbIdx{0};

            bool operator<(const MemoryKey& rhs) const
            {
                if (rankId != rhs.rankId) {
                    return rankId < rhs.rankId;
                }
                if (memType != rhs.memType) {
                    return static_cast<uint32_t>(memType) < static_cast<uint32_t>(rhs.memType);
                }
                return aivUbIdx < rhs.aivUbIdx;
            }
        };

        struct MemoryAccess {
            MemoryKey key;
            uint64_t start{0};
            uint64_t end{0};
            MemoryAccessKind kind{MemoryAccessKind::READ};
            NodeId nodeId{INVALID_NODE_ID};
            const TaskNode* node{nullptr};
        };

        using TaskMemoryAccesses = std::vector<MemoryAccess>;
        using MemoryAccessBuckets = std::map<MemoryKey, TaskMemoryAccesses>;
        using MemConflictClock = std::chrono::steady_clock;

        constexpr uint64_t MEM_CONFLICT_NS_PER_MS = 1000000ULL;

        struct AccessStartLess {
            bool operator()(const MemoryAccess& lhs, const MemoryAccess& rhs) const
            {
                if (lhs.start != rhs.start) {
                    return lhs.start < rhs.start;
                }
                if (lhs.end != rhs.end) {
                    return lhs.end < rhs.end;
                }
                if (lhs.nodeId != rhs.nodeId) {
                    return lhs.nodeId < rhs.nodeId;
                }
                return static_cast<uint32_t>(lhs.kind) < static_cast<uint32_t>(rhs.kind);
            }
        };

        struct AccessExactLess {
            bool operator()(const MemoryAccess& lhs, const MemoryAccess& rhs) const
            {
                if (lhs.key.rankId != rhs.key.rankId) {
                    return lhs.key.rankId < rhs.key.rankId;
                }
                if (lhs.key.memType != rhs.key.memType) {
                    return static_cast<uint32_t>(lhs.key.memType) < static_cast<uint32_t>(rhs.key.memType);
                }
                if (lhs.key.aivUbIdx != rhs.key.aivUbIdx) {
                    return lhs.key.aivUbIdx < rhs.key.aivUbIdx;
                }
                if (lhs.start != rhs.start) {
                    return lhs.start < rhs.start;
                }
                if (lhs.end != rhs.end) {
                    return lhs.end < rhs.end;
                }
                if (lhs.kind != rhs.kind) {
                    return static_cast<uint32_t>(lhs.kind) < static_cast<uint32_t>(rhs.kind);
                }
                return lhs.nodeId < rhs.nodeId;
            }
        };

        struct AccessEndLess {
            bool operator()(const MemoryAccess* lhs, const MemoryAccess* rhs) const
            {
                if (lhs == rhs) {
                    return false;
                }
                if (lhs->end != rhs->end) {
                    return lhs->end < rhs->end;
                }
                if (lhs->start != rhs->start) {
                    return lhs->start < rhs->start;
                }
                if (lhs->nodeId != rhs->nodeId) {
                    return lhs->nodeId < rhs->nodeId;
                }
                if (lhs->kind != rhs->kind) {
                    return static_cast<uint32_t>(lhs->kind) < static_cast<uint32_t>(rhs->kind);
                }
                return std::less<const MemoryAccess*>()(lhs, rhs);
            }
        };

        using ActiveAccesses = std::multiset<const MemoryAccess*, AccessEndLess>;

        struct OverlapCandidateStatsCollector {
            std::unordered_set<NodeId> nodeIds;
            std::map<std::string, size_t> pairCountByBucket;
            std::map<std::string, size_t> pairCountByTaskTypePair;
        };

        struct ConflictCandidate {
            const MemoryAccess* current{nullptr};
            const MemoryAccess* history{nullptr};
        };

        using ConflictCandidates = std::vector<ConflictCandidate>;

        const char* AccessKindName(MemoryAccessKind kind) { return kind == MemoryAccessKind::WRITE ? "WRITE" : "READ"; }

        const char* TaskTypeName(TaskType taskType)
        {
            switch (taskType) {
                case TaskType::TRANS_MEM:
                    return "TRANS_MEM";
                case TaskType::BATCH_TRANS_MEM:
                    return "BATCH_TRANS_MEM";
                case TaskType::REDUCE:
                    return "REDUCE";
                case TaskType::BATCH_REDUCE:
                    return "BATCH_REDUCE";
                case TaskType::RECORD:
                    return "RECORD";
                case TaskType::WAIT:
                    return "WAIT";
                case TaskType::CCU_GRAPH:
                    return "CCU_GRAPH";
                case TaskType::AIV_GRAPH:
                    return "AIV_GRAPH";
                case TaskType::AIV_SET_FLAG:
                    return "AIV_SET_FLAG";
                case TaskType::AIV_WAIT_FLAG:
                    return "AIV_WAIT_FLAG";
                case TaskType::AIV_PIPE_BARRIER:
                    return "AIV_PIPE_BARRIER";
                case TaskType::AIV_SYNC_ALL:
                    return "AIV_SYNC_ALL";
                case TaskType::AIV_SEND_FLAG:
                    return "AIV_SEND_FLAG";
                case TaskType::AIV_RECV_FLAG:
                    return "AIV_RECV_FLAG";
                case TaskType::START:
                    return "START";
                case TaskType::END:
                    return "END";
                default:
                    return "INVALID";
            }
        }

        uint64_t ElapsedMemConflictMs(MemConflictClock::time_point start, MemConflictClock::time_point end)
        {
            const auto elapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            return static_cast<uint64_t>(elapsedNs) / MEM_CONFLICT_NS_PER_MS;
        }

        void LogMemConflictStageTime(const char* stage, const char* status, MemConflictClock::time_point start)
        {
            HCCL_VM_INFO(
                "Stage finished, stage={}, status={}, costMs={}", stage, status,
                ElapsedMemConflictMs(start, MemConflictClock::now()));
        }

        bool IsDataMoveTaskNode(const TaskNode* node)
        {
            return node != nullptr
                   && (node->GetType() == TaskType::TRANS_MEM || node->GetType() == TaskType::BATCH_TRANS_MEM
                       || node->GetType() == TaskType::REDUCE || node->GetType() == TaskType::BATCH_REDUCE);
        }

        std::string DescribeKey(const MemoryKey& key)
        {
            std::ostringstream os;
            os << "rank ";
            if (key.rankId == INVALID_RANK_ID) {
                os << "invalid";
            } else {
                os << key.rankId;
            }
            os << ", " << DescribeMemType(key.memType);
            if (key.memType == MemType::AIV_UB) {
                os << ",(aivBlockIndex=" << key.aivUbIdx << ")";
            }
            return os.str();
        }

        std::string DescribeAccess(const MemoryAccess& access)
        {
            std::ostringstream os;
            os << "    node " << access.nodeId
               << ", action=" << (access.kind == MemoryAccessKind::WRITE ? "write" : "read")
               << "\n    access range : [0x" << std::hex << access.start << ",0x" << access.end << ")" << std::dec;
            if (access.node != nullptr) {
                os << "\n    task         : " << access.node->Describe();
            }
            return os.str();
        }

        std::string MakeTaskTypePairKey(TaskType lhs, TaskType rhs)
        {
            std::string lhsName = TaskTypeName(lhs);
            std::string rhsName = TaskTypeName(rhs);
            if (lhsName > rhsName) {
                std::swap(lhsName, rhsName);
            }
            return lhsName + "<->" + rhsName;
        }

        void LogCountBreakdown(const char* prefix, const std::map<std::string, size_t>& counts)
        {
            (void)prefix;
            (void)counts;
        }

        HcclResult CollectReachableTaskNodes(const TaskNode* start, std::vector<const TaskNode*>& nodes)
        {
            nodes.clear();
            if (start == nullptr) {
                return HCCL_E_PTR;
            }

            // 先用 BFS 收集 start 可达的全部节点，后续所有统计/冲突分析都只在这个闭包里进行。
            std::queue<const TaskNode*> nodeQue;
            std::set<const TaskNode*> visited;
            nodeQue.push(start);
            visited.insert(start);
            while (!nodeQue.empty()) {
                const TaskNode* node = nodeQue.front();
                nodeQue.pop();
                if (node != start) {
                    nodes.push_back(node);
                }

                for (const TaskNode* childNode : node->GetChildren()) {
                    if (childNode == nullptr) {
                        HCCL_VM_ERROR(
                            "{} Graph structure is broken because one child node is null, "
                            "parent={}",
                            MakeErrorCodeText(ErrorCode::MEMCONFLICT_DAG_INVALID), node->Describe());
                        return HCCL_E_PTR;
                    }
                    if (visited.count(childNode) != 0) {
                        continue;
                    }
                    visited.insert(childNode);
                    nodeQue.push(childNode);
                }
            }
            return HCCL_SUCCESS;
        }

        size_t CountDataMoveTaskNodes(const std::vector<const TaskNode*>& nodes)
        {
            size_t count = 0;
            for (const TaskNode* node : nodes) {
                if (IsDataMoveTaskNode(node)) {
                    ++count;
                }
            }
            return count;
        }

        uint64_t MakeAivUbIdx(const TaskPosition& position)
        {
            if (position.launchIdx == std::numeric_limits<uint64_t>::max()
                || position.blockId == std::numeric_limits<uint32_t>::max()) {
                return 0;
            }
            return (position.launchIdx << 32U) | static_cast<uint64_t>(position.blockId);
        }

        uint64_t GetAivUbIdx(const MemSlice& slice, const TaskNode* node)
        {
            if (slice.memType != MemType::AIV_UB || node == nullptr) {
                return 0;
            }
            return MakeAivUbIdx(node->GetPosition());
        }

        HcclResult
        AddAccess(const MemSlice& slice, MemoryAccessKind kind, const TaskNode* node, TaskMemoryAccesses& accesses)
        {
            if (node == nullptr) {
                return HCCL_E_PTR;
            }
            if (node->GetNodeId() < 0) {
                HCCL_VM_ERROR(
                    "{} Data task node id is invalid, nodeId={}, task={}",
                    MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), node->GetNodeId(), node->Describe());
                return HCCL_E_PARA;
            }
            if (slice.len == 0) {
                return HCCL_SUCCESS;
            }
            if (slice.rankId == INVALID_RANK_ID || slice.memType == MemType::INVALID) {
                HCCL_VM_ERROR(
                    "{} One memory slice is missing a valid rank or memory type, "
                    "task={}, rankId={}, memType={}, offset=0x{:x}, length=0x{:x}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(),
                    slice.rankId == INVALID_RANK_ID ? "invalid" : std::to_string(slice.rankId),
                    DescribeMemType(slice.memType), slice.offset, slice.len);
                return HCCL_E_PARA;
            }
            if (slice.len > std::numeric_limits<uint64_t>::max() - slice.offset) {
                HCCL_VM_ERROR(
                    "{} One memory slice is invalid because offset + length overflowed, "
                    "task={}, offset=0x{:x}, length=0x{:x}, endLimit=0x{:x}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), slice.offset, slice.len,
                    std::numeric_limits<uint64_t>::max());
                return HCCL_E_PARA;
            }

            MemoryAccess access;
            access.key.rankId = slice.rankId;
            access.key.memType = slice.memType;
            access.key.aivUbIdx = GetAivUbIdx(slice, node);
            access.start = slice.offset;
            access.end = slice.offset + slice.len;
            access.kind = kind;
            access.nodeId = node->GetNodeId();
            access.node = node;
            accesses.push_back(access);
            return HCCL_SUCCESS;
        }

        HcclResult BuildTaskMemoryAccesses(const TaskNode* node, TaskMemoryAccesses& accesses)
        {
            accesses.clear();
            if (!IsDataMoveTaskNode(node)) {
                return HCCL_SUCCESS;
            }

            if (node->GetType() == TaskType::TRANS_MEM) {
                const auto* transMem = dynamic_cast<const TaskTransMem*>(node);
                if (transMem == nullptr) {
                    return HCCL_E_PTR;
                }
                HcclResult ret = AddAccess(transMem->GetSrc(), MemoryAccessKind::READ, node, accesses);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                return AddAccess(transMem->GetDst(), MemoryAccessKind::WRITE, node, accesses);
            }

            if (node->GetType() == TaskType::REDUCE) {
                const auto* reduce = dynamic_cast<const TaskReduce*>(node);
                if (reduce == nullptr) {
                    return HCCL_E_PTR;
                }
                const std::vector<MemSlice>& srcs = reduce->GetSrcs();
                if (srcs.empty()) {
                    HCCL_VM_ERROR(
                        "{} This reduce task has no source data, task={}",
                        MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe());
                    return HCCL_E_PARA;
                }
                for (size_t index = 0; index < srcs.size(); ++index) {
                    HcclResult ret = AddAccess(srcs[index], MemoryAccessKind::READ, node, accesses);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                }
                return AddAccess(reduce->GetDst(), MemoryAccessKind::WRITE, node, accesses);
            }

            if (node->GetType() == TaskType::BATCH_TRANS_MEM) {
                const auto* transMem = dynamic_cast<const TaskBatchTransMem*>(node);
                if (transMem == nullptr) {
                    return HCCL_E_PTR;
                }
                const auto& srcs = transMem->GetMergedSrcs();
                const auto& dsts = transMem->GetMergedDsts();
                for (const auto& src : srcs) {
                    HcclResult ret = AddAccess(src, MemoryAccessKind::READ, node, accesses);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                }
                for (const auto& dst : dsts) {
                    HcclResult ret = AddAccess(dst, MemoryAccessKind::WRITE, node, accesses);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                }
                return HCCL_SUCCESS;
            }

            if (node->GetType() == TaskType::BATCH_REDUCE) {
                const auto* reduce = dynamic_cast<const TaskBatchReduce*>(node);
                if (reduce == nullptr) {
                    return HCCL_E_PTR;
                }
                const auto& srcGroups = reduce->GetMergedSrcs();
                const auto& dsts = reduce->GetMergedDsts();
                if (srcGroups.empty() || srcGroups.size() != dsts.size()) {
                    HCCL_VM_ERROR(
                        "{} Batch reduce has different counts of source groups and target "
                        "memory slices, task={}, srcGroupCount={}, dstCount={}",
                        MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), srcGroups.size(),
                        dsts.size());
                    return HCCL_E_PARA;
                }
                for (size_t groupIndex = 0; groupIndex < srcGroups.size(); ++groupIndex) {
                    if (srcGroups[groupIndex].empty()) {
                        HCCL_VM_ERROR(
                            "{} One reduce group has no source data at all, task={}, "
                            "groupIndex={}",
                            MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), groupIndex);
                        return HCCL_E_PARA;
                    }
                    for (const auto& src : srcGroups[groupIndex]) {
                        HcclResult ret = AddAccess(src, MemoryAccessKind::READ, node, accesses);
                        if (ret != HCCL_SUCCESS) {
                            return ret;
                        }
                    }
                    HcclResult ret = AddAccess(dsts[groupIndex], MemoryAccessKind::WRITE, node, accesses);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                }
                return HCCL_SUCCESS;
            }

            return HCCL_SUCCESS;
        }

        void DedupTaskMemoryAccesses(TaskMemoryAccesses& accesses)
        {
            if (accesses.size() <= 1) {
                return;
            }

            // 同一task可能抽取出重复区间，先做去重，减少后续扫描线候选数量。
            std::sort(accesses.begin(), accesses.end(), AccessExactLess());
            accesses.erase(
                std::unique(
                    accesses.begin(), accesses.end(),
                    [](const MemoryAccess& lhs, const MemoryAccess& rhs) {
                        return lhs.key.rankId == rhs.key.rankId && lhs.key.memType == rhs.key.memType
                               && lhs.key.aivUbIdx == rhs.key.aivUbIdx && lhs.start == rhs.start && lhs.end == rhs.end
                               && lhs.kind == rhs.kind && lhs.nodeId == rhs.nodeId;
                    }),
                accesses.end());
        }

        bool HasWriteAccess(const MemoryAccess& lhs, const MemoryAccess& rhs)
        {
            return lhs.kind == MemoryAccessKind::WRITE || rhs.kind == MemoryAccessKind::WRITE;
        }

        void RefreshOverlapCandidateStats(const OverlapCandidateStatsCollector& collector, MemConflictCheckStats& stats)
        {
            stats.overlapCandidateTaskNodeCount = collector.nodeIds.size();
        }

        void RecordOverlapCandidateTaskNodes(NodeId lhs, NodeId rhs, OverlapCandidateStatsCollector& collector)
        {
            collector.nodeIds.insert(lhs);
            collector.nodeIds.insert(rhs);
        }

        HcclResult CanRunInParallel(const ReachabilityClosure& closure, NodeId lhs, NodeId rhs, bool& canRunInParallel)
        {
            canRunInParallel = false;
            if (lhs == rhs) {
                return HCCL_SUCCESS;
            }

            bool lhsReachRhs = false;
            HcclResult ret = IsReachable(closure, lhs, rhs, lhsReachRhs);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }

            bool rhsReachLhs = false;
            ret = IsReachable(closure, rhs, lhs, rhsReachLhs);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }

            canRunInParallel = !lhsReachRhs && !rhsReachLhs;
            return HCCL_SUCCESS;
        }

        HcclResult CollectConflictCandidate(
            const MemoryAccess& current, const MemoryAccess& history, bool& needReachabilityClosure,
            ConflictCandidates& candidates, MemConflictCheckStats& stats, OverlapCandidateStatsCollector& collector)
        {
            // 同一节点内部重叠不在这里判冲突；纯读读重叠也不会形成冲突。
            if (current.nodeId == history.nodeId || !HasWriteAccess(current, history)) {
                return HCCL_SUCCESS;
            }

            ++stats.overlapCandidatePairCount;
            RecordOverlapCandidateTaskNodes(current.nodeId, history.nodeId, collector);
            collector.pairCountByBucket[DescribeKey(current.key)]++;
            if (current.node != nullptr && history.node != nullptr) {
                collector
                    .pairCountByTaskTypePair[MakeTaskTypePairKey(current.node->GetType(), history.node->GetType())]++;
            }
            needReachabilityClosure = true;
            candidates.push_back(ConflictCandidate{&current, &history});
            return HCCL_SUCCESS;
        }

        void RemoveInactiveAccesses(ActiveAccesses& activeAccesses, uint64_t sweepStart)
        {
            while (!activeAccesses.empty()) {
                const MemoryAccess* access = *activeAccesses.begin();
                // 扫描线推进到 sweepStart 后，所有已经完全结束的区间都可以移出活动集合。
                if (access->end > sweepStart) {
                    return;
                }
                activeAccesses.erase(activeAccesses.begin());
            }
        }

        HcclResult CheckActiveAccesses(
            const MemoryAccess& current, const ActiveAccesses& activeAccesses, bool& needReachabilityClosure,
            ConflictCandidates& candidates, MemConflictCheckStats& stats, OverlapCandidateStatsCollector& collector)
        {
            for (const MemoryAccess* activeAccess : activeAccesses) {
                if (activeAccess == nullptr) {
                    return HCCL_E_PTR;
                }
                HcclResult ret = CollectConflictCandidate(
                    current, *activeAccess, needReachabilityClosure, candidates, stats, collector);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            return HCCL_SUCCESS;
        }

        HcclResult ScanMemoryBucket(
            TaskMemoryAccesses& bucketAccesses, bool& needReachabilityClosure, ConflictCandidates& candidates,
            MemConflictCheckStats& stats, OverlapCandidateStatsCollector& collector)
        {
            // 同一个 bucket 内 rankId/memType 相同，只需要按区间起点做一次扫描线检查。
            std::sort(bucketAccesses.begin(), bucketAccesses.end(), AccessStartLess());

            ActiveAccesses activeReads;
            ActiveAccesses activeWrites;
            for (const MemoryAccess& current : bucketAccesses) {
                RemoveInactiveAccesses(activeReads, current.start);
                RemoveInactiveAccesses(activeWrites, current.start);

                if (current.kind == MemoryAccessKind::READ) {
                    // 读只需要和当前仍活跃的写比较。
                    HcclResult ret = CheckActiveAccesses(
                        current, activeWrites, needReachabilityClosure, candidates, stats, collector);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                    activeReads.insert(&current);
                    continue;
                }

                // 写需要同时检查读/写两个集合，因为读写、写写都可能冲突。
                HcclResult ret
                    = CheckActiveAccesses(current, activeReads, needReachabilityClosure, candidates, stats, collector);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ret = CheckActiveAccesses(current, activeWrites, needReachabilityClosure, candidates, stats, collector);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                activeWrites.insert(&current);
            }
            return HCCL_SUCCESS;
        }

        HcclResult ScanMemoryBuckets(
            MemoryAccessBuckets& accessBuckets, bool& needReachabilityClosure, ConflictCandidates& candidates,
            MemConflictCheckStats& stats, OverlapCandidateStatsCollector& collector)
        {
            for (auto& bucketItem : accessBuckets) {
                HcclResult ret
                    = ScanMemoryBucket(bucketItem.second, needReachabilityClosure, candidates, stats, collector);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            return HCCL_SUCCESS;
        }

        template <typename ReachabilityClosureT>
        HcclResult CheckCollectedConflictCandidates(
            const ConflictCandidates& candidates, const ReachabilityClosureT& closure, MemConflictCheckStats& stats)
        {
            for (const ConflictCandidate& candidate : candidates) {
                if (candidate.current == nullptr || candidate.history == nullptr) {
                    return HCCL_E_PTR;
                }

                bool canRunInParallel = false;
                HcclResult ret
                    = CanRunInParallel(closure, candidate.current->nodeId, candidate.history->nodeId, canRunInParallel);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                // 若两个任务在图上已有先后约束，则即使地址重叠也不是并发冲突。
                if (!canRunInParallel) {
                    ++stats.orderedCandidatePairCount;
                    continue;
                }

                ++stats.parallelCandidatePairCount;
                const uint64_t overlapStart = std::max(candidate.current->start, candidate.history->start);
                const uint64_t overlapEnd = std::min(candidate.current->end, candidate.history->end);
                HCCL_VM_ERROR(
                    "{} Two tasks may access the same memory range in parallel, and at least one "
                    "access is a write.\n  Conflict memory : {}\n  Overlap range    : [0x{:x},0x{:x})\n"
                    "  Conflict task 1:\n{}\n  Conflict task 2:\n{}",
                    MakeErrorCodeText(ErrorCode::MEMCONFLICT_DETECTED), DescribeKey(candidate.current->key),
                    overlapStart, overlapEnd, DescribeAccess(*candidate.current), DescribeAccess(*candidate.history));
                return HCCL_E_MEMORY;
            }
            return HCCL_SUCCESS;
        }

        HcclResult CheckCandidatesWithDenseReachability(
            const TaskNode* start, const ConflictCandidates& candidates, MemConflictCheckStats& stats)
        {
            const auto closureStart = MemConflictClock::now();
            ReachabilityClosure closure;
            // 只有扫描线阶段确实发现候选重叠时，才构建可达闭包做精确判定。
            HcclResult ret = GenReachabilityClosure(start, closure);
            LogMemConflictStageTime(
                "GenReachabilityClosure(Dense)", ret == HCCL_SUCCESS ? "success" : "failed", closureStart);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }

            const auto checkStart = MemConflictClock::now();
            ret = CheckCollectedConflictCandidates(candidates, closure, stats);
            LogMemConflictStageTime(
                "CheckCollectedConflictCandidates(Dense)", ret == HCCL_SUCCESS ? "success" : "failed", checkStart);
            return ret;
        }

        HcclResult CollectMemoryAccessBuckets(
            const std::vector<const TaskNode*>& nodes, MemoryAccessBuckets& accessBuckets, MemConflictCheckStats& stats)
        {
            accessBuckets.clear();
            std::map<std::string, size_t> accessCountByTaskType;
            std::map<std::string, size_t> accessCountByTaskTypeAndKind;
            std::map<std::string, size_t> accessCountByMemType;
            std::map<std::string, size_t> accessCountByMemTypeAndKind;
            std::map<std::string, size_t> accessCountByBucket;
            for (const TaskNode* node : nodes) {
                if (!IsDataMoveTaskNode(node)) {
                    continue;
                }

                TaskMemoryAccesses currentAccesses;
                HcclResult ret = BuildTaskMemoryAccesses(node, currentAccesses);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                DedupTaskMemoryAccesses(currentAccesses);

                for (const MemoryAccess& access : currentAccesses) {
                    accessBuckets[access.key].push_back(access);
                    ++stats.accessIntervalCount;
                    accessCountByTaskType[TaskTypeName(node->GetType())]++;
                    accessCountByTaskTypeAndKind
                        [std::string(TaskTypeName(node->GetType())) + ":" + AccessKindName(access.kind)]++;
                    accessCountByMemType[DescribeMemType(access.key.memType)]++;
                    accessCountByMemTypeAndKind
                        [std::string(DescribeMemType(access.key.memType)) + ":" + AccessKindName(access.kind)]++;
                    accessCountByBucket[DescribeKey(access.key)]++;
                }
                ++stats.processedDataTaskNodeCount;
            }

            stats.memoryBucketCount = accessBuckets.size();
            LogCountBreakdown("AccessCountByTaskType", accessCountByTaskType);
            LogCountBreakdown("AccessCountByTaskTypeAndKind", accessCountByTaskTypeAndKind);
            LogCountBreakdown("AccessCountByMemType", accessCountByMemType);
            LogCountBreakdown("AccessCountByMemTypeAndKind", accessCountByMemTypeAndKind);
            LogCountBreakdown("AccessCountByBucket", accessCountByBucket);
            return HCCL_SUCCESS;
        }
    } // namespace

    HcclResult CheckDataMoveTaskMemConflict(const TaskNode* start, MemConflictCheckStats* stats)
    {
        const auto scanLineStart = MemConflictClock::now();
        std::vector<const TaskNode*> nodes;
        // 整体流程：
        // 1. BFS 收集可达节点；
        // 2. 为数据搬运类节点抽取读写区间并按内存桶归类；
        // 3. 在每个桶内扫描重叠候选；
        // 4. 只有真正存在前后无序关系时才计算可达闭包做精确判定。
        HcclResult ret = CollectReachableTaskNodes(start, nodes);
        if (ret != HCCL_SUCCESS) {
            LogMemConflictStageTime("CollectReachableTaskNodesToScanMemoryBuckets", "failed", scanLineStart);
            return ret;
        }

        MemConflictCheckStats localStats;
        localStats.nodeCount = nodes.size() + 1U;
        localStats.dataTaskNodeCount = CountDataMoveTaskNodes(nodes);

        MemoryAccessBuckets accessBuckets;
        ret = CollectMemoryAccessBuckets(nodes, accessBuckets, localStats);
        if (ret != HCCL_SUCCESS) {
            LogMemConflictStageTime("CollectReachableTaskNodesToScanMemoryBuckets", "failed", scanLineStart);
            return ret;
        }

        bool needReachabilityClosure = false;
        ConflictCandidates candidates;
        OverlapCandidateStatsCollector overlapCandidateStats;
        ret = ScanMemoryBuckets(accessBuckets, needReachabilityClosure, candidates, localStats, overlapCandidateStats);
        RefreshOverlapCandidateStats(overlapCandidateStats, localStats);
        LogCountBreakdown("OverlapCandidatePairsByBucket", overlapCandidateStats.pairCountByBucket);
        LogCountBreakdown("OverlapCandidatePairsByTaskTypePair", overlapCandidateStats.pairCountByTaskTypePair);
        LogMemConflictStageTime(
            "CollectReachableTaskNodesToScanMemoryBuckets", ret == HCCL_SUCCESS ? "success" : "failed", scanLineStart);
        if (ret != HCCL_SUCCESS) {
            if (stats != nullptr) {
                *stats = localStats;
            }
            return ret;
        }

        if (needReachabilityClosure) {
            ret = CheckCandidatesWithDenseReachability(start, candidates, localStats);
            // ret = CheckCandidatesWithCRoaringReachability(start, candidates, localStats);
            if (ret != HCCL_SUCCESS) {
                if (stats != nullptr) {
                    *stats = localStats;
                }
                return ret;
            }
        } else {
            LogMemConflictStageTime("GenReachabilityClosure", "skipped", MemConflictClock::now());
        }

        if (stats != nullptr) {
            *stats = localStats;
        }
        return HCCL_SUCCESS;
    }
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
