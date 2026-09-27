/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_graph_single_task_check_v3.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "sim_log.h"
#include "storage_manager.h"
#include "utils/check_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {
        constexpr uint64_t AIV_COMM_INFO_SYNC_CELL_BYTES = 128ULL;

        const TaskNode* GetQueueNode(const std::vector<std::unique_ptr<TaskNode>>& nodes, NodeId nodeId)
        {
            if (nodeId < 0 || static_cast<size_t>(nodeId) >= nodes.size()) {
                return nullptr;
            }
            return nodes[static_cast<size_t>(nodeId)].get();
        }

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

        std::string DescribePosition(const TaskPosition& loc)
        {
            std::ostringstream os;
            os << "rankId=";
            if (loc.rankId == INVALID_RANK_ID) {
                os << "invalid";
            } else {
                os << loc.rankId;
            }
            os << ", streamId=";
            if (loc.streamId == INVALID_STREAM_ID) {
                os << "invalid";
            } else {
                os << loc.streamId;
            }
            return os.str();
        }

        std::string DescribeMemSlice(const MemSlice& slice)
        {
            std::ostringstream os;
            os << "{rankId=";
            if (slice.rankId == INVALID_RANK_ID) {
                os << "invalid";
            } else {
                os << slice.rankId;
            }
            os << ", memType=" << DescribeMemType(slice.memType) << ", offset=0x" << std::hex << slice.offset
               << ", length=0x" << slice.len << std::dec << "}";
            return os.str();
        }

        bool IsLocalAicpuRecord(const TaskNode* node)
        {
            if (node == nullptr || node->GetType() != TaskType::RECORD) {
                return false;
            }
            const auto* record = dynamic_cast<const TaskRecordAICPU*>(node);
            if (record == nullptr) {
                return false;
            }
            const AicpuNotify& notify = record->GetNotify();
            return notify.recordRankId == notify.waitRankId;
        }

        bool IsLocalAicpuWait(const TaskNode* node)
        {
            if (node == nullptr || node->GetType() != TaskType::WAIT) {
                return false;
            }
            const auto* wait = dynamic_cast<const TaskWaitAICPU*>(node);
            if (wait == nullptr) {
                return false;
            }
            const AicpuNotify& notify = wait->GetNotify();
            return notify.recordRankId == notify.waitRankId;
        }

        bool IsEmptyLocalCopy(const TaskNode* node)
        {
            if (node == nullptr || node->GetType() != TaskType::TRANS_MEM) {
                return false;
            }
            const auto* transMem = dynamic_cast<const TaskTransMem*>(node);
            if (transMem == nullptr) {
                return false;
            }
            const MemSlice& src = transMem->GetSrc();
            const MemSlice& dst = transMem->GetDst();
            return src.rankId == dst.rankId && src.len == 0 && dst.len == 0;
        }

        bool IsDataMoveTaskNode(const TaskNode* node)
        {
            return node != nullptr
                   && (node->GetType() == TaskType::TRANS_MEM || node->GetType() == TaskType::BATCH_TRANS_MEM
                       || node->GetType() == TaskType::REDUCE || node->GetType() == TaskType::BATCH_REDUCE);
        }

        bool StreamHasAivGraph(const std::vector<std::unique_ptr<TaskNode>>& nodes, const std::vector<NodeId>& stream)
        {
            for (const NodeId nodeId : stream) {
                const TaskNode* node = GetQueueNode(nodes, nodeId);
                if (node != nullptr && node->GetType() == TaskType::AIV_GRAPH) {
                    return true;
                }
            }
            return false;
        }

        BufferType ConvertMemTypeToBufferType(MemType memType)
        {
            switch (memType) {
                case MemType::INPUT:
                    return BufferType::INPUT;
                case MemType::OUTPUT:
                    return BufferType::OUTPUT;
                case MemType::CCL:
                    return BufferType::CCL;
                case MemType::MS_CCU:
                    return BufferType::MS;
                default:
                    return BufferType::RESERVED;
            }
        }

        bool IsSameMemoryType(const MemSlice& lhs, const MemSlice& rhs) { return lhs.memType == rhs.memType; }

        uint64_t GetAivSliceBoundSize(MemType memType)
        {
            if (memType == MemType::AIV_UB) {
                return g_checkerAivUbBufferSize;
            }
            if (memType == MemType::AIV_COMM) {
                return g_checkerAivCommInfoSize;
            }
            return 0;
        }

        HcclResult CheckSingleSlice(const TaskNode* node, const MemSlice& slice);
        HcclResult CheckTwoSliceOverlap(const TaskNode* node, const MemSlice& lhs, const MemSlice& rhs);

        struct MemoryKey {
            RankId rankId{INVALID_RANK_ID};
            MemType memType{MemType::INVALID};

            bool operator==(const MemoryKey& rhs) const { return rankId == rhs.rankId && memType == rhs.memType; }

            bool operator<(const MemoryKey& rhs) const
            {
                if (rankId != rhs.rankId) {
                    return rankId < rhs.rankId;
                }
                return static_cast<uint32_t>(memType) < static_cast<uint32_t>(rhs.memType);
            }
        };

        struct SliceCoverageSummary {
            std::map<MemoryKey, uint64_t> coveredSizeByKey;
            uint64_t totalCoveredSize{0};
        };

        // 将一组 slice 归并成按 {rank, memType} 统计的覆盖大小。
        // 普通内存按区间并集计算；MS 由于是复用内存，按出现次数累计。
        HcclResult AccumulateCoveredSizeByKey(
            const TaskNode* node, const std::vector<MemSlice>& slices, SliceCoverageSummary& summary)
        {
            if (node == nullptr) {
                return HCCL_E_PTR;
            }

            std::map<MemoryKey, std::vector<std::pair<uint64_t, uint64_t>>> rangesByKey;
            std::map<MemoryKey, uint64_t> repeatedSizeByKey;
            for (const auto& slice : slices) {
                if (slice.len == 0) {
                    continue;
                }
                if (slice.len > std::numeric_limits<uint64_t>::max() - slice.offset) {
                    HCCL_VM_ERROR(
                        "{} One memory slice is invalid because its end address overflows "
                        "while total coverage is being calculated, task={}, memorySlice={}",
                        MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(),
                        DescribeMemSlice(slice));
                    return HCCL_E_PARA;
                }
                const MemoryKey key{slice.rankId, slice.memType};
                if (slice.memType == MemType::MS_CCU) {
                    repeatedSizeByKey[key] += slice.len;
                    continue;
                }
                rangesByKey[key].push_back({slice.offset, slice.offset + slice.len});
            }

            for (auto& entry : rangesByKey) {
                auto& ranges = entry.second;
                std::sort(
                    ranges.begin(), ranges.end(),
                    [](const std::pair<uint64_t, uint64_t>& lhs, const std::pair<uint64_t, uint64_t>& rhs) {
                        if (lhs.first != rhs.first) {
                            return lhs.first < rhs.first;
                        }
                        return lhs.second < rhs.second;
                    });
                uint64_t coveredSize = 0;
                uint64_t mergedStart = ranges.front().first;
                uint64_t mergedEnd = ranges.front().second;
                for (size_t index = 1; index < ranges.size(); ++index) {
                    const uint64_t curStart = ranges[index].first;
                    const uint64_t curEnd = ranges[index].second;
                    if (curStart <= mergedEnd) {
                        mergedEnd = std::max(mergedEnd, curEnd);
                        continue;
                    }
                    coveredSize += mergedEnd - mergedStart;
                    mergedStart = curStart;
                    mergedEnd = curEnd;
                }
                coveredSize += mergedEnd - mergedStart;
                summary.coveredSizeByKey[entry.first] += coveredSize;
                summary.totalCoveredSize += coveredSize;
            }

            for (const auto& entry : repeatedSizeByKey) {
                summary.coveredSizeByKey[entry.first] += entry.second;
                summary.totalCoveredSize += entry.second;
            }

            return HCCL_SUCCESS;
        }

        HcclResult
        BuildCoverageSummary(const TaskNode* node, const std::vector<MemSlice>& slices, SliceCoverageSummary& summary)
        {
            summary = SliceCoverageSummary{};
            return AccumulateCoveredSizeByKey(node, slices, summary);
        }

        HcclResult BuildCoverageSummary(
            const TaskNode* node, const std::vector<std::vector<MemSlice>>& sliceGroups, SliceCoverageSummary& summary)
        {
            summary = SliceCoverageSummary{};
            for (const auto& group : sliceGroups) {
                HcclResult ret = AccumulateCoveredSizeByKey(node, group, summary);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            return HCCL_SUCCESS;
        }

        HcclResult
        CheckCoverageMultiple(const TaskNode* node, uint64_t srcSize, uint64_t dstSize, const std::string& group)
        {
            if (node == nullptr) {
                return HCCL_E_PTR;
            }
            if (dstSize == 0) {
                if (srcSize == 0) {
                    return HCCL_SUCCESS;
                }
                HCCL_VM_ERROR(
                    "{} Source data exists but the target range is empty, task={}, "
                    "srcDataSize={}, dstDataSize={}, group={}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), srcSize, dstSize, group);
                return HCCL_E_INTERNAL;
            }
            if (srcSize % dstSize == 0) {
                return HCCL_SUCCESS;
            }
            HCCL_VM_ERROR(
                "{} Source data size is not an integer multiple of target data size, "
                "task={}, srcDataSize={}, dstDataSize={}, group={}",
                MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), srcSize, dstSize, group);
            return HCCL_E_INTERNAL;
        }

        HcclResult CheckSliceLenEqual(
            const TaskNode* node, const MemSlice& src, const MemSlice& dst, const std::string& label, size_t index)
        {
            if (node == nullptr) {
                return HCCL_E_PTR;
            }
            if (src.len == dst.len) {
                return HCCL_SUCCESS;
            }
            HCCL_VM_ERROR(
                "{} In one batch transfer pair, source length and target length are "
                "different, task={}, group={}, pairIndex={}, sourceMemorySlice={}, targetMemorySlice={}",
                MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), label, index,
                DescribeMemSlice(src), DescribeMemSlice(dst));
            return HCCL_E_INTERNAL;
        }

        bool MemSliceLess(const MemSlice& lhs, const MemSlice& rhs)
        {
            if (lhs.rankId != rhs.rankId) {
                return lhs.rankId < rhs.rankId;
            }
            if (lhs.memType != rhs.memType) {
                return static_cast<uint32_t>(lhs.memType) < static_cast<uint32_t>(rhs.memType);
            }
            if (lhs.offset != rhs.offset) {
                return lhs.offset < rhs.offset;
            }
            return lhs.len < rhs.len;
        }

        // merged slices 只用于更快地做 batch 节点内部重叠检查，因此这里先按完整四元组去重。
        void DedupMemSlices(std::vector<MemSlice>& slices)
        {
            if (slices.size() <= 1) {
                return;
            }
            std::sort(slices.begin(), slices.end(), MemSliceLess);
            slices.erase(
                std::unique(
                    slices.begin(), slices.end(),
                    [](const MemSlice& lhs, const MemSlice& rhs) {
                        return lhs.rankId == rhs.rankId && lhs.memType == rhs.memType && lhs.offset == rhs.offset
                               && lhs.len == rhs.len;
                    }),
                slices.end());
        }

        HcclResult CheckBatchTransSlices(
            const TaskNode* node, const std::vector<MemSlice>& srcs, const std::vector<MemSlice>& dsts,
            const std::vector<MemSlice>& mergedSrcs, const std::vector<MemSlice>& mergedDsts, const std::string& label)
        {
            if (node == nullptr) {
                return HCCL_E_PTR;
            }
            // 原始 src/dst 负责逐对检查 memcpy 语义是否满足“长度一一对应”；
            // merged src/dst 只用于更快地检查 batch 节点内部是否存在自冲突。
            if (srcs.size() != dsts.size()) {
                HCCL_VM_ERROR(
                    "{} Batch transfer has different counts of source memory slices and "
                    "target memory slices, task={}, group={}, sourceMemorySliceCount={}, targetMemorySliceCount={}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), label, srcs.size(),
                    dsts.size());
                return HCCL_E_PARA;
            }
            for (size_t index = 0; index < srcs.size(); ++index) {
                HcclResult ret = CheckSingleSlice(node, srcs[index]);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ret = CheckSingleSlice(node, dsts[index]);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ret = CheckSliceLenEqual(node, srcs[index], dsts[index], label, index);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }

            std::vector<MemSlice> dedupMergedSrcs = mergedSrcs;
            std::vector<MemSlice> dedupMergedDsts = mergedDsts;
            DedupMemSlices(dedupMergedSrcs);
            DedupMemSlices(dedupMergedDsts);

            for (const auto& src : dedupMergedSrcs) {
                HcclResult ret = CheckSingleSlice(node, src);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            for (const auto& dst : dedupMergedDsts) {
                HcclResult ret = CheckSingleSlice(node, dst);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            for (const auto& src : dedupMergedSrcs) {
                for (const auto& dst : dedupMergedDsts) {
                    HcclResult ret = CheckTwoSliceOverlap(node, src, dst);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                }
            }
            HCCL_VM_DEBUG(
                "Batch transfer merged-slice check finished, "
                "taskId={}, group={}, originalSourceMemorySliceCount={}, originalTargetMemorySliceCount={}, "
                "mergedSourceMemorySliceCount={}, mergedTargetMemorySliceCount={}, "
                "uniqueMergedSourceMemorySliceCount={}, "
                "uniqueMergedTargetMemorySliceCount={}",
                node->GetNodeId(), label, srcs.size(), dsts.size(), mergedSrcs.size(), mergedDsts.size(),
                dedupMergedSrcs.size(), dedupMergedDsts.size());
            return HCCL_SUCCESS;
        }

        HcclResult CheckBatchReduceSlices(
            const TaskNode* node, const std::vector<std::vector<MemSlice>>& srcGroups,
            const std::vector<MemSlice>& dsts, const std::string& label, SliceCoverageSummary& srcSummary,
            SliceCoverageSummary& dstSummary)
        {
            if (node == nullptr) {
                return HCCL_E_PTR;
            }
            // batch reduce 仍然按原始 group 校验：每组 src 必须能落到同一个 dst，
            // 且所有 src 总覆盖量应当是 dst 总覆盖量的整倍数。
            if (srcGroups.empty() || srcGroups.size() != dsts.size()) {
                HCCL_VM_ERROR(
                    "{} Batch reduce has different counts of source groups and target "
                    "memory slices, task={}, group={}, sourceGroupCount={}, targetMemorySliceCount={}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), label, srcGroups.size(),
                    dsts.size());
                return HCCL_E_PARA;
            }
            for (size_t groupIndex = 0; groupIndex < srcGroups.size(); ++groupIndex) {
                HcclResult ret = CheckSingleSlice(node, dsts[groupIndex]);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                if (srcGroups[groupIndex].empty()) {
                    HCCL_VM_ERROR(
                        "{} One reduce group has no source data at all, task={}, group={}, "
                        "groupIndex={}",
                        MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), label, groupIndex);
                    return HCCL_E_PARA;
                }
                for (const auto& src : srcGroups[groupIndex]) {
                    ret = CheckSingleSlice(node, src);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                    ret = CheckTwoSliceOverlap(node, src, dsts[groupIndex]);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                }
            }

            HcclResult ret = BuildCoverageSummary(node, srcGroups, srcSummary);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            ret = BuildCoverageSummary(node, dsts, dstSummary);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            return CheckCoverageMultiple(node, srcSummary.totalCoveredSize, dstSummary.totalCoveredSize, label);
        }

        HcclResult CheckSingleSlice(const TaskNode* node, const MemSlice& slice)
        {
            if (node == nullptr) {
                return HCCL_E_PTR;
            }
            if (slice.memType == MemType::MS_CCU) {
                return HCCL_SUCCESS;
            }
            if (slice.rankId == INVALID_RANK_ID || slice.memType == MemType::INVALID) {
                HCCL_VM_ERROR(
                    "{} One memory slice is missing a valid rank or memory type, task={}, "
                    "memorySlice={}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), DescribeMemSlice(slice));
                return HCCL_E_PARA;
            }
            if (slice.len > std::numeric_limits<uint64_t>::max() - slice.offset) {
                HCCL_VM_ERROR(
                    "{} One memory slice is invalid because offset + length exceeds the "
                    "numeric limit, task={}, memorySlice={}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), DescribeMemSlice(slice));
                return HCCL_E_PARA;
            }

            if (slice.memType == MemType::AIV_UB || slice.memType == MemType::AIV_COMM) {
                const uint64_t boundSize = GetAivSliceBoundSize(slice.memType);
                if (boundSize != 0 && slice.offset + slice.len > boundSize) {
                    HCCL_VM_ERROR(
                        "{} One AIV memory slice goes past the valid AIV buffer boundary, "
                        "task={}, memorySlice={}, bufferLimit={}",
                        MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(),
                        DescribeMemSlice(slice), boundSize);
                    return HCCL_E_INTERNAL;
                }
                return HCCL_SUCCESS;
            }

            const BufferType storageType = ConvertMemTypeToBufferType(slice.memType);
            if (storageType == BufferType::RESERVED) {
                HCCL_VM_ERROR(
                    "{} This task uses a memory type that the checker does not support, "
                    "task={}, memorySlice={}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), DescribeMemSlice(slice));
                return HCCL_E_NOT_SUPPORT;
            }

            const uint64_t blockSize = StorageManager::GetInstance().GetBlockSize(slice.rankId, storageType);
            if (slice.offset + slice.len > blockSize) {
                HCCL_VM_ERROR(
                    "{} One memory slice goes past the end of its buffer, task={}, "
                    "memorySlice={}, bufferType={}, bufferSize={}, position={}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), DescribeMemSlice(slice),
                    BufferTypeToString(storageType), blockSize, DescribePosition(node->GetPosition()));
                return HCCL_E_INTERNAL;
            }
            return HCCL_SUCCESS;
        }

        HcclResult CheckTwoSliceOverlap(const TaskNode* node, const MemSlice& lhs, const MemSlice& rhs)
        {
            if (node == nullptr) {
                return HCCL_E_PTR;
            }
            if (lhs.rankId != rhs.rankId || !IsSameMemoryType(lhs, rhs) || lhs.len == 0 || rhs.len == 0) {
                return HCCL_SUCCESS;
            }
            if (lhs.len > std::numeric_limits<uint64_t>::max() - lhs.offset
                || rhs.len > std::numeric_limits<uint64_t>::max() - rhs.offset) {
                HCCL_VM_ERROR(
                    "{} Two memory slices cannot be compared because one slice already "
                    "overflowed, task={}, memorySlice1={}, memorySlice2={}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), DescribeMemSlice(lhs),
                    DescribeMemSlice(rhs));
                return HCCL_E_PARA;
            }

            if (node->HasCcuTrace()) {
                // CCU mode下，允许src == dst
                if (lhs.rankId == rhs.rankId && IsSameMemoryType(lhs, rhs) && lhs.offset == rhs.offset
                    && lhs.len == rhs.len) {
                    HCCL_VM_WARN(
                        "Source and destination use the same memory slice, which may hurt "
                        "performance, position={}, task={}, memorySlice1={}, memorySlice2={}",
                        DescribePosition(node->GetPosition()), node->Describe(), DescribeMemSlice(lhs),
                        DescribeMemSlice(rhs));
                    return HCCL_SUCCESS;
                }
            }

            const bool conflictCase1 = lhs.offset >= rhs.offset && lhs.offset < (rhs.offset + rhs.len);
            const bool conflictCase2 = rhs.offset >= lhs.offset && rhs.offset < (lhs.offset + lhs.len);
            if (conflictCase1 || conflictCase2) {
                HCCL_VM_ERROR(
                    "{} Two memory slices overlap inside the same task, task={}, "
                    "memorySlice1={}, memorySlice2={}, position={}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_CONFLICT), node->Describe(), DescribeMemSlice(lhs),
                    DescribeMemSlice(rhs), DescribePosition(node->GetPosition()));
                return HCCL_E_INTERNAL;
            }
            return HCCL_SUCCESS;
        }

        HcclResult CheckAivCommInfoSyncNodeMem(const TaskNode* node)
        {
            if (node == nullptr) {
                return HCCL_E_PTR;
            }
            const AivFlagSync* flag = nullptr;
            if (node->GetType() == TaskType::AIV_SEND_FLAG) {
                const auto* sendFlag = dynamic_cast<const TaskAivSendFlag*>(node);
                if (sendFlag == nullptr) {
                    return HCCL_E_PTR;
                }
                flag = &sendFlag->GetFlag();
            } else if (node->GetType() == TaskType::AIV_RECV_FLAG) {
                const auto* recvFlag = dynamic_cast<const TaskAivRecvFlag*>(node);
                if (recvFlag == nullptr) {
                    return HCCL_E_PTR;
                }
                flag = &recvFlag->GetFlag();
            }
            if (flag == nullptr) {
                return HCCL_SUCCESS;
            }
            if (flag->commInfoOffset % AIV_COMM_INFO_SYNC_CELL_BYTES != 0) {
                HCCL_VM_ERROR(
                    "{} AIV commInfo synchronization address is not cell aligned, "
                    "task={}, commInfoOffset={}, cellSize={}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), flag->commInfoOffset,
                    AIV_COMM_INFO_SYNC_CELL_BYTES);
                return HCCL_E_PARA;
            }
            const uint64_t aivCommInfoSize = GetAivSliceBoundSize(MemType::AIV_COMM);
            if (aivCommInfoSize != 0
                && (aivCommInfoSize < sizeof(int32_t) || flag->commInfoOffset > aivCommInfoSize - sizeof(int32_t))) {
                HCCL_VM_ERROR(
                    "{} AIV commInfo synchronization address goes past the valid buffer boundary, task={}, "
                    "commInfoOffset={}, aivCommInfoSize={}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe(), flag->commInfoOffset,
                    aivCommInfoSize);
                return HCCL_E_INTERNAL;
            }
            return HCCL_SUCCESS;
        }

        HcclResult CheckDataMoveTaskMem(const TaskNode* node)
        {
            if (node == nullptr) {
                return HCCL_E_PTR;
            }
            if (node->GetType() == TaskType::TRANS_MEM) {
                const auto* transMem = dynamic_cast<const TaskTransMem*>(node);
                if (transMem == nullptr) {
                    return HCCL_E_PTR;
                }
                HcclResult ret = CheckSingleSlice(node, transMem->GetSrc());
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ret = CheckSingleSlice(node, transMem->GetDst());
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                return CheckTwoSliceOverlap(node, transMem->GetSrc(), transMem->GetDst());
            }

            if (node->GetType() == TaskType::REDUCE) {
                const auto* reduce = dynamic_cast<const TaskReduce*>(node);
                if (reduce == nullptr) {
                    return HCCL_E_PTR;
                }
                HcclResult ret = CheckSingleSlice(node, reduce->GetDst());
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                const std::vector<MemSlice>& srcs = reduce->GetSrcs();
                if (srcs.empty()) {
                    HCCL_VM_ERROR(
                        "{} This reduce task has no source data at all, task={}",
                        MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), node->Describe());
                    return HCCL_E_PARA;
                }
                for (const auto& src : srcs) {
                    ret = CheckSingleSlice(node, src);
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                    ret = CheckTwoSliceOverlap(node, src, reduce->GetDst());
                    if (ret != HCCL_SUCCESS) {
                        return ret;
                    }
                }
                return HCCL_SUCCESS;
            }

            if (node->GetType() == TaskType::BATCH_TRANS_MEM) {
                const auto* transMem = dynamic_cast<const TaskBatchTransMem*>(node);
                if (transMem == nullptr) {
                    return HCCL_E_PTR;
                }
                HcclResult ret = CheckBatchTransSlices(
                    node, transMem->GetSrcs(), transMem->GetDsts(), transMem->GetMergedSrcs(),
                    transMem->GetMergedDsts(), "original");
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                return HCCL_SUCCESS;
            }

            if (node->GetType() == TaskType::BATCH_REDUCE) {
                const auto* reduce = dynamic_cast<const TaskBatchReduce*>(node);
                if (reduce == nullptr) {
                    return HCCL_E_PTR;
                }
                SliceCoverageSummary originalSrcSummary;
                SliceCoverageSummary originalDstSummary;
                HcclResult ret = CheckBatchReduceSlices(
                    node, reduce->GetSrcs(), reduce->GetDsts(), "original", originalSrcSummary, originalDstSummary);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                return HCCL_SUCCESS;
            }

            HcclResult ret = HCCL_SUCCESS;
            if (node->GetType() == TaskType::AIV_SEND_FLAG || node->GetType() == TaskType::AIV_RECV_FLAG) {
                ret = CheckAivCommInfoSyncNodeMem(node);
            }
            return ret;
        }
    } // namespace

    HcclResult
    CheckSlaveTaskQueue(const std::vector<std::unique_ptr<TaskNode>>& nodes, const AllRankNodeQueues& allRankTaskQueues)
    {
        for (const auto& rankEntry : allRankTaskQueues) {
            const RankId rankId = rankEntry.first;
            const RankNodeQueues& taskQueue = rankEntry.second;
            const size_t queueNum = taskQueue.size();
            for (size_t streamId = 1; streamId < queueNum; ++streamId) {
                const auto& stream = taskQueue[streamId];
                const size_t taskSize = stream.size();
                if (StreamHasAivGraph(nodes, stream)) {
                    HCCL_VM_DEBUG(
                        "Skip this slave-stream structure check because the stream contains "
                        "an AIV graph task, rankId={}, streamId={}",
                        rankId, streamId);
                    continue;
                }
                if (taskSize < 2) {
                    return HCCL_SUCCESS;
                }

                const TaskNode* firstTask = GetQueueNode(nodes, stream.front());
                const TaskNode* lastTask = GetQueueNode(nodes, stream.back());
                if (firstTask == nullptr || lastTask == nullptr) {
                    HCCL_VM_ERROR(
                        "{} This slave stream is missing its start node "
                        "or end node, rankId={}, streamId={}, taskCount={}, startNode={}, endNode={}",
                        MakeErrorCodeText(ErrorCode::SINGLETASK_SLAVE_STREAM_INVALID), rankId, streamId, taskSize,
                        firstTask == nullptr ? "null" : firstTask->Describe(),
                        lastTask == nullptr ? "null" : lastTask->Describe());
                    return HCCL_E_PTR;
                }

                size_t backStep = 1;
                size_t skippedEmptyLocalCopyCount = 0;
                while (IsEmptyLocalCopy(lastTask) && backStep < taskSize) {
                    lastTask = GetQueueNode(nodes, stream[taskSize - 1 - backStep]);
                    if (lastTask == nullptr) {
                        HCCL_VM_ERROR(
                            "{} This slave stream still has no valid end "
                            "node after empty local-copy tasks are skipped, rankId={}, streamId={}, "
                            "skippedEmptyLocalCopyCount={}, currentTailNode={}",
                            MakeErrorCodeText(ErrorCode::SINGLETASK_SLAVE_STREAM_INVALID), rankId, streamId,
                            skippedEmptyLocalCopyCount, "null");
                        return HCCL_E_PTR;
                    }
                    ++skippedEmptyLocalCopyCount;
                    ++backStep;
                }

                const char* hcclvmTopoType = std::getenv("HCCLVM_TOPO_TYPE");
                if (hcclvmTopoType != nullptr && std::string(hcclvmTopoType) == "HF") {
                    continue;
                }

                if (!IsLocalAicpuWait(firstTask)) {
                    HCCL_VM_ERROR(
                        "{} The first task in this slave stream is not a "
                        "local WAIT task, rankId={}, streamId={}, actualFirstTaskType={}, firstTask={}",
                        MakeErrorCodeText(ErrorCode::SINGLETASK_SLAVE_STREAM_INVALID), rankId, streamId,
                        TaskTypeName(firstTask->GetType()), firstTask->Describe());
                    return HCCL_E_INTERNAL;
                }

                if (!IsLocalAicpuRecord(lastTask)) {
                    HCCL_VM_ERROR(
                        "{} The last task in this slave stream is not a "
                        "local RECORD task, rankId={}, streamId={}, actualLastTaskType={}, lastTask={}",
                        MakeErrorCodeText(ErrorCode::SINGLETASK_SLAVE_STREAM_INVALID), rankId, streamId,
                        TaskTypeName(lastTask->GetType()), lastTask->Describe());
                    return HCCL_E_INTERNAL;
                }
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult CheckTaskMem(const TaskNode* dummyStart, SingleTaskCheckStats* stats)
    {
        if (dummyStart == nullptr) {
            return HCCL_E_PTR;
        }

        // 第一阶段先用 BFS 收集整张可达子图，避免同一个 DAG 节点被重复检查。
        std::queue<const TaskNode*> candNode;
        std::set<const TaskNode*> visitedNodes;
        std::set<const TaskNode*> simulatedNodes;
        for (const TaskNode* child : dummyStart->GetChildren()) {
            if (child == nullptr) {
                return HCCL_E_PTR;
            }
            visitedNodes.insert(child);
            candNode.push(child);
        }

        while (!candNode.empty()) {
            const TaskNode* curNode = candNode.front();
            candNode.pop();
            simulatedNodes.insert(curNode);
            for (const TaskNode* child : curNode->GetChildren()) {
                if (child == nullptr) {
                    return HCCL_E_PTR;
                }
                if (visitedNodes.count(child) != 0) {
                    continue;
                }
                visitedNodes.insert(child);
                candNode.push(child);
            }
        }

        // 第二阶段逐节点做单点内存合法性检查。
        // 这里不依赖拓扑序，因为该阶段只校验任务自身的 src/dst 描述是否自洽。
        if (stats != nullptr) {
            stats->nodeCount = simulatedNodes.size() + 1U;
            stats->dataTaskNodeCount = 0;
            for (const TaskNode* node : simulatedNodes) {
                if (IsDataMoveTaskNode(node)) {
                    ++stats->dataTaskNodeCount;
                }
            }
        }

        for (const TaskNode* node : simulatedNodes) {
            HcclResult ret = CheckDataMoveTaskMem(node);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
        }

        return HCCL_SUCCESS;
    }
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
