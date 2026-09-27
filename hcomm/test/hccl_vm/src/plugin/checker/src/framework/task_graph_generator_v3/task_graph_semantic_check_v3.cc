/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_graph_semantic_check_v3.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "framework/semantics_check/all2all_semantics_checker.h"
#include "framework/semantics_check/allgather_semantics_checker.h"
#include "framework/semantics_check/allgather_v_semantics_checker.h"
#include "framework/semantics_check/allreduce_semantics_checker.h"
#include "framework/semantics_check/batchsendrecv_semantics_checker.h"
#include "framework/semantics_check/broadcast_semantics_checker.h"
#include "framework/semantics_check/reduce_scatter_semantics_checker.h"
#include "framework/semantics_check/reduce_scatter_v_semantics_checker.h"
#include "framework/semantics_check/reduce_semantics_checker.h"
#include "framework/semantics_check/scatter_semantics_checker.h"
#include "framework/semantics_check/send_recv_semantics_checker.h"
#include "sim_log.h"
#include "utils/check_utils.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"
#include "utils/storage_manager.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {
        // CCU 的单个 MS 槽位按 4KB 对齐管理；语义校验里用它把复用的 MS 拆成独立桶。
        constexpr u64 kMsSliceSize = 4ULL * 1024ULL;
        enum class SliceOpV3 : uint8_t {
            OVERRIDE = 0,
            REDUCE,
        };

        std::string DescribeMemSliceForSemantic(const MemSlice& slice)
        {
            std::ostringstream os;
            os << "{rankId=";
            if (slice.rankId == INVALID_RANK_ID) {
                os << "invalid";
            } else {
                os << slice.rankId;
            }
            os << ", memoryType=" << DescribeMemType(slice.memType) << ", offset=0x" << std::hex << slice.offset
               << ", length=0x" << slice.len << std::dec << "}";
            return os.str();
        }

        struct SliceOpPairV3 {
            RankId srcRank{INVALID_RANK_ID};
            RankId dstRank{INVALID_RANK_ID};
            MemSlice src;
            MemSlice dst;
            TaskPosition position;
            SliceOpV3 op{SliceOpV3::OVERRIDE};
            HcclReduceOp reduceType{HCCL_REDUCE_RESERVED};
            u64 index{0};
            bool useBatchIndex{false};

            std::string Describe() const
            {
                std::ostringstream os;
                os << "{operation=" << (op == SliceOpV3::REDUCE ? "reduce" : "overwrite")
                   << ", sourceMemorySlice=" << DescribeMemSliceForSemantic(src)
                   << ", targetMemorySlice=" << DescribeMemSliceForSemantic(dst) << ", launchIdx=" << position.launchIdx
                   << ", blockId=" << position.blockId << ", pipeId=" << position.pipe
                   << ", taskId=" << position.taskId;
                if (useBatchIndex) {
                    os << ", batchGroup=" << index;
                }
                if (op == SliceOpV3::REDUCE) {
                    os << ", reduceType=" << DumpReduceOpToString(reduceType);
                }
                os << "}";
                return os.str();
            }
        };

        // 对 MS 语义做“按 rank + msId + batch 下标”分桶。
        // 普通内存仍然共用原来的 INPUT/OUTPUT/CCL 语义表，不会被 batch index 隔离。
        struct MsKey {
            RankId rankId{INVALID_RANK_ID};
            u64 msId{0};
            u64 index{0};

            bool operator<(const MsKey& rhs) const
            {
                if (rankId != rhs.rankId) {
                    return rankId < rhs.rankId;
                }
                if (msId != rhs.msId) {
                    return msId < rhs.msId;
                }
                return index < rhs.index;
            }
        };

        struct AivMemoryKey {
            RankId rankId{INVALID_RANK_ID};
            MemType memType{MemType::INVALID};
            u64 aivUBIdx{0};
            u64 batchIndex{0};
            bool isBatchUB{false};

            bool operator<(const AivMemoryKey& rhs) const
            {
                if (rankId != rhs.rankId) {
                    return rankId < rhs.rankId;
                }
                if (memType != rhs.memType) {
                    return static_cast<uint32_t>(memType) < static_cast<uint32_t>(rhs.memType);
                }
                if (aivUBIdx != rhs.aivUBIdx) {
                    return aivUBIdx < rhs.aivUBIdx;
                }
                if (batchIndex != rhs.batchIndex) {
                    return batchIndex < rhs.batchIndex;
                }
                return isBatchUB < rhs.isBatchUB;
            }
        };

        // 对同一个 task 先把所有源语义拷出来，再统一写回目标，
        // 这样 batch 任务内部的后一个 pair 不会读到前一个 pair 刚刚写出的结果。
        struct PreparedSliceOpPairV3 {
            SliceOpPairV3 pair;
            std::vector<BufferSemantic> srcSemantics;
        };

        using InternalBufferSemanticMap = std::map<u64, BufferSemantic>;
        using InternalRankMemorySemantics = std::map<BufferType, InternalBufferSemanticMap>;
        using LegacyRankMemories = std::map<RankId, RankMemorySemantics>;

        u64 GetSegmentEndAddr(const BufferSemantic& semantic);

        struct SemanticState {
            CheckerParam param;
            u64 dataSize{0};
            u64 inputSize{0};
            u64 outputSize{0};
            std::map<RankId, InternalRankMemorySemantics> mem;
            std::map<MsKey, InternalBufferSemanticMap> ms;
            std::map<AivMemoryKey, InternalBufferSemanticMap> AivDevMem;
        };

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

        bool IsDataTask(const TaskNode* node)
        {
            if (node == nullptr) {
                return false;
            }
            switch (node->GetType()) {
                case TaskType::TRANS_MEM:
                case TaskType::BATCH_TRANS_MEM:
                case TaskType::REDUCE:
                case TaskType::BATCH_REDUCE:
                    return true;
                default:
                    return false;
            }
        }

        bool IsSupportedSemanticMemType(MemType memType)
        {
            switch (memType) {
                case MemType::INPUT:
                case MemType::OUTPUT:
                case MemType::CCL:
                case MemType::MS_CCU:
                case MemType::AIV_UB:
                case MemType::AIV_COMM:
                    return true;
                default:
                    return false;
            }
        }

        bool IsValidMemorySlice(const MemSlice& slice)
        {
            return slice.rankId != INVALID_RANK_ID && slice.memType != MemType::INVALID;
        }

        MsKey MakeMsKey(RankId rankId, u64 offset, u64 index) { return MsKey{rankId, offset / kMsSliceSize, index}; }

        u64 MakeAivUbIdx(const TaskPosition& position)
        {
            if (position.launchIdx == std::numeric_limits<uint64_t>::max()
                || position.blockId == std::numeric_limits<uint32_t>::max()) {
                return 0;
            }
            return (position.launchIdx << 32U) | static_cast<u64>(position.blockId);
        }

        AivMemoryKey
        MakeAivMemoryKey(RankId rankId, MemType memType, const TaskPosition& position, u64 batchIndex, bool isBatchUB)
        {
            if (memType == MemType::AIV_COMM) {
                return AivMemoryKey{rankId, memType, 0U, 0U, false};
            }
            const bool effectiveIsBatchUB = memType == MemType::AIV_UB && isBatchUB;
            return AivMemoryKey{
                rankId, memType, MakeAivUbIdx(position), effectiveIsBatchUB ? batchIndex : 0U, effectiveIsBatchUB};
        }

        // 普通内存直接按 rank/bufferType 取语义表；MS 则额外按 batch 下标拆桶，
        // 从而把“同一个 ms 在不同 batch item 复用”的语义分开记录。
        InternalBufferSemanticMap& GetBufferSemanticMap(
            SemanticState& state, RankId rankId, MemType memType, u64 offset, const TaskPosition& position, u64 index,
            bool useBatchIndex)
        {
            if (memType == MemType::MS_CCU) {
                return state.ms[MakeMsKey(rankId, offset, index)];
            }
            if (memType == MemType::AIV_UB || memType == MemType::AIV_COMM) {
                return state.AivDevMem[MakeAivMemoryKey(rankId, memType, position, index, useBatchIndex)];
            }
            return state.mem[rankId][ConvertMemTypeToBufferType(memType)];
        }

        const InternalBufferSemanticMap& GetBufferSemanticMap(
            const SemanticState& state, RankId rankId, MemType memType, u64 offset, const TaskPosition& position,
            u64 index, bool useBatchIndex)
        {
            static const InternalBufferSemanticMap empty;
            if (memType == MemType::MS_CCU) {
                const auto iter = state.ms.find(MakeMsKey(rankId, offset, index));
                return iter == state.ms.end() ? empty : iter->second;
            }
            if (memType == MemType::AIV_UB || memType == MemType::AIV_COMM) {
                const auto iter
                    = state.AivDevMem.find(MakeAivMemoryKey(rankId, memType, position, index, useBatchIndex));
                return iter == state.AivDevMem.end() ? empty : iter->second;
            }
            const auto rankIter = state.mem.find(rankId);
            if (rankIter == state.mem.end()) {
                return empty;
            }
            const auto memIter = rankIter->second.find(ConvertMemTypeToBufferType(memType));
            return memIter == rankIter->second.end() ? empty : memIter->second;
        }

        void InitRankBuffers(SemanticState& state, RankId rankId, bool initInput)
        {
            state.mem[rankId][BufferType::INPUT];
            state.mem[rankId][BufferType::OUTPUT];
            state.mem[rankId][BufferType::CCL];
            state.mem[rankId][BufferType::MS];
            if (!initInput) {
                return;
            }
            BufferSemantic input(0, state.inputSize);
            input.srcBufs.insert(SrcBufDes(rankId, BufferType::INPUT, 0));
            state.mem[rankId][BufferType::INPUT].emplace(input.startAddr, std::move(input));
        }

        HcclResult InitState(SemanticState& state)
        {
            state.param = StorageManager::GetInstance().GetCheckerParam();
            if (state.param.rankSize == 0) {
                HCCL_VM_ERROR(
                    "{} Semantic check initialization failed because the rank count is 0, "
                    "collectiveType={}, dataType={}, elementCount={}, reduceType={}",
                    MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), HcclCmdTypeToString(state.param.cmdType),
                    HcclDataTypeToString(state.param.dataType), state.param.dataCount,
                    DumpReduceOpToString(state.param.reduceType));
                return HCCL_E_PARA;
            }
            CalcDataSize(state.param.cmdType, state.param.dataCount, state.param.dataType, state.dataSize);
            for (RankId rankId = 0; rankId < state.param.rankSize; ++rankId) {
                RankId srcRank = state.param.srcRank;
                RankId dstRank = state.param.dstRank;
                for (const auto& pair : state.param.sendRecvPairs) {
                    if (pair.srcRank == rankId || pair.dstRank == rankId) {
                        srcRank = pair.srcRank;
                        dstRank = pair.dstRank;
                        break;
                    }
                }
                CalcInputOutputSize(
                    state.param.cmdType, state.param.rankSize, state.param.dataCount, state.param.dataType,
                    state.inputSize, state.outputSize, rankId, srcRank, dstRank, state.param.vDataDes,
                    state.param.all2AllDataDes);
                const bool initInput
                    = !(state.param.cmdType == HCCL_CMD_BROADCAST || state.param.cmdType == HCCL_CMD_SCATTER)
                      || rankId == state.param.root;
                InitRankBuffers(state, rankId, initInput);
            }
            return HCCL_SUCCESS;
        }

        u64 GetSegmentEndAddr(const BufferSemantic& semantic) { return semantic.startAddr + semantic.size; }

        BufferSemantic CloneSegmentRange(const BufferSemantic& semantic, u64 newStartAddr, u64 newEndAddr)
        {
            const u64 srcOffset = newStartAddr - semantic.startAddr;
            BufferSemantic cloned(
                newStartAddr, newEndAddr - newStartAddr, semantic.isReduce, semantic.reduceType,
                OffsetSrcBufs(semantic.srcBufs, srcOffset));
            return cloned;
        }

        InternalBufferSemanticMap::iterator
        FindFirstOverlap(InternalBufferSemanticMap& bufferSemantics, u64 startAddr, u64 endAddr)
        {
            auto iter = bufferSemantics.lower_bound(startAddr);
            if (iter != bufferSemantics.begin()) {
                auto prevIter = std::prev(iter);
                if (GetSegmentEndAddr(prevIter->second) > startAddr) {
                    iter = prevIter;
                }
            }
            while (iter != bufferSemantics.end() && GetSegmentEndAddr(iter->second) <= startAddr) {
                ++iter;
            }
            if (iter != bufferSemantics.end() && iter->second.startAddr >= endAddr) {
                return bufferSemantics.end();
            }
            return iter;
        }

        InternalBufferSemanticMap::const_iterator
        FindFirstOverlap(const InternalBufferSemanticMap& bufferSemantics, u64 startAddr, u64 endAddr)
        {
            auto iter = bufferSemantics.lower_bound(startAddr);
            if (iter != bufferSemantics.begin()) {
                auto prevIter = std::prev(iter);
                if (GetSegmentEndAddr(prevIter->second) > startAddr) {
                    iter = prevIter;
                }
            }
            while (iter != bufferSemantics.end() && GetSegmentEndAddr(iter->second) <= startAddr) {
                ++iter;
            }
            if (iter != bufferSemantics.end() && iter->second.startAddr >= endAddr) {
                return bufferSemantics.end();
            }
            return iter;
        }

        void SplitBufferSemantic(InternalBufferSemanticMap& bufferSemantics, u64 splitAddr)
        {
            if (bufferSemantics.empty()) {
                return;
            }

            auto iter = bufferSemantics.upper_bound(splitAddr);
            if (iter == bufferSemantics.begin()) {
                return;
            }
            --iter;
            BufferSemantic& semantic = iter->second;
            if (splitAddr <= semantic.startAddr || splitAddr >= GetSegmentEndAddr(semantic)) {
                return;
            }

            BufferSemantic rightSemantic = CloneSegmentRange(semantic, splitAddr, GetSegmentEndAddr(semantic));
            semantic.size = splitAddr - semantic.startAddr;
            bufferSemantics.emplace(rightSemantic.startAddr, std::move(rightSemantic));
        }

        void CollectOverlappingBufferSemantics(
            InternalBufferSemanticMap& bufferSemantics, u64 startAddr, u64 endAddr,
            std::vector<BufferSemantic*>& overlaps)
        {
            if (startAddr >= endAddr) {
                return;
            }

            auto iter = FindFirstOverlap(bufferSemantics, startAddr, endAddr);
            while (iter != bufferSemantics.end() && iter->second.startAddr < endAddr) {
                overlaps.push_back(&iter->second);
                ++iter;
            }
        }

        void CollectOverlappingBufferSemantics(
            const InternalBufferSemanticMap& bufferSemantics, u64 startAddr, u64 endAddr,
            std::vector<const BufferSemantic*>& overlaps)
        {
            if (startAddr >= endAddr) {
                return;
            }

            auto iter = FindFirstOverlap(bufferSemantics, startAddr, endAddr);
            while (iter != bufferSemantics.end() && iter->second.startAddr < endAddr) {
                overlaps.push_back(&iter->second);
                ++iter;
            }
        }

        void EraseBufferSemanticsInRange(InternalBufferSemanticMap& bufferSemantics, u64 startAddr, u64 endAddr)
        {
            if (startAddr >= endAddr) {
                return;
            }

            auto iter = bufferSemantics.lower_bound(startAddr);
            while (iter != bufferSemantics.end() && iter->second.startAddr < endAddr) {
                iter = bufferSemantics.erase(iter);
            }
        }

        bool CanMergeSrcBufs(const BufferSemantic& lhs, const BufferSemantic& rhs)
        {
            if (lhs.srcBufs.size() != rhs.srcBufs.size()) {
                return false;
            }

            auto lhsIter = lhs.srcBufs.begin();
            auto rhsIter = rhs.srcBufs.begin();
            while (lhsIter != lhs.srcBufs.end() && rhsIter != rhs.srcBufs.end()) {
                if (lhsIter->rankId != rhsIter->rankId || lhsIter->bufType != rhsIter->bufType
                    || lhsIter->srcAddr + lhs.size != rhsIter->srcAddr) {
                    return false;
                }
                ++lhsIter;
                ++rhsIter;
            }
            return lhsIter == lhs.srcBufs.end() && rhsIter == rhs.srcBufs.end();
        }

        bool CanMergeSemantics(const BufferSemantic& lhs, const BufferSemantic& rhs)
        {
            return GetSegmentEndAddr(lhs) == rhs.startAddr && lhs.isReduce == rhs.isReduce
                   && lhs.reduceType == rhs.reduceType && CanMergeSrcBufs(lhs, rhs);
        }

        void MergeAdjacentBufferSemantics(InternalBufferSemanticMap& bufferSemantics, u64 startAddr, u64 endAddr)
        {
            if (bufferSemantics.empty()) {
                return;
            }

            auto iter = bufferSemantics.lower_bound(startAddr);
            if (iter != bufferSemantics.begin()) {
                --iter;
            }
            while (iter != bufferSemantics.end()) {
                auto nextIter = std::next(iter);
                if (nextIter == bufferSemantics.end()) {
                    break;
                }
                if (iter->second.startAddr > endAddr && nextIter->second.startAddr > endAddr) {
                    break;
                }
                if (!CanMergeSemantics(iter->second, nextIter->second)) {
                    iter = nextIter;
                    continue;
                }

                iter->second.size += nextIter->second.size;
                endAddr = std::max(endAddr, GetSegmentEndAddr(nextIter->second));
                bufferSemantics.erase(nextIter);
            }
        }

        void InsertBufferSemantic(InternalBufferSemanticMap& bufferSemantics, BufferSemantic semantic)
        {
            if (semantic.size == 0) {
                return;
            }
            auto iter = bufferSemantics.lower_bound(semantic.startAddr);
            if (iter != bufferSemantics.end() && iter->first == semantic.startAddr) {
                iter->second = std::move(semantic);
                return;
            }
            bufferSemantics.emplace_hint(iter, semantic.startAddr, std::move(semantic));
        }

        int64_t OffsetDelta(u64 lhs, u64 rhs)
        {
            if (lhs >= rhs) {
                return static_cast<int64_t>(lhs - rhs);
            }
            return -static_cast<int64_t>(rhs - lhs);
        }

        u64 ApplyOffsetDelta(u64 offset, int64_t delta)
        {
            if (delta >= 0) {
                return offset + static_cast<u64>(delta);
            }
            return offset - static_cast<u64>(-delta);
        }

        void GetSrcIntersectionAddr(
            const SliceOpPairV3& pair, const BufferSemantic& srcSemantic, u64& srcStartAddr, u64& srcEndAddr)
        {
            srcStartAddr = pair.src.offset;
            srcEndAddr = srcStartAddr + pair.src.len;
            if (srcSemantic.startAddr > srcStartAddr) {
                srcStartAddr = srcSemantic.startAddr;
            }
            if (srcSemantic.startAddr + srcSemantic.size < srcEndAddr) {
                srcEndAddr = srcSemantic.startAddr + srcSemantic.size;
            }
        }

        HcclResult CheckBufSemantics(
            const std::vector<const BufferSemantic*>& bufSemantics, u64 startAddr, u64 size, bool ignoreError)
        {
            if (size == 0) {
                return HCCL_SUCCESS;
            }
            if (bufSemantics.empty()) {
                if (!ignoreError) {
                    HCCL_VM_ERROR(
                        "{} No source/output information was found for the target memory range, "
                        "startAddr=0x{:x}, size=0x{:x}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_BUFFER_EMPTY), startAddr, size);
                }
                return HCCL_E_PARA;
            }

            u64 totalSize = 0;
            const BufferSemantic* prev = bufSemantics.front();
            if (prev->startAddr > startAddr) {
                if (!ignoreError) {
                    HCCL_VM_ERROR(
                        "{} Output data does not start from the expected address; the beginning is "
                        "missing, expectedStart=0x{:x}, actualStart=0x{:x}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_GAP), startAddr, prev->startAddr);
                }
                return HCCL_E_PARA;
            }
            if (prev->startAddr + prev->size >= startAddr + size) {
                totalSize += size;
            } else {
                totalSize += prev->startAddr + prev->size - startAddr;
            }

            for (size_t index = 1; index < bufSemantics.size(); ++index) {
                const BufferSemantic* cur = bufSemantics[index];
                if (cur->startAddr != prev->startAddr + prev->size) {
                    if (!ignoreError) {
                        HCCL_VM_ERROR(
                            "{} Output data is broken in the middle; one piece ends at 0x{:x} "
                            "but the next starts at 0x{:x}",
                            MakeErrorCodeText(ErrorCode::SEMANTIC_GAP), prev->startAddr + prev->size, cur->startAddr);
                    }
                    return HCCL_E_PARA;
                }
                if (cur->startAddr + cur->size >= startAddr + size) {
                    totalSize += startAddr + size - cur->startAddr;
                } else {
                    totalSize += cur->size;
                }
                prev = cur;
            }

            if (totalSize != size) {
                if (!ignoreError) {
                    HCCL_VM_ERROR(
                        "{} Output data ends too early; the tail is missing, expectedEnd=0x{:x}, "
                        "actualEnd=0x{:x}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_GAP), startAddr + size, startAddr + totalSize);
                }
                return HCCL_E_PARA;
            }
            return HCCL_SUCCESS;
        }

        BufferSemantic BuildDstSemanticFromSrc(const SliceOpPairV3& pair, const BufferSemantic& srcSemantic)
        {
            u64 srcStartAddr = 0;
            u64 srcEndAddr = 0;
            GetSrcIntersectionAddr(pair, srcSemantic, srcStartAddr, srcEndAddr);

            const int64_t dstSrcOffset = OffsetDelta(pair.dst.offset, pair.src.offset);
            const u64 dstStartAddr = ApplyOffsetDelta(srcStartAddr, dstSrcOffset);
            const u64 dstEndAddr = ApplyOffsetDelta(srcEndAddr, dstSrcOffset);

            BufferSemantic dstSemantic(
                dstStartAddr, dstEndAddr - dstStartAddr, srcSemantic.isReduce, srcSemantic.reduceType,
                OffsetSrcBufs(srcSemantic.srcBufs, srcStartAddr - srcSemantic.startAddr));
            return dstSemantic;
        }

        void ApplyOverrideSemantics(
            const SliceOpPairV3& pair, SemanticState& state, const std::vector<const BufferSemantic*>& srcSemantics)
        {
            InternalBufferSemanticMap& dstSemantics = GetBufferSemanticMap(
                state, pair.dstRank, pair.dst.memType, pair.dst.offset, pair.position, pair.index, pair.useBatchIndex);
            const u64 dstStartAddr = pair.dst.offset;
            const u64 dstEndAddr = dstStartAddr + pair.dst.len;
            SplitBufferSemantic(dstSemantics, dstStartAddr);
            SplitBufferSemantic(dstSemantics, dstEndAddr);
            EraseBufferSemanticsInRange(dstSemantics, dstStartAddr, dstEndAddr);

            for (const BufferSemantic* semantic : srcSemantics) {
                if (semantic == nullptr) {
                    continue;
                }
                InsertBufferSemantic(dstSemantics, BuildDstSemanticFromSrc(pair, *semantic));
            }
            MergeAdjacentBufferSemantics(dstSemantics, dstStartAddr, dstEndAddr);
        }

        HcclResult AddReduceSourcesToDstSegments(
            const SliceOpPairV3& pair, const BufferSemantic& srcSemantic,
            const std::vector<BufferSemantic*>& affectedDstSemantics, u64 srcStartAddr)
        {
            u64 srcOffset = srcStartAddr - srcSemantic.startAddr;
            for (BufferSemantic* dstSemantic : affectedDstSemantics) {
                if (dstSemantic == nullptr) {
                    return HCCL_E_PTR;
                }
                if (dstSemantic->srcBufs.size() == 1) {
                    if (dstSemantic->isReduce) {
                        HCCL_VM_ERROR(
                            "{} Target output range is already marked as a reduce result before "
                            "enough source data is merged, dataMapping={}",
                            MakeErrorCodeText(ErrorCode::SEMANTIC_REDUCE_ERROR), pair.Describe());
                        return HCCL_E_PARA;
                    }
                    dstSemantic->isReduce = true;
                    dstSemantic->reduceType = pair.reduceType;
                }
                if (srcSemantic.srcBufs.size() > 1 && dstSemantic->reduceType != srcSemantic.reduceType) {
                    HCCL_VM_ERROR(
                        "{} Reduce result type is inconsistent while merging one source data "
                        "range, dataMapping={}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_REDUCE_ERROR), pair.Describe());
                    return HCCL_E_PARA;
                }

                const std::set<SrcBufDes> shiftedSrcBufs = OffsetSrcBufs(srcSemantic.srcBufs, srcOffset);
                for (const auto& srcBuf : shiftedSrcBufs) {
                    const size_t before = dstSemantic->srcBufs.size();
                    dstSemantic->srcBufs.insert(srcBuf);
                    if (before == dstSemantic->srcBufs.size()) {
                        HCCL_VM_ERROR(
                            "{} The same source data is added twice to one output range during "
                            "reduce, dataMapping={}, sourceOffsetInThisRange=0x{:x}, outputRange=[0x{:x},0x{:x})",
                            MakeErrorCodeText(ErrorCode::SEMANTIC_REDUCE_ERROR), pair.Describe(), srcOffset,
                            dstSemantic->startAddr, GetSegmentEndAddr(*dstSemantic));
                        return HCCL_E_PARA;
                    }
                }
                srcOffset += dstSemantic->size;
            }
            return HCCL_SUCCESS;
        }

        HcclResult
        ApplyReduceSemantic(const SliceOpPairV3& pair, SemanticState& state, const BufferSemantic& srcSemantic)
        {
            u64 srcStartAddr = 0;
            u64 srcEndAddr = 0;
            GetSrcIntersectionAddr(pair, srcSemantic, srcStartAddr, srcEndAddr);
            const int64_t dstSrcOffset = OffsetDelta(pair.dst.offset, pair.src.offset);
            const u64 dstStartAddr = ApplyOffsetDelta(srcStartAddr, dstSrcOffset);
            const u64 dstEndAddr = ApplyOffsetDelta(srcEndAddr, dstSrcOffset);

            InternalBufferSemanticMap& dstSemantics = GetBufferSemanticMap(
                state, pair.dstRank, pair.dst.memType, pair.dst.offset, pair.position, pair.index, pair.useBatchIndex);
            SplitBufferSemantic(dstSemantics, dstStartAddr);
            SplitBufferSemantic(dstSemantics, dstEndAddr);

            std::vector<BufferSemantic*> affectedDstSemantics;
            CollectOverlappingBufferSemantics(dstSemantics, dstStartAddr, dstEndAddr, affectedDstSemantics);
            std::vector<const BufferSemantic*> affectedViews;
            affectedViews.reserve(affectedDstSemantics.size());
            for (const BufferSemantic* semantic : affectedDstSemantics) {
                affectedViews.push_back(semantic);
            }
            HcclResult ret = CheckBufSemantics(affectedViews, dstStartAddr, dstEndAddr - dstStartAddr, false);
            if (ret != HCCL_SUCCESS) {
                HCCL_VM_ERROR(
                    "{} Target output range is only partially filled before reduce continues, "
                    "dataMapping={}, outputRange=[0x{:x},0x{:x}), pieceCount={}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_REDUCE_ERROR), pair.Describe(), dstStartAddr, dstEndAddr,
                    affectedDstSemantics.size());
                return ret;
            }

            ret = AddReduceSourcesToDstSegments(pair, srcSemantic, affectedDstSemantics, srcStartAddr);
            if (ret != HCCL_SUCCESS) {
                HCCL_VM_ERROR(
                    "{} Failed to merge one source data range into the target output range during "
                    "reduce, dataMapping={}, srcRange=[0x{:x},0x{:x}), dstRange=[0x{:x},0x{:x}), ret={}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_REDUCE_ERROR), pair.Describe(), srcStartAddr, srcEndAddr,
                    dstStartAddr, dstEndAddr, static_cast<int32_t>(ret));
                return ret;
            }

            return HCCL_SUCCESS;
        }

        HcclResult ApplySrcSemanticsToDst(
            const SliceOpPairV3& pair, SemanticState& state, const std::vector<const BufferSemantic*>& srcSemantics)
        {
            if (pair.op == SliceOpV3::OVERRIDE) {
                ApplyOverrideSemantics(pair, state, srcSemantics);
                return HCCL_SUCCESS;
            }

            for (const BufferSemantic* semantic : srcSemantics) {
                if (semantic == nullptr) {
                    return HCCL_E_PTR;
                }
                HcclResult ret = ApplyReduceSemantic(pair, state, *semantic);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            return HCCL_SUCCESS;
        }

        HcclResult LoadSliceOpPairSrcSemantics(
            const SliceOpPairV3& pair, const SemanticState& state, std::vector<BufferSemantic>& srcSemantics)
        {
            srcSemantics.clear();
            if (pair.src.len == 0) {
                return HCCL_SUCCESS;
            }
            if (!IsValidMemorySlice(pair.src) || !IsValidMemorySlice(pair.dst)) {
                HCCL_VM_ERROR(
                    "{} One source-to-target mapping uses an invalid memory slice, "
                    "dataMapping={}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), pair.Describe());
                return HCCL_E_PARA;
            }
            if (!IsSupportedSemanticMemType(pair.src.memType) || !IsSupportedSemanticMemType(pair.dst.memType)) {
                HCCL_VM_ERROR(
                    "{} This data mapping uses a memory type that semantic check does not support, "
                    "dataMapping={}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), pair.Describe());
                return HCCL_E_NOT_SUPPORT;
            }

            const InternalBufferSemanticMap& srcSemanticsMap = GetBufferSemanticMap(
                state, pair.srcRank, pair.src.memType, pair.src.offset, pair.position, pair.index, pair.useBatchIndex);
            std::vector<const BufferSemantic*> srcViews;
            CollectOverlappingBufferSemantics(
                srcSemanticsMap, pair.src.offset, pair.src.offset + pair.src.len, srcViews);

            HcclResult ret = CheckBufSemantics(srcViews, pair.src.offset, pair.src.len, true);
            if (ret != HCCL_SUCCESS) {
                if (pair.op == SliceOpV3::REDUCE) {
                    HCCL_VM_ERROR(
                        "{} Source data needed by this reduce is missing, dataMapping={}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_REDUCE_ERROR), pair.Describe());
                    return ret;
                }
                // 老语义实现对 override 的不完整源语义是放行但告警；
                // 这里保持同样语义，至少把“并非完整 memcpy 语义”的风险显式打出来。
                HCCL_VM_WARN(
                    "{} Source data needed by this overwrite is missing, dataMapping={}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_SIMULATE_FAILED), pair.Describe());
            }

            srcSemantics.reserve(srcViews.size());
            for (const BufferSemantic* semantic : srcViews) {
                if (semantic == nullptr) {
                    return HCCL_E_PTR;
                }
                srcSemantics.push_back(*semantic);
            }
            return HCCL_SUCCESS;
        }

        HcclResult ApplyPreparedSliceOpPair(const PreparedSliceOpPairV3& prepared, SemanticState& state)
        {
            if (prepared.pair.src.len == 0) {
                return HCCL_SUCCESS;
            }
            std::vector<const BufferSemantic*> srcViews;
            srcViews.reserve(prepared.srcSemantics.size());
            for (const auto& semantic : prepared.srcSemantics) {
                srcViews.push_back(&semantic);
            }
            HcclResult ret = ApplySrcSemanticsToDst(prepared.pair, state, srcViews);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            return HCCL_SUCCESS;
        }

        void GetSliceOpPairs(const TaskNode* node, std::vector<SliceOpPairV3>& pairs)
        {
            pairs.clear();
            if (node == nullptr || !IsDataTask(node)) {
                return;
            }

            // 这里把不同 task 类型统一翻译成“src -> dst 的语义操作对”，
            // 后续模拟阶段不再关心原始 task 类型细节。
            if (node->GetType() == TaskType::TRANS_MEM) {
                const auto* task = dynamic_cast<const TaskTransMem*>(node);
                if (task == nullptr) {
                    return;
                }
                pairs.push_back(SliceOpPairV3{
                    task->GetSrc().rankId, task->GetDst().rankId, task->GetSrc(), task->GetDst(), node->GetPosition(),
                    SliceOpV3::OVERRIDE, HCCL_REDUCE_RESERVED, 0});
                return;
            }

            if (node->GetType() == TaskType::REDUCE) {
                const auto* task = dynamic_cast<const TaskReduce*>(node);
                if (task == nullptr) {
                    return;
                }
                const auto& srcs = task->GetSrcs();
                for (size_t index = 0; index < srcs.size(); ++index) {
                    const auto& src = srcs[index];
                    SliceOpPairV3 pair;
                    pair.srcRank = src.rankId;
                    pair.dstRank = task->GetDst().rankId;
                    pair.src = src;
                    pair.dst = task->GetDst();
                    pair.position = node->GetPosition();
                    pair.op = SliceOpV3::REDUCE;
                    pair.reduceType = static_cast<HcclReduceOp>(task->GetReduceOp());
                    pairs.push_back(std::move(pair));
                }
                return;
            }

            if (node->GetType() == TaskType::BATCH_TRANS_MEM) {
                const auto* task = dynamic_cast<const TaskBatchTransMem*>(node);
                if (task == nullptr) {
                    return;
                }
                const auto& srcs = task->GetSrcs();
                const auto& dsts = task->GetDsts();
                const size_t count = std::min(srcs.size(), dsts.size());
                for (size_t index = 0; index < count; ++index) {
                    pairs.push_back(SliceOpPairV3{
                        srcs[index].rankId, dsts[index].rankId, srcs[index], dsts[index], node->GetPosition(),
                        SliceOpV3::OVERRIDE, HCCL_REDUCE_RESERVED, static_cast<u64>(index), true});
                }
                return;
            }

            if (node->GetType() == TaskType::BATCH_REDUCE) {
                const auto* task = dynamic_cast<const TaskBatchReduce*>(node);
                if (task == nullptr) {
                    return;
                }
                const auto& srcGroups = task->GetSrcs();
                const auto& dsts = task->GetDsts();
                const size_t count = std::min(srcGroups.size(), dsts.size());
                for (size_t index = 0; index < count; ++index) {
                    for (const auto& src : srcGroups[index]) {
                        SliceOpPairV3 pair;
                        pair.srcRank = src.rankId;
                        pair.dstRank = dsts[index].rankId;
                        pair.src = src;
                        pair.dst = dsts[index];
                        pair.position = node->GetPosition();
                        pair.op = SliceOpV3::REDUCE;
                        pair.reduceType = static_cast<HcclReduceOp>(task->GetReduceOp());
                        pair.index = static_cast<u64>(index);
                        pair.useBatchIndex = true;
                        pairs.push_back(std::move(pair));
                    }
                }
            }
        }

        HcclResult SimulateTask(const TaskNode* node, SemanticState& state)
        {
            if (!IsDataTask(node)) {
                return HCCL_SUCCESS;
            }
            std::vector<SliceOpPairV3> pairs;
            GetSliceOpPairs(node, pairs);
            // 先 load 再 apply，保证一个 batch task 内所有 pair 看到的是同一时刻的源语义快照。
            std::vector<PreparedSliceOpPairV3> preparedPairs;
            preparedPairs.reserve(pairs.size());
            for (const auto& pair : pairs) {
                PreparedSliceOpPairV3 prepared;
                prepared.pair = pair;
                HcclResult ret = LoadSliceOpPairSrcSemantics(pair, state, prepared.srcSemantics);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                preparedPairs.push_back(std::move(prepared));
            }
            for (const auto& prepared : preparedPairs) {
                HcclResult ret = ApplyPreparedSliceOpPair(prepared, state);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            return HCCL_SUCCESS;
        }

        LegacyRankMemories BuildLegacyMemories(const SemanticState& state)
        {
            LegacyRankMemories legacy;
            for (const auto& rankEntry : state.mem) {
                RankMemorySemantics& legacyMem = legacy[rankEntry.first];
                for (const auto& bufEntry : rankEntry.second) {
                    auto& legacyBufSet = legacyMem[bufEntry.first];
                    for (const auto& semanticEntry : bufEntry.second) {
                        legacyBufSet.insert(semanticEntry.second);
                    }
                }
            }
            return legacy;
        }

        HcclResult CheckFinalOutput(const SemanticState& state)
        {
            auto allRankMemSemantics = BuildLegacyMemories(state);
            switch (state.param.cmdType) {
                case HCCL_CMD_ALLREDUCE:
                    return TaskCheckAllReduceSemantics(allRankMemSemantics, state.dataSize, state.param.reduceType);
                case HCCL_CMD_ALLGATHER:
                    return TaskCheckAllGatherSemantics(allRankMemSemantics, state.dataSize);
                case HCCL_CMD_ALLGATHER_V: {
                    auto vDataDes = state.param.vDataDes;
                    return TaskCheckAllGatherVSemantics(allRankMemSemantics, vDataDes);
                }
                case HCCL_CMD_REDUCE_SCATTER:
                    return TaskCheckReduceScatterSemantics(allRankMemSemantics, state.dataSize, state.param.reduceType);
                case HCCL_CMD_REDUCE_SCATTER_V: {
                    auto vDataDes = state.param.vDataDes;
                    return TaskCheckReduceScatterVSemantics(allRankMemSemantics, state.param.reduceType, vDataDes);
                }
                case HCCL_CMD_ALLTOALL:
                case HCCL_CMD_ALLTOALLV:
                case HCCL_CMD_ALLTOALLVC: {
                    auto all2All = state.param.all2AllDataDes;
                    return TaskCheckAll2AllSemantics(allRankMemSemantics, all2All);
                }
                case HCCL_CMD_SEND:
                case HCCL_CMD_RECEIVE:
                    return TaskCheckSendRecvGroupSemantics(
                        allRankMemSemantics, state.dataSize, state.param.sendRecvPairs);
                case HCCL_CMD_BROADCAST:
                    return TaskCheckBroadcastSemantics(allRankMemSemantics, state.dataSize, state.param.root);
                case HCCL_CMD_REDUCE:
                    return TaskCheckReduceSemantics(
                        allRankMemSemantics, state.dataSize, state.param.reduceType, state.param.root);
                case HCCL_CMD_SCATTER:
                    return TaskCheckScatterSemantics(allRankMemSemantics, state.dataSize, state.param.root);
                case HCCL_CMD_BATCH_SEND_RECV:
                    return TaskCheckBatchSendRecvRingSemantics(
                        allRankMemSemantics, state.param.rankSize, state.dataSize);
                default:
                    HCCL_VM_WARN(
                        "{} Final output validation does not support this collective type yet, "
                        "collectiveType={}, rankCount={}, dataType={}, elementCount={}, reduceType={}",
                        MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), HcclCmdTypeToString(state.param.cmdType),
                        state.param.rankSize, HcclDataTypeToString(state.param.dataType), state.param.dataCount,
                        DumpReduceOpToString(state.param.reduceType));
                    return HCCL_E_NOT_SUPPORT;
            }
        }

        void FillStats(const SemanticState& state, size_t handledNodeCount, SemanticCheckStats* stats)
        {
            if (stats == nullptr) {
                return;
            }
            stats->handledNodeCount = handledNodeCount;
            stats->rankCount = state.mem.size();
            stats->normalSemanticCount = 0;
            stats->normalSemanticBytes = 0;
            for (const auto& rankEntry : state.mem) {
                for (const auto& bufEntry : rankEntry.second) {
                    stats->normalSemanticCount += bufEntry.second.size();
                    for (const auto& semanticEntry : bufEntry.second) {
                        stats->normalSemanticBytes += semanticEntry.second.size;
                    }
                }
            }
            stats->msBucketCount = state.ms.size();
        }
    } // namespace

    HcclResult CheckSemantics(const TaskNode* start, SemanticCheckStats* stats)
    {
        if (start == nullptr) {
            HCCL_VM_ERROR(
                "{} Semantic check cannot start because the main start node is missing, "
                "mainStartNode=null",
                MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR));
            return HCCL_E_PTR;
        }

        SemanticState state;
        HcclResult ret = InitState(state);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }

        // 第一步先用 BFS 把从 start 可达的节点全部收集出来，并初始化每个节点的“剩余父节点数”。
        // 这里不直接在 BFS 顺序上做语义模拟，因为 DAG 中子节点可能比某些父节点更早被扫描到。
        std::unordered_map<NodeId, const TaskNode*> nodes;
        std::unordered_map<NodeId, u32> pendingParents;
        std::queue<const TaskNode*> walk;
        std::set<NodeId> visited;
        walk.push(start);
        visited.insert(start->GetNodeId());
        while (!walk.empty()) {
            const TaskNode* node = walk.front();
            walk.pop();
            nodes[node->GetNodeId()] = node;
            pendingParents[node->GetNodeId()] = static_cast<u32>(node->GetParents().size());
            for (const TaskNode* child : node->GetChildren()) {
                if (child == nullptr) {
                    continue;
                }
                if (visited.insert(child->GetNodeId()).second) {
                    walk.push(child);
                }
            }
        }

        // 第二步做一次基于入度的拓扑模拟：只有 pendingParents 归零的节点才允许执行。
        // 这样可以直接复用当前 DAG 上已有的 parent/child 约束，不需要再单独识别 loop 区间。
        std::queue<const TaskNode*> ready;
        for (const auto& entry : nodes) {
            if (pendingParents[entry.first] == 0) {
                ready.push(entry.second);
            }
        }

        size_t handled = 0;
        while (!ready.empty()) {
            const TaskNode* node = ready.front();
            ready.pop();
            ++handled;
            ret = SimulateTask(node, state);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            for (const TaskNode* child : node->GetChildren()) {
                if (child == nullptr) {
                    HCCL_VM_ERROR(
                        "{} Graph structure is broken because one child node is null, parent={}",
                        MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), node->Describe());
                    return HCCL_E_PTR;
                }
                auto iter = pendingParents.find(child->GetNodeId());
                if (iter == pendingParents.end() || iter->second == 0) {
                    continue;
                }
                --iter->second;
                if (iter->second == 0) {
                    ready.push(child);
                }
            }
        }

        if (handled != nodes.size()) {
            const TaskNode* firstRemainingNode = nullptr;
            for (const auto& entry : pendingParents) {
                if (entry.second != 0) {
                    auto nodeIter = nodes.find(entry.first);
                    if (nodeIter != nodes.end()) {
                        firstRemainingNode = nodeIter->second;
                        break;
                    }
                }
            }
            HCCL_VM_ERROR(
                "{} Output simulation stopped because some tasks still have unresolved "
                "dependencies, handledNodeCount={}, totalNodeCount={}, firstRemainingNode={}",
                MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), handled, nodes.size(),
                firstRemainingNode == nullptr ? std::string("node=null") : firstRemainingNode->Describe());
            return HCCL_E_INTERNAL;
        }

        ret = CheckFinalOutput(state);
        if (ret != HCCL_SUCCESS && ret != HCCL_E_NOT_SUPPORT) {
            HCCL_VM_ERROR("Final semantic check failed, ret={}", static_cast<uint32_t>(ret));
            return ret;
        }

        FillStats(state, handled, stats);
        return HCCL_SUCCESS;
    }
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
