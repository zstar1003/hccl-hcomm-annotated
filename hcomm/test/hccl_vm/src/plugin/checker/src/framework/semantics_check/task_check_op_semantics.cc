/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_check_op_semantics.h"

#include <algorithm>
#include <queue>
#include <tuple>

#include "all2all_semantics_checker.h"
#include "allgather_semantics_checker.h"
#include "allgather_v_semantics_checker.h"
#include "allreduce_semantics_checker.h"
#include "batchsendrecv_semantics_checker.h"
#include "broadcast_semantics_checker.h"
#include "check_utils.h"
#include "dump/dump_graph.h"
#include "dump/dump_memory_timeline.h"
#include "dump/validation_issue_record_utils.h"
#include "hccl_types.h"
#include "reduce_scatter_semantics_checker.h"
#include "reduce_scatter_v_semantics_checker.h"
#include "reduce_semantics_checker.h"
#include "scatter_semantics_checker.h"
#include "send_recv_semantics_checker.h"
#include "task_ccu.h"

namespace HcclSim {
namespace {
    using QueueKey = std::pair<RankId, u32>;

    bool TimelineNodeLess(const TimelineEvent* lhs, const TimelineEvent* rhs)
    {
        if (lhs->rankId != rhs->rankId) {
            return lhs->rankId < rhs->rankId;
        }
        if (lhs->queueId != rhs->queueId) {
            return lhs->queueId < rhs->queueId;
        }
        if (lhs->pos != rhs->pos) {
            return lhs->pos < rhs->pos;
        }
        if (lhs->simGlobalStep != rhs->simGlobalStep) {
            return lhs->simGlobalStep < rhs->simGlobalStep;
        }
        return lhs->nodeId < rhs->nodeId;
    }

    bool TimelineReplayLess(const TimelineEvent* lhs, const TimelineEvent* rhs)
    {
        if (lhs->logicalEndStep != rhs->logicalEndStep) {
            return lhs->logicalEndStep < rhs->logicalEndStep;
        }
        if (lhs->logicalStartStep != rhs->logicalStartStep) {
            return lhs->logicalStartStep < rhs->logicalStartStep;
        }
        if (lhs->rankId != rhs->rankId) {
            return lhs->rankId < rhs->rankId;
        }
        if (lhs->queueId != rhs->queueId) {
            return lhs->queueId < rhs->queueId;
        }
        if (lhs->pos != rhs->pos) {
            return lhs->pos < rhs->pos;
        }
        return lhs->simGlobalStep < rhs->simGlobalStep;
    }

    void AppendTimelineRelatedRanks(nlohmann::json& detail, const std::vector<TimelineEvent>& events)
    {
        std::set<RankId> relatedRanks;
        for (const auto& event : events) {
            relatedRanks.insert(event.rankId);
        }
        for (RankId rankId : relatedRanks) {
            HcclSim::AppendRelatedRank(detail, rankId);
        }
    }
} // namespace

void TaskCheckOpSemantics::InitInputBuffer()
{
    for (auto& child : graphHead_->children) {
        RankId rankId = child->rankIdx;
        CalcInputOutputSize(
            opType_, graphHead_->children.size(), dataCount_, dataType_, inputDataSize_, outputDataSize_, rankId,
            srcRank_, dstRank_, vDataDes_, all2AllDataDes_);
        BufferSemantic inputInitStatus(0, inputDataSize_);
        inputInitStatus.srcBufs.insert(SrcBufDes(rankId, BufferType::INPUT, 0));
        allRankMemSemantics_[rankId][BufferType::INPUT].insert(inputInitStatus);
    }
    return;
}

void TaskCheckOpSemantics::InitInputBuffer(RankId root)
{
    for (auto& child : graphHead_->children) {
        RankId rankId = child->rankIdx;
        if (rankId == root) {
            CalcInputOutputSize(
                opType_, graphHead_->children.size(), dataCount_, dataType_, inputDataSize_, outputDataSize_, rankId,
                srcRank_, dstRank_, vDataDes_, all2AllDataDes_);
            BufferSemantic inputInitStatus(0, inputDataSize_);
            inputInitStatus.srcBufs.insert(SrcBufDes(rankId, BufferType::INPUT, 0));
            allRankMemSemantics_[rankId][BufferType::INPUT].insert(inputInitStatus);
        }
    }
    return;
}

bool TaskCheckOpSemantics::IsReadyForSimulate(const TaskNode* node, std::set<TaskNode*>& simulatedNodes) const
{
    for (auto& parent : node->parents) {
        if (simulatedNodes.count(parent) == 0) {
            return false;
        }
    }
    return true;
}

HcclResult TaskCheckOpSemantics::CheckBufSemantics(
    std::vector<BufferSemantic*>& bufSemantics, u64 startAddr, u64 size, bool ignoreError) const
{
    if (bufSemantics.size() == 0) {
        if (!ignoreError) {
            HCCL_ERROR("When check buf semantics in range, buf semantics to check is empty");
        } else {
            HCCL_WARNING("When check buf semantics in range, buf semantics to check is empty");
        }

        nlohmann::json detail = nlohmann::json::object();
        detail["start_addr"] = startAddr;
        detail["size"] = size;
        detail["ignore_error"] = ignoreError;
        detail["buf_semantics_count"] = bufSemantics.size();
        HCCL_VM_RETURN_WITH_ISSUE("step_7_check_op_semantics", "BUF_SEMANTICS_EMPTY", HcclResult::HCCL_E_PARA, detail);
    }

    u64 totalSize = 0;
    BufferSemantic* pre = bufSemantics[0];
    // 头部有空档
    if (pre->startAddr > startAddr) {
        if (!ignoreError) {
            HCCL_ERROR(
                "When check buf semantics in range, there is blank in head."
                "startAddr should be %llu, while first semantic startAddr is %llu",
                startAddr, pre->startAddr);
        } else {
            HCCL_WARNING(
                "When check buf semantics in range, there is blank in head."
                "startAddr should be %llu, while first semantic startAddr is %llu",
                startAddr, pre->startAddr);
        }

        nlohmann::json detail = nlohmann::json::object();
        detail["start_addr"] = startAddr;
        detail["size"] = size;
        detail["first_start_addr"] = pre->startAddr;
        detail["ignore_error"] = ignoreError;
        HCCL_VM_RETURN_WITH_ISSUE(
            "step_7_check_op_semantics", "BUF_SEMANTICS_HEAD_GAP", HcclResult::HCCL_E_PARA, detail);
    }
    if (pre->startAddr + pre->size >= startAddr + size) {
        totalSize += size;
    } else {
        totalSize += pre->startAddr + pre->size - startAddr;
    }

    for (size_t index = 1; index < bufSemantics.size(); index++) {
        BufferSemantic* cur = bufSemantics[index];
        if (cur->startAddr != pre->startAddr + pre->size) {
            if (!ignoreError) {
                HCCL_ERROR(
                    "there is blank in middle, pre semantic endAddr is %llu, cur semantic startAddr is %llu,"
                    "they should be equal",
                    pre->startAddr + pre->size, cur->startAddr);
            } else {
                HCCL_WARNING(
                    "there is blank in middle, pre semantic endAddr is %llu, cur semantic startAddr is %llu,"
                    "they should be equal",
                    pre->startAddr + pre->size, cur->startAddr);
            }

            nlohmann::json detail = nlohmann::json::object();
            detail["pre_end_addr"] = pre->startAddr + pre->size;
            detail["cur_start_addr"] = cur->startAddr;
            detail["ignore_error"] = ignoreError;
            HCCL_VM_RETURN_WITH_ISSUE(
                "step_7_check_op_semantics", "BUF_SEMANTICS_MIDDLE_GAP", HcclResult::HCCL_E_PARA, detail);
        }
        if (cur->startAddr + cur->size >= startAddr + size) {
            totalSize += startAddr + size - cur->startAddr;
        } else {
            totalSize += cur->size;
        }

        pre = cur;
    }

    if (totalSize != size) {
        if (!ignoreError) {
            HCCL_ERROR(
                "When check buf semantics in range, there is blank in tail."
                "endAddr should be %llu, while last semantic endAddr is %llu",
                startAddr + size, startAddr + totalSize);
        } else {
            HCCL_WARNING(
                "When check buf semantics in range, there is blank in tail."
                "endAddr should be %llu, while last semantic endAddr is %llu",
                startAddr + size, startAddr + totalSize);
        }

        nlohmann::json detail = nlohmann::json::object();
        detail["expect_end_addr"] = startAddr + size;
        detail["actual_end_addr"] = startAddr + totalSize;
        detail["ignore_error"] = ignoreError;
        HCCL_VM_RETURN_WITH_ISSUE(
            "step_7_check_op_semantics", "BUF_SEMANTICS_TAIL_GAP", HcclResult::HCCL_E_PARA, detail);
    }

    return HcclResult::HCCL_SUCCESS;
}

// 获取slicePair和srcBufSemantic的交集区域
void TaskCheckOpSemantics::GetSrcIntersectionAddr(
    SliceOpPair& slicePair, const BufferSemantic& srcBufSemantic, u64& srcStartAddr, u64& srcEndAddr) const
{
    srcStartAddr = slicePair.srcSlice.GetOffset();
    srcEndAddr = srcStartAddr + slicePair.srcSlice.GetSize();
    if (srcBufSemantic.startAddr > srcStartAddr) {
        srcStartAddr = srcBufSemantic.startAddr;
    }
    if (srcBufSemantic.startAddr + srcBufSemantic.size < srcEndAddr) {
        srcEndAddr = srcBufSemantic.startAddr + srcBufSemantic.size;
    }
    return;
}

void TaskCheckOpSemantics::GetAffectedBufSemantics(
    SliceOpPair& slicePair, const BufferSemantic& srcBufSemantic,
    std::map<RankId, RankMemorySemantics>& rankMemSemantics, std::vector<BufferSemantic*>& affectedDstBufSemantics)
{
    u64 srcStartAddr;
    u64 srcEndAddr;
    GetSrcIntersectionAddr(slicePair, srcBufSemantic, srcStartAddr, srcEndAddr);

    RankId dstRank = slicePair.dstRank;
    BufferType dstBufType = slicePair.dstSlice.GetType();

    s64 dstSrcOffset = slicePair.dstSlice.GetOffset() - slicePair.srcSlice.GetOffset();
    u64 dstStartAddr = srcStartAddr + dstSrcOffset;
    u64 dstEndAddr = srcEndAddr + dstSrcOffset;

    for (auto& ele : rankMemSemantics[dstRank][dstBufType]) {
        if (ele.startAddr + ele.size <= dstStartAddr) {
            continue;
        }
        if (ele.startAddr >= dstEndAddr) {
            continue;
        }
        affectedDstBufSemantics.push_back(const_cast<BufferSemantic*>(&ele));
    }
    return;
}

void TaskCheckOpSemantics::GetAffectedBufSemantics(
    SliceOpPair& slicePair, std::map<RankId, RankMemorySemantics>& rankMemSemantics,
    std::vector<BufferSemantic*>& affectedDstBufSemantics)
{
    u64 dstStartAddr = slicePair.dstSlice.GetOffset();
    u64 dstEndAddr = dstStartAddr + slicePair.dstSlice.GetSize();

    RankId dstRank = slicePair.dstRank;
    BufferType dstBufType = slicePair.dstSlice.GetType();

    for (auto& ele : rankMemSemantics[dstRank][dstBufType]) {
        if (ele.startAddr + ele.size <= dstStartAddr) {
            continue;
        }
        if (ele.startAddr >= dstEndAddr) {
            continue;
        }
        affectedDstBufSemantics.push_back(const_cast<BufferSemantic*>(&ele));
    }
    return;
}

// 因为srcBufSemantic与affectedDstBufSemantics可能会有重叠，affectedDstBufSemantics处理过程中会被修改，因此srcBufSemantic不能传引用
void TaskCheckOpSemantics::ApplyOverrideSrcBufSemantic(
    SliceOpPair& slicePair, std::map<RankId, RankMemorySemantics>& rankMemSemantics, u32 effectGlobalStep,
    const BufferSemantic srcBufSemantic)
{
    u64 srcStartAddr;
    u64 srcEndAddr;
    GetSrcIntersectionAddr(slicePair, srcBufSemantic, srcStartAddr, srcEndAddr);

    s64 dstSrcOffset = slicePair.dstSlice.GetOffset() - slicePair.srcSlice.GetOffset();
    u64 dstStartAddr = srcStartAddr + dstSrcOffset;
    u64 dstEndAddr = srcEndAddr + dstSrcOffset;

    RankId dstRank = slicePair.dstRank;
    BufferType dstBufType = slicePair.dstSlice.GetType();
    std::set<BufferSemantic>& targetBufferSemantic = rankMemSemantics[dstRank][dstBufType];

    BufferSemantic dstBufSemantic(
        dstStartAddr, dstEndAddr - dstStartAddr, srcBufSemantic.isReduce, srcBufSemantic.reduceType,
        OffsetSrcBufs(srcBufSemantic.srcBufs, srcStartAddr - srcBufSemantic.startAddr));
    dstBufSemantic.affectedGlobalSteps.push_back(effectGlobalStep);

    targetBufferSemantic.insert(dstBufSemantic);

    return;
}

HcclResult TaskCheckOpSemantics::ReduceToAffectedBufSemantic(
    const BufferSemantic& srcBufSemantic, std::vector<BufferSemantic*> toAddReduceInfoSemantics, u64 srcStartAddr)
{
    u64 srcOffset = srcStartAddr - srcBufSemantic.startAddr;
    for (auto& ele : toAddReduceInfoSemantics) {
        if (ele->srcBufs.size() == 1) {
            if (ele->isReduce == true) {
                HCCL_ERROR("buffer semantic srcBufs size is 1, but isReduce is true");
                HCCL_ERROR("invalid buffer semantic is %s", ele->Describe().c_str());
                nlohmann::json detail = nlohmann::json::object();
                detail["src_buf_semantic"] = DumpBufferSemanticToJson(srcBufSemantic, BufferType::RESERVED);
                detail["dst_buf_semantic"] = DumpBufferSemanticToJson(*ele, BufferType::RESERVED);
                HCCL_VM_RETURN_WITH_ISSUE(
                    "step_7_check_op_semantics", "REDUCE_SINGLE_SRC_INCONSISTENT", HcclResult::HCCL_E_PARA, detail);
            }
            ele->isReduce = true;
            ele->reduceType = reduceType_;
        }
        if (srcBufSemantic.srcBufs.size() > 1 && ele->reduceType != srcBufSemantic.reduceType) {
            HCCL_ERROR("reduceType is different");
            HCCL_ERROR("src buf semantic is %s", srcBufSemantic.Describe().c_str());
            HCCL_ERROR("dst buf semantic is %s", ele->Describe().c_str());
            nlohmann::json detail = nlohmann::json::object();
            detail["src_buf_semantic"] = DumpBufferSemanticToJson(srcBufSemantic, BufferType::RESERVED);
            detail["dst_buf_semantic"] = DumpBufferSemanticToJson(*ele, BufferType::RESERVED);
            HCCL_VM_RETURN_WITH_ISSUE(
                "step_7_check_op_semantics", "REDUCE_TYPE_MISMATCH", HcclResult::HCCL_E_PARA, detail);
        }

        const std::set<SrcBufDes> shiftedSrcBufs = OffsetSrcBufs(srcBufSemantic.srcBufs, srcOffset);
        for (const auto& srcBuf : shiftedSrcBufs) {
            // 校验重复reduce的场景
            u32 beforeInsertCnt = ele->srcBufs.size();
            ele->srcBufs.insert(srcBuf);
            u32 afterInsertCnt = ele->srcBufs.size();
            if (beforeInsertCnt == afterInsertCnt) {
                HCCL_ERROR(
                    "after add reduce srcBuf %s, the size of dst srcBufs not changed", srcBuf.Describe().c_str());
                HCCL_ERROR("src buf semantic is %s", srcBufSemantic.Describe().c_str());
                HCCL_ERROR("dst buf semantic is %s", ele->Describe().c_str());
                nlohmann::json detail = nlohmann::json::object();
                detail["src_buf_semantic"] = DumpBufferSemanticToJson(srcBufSemantic, BufferType::RESERVED);
                detail["dst_buf_semantic"] = DumpBufferSemanticToJson(*ele, BufferType::RESERVED);
                detail["src_buf"] = srcBuf.Describe();
                HCCL_VM_RETURN_WITH_ISSUE(
                    "step_7_check_op_semantics", "REDUCE_DUPLICATE_SRC_BUF", HcclResult::HCCL_E_PARA, detail);
            }
        }

        // 用于图形化界面展示，添加影响的节点
        ele->affectedGlobalSteps.insert(
            ele->affectedGlobalSteps.end(), srcBufSemantic.affectedGlobalSteps.begin(),
            srcBufSemantic.affectedGlobalSteps.end());
        srcOffset += ele->size;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskCheckOpSemantics::ApplyReduceSrcBufSemantic(
    SliceOpPair& slicePair, std::map<RankId, RankMemorySemantics>& rankMemSemantics, u32 effectGlobalStep,
    const BufferSemantic& srcBufSemantic, std::vector<BufferSemantic*>& affectedDstBufSemantics)
{
    u64 srcStartAddr;
    u64 srcEndAddr;
    GetSrcIntersectionAddr(slicePair, srcBufSemantic, srcStartAddr, srcEndAddr);

    s64 dstSrcOffset = slicePair.dstSlice.GetOffset() - slicePair.srcSlice.GetOffset();
    u64 dstStartAddr = srcStartAddr + dstSrcOffset;
    u64 dstEndAddr = srcEndAddr + dstSrcOffset;

    // 校验目的地是否已经有了数据
    auto ret = CheckBufSemantics(affectedDstBufSemantics, dstStartAddr, dstEndAddr - dstStartAddr);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR(
            "failed to check dst buf semantics in src semantic range, src buf semantic is %s, affected dst buf "
            "semantics are as follows:",
            srcBufSemantic.Describe().c_str());
        for (auto& ele : affectedDstBufSemantics) {
            HCCL_ERROR("    %s", ele->Describe().c_str());
        }
        nlohmann::json detail = nlohmann::json::object();
        detail["slice_pair"] = slicePair.Describe();
        detail["src_buf_semantic"] = DumpBufferSemanticToJson(srcBufSemantic, BufferType::RESERVED);
        HCCL_VM_RETURN_WITH_ISSUE("step_7_check_op_semantics", "CHECK_DST_BUF_SEMANTICS_FAILED", ret, detail);
    }

    RankId dstRank = slicePair.dstRank;
    BufferType dstBufType = slicePair.dstSlice.GetType();
    // 分割尾节点
    BufferSemantic& lastDstBuf = *(affectedDstBufSemantics.back());
    if (lastDstBuf.startAddr + lastDstBuf.size > dstEndAddr) {
        BufferSemantic dstBufSemantic(
            dstEndAddr, lastDstBuf.startAddr + lastDstBuf.size - dstEndAddr, lastDstBuf.isReduce, lastDstBuf.reduceType,
            OffsetSrcBufs(lastDstBuf.srcBufs, dstEndAddr - lastDstBuf.startAddr));
        dstBufSemantic.affectedGlobalSteps = lastDstBuf.affectedGlobalSteps;

        rankMemSemantics[dstRank][dstBufType].insert(dstBufSemantic);

        // 修改原来的节点
        lastDstBuf.size = dstEndAddr - lastDstBuf.startAddr;
    }

    // 拆分首节点。（必须先新增尾节点，再修改首节点，因为fistDstBuf和lastDstBuf可能指向同一个对象）
    BufferSemantic& fistDstBuf = *(affectedDstBufSemantics[0]);
    if (fistDstBuf.startAddr < dstStartAddr) {
        BufferSemantic dstBufSemantic(
            dstStartAddr, fistDstBuf.startAddr + fistDstBuf.size - dstStartAddr, fistDstBuf.isReduce,
            fistDstBuf.reduceType, OffsetSrcBufs(fistDstBuf.srcBufs, dstStartAddr - fistDstBuf.startAddr));
        dstBufSemantic.affectedGlobalSteps = fistDstBuf.affectedGlobalSteps;

        rankMemSemantics[dstRank][dstBufType].insert(dstBufSemantic);

        fistDstBuf.size = dstStartAddr - fistDstBuf.startAddr;
    }

    std::vector<BufferSemantic*> toAddReduceInfoSemantics;
    GetAffectedBufSemantics(slicePair, srcBufSemantic, rankMemSemantics, toAddReduceInfoSemantics);

    ret = ReduceToAffectedBufSemantic(srcBufSemantic, toAddReduceInfoSemantics, srcStartAddr);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR(
            "reduce to affected buf semantics failed, src buf semantic is %s, affected dst buf semantics are as "
            "follows:",
            srcBufSemantic.Describe().c_str());
        for (auto& ele : toAddReduceInfoSemantics) {
            HCCL_ERROR("    %s", ele->Describe().c_str());
        }
        nlohmann::json detail = nlohmann::json::object();
        detail["slice_pair"] = slicePair.Describe();
        detail["src_buf_semantic"] = DumpBufferSemanticToJson(srcBufSemantic, BufferType::RESERVED);
        HCCL_VM_RETURN_WITH_ISSUE("step_7_check_op_semantics", "REDUCE_TO_AFFECTED_SEMANTICS_FAILED", ret, detail);
    }

    return HcclResult::HCCL_SUCCESS;
}

void TaskCheckOpSemantics::RemoveAffectedBufSemantics(
    SliceOpPair& slicePair, std::map<RankId, RankMemorySemantics>& rankMemSemantics,
    std::vector<BufferSemantic*>& affectedDstBufSemantics)
{
    u64 dstStartAddr = slicePair.dstSlice.GetOffset();
    u64 dstEndAddr = dstStartAddr + slicePair.dstSlice.GetSize();

    RankId dstRank = slicePair.dstRank;
    BufferType dstBufType = slicePair.dstSlice.GetType();
    std::set<BufferSemantic>& targetBufferSemantic = rankMemSemantics[dstRank][dstBufType];

    // 一种特殊情况，对应的dst还未创建起来
    if (affectedDstBufSemantics.size() == 0) {
        return;
    }

    // 新增尾节点
    BufferSemantic& lastDstBuf = *(affectedDstBufSemantics.back());
    if (lastDstBuf.startAddr + lastDstBuf.size > dstEndAddr) {
        BufferSemantic dstBufSemantic(
            dstEndAddr, lastDstBuf.startAddr + lastDstBuf.size - dstEndAddr, lastDstBuf.isReduce, lastDstBuf.reduceType,
            OffsetSrcBufs(lastDstBuf.srcBufs, dstEndAddr - lastDstBuf.startAddr));
        dstBufSemantic.affectedGlobalSteps = lastDstBuf.affectedGlobalSteps;

        targetBufferSemantic.insert(dstBufSemantic);
        // 修改原来的节点
        lastDstBuf.size = dstEndAddr - lastDstBuf.startAddr;
    }

    // 修改首节点。（必须先新增尾节点，再修改首节点，因为fistDstBuf和lastDstBuf可能指向同一个对象）
    BufferSemantic& fistDstBuf = *(affectedDstBufSemantics[0]);
    if (fistDstBuf.startAddr < dstStartAddr) {
        fistDstBuf.size = dstStartAddr - fistDstBuf.startAddr;
    }

    for (auto ele : affectedDstBufSemantics) {
        if (ele->startAddr < dstStartAddr) {
            continue;
        }
        targetBufferSemantic.erase(*ele);
    }

    return;
}

HcclResult TaskCheckOpSemantics::ApplySrcBufSemanticsToDst(
    SliceOpPair& slicePair, std::map<RankId, RankMemorySemantics>& rankMemSemantics,
    std::map<RankId, bool>& memSemanticsChange, u32 effectGlobalStep, std::vector<BufferSemantic*> srcBufSemantics)
{
    SliceOp sliceOp = slicePair.sliceOp;

    if (sliceOp == SliceOp::OVERRIDE) {
        std::vector<BufferSemantic*> affectedDstBufSemantics;
        GetAffectedBufSemantics(slicePair, rankMemSemantics, affectedDstBufSemantics);
        RemoveAffectedBufSemantics(slicePair, rankMemSemantics, affectedDstBufSemantics);

        for (auto& ele : srcBufSemantics) {
            ApplyOverrideSrcBufSemantic(slicePair, rankMemSemantics, effectGlobalStep, *ele);
        }
    } else if (sliceOp == SliceOp::REDUCE) {
        for (auto& ele : srcBufSemantics) {
            std::vector<BufferSemantic*> affectedDstBufSemantics;
            GetAffectedBufSemantics(slicePair, *ele, rankMemSemantics, affectedDstBufSemantics);
            auto ret = ApplyReduceSrcBufSemantic(
                slicePair, rankMemSemantics, effectGlobalStep, *ele, affectedDstBufSemantics);
            if (ret != HcclResult::HCCL_SUCCESS) {
                HCCL_ERROR(
                    "fail to apply reduce for src buf semantic, which is %s, affected dst buf semantics are as "
                    "follows:",
                    ele->Describe().c_str());
                for (auto& ele : affectedDstBufSemantics) {
                    HCCL_ERROR("    %s", ele->Describe().c_str());
                }
                nlohmann::json detail = nlohmann::json::object();
                detail["slice_pair"] = slicePair.Describe();
                detail["src_buf_semantic"] = DumpBufferSemanticToJson(*ele, BufferType::RESERVED);
                HCCL_VM_RETURN_WITH_ISSUE("step_7_check_op_semantics", "APPLY_REDUCE_SRC_SEMANTIC_FAILED", ret, detail);
            }
        }
    }
    memSemanticsChange[slicePair.dstRank] = true;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskCheckOpSemantics::ProcessSliceOpPair(
    SliceOpPair& slicePair, std::map<RankId, RankMemorySemantics>& rankMemSemantics,
    std::map<RankId, bool>& memSemanticsChange, u32 effectGlobalStep)
{
    RankId srcRank = slicePair.srcRank;
    BufferType srcBufType = slicePair.srcSlice.GetType();
    u64 srcStartAddr = slicePair.srcSlice.GetOffset();
    u64 srcSize = slicePair.srcSlice.GetSize();
    if (srcSize == 0) {
        return HcclResult::HCCL_SUCCESS;
    }

    // 根据源slice获取源数据片
    std::vector<BufferSemantic*> srcBufSemantics;
    for (auto& ele : rankMemSemantics[srcRank][srcBufType]) {
        if (ele.startAddr + ele.size <= srcStartAddr) {
            continue;
        }

        if (ele.startAddr >= srcStartAddr + srcSize) {
            continue;
        }

        srcBufSemantics.push_back(const_cast<BufferSemantic*>(&ele));
    }

    auto ret = CheckBufSemantics(srcBufSemantics, srcStartAddr, srcSize, true);
    if (ret != HcclResult::HCCL_SUCCESS) {
        // 对于reduce操作，源slice对应的语义块不能有缺失，因为随机数据做reduce，可能导致概率性溢出问题
        if (slicePair.sliceOp == SliceOp::REDUCE) {
            HCCL_ERROR(
                "failed to check buf semantics in src slice, src slice is %s, bufSemantics in this range are as "
                "follows:",
                slicePair.srcSlice.Describe().c_str());
            for (auto& ele : srcBufSemantics) {
                HCCL_ERROR("    %s", ele->Describe().c_str());
            }

            nlohmann::json detail = nlohmann::json::object();
            detail["slice_op"] = "REDUCE";
            detail["slice_pair"] = slicePair.Describe();
            HCCL_VM_RETURN_WITH_ISSUE(
                "step_7_check_op_semantics", "SRC_SLICE_SEMANTICS_INCOMPLETE_FOR_REDUCE", HCCL_E_PARA, detail);
        } else if (slicePair.sliceOp == SliceOp::OVERRIDE) {
            HCCL_WARNING("incomplete buf semantics in src slice, which may affected performance.");
        }
    }

    // 将源数据片应用到目标数据片
    ret = ApplySrcBufSemanticsToDst(slicePair, rankMemSemantics, memSemanticsChange, effectGlobalStep, srcBufSemantics);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR(
            "failed to apply src buf semantics to dst, src slice is %s, bufSemantics in this range are as follows:",
            slicePair.srcSlice.Describe().c_str());
        for (auto& ele : srcBufSemantics) {
            HCCL_ERROR("    %s", ele->Describe().c_str());
        }
        nlohmann::json detail = nlohmann::json::object();
        detail["slice_pair"] = slicePair.Describe();
        HCCL_VM_RETURN_WITH_ISSUE("step_7_check_op_semantics", "APPLY_SRC_TO_DST_FAILED", ret, detail);
    }
    return HcclResult::HCCL_SUCCESS;
}

void TaskCheckOpSemantics::GetSliceOpPair(TaskNode* simNode, std::vector<SliceOpPair>& sliceOpPairs) const
{
    TaskTypeStub taskType = GetNodeType(simNode);
    if (taskType == TaskTypeStub::LOCAL_COPY) {
        const TaskStubLocalCopy* task = dynamic_cast<const TaskStubLocalCopy*>(simNode->task);
        SliceOpPair pair;
        pair.srcRank = simNode->rankIdx;
        pair.dstRank = simNode->rankIdx;
        pair.srcSlice = task->GetSrcSlice();
        pair.dstSlice = task->GetDstSlice();
        pair.sliceOp = SliceOp::OVERRIDE;
        sliceOpPairs.push_back(pair);
    } else if (taskType == TaskTypeStub::LOCAL_REDUCE) {
        const TaskStubLocalReduce* task = dynamic_cast<const TaskStubLocalReduce*>(simNode->task);
        SliceOpPair pair;
        pair.srcRank = simNode->rankIdx;
        pair.dstRank = simNode->rankIdx;
        pair.srcSlice = task->GetSrcSlice();
        pair.dstSlice = task->GetDstSlice();
        pair.sliceOp = SliceOp::REDUCE;
        sliceOpPairs.push_back(pair);
    } else if (taskType == TaskTypeStub::READ) {
        const TaskStubRead* readTask = dynamic_cast<const TaskStubRead*>(simNode->task);
        SliceOpPair pair;
        pair.srcRank = readTask->GetRemoteRank();
        pair.dstRank = simNode->rankIdx;
        pair.srcSlice = readTask->GetRemoteSlice();
        pair.dstSlice = readTask->GetLocalSlice();
        pair.sliceOp = SliceOp::OVERRIDE;
        sliceOpPairs.push_back(pair);
    } else if (taskType == TaskTypeStub::WRITE) {
        const TaskStubWrite* writeTask = dynamic_cast<const TaskStubWrite*>(simNode->task);
        SliceOpPair pair;
        pair.srcRank = simNode->rankIdx;
        pair.dstRank = writeTask->GetRemoteRank();
        pair.srcSlice = writeTask->GetLocalSlice();
        pair.dstSlice = writeTask->GetRemoteSlice();
        pair.sliceOp = SliceOp::OVERRIDE;
        sliceOpPairs.push_back(pair);
    } else if (taskType == TaskTypeStub::READ_REDUCE) {
        const TaskStubReadReduce* readReduceTask = dynamic_cast<const TaskStubReadReduce*>(simNode->task);
        SliceOpPair pair;
        pair.srcRank = readReduceTask->GetRemoteRank();
        pair.dstRank = simNode->rankIdx;
        pair.srcSlice = readReduceTask->GetRemoteSlice();
        pair.dstSlice = readReduceTask->GetLocalSlice();
        pair.sliceOp = SliceOp::REDUCE;
        sliceOpPairs.push_back(pair);
    } else if (taskType == TaskTypeStub::WRITE_REDUCE) {
        const TaskStubWriteReduce* writeReduceTask = dynamic_cast<const TaskStubWriteReduce*>(simNode->task);
        SliceOpPair pair;
        pair.srcRank = simNode->rankIdx;
        pair.dstRank = writeReduceTask->GetRemoteRank();
        pair.srcSlice = writeReduceTask->GetLocalSlice();
        pair.dstSlice = writeReduceTask->GetRemoteSlice();
        pair.sliceOp = SliceOp::REDUCE;
        sliceOpPairs.push_back(pair);
    } else if (taskType == TaskTypeStub::LOCAL_BATCH_REDUCE) {
        const TaskStubLocalBatchReduce* batchReduceTask = dynamic_cast<const TaskStubLocalBatchReduce*>(simNode->task);
        const std::vector<DataSlice>& srcSlices = batchReduceTask->GetSrcSlices();
        for (u32 i = 0; i < srcSlices.size(); i++) {
            SliceOpPair pair;
            pair.srcRank = simNode->rankIdx;
            pair.dstRank = simNode->rankIdx;
            pair.srcSlice = srcSlices[i];
            pair.dstSlice = batchReduceTask->GetDstSlice();
            pair.sliceOp = SliceOp::REDUCE;
            sliceOpPairs.push_back(pair);
        }
    }
    return;
}

void TaskCheckOpSemantics::UpdateStep(TaskNode* simNode)
{
    RankId rankId = simNode->rankIdx;
    globalStep_++;
    if (localStep_.count(rankId) == 0) {
        localStep_[rankId] = 1;
    } else {
        localStep_[rankId] += 1;
    }

    localStep2GlobalStep_[rankId][localStep_[rankId]] = globalStep_;
    globalStep2LocalStep_[globalStep_] = LocalStep{rankId, localStep_[rankId]};

    simNode->globalStep = globalStep_;
    simNode->localStep = localStep_[rankId];
    return;
}

void TaskCheckOpSemantics::PrepareTimelineContext()
{
    graphNodes_.clear();
    nodeIdMap_.clear();
    timelineEvents_.clear();
    globalStepToEventId_.clear();
    initialRankMemSemantics_.clear();
    rankMemoryTimelines_.clear();

    CollectTaskNodesWithGlobalIds(graphHead_, graphNodes_, nodeIdMap_);

    EnsureRankMemoryEntries(allRankMemSemantics_);
    initialRankMemSemantics_ = allRankMemSemantics_;
}

void TaskCheckOpSemantics::EnsureRankMemoryEntries(std::map<RankId, RankMemorySemantics>& rankMemSemantics) const
{
    for (auto* node : graphNodes_) {
        if (node == nullptr) {
            continue;
        }
        if (rankMemSemantics.count(node->rankIdx) == 0) {
            rankMemSemantics[node->rankIdx] = RankMemorySemantics();
        }
    }
}

HcclResult TaskCheckOpSemantics::RecordNodeSemantics(TaskNode* simNode, const std::vector<SliceOpPair>& sliceOpPairs)
{
    if (simNode == nullptr) {
        return HcclResult::HCCL_SUCCESS;
    }

    TimelineEvent timelineEvent;
    timelineEvent.node = simNode;
    timelineEvent.eventId = static_cast<u32>(timelineEvents_.size()) + 1;
    auto nodeIdIter = nodeIdMap_.find(simNode);
    if (nodeIdIter == nodeIdMap_.end()) {
        nlohmann::json detail = HcclSim::MakeTaskNodeDetail(simNode);
        detail["registered_node_count"] = nodeIdMap_.size();
        HCCL_VM_RETURN_WITH_ISSUE(
            "step_7_check_op_semantics", "TIMELINE_NODE_ID_NOT_REGISTERED", HcclResult::HCCL_E_INTERNAL, detail);
    }
    timelineEvent.nodeId = nodeIdIter->second;
    timelineEvent.simGlobalStep = simNode->globalStep;
    timelineEvent.simLocalStep = simNode->localStep;
    timelineEvent.rankId = simNode->rankIdx;
    timelineEvent.queueId = simNode->queIdx;
    timelineEvent.pos = simNode->pos;
    timelineEvent.taskType = GetNodeType(simNode);
    timelineEvent.sliceOpPairs = sliceOpPairs;
    timelineEvent.isMemoryEvent = !sliceOpPairs.empty();

    std::set<RankId> affectedRanks;
    affectedRanks.insert(simNode->rankIdx);
    for (const auto& sliceOpPair : sliceOpPairs) {
        affectedRanks.insert(sliceOpPair.srcRank);
        affectedRanks.insert(sliceOpPair.dstRank);
    }
    timelineEvent.affectedRanks.assign(affectedRanks.begin(), affectedRanks.end());

    globalStepToEventId_[timelineEvent.simGlobalStep] = timelineEvent.eventId;
    timelineEvents_.push_back(timelineEvent);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskCheckOpSemantics::BuildLogicalSchedule()
{
    std::map<TaskNode*, TimelineEvent*> eventByNode;
    std::map<TaskNode*, u32> pendingParentCnt;
    std::map<QueueKey, u32> queueAvailableStep;
    std::vector<TimelineEvent*> readyEvents;
    readyEvents.reserve(timelineEvents_.size());

    for (auto& timelineEvent : timelineEvents_) {
        eventByNode[timelineEvent.node] = &timelineEvent;
    }

    for (auto& timelineEvent : timelineEvents_) {
        u32 pendingCnt = 0;
        for (auto* parent : timelineEvent.node->parents) {
            if (eventByNode.count(parent) != 0) {
                pendingCnt++;
            }
        }
        pendingParentCnt[timelineEvent.node] = pendingCnt;
        if (pendingCnt == 0) {
            readyEvents.push_back(&timelineEvent);
        }
    }

    u32 scheduledCnt = 0;
    while (!readyEvents.empty()) {
        std::sort(readyEvents.begin(), readyEvents.end(), TimelineNodeLess);
        TimelineEvent* currentEvent = readyEvents.front();
        readyEvents.erase(readyEvents.begin());

        u32 parentEndStep = 0;
        for (auto* parent : currentEvent->node->parents) {
            auto eventIter = eventByNode.find(parent);
            if (eventIter == eventByNode.end()) {
                continue;
            }
            parentEndStep = std::max(parentEndStep, eventIter->second->logicalEndStep);
        }

        QueueKey queueKey = std::make_pair(currentEvent->rankId, currentEvent->queueId);
        u32 queueNextStep = queueAvailableStep[queueKey];
        currentEvent->logicalStartStep = std::max(parentEndStep, queueNextStep);
        const u32 eventCost = currentEvent->isMemoryEvent ? 1 : 0;
        currentEvent->logicalEndStep = currentEvent->logicalStartStep + eventCost;
        queueAvailableStep[queueKey] = currentEvent->logicalEndStep;
        scheduledCnt++;

        for (auto* child : currentEvent->node->children) {
            auto eventIter = eventByNode.find(child);
            if (eventIter == eventByNode.end()) {
                continue;
            }
            pendingParentCnt[child]--;
            if (pendingParentCnt[child] == 0) {
                readyEvents.push_back(eventIter->second);
            }
        }
    }

    if (scheduledCnt != timelineEvents_.size()) {
        HCCL_WARNING(
            "[BuildLogicalSchedule] schedule node count mismatch, scheduled [%u], total [%zu].", scheduledCnt,
            timelineEvents_.size());
        nlohmann::json detail = nlohmann::json::object();
        detail["scheduled_count"] = scheduledCnt;
        detail["total_event_count"] = timelineEvents_.size();
        AppendTimelineRelatedRanks(detail, timelineEvents_);
        HCCL_VM_RETURN_WITH_ISSUE(
            "step_7_check_op_semantics", "TIMELINE_SCHEDULE_COUNT_MISMATCH", HcclResult::HCCL_E_INTERNAL, detail);
    }

    return HcclResult::HCCL_SUCCESS;
}

void TaskCheckOpSemantics::BuildRankTimelineEvents()
{
    rankMemoryTimelines_.clear();
    for (auto* node : graphNodes_) {
        if (node == nullptr) {
            continue;
        }
        RankMemoryTimeline& timeline = rankMemoryTimelines_[node->rankIdx];
        timeline.rankId = node->rankIdx;
        timeline.timelineId = StringFormat("rank_%u_memory_timeline", node->rankIdx);
        timeline.timelineType = "memory_timeline";
    }
}

HcclResult TaskCheckOpSemantics::ReplayTimelineEvents()
{
    std::map<RankId, RankMemorySemantics> replayMemSemantics = initialRankMemSemantics_;
    EnsureRankMemoryEntries(replayMemSemantics);

    std::vector<RankId> rankIds;
    rankIds.reserve(rankMemoryTimelines_.size());
    for (const auto& timelineEntry : rankMemoryTimelines_) {
        rankIds.push_back(timelineEntry.first);
    }

    MemoryTimelineDumpStream dumpStream;
    HcclResult ret = dumpStream.Initialize(rankIds, timelineEvents_, globalStepToEventId_);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_WARNING("[ReplayTimelineEvents] initialize memory timeline dump stream failed.");
        return ret;
    }

    const std::vector<std::string> emptyMemoryTaskIds;
    for (RankId rankId : rankIds) {
        ret = dumpStream.AppendSnapshot(rankId, 0, emptyMemoryTaskIds, replayMemSemantics[rankId]);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_WARNING("[ReplayTimelineEvents] dump init snapshot failed. rank[%u].", rankId);
            return ret;
        }
    }

    std::vector<TimelineEvent*> replayEvents;
    replayEvents.reserve(timelineEvents_.size());
    for (auto& timelineEvent : timelineEvents_) {
        if (timelineEvent.isMemoryEvent) {
            replayEvents.push_back(&timelineEvent);
        }
    }
    std::sort(replayEvents.begin(), replayEvents.end(), TimelineReplayLess);

    size_t replayIndex = 0;
    while (replayIndex < replayEvents.size()) {
        u32 logicalEndStep = replayEvents[replayIndex]->logicalEndStep;
        std::map<RankId, bool> stepChange;
        std::map<RankId, std::vector<std::string>> memoryTaskIdsByRank;

        while (replayIndex < replayEvents.size() && replayEvents[replayIndex]->logicalEndStep == logicalEndStep) {
            TimelineEvent* timelineEvent = replayEvents[replayIndex];
            std::map<RankId, bool> eventChange;
            for (auto& sliceOpPair : timelineEvent->sliceOpPairs) {
                ret = ProcessSliceOpPair(sliceOpPair, replayMemSemantics, eventChange, timelineEvent->simGlobalStep);
                if (ret != HcclResult::HCCL_SUCCESS) {
                    HCCL_WARNING(
                        "[ReplayTimelineEvents] replay slice op failed for event [%u].", timelineEvent->eventId);
                    nlohmann::json detail = nlohmann::json::object();
                    detail["event_id"] = timelineEvent->eventId;
                    detail["slice_pair"] = sliceOpPair.Describe();
                    HcclSim::AppendRelatedSnapshotStep(detail, logicalEndStep);
                    HCCL_VM_RETURN_WITH_ISSUE("step_7_check_op_semantics", "REPLAY_SLICE_OP_FAILED", ret, detail);
                }
            }

            for (const auto& changeEntry : eventChange) {
                if (!changeEntry.second) {
                    continue;
                }
                stepChange[changeEntry.first] = true;
                if (!timelineEvent->nodeId.empty()) {
                    memoryTaskIdsByRank[changeEntry.first].push_back(timelineEvent->nodeId);
                } else {
                    HCCL_ERROR(
                        "[ReplayTimelineEvents] missing task_id for memory event. event_id[%u], rank[%u], "
                        "queue[%u], pos[%u], sim_global_step[%u].",
                        timelineEvent->eventId, timelineEvent->rankId, timelineEvent->queueId, timelineEvent->pos,
                        timelineEvent->simGlobalStep);
                }
            }
            replayIndex++;
        }

        for (const auto& changeEntry : stepChange) {
            if (!changeEntry.second) {
                continue;
            }

            RankId rankId = changeEntry.first;
            ret = dumpStream.AppendSnapshot(
                rankId, logicalEndStep, memoryTaskIdsByRank[rankId], replayMemSemantics[rankId]);
            if (ret != HcclResult::HCCL_SUCCESS) {
                HCCL_WARNING(
                    "[ReplayTimelineEvents] dump snapshot failed. rank[%u], logicalEndStep[%u].", rankId,
                    logicalEndStep);
                return ret;
            }
        }
    }

    return dumpStream.Finalize();
}

HcclResult TaskCheckOpSemantics::GenerateMemoryTimeline()
{
    if (!DumpManager::GetInstance().IsMemorySnapshotEnabled()) {
        return HcclResult::HCCL_SUCCESS;
    }

    if (timelineEvents_.empty()) {
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult ret = BuildLogicalSchedule();
    if (ret != HcclResult::HCCL_SUCCESS) {
        nlohmann::json detail = nlohmann::json::object();
        AppendTimelineRelatedRanks(detail, timelineEvents_);
        HCCL_VM_RETURN_WITH_ISSUE("step_7_check_op_semantics", "BUILD_LOGICAL_SCHEDULE_FAILED", ret, detail);
    }

    BuildRankTimelineEvents();
    ret = ReplayTimelineEvents();
    if (ret != HcclResult::HCCL_SUCCESS) {
        nlohmann::json detail = nlohmann::json::object();
        AppendTimelineRelatedRanks(detail, timelineEvents_);
        HCCL_VM_RETURN_WITH_ISSUE("step_7_check_op_semantics", "REPLAY_TIMELINE_EVENTS_FAILED", ret, detail);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskCheckOpSemantics::ProcessNodeSemantics(TaskNode* simNode)
{
    // 更新localStep与globalStep
    UpdateStep(simNode);

    std::vector<SliceOpPair> sliceOpPairs;
    GetSliceOpPair(simNode, sliceOpPairs);

    HcclResult ret;
    for (auto& ele : sliceOpPairs) {
        ret = ProcessSliceOpPair(ele, allRankMemSemantics_, memSemanticsChange_, globalStep_);
        if (ret != HcclResult::HCCL_SUCCESS) {
            simNode->genSemanticError = true; // 该节点产生语义信息失败，后续在可视化界面进行高亮
            HCCL_ERROR("process SliceOpPair failed,  SliceOpPair is %s", ele.Describe().c_str());
            nlohmann::json detail = HcclSim::MakeTaskNodeDetail(simNode);
            detail["slice_pair"] = ele.Describe();
            HCCL_VM_RETURN_WITH_ISSUE("step_7_check_op_semantics", "PROCESS_SLICE_OP_PAIR_FAILED", ret, detail);
        }
    }

    return RecordNodeSemantics(simNode, sliceOpPairs);
}

void TaskCheckOpSemantics::AddChildrenToQueue(
    TaskNode* node, std::set<TaskNode*>& visitedNodes, std::queue<TaskNode*>& walkQue,
    std::set<TaskNode*>& simulatedNodes) const
{
    node = HcclSim::UpdateNodeForCcuGraph(node, simulatedNodes);
    for (auto& child : node->children) {
        if (visitedNodes.count(child) != 0) {
            continue;
        }
        walkQue.push(child);
        visitedNodes.insert(child);
    }
    return;
}

HcclResult TaskCheckOpSemantics::GenMemSemantics()
{
    std::set<TaskNode*> visitedNodes;
    std::queue<TaskNode*> walkQue;
    std::set<TaskNode*> simulatedNodes;

    // graphHead_是dummy节点，需要忽略掉的
    simulatedNodes.insert(graphHead_);
    for (auto& child : graphHead_->children) {
        visitedNodes.insert(child);
        walkQue.push(child);
    }

    while (!walkQue.empty()) {
        TaskNode* curNode = walkQue.front();
        walkQue.pop();

        // 因为Send/Recv或者SendReduce/RecvReduce必须成对执行，所以队列中有些节点可能已经执行
        if (simulatedNodes.count(curNode) != 0) {
            continue;
        }

        // 父节点还没全部执行完成，不能执行
        if (!IsReadyForSimulate(curNode, simulatedNodes)) {
            walkQue.push(curNode);
            continue;
        }

        auto ret = ProcessNodeSemantics(curNode);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("Process node semantics failed. Node task is %s", curNode->task->Describe().c_str());
            nlohmann::json detail = HcclSim::MakeTaskNodeDetail(curNode);
            detail["op_type"] = static_cast<u32>(opType_);
            HCCL_VM_RETURN_WITH_ISSUE("step_7_check_op_semantics", "PROCESS_NODE_SEMANTICS_FAILED", ret, detail);
        }

        AddChildrenToQueue(curNode, visitedNodes, walkQue, simulatedNodes);
        simulatedNodes.insert(curNode);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskCheckOpSemantics::Execute()
{
    if (opType_ == HcclCMDType::HCCL_CMD_BROADCAST || opType_ == HcclCMDType::HCCL_CMD_SCATTER) {
        InitInputBuffer(root_);
    } else {
        InitInputBuffer();
    }
    PrepareTimelineContext();

    HcclResult ret = HcclResult::HCCL_SUCCESS;
    if (graphHead_->hasAivTask) {
        HCCL_WARNING("[TaskCheckOpSemantics::Execute] not support AIV task");
    } else {
        ret = GenMemSemantics();
    }

    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("Generate memory semantics failed.");
        nlohmann::json detail = nlohmann::json::object();
        detail["op_type"] = static_cast<u32>(opType_);
        detail["rank_size"] = rankSize_;
        std::set<RankId> relatedRanks;
        for (auto* child : graphHead_->children) {
            if (child != nullptr) {
                relatedRanks.insert(child->rankIdx);
            }
        }
        for (RankId rankId : relatedRanks) {
            HcclSim::AppendRelatedRank(detail, rankId);
        }
        HCCL_VM_RETURN_WITH_ISSUE("step_7_check_op_semantics", "GEN_MEM_SEMANTICS_FAILED", ret, detail);
    }
    HCCL_INFO("gen GenMemSemantics success");

    ret = GenerateMemoryTimeline();
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_WARNING("[TaskCheckOpSemantics::Execute] generate memory timeline failed, ignore dump error.");
    }

    if (opType_ == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        ret = TaskCheckAllReduceSemantics(allRankMemSemantics_, dataSize_, reduceType_);
    } else if (opType_ == HcclCMDType::HCCL_CMD_ALLGATHER) {
        ret = TaskCheckAllGatherSemantics(allRankMemSemantics_, dataSize_);
    } else if (opType_ == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        ret = TaskCheckAllGatherVSemantics(allRankMemSemantics_, vDataDes_);
    } else if (opType_ == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        ret = TaskCheckReduceScatterSemantics(allRankMemSemantics_, dataSize_, reduceType_);
    } else if (opType_ == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        ret = TaskCheckReduceScatterVSemantics(allRankMemSemantics_, reduceType_, vDataDes_);
    } else if (opType_ == HcclCMDType::HCCL_CMD_ALLTOALL || opType_ == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        ret = TaskCheckAll2AllSemantics(allRankMemSemantics_, all2AllDataDes_);
    } else if (opType_ == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        HCCL_WARNING("TaskCheckOpSemantics not support HCCL_CMD_ALLTOALLV, please use HCCL_CMD_ALLTOALLVV instead");
    } else if (opType_ == HcclCMDType::HCCL_CMD_SEND || opType_ == HcclCMDType::HCCL_CMD_RECEIVE) {
        ret = TaskCheckSendRecvSemantics(allRankMemSemantics_, dataSize_, srcRank_, dstRank_);
    } else if (opType_ == HcclCMDType::HCCL_CMD_BROADCAST) {
        ret = TaskCheckBroadcastSemantics(allRankMemSemantics_, dataSize_, root_);
    } else if (opType_ == HcclCMDType::HCCL_CMD_REDUCE) {
        ret = TaskCheckReduceSemantics(allRankMemSemantics_, dataSize_, reduceType_, root_);
    } else if (opType_ == HcclCMDType::HCCL_CMD_SCATTER) {
        ret = TaskCheckScatterSemantics(allRankMemSemantics_, dataSize_, root_);
    } else if (opType_ == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        ret = TaskCheckBatchSendRecvRingSemantics(allRankMemSemantics_, rankSize_, dataSize_);
    } else {
        HCCL_ERROR("unsupported op type[%d]", opType_);
        ret = HcclResult::HCCL_E_NOT_SUPPORT;
    }
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("Check memory semantics failed.");
        nlohmann::json detail = nlohmann::json::object();
        detail["op_type"] = static_cast<u32>(opType_);
        detail["rank_size"] = rankSize_;
        for (const auto& entry : allRankMemSemantics_) {
            HcclSim::AppendRelatedRank(detail, entry.first);
        }
        HCCL_VM_RETURN_WITH_ISSUE("step_7_check_op_semantics", "CHECK_MEM_SEMANTICS_FAILED", ret, detail);
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
