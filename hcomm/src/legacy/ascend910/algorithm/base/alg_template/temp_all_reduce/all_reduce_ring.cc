/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_register.h"
#include "all_reduce_ring.h"

namespace hccl {
AllReduceRing::AllReduceRing(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllReduceRing::~AllReduceRing() {}

HcclResult AllReduceRing::Prepare(u64 reduceAttrBitMap, [[maybe_unused]] HcomCollOpInfo* opInfo)
{
    reduceAttr_ = reduceAttrBitMap;
    return HCCL_SUCCESS;
}

// ring allreduce算法的函数入口
HcclResult AllReduceRing::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    ret = PrepareRunAsync(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceRing][RunAsync]rank[%u] count[%llu] "
            "failed in PrepareRunAsync step",
            rank, count_),
        ret);
    CHK_PRT_RET(
        rankSize == 1, HCCL_INFO("[AllReduceRing][RunAsync] rankSize[%u], do nothing.", rankSize), HCCL_SUCCESS);

    // 先执行reducescater
    ret = RunReduceScatter(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceRing][RunAsync]rank[%u] count[%llu] failed in ReduceScatter "
            "step",
            rank, count_),
        ret);

    // 再执行allgather
    ret = RunAllGather(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceRing][RunAsync]rank[%u] count[%llu] failed in AllGather "
            "step",
            rank, count_),
        ret);

    HCCL_INFO("AllReduceRing finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult
AllReduceRing::RunAsyncStaged(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, RunStage stage)
{
    CHK_PRT_RET(
        rankSize == 1 && stage != RunStage::RUN_PREPARE,
        HCCL_INFO("[AllReduceRing][RunAsyncStaged] rankSize[%u], stage[%d], do nothing.", rankSize, stage),
        HCCL_SUCCESS);

    HcclResult ret = HCCL_SUCCESS;
    switch (stage) {
        case RunStage::RUN_PREPARE:
            ret = PrepareRunAsync(rank, rankSize, links);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[AllReduceRing][RunAsyncStaged]rank[%u] count[%llu] failed in PrepareRunAsync step", rank, count_),
                ret);
            break;
        case RunStage::RUN_REDUCE_SCATTER:
            // 先执行reducescater
            ret = RunReduceScatter(rank, rankSize, links, true);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[AllReduceRing][RunAsyncStaged]rank[%u] count[%llu] "
                    "failed in ReduceScatter step",
                    rank, count_),
                ret);
            break;
        case RunStage::RUN_ALLGATHER:
            // 再执行allgather
            ret = RunAllGather(rank, rankSize, links);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[AllReduceRing][RunAsyncStaged]rank[%u] count[%llu] "
                    "failed in AllGather step",
                    rank, count_),
                ret);
            break;
        default:
            HCCL_ERROR("[AllReduceRing][RunAsyncStaged]stage[%d] is not supported", stage);
            return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO("AllReduceRing RunAsyncStaged stage[%d] finished: rank[%u] ranksize[%u]", stage, rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceRing::PrepareRunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR("[AllReduceRing][RunAsync]rank[%u] run_async inputmem or outputmem is null", rank);
        return HCCL_E_PTR;
    }
    HCCL_INFO(
        "AllReduceRing run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    if (links.size() < rankSize) {
        HCCL_ERROR(
            "[AllReduceRing][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]", rank, links.size(), rankSize);
        return HCCL_E_INTERNAL;
    }

    // 如果ranksize为1, inline reduce和普通跨片reduce操作一致，从input->output
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceRing][RunAsync]rank[%u] memcpy async failed", rank), ret);
        }

        return ret;
    }
    // 计算reducescatter 阶段每个rank结果上的offset和size
    if (slices_.size() == 0) {
        slices_.resize(rankSize);
        u64 totalSize = count_ * SIZE_TABLE[dataType_];
        u64 sliceSizeCalculated = (totalSize + (rankSize - 1)) / rankSize;
        u64 sliceSizeAligned = AlgTemplateBase::RoundUpWithDivisor(sliceSizeCalculated, HCCL_MIN_SLICE_ALIGN);
        u64 residueSize = totalSize;

        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = (residueSize > sliceSizeAligned) ? sliceSizeAligned : residueSize;
            slices_[i].offset = totalSize - residueSize;
            residueSize -= slices_[i].size;
        }

        if (HcclCheckLogLevel(DLOG_DEBUG)) {
            for (size_t j = 0; j < slices_.size(); j++) {
                HCCL_DEBUG("rank[%u] slice[%u]: offset[%llu] size[%llu]", rank, j, slices_[j].offset, slices_[j].size);
            }
        }
    }
    HCCL_INFO("AllReduceRing PrepareRunAsync finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceRing::RunReduceScatter(u32 rank, u32 rankSize, const std::vector<LINK>& links, bool needBarrier)
{
    if (links.size() < rankSize) {
        HCCL_ERROR(
            "[AllReduceRing][RunReduceScatter]rank[%u] linkSize[%llu] is less than rankSize[%u]", rank, links.size(),
            rankSize);
        return HCCL_E_INTERNAL;
    }

    // 调用reducescattering算法
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    tempAlg->Prepare(reduceAttr_);
    HCCL_INFO(
        "rank[%u] tempAlg ReduceScattering inputMem[%p] outputMem[%p] mem_size[%llu] "
        "count[%llu] planeID:[%d]",
        rank, inputMem_.ptr(), outputMem_.ptr(), outputMem_.size(), count_, profilerInput_.planeID);
    if (!needBarrier) {
        tempAlg->CloseBarrier();
    }
    CHK_RET(tempAlg->Prepare(
        inputMem_, inputMem_, outputMem_, count_, dataType_, stream_, reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    HcclResult ret = tempAlg->RunAsync(rank, rankSize, links);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceRing::RunAllGather(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    HCCL_INFO(
        "rank[%u] tempAlg AllGathering inputMem[%p] outputMem[%p] mem_size[%llu] "
        "count[%llu] planeID:[%d]",
        rank, inputMem_.ptr(), outputMem_.ptr(), outputMem_.size(), count_, profilerInput_.planeID);
    // 判断是否关闭allgather的barrier
    if (!barrierSwitchOn_) {
        tempAlg->CloseBarrier();
    }
    // 调用allgatherring的算法执行
    CHK_RET(tempAlg->Prepare(
        inputMem_, outputMem_, outputMem_, count_, dataType_, stream_, reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    HcclResult ret = tempAlg->RunAsync(rank, rankSize, links);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    return HCCL_SUCCESS;
}
HcclResult
AllReduceRing::GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo)
{
    u32 ringNextRank = (rank + 1) % rankSize;
    LINK nslbNext = links[ringNextRank];
    CHK_SMART_PTR_NULL(nslbNext);
    // reduce_scatter 的ring
    NslbDpAdjInfo adjInfoStep = {};
    nslbAdjInfo.dstRankNum = 1;
    adjInfoStep.dstLocalRankId = nslbNext->GetRemoteRank();
    adjInfoStep.phaseId = 1;
    adjInfoStep.rev = 0;
    nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);

    LINK right = links[ringNextRank];
    CHK_SMART_PTR_NULL(right);
    // all_gather 的ring
    NslbDpAdjInfo nextAdjInfo = {};
    nslbAdjInfo.dstRankNum = nslbAdjInfo.dstRankNum + 1;
    nextAdjInfo.dstLocalRankId = right->GetRemoteRank();
    nextAdjInfo.phaseId = nslbAdjInfo.nsAdjInfo[0].phaseId + 1;
    nextAdjInfo.rev = 0;
    nslbAdjInfo.nsAdjInfo.push_back(nextAdjInfo);
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_RING, AllReduceRing);
} // namespace hccl
