/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_reduce_nb.h"
#include "alg_template_register.h"

namespace hccl {
AllReduceNB::AllReduceNB(const HcclDispatcher dispatcher) : NBBase(dispatcher) {}

AllReduceNB::~AllReduceNB() {}

HcclResult AllReduceNB::Prepare(u64 reduceAttrBitMap, [[maybe_unused]] HcomCollOpInfo* opInfo)
{
    reduceAttr_ = reduceAttrBitMap;
    return HCCL_SUCCESS;
}

// nb allreduce算法的函数入口
HcclResult AllReduceNB::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    ret = PrepareRunAsync(rank, rankSize, links);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllReduceNB][RunAsync]rank[%u] count[%llu] failed in PrepareRunAsync step", rank, count_), ret);

    CHK_PRT_RET(rankSize == 1, HCCL_INFO("[AllReduceNB][RunAsync] rankSize[%u], do nothing.", rankSize), HCCL_SUCCESS);

    CHK_PRT_RET(count_ == 0, HCCL_INFO("[AllReduceNB][RunAsync] count_[%u], do nothing.", count_), HCCL_SUCCESS);

    // 先执行reducescater
    ret = RunReduceScatter(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceNB][RunAsync]rank[%u] count[%llu] failed in ReduceScatter "
            "step",
            rank, count_),
        ret);

    // 再执行allgather
    ret = RunAllGather(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceNB][RunAsync]rank[%u] count[%llu] failed in AllGather "
            "step",
            rank, count_),
        ret);

    HCCL_INFO("AllReduceNB finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult
AllReduceNB::RunAsyncStaged(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, RunStage stage)
{
    CHK_PRT_RET(
        rankSize == 1 && stage != RunStage::RUN_PREPARE,
        HCCL_INFO("[AllReduceNB][RunAsyncStaged] rankSize[%u], stage[%d], do nothing.", rankSize, stage), HCCL_SUCCESS);

    HcclResult ret = HCCL_SUCCESS;
    switch (stage) {
        case RunStage::RUN_PREPARE:
            ret = PrepareRunAsync(rank, rankSize, links);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[AllReduceNB][RunAsyncStaged]rank[%u] count[%llu] failed in PrepareRunAsync step", rank, count_),
                ret);
            break;
        case RunStage::RUN_REDUCE_SCATTER:
            // 先执行reducescater
            ret = RunReduceScatter(rank, rankSize, links);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[AllReduceNB][RunAsyncStaged]rank[%u] count[%llu] "
                    "failed in ReduceScatter step",
                    rank, count_),
                ret);
            break;
        case RunStage::RUN_ALLGATHER:
            // 再执行AllGather
            ret = RunAllGather(rank, rankSize, links);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[AllReduceNB][RunAsyncStaged]rank[%u] count[%llu] "
                    "failed in AllGather step",
                    rank, count_),
                ret);
            break;
        default:
            HCCL_ERROR("[AllReduceNB][RunAsyncStaged]stage[%d] is not supported", stage);
            return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO("AllReduceNB RunAsyncStaged stage[%d] finished: rank[%u] ranksize[%u]", stage, rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceNB::PrepareRunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR("[AllReduceNB][RunAsync]rank[%u] run_async inputmem or outputmem is null", rank);
        return HCCL_E_PTR;
    }
    HCCL_INFO(
        "AllReduceNB run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    if (links.size() < rankSize) {
        HCCL_ERROR(
            "[AllReduceNB][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]", rank, links.size(), rankSize);
        return HCCL_E_INTERNAL;
    }

    // 如果ranksize为1, inline reduce和普通跨片reduce操作一致，从input->output
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceNB][RunAsync]rank[%u] memcpy async failed", rank), ret);
        }

        return ret;
    }
    // 计算reducescatter 阶段每个rank结果上的offset和size
    if (slices_.size() == 0) {
        slices_.resize(rankSize);
        const u64 totalSize = count_ * SIZE_TABLE[dataType_];
        const u64 sliceSizeAligned = GetSliceSizeOfNB(totalSize, rankSize);
        u64 residueSize = totalSize;

        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = (residueSize > sliceSizeAligned) ? sliceSizeAligned : residueSize;
            slices_[i].offset = totalSize - residueSize;
            residueSize -= slices_[i].size;
        }

        if (HcclCheckLogLevel(HCCL_LOG_DEBUG)) {
            for (size_t j = 0; j < slices_.size(); j++) {
                HCCL_DEBUG("rank[%u] slice[%u]: offset[%llu] size[%llu]", rank, j, slices_[j].offset, slices_[j].size);
            }
        }
    }
    HCCL_INFO("AllReduceNB PrepareRunAsync finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceNB::RunReduceScatter(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    // 调用ReduceScatterNB算法
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(reduceAttr_));
    HCCL_INFO(
        "rank[%u] tempAlg ReduceScatterNB inputMem[%p] outputMem[%p] mem_size[%llu] "
        "count[%llu] planeID:[%d]",
        rank, inputMem_.ptr(), outputMem_.ptr(), outputMem_.size(), count_, profilerInput_.planeID);
    tempAlg->CloseBarrier();
    CHK_RET(tempAlg->Prepare(
        inputMem_, inputMem_, outputMem_, count_, dataType_, stream_, reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    return tempAlg->RunAsync(rank, rankSize, links);
}

HcclResult AllReduceNB::RunAllGather(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    HCCL_INFO(
        "rank[%u] tempAlg AllGatherNB inputMem[%p] outputMem[%p] mem_size[%llu] "
        "count[%llu] planeID:[%d]",
        rank, inputMem_.ptr(), outputMem_.ptr(), outputMem_.size(), count_, profilerInput_.planeID);
    // 判断是否关闭allgather的barrier
    tempAlg->CloseBarrier();

    // 调用allgatherring的算法执行
    CHK_RET(tempAlg->Prepare(
        inputMem_, outputMem_, outputMem_, count_, dataType_, stream_, reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    return tempAlg->RunAsync(rank, rankSize, links);
}

u64 GetSliceSizeOfNB(const u64 dataSize, const u32 rankSize)
{
    const u64 sliceSizeCalculated = (dataSize + (rankSize - 1)) / rankSize;
    u64 sliceSizeAligned = 0;

    // 优化小包性能，小于128k不切片
    if (sliceSizeCalculated > NB_ALLREDUCE_SMALL_SIZE) {
        sliceSizeAligned = AlgTemplateBase::RoundUpWithDivisor(sliceSizeCalculated, HCCL_MIN_SLICE_ALIGN);
    } else {
        sliceSizeAligned = AlgTemplateBase::RoundUpWithDivisor(sliceSizeCalculated, NB_ALLREDUCE_SMALL_SIZE);
    }
    HCCL_INFO(
        "dataSize[%llu], rankSize[%u], sliceSizeCalculated[%llu], sliceSizeAligned[%llu]", dataSize, rankSize,
        sliceSizeCalculated, sliceSizeAligned);

    return sliceSizeAligned;
}

HcclResult
AllReduceNB::GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo)
{
    if (rankSize == 1) {
        return HCCL_SUCCESS;
    }
    if (links.size() < rankSize) {
        return HCCL_SUCCESS;
    }
    u32 nSteps = 0;
    for (u32 temp = rankSize - 1; temp != 0; temp >>= 1, ++nSteps) {
    }

    // 先执行ReduceScatter的NB流程
    for (u32 step = 0; step < nSteps; step++) {
        u32 deltaRank = 1 << step;
        u32 sendTo = (rank + deltaRank) % rankSize;
        LINK linkRight = links[sendTo];
        CHK_SMART_PTR_NULL(linkRight);
        NslbDpAdjInfo adjInfoStep = {};
        adjInfoStep.dstLocalRankId = linkRight->GetRemoteRank();
        adjInfoStep.phaseId = step + 1;
        adjInfoStep.rev = 0;
        nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
    }
    u32 begin = nSteps;
    // 后续执行AllGather的NB流程
    for (u32 step = 0; step < nSteps; step++) {
        u32 deltaRank = 1 << step;
        u32 sendTo = (rank + deltaRank) % rankSize;
        LINK linkRight = links[sendTo];
        CHK_SMART_PTR_NULL(linkRight);
        NslbDpAdjInfo allGatherInfoStep = {};
        allGatherInfoStep.dstLocalRankId = linkRight->GetRemoteRank();
        allGatherInfoStep.phaseId = step + begin + 1;
        allGatherInfoStep.rev = 0;
        nslbAdjInfo.nsAdjInfo.push_back(allGatherInfoStep);
    }
    nslbAdjInfo.dstRankNum = nslbAdjInfo.nsAdjInfo.size();
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_NB, AllReduceNB);
} // namespace hccl
