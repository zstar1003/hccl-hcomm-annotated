/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_reduce_nhr.h"
#include "alg_template_register.h"

namespace hccl {
AllReduceNHR::AllReduceNHR(const HcclDispatcher dispatcher) : NHRBase(dispatcher) {}

AllReduceNHR::~AllReduceNHR() {}

HcclResult AllReduceNHR::Prepare(u64 reduceAttrBitMap, [[maybe_unused]] HcomCollOpInfo* opInfo)
{
    reduceAttr_ = reduceAttrBitMap;
    return HCCL_SUCCESS;
}

HcclResult AllReduceNHR::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 基本的检查
    CHK_RET(SimpleCheck(rank, rankSize, links));
    HCCL_INFO(
        "[AllReduceNHR][RunAsync] run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    HcclResult ret = HCCL_SUCCESS;
    // 如果ranksize为1, inline reduce和普通跨片reduce操作一致，从input->output
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceNHR][RunAsync] rank[%u] memcpy async failed", rank), ret);
        }

        return ret;
    }

    // reducescatter + allgather
    ret = PrepareRunAsync(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceNHR][RunAsync] rank[%u] count[%llu] "
            "failed in PrepareRunAsync step",
            rank, count_),
        ret);

    // 先执行reducescater
    ret = RunReduceScatter(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceNHR][RunAsync] rank[%u] count[%llu] failed in ReduceScatter "
            "step",
            rank, count_),
        ret);

    // 再执行allgather
    ret = RunAllGather(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceNHR][RunAsync] rank[%u] count[%llu] failed in AllGather "
            "step",
            rank, count_),
        ret);

    HCCL_INFO("[AllReduceNHR][RunAsync] finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceNHR::SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 判断stream, dispatcher是否为空
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());

    // 检查memory
    CHK_PRT_RET(
        !outputMem_ || !inputMem_,
        HCCL_ERROR("[AllReduceNHR][SimpleCheck] rank[%u] inputmem or outputmem is null", rank), HCCL_E_PTR);

    // 判断links数量是否正确
    CHK_PRT_RET(
        links.size() < rankSize,
        HCCL_ERROR(
            "[AllReduceNHR][SimpleCheck] rank[%u] link size[%llu] is less than "
            "rank size[%u]",
            rank, links.size(), rankSize),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult AllReduceNHR::PrepareRunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    (void)links;
    // 计算reducescatter阶段每个rank结果上的offset和size
    if (slices_.size() == 0) {
        slices_.resize(rankSize);
        u64 totalSize = count_ * SIZE_TABLE[dataType_];
        u64 sliceSizeCalculated = (totalSize + (rankSize - 1)) / rankSize;
        u64 sliceSizeAligned = AlgTemplateBase::RoundUpWithDivisor(sliceSizeCalculated, HCCL_MIN_SLICE_ALIGN);

        u64 residueSize = totalSize;

        HCCL_DEBUG(
            "[AllReduceNHR][PrepareRunAsync]residueSize is %llu, sliceSizeAligned is %llu", residueSize,
            sliceSizeAligned);
        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = (residueSize > sliceSizeAligned) ? sliceSizeAligned : residueSize;
            slices_[i].offset = totalSize - residueSize;
            residueSize -= slices_[i].size;
        }

        if (HcclCheckLogLevel(HCCL_LOG_DEBUG)) {
            for (size_t j = 0; j < slices_.size(); j++) {
                HCCL_DEBUG(
                    "[AllReduceNHR][PrepareRunAsync] rank[%u] slice[%u]: offset[%llu] size[%llu]", rank, j,
                    slices_[j].offset, slices_[j].size);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceNHR::RunReduceScatter(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(reduceAttr_, true));
    HCCL_INFO(
        "[AllReduceNHR][RunReduceScatter] rank[%u] tempAlg ReduceScatterNHR inputMem[%p] outputMem[%p] "
        "mem_size[%llu] count[%llu] planeID:[%d]",
        rank, inputMem_.ptr(), outputMem_.ptr(), outputMem_.size(), count_, profilerInput_.planeID);

    if (!barrierSwitchOn_) {
        tempAlg->CloseBarrier();
    }

    CHK_RET(tempAlg->Prepare(
        inputMem_, inputMem_, outputMem_, count_, dataType_, stream_, reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    return tempAlg->RunAsync(rank, rankSize, links);
}

HcclResult AllReduceNHR::RunAllGather(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(true));
    HCCL_INFO(
        "[AllReduceNHR][RunAllGather] rank[%u] tempAlg AllGatherNHR inputMem[%p] outputMem[%p] mem_size[%llu] "
        "count[%llu] planeID:[%d]",
        rank, inputMem_.ptr(), outputMem_.ptr(), outputMem_.size(), count_, profilerInput_.planeID);

    CHK_RET(tempAlg->Prepare(
        inputMem_, outputMem_, outputMem_, count_, dataType_, stream_, reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    return tempAlg->RunAsync(rank, rankSize, links);
}

HcclResult
AllReduceNHR::GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo)
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

    // 先执行ReduceScatter的NHR流程
    for (u32 step = 0; step < nSteps; step++) {
        u32 deltaRank = 1 << step;
        u32 sendTo = (rank + rankSize - deltaRank) % rankSize;
        ;
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
        u32 deltaRank = 1 << (nSteps - 1 - step);
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
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_NHR, AllReduceNHR);
} // namespace hccl
