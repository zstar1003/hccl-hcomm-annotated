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
#include "all_reduce_nhr_v1.h"

namespace hccl {
AllReduceNHRV1::AllReduceNHRV1(const HcclDispatcher dispatcher) : NHRV1Base(dispatcher) {}

AllReduceNHRV1::~AllReduceNHRV1() {}

HcclResult AllReduceNHRV1::Prepare(u64 reduceAttrBitMap, [[maybe_unused]] HcomCollOpInfo* opInfo)
{
    reduceAttr_ = reduceAttrBitMap;
    return HCCL_SUCCESS;
}

HcclResult
AllReduceNHRV1::RunAsync(const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport>>& links)
{
    CHK_RET(PrepareRunAsync(rank, rankSize, links));
    CHK_PRT_RET(
        rankSize == 1, HCCL_INFO("[AllReduceNHRV1][RunAsync] rankSize[%u], do nothing.", rankSize), HCCL_SUCCESS);

    HcclResult ret = HCCL_SUCCESS;
    // 获取通信关系
    RingInfo info = GetRingInfo(rankSize);
    // 水平方向做broken reducescatter ring

    ret = RunReduceScatterOnHorizontal(rank, links, info);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceNHRV1][RunAsync]rank[%u] count[%llu] failed in "
            "RunReduceScatterOnHorizontal step",
            rank, count_),
        ret);

    // 垂直方向做allreduce ring
    ret = RunAllReduceOnVertical(rank, links, info);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceNHRV1][RunAsync]rank[%u] count[%llu] failed in "
            "RunAllReduceOnVertical step",
            rank, count_),
        ret);

    // 水平方向做broken allgather ring
    ret = RunAllGatherOnHorizontal(rank, links, info);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceNHRV1][RunAsync]rank[%u] count[%llu] failed in "
            "RunAllGatherOnHorizontal step",
            rank, count_),
        ret);

    HCCL_INFO("AllReduceNHRV1 finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

HcclResult
AllReduceNHRV1::RunAsyncStaged(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, RunStage stage)
{
    CHK_PRT_RET(
        rankSize == 1 && stage != RunStage::RUN_PREPARE,
        HCCL_INFO("[AllReduceNHRV1][RunAsyncStaged] rankSize[%u], stage[%d], do nothing.", rankSize, stage),
        HCCL_SUCCESS);
    // 获取通信关系
    RingInfo info = GetRingInfo(rankSize);

    HcclResult ret = HCCL_SUCCESS;
    switch (stage) {
        case RunStage::RUN_PREPARE:
            ret = PrepareRunAsync(rank, rankSize, links);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[AllReduceNHRV1][RunAsyncStaged]rank[%u] count[%llu] failed in PrepareRunAsync step", rank,
                    count_),
                ret);
            break;
        case RunStage::RUN_REDUCE_SCATTER:
            // 水平方向做broken reducescatter ring
            ret = RunReduceScatterOnHorizontal(rank, links, info);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[AllReduceNHRV1][RunAsync]rank[%u] count[%llu] failed in "
                    "RunReduceScatterOnHorizontal step",
                    rank, count_),
                ret);
            break;
        case RunStage::RUN_ALLREDUCE:
            // 垂直方向做allreduce ring
            ret = RunAllReduceOnVertical(rank, links, info);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[AllReduceNHRV1][RunAsync]rank[%u] count[%llu] failed in "
                    "RunAllReduceOnVertical step",
                    rank, count_),
                ret);
            break;
        case RunStage::RUN_ALLGATHER:
            // 水平方向做broken allgather ring
            ret = RunAllGatherOnHorizontal(rank, links, info);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[AllReduceNHRV1][RunAsync]rank[%u] count[%llu] failed in "
                    "RunAllGatherOnHorizontal step",
                    rank, count_),
                ret);
            break;
        default:
            HCCL_ERROR("[AllReduceNHRV1][RunAsyncStaged]stage[%d] is not supported", stage);
            return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO("AllReduceNHRV1 RunAsyncStaged stage[%d] finished: rank[%u] ranksize[%u]", stage, rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceNHRV1::PrepareRunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    CHK_PRT_RET(
        !outputMem_ || !inputMem_,
        HCCL_ERROR("[AllReduceNHRV1][RunAsync]rank[%u] run_async inputmem or outputmem is null", rank), HCCL_E_PTR);

    HCCL_INFO(
        "AllReduceNHRV1 run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    CHK_PRT_RET(
        links.size() < rankSize,
        HCCL_ERROR(
            "[AllReduceNHRV1][RunAsync]rank[%u] linksize[%llu] is less "
            "than rankSize[%u]",
            rank, links.size(), rankSize),
        HCCL_E_INTERNAL);

    // 如果ranksize为1, inline reduce和普通跨片reduce操作一致，从input->output
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceNHRV1][RunAsync]rank[%u] memcpy async failed", rank), ret);
        }
        return ret;
    }

    // 检查、并清空slices_
    if (slices_.size() != 0) {
        HCCL_WARNING("[AllReduceNHRV1][RunAsync]AllReduceNHRV1 not supported passing in parameter slice_, "
                     "otherwise will be cleared");
        slices_.clear();
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceNHRV1::CalcHSlicesAndLinks(
    const u32 rank, const std::vector<LINK>& links, const RingInfo& info, std::vector<LINK>& hLinks,
    std::vector<Slice>& hSlices) const
{
    u32 ringSize = info.GetHSizeByRank(rank); // 查找自己所处的行长度，也即Ring的大小
    u32 vIndex = info.GetVIndex(rank);        // 查找自己位于第几行

    // 计算水平方向每个rank结果上的offset和size
    u64 sliceSizeCalculated = (count_ + (info.GetRowSize() - 1)) / info.GetRowSize() * DataUnitSize(dataType_);
    u64 totalSize = count_ * DataUnitSize(dataType_);
    u64 residueSize = totalSize;
    u64 sliceSizeAligned = AlgTemplateBase::RoundUpWithDivisor(sliceSizeCalculated, HCCL_MIN_SLICE_ALIGN);

    // 水平方向都为broken ring，故最后一列有可能不需要参与计算，此时size为0
    for (u32 hIdx = 0; hIdx < ringSize; hIdx++) {
        u32 oldRank = info.GetRank(vIndex, hIdx);

        CHK_PRT_RET(
            oldRank >= links.size(),
            HCCL_ERROR(
                "[AllReduceNHRV1] rank[%u] out of range, "
                "oldRank=%u, links.size=%u",
                rank, oldRank, links.size()),
            HCCL_E_INTERNAL);
        hLinks.push_back(links[oldRank]);
        Slice slice;
        if (info.GetVSizeByHIndex(hIdx) == info.GetVSizeByHIndex(0)) {
            slice.size = (residueSize > sliceSizeAligned) ? sliceSizeAligned : residueSize;
            slice.offset = totalSize - residueSize;
            residueSize -= slice.size;
        } else {
            slice.size = 0;
            slice.offset = 0;
        }
        HCCL_DEBUG(
            "[AllReduceNHRV1][CalcHSlicesAndLinks] rank[%u], slices[%u].offset=%llu, slices[%u].size=%llu", rank, hIdx,
            slice.offset, hIdx, slice.size);
        hSlices.push_back(slice);
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceNHRV1::CalcVSlicesAndLinks(
    const u32 rank, const std::vector<LINK>& links, const RingInfo& info, std::vector<LINK>& vLinks,
    std::vector<Slice>& vSlices)
{
    u32 ringSize = info.GetVSizeByRank(rank); // 查找自己所处的列长度，也即Ring的大小
    u32 hIndex = info.GetHIndex(rank);        // 查找自己位于第几列

    std::vector<Slice> hSlices;
    std::vector<LINK> hLinks;
    CHK_RET(CalcHSlicesAndLinks(rank, links, info, hLinks, hSlices));

    // 计算垂直方向每个rank结果上的offset和size
    u64 sliceSizeCalculated
        = (hSlices[hIndex].size / DataUnitSize(dataType_) + (ringSize - 1)) / ringSize * DataUnitSize(dataType_);
    u64 totalSize = hSlices[hIndex].size;
    u64 residueSize = totalSize;
    u64 sliceSizeAligned = AlgTemplateBase::RoundUpWithDivisor(sliceSizeCalculated, HCCL_MIN_SLICE_ALIGN);

    for (u32 vIdx = 0; vIdx < ringSize; vIdx++) {
        u32 oldRank = info.GetRank(vIdx, hIndex);
        CHK_PRT_RET(
            oldRank >= links.size(),
            HCCL_ERROR(
                "[AllReduceNHRV1] rank[%u] out of range, "
                "oldRank=%u, links.size=%u",
                rank, oldRank, links.size()),
            HCCL_E_INTERNAL);
        vLinks.push_back(links[oldRank]);
        Slice slice;
        slice.size = (residueSize > sliceSizeAligned) ? sliceSizeAligned : residueSize;
        slice.offset = hSlices[hIndex].offset + totalSize - residueSize;
        residueSize -= slice.size;
        HCCL_DEBUG(
            "[AllReduceNHRV1][CalcVSlicesAndLinks] rank[%u], slices[%u].offset=%llu, slices[%u].size=%llu", rank, vIdx,
            slice.offset, vIdx, slice.size);
        vSlices.push_back(slice);
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceNHRV1::RunReduceScatterOnHorizontal(u32 rank, const std::vector<LINK>& links, const RingInfo& info)
{
    u32 ringRank = info.GetHIndex(rank); // 查找自己位于第几列，也即处于Ring中的第几个rank

    // 计算reducescatter每个rank结果上的offset和size
    std::vector<Slice> hSlices;
    std::vector<LINK> hLinks;
    CHK_RET(CalcHSlicesAndLinks(rank, links, info, hLinks, hSlices));

    // 长度不足2，直接跳过
    if (hLinks.size() < 2) {
        return HCCL_SUCCESS;
    }

    HCCL_DEBUG("[AllReduceNHRV1][ReduceScatter-H] rank[%u] ringRank=%u, ringSize=%u", rank, ringRank, hLinks.size());
    return RunReduceScatterBrokenRing(ringRank, hLinks, hSlices);
}

HcclResult AllReduceNHRV1::RunAllReduceOnVertical(const u32 rank, const std::vector<LINK>& links, const RingInfo& info)
{
    u32 ringRank = info.GetVIndex(rank);      // 查找自己位于第几行，也即处于Ring中的第几个rank
    u32 ringSize = info.GetVSizeByRank(rank); // 查找自己所处的列长度，也即Ring的大小
    // 若最后一列不完整，则不做allreduce操作直接返回success
    if (ringSize < info.GetVSizeByHIndex(0)) {
        return HCCL_SUCCESS;
    }
    // 计算allreduce 阶段每个rank结果上的offset和size
    std::vector<Slice> vSlices;
    std::vector<LINK> vLinks;
    CHK_RET(CalcVSlicesAndLinks(rank, links, info, vLinks, vSlices));

    std::unique_ptr<AlgTemplateBase> tempAlg;
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(reduceAttr_));

    // 判断是否关闭allreduce的barrier
    if (!barrierSwitchOn_) {
        tempAlg->CloseBarrier();
    }

    CHK_RET(tempAlg->Prepare(
        inputMem_, outputMem_, outputMem_, count_, dataType_, stream_, reductionOp_, root_, vSlices, baseOffset_));

    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    HCCL_DEBUG("[AllReduceNHRV1][AllReduce-V] rank[%u] ringRank=%u, ringSize=%u", rank, ringRank, ringSize);
    return tempAlg->RunAsync(ringRank, ringSize, vLinks);
}

HcclResult AllReduceNHRV1::RunAllGatherOnHorizontal(u32 rank, const std::vector<LINK>& links, const RingInfo& info)
{
    u32 ringRank = info.GetHIndex(rank); // 查找自己位于第几列，也即处于Ring中的第几个rank

    // 计算allgather阶段每个rank结果上的offset和size
    std::vector<Slice> hSlices;
    std::vector<LINK> hLinks;
    CHK_RET(CalcHSlicesAndLinks(rank, links, info, hLinks, hSlices));

    // 长度不足2，直接跳过
    if (hLinks.size() < 2)
        return HCCL_SUCCESS;

    HCCL_DEBUG("[AllReduceNHRV1][AllGather-H] rank[%u] ringRank=%u, ringSize=%u", rank, ringRank, hLinks.size());
    return RunAllGatherBrokenRing(ringRank, hLinks, hSlices);
}

HcclResult AllReduceNHRV1::RunReduceScatterBrokenRing(
    const u32 rank, const std::vector<LINK>& links, const std::vector<Slice>& slices)
{
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(reduceAttr_));

    // 判断是否关闭reducescatter的barrier
    if (!barrierSwitchOn_) {
        tempAlg->CloseBarrier();
    }

    // 调用reducescatter ring的算法执行
    CHK_RET(tempAlg->Prepare(
        inputMem_, inputMem_, outputMem_, count_, dataType_, stream_, reductionOp_, root_, slices, baseOffset_));

    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    return tempAlg->RunAsync(rank, links.size(), links);
}

HcclResult
AllReduceNHRV1::RunAllGatherBrokenRing(const u32 rank, const std::vector<LINK>& links, const std::vector<Slice>& slices)
{
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    // 判断是否关闭allgather的barrier
    if (!barrierSwitchOn_) {
        tempAlg->CloseBarrier();
    }

    // 调用allgather ring的算法执行
    CHK_RET(tempAlg->Prepare(
        outputMem_, outputMem_, outputMem_, count_, dataType_, stream_, reductionOp_, root_, slices, baseOffset_));

    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    return tempAlg->RunAsync(rank, links.size(), links);
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_NHR_V1, AllReduceNHRV1);
} // namespace hccl
