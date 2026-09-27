/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_nhr_v1.h"
#include "alg_template_register.h"

namespace hccl {

BroadcastNHRV1::BroadcastNHRV1(const HcclDispatcher dispatcher) : NHRV1Base(dispatcher) {}

BroadcastNHRV1::~BroadcastNHRV1() {}

HcclResult BroadcastNHRV1::Prepare(PrepareData& param)
{
    return AlgTemplateBase::Prepare(
        param.inputMem, param.outputMem, param.scratchMem, param.count, param.dataType, param.stream,
        HCCL_REDUCE_RESERVED, param.root, std::vector<Slice>(ZERO_SLICE), param.baseOffset);
}

HcclResult BroadcastNHRV1::RunScatterOnHorizontal(const u32 rank, const std::vector<LINK>& links, const RingInfo& info)
{
    // 只有root节点所在的水平Ring做Scatter
    u32 rootVIndex = info.GetVIndex(root_);
    u32 vIndex = info.GetVIndex(rank);
    if (rootVIndex != vIndex) {
        return HCCL_SUCCESS;
    }

    // 收集link
    u32 hSize = info.GetHSizeByVIndex(vIndex);
    std::vector<LINK> subLinks(hSize);
    for (u32 hIdx = 0; hIdx < hSize; hIdx++) {
        u32 rankInRing = info.GetRank(vIndex, hIdx);
        CHK_PRT_RET(
            rankInRing >= links.size(),
            HCCL_ERROR(
                "[BroadcastNHRV1][Scatter-H] rank[%u] out of range, "
                "rankInRing=%u, links.size=%u",
                rank, rankInRing, links.size()),
            HCCL_E_INTERNAL);
        subLinks[hIdx] = links[rankInRing];
        HCCL_DEBUG("[BroadcastNHRV1][Scatter-H] rank[%u], ringRank[%u]=%u", rank, hIdx, rankInRing);
    }

    // 计算新的rank和root
    u32 subRank = info.GetHIndex(rank);
    u32 subRoot = info.GetHIndex(root_);
    HCCL_DEBUG("[BroadcastNHRV1][Scatter-H] rank[%u] subRank=%u, subRoot=%u", rank, subRank, subRoot);

    // 执行Ring - Scatter
    // 此处Prepare的baseOffset给0，因为偏移量已经在加在slices里面
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_RING, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    if (!barrierSwitchOn_) {
        tempAlg->CloseBarrier();
    }
    CHK_RET(tempAlg->Prepare(scratch_, scratch_, scratch_, -1, dataType_, stream_, reductionOp_, subRoot, slices_));
    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));
    return tempAlg->RunAsync(subRank, subLinks.size(), subLinks);
}

HcclResult BroadcastNHRV1::RunBroadcastOnVertical(const u32 rank, const std::vector<LINK>& links, const RingInfo& info)
{
    // 只有不出现在额外列的节点做Broadcast
    u32 hIndex = info.GetHIndex(rank);
    if (hIndex >= info.GetRowSize()) {
        return HCCL_SUCCESS;
    }

    // 收集link
    u32 vSize = info.GetVSizeByHIndex(hIndex);
    std::vector<LINK> subLinks(vSize);
    for (u32 vIdx = 0; vIdx < vSize; vIdx++) {
        u32 rankInRing = info.GetRank(vIdx, hIndex);
        CHK_PRT_RET(
            rankInRing >= links.size(),
            HCCL_ERROR(
                "[BroadcastNHRV1][Broadcast-V] rank[%u] out of range, "
                "rankInRing=%u, links.size=%u",
                rank, rankInRing, links.size()),
            HCCL_E_INTERNAL);
        subLinks[vIdx] = links[rankInRing];
        HCCL_DEBUG("[BroadcastNHRV1][Broadcast-V] rank[%u], ringRank[%u]=%u", rank, vIdx, rankInRing);
    }

    // 计算新的rank和root
    u32 subRank = info.GetVIndex(rank);
    u32 subRoot = info.GetVIndex(root_);
    HCCL_DEBUG("[BroadcastNHRV1][Broadcast-V] rank[%u] subRank=%u, subRoot=%u", rank, subRank, subRoot);

    // 计算新的内存块
    DeviceMem devMem = scratch_.range(slices_[hIndex].offset, slices_[hIndex].size);
    u64 memCount = 0;
    if (DataUnitSize(dataType_) != 0) {
        memCount = devMem.size() / DataUnitSize(dataType_);
    }
    // 执行Ring - Broadcast
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_RING, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    if (!barrierSwitchOn_) {
        tempAlg->CloseBarrier();
    }
    CHK_RET(tempAlg->Prepare(
        devMem, devMem, devMem, memCount, dataType_, stream_, reductionOp_, subRoot, std::vector<Slice>(0),
        slices_[hIndex].offset));
    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));
    return tempAlg->RunAsync(subRank, subLinks.size(), subLinks);
}

HcclResult
BroadcastNHRV1::RunAllGatherOnHorizontal(const u32 rank, const std::vector<LINK>& links, const RingInfo& info)
{
    // 收集link
    u32 vIndex = info.GetVIndex(rank);
    u32 hSize = info.GetHSizeByVIndex(vIndex);
    std::vector<LINK> subLinks(hSize);
    for (u32 hIdx = 0; hIdx < hSize; hIdx++) {
        u32 rankInRing = info.GetRank(vIndex, hIdx);
        CHK_PRT_RET(
            rankInRing >= links.size(),
            HCCL_ERROR(
                "[BroadcastNHRV1][AllGather-H] rank[%u] out of range, "
                "rankInRing=%u, links.size=%u",
                rank, rankInRing, links.size()),
            HCCL_E_INTERNAL);
        subLinks[hIdx] = links[rankInRing];
        HCCL_DEBUG("[BroadcastNHRV1][AllGather-H] rank[%u], ringRank[%u]=%u", rank, hIdx, rankInRing);
    }

    // 计算新的rank
    u32 subRank = info.GetHIndex(rank);
    HCCL_DEBUG("[BroadcastNHRV1][AllGather-H] rank[%u] subRank=%u", rank, subRank);

    // 执行Ring - AllGather
    // 此处Prepare的baseOffset给0，因为偏移量已经在加在slices里面
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    if (!barrierSwitchOn_) {
        tempAlg->CloseBarrier();
    }
    CHK_RET(tempAlg->Prepare(scratch_, scratch_, scratch_, -1, dataType_, stream_, reductionOp_, -1, slices_));
    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));
    return tempAlg->RunAsync(subRank, subLinks.size(), subLinks);
}

HcclResult BroadcastNHRV1::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 基本的检查
    CHK_RET(SimpleCheck(rank, rankSize, links));
    HCCL_DEBUG(
        "BroadcastNHRV1 run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    // 判断rank_size == 1
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            return HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        return HCCL_SUCCESS;
    }

    // 创建scratch
    if (rank == root_) {
        scratch_ = DeviceMem::create(inputMem_.ptr(), inputMem_.size());
    } else {
        scratch_ = DeviceMem::create(outputMem_.ptr(), outputMem_.size());
    }

    HCCL_DEBUG("[BroadcastNHRV1] root[%u] scratch[%p] memsize[%llu]", root_, scratch_.ptr(), scratch_.size());

    // 获取通信关系
    RingInfo info = GetRingInfo(rankSize);

    // 处理和检查Slices
    if (slices_.size() == 0) {
        CHK_RET(SetDefaultSlices(rank, info));
    }
    CHK_RET(CheckSlices(rank, info));

    // 水平方向做Ring Scatter（inputMem -> scratch_）
    CHK_RET(RunScatterOnHorizontal(rank, links, info));

    // 垂直方向做Ring Broadcast（scratch_ -> scratch_）
    CHK_RET(RunBroadcastOnVertical(rank, links, info));

    // 水平方向做Ring AllGather（scratch_ -> scratch_）
    CHK_RET(RunAllGatherOnHorizontal(rank, links, info));

    HCCL_INFO("BroadcastNHRV1 finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult BroadcastNHRV1::SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 判断stream, dispatcher是否为空
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());

    // 判断Memory是否为空
    if (rank == root_) {
        CHK_PRT_RET(!inputMem_, HCCL_ERROR("[BroadcastNHRV1]rank[%u] inputmem is null", rank), HCCL_E_PTR);
    } else {
        CHK_PRT_RET(!outputMem_, HCCL_ERROR("[BroadcastNHRV1]rank[%u] outputmem is null", rank), HCCL_E_PTR);
    }

    // 判断links数量是否正确
    CHK_PRT_RET(
        links.size() < rankSize,
        HCCL_ERROR(
            "[BroadcastNHRV1]rank[%u] link size[%llu] is less than "
            "rank size[%u]",
            rank, links.size(), rankSize),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult BroadcastNHRV1::SetDefaultSlices(const u32 rank, const RingInfo& info)
{
    u32 unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[BroadcastNHRV1] rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }

    // slices_只用于水平方向的Scatter和AllGather
    u32 rowSize = info.GetRowSize();
    u64 sliceCount = (count_ + rowSize - 1) / rowSize;
    u64 sliceSize = RoundUpWithDivisor(sliceCount * unitSize, HCCL_MIN_SLICE_ALIGN);
    u64 restSize = count_ * unitSize;
    slices_.resize(rowSize);
    for (u32 i = 0; i < rowSize; i++) {
        // broadcast逻辑与其他算子不太一样，impl传入的memory是总的大memory而不是预先切出server间的memory，需要在此处做处理
        slices_[i].offset = (i == 0) ? baseOffset_ : (slices_[i - 1].offset + slices_[i - 1].size);
        slices_[i].size = std::min(restSize, sliceSize);
        restSize -= slices_[i].size;
        HCCL_DEBUG(
            "[BroadcastNHRV1] rank[%u], slices[%u].offset=[%llu], slices[%u].size=[%llu] ", rank, i, slices_[i].offset,
            i, slices_[i].size);
    }

    // 如果有多余rank，需要添加空白slice以实现BrokenRing
    if (info.GetHSizeByRank(rank) > info.GetRowSize()) {
        Slice slice;
        slice.offset = 0;
        slice.size = 0;
        slices_.push_back(slice);
        HCCL_DEBUG(
            "[BroadcastNHRV1] rank[%u], slices[%u].offset=[%llu], slices[%u].size=[%llu] ", rank, slices_.size() - 1, 0,
            slices_.size() - 1, 0);
    }
    return HCCL_SUCCESS;
}

HcclResult BroadcastNHRV1::CheckSlices(const u32 rank, const RingInfo& info)
{
    u32 expectedSlices = info.GetHSizeByRank(rank);
    CHK_PRT_RET(
        slices_.size() != expectedSlices,
        HCCL_ERROR(
            "[BroadcastNHRV1]slices.size[%u] should be equal to sqrt of rankSize[%u]", slices_.size(), expectedSlices),
        HCCL_E_INTERNAL);

    for (u32 idx = 1; idx < slices_.size(); idx++) {
        if (slices_[idx].size != 0) {
            CHK_PRT_RET(
                slices_[idx - 1].offset + slices_[idx - 1].size != slices_[idx].offset,
                HCCL_ERROR(
                    "[BroadcastNHRV1]only support continuous slices, but get slices[%u].offset[%u]"
                    ", slices[%u].size[%u], slices[%u].offset[%u]",
                    idx - 1, slices_[idx - 1].offset, idx - 1, slices_[idx - 1].size, idx, slices_[idx].offset),
                HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_BROADCAST_NHR_V1, BroadcastNHRV1);
} // namespace hccl
