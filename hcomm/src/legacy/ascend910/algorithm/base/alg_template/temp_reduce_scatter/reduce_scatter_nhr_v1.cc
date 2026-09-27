/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_nhr_v1.h"
#include "alg_template_register.h"

namespace hccl {

ReduceScatterNHRV1::ReduceScatterNHRV1(const HcclDispatcher dispatcher) : NHRV1Base(dispatcher) {}

ReduceScatterNHRV1::~ReduceScatterNHRV1() {}

HcclResult ReduceScatterNHRV1::Prepare(u64 reduceAttrBitMap, HcomCollOpInfo* opInfo)
{
    (void)opInfo;
    reduceAttr_ = reduceAttrBitMap;
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNHRV1::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 基本的检查
    CHK_RET(SimpleCheck(rank, rankSize, links));
    HCCL_INFO(
        "ReduceScatterNHRV1 run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    // 判断rank_size == 1
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            return HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        return HCCL_SUCCESS;
    }

    // 处理和检查Slices
    if (slices_.size() == 0) {
        CHK_RET(SetDefaultSlices(rank, rankSize));
    }
    CHK_RET(CheckSlices(rankSize));

    // 获取通信关系
    RingInfo info = GetRingInfo(rankSize);

    // 垂直方向做Ring
    CHK_RET(RunReduceScatterOnVertical(rank, links, info));

    // 水平方向做Ring
    CHK_RET(RunReduceScatterOnHorizontal(rank, links, info));

    // 额外的搬运（从(x,sqrt-1)搬运到(x,sqrt)）
    /* 一个可能的优化点：
    以8节点为例：    0   1   2
                    3   4   5
                    6   7
    当前的做法是：{0,1}、{3,4}、{6、7}做水平Ring，0/3/6/7拿到各自的那份结果，1拿到1和2的结果，4拿到4和5的结果，
                 最后1把2的结果发给2，4把5的结果发给5
    一种可能的更优做法是：以ReduceOp=Sum为例，首先把2和5的数据都置为0，
                        然后直接{0,1,2}、{3,4,5}、{6,7}做水平Ring，避免不等分Ring和额外的拷贝步骤，但需要调用TBE-assign
    */
    CHK_RET(RunLastCopyStep(rank, links, info));

    // 搬运数据到OutputMem
    CHK_RET(RunCopyDataToOutputMem(rank));

    HCCL_INFO("ReduceScatterNHRV1 finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNHRV1::SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 判断stream, dispatcher是否为空
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());

    // 检查memory
    CHK_PRT_RET(
        !outputMem_ || !inputMem_, HCCL_ERROR("[ReduceScatterNHRV1]rank[%u] inputmem or outputmem is null", rank),
        HCCL_E_PTR);

    // 判断links数量是否正确
    CHK_PRT_RET(
        links.size() < rankSize,
        HCCL_ERROR(
            "[ReduceScatterNHRV1]rank[%u] link size[%llu] is less than "
            "rank size[%u]",
            rank, links.size(), rankSize),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNHRV1::SetDefaultSlices(const u32 rank, const u32 rankSize)
{
    u32 unitSize = DataUnitSize(dataType_);
    CHK_PRT_RET(
        unitSize == 0, HCCL_ERROR("[ReduceScatterNHRV1]rank[%u] unit data size is zero", rank), HCCL_E_INTERNAL);

    slices_.resize(rankSize);
    u64 sliceSize = count_ * unitSize;
    for (u32 i = 0; i < rankSize; i++) {
        slices_[i].size = sliceSize;
        slices_[i].offset = (i * sliceSize);
        HCCL_DEBUG(
            "rank[%u], slices[%u].offset=[%llu], slices[%u].size=[%llu] ", rank, i, slices_[i].offset, i,
            slices_[i].size);
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNHRV1::CheckSlices(const u32 rankSize)
{
    CHK_PRT_RET(
        slices_.size() != rankSize,
        HCCL_ERROR("[ReduceScatterNHRV1]slices.size[%u] should be equal to rankSize[%u]", slices_.size(), rankSize),
        HCCL_E_INTERNAL);

    for (u32 idx = 1; idx < slices_.size(); idx++) {
        CHK_PRT_RET(
            slices_[idx - 1].offset + slices_[idx - 1].size != slices_[idx].offset,
            HCCL_ERROR(
                "[ReduceScatterNHRV1]only support continuous slices, but get "
                "slices[%u].offset[%u], slices[%u].size[%u], slices[%u].offset[%u]",
                idx - 1, slices_[idx - 1].offset, idx - 1, slices_[idx - 1].size, idx, slices_[idx].offset),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNHRV1::RunReduceScatterBrokenRing(
    const u32 rank, const std::vector<LINK>& links, const std::vector<Slice>& slices)
{
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(reduceAttr_));

    if (!barrierSwitchOn_) {
        tempAlg->CloseBarrier();
    }

    CHK_RET(
        tempAlg->Prepare(inputMem_, inputMem_, scratchMem_, count_, dataType_, stream_, reductionOp_, root_, slices));

    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    return tempAlg->RunAsync(rank, links.size(), links);
}

HcclResult
ReduceScatterNHRV1::RunReduceScatterOnVertical(const u32 rank, const std::vector<LINK>& links, const RingInfo& info)
{
    u32 hIndex = info.GetHIndex(rank); // 查找自己位于第几列

    // 构造新的links和slices
    std::vector<LINK> subLinks;
    std::vector<Slice> subSlices;
    u32 sliceIndexOffset = 0;                                                         // slice数量的累计偏移
    u32 hIndexForRing = (hIndex < info.GetRowSize()) ? hIndex : info.GetVIndex(rank); // 属于第几个垂直Ring
    u32 vSizeForRing = info.GetVSizeByHIndex(hIndexForRing);                          // 所属垂直Ring的大小
    for (u32 vIdx = 0; vIdx < vSizeForRing; vIdx++) {                                 // 处理垂直方向上的Rank
        // 增加link
        u32 rankInRing = info.GetRank(vIdx, hIndexForRing);
        CHK_PRT_RET(
            rankInRing >= links.size(),
            HCCL_ERROR(
                "[ReduceScatterNHRV1][Vertical] rank[%u] out of range, "
                "rankInRing=%u, links.size=%u",
                rank, rankInRing, links.size()),
            HCCL_E_INTERNAL);
        HCCL_DEBUG("[ReduceScatterNHRV1][Vertical] rank[%u] links[%u]=%u", rank, vIdx, rankInRing);
        subLinks.push_back(links[rankInRing]);

        // 寻找要合并的slice
        u32 nSlices = info.GetHSizeByVIndex(vIdx);
        Slice& headSlice = slices_[sliceIndexOffset];
        Slice& tailSlice = slices_[sliceIndexOffset + nSlices - 1];

        // 增加slice
        Slice slice;
        slice.offset = headSlice.offset;
        slice.size = tailSlice.offset + tailSlice.size - headSlice.offset;
        HCCL_DEBUG(
            "[ReduceScatterNHRV1][Vertical] rank[%u] subSlices[%u].offset=%llu, subSlices[%u].size=%llu", rank, vIdx,
            slice.offset, vIdx, slice.size);
        subSlices.push_back(slice);

        // 更新偏移
        sliceIndexOffset += nSlices;
    }

    // -- 可能还涉及跳跃的一个链接，比如8节点
    // ---- 0   1   2
    // ---- 3   4   5
    // ---- 6   7
    // -- 两个垂直Ring分别是{0,3,6,2}和{1,4,7,5}，而不是{0,3,6}和{1,4,7}
    if (info.GetHSizeByVIndex(hIndexForRing) > info.GetRowSize()) {
        // 添加link
        u32 rankInRing = info.GetRank(hIndexForRing, info.GetRowSize());
        CHK_PRT_RET(
            rankInRing >= links.size(),
            HCCL_ERROR(
                "[ReduceScatterNHRV1][Vertical] rank[%u] out of range, "
                "rankInRing=%u, links.size=%u",
                rank, rankInRing, links.size()),
            HCCL_E_INTERNAL);
        HCCL_DEBUG("[ReduceScatterNHRV1][Vertical] rank[%u] links[%u]=%u", rank, subLinks.size(), rankInRing);
        subLinks.push_back(links[rankInRing]);

        // 添加slice
        Slice slice;
        slice.offset = 0;
        slice.size = 0;
        HCCL_DEBUG(
            "[ReduceScatterNHRV1][Vertical] rank[%u] subSlices[%u].offset=%llu, subSlices[%u].size=%llu", rank,
            subLinks.size(), slice.offset, subLinks.size(), slice.size);
        subSlices.push_back(slice);
    }

    // 长度不足2，直接跳过
    if (subLinks.size() < 2) {
        return HCCL_SUCCESS;
    }

    // 计算在垂直Ring中的rank号
    u32 subRank = (hIndex == hIndexForRing) ? info.GetVIndex(rank) : vSizeForRing;
    HCCL_DEBUG("[ReduceScatterNHRV1][Vertical] rank[%u] subRank=%u", rank, subRank);

    // 执行Broken Ring ReduceScatter
    return RunReduceScatterBrokenRing(subRank, subLinks, subSlices);
}

HcclResult
ReduceScatterNHRV1::RunReduceScatterOnHorizontal(const u32 rank, const std::vector<LINK>& links, const RingInfo& info)
{
    u32 hIndex = info.GetHIndex(rank);
    if (hIndex >= info.GetRowSize()) {
        return HCCL_SUCCESS;
    }

    // 构造新的links和slices
    u32 vIndex = info.GetVIndex(rank);
    std::vector<LINK> subLinks;
    std::vector<Slice> subSlices;
    for (u32 hIdx = 0; hIdx < info.GetRowSize(); hIdx++) {
        // 增加link
        u32 rankInRing = info.GetRank(vIndex, hIdx);
        CHK_PRT_RET(
            rankInRing >= links.size(),
            HCCL_ERROR(
                "[ReduceScatterNHRV1][Horizontal] rank[%u] out of range, "
                "rankInRing=%u, links.size=%u",
                rank, rankInRing, links.size()),
            HCCL_E_INTERNAL);
        HCCL_DEBUG("[ReduceScatterNHRV1][Horizontal] rank[%u] links[%u]=%u", rank, hIdx, rankInRing);
        subLinks.push_back(links[rankInRing]);

        // 增肌slice（CheckSlices()已经约束slices_里的Slice都是连续的）
        // -- 比如8节点在做完水平Ring后
        // ---- 0   1   2
        // ---- 3   4   5
        // ---- 6   7
        // -- 0/3/6/7节点只拿到自己那份ReduceScatter结果，而1拿到1和2的两份数据、4拿到4和5的两份数据
        u64 sliceSize = slices_[rankInRing].size;
        if (hIdx == info.GetRowSize() - 1 && info.GetHSizeByVIndex(vIndex) > info.GetRowSize()) {
            sliceSize += slices_[rankInRing + 1].size;
        }

        Slice slice;
        slice.offset = slices_[rankInRing].offset;
        slice.size = sliceSize;
        HCCL_DEBUG(
            "[ReduceScatterNHRV1][Horizontal] rank[%u] subSlices[%u].offset=%llu, subSlices[%u].size=%llu", rank, hIdx,
            slice.offset, hIdx, slice.size);
        subSlices.push_back(slice);
    }

    // 长度不足2，直接跳过
    if (subLinks.size() < 2) {
        return HCCL_SUCCESS;
    }

    // 计算在水平Ring中的rank号
    u32 subRank = hIndex;
    HCCL_DEBUG("[ReduceScatterNHRV1][Horizontal] rank[%u] subRank=%u", rank, subRank);

    // 执行Broken Ring ReduceScatter
    return RunReduceScatterBrokenRing(subRank, subLinks, subSlices);
}

HcclResult ReduceScatterNHRV1::RunLastCopyStep(const u32 rank, const std::vector<LINK>& links, const RingInfo& info)
{
    HcclResult ret;

    u32 hIndex = info.GetHIndex(rank); // 查找自己位于第几列
    u32 vIndex = info.GetVIndex(rank); // 查找自己位于第几行
    if (hIndex >= info.GetRowSize() - 1 && info.GetHSizeByVIndex(vIndex) > info.GetRowSize()) {
        u32 peerRank = (hIndex == info.GetRowSize() - 1) ? rank + 1 : rank - 1;

        // 检查指针
        CHK_SMART_PTR_NULL(links[peerRank]);

        // TxAck
        ret = links[peerRank]->TxAck(stream_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[ReduceScatterNHRV1][RunLastCopyStep]rank[%u] tx ack from peerRank[%u] failed", rank, peerRank),
            ret);

        // RxAck
        ret = links[peerRank]->RxAck(stream_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[ReduceScatterNHRV1][RunLastCopyStep]rank[%u] rx ack from peerRank[%u] failed", rank, peerRank),
            ret);

        if (hIndex == info.GetRowSize() - 1) { // 发数据
            Slice& txSlice = slices_[peerRank];
            DeviceMem srcMem = inputMem_.range(txSlice.offset, txSlice.size);
            HCCL_DEBUG("tx srcMem[%p] range[%llu] size[%llu] ", srcMem.ptr(), txSlice.offset, txSlice.size);
            CHK_RET(ExecuteTxSync(
                links[peerRank], UserMemType::INPUT_MEM, txSlice.offset + baseOffset_, srcMem.ptr(), srcMem.size(),
                stream_));

            ret = links[peerRank]->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ReduceScatterNHRV1][RunLastCopyStep]TxWaitDone failed"), ret);
        } else { // 收数据
            Slice& rxSlice = slices_[rank];
            DeviceMem dstMem = inputMem_.range(rxSlice.offset, rxSlice.size);
            HCCL_DEBUG("rx dstMem[%p] range[%llu], size[%llu] ", dstMem.ptr(), rxSlice.offset, rxSlice.size);
            CHK_RET(ExecuteRxSync(
                links[peerRank], UserMemType::INPUT_MEM, rxSlice.offset + baseOffset_, dstMem.ptr(), dstMem.size(),
                stream_));

            ret = links[peerRank]->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ReduceScatterNHRV1][RunLastCopyStep]RxWaitDone failed"), ret);
        }

        // 如果不Barrier，SDMA结果在数据量超过CCL Buffer之后结果不正确
        CHK_RET(ExecuteBarrier(links[peerRank], stream_));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterNHRV1::RunCopyDataToOutputMem(const u32 rank)
{
    if (inputMem_ != outputMem_) {
        Slice& srcSlice = slices_[rank];
        DeviceMem dst = outputMem_.range(0, srcSlice.size);
        DeviceMem src = inputMem_.range(srcSlice.offset, srcSlice.size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_NHR_V1, ReduceScatterNHRV1);
} // namespace hccl
