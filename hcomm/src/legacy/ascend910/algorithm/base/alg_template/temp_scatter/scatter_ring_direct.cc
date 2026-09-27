/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_ring_direct.h"
#include "alg_template_register.h"

namespace hccl {
ScatterRingDirect::ScatterRingDirect(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

ScatterRingDirect::~ScatterRingDirect() {}

HcclResult ScatterRingDirect::Prepare(
    HcomCollOpInfo* opInfo, const u32 userRank, const std::vector<u32>& ringsOrders,
    const std::vector<Slice>& userMemInputSlices)
{
    opInfo_ = opInfo;
    userRank_ = userRank;
    ringsOrder_ = ringsOrders;
    userMemInputSlices_ = userMemInputSlices;
    return HCCL_SUCCESS;
}

// reduce scatter ring direct算法的函数入口
HcclResult ScatterRingDirect::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 基本的检查
    CHK_RET(CheckParameters(rank, rankSize, links));

    // 判断rank_size == 1, 若inputMem_ != outputMem_，才需要搬运
    if (rankSize == 1) {
        CHK_RET(OneRankMemcpy());
        return HCCL_SUCCESS;
    }
    // 收集邻居信息
    CHK_RET(GetInitializedNeighborLinks(rank, rankSize, links));
    // 填充slice_
    CHK_RET(SetSlices(rank, rankSize));

    // 运行scatter, ring算法
    CHK_RET(RunScatter(rank, rankSize));

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(leftLink_, rightLink_));
    }

    HCCL_INFO("ScatterRingDirect finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

HcclResult ScatterRingDirect::CheckParameters(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    CHK_PTR_NULL(opInfo_);
    CHK_RET(CheckConcurrentDirectParameters(rank, rankSize, links));
    // 判断ringsOrder数量是否正确
    CHK_PRT_RET(
        ringsOrder_.size() != rankSize,
        HCCL_ERROR(
            "[ScatterRingDirect] ringsOrder size[%u] is not equal to rank size[%u]", ringsOrder_.size(), rankSize),
        HCCL_E_PARA);
    // 判断userMemInputSlices数量是否正确
    CHK_PRT_RET(
        userMemInputSlices_.size() != rankSize,
        HCCL_ERROR(
            "[ScatterRingDirect] userMemInputSlices size[%u] is not equal to rank size[%u]", userMemInputSlices_.size(),
            rankSize),
        HCCL_E_PARA);
    HCCL_INFO("ScatterRingDirect CheckParameters success");
    return HCCL_SUCCESS;
}

HcclResult ScatterRingDirect::OneRankMemcpy()
{
    const Slice& srcSlice = userMemInputSlices_[0];
    const Slice& dstSlice = slices_[0];
    DeviceMem src;
    DeviceMem dst;
    if (opInfo_->inputAddr == nullptr) {
        src = inputMem_.range(srcSlice.offset, srcSlice.size);
    } else {
        src = DeviceMem::create(static_cast<u8*>(opInfo_->inputAddr) + srcSlice.offset, srcSlice.size);
    }
    if (opInfo_->outputAddr != nullptr) {
        // opInfo_->outputAddr != nullptr指示要将输出发送至user output
        u64 stepOffset = slices_[ringsOrder_[0]].offset;
        HCCL_DEBUG(
            "Memcpy operation: stream[main], rank[%u] starts to rcv offset[%llu], size[%llu] at userMemOut_", userRank_,
            stepOffset, dstSlice.size);
        dst = DeviceMem::create(static_cast<u8*>(opInfo_->outputAddr) + stepOffset, dstSlice.size);
    } else {
        // opInfo_->outputAddr == nullptr指示要将输出发送至CCL buffer
        HCCL_DEBUG(
            "Memcpy operation: stream[main], rank[%u] starts to rcv offset[%llu], size[%llu] at outputMem_", userRank_,
            dstSlice.offset, dstSlice.size);
        dst = outputMem_.range(dstSlice.offset, dstSlice.size);
    }
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    return HCCL_SUCCESS;
}

HcclResult
ScatterRingDirect::GetInitializedNeighborLinks(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 收集左邻居信息
    leftLink_ = links[(rank + rankSize - 1) % rankSize];
    CHK_SMART_PTR_NULL(leftLink_);

    // 收集右邻居信息
    rightLink_ = links[(rank + 1) % rankSize];
    CHK_SMART_PTR_NULL(rightLink_);
    HCCL_INFO("ScatterRingDirect finished to GetInitializedNeighborLinks");
    return HCCL_SUCCESS;
}

HcclResult ScatterRingDirect::SetSlices(const u32 rank, const u32 rankSize)
{
    if (slices_.size() == 0) {
        slices_.resize(rankSize);

        // 生成std::vector<Slice> slices_
        u64 sliceSize = count_ * SIZE_TABLE[dataType_];
        ;

        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = sliceSize;
            // 用于DMA消减过程中，消除src与dst不对位的风险
            slices_[i].offset = RoundUpWithDivisor(i * sliceSize, HCCL_MIN_SLICE_ALIGN);

            HCCL_DEBUG(
                "rank[%u], slices[%u].offset=[%llu], slices[%u].size=[%llu]", rank, i, slices_[i].offset, i,
                slices_[i].size);
        }
    }
    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_DEBUG))) {
        for (u32 i = 0; i < slices_.size(); i++) {
            HCCL_DEBUG(
                "[ScatterRingDirect][SetSlices]rank[%u], slices[%u].offset=[%llu], slices[%u].size=[%llu]", rank, i,
                slices_[i].offset, i, slices_[i].size);
        }
    }
    // 最后一步搬到userMemOut_的offset, 不同的ring环offset不一样
    lastStepOffset_ = slices_[ringsOrder_[0]].offset;
    HCCL_INFO("ScatterRingDirect finished to SetSlices");
    return HCCL_SUCCESS;
}

HcclResult ScatterRingDirect::RunScatter(const u32 rank, const u32 rankSize)
{
    HCCL_INFO("ScatterRingDirect starts, the input param rank[%u]", rank);
    // 空拷贝用于后续操作附着
    CHK_RET(ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    // 例如rank[0,1,2,3]中，rank0的rxSliceIdx = 2，txSliceIdx = 3, subSliceIdx = 1
    u32 txSliceIdx = (rank + rankSize - 1) % rankSize;
    u32 rxSliceIdx = (rank + rankSize - DMA_REDUCE_TWO_OFFSET) % rankSize;
    u32 subSliceIdx = (rank + rankSize - DMA_REDUCE_TWO_OFFSET) % rankSize; // 只存在于根节点
    u32 stepsFromRank2Root = (rank + rankSize - root_) % rankSize;
    for (u32 step = 0; step < rankSize - 1; step++) {
        const Slice& subSlice = userMemInputSlices_[subSliceIdx];
        const Slice& cclSlice = slices_[subSliceIdx];
        const Slice& txSlice = slices_[txSliceIdx];
        const Slice& rxSlice = slices_[rxSliceIdx];

        CHK_RET(RunScatterOnRootRank(step, subSlice, cclSlice, rank, rankSize));
        CHK_RET(RunScatterOnOtherRank(stepsFromRank2Root, step, txSlice, rxSlice, rankSize));

        // 更新索引
        subSliceIdx = (subSliceIdx + rankSize - 1) % rankSize;
        txSliceIdx = (txSliceIdx + rankSize - 1) % rankSize;
        rxSliceIdx = (rxSliceIdx + rankSize - 1) % rankSize;
    }
    HCCL_INFO("ScatterRingDirect finished to RunScatter");
    return HCCL_SUCCESS;
}

HcclResult ScatterRingDirect::RunScatterOnOtherRank(
    const u32 stepsFromRank2Root, const u32 step, const Slice& txSlice, const Slice& rxSlice, const u32 rankSize)
{
    bool needSend = stepsFromRank2Root <= step;
    bool needReceive = stepsFromRank2Root > 0 && stepsFromRank2Root <= (step + 1);
    // Ack
    if (needReceive) {
        CHK_RET(leftLink_->TxAck(stream_));
    }
    if (needSend) {
        CHK_RET(rightLink_->RxAck(stream_));
    }

    DeviceMem src;
    // 不同的rank会在不同的step开始持续发送操作，距离root节点越近，越早step开始发送操作
    if (needSend) {
        src = inputMem_.range(txSlice.offset, txSlice.size);
        CHK_RET(rightLink_->TxAsync(
            UserMemType::INPUT_MEM, txSlice.offset + baseOffset_, src.ptr(), txSlice.size, stream_));
    }
    // 不同的rank会在不同的step开始持续发送操作，距离root节点越近，越早step开始发送操作
    DeviceMem dst;
    if (needReceive) {
        HCCL_DEBUG(
            "MemcpyAsync operation: step[%u] stream[main], src rank[%u] starts to send offset[%llu] size[%llu] "
            "from leftMem_",
            step, leftLink_->GetRemoteRank(), rxSlice.offset, rxSlice.size);
        if (step == rankSize - DMA_REDUCE_TWO_OFFSET && opInfo_->outputAddr != nullptr) {
            HCCL_DEBUG(
                "MemcpyAsync operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu], "
                "size[%llu] "
                "at userMemOut_",
                step, userRank_, lastStepOffset_, rxSlice.size);
            dst = DeviceMem::create(static_cast<u8*>(opInfo_->outputAddr) + lastStepOffset_, rxSlice.size);
        } else {
            HCCL_DEBUG(
                "MemcpyAsync operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu], "
                "size[%llu] "
                "at inputMem_",
                step, userRank_, rxSlice.offset, rxSlice.size);
            dst = inputMem_.range(rxSlice.offset, rxSlice.size);
        }
        CHK_RET(
            leftLink_->RxAsync(UserMemType::INPUT_MEM, rxSlice.offset + baseOffset_, dst.ptr(), rxSlice.size, stream_));
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterRingDirect::RunScatterOnRootRank(
    const u32 step, const Slice& subSlice, const Slice& cclSlice, const u32 rank, const u32 rankSize)
{
    if (step == rankSize - DMA_REDUCE_TWO_OFFSET && opInfo_->outputAddr != nullptr && rank == root_) {
        HCCL_DEBUG(
            "MemcpyAsync operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu], "
            "size[%llu] at userMemOut_",
            step, userRank_, lastStepOffset_, subSlice.size);
        DeviceMem src = inputMem_.range(cclSlice.offset, cclSlice.size);
        DeviceMem dst = DeviceMem::create(static_cast<u8*>(opInfo_->outputAddr) + lastStepOffset_, subSlice.size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_SCATTER_RING_DIRECT, ScatterRingDirect);
} // namespace hccl
