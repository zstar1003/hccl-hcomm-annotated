/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_ring_direct.h"
#include "alg_template_register.h"

namespace hccl {
AllGatherRingDirect::AllGatherRingDirect(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllGatherRingDirect::~AllGatherRingDirect() {}

HcclResult AllGatherRingDirect::Prepare(
    HcomCollOpInfo* opInfo, u32 userRank, const std::vector<Slice>& userMemOutputSlices, bool isSdma)
{
    opInfo_ = opInfo;
    userRank_ = userRank;
    userMemOutputSlices_ = userMemOutputSlices;
    isSdma_ = isSdma;
    return HCCL_SUCCESS;
}

// allgather的入口函数
HcclResult AllGatherRingDirect::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 基本的检查
    CHK_RET(CheckParameters(rank, rankSize, links));

    if (rankSize == 1) {
        CHK_RET(OneRankMemcpy());
        return HCCL_SUCCESS;
    }
    // 收集邻居信息
    CHK_RET(GetInitializedNeighborLinks(rank, rankSize, links));

    // 填充slice_
    CHK_RET(SetSlices(rank, rankSize));

    // 运行all-gather, ring算法
    CHK_RET(RunAllGather(rank, rankSize));

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(leftLink_, rightLink_));
    }

    HCCL_INFO("AllGatherRingDirect finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult AllGatherRingDirect::CheckParameters(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    CHK_PTR_NULL(opInfo_);
    CHK_RET(CheckConcurrentDirectParameters(rank, rankSize, links));
    // 判断userMemInputSlices数量是否正确
    CHK_PRT_RET(
        userMemOutputSlices_.size() % rankSize != 0,
        HCCL_ERROR(
            "[AllGatherRingDirect] userMemOutputSlices size[%u] can not be divided by rank size[%u]",
            userMemOutputSlices_.size(), rankSize),
        HCCL_E_PARA);

    HCCL_INFO("AllGatherRingDirect finished to CheckParameters");
    return HCCL_SUCCESS;
}

// 单卡场景
HcclResult AllGatherRingDirect::OneRankMemcpy()
{
    for (u32 sliceIdx = 0; sliceIdx < slices_.size(); sliceIdx++) {
        const Slice& srcSlice = slices_[sliceIdx];
        const Slice& dstSlice = userMemOutputSlices_[sliceIdx];
        DeviceMem src;
        DeviceMem dst = DeviceMem::create(static_cast<u8*>(opInfo_->outputAddr) + dstSlice.offset, dstSlice.size);
        if (opInfo_->inputAddr != nullptr) {
            // opInfo_->inputAddr != nullptr指示要从user input获取输入
            u64 stepOffset = slices_[0].offset;
            HCCL_DEBUG(
                "Memcpy operation: stream[main], rank[%u] starts to copy offset[%llu], size[%llu] at userInput",
                userRank_, stepOffset, srcSlice.size);
            src = DeviceMem::create(static_cast<u8*>(opInfo_->inputAddr) + stepOffset, srcSlice.size);
        } else {
            // opInfo_->inputAddr == nullptr指示要从CCL buffer获取输入
            HCCL_DEBUG(
                "Memcpy operation: stream[main], rank[%u] starts to copy offset[%llu], size[%llu] at inputMem_",
                userRank_, srcSlice.offset, srcSlice.size);
            src = inputMem_.range(srcSlice.offset, srcSlice.size);
        }
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    return HCCL_SUCCESS;
}

HcclResult
AllGatherRingDirect::GetInitializedNeighborLinks(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 收集左邻居信息
    leftLink_ = links[(rank + rankSize - 1) % rankSize];
    CHK_SMART_PTR_NULL(leftLink_);

    // 收集右邻居信息
    rightLink_ = links[(rank + 1) % rankSize];
    CHK_SMART_PTR_NULL(rightLink_);

    HCCL_INFO("AllGatherRingDirect finished to GetInitializedNeighborLinks");
    return HCCL_SUCCESS;
}

HcclResult AllGatherRingDirect::SetSlices(const u32 rank, const u32 rankSize)
{
    inputSlices_ = slices_;
    if (slices_.size() == 0) {
        slices_.resize(rankSize);
        inputSlices_.resize(rankSize);

        u64 sliceSize = count_ * DataUnitSize(dataType_);
        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = sliceSize;
            slices_[i].offset = sliceSize * i;
            inputSlices_[i].size = sliceSize;
            inputSlices_[i].offset = (inputMem_.size() < outputMem_.size()) ? 0 : (sliceSize * i);
            HCCL_DEBUG(
                "rank[%u], slices[%u].offset=%llu, slices[%u].size=[%llu]", rank, i, slices_[i].offset, i,
                slices_[i].size);
        }
    }

    if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {
        for (u32 i = 0; i < slices_.size(); i++) {
            HCCL_DEBUG(
                "[AllGatherRingDirect][SetSlices]rank[%u], slices[%u].offset=[%llu], slices[%u].size=[%llu]", rank, i,
                slices_[i].offset, i, slices_[i].size);
        }
    }

    HCCL_INFO("AllGatherRingDirect finished to SetSlices");
    return HCCL_SUCCESS;
}

HcclResult AllGatherRingDirect::RunInitStep(const u32 rank, const u32 rankSize)
{
    // 第一步搬到userMemIn_的offset
    auto firstStepOffset = slices_[0].offset;

    // 第-1步，片内将部分数据从userIn搬到cclIn
    DeviceMem srcInit;
    DeviceMem dstInit;
    u32 initSliceIdx = rank;
    u32 sliceSize = slices_.size() / rankSize;

    for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
        Slice initSlice = slices_[initSliceIdx * sliceSize + sliceIdx];

        // 需要+userMemIn_的offset
        if (opInfo_->inputAddr != nullptr) {
            // AllGather算子调用AllGatherRingDirect场景
            srcInit = DeviceMem::create(static_cast<u8*>(opInfo_->inputAddr) + firstStepOffset, initSlice.size);
        } else {
            // AllReduce算子调用AllGatherRingDirect场景
            srcInit = inputMem_.range(initSlice.offset, initSlice.size);
        }

        dstInit = outputMem_.range(initSlice.offset, initSlice.size);
        HCCL_DEBUG(
            "Memcpy operation: step[-1] stream[main] src rank[%u] starts to copy(rcv) offset[%llu], "
            "size[%llu] on userMemOutput to offset[%llu], size[%llu] on CCL",
            userRank_, firstStepOffset, initSlice.size, initSlice.offset, initSlice.size);

        // 若src与dst一样，则不需要搬运
        if (srcInit != dstInit) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstInit, srcInit, stream_));
        }
    }

    return HCCL_SUCCESS;
}

// 本端cclout -> 本端userout
HcclResult AllGatherRingDirect::RunAllGatherPartOne(const u32 sliceSize, const u32 step, const u32 txSliceIdx)
{
    std::vector<Slice> txSliceVector;
    std::vector<Slice> sliceVector;

    for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
        txSliceVector.push_back(slices_[txSliceIdx * sliceSize + sliceIdx]);
        sliceVector.push_back(userMemOutputSlices_[txSliceIdx * sliceSize + sliceIdx]);
    }

    for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
        DeviceMem src = outputMem_.range(txSliceVector[sliceIdx].offset, txSliceVector[sliceIdx].size);
        DeviceMem dst = DeviceMem::create(
            static_cast<u8*>(opInfo_->outputAddr) + sliceVector[sliceIdx].offset, sliceVector[sliceIdx].size);

        HCCL_DEBUG(
            "Memcpy operation: step[%u] stream[sub], src rank[%u] starts to send offset[%llu] size[%llu], "
            "dst rank starts to rcv offset[%llu] size[%llu] at userMemOutput_",
            step, userRank_, sliceVector[sliceIdx].offset, sliceVector[sliceIdx].size, txSliceVector[sliceIdx].offset,
            txSliceVector[sliceIdx].size);

        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    return HCCL_SUCCESS;
}

// 对端cclout -> 本端cclout, 如果最后一步则：对端cclout -> 本端userout （DMA消减）
HcclResult AllGatherRingDirect::RunAllGatherPartTwo(
    const u32 sliceSize, const u32 step, const u32 txSliceIdx, const u32 rxSliceIdx, const u32 rankSize)
{
    std::vector<Slice> txSliceVector;
    std::vector<Slice> rxSliceVector;
    std::vector<Slice> sliceVector;

    for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
        txSliceVector.push_back(slices_[txSliceIdx * sliceSize + sliceIdx]);
        rxSliceVector.push_back(slices_[rxSliceIdx * sliceSize + sliceIdx]);
        sliceVector.push_back(userMemOutputSlices_[rxSliceIdx * sliceSize + sliceIdx]);
    }

    CHK_RET(leftLink_->TxAck(stream_));
    CHK_RET(rightLink_->RxAck(stream_));

    std::vector<TxMemoryInfo> txMems;
    std::vector<RxMemoryInfo> rxMems;

    for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
        DeviceMem src = outputMem_.range(txSliceVector[sliceIdx].offset, txSliceVector[sliceIdx].size);
        HCCL_DEBUG(
            "tx srcMem[%p] range[%llu] size[%llu] ", src.ptr(), txSliceVector[sliceIdx].offset,
            txSliceVector[sliceIdx].size);
        txMems.emplace_back(TxMemoryInfo{
            UserMemType::OUTPUT_MEM, txSliceVector[sliceIdx].offset + baseOffset_, src.ptr(),
            txSliceVector[sliceIdx].size});

        DeviceMem dst;
        if (isSdma_ && step == rankSize - DMA_REDUCE_TWO_OFFSET) {
            // 最后一步实现DMA消减：对端cclout -> 本端userout
            HCCL_DEBUG(
                "DMAReduce(sdma) MemcpyAsync operation: step[%u] stream[main], dst rank[%u] starts to rcv "
                "offset[%llu] size[%llu] at userMemOutput_",
                step, userRank_, sliceVector[sliceIdx].offset, sliceVector[sliceIdx].size);

            dst = DeviceMem::create(
                static_cast<u8*>(opInfo_->outputAddr) + sliceVector[sliceIdx].offset, sliceVector[sliceIdx].size);
        } else {
            HCCL_DEBUG(
                "MemcpyAsync operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu] size[%llu] "
                "at outputMem_",
                step, userRank_, rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size);

            // 中间步数无DMA消减
            dst = outputMem_.range(rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size);
            if (!isSdma_ && step == rankSize - DMA_REDUCE_TWO_OFFSET) {
                // 最后一步实现DMA消减：对端cclout -> 本端userout
                HCCL_DEBUG("DMAReduce(rdma) record final addr");

                finalSrc_.push_back(outputMem_.range(rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size));
                finalDst_.push_back(DeviceMem::create(
                    static_cast<u8*>(opInfo_->outputAddr) + sliceVector[sliceIdx].offset, sliceVector[sliceIdx].size));
            }
        }

        rxMems.emplace_back(RxMemoryInfo{
            UserMemType::OUTPUT_MEM, rxSliceVector[sliceIdx].offset + baseOffset_, dst.ptr(),
            rxSliceVector[sliceIdx].size});
    }

    CHK_RET(rightLink_->TxAsync(txMems, stream_));

    if (!isSdma_) {
        CHK_RET(leftLink_->RxAsync(rxMems, stream_));
    } else {
        CHK_RET(leftLink_->RxDataSignal(stream_));

        for (auto& mem : rxMems) {
            CHK_PTR_NULL(mem.dst);
            void* srcMemPtr = nullptr;
            CHK_RET(leftLink_->GetRemoteMem(mem.srcMemType, &srcMemPtr));

            DeviceMem srcDevMem(static_cast<s8*>(srcMemPtr) + mem.srcOffset, mem.len);
            DeviceMem dstDevMem(static_cast<s8*>(mem.dst), mem.len);

            CHK_RET(HcclD2DMemcpyAsync(
                dispatcher_, dstDevMem, srcDevMem, stream_, leftLink_->GetRemoteRank(), leftLink_->GetLinkType()));
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AllGatherRingDirect::RunAllGather(const u32 rank, const u32 rankSize)
{
    HCCL_INFO("AllGatherRingDirect starts, the input param rank[%u]", rank);
    CHK_RET(RunInitStep(rank, rankSize));

    finalSrc_.clear();
    finalDst_.clear();

    u32 txSliceIdx = rank;
    u32 sliceSize = slices_.size() / rankSize;
    u32 rxSliceIdx = (rank + rankSize - 1) % rankSize;

    for (u32 step = 0; step < rankSize - 1; step++) {
        // 本端cclout -> 本端userout
        CHK_RET(RunAllGatherPartOne(sliceSize, step, txSliceIdx));
        // 对端cclout -> 本端cclout, 如果最后一步则：对端cclout -> 本端userout （DMA消减）
        CHK_RET(RunAllGatherPartTwo(sliceSize, step, txSliceIdx, rxSliceIdx, rankSize));
        // 更新索引
        txSliceIdx = (txSliceIdx + rankSize - 1) % rankSize;
        rxSliceIdx = (rxSliceIdx + rankSize - 1) % rankSize;
    }

    if (!isSdma_) {
        for (u32 vecIdx = 0; vecIdx < finalSrc_.size(); vecIdx++) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, finalDst_[vecIdx], finalSrc_[vecIdx], stream_));
        }
    }

    HCCL_INFO("AllGatherRingDirect finished to RunAllGather");

    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_RING_DIRECT, AllGatherRingDirect);
} // namespace hccl
