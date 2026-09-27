/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aligned_reduce_scatter_double_ring.h"
#include "alg_template_register.h"

namespace hccl {
AlignedReduceScatterDoubleRing::AlignedReduceScatterDoubleRing(const HcclDispatcher dispatcher)
    : AlgTemplateBase(dispatcher)
{}

AlignedReduceScatterDoubleRing::~AlignedReduceScatterDoubleRing() {}

HcclResult AlignedReduceScatterDoubleRing::Prepare(
    DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
    const Stream& stream, const std::vector<std::vector<Slice>>& multRingsSlices, const HcclReduceOp reductionOp,
    const u32 root, const u64 baseOffset, const bool disableDMAReduce, const u64 reduceAttrBitMap,
    const HcomCollOpInfo* opInfo, const u32 userRank, std::vector<Stream>& subStreams,
    const std::vector<std::shared_ptr<LocalNotify>>& mainSignals,
    const std::vector<std::shared_ptr<LocalNotify>>& subSignals, const std::vector<std::vector<u32>>& ringsOrders,
    const std::vector<std::vector<Slice>>& userMemInputSlicesOfDoubleRing)
{
    reduceAttr_ = reduceAttrBitMap;
    opInfo_ = opInfo;
    userRank_ = userRank;
    subStreams_ = subStreams;
    mainSignals_ = mainSignals;
    subSignals_ = subSignals;
    ringsOrders_ = ringsOrders;
    userMemInputSlicesOfDoubleRing_ = userMemInputSlicesOfDoubleRing;
    return AlgTemplateBase::Prepare(
        inputMem, outputMem, scratchMem, count, dataType, stream, multRingsSlices, reductionOp, root, baseOffset,
        disableDMAReduce);
}

// reduce scatter ring direct算法的函数入口
HcclResult AlignedReduceScatterDoubleRing::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 基本的检查
    CHK_RET(CheckParameters(rank, rankSize, links));

    // 判断rank_size == 1的情况，并拷贝
    if (rankSize == 1) {
        CHK_RET(OneRankMemcpy());
        return HCCL_SUCCESS;
    }
    // 收集本地mem信息
    CHK_RET(InitSenderReducer());

    // 收集邻居信息
    CHK_RET(GetInitializedNeighborLinks(rank, rankSize, links));

    // 填充slice_
    CHK_RET(SetSlices(rank, rankSize));

    // 运行reduce-scatter, ring算法
    CHK_RET(RunReduceScatter(rank, rankSize));

    CHK_RET(LaunchTaskExtend(dispatcher_, stream_, subStreams_));

    HCCL_INFO("AlignedReduceScatterDoubleRing finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult
AlignedReduceScatterDoubleRing::CheckParameters(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    CHK_PTR_NULL(opInfo_);
    CHK_RET(CheckConcurrentDirectParameters(rank, rankSize, links));
    // 判断subStreams数量是否正确
    CHK_PRT_RET(
        subStreams_.size() < 1,
        HCCL_ERROR("[AlignedReduceScatterDoubleRing] subStreams size[%u] is less than 1", subStreams_.size()),
        HCCL_E_PARA);
    for (auto& s : subStreams_) {
        CHK_PTR_NULL(s.ptr());
    }
    // 判断mainSignals数量是否正确
    CHK_PRT_RET(
        mainSignals_.size() < 1,
        HCCL_ERROR("[AlignedReduceScatterDoubleRing] mainSignals size[%u] is less than 1", mainSignals_.size()),
        HCCL_E_PARA);
    // 判断subSignals数量是否正确
    CHK_PRT_RET(
        subSignals_.size() < 1,
        HCCL_ERROR("[AlignedReduceScatterDoubleRing] subSignals size[%u] is less than 1", subSignals_.size()),
        HCCL_E_PARA);
    // 判断ringsOrder数量是否正确
    for (u32 ringIndex = 0; ringIndex < ringsOrders_.size(); ringIndex++) {
        CHK_PRT_RET(
            ringsOrders_[ringIndex].size() != rankSize,
            HCCL_ERROR(
                "[AlignedReduceScatterDoubleRing] ringsOrders[%u] size[%u] is not equal to rank size[%u]", ringIndex,
                ringsOrders_[ringIndex].size(), rankSize),
            HCCL_E_PARA);
    }
    // 判断userMemInputSlices数量是否正确
    for (u32 ringIndex = 0; ringIndex < userMemInputSlicesOfDoubleRing_.size(); ringIndex++) {
        CHK_PRT_RET(
            userMemInputSlicesOfDoubleRing_[ringIndex].size() % rankSize != 0,
            HCCL_ERROR(
                "[AlignedReduceScatterDoubleRing] userMemInputSlicesOfDoubleRing[%u] size[%u] cannot be divided by "
                "size[%u]",
                ringIndex, userMemInputSlicesOfDoubleRing_[ringIndex].size(), rankSize),
            HCCL_E_PARA);
    }
    u32 mainSliceSize = multRingsSlices_[ALIGNED_MAIN_RING_INDEX].size() / rankSize;
    u32 subSliceSize = multRingsSlices_[ALIGNED_SUB_RING_INDEX].size() / rankSize;
    CHK_PRT_RET(
        mainSliceSize != subSliceSize,
        HCCL_ERROR(
            "[AlignedReduceScatterDoubleRing] mainSliceSize[%u] is not equal to subSliceSize[%u].", mainSliceSize,
            subSliceSize),
        HCCL_E_PARA);
    HCCL_INFO("AlignedReduceScatterDoubleRing finished to CheckParameters");
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRing::OneRankMemcpy()
{
    CHK_RET(MainRecordSub()); // 主流通知从流开始通信
    CHK_RET(SubWaitMain());   // 从流等待主流通知
    for (u32 ringIndex = 0; ringIndex < multRingsSlices_.size(); ringIndex++) {
        for (u32 sliceIdx = 0; sliceIdx < multRingsSlices_[ringIndex].size(); sliceIdx++) {
            const Slice& srcSlice = userMemInputSlicesOfDoubleRing_[ringIndex][sliceIdx];
            const Slice& dstSlice = multRingsSlices_[ringIndex][sliceIdx];
            DeviceMem src = DeviceMem::create(static_cast<u8*>(opInfo_->inputAddr) + srcSlice.offset, srcSlice.size);
            DeviceMem dst;
            if (opInfo_->outputAddr != nullptr) {
                // opInfo_->outputAddr != nullptr指示要将输出发送至user output
                u64 stepOffset = multRingsSlices_[ringIndex][ringsOrders_[ringIndex][0]].offset;
                HCCL_DEBUG(
                    "Memcpy operation: stream[main], rank[%u] starts to rcv offset[%llu], size[%llu] at userMemOut_",
                    userRank_, stepOffset, dstSlice.size);
                dst = DeviceMem::create(static_cast<u8*>(opInfo_->outputAddr) + stepOffset, dstSlice.size);
            } else {
                // opInfo_->outputAddr == nullptr指示要将输出发送至CCL buffer
                HCCL_DEBUG(
                    "Memcpy operation: stream[main], rank[%u] starts to rcv offset[%llu], size[%llu] at outputMem_",
                    userRank_, dstSlice.offset, dstSlice.size);
                dst = outputMem_.range(dstSlice.offset, dstSlice.size);
            }
            if (ringIndex == 1) {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
            } else {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStreams_[0]));
            }
        }
        HCCL_DEBUG("[AlignedReduceScatterDoubleRing][OneRankMemcpy] ringIndex[%u] Memcpy success", ringIndex);
    }
    CHK_RET(SubRecordMain()); // 从流通知主流通信完成
    CHK_RET(MainWaitSub());   // 主流等待从流通知
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRing::InitSenderReducer()
{
    // 创建reducer & sender
    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(senderInfo_);

    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(reducerInfo_);
    HCCL_INFO("AlignedReduceScatterDoubleRing finished to InitSenderReducer");
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRing::GetInitializedNeighborLinks(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 收集左邻居信息
    leftLink_ = links[(rank + rankSize - 1) % rankSize];
    CHK_SMART_PTR_NULL(leftLink_);

    // 收集右邻居信息
    rightLink_ = links[(rank + 1) % rankSize];
    CHK_SMART_PTR_NULL(rightLink_);
    HCCL_INFO("AlignedReduceScatterDoubleRing finished to GetInitializedNeighborLinks");
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRing::SetSlices(const u32 rank, const u32 rankSize)
{
    for (u32 ringIndex = 0; ringIndex < multRingsSlices_.size(); ringIndex++) {
        if (multRingsSlices_[ringIndex].size() == 0) {
            multRingsSlices_[ringIndex].resize(rankSize);

            // 生成std::vector<Slice> multRingsSlices_[ringIndex]
            u64 sliceSize = count_ * SIZE_TABLE[dataType_];

            for (u32 i = 0; i < rankSize; i++) {
                multRingsSlices_[ringIndex][i].size = sliceSize;
                // 用于DMA消减过程中，消除src与dst不对位的风险
                multRingsSlices_[ringIndex][i].offset = RoundUpWithDivisor(i * sliceSize, HCCL_MIN_SLICE_ALIGN);

                HCCL_DEBUG(
                    "multRingsSlices_[%u], rank[%u], slices[%u].offset=[%llu], slices[%u].size=[%llu]", ringIndex, rank,
                    i, multRingsSlices_[ringIndex][i].offset, i, multRingsSlices_[ringIndex][i].size);
            }
        }
        for (u32 i = 0; i < multRingsSlices_[ringIndex].size(); i++) {
            HCCL_DEBUG(
                "[AlignedReduceScatterDoubleRing][SetSlices] multRingsSlices_[%u], rank[%u], "
                "slices[%u].offset=[%llu], slices[%u].size=[%llu]",
                ringIndex, rank, i, multRingsSlices_[ringIndex][i].offset, i, multRingsSlices_[ringIndex][i].size);
        }
        // 最后一步搬到userMemOut_的offset, 不同的ring环offset不一样
        u64 toUserMemOffset;
        if (ringIndex == 0) {
            toUserMemOffset = multRingsSlices_[ringIndex][ringsOrders_[ringIndex][0]].offset;
        } else {
            const auto& prevRingSlice = multRingsSlices_[ringIndex - 1][ringsOrders_[ringIndex - 1][rank]];
            const auto& slice = multRingsSlices_[ringIndex][ringsOrders_[ringIndex][rank]];
            toUserMemOffset = slice.offset - prevRingSlice.offset;
        }
        HCCL_DEBUG(
            "[AlignedReduceScatterDoubleRing][SetSlices] rank[%u], ring[%u], toUserMemOffset[%u]", rank, ringIndex,
            toUserMemOffset);
        lastStepOffsets_.emplace_back(toUserMemOffset);
    }
    HCCL_INFO("AlignedReduceScatterDoubleRing finished to SetSlices");
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRing::PrepareInitSlices(
    const u32 rankSize, u64 ringIndex, u32 discontinuousSliceSize, u32 discontinuousSliceIdx, u32 initSlice0Idx,
    u32 initSlice1Idx, DeviceMem& dstInit, DeviceMem& srcInit, DeviceMem& dstSubInit, DeviceMem& srcSubInit)
{
    // 第-1步，片内将部分数据从userIn搬到cclIn
    const Slice& srcInitSlice0
        = userMemInputSlicesOfDoubleRing_[ringIndex][initSlice0Idx * discontinuousSliceSize + discontinuousSliceIdx];
    srcInit = DeviceMem::create(static_cast<u8*>(opInfo_->inputAddr) + srcInitSlice0.offset, srcInitSlice0.size);
    const Slice& dstInitSlice0
        = multRingsSlices_[ringIndex][initSlice0Idx * discontinuousSliceSize + discontinuousSliceIdx];
    dstInit = inputMem_.range(dstInitSlice0.offset, dstInitSlice0.size);

    const Slice& srcInitSlice1
        = userMemInputSlicesOfDoubleRing_[ringIndex][initSlice1Idx * discontinuousSliceSize + discontinuousSliceIdx];
    srcSubInit = DeviceMem::create(static_cast<u8*>(opInfo_->inputAddr) + srcInitSlice1.offset, srcInitSlice1.size);
    const Slice& dstInitSlice1
        = multRingsSlices_[ringIndex][initSlice1Idx * discontinuousSliceSize + discontinuousSliceIdx];
    dstSubInit = inputMem_.range(dstInitSlice1.offset, dstInitSlice1.size);
    // 第-1步并发
    if (rankSize == TWO_RANK_SIZE && opInfo_->outputAddr != nullptr) {
        HCCL_DEBUG(
            "Memcpy operation: step[-1] stream[main] src rank[%u] starts to copy(rcv) offset[%llu], size[%llu] on "
            "userMemInput to offset[%llu], size[%llu] on userMemOut_",
            userRank_, srcInitSlice1.offset, srcInitSlice1.size, lastStepOffsets_[ringIndex], dstInitSlice1.size);
        dstInit = DeviceMem::create(
            static_cast<u8*>(opInfo_->outputAddr) + lastStepOffsets_[ringIndex], dstInitSlice1.size);
    } else {
        HCCL_DEBUG(
            "Memcpy operation: step[-1] stream[main] src rank[%u] starts to copy(rcv) offset[%llu], size[%llu] on "
            "userMemInput to offset[%llu], size[%llu] on CCL",
            userRank_, srcInitSlice1.offset, srcInitSlice1.size, dstInitSlice1.offset, dstInitSlice1.size);
    }
    HCCL_DEBUG(
        "Memcpy operation: step[-1] stream[sub] src rank[%u] starts to copy(rcv) offset[%llu], "
        "size[%llu] on userMemInput to offset[%llu], size[%llu] on CCL",
        userRank_, srcInitSlice0.offset, srcInitSlice0.size, dstInitSlice0.offset, dstInitSlice0.size);
    return HCCL_SUCCESS;
}

HcclResult
AlignedReduceScatterDoubleRing::MemcpyInitSlicesOnMainStreams(u64 ringIndex, DeviceMem& dstInit, DeviceMem& srcInit)
{
    if (ringIndex == 1) {
        CHK_RET(MainWaitSub());
        CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
        CHK_RET(MainRecordSub());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstInit, srcInit, stream_));
    } else {
        CHK_RET(LocalNotify::Post(subStreams_[0], dispatcher_, mainSignals_[0], profilerInput_.stage));
        CHK_RET(LocalNotify::Wait(subStreams_[0], dispatcher_, subSignals_[0], profilerInput_.stage));
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstInit, srcInit, subStreams_[0]));
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRing::MemcpyInitSlices(
    u64 ringIndex, DeviceMem& dstInit, DeviceMem& srcInit, DeviceMem& dstSubInit, DeviceMem& srcSubInit)
{
    CHK_RET(MemcpyInitSlicesOnMainStreams(ringIndex, dstInit, srcInit));
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB && (!disableDMAReduce_)) {
        HCCL_DEBUG("[AlignedReduceScatterDoubleRing][MemcpyInitSlices] no graph mode");
        CHK_RET(LocalNotify::Post(
            subStreams_[ringIndex + 1], dispatcher_, mainSignals_[ringIndex + 1], profilerInput_.stage));
        CHK_RET(LocalNotify::Wait(
            subStreams_[ringIndex + 1], dispatcher_, subSignals_[ringIndex + 1], profilerInput_.stage));
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstSubInit, srcSubInit, subStreams_[ringIndex + 1]));
    } else {
        HCCL_DEBUG("[AlignedReduceScatterDoubleRing][MemcpyInitSlices] graph mode");
        CHK_RET(MemcpyInitSlicesOnMainStreams(ringIndex, dstSubInit, srcSubInit));
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRing::RunInitStep(const u32 rank, const u32 rankSize)
{
    // 主环初始indexes
    u32 initSlice0Idx = (rankSize - rank - 1 + rankSize) % rankSize;
    u32 initSlice1Idx = (rankSize - rank - DMA_REDUCE_TWO_OFFSET + rankSize) % rankSize;
    // 从环初始indexes
    u32 subInitSlice0Idx = (rank + rankSize - 1) % rankSize;
    u32 subInitSlice1Idx = (rank + rankSize - DMA_REDUCE_TWO_OFFSET) % rankSize;
    u32 discontinuousSliceSize = multRingsSlices_[ALIGNED_SUB_RING_INDEX].size() / rankSize;
    DeviceMem dstInit;
    DeviceMem srcInit;
    DeviceMem dstSubInit;
    DeviceMem srcSubInit;
    DeviceMem subDstInit;
    DeviceMem subSrcInit;
    DeviceMem subDstSubInit;
    DeviceMem subSrcSubInit;
    for (u32 discontinuousSliceIdx = 0; discontinuousSliceIdx < discontinuousSliceSize; discontinuousSliceIdx++) {
        CHK_RET(PrepareInitSlices(
            rankSize, ALIGNED_SUB_RING_INDEX, discontinuousSliceSize, discontinuousSliceIdx, subInitSlice0Idx,
            subInitSlice1Idx, subDstInit, subSrcInit, subDstSubInit, subSrcSubInit));
        CHK_RET(PrepareInitSlices(
            rankSize, ALIGNED_MAIN_RING_INDEX, discontinuousSliceSize, discontinuousSliceIdx, initSlice0Idx,
            initSlice1Idx, dstInit, srcInit, dstSubInit, srcSubInit));
        HCCL_DEBUG("Memcpy operation: step[-1] starts on ring[%u]", ALIGNED_SUB_RING_INDEX);
        CHK_RET(MemcpyInitSlices(ALIGNED_SUB_RING_INDEX, subDstInit, subSrcInit, subDstSubInit, subSrcSubInit));
        HCCL_DEBUG("Memcpy operation: step[-1] starts on ring[%u]", ALIGNED_MAIN_RING_INDEX);
        CHK_RET(MemcpyInitSlices(ALIGNED_MAIN_RING_INDEX, dstInit, srcInit, dstSubInit, srcSubInit));
    }
    return HCCL_SUCCESS;
}

HcclResult
AlignedReduceScatterDoubleRing::PrepareRunMainStream(u32 ringIndex, Stream& stream, LINK& preLink, LINK& nextLink)
{
    HCCL_DEBUG("AlignedReduceScatterDoubleRing PrepareRunMainStream start");
    if (ringIndex == 1) {
        stream = stream_;
        preLink = rightLink_;
        nextLink = leftLink_;
    } else {
        stream = subStreams_[0];
        preLink = leftLink_;
        nextLink = rightLink_;
    }
    HCCL_DEBUG("AlignedReduceScatterDoubleRing PrepareRunMainStream end");
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRing::PreSync(const u32 ringIndex)
{
    if (ringIndex == 1) {
        CHK_RET(MainWaitSub());
        CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
        CHK_RET(MainRecordSub());
    } else {
        CHK_RET(LocalNotify::Post(subStreams_[0], dispatcher_, mainSignals_[0], profilerInput_.stage));
        CHK_RET(LocalNotify::Wait(subStreams_[0], dispatcher_, subSignals_[0], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRing::PrepareDeviceMems(
    const u32 step, const u32 ringIndex, const u32 rankSize, const u32 txSliceIdx, const u32 rxSliceIdx,
    const u32 subSliceIdx, std::vector<SenderMemoryInfo>& txReduceMems, std::vector<ReducerMemoryInfo>& rxReduceMems,
    std::vector<DeviceMem>& localSrcMems, std::vector<DeviceMem>& localDstMems)
{
    u32 sliceSize = multRingsSlices_[ringIndex].size() / rankSize;
    for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
        const Slice& rxSlice = multRingsSlices_[ringIndex][rxSliceIdx * sliceSize + sliceIdx];
        const Slice& cclSlice = multRingsSlices_[ringIndex][subSliceIdx * sliceSize + sliceIdx];
        const Slice& txSlice = multRingsSlices_[ringIndex][txSliceIdx * sliceSize + sliceIdx];
        const Slice& subSlice = userMemInputSlicesOfDoubleRing_[ringIndex][subSliceIdx * sliceSize + sliceIdx];
        // PrepareReduceDeviceMems
        // Ack
        DeviceMem dst;
        if (step == rankSize - DMA_REDUCE_TWO_OFFSET && opInfo_->outputAddr != nullptr) {
            HCCL_DEBUG(
                "Reduce operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
                "at userMemOut_",
                step, userRank_, lastStepOffsets_[ringIndex], rxSlice.size);
            dst = DeviceMem::create(static_cast<u8*>(opInfo_->outputAddr) + lastStepOffsets_[ringIndex], rxSlice.size);
        } else {
            HCCL_DEBUG(
                "Reduce operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
                "at inputMem_",
                step, userRank_, rxSlice.offset, rxSlice.size);
            dst = inputMem_.range(rxSlice.offset, rxSlice.size);
        }
        // 在inline reduce场景, 需要利用scratchMem_暂存
        DeviceMem srcMemTemp = scratchMem_.range(rxSlice.offset, rxSlice.size);
        DeviceMem srcMem = inputMem_.range(txSlice.offset, txSlice.size);
        HCCL_DEBUG(
            "Reduce operation: step[%u] stream[main], receiver starts to rcv offset[%llu], size[%llu]", step,
            txSlice.offset, txSlice.size);
        rxReduceMems.emplace_back(ReducerMemoryInfo{baseOffset_ + rxSlice.offset, dst, dst, srcMemTemp});
        txReduceMems.emplace_back(SenderMemoryInfo{baseOffset_ + txSlice.offset, srcMem});

        // PrepareLocalCopyDeviceMems
        DeviceMem localSrt;
        DeviceMem localDst;
        if (step == rankSize - DMA_REDUCE_TWO_OFFSET) {
            // do nothing
        } else if (step == rankSize - DMA_REDUCE_THREE_OFFSET && opInfo_->outputAddr != nullptr) {
            HCCL_DEBUG(
                "Memcpy operation: step[%u] subStream[%u], src rank[%u] sends offset[%llu], size[%llu], "
                "dst rank[%u] starts to rcv offset[%llu], size[%llu], "
                "from userMemIn_ to userMemOut_",
                step, ringIndex + 1, userRank_, subSlice.offset, subSlice.size, userRank_, lastStepOffsets_[ringIndex],
                subSlice.size);
            localSrt = DeviceMem::create(static_cast<u8*>(opInfo_->inputAddr) + subSlice.offset, subSlice.size);
            localDst
                = DeviceMem::create(static_cast<u8*>(opInfo_->outputAddr) + lastStepOffsets_[ringIndex], subSlice.size);
        } else {
            HCCL_DEBUG(
                "Memcpy operation: step[%u] subStream[%u], src rank[%u] sends offset[%llu], size[%llu], "
                "dst rank[%u] starts to rcv offset[%llu], size[%llu], "
                "from userMemIn_ to inputMem_",
                step, ringIndex + 1, userRank_, subSlice.offset, subSlice.size, userRank_, cclSlice.offset,
                cclSlice.size);
            localSrt = DeviceMem::create(static_cast<u8*>(opInfo_->inputAddr) + subSlice.offset, subSlice.size);
            localDst = inputMem_.range(cclSlice.offset, cclSlice.size);
        }
        localSrcMems.emplace_back(localSrt);
        localDstMems.emplace_back(localDst);
    }
    return HCCL_SUCCESS;
}

HcclResult
AlignedReduceScatterDoubleRing::RxAsyncMemcpy(const u32 ringIndex, RxMemoryInfo& mem, Stream& stream, const LINK& link)
{
    // PreSync
    CHK_RET(PreSync(ringIndex));
    CHK_PTR_NULL(mem.dst);
    void* srcMemPtr = nullptr;
    CHK_RET(link->GetRemoteMem(mem.srcMemType, &srcMemPtr));

    DeviceMem srcDevMem(static_cast<s8*>(srcMemPtr) + mem.srcOffset, mem.len);
    DeviceMem dstDevMem(static_cast<s8*>(mem.dst), mem.len);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstDevMem, srcDevMem, stream, link->GetRemoteRank(), link->GetLinkType()));
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRing::ReducerRun(
    const u32 ringIndex, const HcclDispatcher dispatcher, const LINK& link, ReducerMemoryInfo& reduceMem,
    Stream& stream)
{
    CHK_PTR_NULL(stream.ptr());
    bool isSpInlineReduce = link->IsSpInlineReduce();
    HcclResult ret = HCCL_SUCCESS;
    if (isSpInlineReduce && static_cast<bool>((INLINE_REDUCE_BITMASK & reduceAttr_))) {
        void* remoteMem = nullptr;
        CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &remoteMem));
        const u64 dataBytes = reduceMem.remoteRcvTemp.size();
        CHK_RET(PreSync(ringIndex));
        CHK_RET(HcclReduceAsync(
            dispatcher, static_cast<s8*>(remoteMem) + reduceMem.remoteMemOffset, dataBytes / SIZE_TABLE[dataType_],
            dataType_, reductionOp_, stream, reduceMem.localsrc.ptr(), link->GetRemoteRank(), link->GetLinkType(),
            INLINE_REDUCE_BIT));

        if (reduceMem.localsrc != reduceMem.localdst) {
            ret = HcclD2DMemcpyAsync(dispatcher, reduceMem.localdst, reduceMem.localsrc, stream);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[AlignedReduceScatterDoubleRing][Run]memcpy_async localSrc[%p] localDst[%p] failed",
                    reduceMem.localsrc.ptr(), reduceMem.localdst.ptr()),
                ret);
        }
    } else {
        RxMemoryInfo rxMem = RxMemoryInfo{
            UserMemType::INPUT_MEM, reduceMem.remoteMemOffset, reduceMem.remoteRcvTemp.ptr(),
            reduceMem.remoteRcvTemp.size()};

        u64 dataCount = reduceMem.localdst.size() / SIZE_TABLE[dataType_];
        DeviceMem reduceSrc = (reduceMem.localsrc == reduceMem.localdst) ? reduceMem.remoteRcvTemp : reduceMem.localsrc;
        RxWithReduceMemoryInfo rxWithReduceMem = RxWithReduceMemoryInfo{
            UserMemType::INPUT_MEM,
            reduceMem.remoteMemOffset,
            reduceMem.remoteRcvTemp.ptr(),
            reduceMem.remoteRcvTemp.size(),
            reduceSrc.ptr(),
            reduceMem.localdst.ptr(),
            dataCount};
        CHK_RET(RxAsyncMemcpy(ringIndex, rxMem, stream, link));
        RxWithReduceMemoryInfo& rxReduceMem = rxWithReduceMem;
        if (ringIndex == ALIGNED_SUB_RING_INDEX) {
            CHK_PRT_RET(
                stream != subStreams_[0],
                HCCL_ERROR("[%s] subStreams_[0] should be used for ringIndex=%d", __func__, ALIGNED_SUB_RING_INDEX),
                HCCL_E_INTERNAL);
            CHK_RET(LocalNotify::Post(subStreams_[0], dispatcher_, mainSignals_[0], profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[0], profilerInput_.stage));
        }
        CHK_RET(HcclReduceAsync(
            dispatcher, rxReduceMem.reduceSrc, rxReduceMem.reduceDataCount, dataType_, reductionOp_, stream_,
            rxReduceMem.reduceDst, INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP, reduceAttr_));
        if (ringIndex == ALIGNED_SUB_RING_INDEX) {
            CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[0], profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(subStreams_[0], dispatcher_, subSignals_[0], profilerInput_.stage));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRing::LocalMemcpy(
    const u32 step, const u32 rankSize, const u32 ringIndex, DeviceMem& localSrcMem, DeviceMem& localDstMem)
{
    // 通过校验流数判断是单算子模式还是图模式
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB && (!disableDMAReduce_)) {
        CHK_RET(LocalNotify::Post(
            subStreams_[ringIndex + 1], dispatcher_, mainSignals_[ringIndex + 1], profilerInput_.stage));
        CHK_RET(LocalNotify::Wait(
            subStreams_[ringIndex + 1], dispatcher_, subSignals_[ringIndex + 1], profilerInput_.stage));
        if (localSrcMem != localDstMem && step != rankSize - DMA_REDUCE_TWO_OFFSET) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, localDstMem, localSrcMem, subStreams_[ringIndex + 1]));
        }
    } else {
        // 图模式
        CHK_RET(PreSync(ringIndex));
        if (localSrcMem != localDstMem && step != rankSize - DMA_REDUCE_TWO_OFFSET) {
            if (ringIndex == 1) {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, localDstMem, localSrcMem, stream_));
            } else {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, localDstMem, localSrcMem, subStreams_[0]));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRing::RunSubStream(
    const u32 step, const u32 rankSize, u32 ringIndex, std::vector<DeviceMem>& localSrcMems,
    std::vector<DeviceMem>& localDstMems)
{
    for (u32 sliceIdx = 0; sliceIdx < localSrcMems.size(); sliceIdx++) {
        CHK_RET(LocalNotify::Post(
            subStreams_[ringIndex + 1], dispatcher_, mainSignals_[ringIndex + 1], profilerInput_.stage));
        CHK_RET(LocalNotify::Wait(
            subStreams_[ringIndex + 1], dispatcher_, subSignals_[ringIndex + 1], profilerInput_.stage));
        if (step != rankSize - DMA_REDUCE_TWO_OFFSET) {
            CHK_RET(HcclD2DMemcpyAsync(
                dispatcher_, localDstMems[sliceIdx], localSrcMems[sliceIdx], subStreams_[ringIndex + 1]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRing::RunAllStreams(
    const u32 step, const u32 rankSize, std::vector<SenderMemoryInfo>& mainTxReduceMems,
    std::vector<ReducerMemoryInfo>& mainRxReduceMems, std::vector<SenderMemoryInfo>& subTxReduceMems,
    std::vector<ReducerMemoryInfo>& subRxReduceMems, std::vector<DeviceMem>& mainLocalSrcMems,
    std::vector<DeviceMem>& mainLocalDstMems, std::vector<DeviceMem>& subLocalSrcMems,
    std::vector<DeviceMem>& subLocalDstMems)
{
    (void)subTxReduceMems;
    (void)mainTxReduceMems;
    Stream mainStream;
    LINK mainPreLink;
    LINK mainNextLink;
    Stream subStream;
    LINK subPreLink;
    LINK subNextLink;
    CHK_RET(PrepareRunMainStream(ALIGNED_MAIN_RING_INDEX, mainStream, mainPreLink, mainNextLink));
    HCCL_DEBUG(
        "Reduce: step[%u] ring[%u], src rank[%u] starts to send slice to dst rank[%u]", step, ALIGNED_MAIN_RING_INDEX,
        mainPreLink->GetRemoteRank(), mainNextLink->GetRemoteRank());
    CHK_RET(PrepareRunMainStream(ALIGNED_SUB_RING_INDEX, subStream, subPreLink, subNextLink));
    HCCL_DEBUG(
        "Reduce: step[%u] ring[%u], src rank[%u] starts to send slice to dst rank[%u]", step, ALIGNED_SUB_RING_INDEX,
        subPreLink->GetRemoteRank(), subNextLink->GetRemoteRank());

    CHK_RET(mainNextLink->TxAck(mainStream));
    CHK_RET(mainPreLink->RxAck(mainStream));
    CHK_RET(subNextLink->TxAck(subStream));
    CHK_RET(subPreLink->RxAck(subStream));

    u32 sliceSize = multRingsSlices_[ALIGNED_MAIN_RING_INDEX].size() / rankSize;
    for (u32 memIdx = 0; memIdx < sliceSize; memIdx++) {
        CHK_RET(ReducerRun(ALIGNED_MAIN_RING_INDEX, dispatcher_, mainPreLink, mainRxReduceMems[memIdx], mainStream));
        CHK_RET(ReducerRun(ALIGNED_SUB_RING_INDEX, dispatcher_, subPreLink, subRxReduceMems[memIdx], subStream));
        CHK_RET(
            LocalMemcpy(step, rankSize, ALIGNED_MAIN_RING_INDEX, mainLocalSrcMems[memIdx], mainLocalDstMems[memIdx]));
        CHK_RET(LocalMemcpy(step, rankSize, ALIGNED_SUB_RING_INDEX, subLocalSrcMems[memIdx], subLocalDstMems[memIdx]));
    }
    CHK_RET(mainPreLink->TxDataSignal(mainStream));
    CHK_RET(mainNextLink->RxDataSignal(mainStream));
    CHK_RET(subPreLink->TxDataSignal(subStream));
    CHK_RET(subNextLink->RxDataSignal(subStream));
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRing::PreRunStreams(
    const u32 step, const u32 rankSize, const u32 txSliceIdxMain, const u32 rxSliceIdxMain, const u32 subSliceIdxMain,
    const u32 txSliceIdxSub, const u32 rxSliceIdxSub, const u32 subSliceIdxSub,
    std::vector<SenderMemoryInfo>& txReduceMemsMain, std::vector<ReducerMemoryInfo>& rxReduceMemsMain,
    std::vector<SenderMemoryInfo>& txReduceMemsSub, std::vector<ReducerMemoryInfo>& rxReduceMemsSub,
    std::vector<DeviceMem>& localSrcMemsMain, std::vector<DeviceMem>& localDstMemsMain,
    std::vector<DeviceMem>& localSrcMemsSub, std::vector<DeviceMem>& localDstMemsSub)
{
    CHK_RET(PrepareDeviceMems(
        step, ALIGNED_MAIN_RING_INDEX, rankSize, txSliceIdxMain, rxSliceIdxMain, subSliceIdxMain, txReduceMemsMain,
        rxReduceMemsMain, localSrcMemsMain, localDstMemsMain));
    CHK_RET(PrepareDeviceMems(
        step, ALIGNED_SUB_RING_INDEX, rankSize, txSliceIdxSub, rxSliceIdxSub, subSliceIdxSub, txReduceMemsSub,
        rxReduceMemsSub, localSrcMemsSub, localDstMemsSub));
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRing::RunReduceScatter(const u32 rank, const u32 rankSize)
{
    HCCL_INFO("AlignedReduceScatterDoubleRing starts, the input param rank[%u]", rank);
    // 空拷贝用于后续操作附着
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    // 主环主流通知从环主流开始通信
    CHK_RET(MainRecordSub());
    // 从环主流等待主环主流通知
    CHK_RET(SubWaitMain());
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
        CHK_RET(ExecEmptyTasks());
        CHK_RET(RunInitStep(rank, rankSize));
    }
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    CHK_RET(ExecEmptyTasks());

    // 例如rank[0,1,2,3]中，rank0的rxSliceIdx = 2，txSliceIdx = 3, subSliceIdx = 1
    // 从环初始indexes
    u32 txSliceIdxSub = (rank + rankSize - 1) % rankSize;
    u32 rxSliceIdxSub = (rank + rankSize - DMA_REDUCE_TWO_OFFSET) % rankSize;
    u32 subSliceIdxSub = (rank + rankSize - DMA_REDUCE_THREE_OFFSET) % rankSize;
    HCCL_DEBUG(
        "[RunReduceScatter]txSliceIdxSub is [%u], rxSliceIdxSub is [%u], subSliceIdxSub is [%u]", txSliceIdxSub,
        rxSliceIdxSub, subSliceIdxSub);
    // 主环初始indexes
    u32 txSliceIdxMain = (rankSize - rank - 1 + rankSize) % rankSize;
    u32 rxSliceIdxMain = (rankSize - rank - DMA_REDUCE_TWO_OFFSET + rankSize) % rankSize;
    u32 subSliceIdxMain = (rankSize - rank - DMA_REDUCE_THREE_OFFSET + rankSize) % rankSize;

    for (u32 step = 0; step < rankSize - 1; step++) {
        // 并发
        std::vector<SenderMemoryInfo> txReduceMemsMain;
        std::vector<ReducerMemoryInfo> rxReduceMemsMain;
        std::vector<SenderMemoryInfo> txReduceMemsSub;
        std::vector<ReducerMemoryInfo> rxReduceMemsSub;
        std::vector<DeviceMem> localDstMemsMain;
        std::vector<DeviceMem> localSrcMemsMain;
        std::vector<DeviceMem> localSrcMemsSub;
        std::vector<DeviceMem> localDstMemsSub;
        CHK_RET(PreRunStreams(
            step, rankSize, txSliceIdxMain, rxSliceIdxMain, subSliceIdxMain, txSliceIdxSub, rxSliceIdxSub,
            subSliceIdxSub, txReduceMemsMain, rxReduceMemsMain, txReduceMemsSub, rxReduceMemsSub, localSrcMemsMain,
            localDstMemsMain, localSrcMemsSub, localDstMemsSub));
        CHK_RET(RunAllStreams(
            step, rankSize, txReduceMemsMain, rxReduceMemsMain, txReduceMemsSub, rxReduceMemsSub, localSrcMemsMain,
            localDstMemsMain, localSrcMemsSub, localDstMemsSub));
        // 更新索引
        txSliceIdxSub = (txSliceIdxSub + rankSize - 1) % rankSize;
        rxSliceIdxSub = (rxSliceIdxSub + rankSize - 1) % rankSize;
        subSliceIdxSub = (subSliceIdxSub + rankSize - 1) % rankSize;
        txSliceIdxMain = (txSliceIdxMain + rankSize - 1) % rankSize;
        rxSliceIdxMain = (rxSliceIdxMain + rankSize - 1) % rankSize;
        subSliceIdxMain = (subSliceIdxMain + rankSize - 1) % rankSize;
    }
    // 从环主流通知主环主流通信完成
    CHK_RET(SubRecordMain());
    // 主环主流等待从环主流通知
    CHK_RET(MainWaitSub());
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    CHK_RET(ExecEmptyTasks());
    HCCL_INFO("AlignedReduceScatterDoubleRing finished to RunReduceScatter");
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRing::GetActiveSubstreamNum(u32& activeSubstreamNum)
{
    activeSubstreamNum = subStreams_.size();
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB && disableDMAReduce_) {
        if (subStreams_.size() <= 2) {
            HCCL_ERROR(
                "[AlignedReduceScatterDoubleRing][GetActiveSubstreamNum]subStreams_.size()[%zu] <= 2",
                subStreams_.size());
            return HCCL_E_PARA;
        }
        activeSubstreamNum = subStreams_.size() - 2;
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRing::ExecEmptyTasks()
{
    u32 activeSubstreamNum = 0;
    CHK_RET(GetActiveSubstreamNum(activeSubstreamNum));
    for (u32 signalIndex = 0; signalIndex < activeSubstreamNum; signalIndex++) {
        CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, subStreams_[signalIndex], dispatcher_));
    }
    return HCCL_SUCCESS;
}

// 主流通知从流干活
HcclResult AlignedReduceScatterDoubleRing::MainRecordSub()
{
    u32 activeSubstreamNum = 0;
    CHK_RET(GetActiveSubstreamNum(activeSubstreamNum));
    for (u32 signalIndex = 0; signalIndex < activeSubstreamNum; signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 从流等待主流
HcclResult AlignedReduceScatterDoubleRing::SubWaitMain()
{
    u32 activeSubstreamNum = 0;
    CHK_RET(GetActiveSubstreamNum(activeSubstreamNum));
    for (u32 streamIndex = 0; streamIndex < activeSubstreamNum; streamIndex++) {
        CHK_RET(
            LocalNotify::Wait(subStreams_[streamIndex], dispatcher_, subSignals_[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 主流等待从流
HcclResult AlignedReduceScatterDoubleRing::MainWaitSub()
{
    u32 activeSubstreamNum = 0;
    CHK_RET(GetActiveSubstreamNum(activeSubstreamNum));
    for (u32 signalIndex = 0; signalIndex < activeSubstreamNum; signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 从流告诉主流活干完了
HcclResult AlignedReduceScatterDoubleRing::SubRecordMain()
{
    u32 activeSubstreamNum = 0;
    CHK_RET(GetActiveSubstreamNum(activeSubstreamNum));
    for (u32 streamIndex = 0; streamIndex < activeSubstreamNum; streamIndex++) {
        CHK_RET(
            LocalNotify::Post(subStreams_[streamIndex], dispatcher_, mainSignals_[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_DB_RING, AlignedReduceScatterDoubleRing);
} // namespace hccl
