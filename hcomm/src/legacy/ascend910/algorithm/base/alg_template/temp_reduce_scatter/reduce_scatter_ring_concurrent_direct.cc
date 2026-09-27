/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_ring_concurrent_direct.h"
#include "alg_template_register.h"

namespace hccl {
ReduceScatterRingConcurrentDirect::ReduceScatterRingConcurrentDirect(const HcclDispatcher dispatcher)
    : AlgTemplateBase(dispatcher)
{}

ReduceScatterRingConcurrentDirect::~ReduceScatterRingConcurrentDirect() {}

HcclResult ReduceScatterRingConcurrentDirect::Prepare(
    const u64 reduceAttrBitMap, const HcomCollOpInfo* opInfo, const u32 userRank, std::vector<Stream>& subStreams,
    const std::vector<std::shared_ptr<LocalNotify>>& mainSignals,
    const std::vector<std::shared_ptr<LocalNotify>>& subSignals, const std::vector<u32>& ringsOrder,
    const std::vector<Slice>& userMemInputSlices, bool isSdma)
{
    reduceAttr_ = reduceAttrBitMap;
    opInfo_ = opInfo;
    userRank_ = userRank;
    subStreams_ = subStreams;
    mainSignals_ = mainSignals;
    subSignals_ = subSignals;
    ringsOrder_ = ringsOrder;
    userSlices_ = userMemInputSlices;
    isSdma_ = isSdma;
    return HCCL_SUCCESS;
}

// reduce scatter ring direct算法的函数入口
HcclResult
ReduceScatterRingConcurrentDirect::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 基本的检查
    CHK_RET(CheckParameters(rank, rankSize, links));

    // 判断rank_size == 1的情况，并拷贝
    if (rankSize == 1) {
        CHK_RET(OneRankMemcpy());
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("ReduceScatterRingConcurrentDirect starts: rank[%u]", rank);
    // 收集本地mem信息
    CHK_RET(InitSenderReducer());

    // 收集邻居信息
    CHK_RET(GetInitializedNeighborLinks(rank, rankSize, links));

    // 填充slice_
    CHK_RET(SetSlices(rank, rankSize));

    // 运行reduce-scatter, ring算法
    CHK_RET(RunReduceScatter(rank, rankSize));

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(leftLink_, rightLink_));
    }

    CHK_RET(LaunchTaskExtend(dispatcher_, stream_, subStreams_));

    HCCL_INFO("ReduceScatterRingConcurrentDirect finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

HcclResult
ReduceScatterRingConcurrentDirect::CheckParameters(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    CHK_PTR_NULL(opInfo_);
    CHK_RET(CheckConcurrentDirectParameters(rank, rankSize, links));
    // 判断subStreams数量是否正确
    CHK_PRT_RET(
        subStreams_.size() < 1,
        HCCL_ERROR("[ReduceScatterRingConcurrentDirect] subStreams size[%u] is less than 1", subStreams_.size()),
        HCCL_E_PARA);
    for (auto& s : subStreams_) {
        CHK_PTR_NULL(s.ptr());
    }
    // 判断mainSignals数量是否正确
    CHK_PRT_RET(
        mainSignals_.size() < 1,
        HCCL_ERROR("[ReduceScatterRingConcurrentDirect] mainSignals size[%u] is less than 1", mainSignals_.size()),
        HCCL_E_PARA);
    // 判断subSignals数量是否正确
    CHK_PRT_RET(
        subSignals_.size() < 1,
        HCCL_ERROR("[ReduceScatterRingConcurrentDirect] subSignals size[%u] is less than 1", subSignals_.size()),
        HCCL_E_PARA);
    // 判断ringsOrder数量是否正确
    CHK_PRT_RET(
        ringsOrder_.size() != rankSize,
        HCCL_ERROR(
            "[ReduceScatterRingConcurrentDirect] ringsOrder size[%u] is not equal to rank size[%u]", ringsOrder_.size(),
            rankSize),
        HCCL_E_PARA);
    // 判断userMemInputSlices数量是否正确
    CHK_PRT_RET(
        userSlices_.size() % rankSize != 0,
        HCCL_ERROR(
            "[ReduceScatterRingConcurrentDirect] userMemInputSlices size[%u] cannot be divided by size[%u]",
            userSlices_.size(), rankSize),
        HCCL_E_PARA);
    HCCL_INFO("ReduceScatterRingConcurrentDirect CheckParameters success");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::OneRankMemcpy()
{
    for (u32 sliceIdx = 0; sliceIdx < slices_.size(); sliceIdx++) {
        const Slice& srcSlice = userSlices_[sliceIdx];
        const Slice& dstSlice = slices_[sliceIdx];
        DeviceMem src = DeviceMem::create(static_cast<u8*>(opInfo_->inputAddr) + srcSlice.offset, srcSlice.size);
        DeviceMem dst;
        if (opInfo_->outputAddr != nullptr) {
            // opInfo_->outputAddr != nullptr指示要将输出发送至user output
            u64 stepOffset = slices_[ringsOrder_[0]].offset;
            HCCL_DEBUG(
                "[Memcpy operation] stream[main], rank[%u] starts to rcv offset[%llu], size[%llu] at userMemOut_",
                userRank_, stepOffset, dstSlice.size);
            dst = DeviceMem::create(static_cast<u8*>(opInfo_->outputAddr) + stepOffset, dstSlice.size);
        } else {
            // opInfo_->outputAddr == nullptr指示要将输出发送至CCL buffer
            HCCL_DEBUG(
                "[Memcpy operation] stream[main], rank[%u] starts to rcv offset[%llu], size[%llu] at outputMem_",
                userRank_, dstSlice.offset, dstSlice.size);
            dst = outputMem_.range(dstSlice.offset, dstSlice.size);
        }
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::InitSenderReducer()
{
    // 创建reducer & sender
    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(senderInfo_);

    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(reducerInfo_);
    HCCL_INFO("ReduceScatterRingConcurrentDirect finished to InitSenderReducer");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::GetInitializedNeighborLinks(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 收集左邻居信息
    leftLink_ = links[(rank + rankSize - 1) % rankSize];
    CHK_SMART_PTR_NULL(leftLink_);

    // 收集右邻居信息
    rightLink_ = links[(rank + 1) % rankSize];
    CHK_SMART_PTR_NULL(rightLink_);
    HCCL_INFO("ReduceScatterRingConcurrentDirect finished to GetInitializedNeighborLinks");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::SetSlices(const u32 rank, const u32 rankSize)
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
    if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {
        for (u32 i = 0; i < slices_.size(); i++) {
            HCCL_DEBUG(
                "[ReduceScatterRingConcurrentDirect][SetSlices]rank[%u], slices[%u].offset=[%llu], "
                "slices[%u].size=[%llu]",
                rank, i, slices_[i].offset, i, slices_[i].size);
        }
    }
    // 最后一步搬到userMemOut_的offset, 不同的ring环offset不一样
    lastStepOffset_ = slices_[ringsOrder_[0]].offset;
    HCCL_INFO("ReduceScatterRingConcurrentDirect finished to SetSlices");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::RunInitStep(const u32 rank, const u32 rankSize)
{
    // 例如rank[0,1,2,3]中，rank0的rxSliceIdx = 2，txSliceIdx = 3
    u32 initSlice0Idx = 0;
    u32 initSlice1Idx = 0;
    initSlice0Idx = (rank + rankSize - 1) % rankSize;
    initSlice1Idx = (rank + rankSize - DMA_REDUCE_TWO_OFFSET) % rankSize;
    u32 sliceSize = slices_.size() / rankSize;

    for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
        // 第-1步，片内将部分数据从userIn搬到cclIn
        const Slice& srcInitSlice0 = userSlices_[initSlice0Idx * sliceSize + sliceIdx];
        DeviceMem srcSubInit
            = DeviceMem::create(static_cast<u8*>(opInfo_->inputAddr) + srcInitSlice0.offset, srcInitSlice0.size);
        const Slice& dstInitSlice0 = slices_[initSlice0Idx * sliceSize + sliceIdx];
        DeviceMem dstSubInit = inputMem_.range(dstInitSlice0.offset, dstInitSlice0.size);
        const Slice& srcInitSlice1 = userSlices_[initSlice1Idx * sliceSize + sliceIdx];
        DeviceMem srcInit
            = DeviceMem::create(static_cast<u8*>(opInfo_->inputAddr) + srcInitSlice1.offset, srcInitSlice1.size);
        const Slice& dstInitSlice1 = slices_[initSlice1Idx * sliceSize + sliceIdx];
        DeviceMem dstInit = inputMem_.range(dstInitSlice1.offset, dstInitSlice1.size);
        // 第-1步并发
        CHK_RET(MainRecordSub()); // 主流通知从流开始通信
        CHK_RET(SubWaitMain());   // 从流等待主流通知
        if (rankSize == TWO_RANK_SIZE && opInfo_->outputAddr != nullptr) {
            HCCL_DEBUG(
                "Memcpy operation: step[-1] stream[main] src rank[%u] starts to copy(rcv) offset[%llu], size[%llu] on "
                "userMemInput to offset[%llu], size[%llu] on userMemOut_",
                userRank_, srcInitSlice1.offset, srcInitSlice1.size, lastStepOffset_, dstInitSlice1.size);
            dstInit = DeviceMem::create(static_cast<u8*>(opInfo_->outputAddr) + lastStepOffset_, dstInitSlice1.size);
        } else {
            HCCL_DEBUG(
                "Memcpy operation: step[-1] stream[main] src rank[%u] starts to copy(rcv) offset[%llu], size[%llu] on "
                "userMemInput to offset[%llu], size[%llu] on CCL",
                userRank_, srcInitSlice1.offset, srcInitSlice1.size, dstInitSlice1.offset, dstInitSlice1.size);
        }
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstInit, srcInit, stream_));
        HCCL_DEBUG(
            "Memcpy operation: step[-1] stream[sub] src rank[%u] starts to copy(rcv) offset[%llu], "
            "size[%llu] on userMemInput to offset[%llu], size[%llu] on CCL",
            userRank_, srcInitSlice0.offset, srcInitSlice0.size, dstInitSlice0.offset, dstInitSlice0.size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstSubInit, srcSubInit, subStreams_[0]));
        CHK_RET(SubRecordMain()); // 从流通知主流通信完成
        CHK_RET(MainWaitSub());   // 主流等待从流通知
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::PreSync()
{
    CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[0], profilerInput_.stage));
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[0], profilerInput_.stage));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::ReducerSpInlineSlice(
    const HcclDispatcher dispatcher, const LINK& link, void* remoteMem, ReducerMemoryInfo reduceMem, Stream& stream)
{
    const u64 dataBytes = reduceMem.remoteRcvTemp.size();
    CHK_RET(HcclReduceAsync(
        dispatcher, static_cast<s8*>(remoteMem) + reduceMem.remoteMemOffset, dataBytes / SIZE_TABLE[dataType_],
        dataType_, reductionOp_, stream, reduceMem.localsrc.ptr(), link->GetRemoteRank(), link->GetLinkType(),
        INLINE_REDUCE_BIT));

    if (reduceMem.localsrc != reduceMem.localdst) {
        HcclResult ret = HcclD2DMemcpyAsync(dispatcher, reduceMem.localdst, reduceMem.localsrc, stream);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ReducerRun]memcpy_async localSrc[%p] localDst[%p] failed", reduceMem.localsrc.ptr(),
                reduceMem.localdst.ptr()),
            ret);
    }
    return HCCL_SUCCESS;
}

// 仅rdma场景调用：主流一次性PreSync后批量下发inline reduce任务
HcclResult ReduceScatterRingConcurrentDirect::ReducerRunSpInlineReduce(
    const HcclDispatcher dispatcher, const LINK& link, const std::vector<ReducerMemoryInfo>& reducerMems,
    Stream& stream)
{
    CHK_RET(link->RxDataSignal(stream));
    void* remoteMem = nullptr;
    CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &remoteMem));
    CHK_RET(PreSync());
    for (ReducerMemoryInfo reduceMem : reducerMems) {
        CHK_RET(ReducerSpInlineSlice(dispatcher, link, remoteMem, reduceMem, stream));
    }
    return HCCL_SUCCESS;
}

// sdma场景主流单个slice的远端读任务
HcclResult ReduceScatterRingConcurrentDirect::ReducerSdmaRemoteReadSlice(
    const LINK& link, const ReducerMemoryInfo& reduceMem, Stream& stream) const
{
    RxMemoryInfo mem{
        UserMemType::INPUT_MEM, reduceMem.remoteMemOffset, reduceMem.remoteRcvTemp.ptr(),
        reduceMem.remoteRcvTemp.size()};
    CHK_PTR_NULL(mem.dst);
    void* srcMemPtr = nullptr;
    CHK_RET(link->GetRemoteMem(mem.srcMemType, &srcMemPtr));

    DeviceMem srcDevMem(static_cast<s8*>(srcMemPtr) + mem.srcOffset, mem.len);
    DeviceMem dstDevMem(static_cast<s8*>(mem.dst), mem.len);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstDevMem, srcDevMem, stream, link->GetRemoteRank(), link->GetLinkType()));
    return HCCL_SUCCESS;
}

// 数据接收确认 + 主流批量本地reduce任务
HcclResult ReduceScatterRingConcurrentDirect::ReducerLocalReduceSuffix(
    const HcclDispatcher dispatcher, const LINK& link, const std::vector<ReducerMemoryInfo>& reducerMems,
    Stream& stream)
{
    if (link->GetSupportDataReceivedAck()) {
        CHK_RET(link->DataReceivedAck(stream));
    }
    for (ReducerMemoryInfo reduceMem : reducerMems) {
        u64 dataCount = reduceMem.localdst.size() / SIZE_TABLE[dataType_];
        DeviceMem reduceSrc = (reduceMem.localsrc == reduceMem.localdst) ? reduceMem.remoteRcvTemp : reduceMem.localsrc;
        CHK_RET(HcclReduceAsync(
            dispatcher, reduceSrc.ptr(), dataCount, dataType_, reductionOp_, stream, reduceMem.localdst.ptr(),
            INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP, reduceAttr_));
    }
    return HCCL_SUCCESS;
}

// 仅rdma场景调用：主流一次性PreSync、RxAsync后批量下发本地reduce任务
HcclResult ReduceScatterRingConcurrentDirect::ReducerRunNoSpInlineReduce(
    const HcclDispatcher dispatcher, const LINK& link, const std::vector<ReducerMemoryInfo>& reducerMems,
    Stream& stream)
{
    std::vector<RxMemoryInfo> rxMems;
    for (const ReducerMemoryInfo& reduceMem : reducerMems) {
        rxMems.emplace_back(RxMemoryInfo{
            UserMemType::INPUT_MEM, reduceMem.remoteMemOffset, reduceMem.remoteRcvTemp.ptr(),
            reduceMem.remoteRcvTemp.size()});
    }
    CHK_RET(PreSync());
    CHK_RET(link->RxAsync(rxMems, stream));
    CHK_RET(ReducerLocalReduceSuffix(dispatcher, link, reducerMems, stream));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::ReducerRun(
    const HcclDispatcher dispatcher, const LINK& link, const std::vector<ReducerMemoryInfo>& reducerMems,
    Stream& stream)
{
    CHK_PTR_NULL(stream.ptr());
    bool isSpInlineReduce = link->IsSpInlineReduce();
    if (isSpInlineReduce && static_cast<bool>((INLINE_REDUCE_BITMASK & reduceAttr_))) {
        CHK_RET(ReducerRunSpInlineReduce(dispatcher, link, reducerMems, stream));
    } else {
        CHK_RET(ReducerRunNoSpInlineReduce(dispatcher, link, reducerMems, stream));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::RunMainStreamTx(
    const u32 step, const std::vector<Slice>& txSliceVector, const std::vector<Slice>& rxSliceVector, const u32 rank,
    const u32 rankSize, std::vector<ReducerMemoryInfo>& rxReduceMems)
{
    (void)rank;
    CHK_RET(leftLink_->TxAck(stream_));
    CHK_RET(rightLink_->RxAck(stream_));
    u32 sliceSize = slices_.size() / rankSize;

    // 通信，如果是最后一步，则做消减拷贝
    std::vector<SenderMemoryInfo> txMems;
    DeviceMem dst;
    for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
        // Ack
        HCCL_DEBUG(
            "Reduce operation: step[%u] stream[main], src rank[%u] starts to send offset[%llu] size[%llu] "
            "from leftMem_",
            step, leftLink_->GetRemoteRank(), rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size);
        if (isSdma_ && step == rankSize - DMA_REDUCE_TWO_OFFSET && opInfo_->outputAddr != nullptr) {
            HCCL_DEBUG("[RunReduceScatter] sdma DMAReduce step");
            HCCL_DEBUG(
                "Reduce operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
                "at userMemOut_",
                step, userRank_, lastStepOffset_, rxSliceVector[sliceIdx].size);
            dst = DeviceMem::create(
                static_cast<u8*>(opInfo_->outputAddr) + lastStepOffset_, rxSliceVector[sliceIdx].size);
        } else {
            HCCL_DEBUG(
                "Reduce operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
                "at inputMem_",
                step, userRank_, rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size);
            dst = inputMem_.range(rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size);
            if (!isSdma_ && step == rankSize - DMA_REDUCE_TWO_OFFSET && opInfo_->outputAddr != nullptr) {
                HCCL_DEBUG("[RunReduceScatter] rdma DMAReduce step");
                finalSrc_ = inputMem_.range(rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size);
                finalDst_ = DeviceMem::create(
                    static_cast<u8*>(opInfo_->outputAddr) + lastStepOffset_, rxSliceVector[sliceIdx].size);
            }
        }
        // 在inline reduce场景, 需要利用scratchMem_暂存
        DeviceMem srcMemTemp = scratchMem_.range(rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size);
        DeviceMem srcMem = inputMem_.range(txSliceVector[sliceIdx].offset, txSliceVector[sliceIdx].size);
        HCCL_DEBUG(
            "Reduce operation: step[%u] stream[main], senderInfo_ rank[%u] starts to rcv offset[%llu], "
            "size[%llu]",
            step, rightLink_->GetRemoteRank(), txSliceVector[sliceIdx].offset, txSliceVector[sliceIdx].size);
        rxReduceMems.emplace_back(
            ReducerMemoryInfo{baseOffset_ + rxSliceVector[sliceIdx].offset, dst, dst, srcMemTemp});
        txMems.emplace_back(SenderMemoryInfo{baseOffset_ + txSliceVector[sliceIdx].offset, srcMem});
    }
    CHK_RET(senderInfo_->run(rightLink_, txMems, stream_));
    return HCCL_SUCCESS;
}

// 仅rdma场景调用：主流Tx前缀 + ReducerRun整段下发
HcclResult ReduceScatterRingConcurrentDirect::RunMainStream(
    const u32 step, std::vector<Slice> txSliceVector, std::vector<Slice> rxSliceVector, const u32 rank,
    const u32 rankSize)
{
    std::vector<ReducerMemoryInfo> rxReduceMems;
    CHK_RET(RunMainStreamTx(step, txSliceVector, rxSliceVector, rank, rankSize, rxReduceMems));
    CHK_RET(ReducerRun(dispatcher_, leftLink_, rxReduceMems, stream_));
    return HCCL_SUCCESS;
}

// 从流提前下发Post(mainSignals)，使主流Wait(mainSignals)可在队列未积压时及时通过
HcclResult ReduceScatterRingConcurrentDirect::RunSubStreamPrePost()
{
    CHK_RET(LocalNotify::Post(subStreams_[0], dispatcher_, mainSignals_[0], profilerInput_.stage));
    return HCCL_SUCCESS;
}

// 仅rdma场景调用：从流Wait(subSignals)后批量下发本步拷贝任务
HcclResult ReduceScatterRingConcurrentDirect::RunSubStream(
    const u32 step, std::vector<Slice> subSliceVector, std::vector<Slice> cclSliceVector, const u32 rank,
    const u32 rankSize)
{
    (void)rank;
    CHK_RET(LocalNotify::Wait(subStreams_[0], dispatcher_, subSignals_[0], profilerInput_.stage));
    for (u32 sliceIdx = 0; sliceIdx < subSliceVector.size(); sliceIdx++) {
        CHK_RET(RunSubStreamSlice(step, sliceIdx, subSliceVector, cclSliceVector, rankSize));
    }
    return HCCL_SUCCESS;
}

// 从流单个slice的拷贝任务
HcclResult ReduceScatterRingConcurrentDirect::RunSubStreamSlice(
    const u32 step, const u32 sliceIdx, const std::vector<Slice>& subSliceVector,
    const std::vector<Slice>& cclSliceVector, const u32 rankSize)
{
    HCCL_DEBUG(
        "Memcpy operation: step[%u] stream[sub], src rank[%u] starts to send offset[%llu], size[%llu] "
        "from userMemIn_",
        step, userRank_, subSliceVector[sliceIdx].offset, subSliceVector[sliceIdx].size);
    DeviceMem src = DeviceMem::create(
        static_cast<u8*>(opInfo_->inputAddr) + subSliceVector[sliceIdx].offset, subSliceVector[sliceIdx].size);
    DeviceMem dst;
    if (step == rankSize - DMA_REDUCE_TWO_OFFSET) {
        // do nothing
    } else if (isSdma_ && step == rankSize - DMA_REDUCE_THREE_OFFSET && opInfo_->outputAddr != nullptr) {
        HCCL_DEBUG(
            "Memcpy operation: step[%u] stream[sub], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
            "to userMemOut_",
            step, userRank_, lastStepOffset_, subSliceVector[sliceIdx].size);
        dst = DeviceMem::create(static_cast<u8*>(opInfo_->outputAddr) + lastStepOffset_, subSliceVector[sliceIdx].size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStreams_[0]));
    } else {
        HCCL_DEBUG(
            "Memcpy operation: step[%u] stream[sub], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
            "to inputMem_",
            step, userRank_, cclSliceVector[sliceIdx].offset, cclSliceVector[sliceIdx].size);
        dst = inputMem_.range(cclSliceVector[sliceIdx].offset, cclSliceVector[sliceIdx].size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStreams_[0]));
    }
    return HCCL_SUCCESS;
}

// 仅sdma场景调用：RxDataSignal后，主从流逐slice交替下发，最后补本地reduce后缀
HcclResult ReduceScatterRingConcurrentDirect::RunSdmaStepConcurrent(
    const u32 step, const std::vector<ReducerMemoryInfo>& rxReduceMems, const std::vector<Slice>& subSliceVector,
    const std::vector<Slice>& cclSliceVector, const u32 rank, const u32 rankSize)
{
    (void)rank;
    CHK_RET(leftLink_->RxDataSignal(stream_));
    bool isSpInlineReduce = leftLink_->IsSpInlineReduce() && static_cast<bool>((INLINE_REDUCE_BITMASK & reduceAttr_));
    void* remoteMem = nullptr;
    if (isSpInlineReduce) {
        CHK_RET(leftLink_->GetRemoteMem(UserMemType::INPUT_MEM, &remoteMem));
    }
    // 每个slice按 从流Post(mainSignals) -> 主流Wait/Empty/Post -> 从流Wait(subSignals) -> 从流拷贝 ->
    // 主流reduce/远端读 的顺序交替下发
    for (u32 sliceIdx = 0; sliceIdx < subSliceVector.size(); sliceIdx++) {
        CHK_RET(LocalNotify::Post(subStreams_[0], dispatcher_, mainSignals_[0], profilerInput_.stage));
        CHK_RET(PreSync());
        CHK_RET(LocalNotify::Wait(subStreams_[0], dispatcher_, subSignals_[0], profilerInput_.stage));
        CHK_RET(RunSubStreamSlice(step, sliceIdx, subSliceVector, cclSliceVector, rankSize));
        if (isSpInlineReduce) {
            CHK_RET(ReducerSpInlineSlice(dispatcher_, leftLink_, remoteMem, rxReduceMems[sliceIdx], stream_));
        } else {
            CHK_RET(ReducerSdmaRemoteReadSlice(leftLink_, rxReduceMems[sliceIdx], stream_));
        }
    }
    if (!isSpInlineReduce) {
        CHK_RET(ReducerLocalReduceSuffix(dispatcher_, leftLink_, rxReduceMems, stream_));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::RunReduceScatter(const u32 rank, const u32 rankSize)
{
    HCCL_INFO("ReduceScatterRingConcurrentDirect starts, the input param rank[%u]", rank);
    // 空拷贝用于后续操作附着
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));

    CHK_RET(RunInitStep(rank, rankSize));
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    CHK_RET(MainRecordSub()); // 主流通知从流开始通信
    CHK_RET(SubWaitMain());   // 从流等待主流通知
    // 空拷贝用于主从流任务并发
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, subStreams_[0], dispatcher_));
    u32 sliceSize = slices_.size() / rankSize;

    // 例如rank[0,1,2,3]中，rank0的rxSliceIdx = 2，txSliceIdx = 3, subSliceIdx = 1
    u32 txSliceIdx = (rank + rankSize - 1) % rankSize;
    u32 rxSliceIdx = (rank + rankSize - DMA_REDUCE_TWO_OFFSET) % rankSize;
    u32 subSliceIdx = (rank + rankSize - DMA_REDUCE_THREE_OFFSET) % rankSize;
    for (u32 step = 0; step < rankSize - 1; step++) {
        std::vector<Slice> rxSliceVector;
        std::vector<Slice> cclSliceVector;
        std::vector<Slice> txSliceVector;
        std::vector<Slice> subSliceVector;
        for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
            rxSliceVector.push_back(slices_[rxSliceIdx * sliceSize + sliceIdx]);
            cclSliceVector.push_back(slices_[subSliceIdx * sliceSize + sliceIdx]);
            txSliceVector.push_back(slices_[txSliceIdx * sliceSize + sliceIdx]);
            subSliceVector.push_back(userSlices_[subSliceIdx * sliceSize + sliceIdx]);
        }

        // dispatcher_aicpu 单条流的任务队列存在上限，队列满后host会阻塞下发，因此主流与从流的任务必须
        // 交替下发：从流Wait(subSignals)依赖主流Post(subSignals)，主流Wait(mainSignals)依赖从流
        // Post(mainSignals)。若先集中下发某一条流的全部任务，队列被占满后host阻塞，而队列中等待的信号
        // 又需要另一条流尚未下发的任务来产生，两条流互相死等。以下保证每个Wait与其配对的Post在小窗口
        // 内先后完成下发，且每条流上的任务序列保持不变。
        if (!isSdma_) {
            // 从流先Post(mainSignals)，主流整段下发完成后，从流再Wait并下发本步拷贝任务
            CHK_RET(RunSubStreamPrePost());
            // 主流
            CHK_RET(RunMainStream(step, txSliceVector, rxSliceVector, rank, rankSize));
            // 从流
            CHK_RET(RunSubStream(step, subSliceVector, cclSliceVector, rank, rankSize));
        } else {
            // sdma场景：主流Tx前缀下发后，主从流逐slice交替下发
            std::vector<ReducerMemoryInfo> rxReduceMems;
            CHK_RET(RunMainStreamTx(step, txSliceVector, rxSliceVector, rank, rankSize, rxReduceMems));
            CHK_RET(RunSdmaStepConcurrent(step, rxReduceMems, subSliceVector, cclSliceVector, rank, rankSize));
        }

        // 更新索引
        subSliceIdx = (subSliceIdx + rankSize - 1) % rankSize;
        txSliceIdx = (txSliceIdx + rankSize - 1) % rankSize;
        rxSliceIdx = (rxSliceIdx + rankSize - 1) % rankSize;
    }
    CHK_RET(SubRecordMain()); // 从流通知主流通信完成
    CHK_RET(MainWaitSub());   // 主流等待从流通知
    if (!isSdma_ && opInfo_->outputAddr != nullptr) {
        HCCL_DEBUG("[RunReduceScatter] rdma DMAReduce last step");
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, finalDst_, finalSrc_, stream_));
    }
    HCCL_INFO("ReduceScatterRingConcurrentDirect finished to RunReduceScatter");
    return HCCL_SUCCESS;
}

// 主流通知从流干活
HcclResult ReduceScatterRingConcurrentDirect::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < subSignals_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 从流等待主流
HcclResult ReduceScatterRingConcurrentDirect::SubWaitMain()
{
    for (u32 streamIndex = 0; streamIndex < subSignals_.size(); streamIndex++) {
        CHK_RET(
            LocalNotify::Wait(subStreams_[streamIndex], dispatcher_, subSignals_[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 主流等待从流
HcclResult ReduceScatterRingConcurrentDirect::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < mainSignals_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 从流告诉主流活干完了
HcclResult ReduceScatterRingConcurrentDirect::SubRecordMain()
{
    for (u32 streamIndex = 0; streamIndex < mainSignals_.size(); streamIndex++) {
        CHK_RET(
            LocalNotify::Post(subStreams_[streamIndex], dispatcher_, mainSignals_[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_RING_DIRECT, ReduceScatterRingConcurrentDirect);
} // namespace hccl
