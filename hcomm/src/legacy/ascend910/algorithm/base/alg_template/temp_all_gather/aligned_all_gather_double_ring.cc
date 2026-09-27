/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aligned_all_gather_double_ring.h"
#include "alg_template_register.h"

namespace hccl {
AlignedAllGatherDoubleRing::AlignedAllGatherDoubleRing(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AlignedAllGatherDoubleRing::~AlignedAllGatherDoubleRing() {}

HcclResult AlignedAllGatherDoubleRing::Prepare(
    HcomCollOpInfo* opInfo, const u32 userRank, std::vector<Stream>& subStreams,
    std::vector<std::shared_ptr<LocalNotify>>& mainSignals, std::vector<std::shared_ptr<LocalNotify>>& subSignals,
    const std::vector<std::vector<u32>>& ringsOrders,
    const std::vector<std::vector<Slice>>& userMemOutputSlicesOfDoubleRing)
{
    opInfo_ = opInfo;
    userRank_ = userRank;
    subStreams_ = subStreams;
    mainSignals_ = mainSignals;
    subSignals_ = subSignals;
    ringsOrders_ = ringsOrders;
    userMemOutputSlicesOfDoubleRing_ = userMemOutputSlicesOfDoubleRing;
    return HCCL_SUCCESS;
}

// 服务器间allgather的入口函数
HcclResult AlignedAllGatherDoubleRing::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
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

    HCCL_DEBUG("[AlignedAllGatherDoubleRing]RunAsync begins");

    // 运行all-gather, ring算法
    CHK_RET(RunAllGather(rank, rankSize));

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(leftLink_, rightLink_));
    }

    CHK_RET(LaunchTaskExtend(dispatcher_, stream_, subStreams_));

    HCCL_INFO("AlignedAllGatherDoubleRing finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult
AlignedAllGatherDoubleRing::CheckParameters(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    CHK_PTR_NULL(opInfo_);
    CHK_RET(CheckConcurrentDirectParameters(rank, rankSize, links));
    // 判断subStreams数量是否正确
    CHK_PRT_RET(
        subStreams_.size() < 1,
        HCCL_ERROR("[AlignedAllGatherDoubleRing] subStreams size[%u] is less than 1", subStreams_.size()), HCCL_E_PARA);
    for (auto& s : subStreams_) {
        CHK_PTR_NULL(s.ptr());
    }
    // 判断mainSignals数量是否正确
    CHK_PRT_RET(
        mainSignals_.size() < 1,
        HCCL_ERROR("[AlignedAllGatherDoubleRing] mainSignals size[%u] is less than 1", mainSignals_.size()),
        HCCL_E_PARA);
    // 判断subSignals数量是否正确
    CHK_PRT_RET(
        subSignals_.size() < 1,
        HCCL_ERROR("[AlignedAllGatherDoubleRing] subSignals size[%u] is less than 1", subSignals_.size()), HCCL_E_PARA);
    // 判断ringsOrder数量是否正确
    for (u32 ringIndex = 0; ringIndex < ringsOrders_.size(); ringIndex++) {
        CHK_PRT_RET(
            ringsOrders_[ringIndex].size() % rankSize != 0,
            HCCL_ERROR(
                "[AlignedAllGatherDoubleRing] ringsOrders[%u] size[%u] can not be divided by rank size[%u]", ringIndex,
                ringsOrders_[ringIndex].size(), rankSize),
            HCCL_E_PARA);
    }
    // 判断userMemOutputSlices数量是否正确
    for (u32 ringIndex = 0; ringIndex < userMemOutputSlicesOfDoubleRing_.size(); ringIndex++) {
        CHK_PRT_RET(
            userMemOutputSlicesOfDoubleRing_[ringIndex].size() % rankSize != 0,
            HCCL_ERROR(
                "[AlignedAllGatherDoubleRing] userMemOutputSlicesOfDoubleRing[%u] size[%u] can not be divided by rank "
                "size[%u]",
                ringIndex, userMemOutputSlicesOfDoubleRing_[ringIndex].size(), rankSize),
            HCCL_E_PARA);
    }
    u32 mainSliceSize = multRingsSlices_[ALIGNED_MAIN_RING_INDEX].size() / rankSize;
    u32 subSliceSize = multRingsSlices_[ALIGNED_SUB_RING_INDEX].size() / rankSize;
    CHK_PRT_RET(
        mainSliceSize != subSliceSize,
        HCCL_ERROR(
            "[AlignedAllGatherDoubleRing] mainSliceSize[%u] is not equal to subSliceSize[%u].", mainSliceSize,
            subSliceSize),
        HCCL_E_PARA);
    HCCL_INFO("AlignedAllGatherDoubleRing finished to CheckParameters");
    return HCCL_SUCCESS;
}

HcclResult AlignedAllGatherDoubleRing::OneRankMemcpy()
{
    CHK_RET(MainRecordSub()); // 主流通知从流开始通信
    CHK_RET(SubWaitMain());   // 从流等待主流通知
    for (u32 ringIndex = 0; ringIndex < multRingsSlices_.size(); ringIndex++) {
        for (u32 sliceIdx = 0; sliceIdx < multRingsSlices_[ringIndex].size(); sliceIdx++) {
            const Slice& srcSlice = multRingsSlices_[ringIndex][sliceIdx];
            const Slice& dstSlice = userMemOutputSlicesOfDoubleRing_[ringIndex][sliceIdx];
            DeviceMem src;
            DeviceMem dst = DeviceMem::create(static_cast<u8*>(opInfo_->outputAddr) + dstSlice.offset, dstSlice.size);
            if (opInfo_->inputAddr != nullptr) {
                // opInfo_->inputAddr != nullptr指示要从user input获取输入
                u64 stepOffset = multRingsSlices_[ringIndex][ringsOrders_[ringIndex][0]].offset;
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
            if (ringIndex == 1) {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
            } else {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStreams_[0]));
            }
        }
    }
    CHK_RET(SubRecordMain()); // 从流通知主流通信完成
    CHK_RET(MainWaitSub());   // 主流等待从流通知
    return HCCL_SUCCESS;
}

HcclResult AlignedAllGatherDoubleRing::GetInitializedNeighborLinks(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 收集左邻居信息
    leftLink_ = links[(rank + rankSize - 1) % rankSize];
    CHK_SMART_PTR_NULL(leftLink_);

    // 收集右邻居信息
    rightLink_ = links[(rank + 1) % rankSize];
    CHK_SMART_PTR_NULL(rightLink_);
    HCCL_INFO("AlignedAllGatherDoubleRing finished to GetInitializedNeighborLinks");
    return HCCL_SUCCESS;
}

HcclResult AlignedAllGatherDoubleRing::SetSlices(const u32 rank, const u32 rankSize)
{
    for (u32 ringIndex = 0; ringIndex < multRingsSlices_.size(); ringIndex++) {
        if (multRingsSlices_[ringIndex].size() == 0) {
            multRingsSlices_[ringIndex].resize(rankSize);

            u64 sliceSize = count_ * DataUnitSize(dataType_);
            for (u32 i = 0; i < rankSize; i++) {
                multRingsSlices_[ringIndex][i].size = sliceSize;
                multRingsSlices_[ringIndex][i].offset = sliceSize * i;
                HCCL_DEBUG(
                    "multRingsSlices_[%u], rank[%u], slices[%u].offset=%llu, slices[%u].size=[%llu]", ringIndex, rank,
                    i, multRingsSlices_[ringIndex][i].offset, i, multRingsSlices_[ringIndex][i].size);
            }
        }
        for (u32 i = 0; i < multRingsSlices_[ringIndex].size(); i++) {
            HCCL_DEBUG(
                "[AlignedAllGatherDoubleRing][SetSlices] multRingsSlices_[%u], rank[%u], slices[%u].offset=[%llu], "
                "slices[%u].size=[%llu]",
                ringIndex, rank, i, multRingsSlices_[ringIndex][i].offset, i, multRingsSlices_[ringIndex][i].size);
        }
    }
    HCCL_INFO("AlignedAllGatherDoubleRing finished to SetSlices");
    return HCCL_SUCCESS;
}

HcclResult AlignedAllGatherDoubleRing::RunInitStep(const u32 rank, const u32 rankSize)
{
    for (u32 ringIndex = 0; ringIndex < multRingsSlices_.size(); ringIndex++) {
        // 第一步搬到userMemIn_的offset, 不同的ring环offset不一样
        u64 firstStepOffset;
        if (ringIndex == 0) {
            firstStepOffset = multRingsSlices_[ringIndex][ringsOrders_[ringIndex][0]].offset;
        } else {
            const auto& prevRingSlice = multRingsSlices_[ringIndex - 1][ringsOrders_[ringIndex - 1][rank]];
            const auto& slice = multRingsSlices_[ringIndex][ringsOrders_[ringIndex][rank]];
            firstStepOffset = slice.offset - prevRingSlice.offset;
        }
        // 第-1步，片内将部分数据从userIn搬到cclIn
        DeviceMem srcInit;
        DeviceMem dstInit;
        u32 initSliceIdx;
        if (ringIndex == 0) {
            initSliceIdx = rank;
        } else {
            initSliceIdx = (rankSize - rank) % rankSize;
        }
        u32 sliceSize = multRingsSlices_[ringIndex].size() / rankSize;
        for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
            Slice initSlice = multRingsSlices_[ringIndex][initSliceIdx * sliceSize + sliceIdx];
            // 需要+userMemIn_的offset
            if (opInfo_->inputAddr != nullptr) {
                // AllGather算子调用AlignedAllGatherDoubleRing场景
                HCCL_DEBUG(
                    "Memcpy operation: step[-1] stream[main] src rank[%u] starts to copy(rcv) offset[%llu], "
                    "size[%llu] on userMemOutput to offset[%llu], size[%llu] on CCL",
                    userRank_, firstStepOffset, initSlice.size, initSlice.offset, initSlice.size);
                srcInit = DeviceMem::create(static_cast<u8*>(opInfo_->inputAddr) + firstStepOffset, initSlice.size);
            } else {
                // AllReduce算子调用AlignedAllGatherDoubleRing场景
                HCCL_DEBUG(
                    "Memcpy operation: step[-1] stream[main] src rank[%u] starts to copy(rcv) offset[%llu], "
                    "size[%llu] on CCL to offset[%llu], size[%llu] on CCL",
                    userRank_, initSlice.offset, initSlice.size, initSlice.offset, initSlice.size);
                srcInit = inputMem_.range(initSlice.offset, initSlice.size);
            }
            dstInit = outputMem_.range(initSlice.offset, initSlice.size);
            // 若src与dst一样，则不需要搬运
            if (srcInit == dstInit) {
                continue;
            }
            if (ringIndex == 1) {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstInit, srcInit, stream_));
            } else {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstInit, srcInit, subStreams_[0]));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
AlignedAllGatherDoubleRing::PrepareRunMainStream(u32 ringIndex, Stream& stream, LINK& preLink, LINK& nextLink)
{
    HCCL_DEBUG("AlignedAllGatherDoubleRing PrepareRunMainStream start");
    if (ringIndex == 1) {
        stream = stream_;
        preLink = rightLink_;
        nextLink = leftLink_;
    } else {
        stream = subStreams_[0];
        preLink = leftLink_;
        nextLink = rightLink_;
    }
    HCCL_DEBUG("AlignedAllGatherDoubleRing PrepareRunMainStream end");
    return HCCL_SUCCESS;
}

HcclResult AlignedAllGatherDoubleRing::RunAllStreams(
    const u32 step, const u32 rankSize, const std::vector<TxMemoryInfo>& mainTxMems,
    std::vector<RxMemoryInfo>& mainRxMems, const std::vector<TxMemoryInfo>& subTxMems,
    std::vector<RxMemoryInfo>& subRxMems, std::vector<DeviceMem>& mainLocalSrcMems,
    std::vector<DeviceMem>& mainLocalDstMems, std::vector<DeviceMem>& subLocalSrcMems,
    std::vector<DeviceMem>& subLocalDstMems)
{
    (void)mainTxMems;
    (void)subTxMems;
    Stream mainStream;
    LINK mainPreLink;
    LINK mainNextLink;
    Stream subStream;
    LINK subPreLink;
    LINK subNextLink;
    CHK_RET(PrepareRunMainStream(ALIGNED_MAIN_RING_INDEX, mainStream, mainPreLink, mainNextLink));
    CHK_RET(PrepareRunMainStream(ALIGNED_SUB_RING_INDEX, subStream, subPreLink, subNextLink));

    CHK_RET(mainNextLink->TxAck(mainStream));
    CHK_RET(subNextLink->TxAck(subStream));

    CHK_RET(mainPreLink->RxAck(mainStream));
    CHK_RET(subPreLink->RxAck(subStream));
    u32 sliceSize = multRingsSlices_[ALIGNED_MAIN_RING_INDEX].size() / rankSize;
    for (u32 memIdx = 0; memIdx < sliceSize; memIdx++) {
        CHK_RET(RxAsyncMemcpy(step, ALIGNED_SUB_RING_INDEX, subRxMems[memIdx], subStream, subPreLink));
        CHK_RET(LocalMemcpy(ALIGNED_MAIN_RING_INDEX, mainLocalSrcMems[memIdx], mainLocalDstMems[memIdx]));
        CHK_RET(LocalMemcpy(ALIGNED_SUB_RING_INDEX, subLocalSrcMems[memIdx], subLocalDstMems[memIdx]));
        CHK_RET(RxAsyncMemcpy(step, ALIGNED_MAIN_RING_INDEX, mainRxMems[memIdx], mainStream, mainPreLink));
    }
    CHK_RET(mainPreLink->TxDataSignal(mainStream));
    CHK_RET(subPreLink->TxDataSignal(subStream));

    CHK_RET(mainNextLink->RxDataSignal(mainStream));
    CHK_RET(subNextLink->RxDataSignal(subStream));
    return HCCL_SUCCESS;
}

HcclResult AlignedAllGatherDoubleRing::RxAsyncMemcpy(
    const u32 step, const u32 ringIndex, RxMemoryInfo& mem, Stream& stream, LINK& link)
{
    (void)step;
    // PreSync
    if (ringIndex == 1) {
        CHK_RET(MainWaitSub());
        CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
        CHK_RET(MainRecordSub());
    } else {
        CHK_RET(LocalNotify::Post(subStreams_[0], dispatcher_, mainSignals_[0], profilerInput_.stage));
        CHK_RET(LocalNotify::Wait(subStreams_[0], dispatcher_, subSignals_[0], profilerInput_.stage));
    }
    CHK_PTR_NULL(mem.dst);
    void* srcMemPtr = nullptr;
    CHK_RET(link->GetRemoteMem(mem.srcMemType, &srcMemPtr));

    DeviceMem srcDevMem(static_cast<s8*>(srcMemPtr) + mem.srcOffset, mem.len);
    DeviceMem dstDevMem(static_cast<s8*>(mem.dst), mem.len);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstDevMem, srcDevMem, stream, link->GetRemoteRank(), link->GetLinkType()));
    return HCCL_SUCCESS;
}

HcclResult AlignedAllGatherDoubleRing::LocalMemcpy(const u32 ringIndex, DeviceMem& localSrcMem, DeviceMem& localDstMem)
{
    // 校验流数
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        CHK_RET(LocalNotify::Post(
            subStreams_[ringIndex + 1], dispatcher_, mainSignals_[ringIndex + 1], profilerInput_.stage));
        CHK_RET(LocalNotify::Wait(
            subStreams_[ringIndex + 1], dispatcher_, subSignals_[ringIndex + 1], profilerInput_.stage));
        if (localSrcMem != localDstMem) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, localDstMem, localSrcMem, subStreams_[ringIndex + 1]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedAllGatherDoubleRing::PrepareDeviceMems(
    const u32 step, const u32 ringIndex, const u32 rankSize, const u32 txSliceIdx, const u32 rxSliceIdx,
    std::vector<TxMemoryInfo>& txMems, std::vector<RxMemoryInfo>& rxMems, std::vector<DeviceMem>& localSrcMems,
    std::vector<DeviceMem>& localDstMems)
{
    u32 sliceSize = multRingsSlices_[ringIndex].size() / rankSize;
    for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
        const Slice& rxSlice = multRingsSlices_[ringIndex][rxSliceIdx * sliceSize + sliceIdx];
        const Slice& mainSlice = userMemOutputSlicesOfDoubleRing_[ringIndex][rxSliceIdx * sliceSize + sliceIdx];
        const Slice& txSlice = multRingsSlices_[ringIndex][txSliceIdx * sliceSize + sliceIdx];
        const Slice& subSlice = userMemOutputSlicesOfDoubleRing_[ringIndex][txSliceIdx * sliceSize + sliceIdx];
        // PrepareTxRxMems
        DeviceMem src = outputMem_.range(txSlice.offset, txSlice.size);
        HCCL_DEBUG("tx srcMem[%p] range[%llu] size[%llu] ", src.ptr(), txSlice.offset, txSlice.size);
        txMems.emplace_back(
            TxMemoryInfo{UserMemType::OUTPUT_MEM, txSlice.offset + baseOffset_, src.ptr(), txSlice.size});
        DeviceMem dst;
        if (step == rankSize - DMA_REDUCE_TWO_OFFSET) {
            HCCL_DEBUG(
                "DMAReduce(sdma) MemcpyAsync operation: step[%u] stream[main], dst rank[%u] starts to rcv "
                "offset[%llu] size[%llu] at userMemOutput_",
                step, userRank_, mainSlice.offset, mainSlice.size);
            dst = DeviceMem::create(static_cast<u8*>(opInfo_->outputAddr) + mainSlice.offset, mainSlice.size);
        } else {
            HCCL_DEBUG(
                "MemcpyAsync operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu] size[%llu] "
                "at outputMem_",
                step, userRank_, rxSlice.offset, rxSlice.size);
            dst = outputMem_.range(rxSlice.offset, rxSlice.size);
        }
        rxMems.emplace_back(
            RxMemoryInfo{UserMemType::OUTPUT_MEM, rxSlice.offset + baseOffset_, dst.ptr(), rxSlice.size});
        // PrepareLocalCopyDeviceMems
        // 从流
        src = outputMem_.range(txSlice.offset, txSlice.size);
        dst = DeviceMem::create(static_cast<u8*>(opInfo_->outputAddr) + subSlice.offset, subSlice.size);
        HCCL_DEBUG(
            "Memcpy operation: step[%u] stream[sub], src rank[%u] starts to send offset[%llu] size[%llu], "
            "dst rank starts to rcv offset[%llu] size[%llu] at userMemOutput_",
            step, userRank_, subSlice.offset, subSlice.size, txSlice.offset, txSlice.size);
        localSrcMems.emplace_back(src);
        localDstMems.emplace_back(dst);
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedAllGatherDoubleRing::RunAllGather(const u32 rank, const u32 rankSize)
{
    HCCL_INFO("AlignedAllGatherDoubleRing starts, the input param rank[%u]", rank);
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        // 主环主流通知从环主流开始通信
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[0], profilerInput_.stage));
        // 从环主流等待主环主流通知
        CHK_RET(LocalNotify::Wait(subStreams_[0], dispatcher_, subSignals_[0], profilerInput_.stage));
        CHK_RET(RunInitStep(rank, rankSize));
        CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
        CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, subStreams_[0], dispatcher_));
        // 从流通知主流通信完成
        CHK_RET(LocalNotify::Post(subStreams_[0], dispatcher_, mainSignals_[0], profilerInput_.stage));
        // 主流等待从流通知
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[0], profilerInput_.stage));
    }
    // 主环主流通知从环主流开始通信
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    CHK_RET(MainRecordSub());
    // 从环主流等待主环主流通知
    CHK_RET(SubWaitMain());
    u32 txSliceIdxSub = rank;
    u32 rxSliceIdxSub = (rank + rankSize - 1) % rankSize;
    u32 txSliceIdxMain = (rankSize - rank) % rankSize;
    u32 rxSliceIdxMain = (rankSize - rank - 1 + rankSize) % rankSize;
    // 空拷贝用于主从流任务并发
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    CHK_RET(ExecEmptyTasks());
    for (u32 step = 0; step < rankSize - 1; step++) {
        std::vector<TxMemoryInfo> txMemsSub;
        std::vector<RxMemoryInfo> rxMemsSub;
        std::vector<DeviceMem> localSrcMemsSub;
        std::vector<DeviceMem> localDstMemsSub;
        CHK_RET(PrepareDeviceMems(
            step, ALIGNED_SUB_RING_INDEX, rankSize, txSliceIdxSub, rxSliceIdxSub, txMemsSub, rxMemsSub, localSrcMemsSub,
            localDstMemsSub));
        std::vector<TxMemoryInfo> txMemsMain;
        std::vector<RxMemoryInfo> rxMemsMain;
        std::vector<DeviceMem> localSrcMemsMain;
        std::vector<DeviceMem> localDstMemsMain;
        CHK_RET(PrepareDeviceMems(
            step, ALIGNED_MAIN_RING_INDEX, rankSize, txSliceIdxMain, rxSliceIdxMain, txMemsMain, rxMemsMain,
            localSrcMemsMain, localDstMemsMain));
        CHK_RET(RunAllStreams(
            step, rankSize, txMemsMain, rxMemsMain, txMemsSub, rxMemsSub, localSrcMemsMain, localDstMemsMain,
            localSrcMemsSub, localDstMemsSub));

        // 更新索引
        txSliceIdxSub = (txSliceIdxSub + rankSize - 1) % rankSize;
        rxSliceIdxSub = (rxSliceIdxSub + rankSize - 1) % rankSize;
        txSliceIdxMain = (txSliceIdxMain + rankSize - 1) % rankSize;
        rxSliceIdxMain = (rxSliceIdxMain + rankSize - 1) % rankSize;
    }
    // 从环主流通知主环主流通信完成
    CHK_RET(SubRecordMain());
    // 主环主流等待从环主流通知
    CHK_RET(MainWaitSub());
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    HCCL_INFO("AlignedAllGatherDoubleRing finished to RunAllGather");
    return HCCL_SUCCESS;
}

HcclResult AlignedAllGatherDoubleRing::ExecEmptyTasks()
{
    for (u32 signalIndex = 0; signalIndex < subStreams_.size(); signalIndex++) {
        CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, subStreams_[signalIndex], dispatcher_));
    }
    return HCCL_SUCCESS;
}

// 主流通知从流干活
HcclResult AlignedAllGatherDoubleRing::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < subSignals_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 从流等待主流
HcclResult AlignedAllGatherDoubleRing::SubWaitMain()
{
    for (u32 streamIndex = 0; streamIndex < subSignals_.size(); streamIndex++) {
        CHK_RET(
            LocalNotify::Wait(subStreams_[streamIndex], dispatcher_, subSignals_[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 主流等待从流
HcclResult AlignedAllGatherDoubleRing::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < mainSignals_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 从流告诉主流活干完了
HcclResult AlignedAllGatherDoubleRing::SubRecordMain()
{
    for (u32 streamIndex = 0; streamIndex < mainSignals_.size(); streamIndex++) {
        CHK_RET(
            LocalNotify::Post(subStreams_[streamIndex], dispatcher_, mainSignals_[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALIGNED_ALL_GATHER_DOUBLE_RING, AlignedAllGatherDoubleRing);
} // namespace hccl
