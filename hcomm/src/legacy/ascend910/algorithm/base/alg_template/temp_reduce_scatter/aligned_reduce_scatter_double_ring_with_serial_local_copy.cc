/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aligned_reduce_scatter_double_ring_with_serial_local_copy.h"
#include "alg_template_register.h"

namespace hccl {
AlignedReduceScatterDoubleRingWithSerialLocalCopy::AlignedReduceScatterDoubleRingWithSerialLocalCopy(
    const HcclDispatcher dispatcher)
    : AlignedReduceScatterDoubleRing(dispatcher)
{}

AlignedReduceScatterDoubleRingWithSerialLocalCopy::~AlignedReduceScatterDoubleRingWithSerialLocalCopy() {}

// reduce scatter ring direct算法的函数入口
HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::RunAsync(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 基本的检查
    CHK_RET(CheckParameters(rank, rankSize, links));
    HCCL_DEBUG("[AlignedReduceScatterDoubleRingWithSerialLocalCopy]RunAsync start");

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

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(leftLink_, rightLink_));
    }

    CHK_RET(LaunchTaskExtend(dispatcher_, stream_, subStreams_));

    HCCL_INFO("AlignedReduceScatterDoubleRingWithSerialLocalCopy finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::MemcpyInitSlices(
    u64 ringIndex, DeviceMem& dstInit, DeviceMem& srcInit, DeviceMem& dstSubInit, DeviceMem& srcSubInit)
{
    if (ringIndex == 1) {
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstInit, srcInit, stream_));
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstSubInit, srcSubInit, stream_));
    } else {
        if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB && (!disableDMAReduce_)) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstInit, srcInit, subStreams_[0]));
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstSubInit, srcSubInit, subStreams_[1]));
        } else {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstInit, srcInit, subStreams_[0]));
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstSubInit, srcSubInit, subStreams_[0]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::RunMainInitStep(const u32 rank, const u32 rankSize)
{
    // 主环初始indexes
    u32 initSlice0Idx = (rankSize - rank - 1 + rankSize) % rankSize;
    u32 initSlice1Idx = (rankSize - rank - DMA_REDUCE_TWO_OFFSET + rankSize) % rankSize;
    u32 discontinuousSliceSize = multRingsSlices_[ALIGNED_MAIN_RING_INDEX].size() / rankSize;
    HCCL_DEBUG("Memcpy operation: step[-1] starts on ring[%u]", ALIGNED_MAIN_RING_INDEX);
    DeviceMem dstInit;
    DeviceMem srcInit;
    DeviceMem dstSubInit;
    DeviceMem srcSubInit;
    for (u32 discontinuousSliceIdx = 0; discontinuousSliceIdx < discontinuousSliceSize; discontinuousSliceIdx++) {
        CHK_RET(PrepareInitSlices(
            rankSize, ALIGNED_MAIN_RING_INDEX, discontinuousSliceSize, discontinuousSliceIdx, initSlice0Idx,
            initSlice1Idx, dstInit, srcInit, dstSubInit, srcSubInit));
        CHK_RET(MemcpyInitSlices(ALIGNED_MAIN_RING_INDEX, dstInit, srcInit, dstSubInit, srcSubInit));
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::RunSubInitStep(const u32 rank, const u32 rankSize)
{
    // 从环初始indexes
    u32 initSlice0Idx = (rank + rankSize - 1) % rankSize;
    u32 initSlice1Idx = (rank + rankSize - DMA_REDUCE_TWO_OFFSET) % rankSize;
    u32 discontinuousSliceSize = multRingsSlices_[ALIGNED_SUB_RING_INDEX].size() / rankSize;
    HCCL_DEBUG("Memcpy operation: step[-1] starts on ring[%u]", ALIGNED_SUB_RING_INDEX);
    DeviceMem dstInit;
    DeviceMem srcInit;
    DeviceMem dstSubInit;
    DeviceMem srcSubInit;
    for (u32 discontinuousSliceIdx = 0; discontinuousSliceIdx < discontinuousSliceSize; discontinuousSliceIdx++) {
        CHK_RET(PrepareInitSlices(
            rankSize, ALIGNED_SUB_RING_INDEX, discontinuousSliceSize, discontinuousSliceIdx, initSlice0Idx,
            initSlice1Idx, dstInit, srcInit, dstSubInit, srcSubInit));
        CHK_RET(MemcpyInitSlices(ALIGNED_SUB_RING_INDEX, dstInit, srcInit, dstSubInit, srcSubInit));
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::LocalMemcpy(
    const u32 step, const u32 rankSize, const u32 ringIndex, DeviceMem& localSrcMem, DeviceMem& localDstMem)
{
    // 先调通单算子模式
    // 通过校验流数判断是单算子模式还是图模式
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB && (!disableDMAReduce_)) {
        if (ringIndex == 0) {
            CHK_RET(LocalNotify::Post(
                subStreams_[ringIndex + 1], dispatcher_, mainSignals_[ringIndex + 1], profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(
                subStreams_[ringIndex + 1], dispatcher_, subSignals_[ringIndex + 1], profilerInput_.stage));
            if (localSrcMem != localDstMem && step != rankSize - DMA_REDUCE_TWO_OFFSET) {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, localDstMem, localSrcMem, subStreams_[ringIndex + 1]));
            }
        } else {
            if (localSrcMem != localDstMem && step != rankSize - DMA_REDUCE_TWO_OFFSET) {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, localDstMem, localSrcMem, stream_));
            }
        }
    } else {
        if (ringIndex == 0) {
            if (localSrcMem != localDstMem && step != rankSize - DMA_REDUCE_TWO_OFFSET) {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, localDstMem, localSrcMem, subStreams_[0]));
            }
        } else {
            if (localSrcMem != localDstMem && step != rankSize - DMA_REDUCE_TWO_OFFSET) {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, localDstMem, localSrcMem, stream_));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::RunMainRingSubStream(const u32 rank, const u32 rankSize)
{
    // 主环初始indexes
    u32 txSliceIdxMain = (rankSize - rank - 1 + rankSize) % rankSize;
    u32 rxSliceIdxMain = (rankSize - rank - DMA_REDUCE_TWO_OFFSET + rankSize) % rankSize;
    u32 subSliceIdxMain = (rankSize - rank - DMA_REDUCE_THREE_OFFSET + rankSize) % rankSize;
    for (u32 step = 0; step < rankSize - 1; step++) {
        // 并发
        std::vector<SenderMemoryInfo> txReduceMemsMain;
        std::vector<ReducerMemoryInfo> rxReduceMemsMain;
        std::vector<DeviceMem> localSrcMemsMain;
        std::vector<DeviceMem> localDstMemsMain;
        CHK_RET(PrepareDeviceMems(
            step, ALIGNED_MAIN_RING_INDEX, rankSize, txSliceIdxMain, rxSliceIdxMain, subSliceIdxMain, txReduceMemsMain,
            rxReduceMemsMain, localSrcMemsMain, localDstMemsMain));
        // 主环从流
        u32 sliceSize = multRingsSlices_[ALIGNED_MAIN_RING_INDEX].size() / rankSize;
        for (u32 memIdx = 0; memIdx < sliceSize; memIdx++) {
            CHK_RET(LocalMemcpy(
                step, rankSize, ALIGNED_MAIN_RING_INDEX, localSrcMemsMain[memIdx], localDstMemsMain[memIdx]));
        }
        // 更新索引
        subSliceIdxMain = (subSliceIdxMain + rankSize - 1) % rankSize;
        txSliceIdxMain = (txSliceIdxMain + rankSize - 1) % rankSize;
        rxSliceIdxMain = (rxSliceIdxMain + rankSize - 1) % rankSize;
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::PreSync(const u32 ringIndex)
{
    HCCL_DEBUG("[AlignedReduceScatterDoubleRingWithSerialLocalCopy] PreSync starts");
    if (ringIndex == 1) {
        if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB && (!disableDMAReduce_)) {
            CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[0], profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[1], profilerInput_.stage));
            CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
            CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[0], profilerInput_.stage));
            CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[1], profilerInput_.stage));
        } else {
            CHK_RET(MainWaitSub());
            CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
            CHK_RET(MainRecordSub());
        }
    } else {
        CHK_RET(LocalNotify::Post(subStreams_[0], dispatcher_, mainSignals_[0], profilerInput_.stage));
        CHK_RET(LocalNotify::Wait(subStreams_[0], dispatcher_, subSignals_[0], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::RunAllStreams(
    const u32 step, const u32 rankSize, std::vector<SenderMemoryInfo>& mainTxReduceMems,
    std::vector<ReducerMemoryInfo>& mainRxReduceMems, std::vector<SenderMemoryInfo>& subTxReduceMems,
    std::vector<ReducerMemoryInfo>& subRxReduceMems, std::vector<DeviceMem>& mainLocalSrcMems,
    std::vector<DeviceMem>& mainLocalDstMems, std::vector<DeviceMem>& subLocalSrcMems,
    std::vector<DeviceMem>& subLocalDstMems)
{
    (void)mainLocalDstMems;
    (void)mainTxReduceMems;
    (void)subTxReduceMems;
    (void)mainLocalSrcMems;
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
        CHK_RET(LocalMemcpy(step, rankSize, ALIGNED_SUB_RING_INDEX, subLocalSrcMems[memIdx], subLocalDstMems[memIdx]));
    }
    CHK_RET(mainPreLink->TxDataSignal(mainStream));
    CHK_RET(mainNextLink->RxDataSignal(mainStream));
    CHK_RET(subPreLink->TxDataSignal(subStream));
    CHK_RET(subNextLink->RxDataSignal(subStream));
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::RunReduceScatter(const u32 rank, const u32 rankSize)
{
    HCCL_INFO("AlignedReduceScatterDoubleRingWithSerialLocalCopy starts, the input param rank[%u]", rank);
    // 空拷贝用于后续操作附着
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    // 先完成主环主流操作
    CHK_RET(RunMainRingSubStream(rank, rankSize));
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        CHK_RET(RunMainInitStep(rank, rankSize));
        // 主环主流通知从环主流开始通信
        CHK_RET(MainRecordSub());
        // 从环主流等待主环主流通知
        CHK_RET(SubWaitMain());
        CHK_RET(RunSubInitStep(rank, rankSize));
        // 从流通知主流通信完成
        CHK_RET(SubRecordMain());
        // 主流等待从流通知
        CHK_RET(MainWaitSub());
    }
    // 主环主流通知从环主流开始通信
    CHK_RET(MainRecordSub());
    // 从环主流等待主环主流通知
    CHK_RET(SubWaitMain());
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    CHK_RET(ExecEmptyTasks());
    // 例如rank[0,1,2,3]中，rank0的rxSliceIdx = 2，txSliceIdx = 3, subSliceIdx = 1
    // 从环初始indexes
    u32 txSliceIdxSub = (rank + rankSize - 1) % rankSize;
    u32 rxSliceIdxSub = (rank + rankSize - DMA_REDUCE_TWO_OFFSET) % rankSize;
    u32 subSliceIdxSub = (rank + rankSize - DMA_REDUCE_THREE_OFFSET) % rankSize;
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
        std::vector<DeviceMem> localSrcMemsMain;
        std::vector<DeviceMem> localDstMemsMain;
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
        subSliceIdxSub = (subSliceIdxSub + rankSize - 1) % rankSize;
        txSliceIdxSub = (txSliceIdxSub + rankSize - 1) % rankSize;
        rxSliceIdxSub = (rxSliceIdxSub + rankSize - 1) % rankSize;
        subSliceIdxMain = (subSliceIdxMain + rankSize - 1) % rankSize;
        txSliceIdxMain = (txSliceIdxMain + rankSize - 1) % rankSize;
        rxSliceIdxMain = (rxSliceIdxMain + rankSize - 1) % rankSize;
    }
    // 从环主流通知主环主流通信完成
    CHK_RET(SubRecordMain());
    // 主环主流等待从环主流通知
    CHK_RET(MainWaitSub());
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    HCCL_INFO("AlignedReduceScatterDoubleRingWithSerialLocalCopy finished to RunReduceScatter");
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::GetActiveSubstreamNumWithSerial(u32& activeSubstreamNum)
{
    constexpr u32 IDX_2 = 2;
    activeSubstreamNum = subStreams_.size();
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        if (subStreams_.size() <= IDX_2) {
            HCCL_ERROR("[GetActiveSubstreamNumWithSerial]subStreams_.size()[%zu] <= 2", subStreams_.size());
            return HCCL_E_PARA;
        }
        if (disableDMAReduce_) {
            activeSubstreamNum = subStreams_.size() - IDX_2;
        } else {
            activeSubstreamNum = subStreams_.size() - 1;
        }
        HCCL_DEBUG(
            "[AlignedReduceScatterDoubleRingWithSerialLocalCopy] disableDMAReduce_[%d], activeSubstreamNum[%u]",
            disableDMAReduce_, activeSubstreamNum);
    }
    return HCCL_SUCCESS;
}

HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::ExecEmptyTasks()
{
    HCCL_DEBUG("[AlignedReduceScatterDoubleRingWithSerialLocalCopy] ExecEmptyTasks");
    u32 activeSubstreamNum = 0;
    CHK_RET(GetActiveSubstreamNumWithSerial(activeSubstreamNum));
    for (u32 signalIndex = 0; signalIndex < activeSubstreamNum; signalIndex++) {
        CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, subStreams_[signalIndex], dispatcher_));
    }
    return HCCL_SUCCESS;
}

// 主流通知从流干活
HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::MainRecordSub()
{
    HCCL_DEBUG("[AlignedReduceScatterDoubleRingWithSerialLocalCopy] MainRecordSub");
    u32 activeSubstreamNum = 0;
    CHK_RET(GetActiveSubstreamNumWithSerial(activeSubstreamNum));
    for (u32 signalIndex = 0; signalIndex < activeSubstreamNum; signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 从流等待主流
HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::SubWaitMain()
{
    HCCL_DEBUG("[AlignedReduceScatterDoubleRingWithSerialLocalCopy] SubWaitMain");
    u32 activeSubstreamNum = 0;
    CHK_RET(GetActiveSubstreamNumWithSerial(activeSubstreamNum));
    for (u32 streamIndex = 0; streamIndex < activeSubstreamNum; streamIndex++) {
        CHK_RET(
            LocalNotify::Wait(subStreams_[streamIndex], dispatcher_, subSignals_[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 主流等待从流
HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::MainWaitSub()
{
    HCCL_DEBUG("[AlignedReduceScatterDoubleRingWithSerialLocalCopy] MainWaitSub");
    u32 activeSubstreamNum = 0;
    CHK_RET(GetActiveSubstreamNumWithSerial(activeSubstreamNum));
    for (u32 signalIndex = 0; signalIndex < activeSubstreamNum; signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 从流告诉主流活干完了
HcclResult AlignedReduceScatterDoubleRingWithSerialLocalCopy::SubRecordMain()
{
    HCCL_DEBUG("[AlignedReduceScatterDoubleRingWithSerialLocalCopy] SubRecordMain");
    u32 activeSubstreamNum = 0;
    CHK_RET(GetActiveSubstreamNumWithSerial(activeSubstreamNum));
    for (u32 streamIndex = 0; streamIndex < activeSubstreamNum; streamIndex++) {
        CHK_RET(
            LocalNotify::Post(subStreams_[streamIndex], dispatcher_, mainSignals_[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_DB_RING_SLC, AlignedReduceScatterDoubleRingWithSerialLocalCopy);
} // namespace hccl
