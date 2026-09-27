/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_double_ring_direct.h"
#include "alg_template_register.h"

namespace hccl {

constexpr u32 RANK_SIZE_THREE = 3;

ScatterDoubleRingDirect::ScatterDoubleRingDirect(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

ScatterDoubleRingDirect::~ScatterDoubleRingDirect() {}

HcclResult ScatterDoubleRingDirect::Prepare(
    HcomCollOpInfo* opInfo, const u32 userRank, const u32 subRingRank, std::vector<Stream>& subStreams,
    const std::vector<std::shared_ptr<LocalNotify>>& mainSignals,
    const std::vector<std::shared_ptr<LocalNotify>>& subSignals, const std::vector<std::vector<u32>>& ringsOrders,
    const std::vector<std::vector<Slice>>& multiRingSlices, const std::vector<std::vector<Slice>>& userMemInputSlices)
{
    opInfo_ = opInfo;
    userRank_ = userRank;
    subRingRank_ = subRingRank;
    subStreams_ = subStreams;
    mainSignals_ = mainSignals;
    subSignals_ = subSignals;
    ringsOrders_ = ringsOrders;
    multiRingSlices_ = multiRingSlices;
    userMemInputSlices_ = userMemInputSlices;
    return HCCL_SUCCESS;
}

// reduce scatter ring direct算法的函数入口
HcclResult ScatterDoubleRingDirect::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 基本的检查
    CHK_RET(CheckParameters(rank, rankSize, links));

    // 判断rank_size == 1
    if (rankSize == 1) {
        CHK_RET(MemcpyByOneRank());
        return HCCL_SUCCESS;
    }
    // 收集邻居信息
    CHK_RET(GetInitializedNeighborLinks(rank, rankSize, links));

    // 运行scatter, ring算法
    CHK_RET(RunScatter(rank, rankSize));
    CHK_RET(LaunchTaskExtend(dispatcher_, stream_, subStreams_));

    HCCL_INFO("ScatterDoubleRingDirect finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

HcclResult ScatterDoubleRingDirect::CheckParameters(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    CHK_PTR_NULL(opInfo_);
    CHK_RET(CheckConcurrentDirectParameters(rank, rankSize, links));
    // 判断ranksize大小
    CHK_PRT_RET(
        rankSize < 1, HCCL_ERROR("[ScatterDoubleRingDirect] rankSize size[%u] is less than 1", rankSize), HCCL_E_PARA);
    // 判断subStreams数量是否正确
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        CHK_PRT_RET(
            subStreams_.size() != 1,
            HCCL_ERROR("[ScatterDoubleRingDirect] subStreams size[%u] must equal to 1", subStreams_.size()),
            HCCL_E_PARA);
    } else {
        CHK_PRT_RET(
            subStreams_.size() != DOUBLE_RING_STREAM_NUM,
            HCCL_ERROR("[ScatterDoubleRingDirect] subStreams size[%u] must equal to 3", subStreams_.size()),
            HCCL_E_PARA);
    }
    for (auto& s : subStreams_) {
        CHK_PTR_NULL(s.ptr());
    }
    // 判断mainSignals数量是否正确
    CHK_PRT_RET(
        mainSignals_.size() < 1,
        HCCL_ERROR("[ScatterDoubleRingDirect] mainSignals size[%u] is less than 1", mainSignals_.size()), HCCL_E_PARA);
    // 判断subSignals数量是否正确
    CHK_PRT_RET(
        subSignals_.size() < 1,
        HCCL_ERROR("[ScatterDoubleRingDirect] subSignals size[%u] is less than 1", subSignals_.size()), HCCL_E_PARA);
    // 判断ringsOrder size， multiRingSlices size, userMemInputSlices size是否正确
    if (ringsOrders_.size() != DOUBLE_RING_NUM || multiRingSlices_.size() != DOUBLE_RING_NUM
        || userMemInputSlices_.size() != DOUBLE_RING_NUM) {
        HCCL_ERROR(
            "[ScatterDoubleRingDirect] ringsOrder size[%u], multiRingSlices size[%u], userMemInputSlices"
            "size[%u] must equal to 2",
            ringsOrders_.size(), multiRingSlices_.size(), userMemInputSlices_.size());
        return HCCL_E_PARA;
    }
    // 判断ringsOrder数量是否正确
    for (u32 ringIndex = 0; ringIndex < ringsOrders_.size(); ringIndex++) {
        CHK_PRT_RET(
            ringsOrders_[ringIndex].size() != rankSize,
            HCCL_ERROR(
                "[ScatterDoubleRingDirect] ringsOrders[%u] size[%u] must equal to rank size[%u]", ringIndex,
                ringsOrders_[ringIndex].size(), rankSize),
            HCCL_E_PARA);
    }
    // 判断multiRingSlices数量是否正确
    for (u32 ringIndex = 0; ringIndex < multiRingSlices_.size(); ringIndex++) {
        CHK_PRT_RET(
            multiRingSlices_[ringIndex].size() != rankSize,
            HCCL_ERROR(
                "[ScatterDoubleRingDirect] multiRingSlices[%u] size[%u] must equal to rank size[%u]", ringIndex,
                multiRingSlices_[ringIndex].size(), rankSize),
            HCCL_E_PARA);
    }
    // 判断userMemInputSlices数量是否正确
    for (u32 ringIndex = 0; ringIndex < userMemInputSlices_.size(); ringIndex++) {
        CHK_PRT_RET(
            userMemInputSlices_[ringIndex].size() != rankSize,
            HCCL_ERROR(
                "[ScatterDoubleRingDirect] userMemInputSlices_[%u] size[%u] must equal to rank size[%u]", ringIndex,
                userMemInputSlices_[ringIndex].size(), rankSize),
            HCCL_E_PARA);
    }
    HCCL_INFO("ScatterDoubleRingDirect CheckParameters success");
    return HCCL_SUCCESS;
}

HcclResult ScatterDoubleRingDirect::MemcpyByOneRank()
{
    for (u32 ringIndex = 0; ringIndex < multiRingSlices_.size(); ringIndex++) {
        const Slice& srcSlice = userMemInputSlices_[ringIndex][0];
        const Slice& dstSlice = multiRingSlices_[ringIndex][0];
        DeviceMem src = DeviceMem::create(static_cast<u8*>(opInfo_->inputAddr) + srcSlice.offset, srcSlice.size);
        DeviceMem dst;
        if (opInfo_->outputAddr != nullptr) {
            // opInfo_->outputAddr != nullptr指示要将输出发送至user output
            u64 stepOffset = multiRingSlices_[ringIndex][ringsOrders_[ringIndex][0]].offset;
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
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }
    return HCCL_SUCCESS;
}

HcclResult
ScatterDoubleRingDirect::GetInitializedNeighborLinks(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 收集左邻居信息
    leftLink_ = links[(rank + rankSize - 1) % rankSize];
    CHK_SMART_PTR_NULL(leftLink_);

    // 收集右邻居信息
    rightLink_ = links[(rank + 1) % rankSize];
    CHK_SMART_PTR_NULL(rightLink_);
    HCCL_INFO("ScatterDoubleRingDirect finished to GetInitializedNeighborLinks");
    return HCCL_SUCCESS;
}

HcclResult ScatterDoubleRingDirect::RunInitStep(const u32 rank, const u32 rankSize)
{
    if (rank != root_) {
        return HCCL_SUCCESS;
    }
    for (u32 ringIndex = 0; ringIndex < multiRingSlices_.size(); ringIndex++) {
        u32 initSlice0Idx = 0;
        if (ringIndex == 0) {
            initSlice0Idx = (rank + rankSize - 1) % rankSize;
        } else {
            initSlice0Idx = (subRingRank_ + rankSize - 1) % rankSize;
        }
        const Slice& srcInitSlice0 = userMemInputSlices_[ringIndex][initSlice0Idx];
        DeviceMem srcInit
            = DeviceMem::create(static_cast<u8*>(opInfo_->inputAddr) + srcInitSlice0.offset, srcInitSlice0.size);
        const Slice& dstInitSlice0 = multiRingSlices_[ringIndex][initSlice0Idx];
        DeviceMem dstInit = inputMem_.range(dstInitSlice0.offset, dstInitSlice0.size);
        HCCL_DEBUG(
            "Memcpy operation: step[-1] stream[sub] src rank[%u] starts to copy(rcv) offset[%llu], size[%llu] "
            "on userMemInput to offset[%llu], size[%llu] on CCL",
            userRank_, srcInitSlice0.offset, srcInitSlice0.size, dstInitSlice0.offset, dstInitSlice0.size);
        if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstInit, srcInit, stream_));
        }
    }
    // 第-1步，片内将部分数据从userIn搬到cclIn
    return HCCL_SUCCESS;
}

HcclResult ScatterDoubleRingDirect::RunAllStreams(
    const u32 rank, const u32 step, const u32 rankSize, RxMemoryInfo& mainRxMem, RxMemoryInfo& subRxMem,
    DeviceMem& mainLocalSrcMem, DeviceMem& mainLocalDstMem, DeviceMem& subLocalSrcMem, DeviceMem& subLocalDstMem)
{
    (void)step;
    (void)rankSize;
    Stream mainStream = stream_;
    LINK mainPreLink = rightLink_;
    LINK mainNextLink = leftLink_;
    Stream subStream = subStreams_[0];
    LINK subPreLink = leftLink_;
    LINK subNextLink = rightLink_;

    // 唤醒所有环的主流做跨片同步
    CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[0], profilerInput_.stage));
    CHK_RET(LocalNotify::Wait(subStreams_[0], dispatcher_, subSignals_[0], profilerInput_.stage));

    CHK_RET(mainNextLink->TxAck(mainStream));
    CHK_RET(subNextLink->TxAck(subStream));

    CHK_RET(mainPreLink->RxAck(mainStream));
    CHK_RET(subPreLink->RxAck(subStream));

    // 回到主流
    CHK_RET(LocalNotify::Post(subStreams_[0], dispatcher_, mainSignals_[0], profilerInput_.stage));
    CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[0], profilerInput_.stage));

    // 主流唤醒所有流做跨片拷贝和片内拷贝
    CHK_RET(MainRecordSub());

    // 小数据量减少冗余通信，
    const u64 SMALL_DATASIZE = 4 * 1024 * 1024;
    bool isLargeCount = opInfo_->count * SIZE_TABLE[opInfo_->dataType] > SMALL_DATASIZE;
    if (isLargeCount || (step + 1 >= (root_ + rankSize - rank) % rankSize)) {
        CHK_RET(RxAsyncMemcpy(mainRxMem, mainStream, mainPreLink));
    }
    if (isLargeCount || (step + 1 >= (rank + rankSize - root_) % rankSize)) {
        CHK_RET(RxAsyncMemcpy(subRxMem, subStream, subPreLink));
    }

    // 本地拷贝
    if (rank == root_ && GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        if (mainLocalDstMem != mainLocalSrcMem) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, mainLocalDstMem, mainLocalSrcMem, subStreams_[1]));
        }
        if (subLocalDstMem != subLocalSrcMem) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, subLocalDstMem, subLocalSrcMem, subStreams_[2]));
        }
    }

    CHK_RET(mainPreLink->TxDataSignal(mainStream));
    CHK_RET(subPreLink->TxDataSignal(subStream));

    CHK_RET(mainNextLink->RxDataSignal(mainStream));
    CHK_RET(subNextLink->RxDataSignal(subStream));

    CHK_RET(MainWaitSub());
    return HCCL_SUCCESS;
}

HcclResult ScatterDoubleRingDirect::RxAsyncMemcpy(RxMemoryInfo& mem, Stream& stream, LINK& link)
{
    CHK_PTR_NULL(mem.dst);
    void* srcMemPtr = nullptr;
    CHK_RET(link->GetRemoteMem(mem.srcMemType, &srcMemPtr));

    DeviceMem srcDevMem(static_cast<s8*>(srcMemPtr) + mem.srcOffset, mem.len);
    DeviceMem dstDevMem(static_cast<s8*>(mem.dst), mem.len);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstDevMem, srcDevMem, stream, link->GetRemoteRank(), link->GetLinkType()));
    return HCCL_SUCCESS;
}

HcclResult ScatterDoubleRingDirect::PrepareDeviceMems(
    const u32 rank, const u32 step, const u32 ringIndex, const u32 rankSize, const u32 subSliceIdx,
    const u32 rxSliceIdx, RxMemoryInfo& rxMem, DeviceMem& localSrcMem, DeviceMem& localDstMem)
{
    const Slice& subSlice = userMemInputSlices_[ringIndex][subSliceIdx];
    const Slice& cclSlice = multiRingSlices_[ringIndex][subSliceIdx];
    const Slice& rxSlice = multiRingSlices_[ringIndex][rxSliceIdx];

    u64 lastStepOffset = multiRingSlices_[ringIndex][ringsOrders_[ringIndex][0]].offset;

    DeviceMem dst;
    if (step == rankSize - DMA_REDUCE_TWO_OFFSET && opInfo_->outputAddr != nullptr) {
        HCCL_DEBUG(
            "MemcpyAsync operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu], "
            "size[%llu] at userMemOut_",
            step, userRank_, lastStepOffset, rxSlice.size);
        dst = DeviceMem::create(static_cast<u8*>(opInfo_->outputAddr) + lastStepOffset, rxSlice.size);
    } else {
        HCCL_DEBUG(
            "MemcpyAsync operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu], "
            "size[%llu] at inputMem_",
            step, userRank_, rxSlice.offset, rxSlice.size);
        dst = inputMem_.range(rxSlice.offset, rxSlice.size);
    }
    rxMem = RxMemoryInfo{UserMemType::INPUT_MEM, rxSlice.offset + baseOffset_, dst.ptr(), rxSlice.size};

    if (rank == root_) {
        // root节点参与通信，避免片内拷贝带宽过高过分抢占HBM带宽导致其他节点读取数据时性能下降
        u32 rxSliceIdxForRoot = (rxSliceIdx + DMA_REDUCE_TWO_OFFSET) % rankSize;
        Slice rxSliceForRoot = multiRingSlices_[ringIndex][rxSliceIdxForRoot];
        if ((rankSize < RANK_SIZE_THREE && step == 0) || (inputMem_.ptr() == opInfo_->inputAddr))
            rxSliceForRoot.size = 0;
        dst = inputMem_.range(rxSliceForRoot.offset, rxSliceForRoot.size);
        rxMem
            = RxMemoryInfo{UserMemType::INPUT_MEM, rxSliceForRoot.offset + baseOffset_, dst.ptr(), rxSliceForRoot.size};
        HCCL_DEBUG(
            "Memcpy operation: step[%u] stream[sub], src rank[%u] starts to send offset[%llu], size[%llu] "
            "from userMemIn_",
            step, userRank_, subSlice.offset, subSlice.size);
        localSrcMem = DeviceMem::create(static_cast<u8*>(opInfo_->inputAddr) + subSlice.offset, subSlice.size);
        if (step == rankSize - DMA_REDUCE_TWO_OFFSET && opInfo_->outputAddr != nullptr) {
            HCCL_DEBUG(
                "Memcpy operation: step[%u] stream[sub], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
                "to userMemOut_",
                step, userRank_, lastStepOffset, subSlice.size);
            localDstMem = DeviceMem::create(static_cast<u8*>(opInfo_->outputAddr) + lastStepOffset, subSlice.size);
        } else {
            HCCL_DEBUG(
                "Memcpy operation: step[%u] stream[sub], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
                "to inputMem_",
                step, userRank_, cclSlice.offset, cclSlice.size);
            localDstMem = inputMem_.range(cclSlice.offset, cclSlice.size);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterDoubleRingDirect::RunScatter(const u32 rank, const u32 rankSize)
{
    HCCL_INFO("ScatterDoubleRingDirect starts, the input param rank[%u]", rank);

    CHK_RET(RunInitStep(rank, rankSize));

    // 例如rank[0,1,2,3]中，rank0的rxSliceIdx = 2，txSliceIdx = 3, subSliceIdx = 1
    u32 subSliceIdx = (rank + rankSize - DMA_REDUCE_TWO_OFFSET) % rankSize;
    u32 mainSliceIdx = (subRingRank_ + rankSize - DMA_REDUCE_TWO_OFFSET) % rankSize;

    for (u32 step = 0; step < rankSize - 1; step++) {
        RxMemoryInfo rxMemSub;
        DeviceMem localSrcMemSub;
        DeviceMem localDstMemSub;
        CHK_RET(PrepareDeviceMems(
            rank, step, ALIGNED_SUB_RING_INDEX, rankSize, subSliceIdx, subSliceIdx, rxMemSub, localSrcMemSub,
            localDstMemSub));
        RxMemoryInfo rxMemMain;
        DeviceMem localSrcMemMain;
        DeviceMem localDstMemMain;
        CHK_RET(PrepareDeviceMems(
            rank, step, ALIGNED_MAIN_RING_INDEX, rankSize, mainSliceIdx, mainSliceIdx, rxMemMain, localSrcMemMain,
            localDstMemMain));

        CHK_RET(RunAllStreams(
            rank, step, rankSize, rxMemMain, rxMemSub, localSrcMemMain, localDstMemMain, localSrcMemSub,
            localDstMemSub));

        // 更新索引
        mainSliceIdx = (mainSliceIdx + rankSize - 1) % rankSize;
        subSliceIdx = (subSliceIdx + rankSize - 1) % rankSize;
    }
    HCCL_INFO("ScatterDoubleRingDirect finished to RunScatter");
    return HCCL_SUCCESS;
}

// 主流通知从流干活, 从流等待主流
HcclResult ScatterDoubleRingDirect::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < subSignals_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[signalIndex], profilerInput_.stage));
    }
    for (u32 signalIndex = 0; signalIndex < subSignals_.size(); signalIndex++) {
        CHK_RET(
            LocalNotify::Wait(subStreams_[signalIndex], dispatcher_, subSignals_[signalIndex], profilerInput_.stage));
    }
    for (u32 signalIndex = 0; signalIndex < subSignals_.size(); signalIndex++) {
        CHK_RET(ExecEmptyTask(inputMem_, outputMem_, subStreams_[signalIndex], dispatcher_));
    }
    CHK_RET(ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    return HCCL_SUCCESS;
}

// 主流等待从流, 从流告诉主流活干完了
HcclResult ScatterDoubleRingDirect::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < mainSignals_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[signalIndex], profilerInput_.stage));
    }
    for (u32 signalIndex = 0; signalIndex < mainSignals_.size(); signalIndex++) {
        CHK_RET(
            LocalNotify::Post(subStreams_[signalIndex], dispatcher_, mainSignals_[signalIndex], profilerInput_.stage));
    }
    for (u32 signalIndex = 0; signalIndex < mainSignals_.size(); signalIndex++) {
        CHK_RET(ExecEmptyTask(inputMem_, outputMem_, subStreams_[signalIndex], dispatcher_));
    }
    CHK_RET(ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_SCATTER_DOUBLE_RING_DIRECT, ScatterDoubleRingDirect);
} // namespace hccl
