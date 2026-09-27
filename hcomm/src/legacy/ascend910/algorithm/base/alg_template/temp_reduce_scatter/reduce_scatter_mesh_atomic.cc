/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_mesh_atomic.h"
#include "alg_template_register.h"

namespace hccl {
using namespace std;

ReduceScatterMeshAtomic::ReduceScatterMeshAtomic(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

ReduceScatterMeshAtomic::~ReduceScatterMeshAtomic() {}

HcclResult ReduceScatterMeshAtomic::Prepare(
    DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
    const Stream& stream, const HcclReduceOp reductionOp, const u32 root, const std::vector<Slice>& slices,
    const u64 baseOffset, const u64 reduceAttrBitMap, std::vector<Stream>& meshStreams,
    std::vector<std::shared_ptr<LocalNotify>>& meshSignal, std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux,
    u32 userRank, const HcomCollOpInfo* opInfo)
{
    reduceAttr_ = reduceAttrBitMap;
    userRank_ = userRank;
    meshStreams_ = meshStreams;
    meshSignalPtr_ = &meshSignal;
    meshSignalAuxPtr_ = &meshSignalAux;
    (void)opInfo;
    return AlgTemplateBase::Prepare(
        inputMem, outputMem, scratchMem, count, dataType, stream, reductionOp, root, slices, baseOffset);
}

HcclResult ReduceScatterMeshAtomic::RunReduceScatter(const std::vector<LINK>& links)
{
    // 拼接所有stream
    vector<Stream> streamVct;
    streamVct.reserve(localRankSize_ - 1); // 有ranksize-1个对端，每个对端对应一条stream
    streamVct.push_back(stream_);          // 增加主stream
    streamVct.insert(streamVct.end(), meshStreams_.begin(), meshStreams_.end()); // 增加从stream

    // 每个stream只负责一个对端的交互
    for (u32 streamIndex = 0; streamIndex < localRankSize_ - 1; streamIndex++) {
        u32 remoteRank = (streamIndex + localRank_ + 1) % localRankSize_;
        const LINK& dstLink = links[remoteRank];
        Stream& stream = streamVct[streamIndex];

        CHK_RET(dstLink->TxAck(stream));
        CHK_RET(dstLink->RxAck(stream));
    }

    for (u32 streamIndex = 0; streamIndex < localRankSize_ - 1; streamIndex++) {
        u32 remoteRank = (streamIndex + localRank_ + 1) % localRankSize_;
        const LINK& dstLink = links[remoteRank];
        Stream& stream = streamVct[streamIndex];
        profilerInput_.streamID = stream.id();
        profilerInput_.planeID = streamIndex - 1;
        profilerInput_.step = HCCL_EXEC_STEP_NOT_SET;

        Slice& rxSlice = slices_[localRank_];
        if (streamIndex == 0) {
            for (u32 signalIndex = 0; signalIndex < localRankSize_ - 2; signalIndex++) { // rankSize-2: stream num
                CHK_RET(LocalNotify::Wait(stream, dispatcher_, (*meshSignalPtr_)[signalIndex], profilerInput_.stage));
            }
            for (u32 signalIndex = 0; signalIndex < localRankSize_ - 2; signalIndex++) { // rankSize-2: stream num
                CHK_RET(
                    LocalNotify::Post(stream, dispatcher_, (*meshSignalAuxPtr_)[signalIndex], profilerInput_.stage));
            }
        } else {
            u32 signalIndex = streamIndex - 1;
            CHK_RET(LocalNotify::Post(stream, dispatcher_, (*meshSignalPtr_)[signalIndex], profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(stream, dispatcher_, (*meshSignalAuxPtr_)[signalIndex], profilerInput_.stage));
        }

        HCCL_DEBUG(
            "ReduceScatterMeshAtomic RX rank[%u] inputMemSize[%llu], rxSlice.offset[%llu], rxSlice.size[%llu] "
            "baseOffset_[%llu]",
            localRank_, inputMem_.size(), rxSlice.offset, rxSlice.size, baseOffset_);
        DeviceMem dstMem = inputMem_.range(rxSlice.offset, rxSlice.size);
        void* remoteMem = nullptr;
        CHK_RET(dstLink->GetRemoteMem(UserMemType::INPUT_MEM, &remoteMem));
        CHK_RET(HcclReduceAsync(
            dispatcher_, static_cast<s8*>(remoteMem) + baseOffset_ + rxSlice.offset,
            dstMem.size() / SIZE_TABLE[dataType_], dataType_, reductionOp_, stream, dstMem.ptr(),
            dstLink->GetRemoteRank(), dstLink->GetLinkType(), INLINE_REDUCE_BIT));

        CHK_RET(dstLink->TxDataSignal(stream));
        CHK_RET(dstLink->RxDataSignal(stream));
    }
    // 添加空task,保证执行时不乱序
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, streamVct[0], dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshAtomic::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO(
        "ReduceScatterMeshAtomic run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    localRank_ = rank;
    localRankSize_ = rankSize;

    if (localRankSize_ == 1) {
        if (inputMem_ != outputMem_) {
            return HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        return HCCL_SUCCESS;
    }

    if (links.size() < rankSize) {
        HCCL_ERROR("[ReduceScatterMeshAtomic][RunAsync]rank[%u] linksize[%zu] error", rank, links.size());
        return HCCL_E_INTERNAL;
    }
    CHK_RET(MemSlice());

    for (u32 streamIndex = 0; streamIndex < rankSize - 2; streamIndex++) { // rankSize-2: stream num
        HCCL_DEBUG(
            "rank[%u] streamindex[%u] wait signalaux[%p]", rank, streamIndex, (*meshSignalAuxPtr_)[streamIndex]->ptr());
        CHK_RET(LocalNotify::Wait(
            meshStreams_[streamIndex], dispatcher_, (*meshSignalAuxPtr_)[streamIndex], profilerInput_.stage));
    }
    for (u32 streamIndex = 0; streamIndex < rankSize - 2; streamIndex++) { // rankSize-2: stream num
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAuxPtr_)[streamIndex], profilerInput_.stage));
    }

    CHK_RET(RunReduceScatter(links));

    for (u32 streamIndex = 0; streamIndex < rankSize - 2; streamIndex++) { // rankSize - 2 stream num
        HCCL_DEBUG(
            "rank[%u] streamindex[%u] wait signal[%p] ", rank, streamIndex, (*meshSignalPtr_)[streamIndex]->ptr());
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignalPtr_)[streamIndex], profilerInput_.stage));
        CHK_RET(LocalNotify::Post(
            meshStreams_[streamIndex], dispatcher_, (*meshSignalPtr_)[streamIndex], profilerInput_.stage));
    }

    // 添加空task,保证执行时不乱序
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));

    if (inputMem_ != outputMem_) {
        DeviceMem src = inputMem_.range(slices_[localRank_].offset, slices_[localRank_].size);
        HCCL_DEBUG(
            "rank[%u] copy result from to output[%p] offset[%llu] size[%llu] ", localRank_, outputMem_.ptr(),
            slices_[localRank_].offset, slices_[localRank_].size);
        HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, src, stream_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ReduceScatterMeshAtomic][RunAsync]rank[%u] memcpy async from mem[%p] "
                "to outputmem[%p] failed",
                rank, src.ptr(), outputMem_.ptr()),
            ret);
    }

    HCCL_INFO("ReduceScatterMeshAtomic finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshAtomic::MemSlice()
{
    u32 unitSize = SIZE_TABLE[dataType_];
    if (unitSize == 0) {
        HCCL_ERROR("[ReduceScatterMeshAtomic][RunAsync]rank[%u] unit data size is zero", localRank_);
        return HCCL_E_INTERNAL;
    }

    if (HcclCheckLogLevel(DLOG_DEBUG)) {
        for (size_t i = 0; i < slices_.size(); i++) {
            HCCL_DEBUG(
                "[ReduceScatterMeshAtomic] rank[%u] index[%zu] size[%llu] offset[%llu]", localRank_, i, slices_[i].size,
                slices_[i].offset);
        }
        HCCL_DEBUG("[ReduceScatterMeshAtomic] localRankSize[%u]", localRankSize_);
    }

    if (slices_.size() == 0) {
        slices_.resize(localRankSize_);
        u64 sliceSize = count_ * unitSize;
        for (u32 i = 0; i < localRankSize_; i++) {
            slices_[i].size = sliceSize;
            slices_[i].offset = (i * sliceSize);
        }
    }

    scratchSlices_.resize(localRankSize_);
    for (u32 i = 0; i < localRankSize_; i++) {
        scratchSlices_[i].size = slices_[i].size;
        scratchSlices_[i].offset = (scratchMem_.size() < inputMem_.size()) ? 0 : slices_[i].offset;
    }

    if (HcclCheckLogLevel(DLOG_DEBUG)) {
        for (size_t i = 0; i < slices_.size(); i++) {
            HCCL_DEBUG(
                "ReduceScatterMeshAtomic rank[%u] index[%zu] "
                "size[%llu] offset[%llu] scratch_size[%llu] scratch_offset[%llu]",
                localRank_, i, slices_[i].size, slices_[i].offset, scratchSlices_[i].size, scratchSlices_[i].offset);
        }
    }

    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_MESH_ATOMIC, ReduceScatterMeshAtomic);
} // namespace hccl
