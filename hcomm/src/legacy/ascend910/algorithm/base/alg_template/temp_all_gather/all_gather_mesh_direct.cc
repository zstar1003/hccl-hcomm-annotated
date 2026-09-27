/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_mesh_direct.h"
#include "alg_template_register.h"
// userin -> dmaout -> userout
namespace hccl {
AllgatherMeshDirect::AllgatherMeshDirect(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllgatherMeshDirect::~AllgatherMeshDirect() {}

HcclResult AllgatherMeshDirect::Prepare(
    std::vector<Stream>& meshStreams, std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
    std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, u32 userRank, HcomCollOpInfo* opInfo, u32 interRank,
    u32 interRankSize)
{
    meshStreams_ = meshStreams;
    meshSignal_ = &meshSignal;
    meshSignalAux_ = &meshSignalAux;
    opInfo_ = opInfo;
    interRank_ = interRank;
    interRankSize_ = interRankSize;
    userRank_ = userRank;
    return HCCL_SUCCESS;
}

HcclResult AllgatherMeshDirect::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < (*meshSignalAux_).size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAux_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllgatherMeshDirect::SubWaitMain()
{
    for (u32 streamIndex = 0; streamIndex < (*meshSignalAux_).size(); streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            meshStreams_[streamIndex], dispatcher_, (*meshSignalAux_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllgatherMeshDirect::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < (*meshSignal_).size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignal_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllgatherMeshDirect::SubRecordMain()
{
    for (u32 streamIndex = 0; streamIndex < (*meshSignal_).size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(
            meshStreams_[streamIndex], dispatcher_, (*meshSignal_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

// allgather的入口函数
HcclResult AllgatherMeshDirect::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO(
        "AllGatherMeshDirect run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    char* curUerMemInPtr = static_cast<char*>(opInfo_->inputAddr);
    char* curUerMemOutPtr = static_cast<char*>(opInfo_->outputAddr);
    char* curCommMemOutPtr = static_cast<char*>(outputMem_.ptr());

    u32 unitSize = DataUnitSize(dataType_);
    u64 curSize = count_ * unitSize;           // 当前count
    u64 sliceSize = opInfo_->count * unitSize; // 总输入count

    if (rankSize == 1) {
        if (opInfo_->inputAddr != opInfo_->outputAddr) {
            HCCL_DEBUG("rank[%u] mem copy async from input to output", rank);
            DeviceMem userMemIn = DeviceMem::create(curUerMemInPtr, curSize);
            DeviceMem userMemOut = DeviceMem::create(curUerMemOutPtr, curSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, userMemOut, userMemIn, stream_));
        }
        return HCCL_SUCCESS;
    }

    DeviceMem emptyMem = outputMem_.range(0, 0);

    std::vector<Slice> inputSlices(slices_);
    if (slices_.size() == 0) {
        // slices_为空，临时构造等长slices
        slices_.resize(interRankSize_);
        inputSlices.resize(interRankSize_);

        for (u32 i = 0; i < interRankSize_; i++) {
            slices_[i].size = curSize;
            slices_[i].offset = (i * sliceSize);

            inputSlices[i].size = curSize;
            inputSlices[i].offset = (inputMem_.size() < outputMem_.size()) ? 0 : (sliceSize * i);
        }
    } else {
        // allgather_v场景下走else分支，每张卡的数据在CCLbuffer上偏移地址相同
        for (u32 i = 0; i < interRankSize_; i++) {
            inputSlices[i].offset = 0;
        }
    }

    for (u32 i = 0; i < interRankSize_; i++) {
        HCCL_DEBUG(
            "[AllGatherMeshDirect][Slice]: rank[%u], outputslice: size[%llu] offset[%llu] "
            "inputslice: size[%llu] offset[%llu]",
            i, slices_[i].size, slices_[i].offset, inputSlices[i].size, inputSlices[i].offset);
    }

    DeviceMem src;
    DeviceMem dst;
    src = DeviceMem::create(curUerMemInPtr, inputSlices[rank].size);
    u64 localOffsetByte = inputSlices[rank].offset % HCCL_MIN_SLICE_ALIGN_910B;
    dst = DeviceMem::create(curCommMemOutPtr + localOffsetByte, inputSlices[rank].size);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = BackwardRank(rank, rankSize, round);
        Stream& subStream = meshStreams_[round - 1];
        CHK_RET(links[dstRank]->TxAck(subStream));
        CHK_RET(links[dstRank]->RxAck(subStream));
    }

    HCCL_DEBUG("[AllgatherMeshDirect]runAsync now rankSize is %u", rankSize);
    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyMem, emptyMem, stream_));

    CHK_RET(SubWaitMain());
    CHK_RET(MainRecordSub());

    src = dst;
    dst = DeviceMem::create(curUerMemOutPtr + slices_[rank].offset, slices_[rank].size);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = BackwardRank(rank, rankSize, round);
        Stream& subStream = meshStreams_[round - 1];
        // 本rank要收数据
        void* remMemPtr = nullptr;
        // DMA消减场景，从对端的ccl out内存拿数据到本端的user out
        CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
        u64 remoteOffsetByte = inputSlices[dstRank].offset % HCCL_MIN_SLICE_ALIGN_910B;
        src = DeviceMem::create(static_cast<char*>(remMemPtr) + remoteOffsetByte, inputSlices[dstRank].size);
        dst = DeviceMem::create(curUerMemOutPtr + slices_[dstRank].offset, slices_[dstRank].size);
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dst, src, subStream, links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType()));
        CHK_RET(links[dstRank]->TxDataSignal(subStream));
        CHK_RET(links[dstRank]->RxDataSignal(subStream));
    }
    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyMem, emptyMem, stream_));

    HCCL_INFO("AllGatherMeshDirect finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_MESH_DIRECT, AllgatherMeshDirect);
} // namespace hccl
