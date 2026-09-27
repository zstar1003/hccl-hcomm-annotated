/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_mesh_mix.h"
#include "alg_template_register.h"

namespace hccl {
using namespace std;

ReduceScatterMeshMix::ReduceScatterMeshMix(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

ReduceScatterMeshMix::~ReduceScatterMeshMix() {}

HcclResult ReduceScatterMeshMix::Prepare(
    DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
    const Stream& stream, const HcclReduceOp reductionOp, const u32 root, const std::vector<Slice>& slices,
    const u64 baseOffset, const u64 reduceAttrBitMap, std::vector<Stream>& meshStreams,
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
    const std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, u32 interRank, u32 interRankSize,
    HcomCollOpInfo* opInfo)
{
    reduceAttr_ = reduceAttrBitMap;
    meshStreams_ = meshStreams;
    meshSignalPtr_ = &meshSignal;
    meshSignalAuxPtr_ = &meshSignalAux;
    interRank_ = interRank;
    interRankSize_ = interRankSize;
    opInfo_ = opInfo;
    return AlgTemplateBase::Prepare(
        inputMem, outputMem, scratchMem, count, dataType, stream, reductionOp, root, slices, baseOffset);
}

HcclResult ReduceScatterMeshMix::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignalAuxPtr_->size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAuxPtr_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshMix::SubWaitMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignalAuxPtr_->size(); streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            meshStreams_[streamIndex], dispatcher_, (*meshSignalAuxPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshMix::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignalPtr_->size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignalPtr_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshMix::SubRecordMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignalPtr_->size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(
            meshStreams_[streamIndex], dispatcher_, (*meshSignalPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshMix::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO(
        "ReduceScatterMeshMix run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    // 数据准备
    u32 unitSize = SIZE_TABLE[dataType_];
    u64 totalSize = opInfo_->count * unitSize;
    u64 sliceSize = count_ * unitSize;

    u8* curUerMemInPtr = static_cast<u8*>(opInfo_->inputAddr);
    u8* curCommMemOutPtr = static_cast<u8*>(outputMem_.ptr());

    DeviceMem userMemIn = DeviceMem::create(curUerMemInPtr + totalSize * rank, sliceSize);
    DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), outputMem_.size());

    DeviceMem src;
    DeviceMem dst;

    for (u32 i = 0; i < interRankSize_; i++) {
        src = DeviceMem::create(curUerMemInPtr + (i * rankSize + rank) * totalSize, sliceSize);
        dst = DeviceMem::create(curCommMemOutPtr + (i * rankSize + rank) * sliceSize, sliceSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    // 每个stream只负责一个对端的交互
    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = (round + rank) % rankSize;
        const LINK& dstLink = links[dstRank];
        Stream& subStream = meshStreams_[round - 1];
        CHK_RET(dstLink->TxAck(subStream));
        CHK_RET(dstLink->RxAck(subStream));
    }
    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());
    // 为子图增加一个从stream到主stream的附着点
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));

    CHK_RET(SubWaitMain());
    CHK_RET(MainRecordSub());

    // inline执行notice reduce
    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = (round + rank) % rankSize;
        const LINK& dstLink = links[dstRank];
        Stream& subStream = meshStreams_[round - 1];
        // 本rank要发数据
        void* remMemPtr = nullptr;
        // 获取远端的commoutMem
        CHK_RET(dstLink->GetRemoteMem(UserMemType::INPUT_MEM, &remMemPtr));

        for (u32 i = 0; i < interRankSize_; i++) {
            src = DeviceMem::create(curUerMemInPtr + (i * rankSize + dstRank) * totalSize, sliceSize);
            dst = DeviceMem::create(static_cast<u8*>(remMemPtr) + (i * rankSize + dstRank) * sliceSize, sliceSize);
            CHK_RET(HcclReduceAsync(
                dispatcher_, static_cast<void*>(src.ptr()), count_, dataType_, reductionOp_, subStream,
                static_cast<void*>(dst.ptr()), dstLink->GetRemoteRank(), dstLink->GetLinkType(), INLINE_REDUCE_BIT));
        }

        CHK_RET(dstLink->TxDataSignal(subStream));
        CHK_RET(dstLink->RxDataSignal(subStream));
    }

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));

    HCCL_INFO("ReduceScatterMeshMix finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_MESH_MIX, ReduceScatterMeshMix);
} // namespace hccl
