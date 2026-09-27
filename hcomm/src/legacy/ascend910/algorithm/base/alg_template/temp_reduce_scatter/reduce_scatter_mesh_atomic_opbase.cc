/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_mesh_atomic_opbase.h"
#include "alg_template_register.h"

namespace hccl {
using namespace std;

ReduceScatterMeshDirect::ReduceScatterMeshDirect(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

ReduceScatterMeshDirect::~ReduceScatterMeshDirect() {}

HcclResult ReduceScatterMeshDirect::Prepare(
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
    opInfo_ = opInfo;
    return AlgTemplateBase::Prepare(
        inputMem, outputMem, scratchMem, count, dataType, stream, reductionOp, root, slices, baseOffset);
}

HcclResult ReduceScatterMeshDirect::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignalAuxPtr_->size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAuxPtr_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshDirect::SubWaitMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignalAuxPtr_->size(); streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            meshStreams_[streamIndex], dispatcher_, (*meshSignalAuxPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshDirect::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignalPtr_->size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignalPtr_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshDirect::SubRecordMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignalPtr_->size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(
            meshStreams_[streamIndex], dispatcher_, (*meshSignalPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMeshDirect::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO(
        "ReduceScatterMeshDirect run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    // 数据准备
    u32 unitSize = SIZE_TABLE[dataType_];

    if (slices_.size() == 0) {
        // slices_为空，临时构造等长slices
        slices_.resize(rankSize);
        u64 curSize = count_ * unitSize;
        u64 sliceSize = (opInfo_->count) * unitSize;

        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = curSize;
            slices_[i].offset = (i * sliceSize);
        }
    }

    DeviceMem userMemIn
        = DeviceMem::create(static_cast<char*>(opInfo_->inputAddr) + slices_[rank].offset, slices_[rank].size);
    DeviceMem userMemOut = DeviceMem::create(static_cast<char*>(opInfo_->outputAddr), slices_[rank].size);
    DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), outputMem_.size());

    DeviceMem src;
    DeviceMem dst;

    dst = commMemOut.range(0, slices_[rank].size);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, userMemIn, stream_));

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    // 每个stream只负责一个对端的交互
    HCCL_DEBUG("[ReduceScatterMeshDirect][RunAsync]rankSize is %u", rankSize);
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
    DeviceMem srcZero = DeviceMem::create(inputMem_.ptr(), 0);
    DeviceMem dstZero = DeviceMem::create(outputMem_.ptr(), 0);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstZero, srcZero, stream_));

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
        dst = DeviceMem::create(static_cast<char*>(remMemPtr), slices_[dstRank].size);
        src = DeviceMem::create(
            static_cast<char*>(opInfo_->inputAddr) + slices_[dstRank].offset, slices_[dstRank].size);
        u64 curCount = slices_[dstRank].size / unitSize;
        CHK_RET(HcclReduceAsync(
            dispatcher_, static_cast<void*>(src.ptr()), curCount, dataType_, reductionOp_, subStream,
            static_cast<void*>(dst.ptr()), dstLink->GetRemoteRank(), dstLink->GetLinkType(), INLINE_REDUCE_BIT));

        CHK_RET(dstLink->TxDataSignal(subStream));
        CHK_RET(dstLink->RxDataSignal(subStream));
    }

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    // commout--> useroutput
    DeviceMem srcMem = commMemOut.range(0, slices_[rank].size);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, userMemOut, srcMem, stream_));

    HCCL_INFO("ReduceScatterMeshDirect finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_MESH_DIRECT, ReduceScatterMeshDirect);
} // namespace hccl
