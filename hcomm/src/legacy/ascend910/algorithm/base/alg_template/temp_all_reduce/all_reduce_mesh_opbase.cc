/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_register.h"
#include "all_reduce_mesh_opbase.h"

namespace hccl {
AllReduceMeshDirect::AllReduceMeshDirect(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllReduceMeshDirect::~AllReduceMeshDirect() {}

HcclResult AllReduceMeshDirect::Prepare(
    u64 reduceAttrBitMap, std::vector<Stream>& meshStreams, std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
    std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, u32 interRank, u32 interRankSize, u32 userRank,
    HcomCollOpInfo* opInfo)
{
    reduceAttr_ = reduceAttrBitMap;
    localRank_ = interRank;
    localRankSize_ = interRankSize;
    userRank_ = userRank;
    meshStreams_ = meshStreams;
    meshSignal_ = &meshSignal;
    meshSignalAux_ = &meshSignalAux;
    opInfo_ = opInfo;
    return HCCL_SUCCESS;
}

HcclResult AllReduceMeshDirect::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignalAux_->size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAux_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceMeshDirect::SubWaitMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignalAux_->size(); streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            meshStreams_[streamIndex], dispatcher_, (*meshSignalAux_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceMeshDirect::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignal_->size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignal_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceMeshDirect::SubRecordMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignal_->size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(
            meshStreams_[streamIndex], dispatcher_, (*meshSignal_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

// 将数据均分，最小单位是128
HcclResult AllReduceMeshDirect::PrepareSlice(u64 dataCount, u32 unitSize, u32 sliceNum, std::vector<Slice>& dataSlice)
{
    Slice temp;
    u64 totalSize = dataCount * unitSize;
    dataSlice.clear();
    dataSlice.reserve(sliceNum);
    if (sliceNum == 0) {
        HCCL_ERROR("[Prepare][SliceData]data slice prepare, sliceNum is 0");
        return HCCL_E_PARA;
    }
    u64 sizePerSlice = (totalSize + sliceNum - 1) / sliceNum; /* 1是为了向上取整 */
    sizePerSlice = RoundUpWithDivisor(sizePerSlice, HCCL_MIN_SLICE_ALIGN_910B);
    u64 residueSize = totalSize;
    u32 i = 0;
    while (residueSize > 0) {
        u64 sliceSize = sizePerSlice < residueSize ? sizePerSlice : residueSize;
        temp.size = sliceSize;
        temp.offset = totalSize - residueSize;
        i++;
        if (sliceSize <= 0) {
            HCCL_ERROR("[Prepare][SliceData]data_slice_prepare sliceSize[%llu]", sliceSize);
            return HCCL_E_PARA;
        }
        residueSize -= sliceSize;
        dataSlice.push_back(temp);
    }
    while (i < sliceNum) {
        temp.size = 0;
        temp.offset = totalSize;
        i++;
        dataSlice.push_back(temp);
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceMeshDirect::PrepareAllreduceSliceData()
{
    return PrepareSlice(count_, DataUnitSize(dataType_), localRankSize_, slices_);
}

// ring allreduce算法的函数入口
HcclResult AllReduceMeshDirect::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO(
        "AllReduceMeshDirect run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    if (links.size() < rankSize) {
        HCCL_ERROR(
            "[AllReduceMeshDirect][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]", rank, links.size(),
            rankSize);
        return HCCL_E_INTERNAL;
    }

    // 如果ranksize为1, inline reduce和普通跨片reduce操作一致，从input->output
    if (rankSize == 1) {
        if (opInfo_->inputAddr != opInfo_->outputAddr) {
            DeviceMem userMemIn = DeviceMem::create(opInfo_->inputAddr, count_ * DataUnitSize(dataType_));
            DeviceMem userMemOut = DeviceMem::create(opInfo_->outputAddr, count_ * DataUnitSize(dataType_));
            ret = HcclD2DMemcpyAsync(dispatcher_, userMemOut, userMemIn, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceRing][RunAsync]rank[%u] memcpy async failed", rank), ret);
        }
        return ret;
    }

    ret = PrepareAllreduceSliceData();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceRing][RunAsync]rank[%u] count[%llu] failed in PrepareSliceData "
            "step",
            rank, count_),
        ret);

    ret = RunReduceScatter(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceRing][RunAsync]rank[%u] count[%llu] failed in ReduceScatter "
            "step",
            rank, count_),
        ret);

    ret = RunAllGather(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceRing][RunAsync]rank[%u] count[%llu] failed in AllGather "
            "step",
            rank, count_),
        ret);

    HCCL_INFO("AllReduceMeshDirect finished: rank[%u] ranksize[%u].", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceMeshDirect::RunReduceScatter(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO(
        "ReduceScatterMeshAtomicOpbase run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu].", rank,
        rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    // 数据准备
    u32 unitSize = DataUnitSize(dataType_);

    DeviceMem userMemIn = DeviceMem::create(opInfo_->inputAddr, count_ * unitSize);
    DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), outputMem_.size());

    DeviceMem src;
    DeviceMem dst;

    src = DeviceMem::create(static_cast<char*>(opInfo_->inputAddr) + slices_[rank].offset, slices_[rank].size);
    dst = commMemOut.range(0, slices_[rank].size);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = (round + rank) % rankSize;
        Stream& subStream = meshStreams_[round - 1];
        CHK_RET(links[dstRank]->TxAck(subStream));
        CHK_RET(links[dstRank]->RxAck(subStream));
    }

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());
    DeviceMem srcTmp = DeviceMem::create(inputMem_.ptr(), 0);
    DeviceMem dstTmp = DeviceMem::create(outputMem_.ptr(), 0);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstTmp, srcTmp, stream_));
    CHK_RET(SubWaitMain());
    CHK_RET(MainRecordSub());

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = (round + rank) % rankSize;
        Stream& subStream = meshStreams_[round - 1];

        void* remMemPtr = nullptr;
        CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));

        dst = DeviceMem::create(static_cast<char*>(remMemPtr), slices_[dstRank].size);
        src = userMemIn.range(slices_[dstRank].offset, slices_[dstRank].size);
        CHK_RET(HcclReduceAsync(
            dispatcher_, static_cast<void*>(src.ptr()), slices_[dstRank].size / unitSize, dataType_, reductionOp_,
            subStream, static_cast<void*>(dst.ptr()), links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType(),
            INLINE_REDUCE_BIT));

        CHK_RET(links[dstRank]->TxDataSignal(subStream));
        CHK_RET(links[dstRank]->RxDataSignal(subStream));
    }

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());
    return HCCL_SUCCESS;
}

HcclResult AllReduceMeshDirect::RunAllGather(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO(
        "AllGatherMesh run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu].", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);
    u32 unitSize = DataUnitSize(dataType_);

    DeviceMem userMemOut = DeviceMem::create(opInfo_->outputAddr, count_ * unitSize);
    DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), outputMem_.size());

    DeviceMem src;
    DeviceMem dst;

    src = commMemOut.range(0, slices_[rank].size);
    dst = userMemOut.range(slices_[rank].offset, slices_[rank].size);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = BackwardRank(rank, rankSize, round);
        Stream& subStream = meshStreams_[round - 1];
        CHK_RET(links[dstRank]->TxAck(subStream));
        CHK_RET(links[dstRank]->RxAck(subStream));
    }

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());
    DeviceMem srcTmp = DeviceMem::create(inputMem_.ptr(), 0);
    DeviceMem dstTmp = DeviceMem::create(outputMem_.ptr(), 0);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstTmp, srcTmp, stream_));
    CHK_RET(SubWaitMain());
    CHK_RET(MainRecordSub());

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = BackwardRank(rank, rankSize, round);
        Stream& subStream = meshStreams_[round - 1];
        void* remMemPtr = nullptr;
        CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
        src = DeviceMem::create(static_cast<char*>(remMemPtr), slices_[dstRank].size);
        dst = userMemOut.range(slices_[dstRank].offset, slices_[dstRank].size);
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dst, src, subStream, links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType()));
        CHK_RET(links[dstRank]->TxDataSignal(subStream));
        CHK_RET(links[dstRank]->RxDataSignal(subStream));
    }

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    HCCL_INFO("AllGatherMesh finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_MESH_DIRECT, AllReduceMeshDirect);
} // namespace hccl
