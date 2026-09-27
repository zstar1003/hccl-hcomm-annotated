/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include "alg_template_register.h"
#include "all_reduce_local_reduce.h"

namespace hccl {
AllReduceLocalReduce::AllReduceLocalReduce(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AllReduceLocalReduce::~AllReduceLocalReduce() {}

HcclResult AllReduceLocalReduce::Prepare(
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
HcclResult AllReduceLocalReduce::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignalAux_->size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAux_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduce::SubWaitMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignalAux_->size(); streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            meshStreams_[streamIndex], dispatcher_, (*meshSignalAux_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduce::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignal_->size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignal_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduce::SubRecordMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignal_->size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(
            meshStreams_[streamIndex], dispatcher_, (*meshSignal_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

// 将数据均分，最小单位是128
HcclResult AllReduceLocalReduce::PrepareSlice(
    u64 dataCount, u32 unitSize, u32 sliceNum, std::vector<Slice>& dataSlice, std::vector<Slice>& startSlice)
{
    Slice temp;
    Slice startTemp;
    u64 totalSize = dataCount * unitSize;
    dataSlice.clear();
    dataSlice.reserve(sliceNum);
    if (sliceNum == 0) {
        HCCL_ERROR("[Prepare][SliceData]data slice prepare, sliceNum is 0");
        return HCCL_E_PARA;
    }
    u64 sizePerSliceOri = (totalSize + sliceNum - 1) / sliceNum; /* 1是为了向上取整 */
    u64 sizeLimit = 0;
    if (outputMem_.ptr() == opInfo_->outputAddr) {
        sizeLimit = outputMem_.size();
    } else {
        sizeLimit = totalSize;
    }

    u64 sizePerSlice = RoundUpWithDivisor(sizePerSliceOri, HCCL_MIN_SLICE_ALIGN_910B); // 512B对齐
    if (sizePerSlice * (localRankSize_ - 1) > sizeLimit) {
        sizePerSlice = RoundUpWithDivisor(sizePerSliceOri, HCCL_MIN_SLICE_ALIGN_ONCHIP);
    }
    if (sizePerSlice * (localRankSize_ - 1) > sizeLimit) {
        sizePerSlice = RoundUpWithDivisor(sizePerSliceOri, unitSize);
    }
    u64 residueSize = totalSize;
    u32 i = 0;
    while (residueSize > 0) {
        u64 sliceSize = sizePerSlice < residueSize ? sizePerSlice : residueSize;
        temp.size = sliceSize;
        temp.offset = totalSize - residueSize;
        i++;
        if (sliceSize <= 0) {
            HCCL_ERROR("[Prepare][SliceData]data_slices_prepare sliceSize[%llu]", sliceSize);
            return HCCL_E_PARA;
        }
        residueSize -= sliceSize;
        dataSlice.push_back(temp);
        if (i != sliceNum) {
            startTemp.size = sizePerSlice;
            startTemp.offset = 0;
            startSlice.push_back(startTemp);
        } else {
            startTemp.size = sizePerSlice;
            startTemp.offset = sizePerSlice;
            startSlice.push_back(startTemp);
        }
    }
    while (i < sliceNum) {
        temp.size = 0;
        temp.offset = totalSize;
        i++;
        dataSlice.push_back(temp);
        startTemp.size = 0;
        startTemp.offset = 0;
        startSlice.push_back(startTemp);
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduce::PrepareAllreduceSliceData()
{
    return PrepareSlice(count_, DataUnitSize(dataType_), localRankSize_, slices_, startOffset);
}

// ring allreduce算法的函数入口
HcclResult AllReduceLocalReduce::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO(
        "AllReduceLocalReduce run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);

    if (links.size() < rankSize) {
        HCCL_ERROR(
            "[AllReduceLocalReduce][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]", rank, links.size(),
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
                ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceLocalReduce][RunAsync]rank[%u] memcpy async failed", rank),
                ret);
        }
        return ret;
    }

    ret = PrepareAllreduceSliceData();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceLocalReduce][RunAsync]rank[%u] count[%llu] failed in PrepareSliceData step", rank, count_),
        ret);

    ret = RunReduceScatter(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllReduceLocalReduce][RunAsync]rank[%u] count[%llu] failed in ReduceScatter step", rank, count_),
        ret);

    ret = RunAllGather(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllReduceLocalReduce][RunAsync]rank[%u] count[%llu] failed in AllGather step", rank, count_), ret);

    HCCL_INFO("AllReduceLocalReduce finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduce::RunReduceScatter(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO(
        "ReduceScatterMeshLocalReduce run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", rank,
        rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    // 数据准备
    u32 unitSize = DataUnitSize(dataType_);
    DeviceMem userMemIn = DeviceMem::create(opInfo_->inputAddr, count_ * unitSize);
    DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), outputMem_.size());

    DeviceMem src;
    DeviceMem dst;

    src = DeviceMem::create(static_cast<char*>(opInfo_->inputAddr) + slices_[rank].offset, slices_[rank].size);
    dst = commMemOut.range(slices_[rank].offset, slices_[rank].size);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));

    DeviceMem emptySrc = userMemIn.range(0, 0);
    DeviceMem emptyDst = commMemOut.range(0, 0);

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = (round + rank) % rankSize;
        Stream& subStream = (round == rankSize - 1) ? stream_ : meshStreams_[round - 1];

        CHK_RET(links[dstRank]->TxAck(subStream));
        CHK_RET(links[dstRank]->RxAck(subStream));
    }
    HCCL_DEBUG("[ReduceScatterMeshLocalReduce] D2DMemcpy start");
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    for (u32 round = 1; round < rankSize; round++) {
        Stream& subStream = (round == rankSize - 1) ? stream_ : meshStreams_[round - 1];
        void* remMemPtr = nullptr;
        u32 dstRank = (rank + round) % rankSize;
        u32 dstSlice = (dstRank + round) % (rankSize - 1);
        if (dstRank == (rankSize - 1)) {
            dstSlice = (dstSlice + rankSize - 1 - 1) % (rankSize - 1);
        }
        if (round == (rankSize - 1)) {
            CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));

            dstSlice = (dstRank == (rankSize - 1)) ? (dstRank - 1) : dstRank;
            dst = DeviceMem::create(
                static_cast<char*>(remMemPtr) + startOffset[dstRank].offset + dstSlice * startOffset[dstRank].size,
                slices_[dstRank].size);
            src = userMemIn.range(slices_[dstRank].offset, slices_[dstRank].size);

            HCCL_INFO(
                "AllReducelocalreduce reduce dst offset1 %llu offset2 %llu size %llu, rank %u, dstrank %u",
                startOffset[dstRank].offset, dstSlice * startOffset[dstRank].size, slices_[dstRank].size, rank,
                dstRank);

            HCCL_INFO(
                "AllReducelocalreduce reduce src offset %llu size %llu, rank %u, dstrank %u", slices_[dstRank].offset,
                slices_[dstRank].size, rank, dstRank);

            CHK_RET(HcclReduceAsync(
                dispatcher_, static_cast<void*>(src.ptr()), slices_[dstRank].size / unitSize, dataType_, reductionOp_,
                subStream, static_cast<void*>(dst.ptr()), links[dstRank]->GetRemoteRank(),
                links[dstRank]->GetLinkType(), INLINE_REDUCE_BIT));
        } else {
            CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));

            dst = DeviceMem::create(
                static_cast<char*>(remMemPtr) + startOffset[dstRank].offset + dstSlice * startOffset[dstRank].size,
                slices_[dstRank].size);
            src = userMemIn.range(slices_[dstRank].offset, slices_[dstRank].size);

            HCCL_INFO(
                "AllReducelocalreduce memcpy dst offset1 %llu offset2 %llu size %llu, rank %u, dstrank %u",
                startOffset[dstRank].offset, dstSlice * startOffset[dstRank].size, slices_[dstRank].size, rank,
                dstRank);

            HCCL_INFO(
                "AllReducelocalreduce memcpy src offset %llu size %llu, rank %u, dstrank %u", slices_[dstRank].offset,
                slices_[dstRank].size, rank, dstRank);

            CHK_RET(HcclD2DMemcpyAsync(
                dispatcher_, dst, src, subStream, links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType()));
        }

        CHK_RET(links[dstRank]->TxDataSignal(subStream));
        CHK_RET(links[dstRank]->RxDataSignal(subStream));
    }

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    HcclResult ret = HCCL_SUCCESS;
    ret = RunLocalReduce(rank, rankSize);

    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceLocalReduce]rank[%u] ReduceScatter failed", rank), ret);
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduce::RunLocalReduce(u32 rank, u32 rankSize)
{
    DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), outputMem_.size());
    u32 power = static_cast<u32>(log2(rankSize - 1));
    u32 rankPower = static_cast<u32>(pow(2, power));
    u32 unitSize = SIZE_TABLE[dataType_];
    u64 align = startOffset[rank].size;
    u64 totalSize = slices_[rank].size;
    DeviceMem src;
    DeviceMem dst;
    for (u32 i = 0u; i < rankSize - rankPower - 1; ++i) {
        u64 size = totalSize;
        if (rank < rankPower) {
            src = commMemOut.range(startOffset[rank].offset + (rankPower + i) * align, size);
            dst = commMemOut.range(startOffset[rank].offset + i * align, size);
            HCCL_INFO(
                "[RunLocalReduce]LocalReduce rank[%u] src[%llu], dst[%llu] size[%llu]", rank,
                startOffset[rank].offset + (rankPower + i) * align, startOffset[rank].offset + i * align, size);
        } else {
            dst = commMemOut.range(startOffset[rank].offset + (rankPower + i) * align, size);
            src = commMemOut.range(startOffset[rank].offset + i * align, size);
            HCCL_INFO(
                "[RunLocalReduce]LocalReduce rank[%u] src[%llu], dst[%llu] size[%llu]", rank,
                startOffset[rank].offset + i * align, startOffset[rank].offset + (rankPower + i) * align, size);
        }
        CHK_RET(HcclReduceAsync(
            dispatcher_, static_cast<void*>(src.ptr()), size / unitSize, dataType_, reductionOp_, stream_,
            static_cast<void*>(dst.ptr()), INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP, INLINE_REDUCE_BIT));
    }
    u32 center = rank < rankPower ? rank : (rank - rankPower + 1);
    center = std::min(center, rankPower - 1);
    u64 offset = rank < rankPower ? 0 : ((rankSize - rankPower - 1) * align);
    offset += startOffset[rank].offset;
    for (u32 round = 0; round < power; round++) {
        u32 slices_num = static_cast<u32>(rankPower / pow(2, round + 1));
        u64 size = totalSize;
        if (center < slices_num) {
            for (auto i = 0u; i < slices_num; ++i) {
                src = commMemOut.range(offset + (slices_num + i) * align, size);
                dst = commMemOut.range(offset + i * align, size);
                HCCL_INFO(
                    "[RunLocalReduce]LocalReduce rank[%u] src[%llu], dst[%llu] size[%llu]", rank,
                    offset + (slices_num + i) * align, offset + i * align, size);
                CHK_RET(HcclReduceAsync(
                    dispatcher_, static_cast<void*>(src.ptr()), src.size() / unitSize, dataType_, reductionOp_, stream_,
                    static_cast<void*>(dst.ptr()), INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP, INLINE_REDUCE_BIT));
            }
        } else {
            for (auto i = 0u; i < slices_num; ++i) {
                dst = commMemOut.range(offset + (slices_num + i) * align, size);
                src = commMemOut.range(offset + i * align, size);
                HCCL_INFO(
                    "[RunLocalReduce]LocalReduce rank[%u] src[%llu], dst[%llu] size[%llu]", rank, offset + i * align,
                    offset + (slices_num + i) * align, size);
                CHK_RET(HcclReduceAsync(
                    dispatcher_, static_cast<void*>(src.ptr()), src.size() / unitSize, dataType_, reductionOp_, stream_,
                    static_cast<void*>(dst.ptr()), INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP, INLINE_REDUCE_BIT));
            }
            offset = offset + slices_num * align;
            center -= slices_num;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceLocalReduce::RunAllGather(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO(
        "AllGatherMesh run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr(), count_);
    u32 unitSize = DataUnitSize(dataType_);
    DeviceMem userMemOut = DeviceMem::create(opInfo_->outputAddr, count_ * unitSize);
    DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), outputMem_.size());

    DeviceMem src;
    DeviceMem dst;

    DeviceMem emptySrc = commMemOut.range(0, 0);
    DeviceMem emptyDst = userMemOut.range(0, 0);

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = BackwardRank(rank, rankSize, round);
        Stream& subStream = (round == rankSize - 1) ? stream_ : meshStreams_[round - 1];
        CHK_RET(links[dstRank]->TxAck(subStream));
        CHK_RET(links[dstRank]->RxAck(subStream));
    }

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    if (userMemOut.ptr() != commMemOut.ptr()) {
        src = commMemOut.range(slices_[rank].offset, slices_[rank].size);
        dst = userMemOut.range(slices_[rank].offset, slices_[rank].size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, meshStreams_[meshStreams_.size() - 1]));
    }

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = BackwardRank(rank, rankSize, round);
        Stream& subStream = (round == rankSize - 1) ? stream_ : meshStreams_[round - 1];
        void* remMemPtr = nullptr;
        CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));

        src = DeviceMem::create(static_cast<char*>(remMemPtr) + dstRank * slices_[0].size, slices_[dstRank].size);
        dst = userMemOut.range(slices_[dstRank].offset, slices_[dstRank].size);

        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dst, src, subStream, links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType()));
        CHK_RET(links[dstRank]->TxDataSignal(subStream));
        CHK_RET(links[dstRank]->RxDataSignal(subStream));
    }

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    HCCL_INFO("AllGatherMesh finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_LOCAL_REDUCE, AllReduceLocalReduce);
} // namespace hccl
