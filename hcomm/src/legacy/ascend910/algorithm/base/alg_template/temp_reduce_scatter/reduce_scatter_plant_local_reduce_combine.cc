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
#include "reduce_scatter_plant_local_reduce_combine.h"

namespace hccl {
constexpr u32 DEVICE_EIGHT = 8;
constexpr u32 FACTOR_NUM_TWO = 2;
ReduceScatterPlantLocalReduceCombine::ReduceScatterPlantLocalReduceCombine(const HcclDispatcher dispatcher)
    : AlgTemplateBase(dispatcher)
{}

ReduceScatterPlantLocalReduceCombine::~ReduceScatterPlantLocalReduceCombine() {}

HcclResult ReduceScatterPlantLocalReduceCombine::Prepare(
    DeviceMem& cclInMem, DeviceMem& outputMem, const Stream& stream, std::vector<Stream>& subStreams,
    std::vector<std::shared_ptr<LocalNotify>>& meshSignal, std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux,
    MemBlockInfo& memBlockInfo, const HcclReduceOp reductionOp, const HcclDataType dataType, bool isUseCclIn,
    bool isLevel0LastRank, bool isNeedSpaceBorrow)
{
    inputMem_ = cclInMem;   // 空拷贝 & 存放最后一块数据（Allreduce非整除场景）
    outputMem_ = outputMem; // 单算子CclOut 图模式Scratch/UserOut，LocalReduce使用
    stream_ = stream;
    subStreams_ = subStreams;
    meshSignalPtr_ = &meshSignal;
    meshSignalAuxPtr_ = &meshSignalAux;
    memBlockInfo_ = std::move(memBlockInfo);
    reductionOp_ = reductionOp;
    dataType_ = dataType;
    isUseCclIn_ = isUseCclIn; // 本卡在level0执行完毕后,需要告知level1数据是否存放被存放在CCLin的标识(rank维度)
    isLevel0LastRank_ = isLevel0LastRank;
    isNeedSpaceBorrow_ = isNeedSpaceBorrow; // 是否需要借用CCLIN空间完成LocalReuce\alltoall(算子维度)
    return HCCL_SUCCESS;
}

HcclResult
ReduceScatterPlantLocalReduceCombine::MainRecordSub(Stream& mainStream, u32 firstSubStreamIndex, u32 totalTask)
{
    for (u32 streamIndex = firstSubStreamIndex; streamIndex < totalTask; streamIndex++) {
        CHK_RET(LocalNotify::Post(mainStream, dispatcher_, (*meshSignalAuxPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPlantLocalReduceCombine::SubWaitMain(u32 firstSubStreamIndex, u32 totalTask)
{
    for (u32 streamIndex = firstSubStreamIndex; streamIndex < totalTask; streamIndex++) {
        CHK_RET(LocalNotify::Wait(
            subStreams_[streamIndex], dispatcher_, (*meshSignalAuxPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPlantLocalReduceCombine::MainWaitSub(Stream& mainStream, u32 firstSubStreamIndex, u32 totalTask)
{
    for (u32 streamIndex = firstSubStreamIndex; streamIndex < totalTask; streamIndex++) {
        CHK_RET(LocalNotify::Wait(mainStream, dispatcher_, (*meshSignalPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPlantLocalReduceCombine::SubRecordMain(u32 firstSubStreamIndex, u32 totalTask)
{
    for (u32 streamIndex = firstSubStreamIndex; streamIndex < totalTask; streamIndex++) {
        CHK_RET(LocalNotify::Post(
            subStreams_[streamIndex], dispatcher_, (*meshSignalPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

u32 ReduceScatterPlantLocalReduceCombine::CalcOutputIndex(const u32 round) const
{
    return (round + localRank_) % rankSize_;
}

bool ReduceScatterPlantLocalReduceCombine::isLastRank(const u32 rankId) const { return rankId == rankSize_ - 1; }

bool ReduceScatterPlantLocalReduceCombine::isLastBlockData(const u32 outputIndex)
{
    return outputIndex == rankSize_ - 1;
}

HcclResult
ReduceScatterPlantLocalReduceCombine::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    HCCL_INFO(
        "ReduceScatterPlantLocalReduceCombine run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p].", rank, rankSize,
        inputMem_.ptr(), outputMem_.ptr());
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    CHK_PRT_RET(
        links.size() < rankSize,
        HCCL_ERROR(
            "[ReduceScatterPlantLocalReduceCombine][RunAsync]rank[%u] "
            "linksize[%llu] is less than rankSize[%u]",
            rank, links.size(), rankSize),
        HCCL_E_INTERNAL);

    rankSize_ = rankSize;
    localRank_ = rank;

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    CHK_RET(RunAlltoAll(links));
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));

    // 执行LocalReduce
    HcclResult ret = RunLocalReduce();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]localRank[%u] LocalReduce failed", __func__, localRank_), ret);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPlantLocalReduceCombine::LocalCopy()
{
    u64 sliceSize = memBlockInfo_.size[localRank_];
    if (sliceSize == 0) {
        return HCCL_SUCCESS;
    }

    DeviceMem src;
    if (isNeedSpaceBorrow_ && isUseCclIn_) {
        src = inputMem_.range(memBlockInfo_.userInputOffsets[localRank_], sliceSize);
    } else {
        src = outputMem_.range(memBlockInfo_.inputOffsets[localRank_], sliceSize);
    }

    DeviceMem dst;
    u32 outputIndex = CalcOutputIndex(localRank_);
    if (isNeedSpaceBorrow_ && isLevel0LastRank_ && isLastBlockData(outputIndex) && !isLastRank(localRank_)) {
        dst = inputMem_.range(memBlockInfo_.userInputOffsets[localRank_], sliceSize);
    } else {
        dst = outputMem_.range(memBlockInfo_.outputOffsets[outputIndex], sliceSize);
    }

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    return HCCL_SUCCESS;
}

HcclResult
ReduceScatterPlantLocalReduceCombine::RunAlltoAllRDMA(u32 round, u64 sliceSize, const std::vector<LINK>& links)
{
    u64 srcOffset = memBlockInfo_.inputOffsets[round];
    void* srcPtr = static_cast<u8*>(outputMem_.ptr()) + srcOffset;
    if (isNeedSpaceBorrow_ && isUseCclIn_) {
        srcOffset = sliceSize == 0 ? 0 : memBlockInfo_.userInputOffsets[round];
        srcPtr = static_cast<u8*>(inputMem_.ptr()) + srcOffset;
    }

    u32 outputIndex = CalcOutputIndex(round);
    u64 dstOffset = memBlockInfo_.outputOffsets[outputIndex];
    if (isNeedSpaceBorrow_ && isLevel0LastRank_ && !isLastRank(round) && isLastBlockData(outputIndex)) {
        // 只有level0最后一组的最后一块数据需要放到对方的input上（且非全局最后一张卡）
        dstOffset = sliceSize == 0 ? 0 : memBlockInfo_.userInputOffsets[round];
        CHK_RET(links[round]->TxAsync(UserMemType::INPUT_MEM, dstOffset, srcPtr, sliceSize, stream_));
    } else {
        CHK_RET(links[round]->TxAsync(UserMemType::OUTPUT_MEM, dstOffset, srcPtr, sliceSize, stream_));
    }

    u32 localOutputIndex = CalcOutputIndex(localRank_);
    u64 localDstOffset = memBlockInfo_.outputOffsets[localRank_];
    void* dstPtr = static_cast<u8*>(outputMem_.ptr()) + localDstOffset;
    if (isNeedSpaceBorrow_ && isLevel0LastRank_ && isLastBlockData(localOutputIndex) && !isLastRank(localRank_)) {
        localDstOffset = memBlockInfo_.userInputOffsets[round];
        dstPtr = static_cast<u8*>(inputMem_.ptr()) + localDstOffset;
    }

    u64 remoteSrcOffset = memBlockInfo_.inputOffsets[round];
    CHK_RET(links[round]->RxAsync(UserMemType::OUTPUT_MEM, remoteSrcOffset, dstPtr, sliceSize, stream_));

    CHK_RET(links[round]->PostFinAck(stream_));
    CHK_RET(links[round]->WaitFinAck(stream_));
    return HCCL_SUCCESS;
}

HcclResult
ReduceScatterPlantLocalReduceCombine::RunAlltoAllSDMA(u32 round, u64 sliceSize, const std::vector<LINK>& links)
{
    if (sliceSize != 0) {
        DeviceMem src;
        if (isNeedSpaceBorrow_ && isUseCclIn_) {
            src = inputMem_.range(memBlockInfo_.userInputOffsets[round], sliceSize);
        } else {
            src = outputMem_.range(memBlockInfo_.inputOffsets[round], sliceSize);
        }

        u32 outputIndex = CalcOutputIndex(round);
        u64 dstOffset = memBlockInfo_.outputOffsets[outputIndex];
        void* remMemPtr = nullptr;
        if (isNeedSpaceBorrow_ && isLevel0LastRank_ && !isLastRank(round) && isLastBlockData(outputIndex)) {
            CHK_RET(links[round]->GetRemoteMem(UserMemType::INPUT_MEM, &remMemPtr));
            dstOffset = memBlockInfo_.userInputOffsets[round];
        } else {
            CHK_RET(links[round]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
        }
        DeviceMem dst = DeviceMem::create(static_cast<u8*>(remMemPtr) + dstOffset, sliceSize);

        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dst, src, stream_, links[round]->GetRemoteRank(), links[round]->GetLinkType()));
    }

    CHK_RET(links[round]->TxDataSignal(stream_));
    CHK_RET(links[round]->RxDataSignal(stream_));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPlantLocalReduceCombine::RunAlltoAll(const std::vector<LINK>& links)
{
    CHK_RET(LocalCopy());
    for (u32 round = 0; round < rankSize_; round++) {
        if (round == localRank_) {
            continue;
        }
        CHK_SMART_PTR_NULL(links[round]);
        CHK_RET(links[round]->TxAck(stream_));
        CHK_RET(links[round]->RxAck(stream_));

        u64 sliceSize = memBlockInfo_.size[round];
        if (links[round]->GetLinkType() == LinkType::LINK_ROCE) {
            CHK_RET(RunAlltoAllRDMA(round, sliceSize, links));
        } else {
            CHK_RET(RunAlltoAllSDMA(round, sliceSize, links));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPlantLocalReduceCombine::RunLocalReduce()
{
    u32 reduceStep = static_cast<u32>(std::ceil(log2(rankSize_)));
    u64 sliceSize = memBlockInfo_.size[localRank_];
    u32 dataUnitSize = DataUnitSize(dataType_);
    if (dataUnitSize == 0) {
        HCCL_ERROR(
            "[ReduceScatterPlantLocalReduceCombine][RunLocalReduce]data type[%s] out of range[%d, %d]",
            GetDataTypeEnumStr(dataType_).c_str(), HCCL_DATA_TYPE_INT8, static_cast<int>(HCCL_DATA_TYPE_RESERVED) - 1);
        return HCCL_E_INTERNAL;
    }
    u64 count = sliceSize / dataUnitSize;

    for (u32 round = 0; round < reduceStep; round++) {
        u32 tailIndex = std::min(rankSize_, static_cast<u32>(1 << static_cast<int>(reduceStep - round))) - 1;
        u32 headIndex = static_cast<u32>(1 << static_cast<int>((reduceStep - round - 1)));
        u32 reduceSubStreamNum = std::min(tailIndex - headIndex, DEVICE_EIGHT / FACTOR_NUM_TWO - 1);

        // LR主流通知从流可以开始接受数据
        for (u32 offset = 0; offset < reduceSubStreamNum; offset++) {
            u32 streamId = offset;
            // 只有reduce任务 > 1时才需要主从流同步: LR主流通知从流, 从流Wait LR主流
            CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAuxPtr_)[streamId], profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(
                subStreams_[streamId], dispatcher_, (*meshSignalAuxPtr_)[streamId], profilerInput_.stage));
        }

        // LocalReduce操作
        for (u32 offset = 0; offset <= tailIndex - headIndex; offset++) {
            u32 inputIndex = CalcOutputIndex(headIndex + offset); // reduce的源数据offset
            u32 outputIndex = CalcOutputIndex(offset);            // reduce的目标offset

            u32 streamOffset = offset % (reduceSubStreamNum + 1);
            Stream& subStream = streamOffset == 0 ? stream_ : subStreams_[streamOffset - 1];

            if (sliceSize == 0) {
                continue;
            }

            void* srcPtr;
            void* dstPtr;
            if (isNeedSpaceBorrow_ && isLevel0LastRank_ && !isLastRank(localRank_) && isLastBlockData(inputIndex)) {
                srcPtr = static_cast<u8*>(inputMem_.ptr()) + memBlockInfo_.userInputOffsets[localRank_];
            } else {
                srcPtr = static_cast<u8*>(outputMem_.ptr()) + memBlockInfo_.outputOffsets[inputIndex];
            }

            if (isNeedSpaceBorrow_ && isLevel0LastRank_ && !isLastRank(localRank_) && isLastBlockData(outputIndex)) {
                dstPtr = static_cast<u8*>(inputMem_.ptr()) + memBlockInfo_.userInputOffsets[localRank_];
            } else {
                dstPtr = static_cast<u8*>(outputMem_.ptr()) + memBlockInfo_.outputOffsets[outputIndex];
            }

            CHK_RET(HcclReduceAsync(
                dispatcher_, srcPtr, count, dataType_, reductionOp_, subStream, dstPtr, INVALID_VALUE_RANKID,
                LinkType::LINK_ONCHIP, INLINE_REDUCE_BIT));
        }

        // 从流通知LR主流可以开始下一轮
        for (u32 offset = 0; offset < reduceSubStreamNum; offset++) {
            u32 streamId = offset;
            // 只有reduce任务 > 1时才需要主从流同步: LR主流通知从流, 从流Wait LR主流
            CHK_RET(LocalNotify::Post(
                subStreams_[streamId], dispatcher_, (*meshSignalPtr_)[streamId], profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignalPtr_)[streamId], profilerInput_.stage));
        }

        CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    }

    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(
    TemplateType::TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE_COMBINE, ReduceScatterPlantLocalReduceCombine);
} // namespace hccl
