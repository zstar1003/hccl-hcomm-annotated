/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_multi_deter_pipeline.h"
#include "alg_template_register.h"

namespace hccl {
ReduceScatterMultiDeterPipeline::ReduceScatterMultiDeterPipeline(const HcclDispatcher dispatcher)
    : MultiDeterPipeline(dispatcher)
{}

ReduceScatterMultiDeterPipeline::~ReduceScatterMultiDeterPipeline() {}

HcclResult ReduceScatterMultiDeterPipeline::GetRemoteCclbufferDeviceMem(
    u32 inputSliceIndex, LINK link, u32 outputSliceIndex, DeviceMem& remoteMem)
{
    u64 inputSliceOffset = memSliceSize_ * inputSliceIndex + offset_;
    u64 eachOffset = eachRankCclbufferSize_;
    u64 outputSliceOffset = eachOffset * outputSliceIndex;
    u64 outputInSliceOffset = (HCCL_MIN_SLICE_ALIGN_910B + (inputSliceOffset % HCCL_MIN_SLICE_ALIGN_910B)
                               - (outputSliceOffset % HCCL_MIN_SLICE_ALIGN_910B))
                              % HCCL_MIN_SLICE_ALIGN_910B;
    void* remoteMemPtr = nullptr;
    CHK_RET(link->GetRemoteMem(UserMemType::OUTPUT_MEM, &remoteMemPtr)); // 图模式不一定是input，统一output
    u8* beginAddrU8 = static_cast<u8*>(remoteMemPtr);
    u8* intraSrcAddr = beginAddrU8 + outputSliceOffset + outputInSliceOffset;
    remoteMem = DeviceMem::create(intraSrcAddr, curSize_);
    if (remoteMem.ptr() == nullptr) {
        HCCL_ERROR(
            "[%s] outputSliceOffset + outputInSliceOffset + curSize_ = [%llu] > cclBufferSize[%llu]", __func__,
            outputSliceOffset + outputInSliceOffset + curSize_, cclBuffer_.size());
        return HCCL_E_MEMORY;
    }
    HCCL_DEBUG(
        "[%s] rank[%u], beginAddr[%p], outputSliceOffset[%llu](outputSliceIndex * eachOffset[%llu]), "
        "outputInSliceOffset[%llu], curSize[%llu], totalBufferSize[%llu]",
        __func__, outputSliceIndex, remoteMem.ptr(), outputSliceOffset, eachOffset, outputInSliceOffset, curSize_,
        cclBuffer_.size());
    return HCCL_SUCCESS;
}

// RDMA 发送时顶格收，所以不需要128K对齐，故sliceOffset为0
// SDMA 发送后 取本地 sliceOffset = slices_[rankIdInAllRanks].offset偏移处存放地址
HcclResult
ReduceScatterMultiDeterPipeline::GetLocalCclbufferDeviceMem(u32 rankIdInAllRanks, DeviceMem& localMem, u64 sliceOffset)
{
    u64 eachOffset = eachRankCclbufferSize_; // 当前轮有效数据大小 + HCCL_MIN_SLICE_ALIGN_910B作为偏移
    u64 rdmaOffset = eachOffset * rankIdInAllRanks;
    u64 sdmaOffset = sliceOffset;
    u64 offset = sliceOffset == 0 ? rdmaOffset : sdmaOffset;
    localMem = cclBuffer_.range(offset, curSize_);
    if (localMem.ptr() == nullptr) {
        HCCL_ERROR(
            "[%s] get localMem failed, rdmaOffset + curSize_"
            "= [%llu] or sdmaOffset + curSize = [%llu] > cclBufferSize[%llu]",
            __func__, rdmaOffset + curSize_, sdmaOffset + curSize_, cclBuffer_.size());
        return HCCL_E_MEMORY;
    }
    HCCL_DEBUG(
        "[%s] rank[%u], beginAddr[%p], offset[%llu](rdmaOffset[%u] or sdmaOffset[%llu]), "
        "curSize[%llu], totalBufferSize[%llu]",
        __func__, rankIdInAllRanks, localMem.ptr(), sliceOffset, rdmaOffset, sdmaOffset, curSize_, cclBuffer_.size());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMultiDeterPipeline::GetLocalUserInDeviceMem(u32 rankIdInAllRanks, DeviceMem& localMem)
{
    u8* beginAddrU8 = static_cast<u8*>(usrInMemPtr_);
    u64 eachOffset = memSliceSize_;
    u8* intraSrcAddr = beginAddrU8 + (rankIdInAllRanks * eachOffset); // 不用 + offset_，因为usrInMem_已经加过了
    localMem = DeviceMem::create(intraSrcAddr, curSize_);
    if (localMem.ptr() == nullptr) {
        HCCL_ERROR(
            "[%s] get localMem failed, rankIdInAllRanks * eachOffset + curSize_ = [%u] is too big", __func__,
            rankIdInAllRanks * eachOffset + curSize_, cclBuffer_.size());
        return HCCL_E_MEMORY;
    }
    HCCL_DEBUG(
        "[%s] ranks[%u], offset[%llu](rankIdInAllRanks * eachOffset[%llu])"
        "intraSrcAddr[%p], memSliceSize[%llu], usrInMemPtr[%p]",
        __func__, rankIdInAllRanks, rankIdInAllRanks * eachOffset, eachOffset, intraSrcAddr, memSliceSize_,
        usrInMemPtr_);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMultiDeterPipeline::RunLocalCopy()
{
    // 每张卡将自己的input的第user rank块数据搬到output，例如0A 1B 2C
    DeviceMem userIn;
    CHK_RET(GetLocalUserInDeviceMem(userRank_, userIn));
    DeviceMem userOut = DeviceMem::create(usrOutMemPtr_, curSize_);
    // 使用主流搬迁卡内数据
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, userOut, userIn, mainStream_));
    HCCL_DEBUG(
        "[%s] intra-card copy data from [%u] to usrOutMem[%p] size[%llu]", __func__, userRank_, usrOutMemPtr_,
        curSize_);
    return HCCL_SUCCESS;
}

// 机内alltoall full mesh收集数据, #step表示pairwise的第step步
HcclResult ReduceScatterMultiDeterPipeline::RunIntraAlltoallPreSync(u32 step)
{
    HCCL_DEBUG("[%s] intra-server alltoall begin, step[%u]", __func__, step);
    HCCL_DEBUG(
        "[%s] intra-server SDMA send begin, [serverId, intraRankId] = [%u, %u]", __func__, serverId_, intraRankId_);
    for (u32 i = 0; i < intraRankSize_ - 1; ++i) {
        HCCL_DEBUG(
            "[%s] intra-server SDMA send begin, userRank[%u] step[%u] pro[%u/%u]", __func__, userRank_, i, i + 1,
            intraRankSize_ - 1);
        // 从机内rankId为recvIntraRankId收集数据，也发给机内rankId为sendIntraRankId数据
        u32 recvIntraRankId = GetPreIntraRankIdByStep(i + 1);
        u32 sendIntraRankId = GetNextIntraRankIdByStep(i + 1);
        LINK recvIntraLink = intraLinks_[recvIntraRankId];
        LINK sendIntraLink = intraLinks_[sendIntraRankId];
        CHK_RET(sendIntraLink->TxAck(subStreams_[i]));
        CHK_RET(sendIntraLink->RxAck(subStreams_[i]));
    }
    // 增加主从流同步，目的是让SDMA同时进行
    CHK_RET(MainWaitSub(all2allStreamBegin_, all2allStreamBegin_ + all2allStreamSize_));
    CHK_RET(SubRecordMain(all2allStreamBegin_, all2allStreamBegin_ + all2allStreamSize_));
    CHK_RET(MainRecordSub(all2allStreamBegin_, all2allStreamBegin_ + all2allStreamSize_));
    CHK_RET(SubWaitMain(all2allStreamBegin_, all2allStreamBegin_ + all2allStreamSize_));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMultiDeterPipeline::BatchPostNotifyForStreams(
    const std::vector<std::vector<std::pair<u32, u32>>>& streamTasks, bool isStartPhase, bool useMainStream)
{
    if (useMainStream) {
        HCCL_DEBUG("[%s] use mainStream, skip notify wait", __func__);
        return HCCL_SUCCESS;
    }
    for (u32 s = 0; s < MAX_REDUCE_STREAM_NUM; s++) {
        if (streamTasks[s].empty())
            continue; // 无任务的流跳过
        u32 streamIdx = reduceStreamBegin_ + s;
        if (reduceMainStreamIdx_ == streamIdx) {
            continue;
        }
        if (isStartPhase) {
            // 启动阶段：主流→子流 通知（Post主流，Wait子流）
            CHK_RET(LocalNotify::Post(
                subStreams_[reduceMainStreamIdx_], dispatcher_, streamNotifySub_[streamIdx], profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(
                subStreams_[streamIdx], dispatcher_, streamNotifySub_[streamIdx], profilerInput_.stage));
            HCCL_DEBUG("[%s] stream[%u] start phase notify done", __func__, streamIdx);
        } else {
            // 同步阶段：子流→主流 通知（Post子流，Wait主流）
            CHK_RET(LocalNotify::Post(
                subStreams_[streamIdx], dispatcher_, streamNotifyMain_[streamIdx], profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(
                subStreams_[reduceMainStreamIdx_], dispatcher_, streamNotifyMain_[streamIdx], profilerInput_.stage));
            HCCL_DEBUG("[%s] stream[%u] sync phase notify done", __func__, streamIdx);
        }
    }
    return HCCL_SUCCESS;
}

// 机内localreduce首先按序收集所有内存块，接着二分归并reduce，最多使用4条流并行
HcclResult ReduceScatterMultiDeterPipeline::RunIntraLocalReduce(u32 step)
{
    HCCL_DEBUG("[%s] inter-server local reduce begin, step[%u]", __func__, step);
    u32 recvServerId = GetPreServerIdByStep(step);  // 从上一个收
    u32 sendServerId = GetNextServerIdByStep(step); // 发给发下一个
    std::vector<DeviceMem> reduceMem;
    std::vector<bool> isReduceBlock;
    u32 retIndex = 0;
    isReduceBlock.resize(intraRankSize_);
    // 机内，最后rank的规约结果放在倒数第2块，其他放在倒数第1块
    if (intraRankId_ == intraRankSize_ - 1) {
        retIndex = intraRankSize_ - SECOND_TO_LAST;
    } else {
        retIndex = intraRankSize_ - 1;
    }
    // 机内第0步，规约到usrOut，即rank所在机间的intraRankId_处
    if (step == 0) {
        retIndex = intraRankId_;
    }
    HCCL_DEBUG(
        "[%s] intra-server local reduce, retIndex[%u], intraRankSize[%u] intraRankId[%u]", __func__, retIndex,
        intraRankSize_, intraRankId_);
    reduceMem.resize(intraRankSize_);
    u32 userInIdx = GetRankIdx(sendServerId, intraRankId_);
    // 最后一块留给allreduce
    u32 idx = 0;
    const u32 serverId = (step == 0) ? serverId_ : recvServerId;
    for (u32 i = 0; i < intraRankSize_; ++i) {
        if (i == intraRankId_) {
            if (step == 0) {
                // step=0：归到usrOut
                isReduceBlock[i] = true;
                DeviceMem usrOutInraMem = DeviceMem::create(usrOutMemPtr_, curSize_);
                reduceMem[i] = std::move(usrOutInraMem);
                HCCL_DEBUG("[%s] inter-server local reduce, NO.%u reduceMem stores userOut", __func__, i);
            } else {
                // step≠0：填充usrIn
                isReduceBlock[i] = false;
                DeviceMem usrInIntraMem;
                CHK_RET(GetLocalUserInDeviceMem(userInIdx, usrInIntraMem));
                reduceMem[i] = std::move(usrInIntraMem);
                HCCL_DEBUG(
                    "[%s] inter-server local reduce, NO.%u reduceMem stores userInIdx[%u],", __func__, i, userInIdx);
            }
            continue;
        }
        // 其他情况：统一处理CCLBuffer
        isReduceBlock[i] = true;
        const u32 outCCLbufferIdx = GetRankIdx(serverId, idx);
        DeviceMem cclbufferIntraMem;
        CHK_RET(GetLocalCclbufferDeviceMem(outCCLbufferIdx, cclbufferIntraMem, slices_[outCCLbufferIdx].offset));
        reduceMem[i] = std::move(cclbufferIntraMem);
        HCCL_DEBUG(
            "[%s] inter-server local reduce, NO.%u reduceMem stores outCCLbufferIdx[%u]", __func__, i, outCCLbufferIdx);
        idx++;
    }
    CHK_RET(LocalReduce(reduceMem, isReduceBlock, retIndex, false));
    HCCL_INFO("[%s] intra-server step[%u] run local reduce success", __func__, step);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMultiDeterPipeline::RunInterSend(u32 step)
{
    HCCL_DEBUG("[%s] inter-server RDMA write begin, step[%u]", __func__, step);
    // 使用主流进行rdma
    u32 recvServerId = GetPreServerIdByStep(step);  // 从上一个收
    u32 sendServerId = GetNextServerIdByStep(step); // 发给下一个
    LINK recvInterLink = serverLinks_[recvServerId];
    LINK sendInterLink = serverLinks_[sendServerId];
    // 跨机 输出的位置是buffer的第(Rn+Sn-n)%Sn组分块的第1块
    // 跨机 输入是input的第(R1+1)%S1组数据（往后1）的倒数第2块
    u32 reduceRetIndex = intraRankSize_ - SECOND_TO_LAST;
    u32 recvCCLbufferIdx = GetRankIdx(recvServerId, 0);              // 本端rank0：收的位置
    u32 sendCCLbufferIdx = GetRankIdx(recvServerId, reduceRetIndex); // 本端rank0：发的位置
    // 跨机写对端
    u32 sendRankId = GetRankIdx(sendServerId, intraRankId_); // 发送到sendRankId
    u32 recvRankId = GetRankIdx(recvServerId, intraRankId_); // 从recvRankId接收
    u32 remoteRecvCCLbufferIdx = GetRankIdx(serverId_, 0);   // 接收端：收的位置
    // 2机又从serverId_收也从serverId_发
    u32 remoteSendCCLbufferIdx = serverSize_ == MIN_SERVER_NUM ?
                                     GetRankIdx(serverId_, reduceRetIndex) :
                                     GetRankIdx(sendServerId, reduceRetIndex); // 发送端：发的位置
    DeviceMem recvMem;
    DeviceMem sendMem;
    CHK_RET(GetLocalCclbufferDeviceMem(recvCCLbufferIdx, recvMem, eachRankCclbufferSize_ * recvCCLbufferIdx));
    CHK_RET(GetLocalCclbufferDeviceMem(sendCCLbufferIdx, sendMem, slices_[sendCCLbufferIdx].offset));
    HCCL_DEBUG(
        "[%s] inter-server RDMA write begin, cclbufferIdx: [%u] send to [%u], [%u] recv from [%u]", __func__,
        sendCCLbufferIdx, remoteRecvCCLbufferIdx, recvCCLbufferIdx, remoteSendCCLbufferIdx);
    HCCL_DEBUG(
        "[%s] inter-server RDMA write begin, rankId: [%u] send to [%u], [%u] recv from [%u]", __func__, userRank_,
        sendRankId, userRank_, recvRankId);
    // A + X 单机16卡为SDMA读语义
    if (recvInterLink->IsSpInlineReduce() && sendInterLink->IsSpInlineReduce()) {
        CHK_RET(sendInterLink->TxAck(mainStream_));
        CHK_RET(recvInterLink->RxAck(mainStream_));
        DeviceMem dstMem = std::move(recvMem);
        DeviceMem srcMem;
        void* remoteMemPtr = nullptr;
        CHK_RET(recvInterLink->GetRemoteMem(UserMemType::OUTPUT_MEM, &remoteMemPtr));
        u8* beginAddrU8 = static_cast<u8*>(remoteMemPtr);
        u8* intraSrcAddr = beginAddrU8 + slices_[remoteSendCCLbufferIdx].offset;
        srcMem = DeviceMem::create(intraSrcAddr, curSize_);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, mainStream_, recvRankId, recvInterLink->GetLinkType()));
        CHK_RET(sendInterLink->TxDataSignal(mainStream_));
        CHK_RET(recvInterLink->RxDataSignal(mainStream_));
    } else {
        CHK_RET(recvInterLink->TxAck(mainStream_));
        CHK_RET(sendInterLink->RxAck(mainStream_));

        CHK_RET(sendInterLink->TxAsync(
            UserMemType::OUTPUT_MEM, remoteRecvCCLbufferIdx * eachRankCclbufferSize_, sendMem.ptr(), curSize_,
            mainStream_));
        CHK_RET(recvInterLink->RxAsync(
            UserMemType::OUTPUT_MEM, slices_[remoteSendCCLbufferIdx].offset, recvMem.ptr(), curSize_, mainStream_));

        CHK_RET(recvInterLink->PostFinAck(mainStream_));
        CHK_RET(sendInterLink->WaitFinAck(mainStream_));
    }
    HCCL_INFO("[%s] inter-server step[%u] run RDMA send success", __func__, step);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMultiDeterPipeline::RunFinalReduce()
{
    // 主从流同步
    HCCL_DEBUG("[%s] intra-server final reduce begin", __func__);
    std::vector<DeviceMem> reduceMem;
    std::vector<bool> isReduceBlock;
    u32 retIndex = serverId_;
    DeviceMem usrOutInraMem = DeviceMem::create(usrOutMemPtr_, curSize_);
    isReduceBlock.resize(serverSize_);
    reduceMem.resize(serverSize_);

    HCCL_DEBUG("[%s] intra-server retIndex[%u], interRankSize[%u]", __func__, retIndex, serverSize_);
    // 收集每个机子的数据进行最后的reduce
    for (u32 i = 0; i < serverSize_; ++i) {
        if (i == serverId_) {
            isReduceBlock[i] = true;
            reduceMem[i] = std::move(usrOutInraMem);
            HCCL_DEBUG("[%s] inter-server final local reduce, NO.%u reduceMem stores userOut", __func__, i);
            continue;
        }
        // 所有local reduce数据为第serverId大块的第0块
        isReduceBlock[i] = true;
        u32 cclbufferIdx = GetRankIdx(i, 0);
        DeviceMem cclbufferIntraMem;
        CHK_RET(GetLocalCclbufferDeviceMem(cclbufferIdx, cclbufferIntraMem, 0));
        reduceMem[i] = std::move(cclbufferIntraMem);
        HCCL_DEBUG(
            "[%s] inter-server final local reduce, NO.%u reduceMem stores cclbufferIdx[%u]", __func__, i, cclbufferIdx);
    }
    CHK_RET(LocalReduce(reduceMem, isReduceBlock, retIndex, true));
    HCCL_INFO("[%s] intra-server run final local reduce success", __func__);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMultiDeterPipeline::AlltoallSync(u32 step, bool isStartPhase)
{
    if (isStartPhase) {
        CHK_RET(MainRecordSub(all2allStreamBegin_, all2allStreamBegin_ + all2allStreamSize_));
        CHK_RET(SubWaitMain(all2allStreamBegin_, all2allStreamBegin_ + all2allStreamSize_));
        HCCL_DEBUG("[%s] userRank[%u], step[%u/%u] begin sync", __func__, userRank_, step, allSteps_);
    } else {
        CHK_RET(SubRecordMain(all2allStreamBegin_, all2allStreamBegin_ + all2allStreamSize_));
        CHK_RET(MainWaitSub(all2allStreamBegin_, all2allStreamBegin_ + all2allStreamSize_));
        HCCL_DEBUG("[%s] userRank[%u], step[%u/%u] end sync", __func__, userRank_, step, allSteps_);
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterMultiDeterPipeline::LocalReduceSync(u32 step, bool isStartPhase)
{
    if (isStartPhase) {
        CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, streamNotifySub_[reduceMainStreamIdx_], -1));
        CHK_RET(LocalNotify::Wait(
            subStreams_[reduceMainStreamIdx_], dispatcher_, streamNotifySub_[reduceMainStreamIdx_],
            INVALID_VALUE_STAGE));
        HCCL_DEBUG("[%s] userRank[%u], step[%u/%u] begin sync", __func__, userRank_, step, allSteps_);
    } else {
        CHK_RET(LocalNotify::Post(
            subStreams_[reduceMainStreamIdx_], dispatcher_, streamNotifyMain_[reduceMainStreamIdx_], -1));
        CHK_RET(
            LocalNotify::Wait(mainStream_, dispatcher_, streamNotifyMain_[reduceMainStreamIdx_], INVALID_VALUE_STAGE));
        HCCL_DEBUG("[%s] userRank[%u], step[%u/%u] end sync", __func__, userRank_, step, allSteps_);
    }
    return HCCL_SUCCESS;
}

// 每个server内首先要进行alltoall full mesh收集数据，再进行机内local reduce，最后发送给指定server
HcclResult ReduceScatterMultiDeterPipeline::RunAsync()
{
    CHK_RET(RunAsyncReduceScatterPipeline());
    return HCCL_SUCCESS;
}

// 适配新CollExecutor接口
HcclResult ReduceScatterMultiDeterPipeline::Prepare(
    HcomCollOpInfo* opInfo, DeviceMem& cclBuffer, const u64 count, const u64 offset, const std::vector<Slice>& slices,
    const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo, Stream& mainStream,
    std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
    std::vector<std::shared_ptr<LocalNotify>>& notifySub)
{
    // stream
    subStreams_ = subStream;
    mainStream_ = mainStream;
    subStreamNum_ = subStreams_.size();
    CHK_RET(PrepareTopoInfo(level0CommInfo, level1CommInfo));
    all2allStreamBegin_ = 0;
    all2allStreamSize_ = intraRankSize_ - 1; // alltoall 从流只需要 intraRankSize_ - 1条
    reduceStreamBegin_ = intraRankSize_ - 1;
    reduceMainStreamIdx_ = intraRankSize_ - 1;
    reduceStreamSize_ = MAX_REDUCE_STREAM_NUM; // reduce 从流只需要MAX_REDUCE_STREAM_NUM条
    HCCL_INFO(
        "[%s] stream: all2allStreamBegin[%u], size[%u], reduceStreamBegin[%u], reduceMainStreamIdx[%u], size[%u]",
        __func__, all2allStreamBegin_, all2allStreamSize_, reduceStreamBegin_, reduceMainStreamIdx_, reduceStreamSize_);

    // opInfo
    opInfo_ = opInfo;
    reductionOp_ = opInfo_->reduceOp;
    usrInMemPtr_ = opInfo_->inputAddr;
    usrOutMemPtr_ = opInfo_->outputAddr;
    dataType_ = opInfo_->dataType;
    unitSize_ = SIZE_TABLE[opInfo_->dataType];
    memSliceSize_ = opInfo_->count * unitSize_; // 一整块rank的内存大小

    // streamNotify, size: n
    streamNotifySub_ = notifySub;
    if (streamNotifySub_.size() < intraRankSize_) {
        HCCL_ERROR(
            "[%s] rank[%u] streamNotifySub_ size[%u] is smaller than intraRankSize[%u]", __func__, userRank_,
            streamNotifySub_.size(), intraRankSize_);
        return HCCL_E_INTERNAL;
    }
    streamNotifyMain_ = notifyMain;
    if (streamNotifyMain_.size() < intraRankSize_) {
        HCCL_ERROR(
            "[%s] rank[%u] streamNotifyMain_ size[%u] is smaller than intraRankSize[%u]", __func__, userRank_,
            streamNotifyMain_.size(), intraRankSize_);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO(
        "[%s] notify: streamNum[%u], streamNotifySubNum[%u], streamNotifyMainNum[%u]", __func__, subStreams_.size(),
        streamNotifySub_.size(), streamNotifyMain_.size());

    // 此次reduce scatter数据信息
    cclBuffer_ = cclBuffer;
    count_ = count;
    curSize_ = count_ * unitSize_;
    bufferSize_ = cclBuffer.size();
    offset_ = offset;
    slices_ = slices;
    eachRankCclbufferSize_ = curSize_ + HCCL_MIN_SLICE_ALIGN_910B;
    if (slices_.size() != userRankSize_) {
        HCCL_ERROR("[%s] slices size[%llu] not match userRankSize[%u]", __func__, slices_.size(), userRankSize_);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO(
        "[%s] this time: bufferSize[%u], count[%u], curSize[%u], offset[%u], slicesNum[%u]", __func__, bufferSize_,
        count_, curSize_, offset_, slices_.size());
    return HCCL_SUCCESS;
}

u64 ReduceScatterMultiDeterPipeline::GetLocalReduceSerialThresh() { return curSize_; }

REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_MULTI_DETERMINISTIC_PIPELINE, ReduceScatterMultiDeterPipeline);
} // namespace hccl
