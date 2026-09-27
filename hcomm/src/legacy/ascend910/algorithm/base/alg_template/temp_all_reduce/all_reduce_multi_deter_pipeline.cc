/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_reduce_multi_deter_pipeline.h"
#include "alg_template_register.h"

namespace hccl {
AllReduceMultiDeterPipeline::AllReduceMultiDeterPipeline(const HcclDispatcher dispatcher)
    : MultiDeterPipeline(dispatcher)
{}

AllReduceMultiDeterPipeline::~AllReduceMultiDeterPipeline() {}

HcclResult AllReduceMultiDeterPipeline::GetRemoteCclbufferDeviceMem(
    u32 inputSliceIndex, LINK link, u32 outputSliceIndex, DeviceMem& remoteMem)
{
    void* remoteMemPtr = nullptr;
    CHK_RET(link->GetRemoteMem(UserMemType::OUTPUT_MEM, &remoteMemPtr)); // 图模式不一定是input，统一output
    u8* beginAddrU8 = static_cast<u8*>(remoteMemPtr);
    u64 size = slices_[inputSliceIndex].size;
    u64 offset = slices_[outputSliceIndex].offset;
    u8* intraSrcAddr = beginAddrU8 + offset;
    remoteMem = DeviceMem::create(intraSrcAddr, size);
    if (remoteMem.ptr() == nullptr) {
        HCCL_ERROR("[%s] offset + size = [%llu] > cclBufferSize[%llu]", __func__, offset + size, outCclBuffer_.size());
        return HCCL_E_MEMORY;
    }
    HCCL_DEBUG(
        "[%s] rank[%u], beginAddr[%p], offset[%llu](slices_[outputSliceIndex].offset), "
        "curSize[%llu], totalBufferSize[%llu]",
        __func__, outputSliceIndex, remoteMem.ptr(), offset, size, outCclBuffer_.size());
    return HCCL_SUCCESS;
}

HcclResult
AllReduceMultiDeterPipeline::GetLocalInCclbufferDeviceMem(u32 rankIdInAllRanks, DeviceMem& localMem, bool ifUseLastSize)
{
    u64 size = ifUseLastSize ? lastSize_ : slices_[rankIdInAllRanks].size;
    u64 offset = slices_[rankIdInAllRanks].offset;
    localMem = inCclBuffer_.range(offset, size);
    if (localMem.ptr() == nullptr) {
        HCCL_ERROR(
            "[%s] get localMem failed, offset + size = [%llu] > cclBufferSize[%llu]", __func__, offset + size,
            inCclBuffer_.size());
        return HCCL_E_MEMORY;
    }
    HCCL_DEBUG(
        "[%s] rank[%u], beginAddr[%p], offset[%llu], curSize[%llu], totalBufferSize[%llu]", __func__, rankIdInAllRanks,
        localMem.ptr(), offset, size, inCclBuffer_.size());
    return HCCL_SUCCESS;
}

// reduce scatter RDMA、SDMA 发送后 取本地 sliceOffset = slices_[rankIdInAllRanks].offset偏移处存放地址
// 什么时候取小内存额外进行判断
// allgather 都是发到相同内存块，所以不需要额外判断是否为小内存
HcclResult AllReduceMultiDeterPipeline::GetLocalOutCclbufferDeviceMem(
    u32 rankIdInAllRanks, DeviceMem& localMem, bool ifUseLastSize)
{
    u64 size = ifUseLastSize ? lastSize_ : slices_[rankIdInAllRanks].size;
    u64 offset = slices_[rankIdInAllRanks].offset;
    localMem = outCclBuffer_.range(offset, size);
    if (localMem.ptr() == nullptr) {
        HCCL_ERROR(
            "[%s] get localMem failed, offset + size = [%llu] > cclBufferSize[%llu]", __func__, offset + size,
            outCclBuffer_.size());
        return HCCL_E_MEMORY;
    }
    HCCL_DEBUG(
        "[%s] rank[%u], beginAddr[%p], offset[%llu], curSize[%llu], totalBufferSize[%llu]", __func__, rankIdInAllRanks,
        localMem.ptr(), offset, size, outCclBuffer_.size());
    return HCCL_SUCCESS;
}

HcclResult AllReduceMultiDeterPipeline::GetLocalUserDeviceMem(u32 rankIdInAllRanks, DeviceMem& localMem, bool isUserIn)
{
    u8* beginAddrU8 = isUserIn ? static_cast<u8*>(usrInMemPtr_) : static_cast<u8*>(usrOutMemPtr_);
    u64 offset = slices_[rankIdInAllRanks].offset;
    u64 size = slices_[rankIdInAllRanks].size;
    u8* intraSrcAddr = beginAddrU8 + offset; // 不用 + offset_，因为usrInMem_已经加过了
    localMem = DeviceMem::create(intraSrcAddr, size);
    if (localMem.ptr() == nullptr) {
        HCCL_ERROR(
            "[%s] get localMem failed, offset + size = [%llu] > cclBufferSize[%llu]", __func__, offset + size,
            outCclBuffer_.size());
        return HCCL_E_MEMORY;
    }
    HCCL_DEBUG(
        "[%s] rank[%u], beginAddr[%p], offset[%llu], curSize[%llu], totalBufferSize[%llu] isUserIn[%u]", __func__,
        rankIdInAllRanks, localMem.ptr(), offset, size, outCclBuffer_.size(), isUserIn);
    return HCCL_SUCCESS;
}

HcclResult AllReduceMultiDeterPipeline::GetLocalUserInDeviceMem(u32 rankIdInAllRanks, DeviceMem& localMem)
{
    CHK_RET(GetLocalUserDeviceMem(rankIdInAllRanks, localMem, true));
    return HCCL_SUCCESS;
}

HcclResult AllReduceMultiDeterPipeline::GetLocalUserOutDeviceMem(u32 rankIdInAllRanks, DeviceMem& localMem)
{
    CHK_RET(GetLocalUserDeviceMem(rankIdInAllRanks, localMem, false));
    return HCCL_SUCCESS;
}

HcclResult AllReduceMultiDeterPipeline::RunLocalCopy()
{
    if (intraRankId_ != intraRankSize_ - 1) {
        HCCL_DEBUG("[%s] intra-card no need to copy userRank[%u], intraRankId_[%u]", __func__, userRank_, intraRankId_);
        return HCCL_SUCCESS;
    }

    DeviceMem userIn;
    CHK_RET(GetLocalUserInDeviceMem(userRank_, userIn));
    DeviceMem cclbuffer;
    CHK_RET(GetLocalOutCclbufferDeviceMem(userRank_, cclbuffer, false));
    // 使用主流搬迁卡内数据
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, cclbuffer, userIn, mainStream_));
    HCCL_DEBUG(
        "[%s] intra-card copy data from userInMem to number[%u] cclbuffer[%p] size[%llu]", __func__, userRank_,
        cclbuffer.ptr(), curSize_);
    return HCCL_SUCCESS;
}

// 机内alltoall full mesh收集数据, #step表示pairwise的第step步
HcclResult AllReduceMultiDeterPipeline::RunIntraAlltoallPreSync(u32 step)
{
    HCCL_DEBUG("[%s] intra-server alltoall begin, step[%u]", __func__, step);
    // alltoall需要准备跨机要的reduce数据 输出的位置是buffer的第(Rn+Sn-n)%Sn组分块
    // 输入是input的第(R1+1)%S1组数据（往后1）
    // 每个rank机内只需拷贝intraRankSize_ - 1次
    HCCL_DEBUG(
        "[%s] intra-server SDMA send begin, [serverId, intraRankId] = [%u, %u]", __func__, serverId_, intraRankId_);
    for (u32 i = 0; i < intraRankSize_ - 1; ++i) {
        HCCL_DEBUG(
            "[%s] intra-server SDMA send begin, userRank[%u] step[%u] pro[%u/%u]", __func__, userRank_, i, i + 1,
            intraRankSize_ - 1);
        // 从机内rankId为recvIntraRankId收集数据，也发给机内rankId为sendIntraRankId数据
        u32 sendIntraRankId = GetNextIntraRankIdByStep(i + 1);
        LINK sendIntraLink = intraLinks_[sendIntraRankId];
        CHK_RET(sendIntraLink->TxAck(subStreams_[i]));
        CHK_RET(sendIntraLink->RxAck(subStreams_[i]));
    }
    // 增加主从流同步，目的是让SDMA同时进行
    CHK_RET(MainWaitSub(all2allStreamBegin_, all2allStreamBegin_ + all2allStreamSize_));
    CHK_RET(SubRecordMain(all2allStreamBegin_, all2allStreamBegin_ + all2allStreamSize_));
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inCclBuffer_, outCclBuffer_, mainStream_, dispatcher_));
    CHK_RET(MainRecordSub(all2allStreamBegin_, all2allStreamBegin_ + all2allStreamSize_));
    CHK_RET(SubWaitMain(all2allStreamBegin_, all2allStreamBegin_ + all2allStreamSize_));
    return HCCL_SUCCESS;
}

HcclResult AllReduceMultiDeterPipeline::BatchPostNotifyForStreams(
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
            CHK_RET(AlgTemplateBase::ExecEmptyTask(
                inCclBuffer_, outCclBuffer_, subStreams_[reduceMainStreamIdx_], dispatcher_));
            CHK_RET(LocalNotify::Post(
                subStreams_[reduceMainStreamIdx_], dispatcher_, streamNotifySub_[streamIdx], profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(
                subStreams_[streamIdx], dispatcher_, streamNotifySub_[streamIdx], profilerInput_.stage));
            HCCL_DEBUG("[%s] stream[%u] start phase notify done", __func__, streamIdx);
        } else {
            CHK_RET(LocalNotify::Post(
                subStreams_[streamIdx], dispatcher_, streamNotifyMain_[streamIdx], profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(
                subStreams_[reduceMainStreamIdx_], dispatcher_, streamNotifyMain_[streamIdx], profilerInput_.stage));
            CHK_RET(AlgTemplateBase::ExecEmptyTask(
                inCclBuffer_, outCclBuffer_, subStreams_[reduceMainStreamIdx_], dispatcher_));
            HCCL_DEBUG("[%s] stream[%u] sync phase notify done", __func__, streamIdx);
        }
    }
    return HCCL_SUCCESS;
}

bool AllReduceMultiDeterPipeline::IfUseLastSize(u32 step, u32 sendServerId)
{
    // 第0步的最后一块rank，使用小块内存
    if (step == 0 && userRank_ == userRankSize_ - 1) {
        return true;
    }
    // 其他步骤，接收数据的rank是机内最后一个且allreduce后的结果是发给最后一个server
    if (step != 0 && (sendServerId == serverSize_ - 1) && (intraRankId_ == intraRankSize_ - 1)) {
        return true;
    }
    return false;
}

// 机内localreduce首先按序收集所有内存块，接着二分归并reduce，最多使用4条流并行
HcclResult AllReduceMultiDeterPipeline::RunIntraLocalReduce(u32 step)
{
    HCCL_DEBUG("[%s] inter-server local reduce begin, step[%u]", __func__, step);
    u32 recvServerId = GetPreServerIdByStep(step);  // 从上一个收
    u32 sendServerId = GetNextServerIdByStep(step); // 发给发下一个
    std::vector<DeviceMem> reduceMem;
    std::vector<bool> isReduceBlock;
    isReduceBlock.resize(intraRankSize_);
    reduceMem.resize(intraRankSize_);
    u32 retIndex = 0;
    // 机内，最后rank的规约结果放在倒数第2块，其他放在倒数第1块
    // 机内第0步，规约结果放在第userRank块cclbuffer
    if (intraRankId_ == intraRankSize_ - 1) {
        retIndex = intraRankSize_ - SECOND_TO_LAST;
    } else {
        retIndex = intraRankSize_ - 1;
    }
    bool ifUseLastSize = IfUseLastSize(step, sendServerId);
    u32 userInIdx = GetRankIdx(sendServerId, intraRankId_);
    // 最后一块留给allreduce，idx为第sendServerId大块内存的第idx小块
    u32 idx = 0;
    for (u32 i = 0; i < intraRankSize_; ++i) {
        u32 outCCLbufferIdx = 0;
        // i == intraRankId_时，需要取userIn内存，
        if (i == intraRankId_) {
            // 第0步，机内最后一个rank取cclbuffer，因为localcopy时将该内存搬到了第userRank_块cclbuffer
            if (step == 0 && i == intraRankSize_ - 1) {
                isReduceBlock[i] = true;
                outCCLbufferIdx = userRank_;
                DeviceMem cclbufferIntraMem;
                CHK_RET(GetLocalOutCclbufferDeviceMem(outCCLbufferIdx, cclbufferIntraMem, ifUseLastSize));
                reduceMem[i] = std::move(cclbufferIntraMem);
                HCCL_DEBUG(
                    "[%s] inter-server local reduce, NO.%u reduceMem stores outCCLbufferIdx[%u]", __func__, i,
                    outCCLbufferIdx);
                retIndex = i;
            } else {
                isReduceBlock[i] = false;
                DeviceMem usrInIntraMem;
                CHK_RET(GetLocalUserInDeviceMem(userInIdx, usrInIntraMem));
                reduceMem[i] = std::move(usrInIntraMem);
                HCCL_DEBUG(
                    "[%s] inter-server local reduce, NO.%u reduceMem stores userInIdx[%u]", __func__, i, userInIdx);
            }
            continue;
        }
        // 其他情况：统一处理CCLBuffer，
        isReduceBlock[i] = true;
        outCCLbufferIdx = GetRankIdx(recvServerId, idx);
        DeviceMem cclbufferIntraMem;
        CHK_RET(GetLocalOutCclbufferDeviceMem(outCCLbufferIdx, cclbufferIntraMem, ifUseLastSize));
        reduceMem[i] = std::move(cclbufferIntraMem);
        HCCL_DEBUG(
            "[%s] inter-server local reduce, NO.%u reduceMem stores outCCLbufferIdx[%u]", __func__, i, outCCLbufferIdx);
        idx++;
        // step 0, localreduce到userRank_块内存上
        if (step == 0 && outCCLbufferIdx == userRank_) {
            retIndex = i;
        }
    }
    HCCL_DEBUG(
        "[%s] intra-server local reduce, retIndex[%u], intraRankId[%u], intraRankSize[%u]", __func__, retIndex,
        intraRankId_, intraRankSize_);
    CHK_RET(LocalReduce(reduceMem, isReduceBlock, retIndex, false));
    HCCL_INFO("[%s] intra-server step[%u] run local reduce success", __func__, step);
    return HCCL_SUCCESS;
}

HcclResult AllReduceMultiDeterPipeline::RunInterSend(u32 step)
{
    HCCL_DEBUG("[%s] inter-server RDMA write begin, step[%u]", __func__, step);
    // 使用主流进行rdma
    u32 recvServerId = GetPreServerIdByStep(step);  // 从上一个收
    u32 sendServerId = GetNextServerIdByStep(step); // 发给下一个
    LINK recvInterLink = serverLinks_[recvServerId];
    LINK sendInterLink = serverLinks_[sendServerId];
    // 跨机 输出的位置是buffer的第(Rn+Sn-n)%Sn组分块的第1块
    // 跨机 输入是input的第(R1+1)%S1组数据（往后1）的第2块
    u32 reduceRetIndex = intraRankSize_ - SECOND_TO_LAST;
    u32 recvCCLbufferIdx = GetRankIdx(recvServerId, 0);
    u32 sendCCLbufferIdx = GetRankIdx(recvServerId, reduceRetIndex);
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
    bool ifSendToLastServer = IfUseLastSize(step, sendServerId);
    // 最后一个rank则只收小块内存
    CHK_RET(GetLocalOutCclbufferDeviceMem(recvCCLbufferIdx, recvMem, isLastRank_));
    // 如果是发给最后一个rank则只发小块内存
    CHK_RET(GetLocalOutCclbufferDeviceMem(sendCCLbufferIdx, sendMem, ifSendToLastServer));

    HCCL_DEBUG(
        "[%s] inter-server RDMA write begin, cclbufferIdx: [%u] send to [%u], [%u] recv from [%u]", __func__,
        sendCCLbufferIdx, remoteRecvCCLbufferIdx, recvCCLbufferIdx, remoteSendCCLbufferIdx);
    HCCL_DEBUG(
        "[%s] inter-server RDMA write begin, rankId: [%u] send to [%u], [%u] recv from [%u]", __func__, userRank_,
        sendRankId, userRank_, recvRankId);
    HCCL_DEBUG(
        "[%s] inter-server RDMA write begin, if use last small mem? : isLastRank[%u], ifSendToLastServer[%u]", __func__,
        isLastRank_, ifSendToLastServer);
    if (recvInterLink->IsSpInlineReduce() && sendInterLink->IsSpInlineReduce()) {
        CHK_RET(sendInterLink->TxAck(mainStream_));
        CHK_RET(recvInterLink->RxAck(mainStream_));
        DeviceMem dstMem = std::move(recvMem);
        void* remoteMemPtr = nullptr;
        CHK_RET(recvInterLink->GetRemoteMem(UserMemType::OUTPUT_MEM, &remoteMemPtr)); // 图模式不一定是input，统一output
        u8* beginAddrU8 = static_cast<u8*>(remoteMemPtr);
        u64 size = slices_[recvCCLbufferIdx].size;
        u64 offset = slices_[remoteSendCCLbufferIdx].offset;
        u8* intraSrcAddr = beginAddrU8 + offset;
        DeviceMem srcMem = DeviceMem::create(intraSrcAddr, isLastRank_ ? lastSize_ : size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, mainStream_, recvRankId, recvInterLink->GetLinkType()));
        CHK_RET(sendInterLink->TxDataSignal(mainStream_));
        CHK_RET(recvInterLink->RxDataSignal(mainStream_));
    } else {
        CHK_RET(recvInterLink->TxAck(mainStream_));
        CHK_RET(sendInterLink->RxAck(mainStream_));

        u64 size = slices_[sendCCLbufferIdx].size; // 没有特殊情况发送接收内存大小都是一样
        // 发送的size取本地发的数据大小，如果是发给最后一个rank则只发小块内存
        CHK_RET(sendInterLink->TxAsync(
            UserMemType::OUTPUT_MEM, slices_[remoteRecvCCLbufferIdx].offset, sendMem.ptr(),
            ifSendToLastServer ? lastSize_ : size, mainStream_));
        // 接收的size取远端发的数据大小, 如果是最后一个rank则只收小块内存
        CHK_RET(recvInterLink->RxAsync(
            UserMemType::OUTPUT_MEM, slices_[remoteSendCCLbufferIdx].offset, recvMem.ptr(),
            isLastRank_ ? lastSize_ : size, mainStream_));
        CHK_RET(recvInterLink->PostFinAck(mainStream_));
        CHK_RET(sendInterLink->WaitFinAck(mainStream_));
    }
    HCCL_INFO("[%s] inter-server step[%u] run RDMA send success", __func__, step);
    return HCCL_SUCCESS;
}

HcclResult AllReduceMultiDeterPipeline::RunFinalReduce()
{
    HCCL_DEBUG("[%s] intra-server final reduce begin", __func__);
    std::vector<DeviceMem> reduceMem;
    std::vector<bool> isReduceBlock;
    isReduceBlock.resize(serverSize_);
    reduceMem.resize(serverSize_);

    u32 retIndex = serverId_;
    // userRank_ == userRankSize_ - 1时取小内存进行最后一次reduce
    bool ifUseLastSize = isLastRank_;
    HCCL_DEBUG(
        "[%s] intra-server retIndex[%u], interRankSize[%u], ifUseLastSize[%u]", __func__, retIndex, serverSize_,
        ifUseLastSize);
    // 收集每个机子的数据进行最后的reduce
    for (u32 i = 0; i < serverSize_; ++i) {
        DeviceMem cclbufferIntraMem;
        u32 cclbufferIdx = 0;
        if (i == serverId_) {
            cclbufferIdx = userRank_;
        } else {
            cclbufferIdx = GetRankIdx(i, 0);
        }
        // 所有local reduce数据为第serverId大块的第0块
        isReduceBlock[i] = true;
        CHK_RET(GetLocalOutCclbufferDeviceMem(cclbufferIdx, cclbufferIntraMem, ifUseLastSize));
        reduceMem[i] = std::move(cclbufferIntraMem);
        HCCL_DEBUG(
            "[%s] inter-server final local reduce, NO.%u reduceMem stores cclbufferIdx[%u]", __func__, i, cclbufferIdx);
    }
    CHK_RET(LocalReduce(reduceMem, isReduceBlock, retIndex, true)); // final reduce使用主流进行操作
    HCCL_INFO("[%s] intra-server run final local reduce success", __func__);
    return HCCL_SUCCESS;
}

HcclResult AllReduceMultiDeterPipeline::AlltoallSync(u32 step, bool isStartPhase)
{
    if (isStartPhase) {
        CHK_RET(AlgTemplateBase::ExecEmptyTask(inCclBuffer_, outCclBuffer_, mainStream_, dispatcher_));
        CHK_RET(MainRecordSub(all2allStreamBegin_, all2allStreamBegin_ + all2allStreamSize_));
        CHK_RET(SubWaitMain(all2allStreamBegin_, all2allStreamBegin_ + all2allStreamSize_));
        HCCL_DEBUG("[%s] userRank[%u], step[%u/%u] begin sync", __func__, userRank_, step, allSteps_);
    } else {
        CHK_RET(SubRecordMain(all2allStreamBegin_, all2allStreamBegin_ + all2allStreamSize_));
        CHK_RET(MainWaitSub(all2allStreamBegin_, all2allStreamBegin_ + all2allStreamSize_));
        CHK_RET(AlgTemplateBase::ExecEmptyTask(inCclBuffer_, outCclBuffer_, mainStream_, dispatcher_));
        HCCL_DEBUG("[%s] userRank[%u], step[%u/%u] end sync", __func__, userRank_, step, allSteps_);
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceMultiDeterPipeline::LocalReduceSync(u32 step, bool isStartPhase)
{
    if (isStartPhase) {
        CHK_RET(AlgTemplateBase::ExecEmptyTask(inCclBuffer_, outCclBuffer_, mainStream_, dispatcher_));
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
        CHK_RET(AlgTemplateBase::ExecEmptyTask(inCclBuffer_, outCclBuffer_, mainStream_, dispatcher_));
        HCCL_DEBUG("[%s] userRank[%u], step[%u/%u] end sync", __func__, userRank_, step, allSteps_);
    }
    return HCCL_SUCCESS;
}

HcclResult
AllReduceMultiDeterPipeline::RunAllGatherInterServer(u32 step, const LINK& prevInterLink, const LINK& nextInterLink)
{
    HCCL_INFO(
        "[%s] inter-server allgather run, userRank[%u], step[%u/%u]", __func__, userRank_, step,
        serverSize_ - STEP_OFFSET_TWO);
    CHK_RET(prevInterLink->TxAck(mainStream_));
    CHK_RET(nextInterLink->RxAck(mainStream_));
    u32 rxDMAMemSliceId = (serverSize_ + step) % PARITY_BASE;
    u32 txDMAMemSliceId = (serverSize_ + step - 1) % PARITY_BASE;
    UserMemType srcMemType = txDMAMemSliceId == serverSizeParity_ ? UserMemType::OUTPUT_MEM : UserMemType::INPUT_MEM;
    UserMemType dstMemType = rxDMAMemSliceId == serverSizeParity_ ? UserMemType::OUTPUT_MEM : UserMemType::INPUT_MEM;
    u32 txSliceId = ((serverId_ + step) % serverSize_) * intraRankSize_ + intraRankId_;
    u32 txDataSize = slices_[txSliceId].size;
    DeviceMem txlocalMem;
    if (txDMAMemSliceId == serverSizeParity_) {
        CHK_RET(GetLocalOutCclbufferDeviceMem(txSliceId, txlocalMem, false));
    } else {
        CHK_RET(GetLocalInCclbufferDeviceMem(txSliceId, txlocalMem, false));
    }
    CHK_RET(nextInterLink->TxAsync(dstMemType, slices_[txSliceId].offset, txlocalMem.ptr(), txDataSize, mainStream_));

    u32 rxSliceId = ((serverId_ + step + 1) % serverSize_) * intraRankSize_ + intraRankId_;
    DeviceMem rxLocalMem;
    if (rxDMAMemSliceId == serverSizeParity_) {
        CHK_RET(GetLocalOutCclbufferDeviceMem(rxSliceId, rxLocalMem, false));
    } else {
        CHK_RET(GetLocalInCclbufferDeviceMem(rxSliceId, rxLocalMem, false));
    }
    u64 rxDataSize = slices_[rxSliceId].size;
    CHK_RET(prevInterLink->RxAsync(srcMemType, slices_[rxSliceId].offset, rxLocalMem.ptr(), rxDataSize, mainStream_));
    HCCL_DEBUG(
        "[%s] step[%u], txId[%u], rxId[%u], srcMemType[%u], dstMemType[%u]", __func__, step, txDMAMemSliceId,
        rxDMAMemSliceId, srcMemType, dstMemType);
    HCCL_DEBUG(
        "[%s] txlocalMem: ptr[%p], size[%llu], rxLocalMem: ptr[%p], size[%llu]", __func__, txlocalMem.ptr(),
        txlocalMem.size(), rxLocalMem.ptr(), rxLocalMem.size());
    HCCL_DEBUG(
        "[%s] send txlocalMem to txSliceId[%llu], recv rxLocalMem from rxSliceId[%llu]", __func__, txSliceId,
        rxSliceId);
    HCCL_INFO("[%s] inter-server allgather success", __func__);
    return HCCL_SUCCESS;
}

HcclResult AllReduceMultiDeterPipeline::RunAllGatherIntraServer(u32 step)
{
    HCCL_INFO("[%s] intra-server allgather run, userRank[%u], step[%u/%u]", __func__, userRank_, step, serverSize_ - 1);
    u32 dmaMemSliceId = (serverSize_ + step - 1) % PARITY_BASE;
    for (u32 i = 1; i < intraRankSize_; i++) {
        u32 remIntraRankId = (intraRankId_ + i) % intraRankSize_;
        CHK_RET(intraLinks_[remIntraRankId]->TxAck(subStreams_[i - 1]));
        CHK_RET(intraLinks_[remIntraRankId]->RxAck(subStreams_[i - 1]));
        void* remoteMemPtr = nullptr;
        CHK_RET(intraLinks_[remIntraRankId]->GetRemoteMem(
            dmaMemSliceId == serverSizeParity_ ? UserMemType::OUTPUT_MEM : UserMemType::INPUT_MEM, &remoteMemPtr));
        u32 remoteCclbufferId = ((serverId_ + step) % serverSize_) * intraRankSize_ + remIntraRankId;
        DeviceMem src = DeviceMem::create(
            static_cast<u8*>(remoteMemPtr) + slices_[remoteCclbufferId].offset, slices_[remoteCclbufferId].size);
        DeviceMem dst = DeviceMem::create(
            static_cast<u8*>(usrOutMemPtr_) + slices_[remoteCclbufferId].offset, slices_[remoteCclbufferId].size);
        CHK_RET(HcclD2DMemcpyAsync(
            dispatcher_, dst, src, subStreams_[i - 1], intraLinks_[remIntraRankId]->GetRemoteRank(),
            intraLinks_[remIntraRankId]->GetLinkType()));
        CHK_RET(intraLinks_[remIntraRankId]->TxDataSignal(subStreams_[i - 1]));
        CHK_RET(intraLinks_[remIntraRankId]->RxDataSignal(subStreams_[i - 1]));
    }
    HCCL_INFO("[%s] intra-server allgather success", __func__);
    return HCCL_SUCCESS;
}

HcclResult AllReduceMultiDeterPipeline::RunAsyncAllgatherPipeline()
{
    HCCL_INFO("[%s] begin, userRank[%u]", __func__, userRank_);
    //  机间 ring algo 逆时针，从后往前
    u32 prevInterRankId = GetNextServerIdByStep(1);
    u32 nextInterRankId = GetPreServerIdByStep(1);
    LINK prevInterLink = serverLinks_[prevInterRankId];
    LINK nextInterLink = serverLinks_[nextInterRankId];
    for (u32 step = 0; step < serverSize_; step++) {
        HCCL_INFO("[%s] allgather pipeline, userRank[%u], step[%u/%u]", __func__, userRank_, step, serverSize_ - 1);
        CHK_RET(MainRecordSub(0, subStreamNum_));
        CHK_RET(SubWaitMain(0, subStreamNum_));
        if (step < serverSize_ - 1) {
            CHK_RET(RunAllGatherInterServer(step, prevInterLink, nextInterLink));
            CHK_RET(prevInterLink->PostFinAck(mainStream_));
            CHK_RET(nextInterLink->WaitFinAck(mainStream_));
            // inter的最后一步需要barrier确保数据发完
            if (step == serverSize_ - STEP_OFFSET_TWO) {
                CHK_RET(ExecuteBarrier(prevInterLink, nextInterLink, mainStream_));
            }
        }
        CHK_RET(RunAllGatherIntraServer(step));
        CHK_RET(SubRecordMain(0, subStreamNum_));
        CHK_RET(MainWaitSub(0, subStreamNum_));
        u32 cclbufferFlag = (serverSize_ + step - 1) % PARITY_BASE;
        u32 sliceId = ((serverId_ + step) % serverSize_) * intraRankSize_ + intraRankId_;
        DeviceMem srcMem;
        if (cclbufferFlag == serverSizeParity_) {
            CHK_RET(GetLocalOutCclbufferDeviceMem(sliceId, srcMem, false));
        } else {
            CHK_RET(GetLocalInCclbufferDeviceMem(sliceId, srcMem, false));
        }
        DeviceMem dstMem;
        CHK_RET(GetLocalUserOutDeviceMem(sliceId, dstMem));
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, mainStream_));
        HCCL_DEBUG(
            "[%s] step[%u], cclbufferFlag[%u], sliceId[%u], cclBufferSrcMem: ptr[%p], size[%llu]", __func__, step,
            cclbufferFlag, sliceId, srcMem.ptr(), srcMem.size());
    }
    HCCL_INFO("[%s] end, userRank[%u]", __func__, userRank_);
    return HCCL_SUCCESS;
}

// 实现为确定性reduce scatter pipeline + all gather pipeline
HcclResult AllReduceMultiDeterPipeline::RunAsync()
{
    HCCL_INFO(
        "[AllReduceMultiDeterPipeline] run begin: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] "
        "cclBuffer[%p].",
        userRank_, userRankSize_, usrInMemPtr_, usrOutMemPtr_, outCclBuffer_.ptr());
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_RET(RunAsyncReduceScatterPipeline());
    CHK_RET(RunAsyncAllgatherPipeline());
    HCCL_INFO("[AllReduceMultiDeterPipeline] AllReduceMultiDeterPipeline success userRank[%u] ", userRank_);
    return HCCL_SUCCESS;
}

// 适配新CollExecutor接口
HcclResult AllReduceMultiDeterPipeline::Prepare(
    HcomCollOpInfo* opInfo, DeviceMem& inBuffer, DeviceMem& outBuffer, const u64 count,
    const std::vector<Slice>& slices, const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo,
    Stream& mainStream, std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
    std::vector<std::shared_ptr<LocalNotify>>& notifySub)
{
    // opInfo
    opInfo_ = opInfo;
    dataType_ = opInfo_->dataType;
    unitSize_ = SIZE_TABLE[opInfo_->dataType];
    memSliceSize_ = opInfo_->count * unitSize_; // 一整块rank的内存大小
    usrInMemPtr_ = opInfo_->inputAddr;
    usrOutMemPtr_ = opInfo_->outputAddr;
    reductionOp_ = opInfo_->reduceOp;

    // stream
    mainStream_ = mainStream;
    subStreams_ = subStream;
    subStreamNum_ = subStreams_.size();
    CHK_RET(PrepareTopoInfo(level0CommInfo, level1CommInfo));
    all2allStreamBegin_ = 0;
    all2allStreamSize_ = intraRankSize_ - 1; // alltoall 从流只需要 intraRankSize_ - 1条
    reduceMainStreamIdx_ = intraRankSize_ - 1;
    reduceStreamBegin_ = intraRankSize_ - 1;
    reduceStreamSize_ = MAX_REDUCE_STREAM_NUM; // reduce 从流只需要MAX_REDUCE_STREAM_NUM条
    HCCL_INFO(
        "[%s] stream: all2allStreamBegin[%u], size[%u], reduceStreamBegin[%u], size[%u], reduceMainStreamIdx[%u]",
        __func__, all2allStreamBegin_, all2allStreamSize_, reduceStreamBegin_, reduceStreamSize_, reduceMainStreamIdx_);

    // streamNotify, size: n
    streamNotifyMain_ = notifyMain;
    if (streamNotifyMain_.size() < intraRankSize_) {
        HCCL_ERROR(
            "[%s] rank[%u] streamNotifyMain_ size[%u] is smaller than intraRankSize[%u]", __func__, userRank_,
            streamNotifyMain_.size(), intraRankSize_);
        return HCCL_E_INTERNAL;
    }
    streamNotifySub_ = notifySub;
    if (streamNotifySub_.size() < intraRankSize_) {
        HCCL_ERROR(
            "[%s] rank[%u] streamNotifySub_ size[%u] is smaller than intraRankSize[%u]", __func__, userRank_,
            streamNotifySub_.size(), intraRankSize_);
        return HCCL_E_INTERNAL;
    }

    // 此次reduce scatter数据信息
    inCclBuffer_ = inBuffer;
    outCclBuffer_ = outBuffer;
    bufferSize_ = inBuffer.size();
    slices_ = slices;
    // allreduce count为此次处理的数据总数
    curSize_ = slices_[userRank_].size;
    count_ = slices_[userRank_].size / unitSize_;
    lastSize_ = slices_[userRankSize_ - 1].size;
    isLastRank_ = (userRank_ == userRankSize_ - 1) ? true : false;
    // serverSize_是偶数，与正常allreduce pipeline中的allgather pipeline流程一样；若为奇数，则颠倒内存
    serverSizeParity_ = (serverSize_ % PARITY_BASE == 0) ? 1 : 0;
    perRankAvgDataSize_ = count * unitSize_ / userRankSize_;
    if (slices_.size() != userRankSize_) {
        HCCL_ERROR("[%s] slices size[%llu] not match userRankSize[%u]", __func__, slices_.size(), userRankSize_);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO(
        "[%s] this time: bufferSize[%u], count[%u], curSize[%u], lastSize[%u], slicesNum[%u] "
        "serverSizeParity[%u]",
        __func__, bufferSize_, count_, curSize_, lastSize_, slices_.size(), serverSizeParity_);
    return HCCL_SUCCESS;
}

u64 AllReduceMultiDeterPipeline::GetLocalReduceSerialThresh() { return perRankAvgDataSize_; }

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_MULTI_DETERMINISTIC_PIPELINE, AllReduceMultiDeterPipeline);
} // namespace hccl
