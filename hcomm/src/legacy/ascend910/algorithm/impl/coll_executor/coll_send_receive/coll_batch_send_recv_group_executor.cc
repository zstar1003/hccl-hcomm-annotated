/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_batch_send_recv_group_executor.h"

namespace hccl {

CollBatchSendRecvGroupExecutor::CollBatchSendRecvGroupExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollBatchSendRecvExecutor(dispatcher, topoMatcher)
{}

void CollBatchSendRecvGroupExecutor::CalcIsSuPodAsym(bool isA2MultiModule)
{
    // isSuPodAsym 表示A2A3卡数不一致场景或者A3多超节点server数不同场景
    // 参考alltoallv_direct_fullmesh executor中的判断逻辑
    if (topoAttr_.superPodNum > 1) {
        isSuPodAsym_
            = (static_cast<bool>(topoAttr_.multiModuleDiffDeviceNumMode)
               || static_cast<bool>(topoAttr_.multiSuperPodDiffServerNumMode));
    } else {
        isSuPodAsym_ = (static_cast<bool>(topoMatcher_->GetExternalInputInterHccsDisable()) || isA2MultiModule)
                       && static_cast<bool>(topoAttr_.multiModuleDiffDeviceNumMode);
    }
    HCCL_INFO(
        "[CalcIsSuPodAsym] superPodNum[%u] isA2MultiModule[%d] isSuPodAsym[%d]", topoAttr_.superPodNum, isA2MultiModule,
        isSuPodAsym_);
}

HcclResult CollBatchSendRecvGroupExecutor::CalcPingPongHalfSize()
{
    u32 pingPongSliceNum = GROUP_MAX_CONCURRENT * 2;
    u32 alignSize = HCCL_MIN_SLICE_ALIGN_910B;
    bufferSliceSize_ = algResResp_->cclInputMem.size() / alignSize / pingPongSliceNum * alignSize;
    // RDMA单流slot大小 = CCLOut / 2：A半区(send, 单流单slot)、B半区(recv, 单流单slot)
    rdmaDataBlockSize_ = algResResp_->cclOutputMem.size() / alignSize / RDMA_CCLOUT_HALF_NUM * alignSize;
    HCCL_INFO(
        "[CollBatchSendRecvGroupExecutor][CalcPingPongHalfSize] pingPong halfSize[%llu] rdmaDataBlockSize_[%llu]",
        bufferSliceSize_, rdmaDataBlockSize_);
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::OrganizeSendItemByStream()
{
    sendQueueBySendstream_.resize(sendStreamNum_);
    HCCL_INFO("[OrganizeSendItemByStream] sendStreamNum_[%u]", sendStreamNum_);
    while (!sendDeque_.empty()) {
        HcclSendRecvItem* curr = sendDeque_.front();
        CHK_PTR_NULL(curr);
        sendQueueBySendstream_[curr->remoteRank % sendStreamNum_].push_back(curr);
        sendDeque_.pop_front();
    }
    HCCL_INFO("OrganizeSendItemByStream Done!");
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::OrganizeRecvItemByStream()
{
    recvQueueByRecvstream_.resize(recvStreamNum_);
    HCCL_INFO("[OrganizeRecvItemByStream] recvStreamNum_[%u]", recvStreamNum_);
    while (!recvDeque_.empty()) {
        HcclSendRecvItem* curr = recvDeque_.front();
        CHK_PTR_NULL(curr);
        recvQueueByRecvstream_[curr->remoteRank % recvStreamNum_].push_back(curr);
        recvDeque_.pop_front();
    }
    HCCL_INFO("OrganizeRecvItemByStream Done!");
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::CalcPodRange()
{
    // Determine pod (server/supernode) membership — same approach as alltoallv_direct_fullmesh
    u32 devNumInlocalPod = INVALID_VALUE_RANKSIZE;
    u32 rankIdxInPod = INVALID_VALUE_RANKID;
    bool isA2MultiModule = topoAttr_.deviceType == DevType::DEV_TYPE_910B && !topoAttr_.isSingleMeshAggregation;
    if (static_cast<bool>(topoMatcher_->GetExternalInputInterHccsDisable()) || isA2MultiModule) {
        CHK_RET(topoMatcher_->GetLocalServerRankSize(topoAttr_.userRank, devNumInlocalPod, rankIdxInPod));
    } else {
        CHK_RET(topoMatcher_->GetLocalSuperPodRankSize(topoAttr_.userRank, devNumInlocalPod, rankIdxInPod));
    }
    podStartRank_ = topoAttr_.userRank - rankIdxInPod;
    podEndRank_ = podStartRank_ + devNumInlocalPod - 1;
    devNumInlocalPod_ = devNumInlocalPod;
    CalcIsSuPodAsym(isA2MultiModule);
    HCCL_INFO(
        "[CalcPodRange] userRank[%u] pod[%u-%u] devNumInlocalPod[%u] rankIdxInPod[%u] isSuPodAsym[%d]",
        topoAttr_.userRank, podStartRank_, podEndRank_, devNumInlocalPod, rankIdxInPod, isSuPodAsym_);
    return HCCL_SUCCESS;
}

bool CollBatchSendRecvGroupExecutor::IsRemoteRankRdma(u32 remoteRank) const
{
    // pod内为SDMA，跨pod为RDMA
    return !(remoteRank >= podStartRank_ && remoteRank <= podEndRank_);
}

HcclResult CollBatchSendRecvGroupExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    CHK_RET(CollBatchSendRecvExecutor::CalcCommInfo(opTransport));

    LevelNSubCommTransport& commTransport = opTransport[COMM_COMBINE_ORDER];
    for (u32 subCommIndex = 0; subCommIndex < commTransport.size(); subCommIndex++) {
        for (auto& transportRequest : commTransport[subCommIndex].transportRequests) {
            transportRequest.isUsedRdma = topoAttr_.isUsedRdmaMap.at(transportRequest.remoteUserRank);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algResource)
{
    HcclUs startut = TIME_NOW();
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollBatchSendRecvGroupExecutor] groupsendrecv starts.");

    sendStreamNum_ = GROUP_MAX_CONCURRENT;
    recvStreamNum_ = GROUP_MAX_CONCURRENT;
    HCCL_INFO("[Orchestrate] sendStreamNum_[%u], recvStreamNum_[%u]", sendStreamNum_, recvStreamNum_);

    algResResp_ = &algResource;
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_SIZE_TWO));
    CHK_RET(GetPairWiseList(param.BatchSendRecvDataDes.sendRecvItemsPtr, param.BatchSendRecvDataDes.itemNum));
    CHK_RET(ProcessSelfSendRecvTasks(param.stream));
    if (topoAttr_.userRankSize == 1) {
        HCCL_INFO(
            "tag[%s] BatchSendRecvGroup Executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
            DURATION_US(TIME_NOW() - startut));
        return HCCL_SUCCESS;
    }
    CHK_RET(CalcPodRange());
    CHK_RET(CalcPingPongHalfSize()); // ping-pong double buffering + RDMA slot
    CHK_RET(OrganizeSendItemByStream());
    CHK_RET(OrganizeRecvItemByStream());

    CHK_RET(CalcSendSlices());
    CHK_RET(CalcRecvSlices());

    CHK_RET(RunLoop(param));

    HCCL_INFO(
        "tag[%s] BatchSendRecvGroup Executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::CalcStreamTaskStatus(u32& nonEmptySendStream, u32& nonEmptyRecvStream)
{
    nonEmptySendStream = 0;
    nonEmptyRecvStream = 0;
    // 记录各从流是否有任务，供头尾同步只唤醒有任务的从流(循环中不再更新)。
    sendStreamHasTask_.assign(sendStreamNum_, false);
    recvStreamHasTask_.assign(recvStreamNum_, false);
    for (u32 i = 0; i < sendStreamNum_; i++) {
        if (!sendDataSlicesBySendStream_[i].empty()) {
            nonEmptySendStream++;
            sendStreamHasTask_[i] = true;
        }
    }
    for (u32 i = 0; i < recvStreamNum_; i++) {
        if (!recvDataSlicesByRecvStream_[i].empty()) {
            nonEmptyRecvStream++;
            recvStreamHasTask_[i] = true;
        }
    }
    rdmaSendHasTask_ = !rdmaSendSlices_.empty();
    rdmaRecvHasTask_ = !rdmaRecvSlices_.empty();
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::MainPostSubWait(Stream& mainStream)
{
    // 主流只通知有任务的从流开始
    for (u32 i = 0; i < sendStreamNum_; i++) {
        if (!sendStreamHasTask_[i]) {
            continue;
        }
        CHK_RET(LocalNotify::Post(mainStream, dispatcher_, algResResp_->notifiesAux[i], PROF_STAGE_0));
        CHK_RET(
            LocalNotify::Wait(algResResp_->slaveStreams[i], dispatcher_, algResResp_->notifiesAux[i], PROF_STAGE_0));
        HCCL_DEBUG("MainPost, Send[%u] Wait", i);
    }

    for (u32 i = 0; i < recvStreamNum_; i++) {
        if (!recvStreamHasTask_[i]) {
            continue;
        }
        CHK_RET(LocalNotify::Post(mainStream, dispatcher_, algResResp_->notifiesAux[i + sendStreamNum_], PROF_STAGE_0));
        CHK_RET(LocalNotify::Wait(
            algResResp_->slaveStreams[i + sendStreamNum_], dispatcher_, algResResp_->notifiesAux[i + sendStreamNum_],
            PROF_STAGE_0));
        HCCL_DEBUG("MainPost, Recv[%u] Wait", i);
    }

    // RDMA专用从流(send/recv各一条)
    u32 rdmaSendIdx = RdmaSendStreamIdx();
    u32 rdmaRecvIdx = RdmaRecvStreamIdx();
    if (rdmaSendHasTask_) {
        CHK_RET(LocalNotify::Post(mainStream, dispatcher_, algResResp_->notifiesAux[rdmaSendIdx], PROF_STAGE_0));
        CHK_RET(LocalNotify::Wait(
            algResResp_->slaveStreams[rdmaSendIdx], dispatcher_, algResResp_->notifiesAux[rdmaSendIdx], PROF_STAGE_0));
        HCCL_DEBUG("MainPost, RdmaSend Wait");
    }
    if (rdmaRecvHasTask_) {
        CHK_RET(LocalNotify::Post(mainStream, dispatcher_, algResResp_->notifiesAux[rdmaRecvIdx], PROF_STAGE_0));
        CHK_RET(LocalNotify::Wait(
            algResResp_->slaveStreams[rdmaRecvIdx], dispatcher_, algResResp_->notifiesAux[rdmaRecvIdx], PROF_STAGE_0));
        HCCL_DEBUG("MainPost, RdmaRecv Wait");
    }

    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::MainWaitSubPost(Stream& mainStream)
{
    // 最后主流只等待有任务的从流结束
    for (u32 i = 0; i < sendStreamNum_; i++) {
        if (!sendStreamHasTask_[i]) {
            continue;
        }
        CHK_RET(
            LocalNotify::Post(algResResp_->slaveStreams[i], dispatcher_, algResResp_->notifiesMain[i], PROF_STAGE_0));
        CHK_RET(LocalNotify::Wait(mainStream, dispatcher_, algResResp_->notifiesMain[i], PROF_STAGE_0));
        HCCL_DEBUG("MainWait, Send[%u] Post", i);
    }

    for (u32 i = 0; i < recvStreamNum_; i++) {
        if (!recvStreamHasTask_[i]) {
            continue;
        }
        CHK_RET(LocalNotify::Post(
            algResResp_->slaveStreams[i + sendStreamNum_], dispatcher_, algResResp_->notifiesMain[i + sendStreamNum_],
            PROF_STAGE_0));
        CHK_RET(
            LocalNotify::Wait(mainStream, dispatcher_, algResResp_->notifiesMain[i + sendStreamNum_], PROF_STAGE_0));
        HCCL_DEBUG("MainWait, Recv[%u] Post", i);
    }

    // RDMA专用从流(send/recv各一条)
    u32 rdmaSendIdx = RdmaSendStreamIdx();
    u32 rdmaRecvIdx = RdmaRecvStreamIdx();
    if (rdmaSendHasTask_) {
        CHK_RET(LocalNotify::Post(
            algResResp_->slaveStreams[rdmaSendIdx], dispatcher_, algResResp_->notifiesMain[rdmaSendIdx], PROF_STAGE_0));
        CHK_RET(LocalNotify::Wait(mainStream, dispatcher_, algResResp_->notifiesMain[rdmaSendIdx], PROF_STAGE_0));
        HCCL_DEBUG("MainWait, RdmaSend Post");
    }
    if (rdmaRecvHasTask_) {
        CHK_RET(LocalNotify::Post(
            algResResp_->slaveStreams[rdmaRecvIdx], dispatcher_, algResResp_->notifiesMain[rdmaRecvIdx], PROF_STAGE_0));
        CHK_RET(LocalNotify::Wait(mainStream, dispatcher_, algResResp_->notifiesMain[rdmaRecvIdx], PROF_STAGE_0));
        HCCL_DEBUG("MainWait, RdmaRecv Post");
    }

    return HCCL_SUCCESS;
}

HcclResult
CollBatchSendRecvGroupExecutor::ProcessPreloadedSendSlice(u32 streamIdx, u32& pendingSendCount, u32& nonEmptySendStream)
{
    u32 curPhase = sendCurPhase_[streamIdx];
    u32 loadedRank = sendLoadedRemoteRank_[streamIdx];
    u64 loadedSize = sendLoadedSize_[streamIdx];
    u64 kernelOffset = bufferSliceSize_ * (streamIdx * 2 + curPhase);
    u64 phaseOffset = bufferSliceSize_ * curPhase;

    HCCL_INFO(
        "[RunTasks] SendStream[%u](loaded) phase[%u] offset[%llu] size[%llu] rank[%u]", streamIdx, curPhase,
        kernelOffset, loadedSize, loadedRank);

    // Step A.1: Record — notify remote data ready (TxPrepare + TxData)
    LINK sendTargetLink;
    CHK_RET(GetSendTargetLink(loadedRank, sendTargetLink));
    DeviceMem inCommMem = algResResp_->cclInputMem.range(kernelOffset, loadedSize);
    CHK_RET(sendTargetLink->TxPrepare(algResResp_->slaveStreams[streamIdx]));
    CHK_RET(sendTargetLink->TxData(
        UserMemType::OUTPUT_MEM, phaseOffset, inCommMem.ptr(), loadedSize, algResResp_->slaveStreams[streamIdx]));

    sendLoadedSize_[streamIdx] = 0;
    pendingSendCount--;

    // Step A.2: D2D next slice to OTHER half (only if same rank)
    if (!sendDataSlicesBySendStream_[streamIdx].empty()) {
        SendRecvSlice& nextSlice = sendDataSlicesBySendStream_[streamIdx].front();
        if (nextSlice.remoteRank == loadedRank) {
            u32 nextPhase = 1 - curPhase;
            u64 d2dOffset = bufferSliceSize_ * (streamIdx * 2 + nextPhase);
            DeviceMem d2dCommMem = algResResp_->cclInputMem.range(d2dOffset, nextSlice.size);
            DeviceMem inMem(nextSlice.addr, nextSlice.size);
            HCCL_INFO(
                "[RunTasks] SendStream[%u] load next to phase[%u] offset[%llu] size[%llu]", streamIdx, nextPhase,
                d2dOffset, nextSlice.size);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, d2dCommMem, inMem, algResResp_->slaveStreams[streamIdx]));
            sendLoadedSize_[streamIdx] = nextSlice.size;
            sendLoadedRemoteRank_[streamIdx] = nextSlice.remoteRank;
            sendCurPhase_[streamIdx] = nextPhase;
            sendDataSlicesBySendStream_[streamIdx].pop_front();
            pendingSendCount++;
            if (sendDataSlicesBySendStream_[streamIdx].empty()) {
                nonEmptySendStream--;
                HCCL_INFO("[RunTasks] nonEmptySendStream[%u]", nonEmptySendStream);
            }
        }
    }

    // Step A.3: Wait for remote ack (TxDone)
    CHK_RET(sendTargetLink->TxDone(algResResp_->slaveStreams[streamIdx]));
    return HCCL_SUCCESS;
}

HcclResult
CollBatchSendRecvGroupExecutor::ProcessNewRankSendSlice(u32 streamIdx, u32& pendingSendCount, u32& nonEmptySendStream)
{
    SendRecvSlice& firstSlice = sendDataSlicesBySendStream_[streamIdx].front();
    u32 newRank = firstSlice.remoteRank;

    // Step B.1: D2D first slice to half A (phase 0)
    u64 offsetA = bufferSliceSize_ * (streamIdx * 2 + 0);
    DeviceMem commMemA = algResResp_->cclInputMem.range(offsetA, firstSlice.size);
    DeviceMem inMem(firstSlice.addr, firstSlice.size);
    HCCL_INFO(
        "[RunTasks] SendStream[%u] preload rank[%u] to A offset[%llu] size[%llu]", streamIdx, newRank, offsetA,
        firstSlice.size);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, commMemA, inMem, algResResp_->slaveStreams[streamIdx]));

    sendDataSlicesBySendStream_[streamIdx].pop_front();

    // Step B.2: Record from A
    LINK sendTargetLink;
    CHK_RET(GetSendTargetLink(newRank, sendTargetLink));
    CHK_RET(sendTargetLink->TxPrepare(algResResp_->slaveStreams[streamIdx]));
    CHK_RET(sendTargetLink->TxData(
        UserMemType::OUTPUT_MEM, 0, commMemA.ptr(), firstSlice.size, algResResp_->slaveStreams[streamIdx]));

    // Step B.3: D2D next to B (only if same rank)
    sendLoadedSize_[streamIdx] = 0;
    sendLoadedRemoteRank_[streamIdx] = newRank;
    sendCurPhase_[streamIdx] = 0;
    if (!sendDataSlicesBySendStream_[streamIdx].empty()) {
        SendRecvSlice& nextSlice = sendDataSlicesBySendStream_[streamIdx].front();
        if (nextSlice.remoteRank == newRank) {
            u64 offsetB = bufferSliceSize_ * (streamIdx * 2 + 1);
            DeviceMem commMemB = algResResp_->cclInputMem.range(offsetB, nextSlice.size);
            DeviceMem inMemB(nextSlice.addr, nextSlice.size);
            HCCL_INFO(
                "[RunTasks] SendStream[%u] load next to B offset[%llu] size[%llu]", streamIdx, offsetB, nextSlice.size);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, commMemB, inMemB, algResResp_->slaveStreams[streamIdx]));
            sendLoadedSize_[streamIdx] = nextSlice.size;
            sendLoadedRemoteRank_[streamIdx] = nextSlice.remoteRank;
            sendCurPhase_[streamIdx] = 1;
            sendDataSlicesBySendStream_[streamIdx].pop_front();
            pendingSendCount++;
            if (sendDataSlicesBySendStream_[streamIdx].empty()) {
                nonEmptySendStream--;
                HCCL_INFO("[RunTasks] nonEmptySendStream[%u]", nonEmptySendStream);
            }
        }
    } else {
        nonEmptySendStream--;
        HCCL_INFO("[RunTasks] nonEmptySendStream[%u]", nonEmptySendStream);
    }

    // Step B.4: Wait for remote ack (TxDone)
    CHK_RET(sendTargetLink->TxDone(algResResp_->slaveStreams[streamIdx]));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::ProcessRecvSlice(u32 streamIdx, u32& nonEmptyRecvStream)
{
    SendRecvSlice& slice = recvDataSlicesByRecvStream_[streamIdx].front();

    // Reset phase to 0 when encountering a new rank
    if (slice.remoteRank != recvCurRemoteRank_[streamIdx]) {
        recvCurPhase_[streamIdx] = 0;
        recvCurRemoteRank_[streamIdx] = slice.remoteRank;
    }

    u32 curPhase = recvCurPhase_[streamIdx];
    u64 offset = bufferSliceSize_ * (streamIdx * 2 + curPhase);
    u64 phaseOffset = bufferSliceSize_ * (curPhase + (topoAttr_.userRank % sendStreamNum_) * 2);
    HCCL_INFO(
        "[RunTasks] RecvStream[%u] phase[%u] offset[%llu] phaseOffset[%llu] size[%llu] rank[%u]", streamIdx, curPhase,
        offset, phaseOffset, slice.size, slice.remoteRank);

    LINK recvTargetLink;
    CHK_RET(GetRecvTargetLink(slice.remoteRank, recvTargetLink));

    CHK_RET(recvTargetLink->RxPrepare(algResResp_->slaveStreams[streamIdx + sendStreamNum_]));
    DeviceMem outMem(slice.addr, slice.size);
    CHK_RET(recvTargetLink->RxData(
        UserMemType::INPUT_MEM, phaseOffset, outMem.ptr(), slice.size,
        algResResp_->slaveStreams[streamIdx + sendStreamNum_]));
    HCCL_INFO("[RunTasks] RecvStream[%u] direct, outMem ptr[%p], size[%llu]", streamIdx, outMem.ptr(), outMem.size());
    CHK_RET(recvTargetLink->RxDone(algResResp_->slaveStreams[streamIdx + sendStreamNum_]));

    // Toggle phase for next slice of same rank
    recvCurPhase_[streamIdx] = 1 - curPhase;
    recvDataSlicesByRecvStream_[streamIdx].pop_front();
    if (recvDataSlicesByRecvStream_[streamIdx].empty()) {
        nonEmptyRecvStream--;
        HCCL_INFO("[RunTasks] nonEmptyRecvStream[%u]", nonEmptyRecvStream);
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::RunTasks(OpParam& param)
{
    u32 nonEmptySendStream = 0;
    u32 nonEmptyRecvStream = 0;
    CHK_RET(CalcStreamTaskStatus(nonEmptySendStream, nonEmptyRecvStream));

    CHK_RET(MainPostSubWait(param.stream));

    // Initialize ping-pong state (SDMA only; RDMA不做ping-pong)
    sendCurPhase_.resize(sendStreamNum_, 0);
    sendLoadedSize_.resize(sendStreamNum_, 0);
    sendLoadedRemoteRank_.resize(sendStreamNum_, 0);
    recvCurPhase_.resize(recvStreamNum_, 0);
    recvCurRemoteRank_.resize(recvStreamNum_, 0);

    u32 pendingSendCount = 0;

    while (pendingSendCount > 0 || nonEmptySendStream > 0 || nonEmptyRecvStream > 0 || !rdmaSendSlices_.empty()
           || !rdmaRecvSlices_.empty()) {
        HCCL_INFO(
            "[RunTasks] pending[%u] sendStream[%u] recvStream[%u] rdmaSend[%zu] rdmaRecv[%zu]", pendingSendCount,
            nonEmptySendStream, nonEmptyRecvStream, rdmaSendSlices_.size(), rdmaRecvSlices_.size());

        for (u32 i = 0; i < sendStreamNum_; i++) {
            if (sendLoadedSize_[i] > 0) {
                CHK_RET(ProcessPreloadedSendSlice(i, pendingSendCount, nonEmptySendStream));
            } else if (!sendDataSlicesBySendStream_[i].empty()) {
                CHK_RET(ProcessNewRankSendSlice(i, pendingSendCount, nonEmptySendStream));
            }
        }

        for (u32 i = 0; i < recvStreamNum_; i++) {
            if (recvDataSlicesByRecvStream_[i].empty()) {
                continue;
            }
            // SDMA任务：走ping-pong
            CHK_RET(ProcessRecvSlice(i, nonEmptyRecvStream));
        }

        if (!rdmaSendSlices_.empty()) {
            CHK_RET(ProcessRdmaSendSlice());
        }
        if (!rdmaRecvSlices_.empty()) {
            CHK_RET(ProcessRdmaRecvSlice());
        }

        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }
    CHK_RET(MainWaitSubPost(param.stream));
    CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::ProcessRdmaSendSlice()
{
    SendRecvSlice& slice = rdmaSendSlices_.front();
    // RDMA send使用CCLOut A半区(单流，整半区作为单slot)：send scratch offset = 0
    const u64 sendScratchOffset = 0;
    DeviceMem sendScratchMem = algResResp_->cclOutputMem.range(sendScratchOffset, slice.size);
    DeviceMem inMem(slice.addr, slice.size);
    HCCL_INFO(
        "[RunTasks] RdmaSend D2D user[%p] -> CCLOut_A[%p] offset[%llu] size[%llu] rank[%u]", inMem.ptr(),
        sendScratchMem.ptr(), sendScratchOffset, slice.size, slice.remoteRank);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, sendScratchMem, inMem, algResResp_->slaveStreams[RdmaSendStreamIdx()]));

    // Tx: 本地CCLOut_A(单slot) -> 远端CCLOut_B(单slot)。
    // 远端RDMA recv为单流，其recv scratch offset = rdmaDataBlockSize_(B半区基址)。
    const u64 remoteDstOffset = rdmaDataBlockSize_;
    LINK sendTargetLink;
    CHK_RET(GetSendTargetLink(slice.remoteRank, sendTargetLink));
    CHK_RET(sendTargetLink->TxPrepare(algResResp_->slaveStreams[RdmaSendStreamIdx()]));
    CHK_RET(sendTargetLink->TxData(
        UserMemType::OUTPUT_MEM, remoteDstOffset, sendScratchMem.ptr(), slice.size,
        algResResp_->slaveStreams[RdmaSendStreamIdx()]));
    CHK_RET(sendTargetLink->TxDone(algResResp_->slaveStreams[RdmaSendStreamIdx()]));

    rdmaSendSlices_.pop_front();
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::ProcessRdmaRecvSlice()
{
    SendRecvSlice& slice = rdmaRecvSlices_.front();
    // RDMA recv使用CCLOut B半区(单流，整半区作为单slot)：recv scratch offset = rdmaDataBlockSize_(B半区基址)。
    // 与发送方写入的远端offset一致。
    const u64 recvScratchOffset = rdmaDataBlockSize_;
    DeviceMem recvScratchMem = algResResp_->cclOutputMem.range(recvScratchOffset, slice.size);

    LINK recvTargetLink;
    CHK_RET(GetRecvTargetLink(slice.remoteRank, recvTargetLink));
    CHK_RET(recvTargetLink->RxPrepare(algResResp_->slaveStreams[RdmaRecvStreamIdx()]));
    CHK_RET(recvTargetLink->RxData(
        UserMemType::OUTPUT_MEM, recvScratchOffset, recvScratchMem.ptr(), slice.size,
        algResResp_->slaveStreams[RdmaRecvStreamIdx()]));
    CHK_RET(recvTargetLink->RxDone(algResResp_->slaveStreams[RdmaRecvStreamIdx()]));

    // D2D: CCLOut_B -> user output
    DeviceMem outMem(slice.addr, slice.size);
    HCCL_INFO(
        "[RunTasks] RdmaRecv D2D CCLOut_B[%p] offset[%llu] -> user[%p] size[%llu] rank[%u]", recvScratchMem.ptr(),
        recvScratchOffset, outMem.ptr(), slice.size, slice.remoteRank);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outMem, recvScratchMem, algResResp_->slaveStreams[RdmaRecvStreamIdx()]));

    rdmaRecvSlices_.pop_front();
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::SetNormalModeIfDeviceDirect()
{
    for (const auto& q : sendDataSlicesBySendStream_) {
        for (const auto& slice : q) {
            LINK targetLink;
            CHK_RET(GetSendTargetLink(slice.remoteRank, targetLink));
            if (targetLink->GetTransportType() == TransportType::TRANS_TYPE_DEVICE_DIRECT) {
                CHK_RET(SetNormalMode(dispatcher_));
                HCCL_INFO("[CollBatchSendRecvGroupExecutor]Send Set dispatcher NormalMode");
                return HCCL_SUCCESS;
            }
        }
    }

    for (const auto& q : recvDataSlicesByRecvStream_) {
        for (const auto& slice : q) {
            LINK targetLink;
            CHK_RET(GetRecvTargetLink(slice.remoteRank, targetLink));
            if (targetLink->GetTransportType() == TransportType::TRANS_TYPE_DEVICE_DIRECT) {
                CHK_RET(SetNormalMode(dispatcher_));
                HCCL_INFO("[CollBatchSendRecvGroupExecutor]Recv Set NormalMode dispatcher");
                return HCCL_SUCCESS;
            }
        }
    }

    for (const auto& slice : rdmaSendSlices_) {
        LINK targetLink;
        CHK_RET(GetSendTargetLink(slice.remoteRank, targetLink));
        if (targetLink->GetTransportType() == TransportType::TRANS_TYPE_DEVICE_DIRECT) {
            CHK_RET(SetNormalMode(dispatcher_));
            HCCL_INFO("[CollBatchSendRecvGroupExecutor]RdmaSend Set NormalMode dispatcher");
            return HCCL_SUCCESS;
        }
    }

    for (const auto& slice : rdmaRecvSlices_) {
        LINK targetLink;
        CHK_RET(GetRecvTargetLink(slice.remoteRank, targetLink));
        if (targetLink->GetTransportType() == TransportType::TRANS_TYPE_DEVICE_DIRECT) {
            CHK_RET(SetNormalMode(dispatcher_));
            HCCL_INFO("[CollBatchSendRecvGroupExecutor]RdmaRecv Set NormalMode dispatcher");
            return HCCL_SUCCESS;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::RunLoop(OpParam& param)
{
    if (static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())) {
        auto meta = HcclOpMetaInfo::GetOneForBatchSendRecv();
        CHK_RET(InitTask(dispatcher_, param.stream, meta.isEnableCache, meta.GetCacheKey()));
        // 多流子图前后需加空拷贝
        CHK_RET(AlgTemplateBase::ExecEmptyTask(
            algResResp_->cclInputMem, algResResp_->cclOutputMem, param.stream, dispatcher_));
    }
    CHK_RET(SetNormalModeIfDeviceDirect());
    CHK_RET(RunTasks(param));
    if (static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())) {
        // 多流子图前后需加空拷贝
        CHK_RET(AlgTemplateBase::ExecEmptyTask(
            algResResp_->cclInputMem, algResResp_->cclOutputMem, param.stream, dispatcher_));
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
        HCCL_INFO("LaunchTaskExtend!");
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::CalcSendSlices()
{
    // SDMA slice按remoteRank % streamNum分发到sendDataSlicesBySendStream_(CCLIn, ping-pong)；
    // RDMA slice按rank分组到rdmaByRank，随后按对称场景规则排序到rdmaSendSlices_(CCLOut A半区, 无ping-pong)。
    sendDataSlicesBySendStream_.resize(sendStreamNum_);
    std::map<u32, std::deque<SendRecvSlice>> rdmaByRank;
    for (u32 i = 0; i < sendStreamNum_; i++) {
        const auto& sendQueueInner = sendQueueBySendstream_[i];
        for (u32 j = 0; j < sendQueueInner.size(); j++) {
            HcclSendRecvItem* sendRecvItem = sendQueueInner[j];
            u32 unitSize = SIZE_TABLE[sendRecvItem->dataType];
            bool isRdma = IsRemoteRankRdma(sendRecvItem->remoteRank);
            u64 maxCountPerLoop = isRdma ? (rdmaDataBlockSize_ / unitSize) : CalcSendLoopMaxCount(unitSize);
            while (sendRecvItem->count > 0) {
                u8* curInputPtr = static_cast<u8*>(sendRecvItem->buf);
                CHK_PTR_NULL(curInputPtr);
                u64 curCount = (sendRecvItem->count > maxCountPerLoop) ? maxCountPerLoop : sendRecvItem->count;
                u64 curSize = curCount * unitSize;
                SendRecvSlice slice(curInputPtr, curSize, sendRecvItem->remoteRank, isRdma);
                if (isRdma) {
                    rdmaByRank[sendRecvItem->remoteRank].push_back(slice);
                } else {
                    sendDataSlicesBySendStream_[i].push_back(slice);
                }
                sendRecvItem->count -= curCount;
                sendRecvItem->buf = static_cast<u8*>(sendRecvItem->buf) + curSize;
            }
        }
    }
    // RDMA按对称场景规则排序(send前向递增，跳过本pod与无任务对端)
    OrderRdmaSlices(true, rdmaByRank, rdmaSendSlices_);

    for (u32 i = 0; i < sendStreamNum_; i++) {
        const auto& sendQueueInner = sendDataSlicesBySendStream_[i];
        for (const auto& slice : sendQueueInner) {
            HCCL_INFO(
                "[CalcSendSlices] sendstream[%u] addr[%p] size[%llu] rank[%u] isRdma[%d]", i, slice.addr, slice.size,
                slice.remoteRank, slice.isRdma);
        }
    }
    for (const auto& slice : rdmaSendSlices_) {
        HCCL_INFO("[CalcSendSlices] rdmaSend addr[%p] size[%llu] rank[%u]", slice.addr, slice.size, slice.remoteRank);
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvGroupExecutor::CalcRecvSlices()
{
    // SDMA slice按remoteRank % streamNum分发到recvDataSlicesByRecvStream_(CCLIn, ping-pong)；
    // RDMA slice按rank分组到rdmaByRank，随后按对称场景规则排序到rdmaRecvSlices_(CCLOut B半区, 无ping-pong)。
    recvDataSlicesByRecvStream_.resize(recvStreamNum_);
    std::map<u32, std::deque<SendRecvSlice>> rdmaByRank;
    for (u32 i = 0; i < recvStreamNum_; i++) {
        const auto& recvQueueInner = recvQueueByRecvstream_[i];
        for (u32 j = 0; j < recvQueueInner.size(); j++) {
            HcclSendRecvItem* sendRecvItem = recvQueueInner[j];
            u32 unitSize = SIZE_TABLE[sendRecvItem->dataType];
            bool isRdma = IsRemoteRankRdma(sendRecvItem->remoteRank);
            u64 maxCountPerLoop = isRdma ? (rdmaDataBlockSize_ / unitSize) : CalcRecvLoopMaxCount(unitSize);
            while (sendRecvItem->count > 0) {
                u8* curOutputPtr = static_cast<u8*>(sendRecvItem->buf);
                CHK_PTR_NULL(curOutputPtr);
                u64 curCount = (sendRecvItem->count > maxCountPerLoop) ? maxCountPerLoop : sendRecvItem->count;
                u64 curSize = curCount * unitSize;
                SendRecvSlice slice(curOutputPtr, curSize, sendRecvItem->remoteRank, isRdma);
                if (isRdma) {
                    rdmaByRank[sendRecvItem->remoteRank].push_back(slice);
                } else {
                    recvDataSlicesByRecvStream_[i].push_back(slice);
                }
                sendRecvItem->count -= curCount;
                sendRecvItem->buf = static_cast<u8*>(sendRecvItem->buf) + curSize;
            }
        }
    }
    // RDMA按对称场景规则排序(recv后向递减，跳过本pod与无任务对端)
    OrderRdmaSlices(false, rdmaByRank, rdmaRecvSlices_);

    for (u32 i = 0; i < recvStreamNum_; i++) {
        const auto& recvQueueInner = recvDataSlicesByRecvStream_[i];
        for (const auto& slice : recvQueueInner) {
            HCCL_INFO(
                "[CalcRecvSlices] recvstream[%u] addr[%p] size[%llu] rank[%u] isRdma[%d]", i, slice.addr, slice.size,
                slice.remoteRank, slice.isRdma);
        }
    }
    for (const auto& slice : rdmaRecvSlices_) {
        HCCL_INFO("[CalcRecvSlices] rdmaRecv addr[%p] size[%llu] rank[%u]", slice.addr, slice.size, slice.remoteRank);
    }
    return HCCL_SUCCESS;
}

u32 CollBatchSendRecvGroupExecutor::GetNextDstRank(u32& curDstRank) const
{
    // 对称场景send方向：沿rank id递增环绕遍历，跳过本pod。移植自alltoallv_direct_fullmesh。
    if (curDstRank >= topoAttr_.userRankSize) {
        curDstRank = curDstRank % topoAttr_.userRankSize;
    }
    if (curDstRank == podStartRank_) {
        curDstRank += devNumInlocalPod_;
    }
    curDstRank = curDstRank % topoAttr_.userRankSize;
    return curDstRank++;
}

u32 CollBatchSendRecvGroupExecutor::GetPreSrcRank(u32& curSrcRank) const
{
    // 对称场景recv方向：沿rank id递减环绕遍历，跳过本pod。移植自alltoallv_direct_fullmesh。
    if (curSrcRank == podStartRank_ + devNumInlocalPod_ - 1) {
        curSrcRank = (curSrcRank + topoAttr_.userRankSize - devNumInlocalPod_) % topoAttr_.userRankSize;
    }
    if (curSrcRank == 0) {
        curSrcRank = topoAttr_.userRankSize - 1;
        return 0;
    }
    return curSrcRank--;
}

void CollBatchSendRecvGroupExecutor::OrderRdmaSlices(
    bool isSend, const std::map<u32, std::deque<SendRecvSlice>>& byRank, std::deque<SendRecvSlice>& out)
{
    // 跨pod对端总数 = userRankSize - devNumInlocalPod。遍历每个候选rank一次。
    u32 totalRdmaRankNum = topoAttr_.userRankSize - devNumInlocalPod_;
    u32 curRank = INVALID_VALUE_RANKID;
    if (isSuPodAsym_) {
        // 非对称场景：send/recv使用相同遍历顺序(均前向递增GetNextDstRank)。
        // 起点为本pod之外的第一个rank(从0开始扫描，取首个不在[podStartRank_, podEndRank_]内的rank)。
        for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
            if (i < podStartRank_ || i > podEndRank_) {
                curRank = i;
                break;
            }
        }
        HCCL_INFO(
            "[OrderRdmaSlices] %s asym startRank[%u] totalRdmaRankNum[%u]", isSend ? "send" : "recv", curRank,
            totalRdmaRankNum);
        for (u32 i = 0; i < totalRdmaRankNum; i++) {
            u32 rank = GetNextDstRank(curRank);
            auto it = byRank.find(rank);
            if (it == byRank.end()) {
                HCCL_INFO("[OrderRdmaSlices] %s asym skip rank[%u] (no task)", isSend ? "send" : "recv", rank);
                continue;
            }
            for (const auto& s : it->second) {
                out.push_back(s);
            }
            HCCL_INFO(
                "[OrderRdmaSlices] %s asym append rank[%u] sliceNum[%zu]", isSend ? "send" : "recv", rank,
                it->second.size());
        }
    } else {
        // 对称场景：send前向递增(GetNextDstRank)，recv后向递减(GetPreSrcRank)。
        // send起点取"下一个pod中相同pod内位置的rank"；recv起点取"上一个pod中相同pod内位置的rank"。
        curRank = isSend ? (topoAttr_.userRank + devNumInlocalPod_) % topoAttr_.userRankSize :
                           (topoAttr_.userRank + topoAttr_.userRankSize - devNumInlocalPod_) % topoAttr_.userRankSize;
        HCCL_INFO(
            "[OrderRdmaSlices] %s sym startRank[%u] totalRdmaRankNum[%u]", isSend ? "send" : "recv", curRank,
            totalRdmaRankNum);
        for (u32 i = 0; i < totalRdmaRankNum; i++) {
            u32 rank = isSend ? GetNextDstRank(curRank) : GetPreSrcRank(curRank);
            auto it = byRank.find(rank);
            if (it == byRank.end()) {
                HCCL_INFO("[OrderRdmaSlices] %s sym skip rank[%u] (no task)", isSend ? "send" : "recv", rank);
                continue;
            }
            for (const auto& s : it->second) {
                out.push_back(s);
            }
            HCCL_INFO(
                "[OrderRdmaSlices] %s sym append rank[%u] sliceNum[%zu]", isSend ? "send" : "recv", rank,
                it->second.size());
        }
    }
}

u64 CollBatchSendRecvGroupExecutor::CalcSendLoopMaxCount(const u32 unitSize) const
{
    // 中转内存单次最多能够接受的input count
    u64 maxCountPerLoop = bufferSliceSize_ / unitSize;
    HCCL_INFO(
        "[CollBatchSendRecvGroupExecutor][CalcSendLoopMaxCount]"
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.",
        maxCountPerLoop);
    return maxCountPerLoop;
}

u64 CollBatchSendRecvGroupExecutor::CalcRecvLoopMaxCount(const u32 unitSize) const
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = bufferSliceSize_ / unitSize;
    HCCL_INFO(
        "[CollBatchSendRecvGroupExecutor][CalcRecvLoopMaxCount]"
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.",
        maxCountPerLoop);
    return maxCountPerLoop;
}

HcclResult CollBatchSendRecvGroupExecutor::CalcStreamNum(u32& streamNum)
{
    if (topoAttr_.userRankSize == 1) {
        sendStreamNum_ = 0;
        recvStreamNum_ = 0;
        streamNum = 0;
        HCCL_INFO("[CollBatchSendRecvGroupExecutor] Only one rank, do not need substream, streamNum[%u]", streamNum);
        return HCCL_SUCCESS;
    }
    sendStreamNum_ = GROUP_MAX_CONCURRENT;
    recvStreamNum_ = GROUP_MAX_CONCURRENT;
    // SDMA占sendStreamNum_+recvStreamNum_条从流；RDMA单独占RDMA_STREAM_NUM条(1 send + 1 recv)。
    streamNum = sendStreamNum_ + recvStreamNum_ + RDMA_STREAM_NUM;
    HCCL_INFO("[CollBatchSendRecvGroupExecutor][CalcStreamNum] tag_[%s], streamNum[%u].", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

REGISTER_EXEC("BatchSendRecvGroup", BatchSendRecvGroupExecutor, CollBatchSendRecvGroupExecutor);
} // namespace hccl
