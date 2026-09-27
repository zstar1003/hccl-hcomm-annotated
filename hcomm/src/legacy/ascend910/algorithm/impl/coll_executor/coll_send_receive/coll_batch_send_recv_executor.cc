/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_batch_send_recv_executor.h"

namespace hccl {
constexpr u32 RANKSIZE_TWO = 2;

CollBatchSendRecvExecutor::CollBatchSendRecvExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollCommExecutor(dispatcher, topoMatcher)
{}

void CollBatchSendRecvExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    HcclSendRecvItem* itemPtr = param.BatchSendRecvDataDes.sendRecvItemsPtr;
    u32 itemNum = param.BatchSendRecvDataDes.itemNum;
    if (itemPtr == nullptr) {
        HCCL_ERROR("[CollBatchSendRecvExecutor][ParseParam] sendRecvInfo is nullptr.");
    }
    commTargetUserRankSet_.clear();
    for (u32 i = 0; i < itemNum; i++) {
        commTargetUserRankSet_.insert((itemPtr + i)->remoteRank);
        HCCL_INFO(
            "[CollBatchSendRecvExecutor][ParseParam] insert remoteUserRank[%u] to Set ", (itemPtr + i)->remoteRank);
    }
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
}

HcclResult CollBatchSendRecvExecutor::CalcIncreLinkRequest(
    const OpParam& param, std::set<u32>& ranksLinked, AlgResourceRequest& resourceRequest, bool& needIncreLink)
{
    needIncreLink = false;
    (void)ParseParam(param);
    for (auto& remoteRank : commTargetUserRankSet_) {
        if (ranksLinked.find(remoteRank) == ranksLinked.end()) {
            needIncreLink = true;
            ranksLinked.insert(remoteRank);
            HCCL_INFO(
                "[CollBatchSendRecvExecutor][CalcIncreLinkRequest] Start insert remoteUserRank[%u] to "
                "ranksLinked Set.",
                remoteRank);
        }
    }
    CHK_PRT_RET(
        !needIncreLink,
        HCCL_INFO("[CollBatchSendRecvExecutor][CalcIncreLinkRequest] It's "
                  "unnecessary to incre alloc link."),
        HCCL_SUCCESS);

    u64 scratchMemSize = 0U;
    u32 streamNum = 0U;
    u32 notifyNum = 0U;
    u64 aivBufferRequest = 0U;

    std::vector<LevelNSubCommTransport> opTransport{
        std::vector<LevelNSubCommTransport>(static_cast<u32>(COMM_LEVEL_RESERVED))};
    CHK_RET(CalcCommInfo(opTransport));
    CHK_RET(BuildResourceRequest(scratchMemSize, streamNum, notifyNum, aivBufferRequest, opTransport, resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::GetPairWiseList(HcclSendRecvItem* sendRecvInfo, u32 itemNum)
{
    HCCL_INFO("[CollBatchSendRecvExecutor][GetPairWiseList] Start sort the batchSendRecv tasklist.");
    CHK_PTR_NULL(sendRecvInfo);

    for (u32 i = 0; i < itemNum; i++) {
        HCCL_INFO(
            "[CollBatchSendRecvExecutor][GetPairWiseList] index is %u, itemNum is %u, localRankID is %u, remoteRank is "
            "%u, "
            "sendRecvType is %u, rankSize is %u.",
            i, itemNum, topoAttr_.userRank, sendRecvInfo->remoteRank, static_cast<u32>(sendRecvInfo->sendRecvType),
            topoAttr_.userRankSize);
        CHK_PTR_NULL(sendRecvInfo->buf);

        if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_SEND) {
            sendDeque_.push_back(sendRecvInfo);
        } else if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_RECV) {
            recvDeque_.push_back(sendRecvInfo);
        } else {
            HCCL_ERROR(
                "[CollBatchSendRecvExecutor][GetPairWiseList] sendRecvType wrong sendrecvType is %d, "
                "rankID is %u, remoteRank is %u.",
                sendRecvInfo->sendRecvType, topoAttr_.userRank, sendRecvInfo->remoteRank);
            return HCCL_E_PARA;
        }
        sendRecvInfo++;
    }

    /* 此处的排序逻辑(pair-wise算法):
        1.sendDeque元素顺序是:先放remoteRank号小于等于root rank的第一个任务，依次减小(循环索引)直至放完
        2.recvDeque元素顺序是:先放remoteRank号大于等于root rank的第一个任务，依次增大(循环索引)直至放完
    */
    auto sendCompare = [this](HcclSendRecvItem* a, HcclSendRecvItem* b) {
        u32 aFlag = (a->remoteRank <= topoAttr_.userRank) ? (a->remoteRank + topoAttr_.userRankSize) : a->remoteRank;
        u32 bFlag = (b->remoteRank <= topoAttr_.userRank) ? (b->remoteRank + topoAttr_.userRankSize) : b->remoteRank;
        return aFlag > bFlag;
    };

    auto recvCompare = [this](HcclSendRecvItem* a, HcclSendRecvItem* b) {
        u32 aFlag = (a->remoteRank < topoAttr_.userRank) ? (a->remoteRank + topoAttr_.userRankSize) : a->remoteRank;
        u32 bFlag = (b->remoteRank < topoAttr_.userRank) ? (b->remoteRank + topoAttr_.userRankSize) : b->remoteRank;
        return aFlag < bFlag;
    };

    std::stable_sort(sendDeque_.begin(), sendDeque_.end(), sendCompare);
    std::stable_sort(recvDeque_.begin(), recvDeque_.end(), recvCompare);

    while ((!sendDeque_.empty() && sendDeque_.front()->remoteRank == topoAttr_.userRank)
           && (!recvDeque_.empty() && recvDeque_.front()->remoteRank == topoAttr_.userRank)) {
        sendToSelfDeque_.push_back(sendDeque_.front());
        recvFromSelfDeque_.push_back(recvDeque_.front());
        sendDeque_.pop_front();
        recvDeque_.pop_front();
    }
    // 如果自发自收任务没有完全匹配
    if ((!sendDeque_.empty() && sendDeque_.front()->remoteRank == topoAttr_.userRank)
        || (!recvDeque_.empty() && recvDeque_.front()->remoteRank == topoAttr_.userRank)) {
        HCCL_ERROR("[CollBatchSendRecvExecutor] SendTask and Recv Task to rank itself do not match,"
                   "please check the task list.");
        return HCCL_E_PARA;
    }
    HCCL_INFO("[CollBatchSendRecvExecutor][GetPairWiseList] End sort the batchSendRecv tasklist.");
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::ProcessSelfSendRecvTasks(Stream& stream)
{
    while (!sendToSelfDeque_.empty() && !recvFromSelfDeque_.empty()) {
        if (sendToSelfDeque_.front()->count == recvFromSelfDeque_.front()->count
            && sendToSelfDeque_.front()->dataType == recvFromSelfDeque_.front()->dataType) {
            u64 dataSize = sendToSelfDeque_.front()->count * SIZE_TABLE[sendToSelfDeque_.front()->dataType];

            DeviceMem inUserMem = DeviceMem::create(static_cast<u8*>(sendToSelfDeque_.front()->buf), dataSize);
            DeviceMem outUserMem = DeviceMem::create(static_cast<u8*>(recvFromSelfDeque_.front()->buf), dataSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outUserMem, inUserMem, stream));
            sendToSelfDeque_.pop_front();
            recvFromSelfDeque_.pop_front();
        } else {
            HCCL_ERROR("[HcclBatchSendRecv] Send task and recv task to self : count or dataType do not equal, please "
                       "check the task list.");
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algResource)
{
    HcclUs startut = TIME_NOW();
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollBatchSendRecvExecutor] batchsendrecv starts.");

    algResResp_ = &algResource;
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_SIZE_TWO));
    CHK_RET(GetPairWiseList(param.BatchSendRecvDataDes.sendRecvItemsPtr, param.BatchSendRecvDataDes.itemNum));
    CHK_RET(ProcessSelfSendRecvTasks(param.stream));
    if (topoAttr_.userRankSize == 1) {
        HCCL_INFO(
            "tag[%s] BatchSendRecv Executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
            DURATION_US(TIME_NOW() - startut));
        return HCCL_SUCCESS;
    }
    CHK_RET(CalcSendSlices(algResource));
    CHK_RET(CalcRecvSlices(algResource));

    HCCL_INFO("[CollBatchSendRecvExecutor][Orchestrate] aicpuUnfoldMode %d", param.aicpuUnfoldMode);
    if (param.aicpuUnfoldMode) {
        CHK_RET(RunLoopInAicpuUnfoldMode(param));
    } else {
        CHK_RET(RunLoopInHostUnfoldMode(param));
    }
    HCCL_INFO(
        "tag[%s] BatchSendRecv Executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::GetSendTargetLink(u32 remoteUserRank, LINK& targetLink)
{
    u32 commIndex = 0;
    HCCL_INFO(
        "[CollBatchSendRecvExecutor][GetSendTargetLink] remoteUserRank[%u], localUserRank_[%u].", remoteUserRank,
        topoAttr_.userRank);
    if (remoteUserRank < topoAttr_.userRank) {
        HCCL_INFO("[CollBatchSendRecvExecutor][GetSendTargetLink] CommIndex is 0.");
        commIndex = COMM_INDEX_0;
    } else if (remoteUserRank > topoAttr_.userRank) {
        HCCL_INFO("[CollBatchSendRecvExecutor][GetSendTargetLink] CommIndex is 1.");
        commIndex = COMM_INDEX_1;
    } else {
        HCCL_ERROR("[CollBatchSendRecvExecutor][GetSendTargetLink] CommIndex doesn't match.");
        return HCCL_E_PARA;
    }
    CHK_RET(GetTransport(commIndex, remoteUserRank, targetLink));
    CHK_SMART_PTR_NULL(targetLink);

    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::GetRecvTargetLink(u32 remoteUserRank, LINK& targetLink)
{
    u32 commIndex = 0;
    HCCL_INFO(
        "[CollBatchSendRecvExecutor][GetRecvTargetLink] remoteUserRank[%u], localUserRank_[%u].", remoteUserRank,
        topoAttr_.userRank);
    if (remoteUserRank > topoAttr_.userRank) {
        HCCL_INFO("[CollBatchSendRecvExecutor][GetRecvTargetLink] CommIndex is 0.");
        commIndex = COMM_INDEX_0;
    } else if (remoteUserRank < topoAttr_.userRank) {
        HCCL_INFO("[CollBatchSendRecvExecutor][GetRecvTargetLink] CommIndex is 1.");
        commIndex = COMM_INDEX_1;
    } else {
        HCCL_ERROR("[CollBatchSendRecvExecutor][GetRecvTargetLink] CommIndex doesn't match.");
        return HCCL_E_PARA;
    }
    CHK_RET(GetTransport(commIndex, remoteUserRank, targetLink));
    CHK_SMART_PTR_NULL(targetLink);

    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::RunLoopInHostUnfoldMode(OpParam& param)
{
    if (static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())) {
        auto meta = HcclOpMetaInfo::GetOneForBatchSendRecv();
        CHK_RET(InitTask(dispatcher_, param.stream, meta.isEnableCache, meta.GetCacheKey()));
        // 多流子图前后需加空拷贝
        CHK_RET(AlgTemplateBase::ExecEmptyTask(
            algResResp_->cclInputMem, algResResp_->cclOutputMem, param.stream, dispatcher_));
    }
    bool isSetNormalMode = false; // 设置过一次就不需要再设置了
    for (u32 i = 0; i < sendDataSilces_.size(); ++i) {
        SendRecvSlice& slice = sendDataSilces_[i];
        LINK targetLink;
        CHK_RET(GetSendTargetLink(slice.remoteRank, targetLink));
        if (TransportType::TRANS_TYPE_DEVICE_DIRECT == targetLink->GetTransportType()) {
            CHK_RET(SetNormalMode(dispatcher_));
            isSetNormalMode = true;
            HCCL_INFO("[CollBatchSendRecvExecutor][RunLoopInHostUnfoldMode]Send Set NormalMode true");
            break;
        }
    }

    for (u32 i = 0; i < recvDataSilces_.size() && !isSetNormalMode; ++i) {
        SendRecvSlice& slice = recvDataSilces_[i];
        LINK targetLink;
        CHK_RET(GetRecvTargetLink(slice.remoteRank, targetLink));
        if (targetLink->GetTransportType() == TransportType::TRANS_TYPE_DEVICE_DIRECT) {
            CHK_RET(SetNormalMode(dispatcher_));
            HCCL_INFO("[CollBatchSendRecvExecutor][RunLoopInHostUnfoldMode]Recv Set NormalMode dispatcher");
            break;
        }
    }

    CHK_RET(MainPostSubWait(param.stream, algResResp_->slaveStreams[STREAM_INDEX_0]));
    HCCL_INFO("[BatchSendRecv] Stream sync: main stream record, subStream wait.");
    while (!sendDataSilces_.empty() || !recvDataSilces_.empty()) {
        if (!sendDataSilces_.empty()) {
            CHK_RET(ProcessSendDataSlice(param.stream, false, false));
            sendDataSilces_.pop_front();
        }
        if (!recvDataSilces_.empty()) {
            CHK_RET(ProcessRecvDataSlice(algResResp_->slaveStreams[STREAM_INDEX_0], false));
            recvDataSilces_.pop_front();
        }
    }

    CHK_RET(SubPostMainWait(param.stream, algResResp_->slaveStreams[STREAM_INDEX_0]));
    HCCL_INFO("[BatchSendRecv] Stream sync: subStream record, main stream wait.");
    if (static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())) {
        // 多流子图前后需加空拷贝
        CHK_RET(AlgTemplateBase::ExecEmptyTask(
            algResResp_->cclInputMem, algResResp_->cclOutputMem, param.stream, dispatcher_));
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo)
{
    algResResp_ = &algRes;
    SubCommInfo level1CommInfo = {};
    AdjInfo nslbAdjInfo = {};
    if (Getlevel1CommRank(level1CommInfo) != HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }
    u32 localRank = level1CommInfo.localRank;
    u32 localRankSize = level1CommInfo.localRankSize;

    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    if (SelectTempAlg(level1TempAlg, localRankSize) != HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }
    if (level1TempAlg == nullptr) {
        return HCCL_SUCCESS;
    }
    CHK_RET(level1TempAlg->GetNslbAdjInfo(localRank, localRankSize, level1CommInfo.links, nslbAdjInfo));

    adjInfo.dstRankNum = nslbAdjInfo.dstRankNum;
    HCCL_INFO("[CollBatchSendRecvExecutor][GetAdjInfo] adjInfo.dstRankNum[%u].", adjInfo.dstRankNum);

    for (size_t i = 0; i < nslbAdjInfo.nsAdjInfo.size(); i++) {
        NslbDpAdjInfo dpAdjInfo = {};
        dpAdjInfo.dstLocalRankId = nslbAdjInfo.nsAdjInfo[i].dstLocalRankId;
        dpAdjInfo.phaseId = nslbAdjInfo.nsAdjInfo[i].phaseId;
        dpAdjInfo.rev = 0;
        adjInfo.nsAdjInfo.push_back(dpAdjInfo);
        HCCL_INFO(
            "[CollBatchSendRecvExecutor][GetAdjInfo] dstLocalRankId[%u], phaseId[%u].",
            nslbAdjInfo.nsAdjInfo[i].dstLocalRankId, nslbAdjInfo.nsAdjInfo[i].phaseId);
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::RunLoopInAicpuUnfoldMode(OpParam& param)
{
    CHK_RET(MainPostSubWait(param.stream, algResResp_->slaveStreams[STREAM_INDEX_0]));
    u32 loopInOnceLaunch = 0;
    // 每隔200个loop launch一次
    while (!sendDataSilces_.empty() || !recvDataSilces_.empty()) {
        if (!sendDataSilces_.empty()) {
            CHK_RET(ProcessSendDataSlice(param.stream, false, false));
            sendDataSilces_.pop_front();
        }
        if (!recvDataSilces_.empty()) {
            CHK_RET(ProcessRecvDataSlice(algResResp_->slaveStreams[STREAM_INDEX_0], false));
            recvDataSilces_.pop_front();
        }
        loopInOnceLaunch++;
        if (loopInOnceLaunch == MAX_LOOP_IN_ONCE_LAUNCH) {
            CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
            loopInOnceLaunch = 0;
            HCCL_INFO(
                "[BatchSendRecv] LaunchTaskExtend, unprocessed send slices[%u], recv slices[%u].",
                sendDataSilces_.size(), recvDataSilces_.size());
        }
    }
    CHK_RET(SubPostMainWait(param.stream, algResResp_->slaveStreams[STREAM_INDEX_0]));
    CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::MainPostSubWait(Stream& mainStream, Stream& subStream)
{
    CHK_RET(LocalNotify::Post(mainStream, dispatcher_, algResResp_->notifiesAux[STREAM_INDEX_0], PROF_STAGE_0));
    CHK_RET(LocalNotify::Wait(subStream, dispatcher_, algResResp_->notifiesAux[STREAM_INDEX_0], PROF_STAGE_0));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::SubPostMainWait(Stream& mainStream, Stream& subStream)
{
    CHK_RET(LocalNotify::Post(subStream, dispatcher_, algResResp_->notifiesMain[STREAM_INDEX_0], PROF_STAGE_0));

    CHK_RET(LocalNotify::Wait(mainStream, dispatcher_, algResResp_->notifiesMain[STREAM_INDEX_0], PROF_STAGE_0));
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::CalcSendSlices(AlgResourceResponse& algRes)
{
    while (!sendDeque_.empty()) {
        HcclSendRecvItem* sendRecvItem = sendDeque_.front();
        HCCL_INFO(
            "[CollBatchSendRecvExecutor][CalcSendSlices] tag[%s], remoteRank[%u], buf[%p], count[%llu],"
            "dataType[%s], sendRecvType[%d].",
            tag_.c_str(), sendRecvItem->remoteRank, sendRecvItem->buf, sendRecvItem->count,
            GetDataTypeEnumStr(sendRecvItem->dataType).c_str(), sendRecvItem->sendRecvType);
        u8* curInputPtr = static_cast<u8*>(sendRecvItem->buf);
        CHK_PTR_NULL(curInputPtr);
        u32 unitSize = SIZE_TABLE[sendRecvItem->dataType];
        u64 maxCountPerLoop = CalcSendLoopMaxCount(const_cast<DeviceMem&>(algRes.cclInputMem), unitSize);

        for (u64 countLeft = sendRecvItem->count, curCount = 0, curOffset = 0; countLeft > 0; countLeft -= curCount) {
            curInputPtr += curOffset;
            curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
            u64 curSize = curCount * unitSize; // 单位：字节
            sendDataSilces_.emplace_back(curInputPtr, curSize, sendRecvItem->remoteRank);
            HCCL_DEBUG(
                "[CollBatchSendRecvExecutor][CalcSendSlices] tag[%s], slice userAddr[%p], slice size[%llu].",
                tag_.c_str(), curInputPtr, curSize);
            curOffset = curSize;
        }
        sendDeque_.pop_front();
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::CalcRecvSlices(AlgResourceResponse& algRes)
{
    while (!recvDeque_.empty()) {
        HcclSendRecvItem* sendRecvItem = recvDeque_.front();
        HCCL_INFO(
            "[CollBatchSendRecvExecutor][CalcSendSlices] tag[%s], remoteRank[%u], buf[%p], count[%llu],"
            "dataType[%s], sendRecvType[%d].",
            tag_.c_str(), sendRecvItem->remoteRank, sendRecvItem->buf, sendRecvItem->count,
            GetDataTypeEnumStr(sendRecvItem->dataType).c_str(), sendRecvItem->sendRecvType);
        u8* curOutputPtr = static_cast<u8*>(sendRecvItem->buf);
        CHK_PTR_NULL(curOutputPtr);
        u32 unitSize = SIZE_TABLE[sendRecvItem->dataType];
        u64 maxCountPerLoop = CalcRecvLoopMaxCount(const_cast<DeviceMem&>(algRes.cclOutputMem), unitSize);

        for (u64 countLeft = sendRecvItem->count, curCount = 0, curOffset = 0; countLeft > 0; countLeft -= curCount) {
            curOutputPtr += curOffset;
            curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
            u64 curSize = curCount * unitSize; // 单位：字节
            recvDataSilces_.emplace_back(curOutputPtr, curSize, sendRecvItem->remoteRank);
            HCCL_DEBUG(
                "[CollBatchSendRecvExecutor][CalcRecvSlices] tag[%s], slice userAddr[%p], slice size[%llu].",
                tag_.c_str(), curOutputPtr, curSize);
            curOffset = curSize;
        }
        recvDeque_.pop_front();
    }
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::ProcessSendDataSlice(Stream& stream, bool needStreamSync, bool retryEnable)
{
    SendRecvSlice& slice = sendDataSilces_.front();
    DeviceMem inMem(slice.addr, slice.size);
    DeviceMem inCommMem = algResResp_->cclInputMem.range(0, slice.size);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCommMem, inMem, stream));
    if (needStreamSync) {
        CHK_RET(MainPostSubWait(stream, algResResp_->slaveStreams[STREAM_INDEX_0]));
    }

    ExecMem execMem;
    execMem.inputMem = inCommMem;
    HcclResult ret = SendKernelRun(stream, execMem, slice.remoteRank, retryEnable);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollBatchSendRecvExecutor][ProcessSendDataSlice]errNo[0x%016llx]kernel run error, tag[%s], "
            "input_ptr[%p], size[%llu]",
            HCCL_ERROR_CODE(ret), tag_.c_str(), execMem.inputMem.ptr(), slice.size),
        ret);
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::ProcessRecvDataSlice(Stream& stream, bool retryEnable)
{
    SendRecvSlice& slice = recvDataSilces_.front();
    ExecMem execMem;
    execMem.outputMem = algResResp_->cclOutputMem.range(0, slice.size);

    HcclResult ret = RecvKernelRun(stream, execMem, slice.remoteRank, retryEnable);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollBatchSendRecvExecutor][ProcessRecvDataSlice]errNo[0x%016llx]kernel run error, tag[%s], "
            "output_ptr[%p], size[%llu]",
            HCCL_ERROR_CODE(ret), tag_.c_str(), execMem.outputMem.ptr(), slice.size),
        ret);

    DeviceMem outMem(slice.addr, slice.size);
    DeviceMem outCommMem = execMem.outputMem;
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outMem, outCommMem, stream));
    return HCCL_SUCCESS;
}

HcclResult
CollBatchSendRecvExecutor::SendKernelRun(Stream& stream, ExecMem& execMem, u32 remoteUserRank, bool retryEnable)
{
    LINK targetLink;
    CHK_RET(GetSendTargetLink(remoteUserRank, targetLink));
    SendReceive executor(dispatcher_, targetLink, INVALID_VALUE_RANKID, HCCL_CHUNK_SIZE, retryEnable);
    CHK_RET(executor.SendPrepare(execMem.inputMem, remoteUserRank, stream));
    CHK_RET(executor.RegisterProfiler(0, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));
    CHK_RET(executor.BatchSendRunAsync());

    return HCCL_SUCCESS;
}

HcclResult
CollBatchSendRecvExecutor::RecvKernelRun(Stream& stream, ExecMem& execMem, u32 remoteUserRank, bool retryEnable)
{
    LINK targetLink;
    CHK_RET(GetRecvTargetLink(remoteUserRank, targetLink));
    SendReceive executor(dispatcher_, targetLink, INVALID_VALUE_RANKID, HCCL_CHUNK_SIZE, retryEnable);
    CHK_RET(executor.ReceivePrepare(execMem.outputMem, remoteUserRank, stream));
    CHK_RET(executor.RegisterProfiler(0, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));
    CHK_RET(executor.BatchReceiveRunAsync());
    return HCCL_SUCCESS;
}

HcclResult CollBatchSendRecvExecutor::GetTransport(u32 commIndex, u32 remoteUserRank, LINK& targetLink)
{
    CHK_PRT_RET(
        commIndex >= algResResp_->opTransportResponse[COMM_COMBINE_ORDER].size(),
        HCCL_ERROR(
            "[CollBatchSendRecvExecutor][KernelRun] batchsendrecv op commIndex[%u] is larger than "
            "opTransportResponse size[%zu]",
            remoteUserRank, algResResp_->opTransportResponse[COMM_COMBINE_ORDER].size()),
        HCCL_E_PARA);
    SingleSubCommTransport& commCombined
        = const_cast<SingleSubCommTransport&>(algResResp_->opTransportResponse[COMM_COMBINE_ORDER][commIndex]);

    CHK_PRT_RET(
        remoteUserRank >= commCombined.userRank2subCommRank.size(),
        HCCL_ERROR(
            "[CollBatchSendRecvExecutor][KernelRun] batchsendrecv op remoteUserRank[%u] is larger than "
            "userRank2subCommRank map size[%zu]",
            remoteUserRank, commCombined.userRank2subCommRank.size()),
        HCCL_E_PARA);

    u32 remoteRank = commCombined.userRank2subCommRank[remoteUserRank];
    CHK_PRT_RET(
        remoteRank >= commCombined.links.size(),
        HCCL_ERROR(
            "[CollBatchSendRecvExecutor][KernelRun] batchsendrecv op remoteUserRank[%u], get remoteRank[%u],"
            "the size of combinedComm links is [%zu]",
            remoteUserRank, remoteRank, commCombined.links.size()),
        HCCL_E_PARA);
    targetLink = commCombined.links[remoteRank];
    return HCCL_SUCCESS;
}

u64 CollBatchSendRecvExecutor::CalcSendLoopMaxCount(DeviceMem& inCCLBuffer, const u32 unitSize)
{
    // 中转内存单次最多能够接受的input count
    u64 maxCountPerLoop = inCCLBuffer.size() / unitSize;
    HCCL_WARNING(
        "[CollBatchSendRecvExecutor][CalcSendLoopMaxCount]"
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.",
        maxCountPerLoop);
    return maxCountPerLoop;
}

u64 CollBatchSendRecvExecutor::CalcRecvLoopMaxCount(DeviceMem& outCCLBuffer, const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = outCCLBuffer.size() / unitSize;
    HCCL_WARNING(
        "[CollBatchSendRecvExecutor][CalcRecvLoopMaxCount]"
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.",
        maxCountPerLoop);
    return maxCountPerLoop;
}

HcclResult CollBatchSendRecvExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = 1U;
    HCCL_INFO("[CollBatchSendRecvExecutor][CalcScratchMemSize] tag_[%s], streamNum[%u].", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}
HcclResult CollBatchSendRecvExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(
        COMM_COMBINE_ORDER, CommType::COMM_TAG_PARTIAL_MESH_COMBINED, INVALID_VALUE_RANKID, INVALID_VALUE_RANKID, false,
        false, commTargetUserRankSet_);
    TransportMemType inputType = TransportMemType::CCL_INPUT;
    TransportMemType outputType = TransportMemType::CCL_OUTPUT;

    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_COMBINE_ORDER], inputType, outputType));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("BatchSendRecv", BatchSendRecvExecutor, CollBatchSendRecvExecutor);
} // namespace hccl
