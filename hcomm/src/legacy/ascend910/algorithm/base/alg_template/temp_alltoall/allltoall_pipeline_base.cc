/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allltoall_pipeline_base.h"

namespace hccl {
AlltoallPipelineBase::AlltoallPipelineBase(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

AlltoallPipelineBase::~AlltoallPipelineBase() {}

HcclResult AlltoallPipelineBase::Prepare(
    u32 userRank, A2aPipelineMemory A2aPipelineMemory, const SubCommInfo& level0CommInfo,
    const SubCommInfo& level1CommInfo, Stream& mainStream, std::vector<Stream>& subStream,
    std::vector<std::shared_ptr<LocalNotify>>& notifyMain, std::vector<std::shared_ptr<LocalNotify>>& notifySub,
    std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, HcclWorkflowMode workMode)
{
    allMeshAggregationSendRecvInfo_ = &allMeshAggregationSendRecvInfo;
    workMode_ = workMode;

    localSendRecvInfo_ = (*allMeshAggregationSendRecvInfo_)[userRank];

    inputMem_ = A2aPipelineMemory.userInput;
    outputMem_ = A2aPipelineMemory.userOutput;
    scratchMem_ = A2aPipelineMemory.scratchMem;
    cclIn_ = A2aPipelineMemory.cclInBuffer;
    cclOut_ = A2aPipelineMemory.cclOutBuffer;

    intraRankSize_ = level0CommInfo.localRankSize;
    interRankSize_ = level1CommInfo.localRankSize;
    groupRankSize_ = intraRankSize_ * interRankSize_;

    userRank_ = userRank;
    intraRankId_ = level0CommInfo.localRank;
    interRankId_ = level1CommInfo.localRank;

    meshRankStart_ = userRank - intraRankId_;
    meshRankEnd_ = meshRankStart_ + intraRankSize_ - 1;

    mainStream_ = mainStream;
    subStream_ = subStream;
    streamNotifyMain_ = notifyMain;
    streamNotifySub_ = notifySub;

    intraLinks_ = level0CommInfo.links;
    interLinks_ = level1CommInfo.links;

    HCCL_DEBUG(
        "[AlltoallPipelineBase]streamNum[%u], streamNotifyMainNum[%u], streamNotifySubNum[%u]", subStream_.size(),
        streamNotifyMain_.size(), streamNotifySub_.size());
    HCCL_DEBUG("[AlltoallPipelineBase]interLinksNum[%u], intraLinksNum[%u]", interLinks_.size(), intraLinks_.size());

    return HCCL_SUCCESS;
}

HcclResult AlltoallPipelineBase::CheckResourceValid()
{
    if (workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        CHK_PRT_RET(
            cclIn_.size() != cclOut_.size(),
            HCCL_ERROR(
                "[AlltoallPipelineBase][CheckResourceValid] cclIn mem and cclOut mem should be the same size, "
                "ScratchInputMem[%llu] ScratchOutputMem[%llu]",
                cclIn_.size(), cclOut_.size()),
            HCCL_E_MEMORY);
    }
    CHK_PRT_RET(
        subStream_.size() < intraRankSize_ || streamNotifyMain_.size() < intraRankSize_
            || streamNotifySub_.size() < intraRankSize_,
        HCCL_DEBUG(
            "[AlltoallPipelineBase][CheckResourceValid] "
            "stream resource not enough, num sub stream[%llu], num notify main signal[%llu] num notify sub "
            "signal[%llu], "
            "should be more than or equal to intraRankSize %llu",
            subStream_.size(), streamNotifyMain_.size(), streamNotifySub_.size(), intraRankSize_),
        HCCL_E_UNAVAIL);
    return HCCL_SUCCESS;
}

// alltoall 系列算法抽象行为应该都可以分为第一次发送前的数据准备，中间的每一步同步发送，以及本地数据搬移
HcclResult AlltoallPipelineBase::RunAsync()
{
    CHK_RET(CheckResourceValid());
    CHK_RET(PreProcess());
    for (u32 step = 0, numStep = CalcInterNumSteps(); step < numStep; step++) {
        CHK_RET(PipelineSend(step, step == (numStep - 1)));
    }
    CHK_RET(PostProcess());
    return HCCL_SUCCESS;
}

std::string AlltoallPipelineBase::GetCurrClassName()
{
    std::string className = typeid(*this).name();
    if (className.find("class") != className.npos) {
        size_t classNamePrefixLen = 6;
        className = className.substr(classNamePrefixLen);
    }
    return className;
}

std::string AlltoallPipelineBase::GetStreamIndexString()
{
    std::string res = "";
    for (auto& info : intraStreamInfo_) {
        res += std::to_string(info.first) + ", ";
    }
    return res;
}

HcclResult AlltoallPipelineBase::NotifyInterStreamStart()
{
    CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, streamNotifySub_[intraRankId_], INVALID_VALUE_STAGE));
    CHK_RET(
        LocalNotify::Wait(subStream_[intraRankId_], dispatcher_, streamNotifySub_[intraRankId_], INVALID_VALUE_STAGE));
    HCCL_DEBUG(
        "[%s][NotifyInterStreamStart] userRank %u, interRank %u, "
        "intraRank %u, main stream notify sdma stream %s",
        GetCurrClassName().c_str(), userRank_, interRankId_, intraRankId_, GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

HcclResult AlltoallPipelineBase::WaitInterStreamFinish()
{
    CHK_RET(
        LocalNotify::Post(subStream_[intraRankId_], dispatcher_, streamNotifyMain_[intraRankId_], INVALID_VALUE_STAGE));
    CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, streamNotifyMain_[intraRankId_], INVALID_VALUE_STAGE));
    HCCL_DEBUG(
        "[%s][WaitInterStreamFinish] userRank %u, interRank %u, intraRank %u, "
        "main stream notify sdma stream %s",
        GetCurrClassName().c_str(), userRank_, interRankId_, intraRankId_, GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

// 主流只需要通知当前子步骤需要收发数据的 SDMA 流，减少同步开销
HcclResult AlltoallPipelineBase::NotifyIntraStreamStart()
{
    for (auto& sdmaInfo : intraStreamInfo_) {
        u32 streamIndex = sdmaInfo.first;
        CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, streamNotifySub_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(
            subStream_[streamIndex], dispatcher_, streamNotifySub_[streamIndex], INVALID_VALUE_STAGE));
    }
    HCCL_DEBUG(
        "[%s][NotifyIntraStreamStart] userRank %u, interRank %u, "
        "intraRank %u, main stream notify sdma stream %s",
        GetCurrClassName().c_str(), userRank_, interRankId_, intraRankId_, GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

HcclResult AlltoallPipelineBase::WaitIntraStreamFinish()
{
    for (auto& sdmaInfo : intraStreamInfo_) {
        u32 streamIndex = sdmaInfo.first;
        CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, streamNotifyMain_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Post(
            subStream_[streamIndex], dispatcher_, streamNotifyMain_[streamIndex], INVALID_VALUE_STAGE));
    }
    HCCL_DEBUG(
        "[%s][WaitIntraStreamFinish] userRank %u, interRank %u, "
        "intraRank %u, main stream wait sdma stream %s",
        GetCurrClassName().c_str(), userRank_, interRankId_, intraRankId_, GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

HcclResult AlltoallPipelineBase::GetNslbAdjInfo(
    const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo)
{
    u32 numStep = rankSize - 1;

    for (u32 step = 0; step < numStep; step++) {
        u32 nextRank = (rank + 1 + step) % rankSize;
        LINK nslbNext = links[nextRank];
        CHK_SMART_PTR_NULL(nslbNext);
        NslbDpAdjInfo nextInfoStep = {};
        nextInfoStep.dstLocalRankId = nslbNext->GetRemoteRank();
        nextInfoStep.phaseId = step + 1;
        nextInfoStep.rev = 0;
        nslbAdjInfo.nsAdjInfo.push_back(nextInfoStep);
    }

    nslbAdjInfo.dstRankNum = nslbAdjInfo.nsAdjInfo.size();
    return HCCL_SUCCESS;
}
} // namespace hccl
