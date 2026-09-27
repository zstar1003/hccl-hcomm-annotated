/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_native_executor_base.h"
#include "profiling_manager_pub.h"
namespace hccl {

CollNativeExecutorBase::CollNativeExecutorBase(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollExecutorBase(dispatcher, topoMatcher),
      topoAttr_(topoMatcher_->GetTopoInfo()),
      algoAttr_(topoMatcher_->GetAlgoInfo()),
      topoType_(topoAttr_.topoType),
      is310P3Common_(topoAttr_.is310P3Common),
      workflowMode_(GetWorkflowMode())
{}

void CollNativeExecutorBase::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    root_ = param.root;
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
    opType_ = param.opType;
}

// ----------------------资源计算接口----------------------
HcclResult CollNativeExecutorBase::CalcResRequest(const OpParam& param, AlgResourceRequest& resourceRequest)
{
    (void)ParseParam(param);

    u64 scratchMemSize = 0U;
    u32 streamNum = 0U;
    u32 notifyNum = 0U;
    u64 aivBufferRequest = 0U;
    std::vector<LevelNSubCommTransport> opTransport{
        std::vector<LevelNSubCommTransport>(static_cast<u32>(COMM_LEVEL_RESERVED))};

    CHK_RET(CalcScratchMemSize(scratchMemSize));
    CHK_RET(CalcOptimalIntraRing(param));
    CHK_RET(CalcStreamNum(streamNum));
    CHK_RET(CalcNotifyNum(streamNum, notifyNum));
    CHK_RET(CalcAivBufferRequest(aivBufferRequest));
    CHK_RET(CalcCommInfo(opTransport));

    CHK_RET(BuildResourceRequest(scratchMemSize, streamNum, notifyNum, aivBufferRequest, opTransport, resourceRequest));
    HCCL_INFO(
        "streamNum[%u], notifyNum[%u], scratchMemSize[%llu], aivBufferRequest[%llu]", resourceRequest.streamNum,
        resourceRequest.notifyNum, resourceRequest.scratchMemSize, resourceRequest.aivBufferRequest);
    // 打印建链诉求
    PrintTransportRequest(resourceRequest);
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::CalcScratchMemSize(u64& scratchMemSize)
{
    scratchMemSize = 0U;
    HCCL_INFO(
        "[CollNativeExecutorBase][CalcScratchMemSize]tag[%s] scratchMemSize_ is [%llu]", tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::CalcStreamNum(u32& streamNum)
{
    // 只传递从流数量
    streamNum = 0;
    HCCL_INFO("[CollNativeExecutorBase][CalcStreamNum]tag[%s] streamNum_ is [%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::CalcNotifyNum(u32 streamNum, u32& notifyNum)
{
    // notify数量是从流的两倍
    notifyNum = 2U * streamNum;
    HCCL_INFO("[CollNativeExecutorBase][CalcNotifyNum]tag[%s] notifyNum_ is [%u]", tag_.c_str(), notifyNum);
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::CalcAivBufferRequest(u64& aivBufferRequest)
{
    if (desc_.isAivMode) {
        SalSetBitOne(aivBufferRequest, ATTR_POS_AIV_COMM_BUFFER);
    }
    if (desc_.isAivCrossNode) {
        SalSetBitOne(aivBufferRequest, ATTR_POS_AIV_COMM_INFO_BUFFER);
    }
    HCCL_INFO(
        "[CollNativeExecutorBase][CalcAivBufferRequest]tag[%s] aivBufferRequest is [%llu]", tag_.c_str(),
        aivBufferRequest);
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::CalcCommInfo([[maybe_unused]] std::vector<LevelNSubCommTransport>& opTransport)
{
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::CalcOptimalIntraRing([[maybe_unused]] const OpParam& param) { return HCCL_SUCCESS; }

HcclResult CollNativeExecutorBase::SetCommInfoForARS(u32 ringSize)
{
    std::vector<u32> commPlaneVector = topoMatcher_->GetCommPlaneRanks(COMM_ARS)[0];
    std::sort(commPlaneVector.begin(), commPlaneVector.end());
    u32 intraRingsize = ringSize;
    u32 userRank = topoAttr_.userRank;
    u32 userRankSize = topoAttr_.userRankSize;
    HCCL_DEBUG("[SetCommInfoForARS]set topo info for ARS, USERRANK:%u, userRankSize:%u", userRank, userRankSize);

    SetCommInfoForIntraARS(intraRingsize, commPlaneVector);
    SetCommInfoForInterARS(intraRingsize, commPlaneVector);
    topoMatcher_->SetRankMap(); // 一定要刷新RankMap
    HCCL_DEBUG(
        "[SetTopoInfoForARS] outer userRank[%u] ,COMM_LEVEL0_LOGICAL total num [%d]", userRank,
        topoMatcher_->GetCommPlaneRanks(COMM_LEVEL0_LOGICAL).size());
    HCCL_DEBUG(
        "[SetTopoInfoForARS] outer userRank[%u] ,COMM_LEVEL1_LOGICAL total num [%d]", userRank,
        topoMatcher_->GetCommPlaneRanks(COMM_LEVEL1_LOGICAL).size());
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::SetCommInfoForIntraARS(u32 intraRingsize, std::vector<u32> commPlaneVector)
{
    std::vector<u32> comLevelARSVector = topoMatcher_->GetCommPlaneRanks(COMM_ARS)[0];
    u32 superPodRankSize = commPlaneVector.size();
    bool ringIntra = (comLevelARSVector.size() > 2 && topoAttr_.isARSDoubleRing);
    std::vector<u32> ringVectorIntra;
    for (u32 i = 0; i < superPodRankSize && ringIntra; i += intraRingsize) {
        u32 maxValue = i + intraRingsize;
        u32 rankval = topoAttr_.userRank % superPodRankSize;
        if (rankval < i || rankval >= maxValue) {
            continue;
        }
        for (u32 j = 0; j < intraRingsize; j++) {
            ringVectorIntra.push_back(commPlaneVector[i + j]);
        }
    }
    std::vector<std::vector<u32>> ARSmultiOuterOrder;
    std::vector<std::vector<u32>> intraRingVec;
    if (ringIntra) {
        ARSmultiOuterOrder = GetARSRingsOrder(intraRingsize, TopoType::TOPO_TYPE_NP_DOUBLE_RING, ringVectorIntra);
        for (u32 ringIndex = 0; ringIndex < ARSmultiOuterOrder.size(); ringIndex++) {
            std::string outLogInfo = "userRank:";
            std::vector<u32> tmpOuterVector;
            for (u32 startIndex = 0; startIndex < ARSmultiOuterOrder[ringIndex].size(); startIndex++) {
                u32 userRank = ARSmultiOuterOrder[ringIndex][startIndex];
                outLogInfo.append(std::to_string(userRank));
                outLogInfo.append("/");
                tmpOuterVector.push_back(userRank);
            }
            outLogInfo.append("; ");
            intraRingVec.push_back(tmpOuterVector);
            HCCL_INFO(
                "[COMM_LEVEL0_LOGICAL]: userRank[%u], userRankSize[%u], topoRankInfo[%s]", topoAttr_.userRank,
                topoAttr_.userRankSize, outLogInfo.c_str());
        }
    } else {
        std::string outLogInfo = "userRank: ";
        std::vector<u32> tmpOuterVector;
        outLogInfo.append(std::to_string(topoAttr_.userRank));
        tmpOuterVector.push_back(topoAttr_.userRank);
        intraRingVec.push_back(tmpOuterVector);
        HCCL_INFO(
            "[COMM_LEVEL0_LOGICAL]: userRank[%u], userRankSize[%u], topoRankInfo[%s]", topoAttr_.userRank,
            topoAttr_.userRankSize, outLogInfo.c_str());
    }
    topoMatcher_->EditCommPlaneVector(COMM_LEVEL0_LOGICAL, intraRingVec);
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::SetCommInfoForInterARS(u32 intraRingsize, std::vector<u32> commPlaneVector)
{
    u32 superPodRankSize = commPlaneVector.size();
    std::vector<u32> ringVectorInter;
    std::vector<std::vector<u32>> ringVectorInterOrder;
    for (u32 i = 0; i < intraRingsize; i++) {
        ringVectorInter.clear();
        for (u32 j = 0; j < superPodRankSize; j += intraRingsize) {
            ringVectorInter.push_back(commPlaneVector[i + j]);
        }
        ringVectorInterOrder.push_back(ringVectorInter);
    }
    std::vector<std::vector<u32>> interRingVec;
    for (u32 ringIndex = 0; ringIndex < ringVectorInterOrder.size(); ringIndex++) {
        std::string outLogInfo = "userRank: ";
        std::vector<u32> tmpOuterVector;
        for (u32 startIndex = 0; startIndex < ringVectorInterOrder[ringIndex].size(); startIndex++) {
            u32 userRank = ringVectorInterOrder[ringIndex][startIndex];
            outLogInfo.append(std::to_string(userRank));
            outLogInfo.append("/");
            tmpOuterVector.push_back(userRank);
        }
        outLogInfo.append("; ");
        interRingVec.push_back(tmpOuterVector);
        HCCL_INFO(
            "[COMM_LEVEL1_LOGICAL]:userRank[%u], userRankSize[%u], topoRankInfo[%s]", topoAttr_.userRank,
            topoAttr_.userRankSize, outLogInfo.c_str());
    }
    topoMatcher_->EditCommPlaneVector(COMM_LEVEL1_LOGICAL, interRingVec);
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::CalcCommPlaneInfo(
    const std::string& tag, const CommParaInfo& commParaInfo, std::vector<SingleSubCommTransport>& commTransport,
    TransportMemType inPutMemType, TransportMemType outPutMemType)
{
    return topoMatcher_->CalcCommPlaneInfo(tag, commParaInfo, commTransport, inPutMemType, outPutMemType);
}

HcclResult CollNativeExecutorBase::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] start", tag_.c_str());
    u32 root = root_;
    if (opType_ == HcclCMDType::HCCL_CMD_BROADCAST && topoAttr_.superPodNum > 1) {
        root = topoMatcher_->GetSubRootWithSuperPod(topoAttr_.userRank, root_);
        HCCL_DEBUG(
            "[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] subroot is %u usrRank is %u root_ is %u", tag_.c_str(),
            root, topoAttr_.userRank, root_);
    }
    CommParaInfo commParaLevel1(COMM_LEVEL1, CommType::COMM_TAG_MAX, root);

    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        commParaLevel1.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] Calc RingCommInfo", tag_.c_str());
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] Calc NHRCommInfo", tag_.c_str());
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1;
        HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] Calc NHRV1CommInfo", tag_.c_str());
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
        commParaLevel1.commPlane = CommPlane::COMM_LEVEL1_AHC;
        commParaLevel1.commType = CommType::COMM_TAG_ASYMMETRIC_HIERARCHICAL_CONCATENATE;
        HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] Calc AHCCommInfo", tag_.c_str());
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
        commParaLevel1.commPlane = CommPlane::COMM_LEVEL1_AHC;
        commParaLevel1.commType = CommType::COMM_TAG_ASYMMETRIC_HIERARCHICAL_CONCATENATE_BROKE;
        HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] Calc AHCBrokeCommInfo", tag_.c_str());
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] Calc NBCommInfo", tag_.c_str());
    } else {
        commParaLevel1.commType = CommType::COMM_TAG_HALVING_DOUBLING;
        HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] Calc HDCommInfo", tag_.c_str());
    }
    commParaLevel1.forceRdma = false;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel1, opTransport[commParaLevel1.commPlane], inputType, outputType));
    HCCL_INFO("[CollNativeExecutorBase][COMM_LEVEL1]tag[%s] Calc CommInfo Finish", tag_.c_str());

    HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] Calc CommInfo Finish", tag_.c_str());

    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::CalcLevel0CommInfo(
    [[maybe_unused]] TransportMemType inputType, [[maybe_unused]] TransportMemType outputType,
    [[maybe_unused]] std::vector<LevelNSubCommTransport>& opTransport)
{
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::CalcLevel2CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
        || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
        HCCL_INFO("[%s] select AHC bypass level2 comm calculate", __func__);
        return HCCL_SUCCESS;
    }

    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX, root_);
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[%s] Calc NHRCommInfo", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        HCCL_INFO("[%s] Calc NBCommInfo", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_HD) {
        commParaLevel2.commType = CommType::COMM_TAG_HALVING_DOUBLING;
        HCCL_INFO("[%s] Calc HDCommInfo", __func__);
    } else {
        commParaLevel2.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[%s] Calc RingCommInfo", __func__);
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));

    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::PrintTransportRequest(AlgResourceRequest& resourceRequest)
{
    for (u32 levelIndex = 0; levelIndex < COMM_LEVEL_RESERVED; levelIndex++) {
        LevelNSubCommTransport& levelTransport = resourceRequest.opTransport[levelIndex];
        u32 ringSize = levelTransport.size();
        for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
            SingleSubCommTransport& subCommTransport = levelTransport[ringIndex];
            u32 rankSize = subCommTransport.transportRequests.size();
            for (u32 rankIndex = 0; rankIndex < rankSize; rankIndex++) {
                if (subCommTransport.transportRequests[rankIndex].isValid == true) {
                    HCCL_INFO(
                        "[CollNativeExecutorBase][PrintTransportRequest]"
                        "levelIndex[%u], ringIndex[%u], rankIndex[%u], userRank[%u], remoteRank[%u], isUsedRdma[%d]",
                        levelIndex, ringIndex, rankIndex, subCommTransport.transportRequests[rankIndex].localUserRank,
                        subCommTransport.transportRequests[rankIndex].remoteUserRank,
                        subCommTransport.transportRequests[rankIndex].isUsedRdma);
                }
            }
        }
    }
    return HCCL_SUCCESS;
}
// ----------------------算法编排接口----------------------
HcclResult CollNativeExecutorBase::KernelRun([[maybe_unused]] const OpParam& param, [[maybe_unused]] ExecMem& execMem)
{
    HCCL_WARNING("[CollNativeExecutorBase][KernelRun]Using the default kernel run, nothing is done.");
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::ActiveSlaveStreams(const Stream& stream)
{
    HcclResult ret = HCCL_SUCCESS;
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
        for (u32 streamIndex = 0; streamIndex < algResResp_->slaveStreams.size(); streamIndex++) {
            ret = StreamActiveManager::GetInstance(topoAttr_.deviceLogicId)
                      .StreamActive(algResResp_->slaveStreams[streamIndex].ptr(), stream.ptr());
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollNativeExecutorBase][ActiveSlaveStreams]tag[%s], stream[%u] active failed,return[%d]",
                    tag_.c_str(), streamIndex, ret),
                ret);
        }
    }
    return ret;
}

HcclResult CollNativeExecutorBase::AddSubStreamToProfiling()
{
#ifndef OPEN_HCCL_TEST
    if (((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
         && !hccl::ProfilingManagerPub::GetAdditionInfoState() && !hccl::ProfilingManagerPub::GetTaskApiState()
         && !hccl::ProfilingManagerPub::GetThreadCaptureStatus())) {
        return HCCL_SUCCESS;
    }

    for (u32 streamIndex = 0; streamIndex < algResResp_->slaveStreams.size(); streamIndex++) {
        // profiling加入从环的stream
        HCCL_PROFILER_ADD_STREAM_BY_STREAMID(
            algResResp_->slaveStreams[streamIndex].id(), tag_, streamIndex + 1, algType_);
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::CheckCommSize(const CommPlane levelIndex, const u32 subLevelIndex)
{
    if (algResResp_->opTransportResponse[levelIndex].size() < subLevelIndex) {
        HCCL_ERROR(
            "[CollNativeExecutorBase][CheckCommSize]tag[%s], levelIndex[%u], "
            "ring size[%zu] is less than expected[%u]",
            tag_.c_str(), levelIndex, algResResp_->opTransportResponse[levelIndex].size(), subLevelIndex);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

SubCommInfo CollNativeExecutorBase::GetSubCommInfo(const CommPlane levelIndex, const u32 subLevelIndex)
{
    SubCommInfo info;
    SingleSubCommTransport& transportInfo
        = const_cast<SingleSubCommTransport&>(algResResp_->opTransportResponse[levelIndex][subLevelIndex]);
    info.localRank = transportInfo.userRank2subCommRank[topoAttr_.userRank];
    info.localRankSize = transportInfo.transportRequests.size();
    info.links = transportInfo.links;
    info.virtualLinks = transportInfo.virtualLinks;
    HCCL_INFO(
        "[CollNativeExecutorBase][GetSubCommInfo] localRank[%u], localRankSize[%u]", info.localRank,
        info.localRankSize);
    return info;
}

HcclResult CollNativeExecutorBase::BuildResourceRequest(
    u64 scratchMemSize, u32 streamNum, u32 notifyNum, u64 aivBufferRequest,
    std::vector<LevelNSubCommTransport>& opTransport, AlgResourceRequest& resourceRequest) const
{
    resourceRequest.scratchMemSize = scratchMemSize;
    resourceRequest.streamNum = streamNum;
    resourceRequest.notifyNum = notifyNum;
    resourceRequest.aivBufferRequest = aivBufferRequest;
    resourceRequest.opTransport = opTransport;
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::GetRankByUserRank(CommPlane levelIndex, u32 subLevelIndex, u32 userRank, u32& rank)
{
    CHK_RET(CheckCommSize(levelIndex, subLevelIndex + 1));
    SingleSubCommTransport& transportInfo
        = const_cast<SingleSubCommTransport&>(algResResp_->opTransportResponse[levelIndex][subLevelIndex]);
    rank = transportInfo.userRank2subCommRank[userRank];
    HCCL_DEBUG(
        "[GetRankByUserRank]levelIndex[%u] subLevelIndex[%u], userRank[%u], rank[%u]", levelIndex, subLevelIndex,
        userRank, rank);
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::GetUserRankByRank(CommPlane levelIndex, u32 subLevelIndex, u32 rank, u32& userRank)
{
    CHK_RET(CheckCommSize(levelIndex, subLevelIndex + 1));
    SingleSubCommTransport& transportInfo
        = const_cast<SingleSubCommTransport&>(algResResp_->opTransportResponse[levelIndex][subLevelIndex]);
    userRank = transportInfo.subCommRank2UserRank[rank];
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::GenerateStreams(PrepareData& prepareData, std::vector<Stream>& streams)
{
    // 主流 + 从流
    std::vector<Stream> substreams = *prepareData.subStreamsPtr;
    u32 streamIndexNum = substreams.size() + 1;
    u32 index = 0;
    for (u32 streamIndex = 0; streamIndex < streamIndexNum; streamIndex++) {
        if (streamIndex == 0) {
            streams.push_back(prepareData.stream);
        } else {
            streams.push_back(substreams[index]);
            index++;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::NotifySubStreamStart(
    Stream& stream, std::vector<Stream>& substreams, std::vector<std::shared_ptr<LocalNotify>>& signalsSubToMain,
    u32 substreamNum)
{
    for (u32 streamIndex = 0; streamIndex < substreamNum; streamIndex++) {
        CHK_RET(LocalNotify::Post(stream, dispatcher_, signalsSubToMain[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(
            substreams[streamIndex], dispatcher_, signalsSubToMain[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::WaitSubStreamFinish(
    Stream& stream, std::vector<Stream>& substreams, std::vector<std::shared_ptr<LocalNotify>>& signalsMainToSub,
    u32 substreamNum)
{
    for (u32 streamIndex = 0; streamIndex < substreamNum; streamIndex++) {
        CHK_RET(LocalNotify::Post(
            substreams[streamIndex], dispatcher_, signalsMainToSub[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(stream, dispatcher_, signalsMainToSub[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::GenerateRecordWaitStreams(
    std::vector<Stream>& streams, u32 recordStreamNum, u32 waitStreamNum, std::vector<Stream>& recordStreams,
    std::vector<Stream>& waitStreams)
{
    // 生成 record wait Streams
    for (u32 i = 0; i < recordStreamNum; i++) {
        recordStreams.push_back(streams[i]);
    }
    for (u32 i = recordStreamNum; i < recordStreamNum + waitStreamNum; i++) {
        waitStreams.push_back(streams[i]);
    }
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::HoldAllRanksOnCurrentOp(
    const OpParam& param, ExecMem& execMem, PrepareData& prepareData, std::vector<LINK> links)
{
    (void)param;
    u32 subStreamsNum = (*prepareData.subStreamsPtr).size();
    u32 signalNum = (*prepareData.signalPtr).size();
    u32 signalAuxNum = (*prepareData.signalAuxPtr).size();
    std::vector<Stream> substreams = *prepareData.subStreamsPtr;
    std::vector<std::shared_ptr<LocalNotify>> signalsMainToSub = *prepareData.signalPtr;
    std::vector<std::shared_ptr<LocalNotify>> signalsSubToMain = *prepareData.signalAuxPtr;
    // 校验数据是否对齐
    if (subStreamsNum != signalNum || subStreamsNum != signalAuxNum) {
        HCCL_ERROR(
            "[CollNativeExecutorBase][HoldAllRanksOnCurrentOp] The subStreamsNum[%u] != signalNum[%u] or "
            "subStreamsNum[%u] != signalAuxNum[%u]",
            subStreamsNum, signalNum, subStreamsNum, signalAuxNum);
        return HCCL_E_PARA;
    }

    std::vector<Stream> streams;
    CHK_RET(GenerateStreams(prepareData, streams));
    // 支持Record和Wait信号分stream排版布
    u32 recordStreamNum = (subStreamsNum + 1) / 2; // 2代表均分所有流
    u32 waitStreamNum = (subStreamsNum + 1) / 2;   // 2代表均分所有流
    std::vector<Stream> recordStreams;
    std::vector<Stream> waitStreams;
    CHK_RET(GenerateRecordWaitStreams(streams, recordStreamNum, waitStreamNum, recordStreams, waitStreams));

    // 主流record从流
    u32 neededSubstreamNum = recordStreamNum + waitStreamNum - 1;
    CHK_RET(NotifySubStreamStart(prepareData.stream, substreams, signalsSubToMain, neededSubstreamNum));

    u32 recordIndex = 0;
    u32 waitIndex = 0;
    // 防止某一个rank在link未通的情况下继续执行下一个算子
    for (size_t i = 0; i < links.size(); i++) {
        if (links[i] == nullptr) {
            HCCL_DEBUG("[CollNativeExecutorBase][HoldAllRanksOnCurrentOp]links[%zu] == nullptr.", i);
            continue;
        }
        HCCL_INFO(
            "[CollNativeExecutorBase][HoldAllRanksOnCurrentOp]links[%zu]. recordIndex[%u], waitIndex[%u], "
            "recordStreams.size()[%zu], waitStreams.size()[%zu]",
            i, recordIndex, waitIndex, recordStreams.size(), waitStreams.size());
        CHK_RET(links[i]->TxAck(recordStreams[recordIndex]));
        CHK_RET(links[i]->RxAck(waitStreams[waitIndex]));
        recordIndex = (recordIndex + 1) % recordStreams.size();
        waitIndex = (waitIndex + 1) % waitStreams.size();
    }
    CHK_RET(WaitSubStreamFinish(prepareData.stream, substreams, signalsMainToSub, neededSubstreamNum));
    CHK_RET(NotifySubStreamStart(prepareData.stream, substreams, signalsSubToMain, neededSubstreamNum));
    recordIndex = 0;
    waitIndex = 0;
    for (size_t i = 0; i < links.size(); i++) {
        if (links[i] == nullptr) {
            HCCL_DEBUG("[CollNativeExecutorBase][HoldAllRanksOnCurrentOp]links[%zu] == nullptr.", i);
            continue;
        }
        u64 size = std::min(execMem.inputMem.size(), HCCL_POST_SYNC_MEMCOPY_SIZE); // 传128K数据量占满所有端口
        HCCL_INFO(
            "[CollNativeExecutorBase][HoldAllRanksOnCurrentOp]links[%zu] start to memcopy data [%llu]B.", i, size);
        CHK_RET(links[i]->TxAsync(UserMemType::INPUT_MEM, 0, execMem.inputMem.ptr(), size, recordStreams[recordIndex]));
        CHK_RET(links[i]->RxAsync(UserMemType::INPUT_MEM, 0, execMem.inputMem.ptr(), size, waitStreams[waitIndex]));
        HCCL_INFO(
            "[CollNativeExecutorBase][HoldAllRanksOnCurrentOp]links[%zu]. recordIndex[%u], waitIndex[%u], "
            "recordStreams.size()[%zu], waitStreams.size()[%zu]",
            i, recordIndex, waitIndex, recordStreams.size(), waitStreams.size());
        CHK_RET(links[i]->PostFinAck(recordStreams[recordIndex]));
        CHK_RET(links[i]->WaitFinAck(waitStreams[waitIndex]));
        recordIndex = (recordIndex + 1) % recordStreams.size();
        waitIndex = (waitIndex + 1) % waitStreams.size();
    }
    // 从流record主流
    CHK_RET(WaitSubStreamFinish(prepareData.stream, substreams, signalsMainToSub, neededSubstreamNum));
    CHK_RET(LaunchTaskExtend(dispatcher_, prepareData.stream, substreams));
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::HoldAllRanksOnCurrentOpWithSingleStream(
    OpParam& param, ExecMem& execMem, std::vector<LINK> links)
{
    // 防止某一个rank在link未通的情况下继续执行下一个算子
    for (size_t i = 0; i < links.size(); i++) {
        if (links[i] == nullptr) {
            HCCL_DEBUG("[CollNativeExecutorBase][HoldAllRanksOnCurrentOpWithSingleStream]links[%zu] == nullptr.", i);
            continue;
        }
        HCCL_INFO("[CollNativeExecutorBase][HoldAllRanksOnCurrentOpWithSingleStream]links[%zu].", i);
        CHK_RET(links[i]->TxAck(param.stream));
        CHK_RET(links[i]->RxAck(param.stream));
        u64 size = std::min(execMem.inputMem.size(), HCCL_POST_SYNC_MEMCOPY_SIZE); // 传128K数据量占满所有端口
        HCCL_INFO(
            "[CollNativeExecutorBase][HoldAllRanksOnCurrentOpWithSingleStream]"
            "links[%zu] start to memcopy data [%llu]B.",
            i, size);
        CHK_RET(links[i]->TxAsync(UserMemType::INPUT_MEM, 0, execMem.inputMem.ptr(), size, param.stream));
        CHK_RET(links[i]->RxAsync(UserMemType::INPUT_MEM, 0, execMem.inputMem.ptr(), size, param.stream));
        CHK_RET(links[i]->PostFinAck(param.stream));
        CHK_RET(links[i]->WaitFinAck(param.stream));
    }
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::SendRecvSignalOnLinks(OpParam& param, ExecMem& execMem, std::vector<LINK> links)
{
    // 实验结果: 算子间隔1s能够被PreSync阻拦
    // 收发信号校验
    for (size_t i = 0; i < links.size(); i++) {
        if (links[i] == nullptr) {
            HCCL_DEBUG("[CollNativeExecutorBase][SendRecvSignalOnLinks]links[%zu] == nullptr.", i);
            continue;
        }
        HCCL_INFO("[CollNativeExecutorBase][SendRecvSignalOnLinks]links[%zu].", i);
        CHK_RET(links[i]->TxAck(param.stream));
        CHK_RET(links[i]->RxAck(param.stream));
    }
    // 拷贝数据从而占满端口，才能在注入故障时在PreSync算子触发重执行
    for (size_t i = 0; i < links.size(); i++) {
        if (links[i] == nullptr) {
            HCCL_DEBUG("[CollNativeExecutorBase][SendRecvSignalOnLinks]links[%zu] == nullptr.", i);
            continue;
        }
        u64 size = std::min(execMem.inputMem.size(), HCCL_INPLACE_MEMCOPY_SIZE); // 传128K数据量占满所有端口
        HCCL_INFO(
            "[CollNativeExecutorBase][SendRecvSignalOnLinks]"
            "links[%zu] start to memcopy data [%llu]B.",
            i, size);
        CHK_RET(links[i]->TxAsync(UserMemType::INPUT_MEM, 0, execMem.inputMem.ptr(), size, param.stream));
        CHK_RET(links[i]->RxAsync(UserMemType::INPUT_MEM, 0, execMem.inputMem.ptr(), size, param.stream));
        CHK_RET(links[i]->PostFinAck(param.stream));
        CHK_RET(links[i]->WaitFinAck(param.stream));
    }
    // 防止某一个rank在link未通的情况下继续执行下一个算子
    for (size_t i = 0; i < links.size(); i++) {
        if (links[i] == nullptr) {
            HCCL_DEBUG("[CollNativeExecutorBase][SendRecvSignalOnLinks]links[%zu] == nullptr.", i);
            continue;
        }
        HCCL_INFO("[CollNativeExecutorBase][SendRecvSignalOnLinks]links[%zu].", i);
        CHK_RET(links[i]->TxAck(param.stream));
        CHK_RET(links[i]->RxAck(param.stream));
        CHK_RET(links[i]->TxDataSignal(param.stream));
        CHK_RET(links[i]->RxDataSignal(param.stream));
    }
    return HCCL_SUCCESS;
}

bool CollNativeExecutorBase::OpSyncCheckCommSize(const CommPlane levelIndex, const u32 expectedSize)
{
    if (algResResp_->opTransportResponse[levelIndex].size() < expectedSize) {
        HCCL_WARNING(
            "[CollNativeExecutorBase][CheckCommSize]tag[%s], levelIndex[%u], "
            "ring size[%zu] is less than expected[%u]",
            tag_.c_str(), levelIndex, algResResp_->opTransportResponse[levelIndex].size(), expectedSize);
        return false;
    }
    return true;
}

HcclResult CollNativeExecutorBase::PostSyncWithSubstream(OpParam& param, ExecMem& execMem, PrepareData& prepareData)
{
    // COMM_COMBINE_ORDER 是不是只有alltoall类算子使用? 不是，有一些打平场景也会用到
    // 所以需要另起新函数，用于alltoall类算子的postsync调用
    HCCL_INFO(
        "[CollNativeExecutorBase][PostSyncWithSubstream] "
        "The op with algOpContext_.opRetryHandler.isPostSync[%d] starts.",
        algOpContext_.opRetryHandler.isPostSync);
    u32 level0ServerIndex = 0;
    if (OpSyncCheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1)) {
        SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
        level0ServerIndex = level0CommInfo.localRank;
        HCCL_INFO("[CollNativeExecutorBase][PostSyncWithSubstream]level0CommInfo.links check starts.");
        CHK_RET(HoldAllRanksOnCurrentOp(param, execMem, prepareData, level0CommInfo.links));
    }
    if (OpSyncCheckCommSize(COMM_LEVEL1, level0ServerIndex + 1)) {
        SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, level0ServerIndex);
        HCCL_INFO("[CollNativeExecutorBase][PostSyncWithSubstream]level1CommInfo.links check starts.");
        CHK_RET(HoldAllRanksOnCurrentOp(param, execMem, prepareData, level1CommInfo.links));
    }
    if (OpSyncCheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1)) {
        SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
        HCCL_INFO("[CollNativeExecutorBase][PostSyncWithSubstream]level2CommInfo.links check starts.");
        CHK_RET(HoldAllRanksOnCurrentOp(param, execMem, prepareData, level2CommInfo.links));
    }
    if (OpSyncCheckCommSize(COMM_LEVEL1, level0ServerIndex + 1)) {
        SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, level0ServerIndex);
        HCCL_INFO("[CollNativeExecutorBase][PostSyncWithSubstream]level1CommInfo.links check starts again.");
        CHK_RET(HoldAllRanksOnCurrentOp(param, execMem, prepareData, level1CommInfo.links));
    }
    if (OpSyncCheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1)) {
        SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
        HCCL_INFO("[CollNativeExecutorBase][PostSyncWithSubstream]level0CommInfo.links check starts again.");
        CHK_RET(HoldAllRanksOnCurrentOp(param, execMem, prepareData, level0CommInfo.links));
    }
    if (OpSyncCheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1)) {
        SubCommInfo combineOrderCommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
        HCCL_INFO("[CollNativeExecutorBase][PostSyncWithSubstream]combineOrderCommInfo.links check starts.");
        CHK_RET(HoldAllRanksOnCurrentOp(param, execMem, prepareData, combineOrderCommInfo.links));
    }
    if (OpSyncCheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1)) {
        SubCommInfo combineOrderCommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
        HCCL_INFO("[CollNativeExecutorBase][PostSyncWithSubstream]combineOrderCommInfo.links check starts again.");
        CHK_RET(HoldAllRanksOnCurrentOp(param, execMem, prepareData, combineOrderCommInfo.links));
    }
    HCCL_INFO(
        "[CollNativeExecutorBase][PostSyncWithSubstream] "
        "The op with algOpContext_.opRetryHandler.isPostSync[%d] ends.",
        algOpContext_.opRetryHandler.isPostSync);
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::PostSyncWithoutSubstream(OpParam& param, ExecMem& execMem)
{
    HCCL_INFO(
        "[CollNativeExecutorBase][PostSyncWithoutSubstream] "
        "The op with algOpContext_.opRetryHandler.isPostSync[%d] starts.",
        algOpContext_.opRetryHandler.isPostSync);
    u32 level0ServerIndex = 0;
    if (OpSyncCheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1)) {
        SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
        level0ServerIndex = level0CommInfo.localRank;
        HCCL_INFO("[CollNativeExecutorBase][PostSyncWithoutSubstream]level0CommInfo.links check starts.");
        CHK_RET(HoldAllRanksOnCurrentOpWithSingleStream(param, execMem, level0CommInfo.links));
    }
    if (OpSyncCheckCommSize(COMM_LEVEL1, level0ServerIndex + 1)) {
        SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, level0ServerIndex);
        HCCL_INFO("[CollNativeExecutorBase][PostSyncWithoutSubstream]level1CommInfo.links check starts.");
        CHK_RET(HoldAllRanksOnCurrentOpWithSingleStream(param, execMem, level1CommInfo.links));
    }
    if (OpSyncCheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1)) {
        SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
        HCCL_INFO("[CollNativeExecutorBase][PostSyncWithoutSubstream]level2CommInfo.links check starts.");
        CHK_RET(HoldAllRanksOnCurrentOpWithSingleStream(param, execMem, level2CommInfo.links));
    }
    if (OpSyncCheckCommSize(COMM_LEVEL1, level0ServerIndex + 1)) {
        SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, level0ServerIndex);
        HCCL_INFO("[CollNativeExecutorBase][PostSyncWithoutSubstream]level1CommInfo.links check starts again.");
        CHK_RET(HoldAllRanksOnCurrentOpWithSingleStream(param, execMem, level1CommInfo.links));
    }
    if (OpSyncCheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1)) {
        SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
        HCCL_INFO("[CollNativeExecutorBase][PostSyncWithoutSubstream]level0CommInfo.links check starts again.");
        CHK_RET(HoldAllRanksOnCurrentOpWithSingleStream(param, execMem, level0CommInfo.links));
    }
    if (OpSyncCheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1)) {
        SubCommInfo combineOrderCommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
        HCCL_INFO("[CollNativeExecutorBase][PostSyncWithoutSubstream]combineOrderCommInfo.links check starts.");
        CHK_RET(HoldAllRanksOnCurrentOpWithSingleStream(param, execMem, combineOrderCommInfo.links));
    }
    if (OpSyncCheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1)) {
        SubCommInfo combineOrderCommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
        HCCL_INFO("[CollNativeExecutorBase][PostSyncWithoutSubstream]combineOrderCommInfo.links check starts again.");
        CHK_RET(HoldAllRanksOnCurrentOpWithSingleStream(param, execMem, combineOrderCommInfo.links));
    }
    HCCL_INFO(
        "[CollNativeExecutorBase][PostSyncWithoutSubstream] "
        "The op with algOpContext_.opRetryHandler.isPostSync[%d] ends.",
        algOpContext_.opRetryHandler.isPostSync);

    CHK_RET(LaunchTaskExtend(
        dispatcher_, const_cast<Stream&>(param.stream), const_cast<std::vector<Stream>&>(algResResp_->slaveStreams)));

    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::InplaceOpSync(OpParam& param, ExecMem& execMem)
{
    HCCL_INFO(
        "[CollNativeExecutorBase][InplaceOpSync] The op with algOpContext_.opRetryHandler.isInplacePreSync[%d] "
        "or algOpContext_.opRetryHandler.isPostSync[%d] starts.",
        algOpContext_.opRetryHandler.isInplacePreSync, algOpContext_.opRetryHandler.isPostSync);
    u32 level0ServerIndex = 0;
    if (OpSyncCheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1)) {
        SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
        level0ServerIndex = level0CommInfo.localRank;
        HCCL_INFO("[CollNativeExecutorBase][InplaceOpSync]level0CommInfo.links check starts.");
        CHK_RET(SendRecvSignalOnLinks(param, execMem, level0CommInfo.links));
    }
    if (OpSyncCheckCommSize(COMM_LEVEL1, level0ServerIndex + 1)) {
        SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, level0ServerIndex);
        HCCL_INFO("[CollNativeExecutorBase][InplaceOpSync]level1CommInfo.links check starts.");
        CHK_RET(SendRecvSignalOnLinks(param, execMem, level1CommInfo.links));
    }
    if (OpSyncCheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1)) {
        SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
        HCCL_INFO("[CollNativeExecutorBase][InplaceOpSync]level2CommInfo.links check starts.");
        CHK_RET(SendRecvSignalOnLinks(param, execMem, level2CommInfo.links));
    }
    if (OpSyncCheckCommSize(COMM_LEVEL1, level0ServerIndex + 1)) {
        SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, level0ServerIndex);
        HCCL_INFO("[CollNativeExecutorBase][InplaceOpSync]level1CommInfo.links check starts again.");
        CHK_RET(SendRecvSignalOnLinks(param, execMem, level1CommInfo.links));
    }
    if (OpSyncCheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1)) {
        SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
        HCCL_INFO("[CollNativeExecutorBase][InplaceOpSync]level0CommInfo.links check starts again.");
        CHK_RET(SendRecvSignalOnLinks(param, execMem, level0CommInfo.links));
    }
    // alltoall-like opType
    if (OpSyncCheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1)) {
        SubCommInfo combineOrderCommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
        HCCL_INFO("[CollNativeExecutorBase][InplaceOpSync]combineOrderCommInfo.links check starts.");
        CHK_RET(SendRecvSignalOnLinks(param, execMem, combineOrderCommInfo.links));
    }
    if (OpSyncCheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1)) {
        SubCommInfo combineOrderCommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
        HCCL_INFO("[CollNativeExecutorBase][InplaceOpSync]combineOrderCommInfo.links check starts again.");
        CHK_RET(SendRecvSignalOnLinks(param, execMem, combineOrderCommInfo.links));
    }
    HCCL_INFO(
        "[CollNativeExecutorBase][InplaceOpSync] The op with algOpContext_.opRetryHandler.isInplacePreSync[%d] "
        "or algOpContext_.opRetryHandler.isPostSync[%d] ends.",
        algOpContext_.opRetryHandler.isInplacePreSync, algOpContext_.opRetryHandler.isPostSync);

    CHK_RET(LaunchTaskExtend(
        dispatcher_, const_cast<Stream&>(param.stream), const_cast<std::vector<Stream>&>(algResResp_->slaveStreams)));

    return HCCL_SUCCESS;
}

std::vector<std::vector<u32>> GetARSRingsOrder(u32 ranksSize, TopoType topoType, std::vector<u32>& RingList)
{
    std::vector<std::vector<u32>> ARSmultiRingOrder;
    std::vector<u32> tmpOuter0 = RingList;                                          // 环0
    if (topoType == TopoType::TOPO_TYPE_NP_DOUBLE_RING && ranksSize > FACTOR_TWO) { // 两环
        std::vector<u32> tmpOuter1;                                                 // 环1
        tmpOuter1.reserve(ranksSize);
        tmpOuter1.push_back(RingList[0]);
        tmpOuter1.insert(tmpOuter1.end(), tmpOuter0.rbegin(), tmpOuter0.rend() - 1);
        ARSmultiRingOrder.push_back(tmpOuter0);
        ARSmultiRingOrder.push_back(tmpOuter1);
    } else {
        ARSmultiRingOrder.push_back(tmpOuter0);
    }
    return ARSmultiRingOrder;
}

HcclResult CollNativeExecutorBase::CopyAivCommInfoToDevice(
    const CommPlane levelIndex, const u32 subLevelIndex, AlgResourceResponse& algResource)
{
    algResResp_ = &algResource;
    CHK_RET(CheckCommSize(levelIndex, subLevelIndex + 1));
    SubCommInfo commInfo = GetSubCommInfo(levelIndex, subLevelIndex);
    u32 localRank = commInfo.localRank;
    u32 localRankSize = commInfo.localRankSize;

    void* buffersInOut[MAX_RANK_SIZE_A3 * 2] = {};
    bool isOpbaseMode = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;

    for (u32 i = 0; i < localRankSize; i++) {
        u32 idx = (i << 1);
        if (i != localRank) {
            CHK_RET(commInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(buffersInOut[idx])));
            CHK_RET(commInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(buffersInOut[idx + 1])));
        } else {
            buffersInOut[idx] = isOpbaseMode ? algResource.cclInputMem.ptr() : algResource.paramInputMem.ptr();
            buffersInOut[idx + 1] = algResource.aivOutputMem.ptr();
        }
    }
    const u32 bufferNum = 2;
    CHK_RET(hrtMemSyncCopy(
        algResource.aivCommInfoMem.ptr(), sizeof(u64) * localRankSize * bufferNum, buffersInOut,
        sizeof(u64) * localRankSize * bufferNum, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::Getlevel1CommRank(SubCommInfo& level1CommInfo)
{
    (void)level1CommInfo;
    return HCCL_SUCCESS;
}
HcclResult CollNativeExecutorBase::SelectTempAlg(std::unique_ptr<AlgTemplateBase>& level1TempAlg, u32 level1RankSize)
{
    (void)level1TempAlg;
    (void)level1RankSize;
    return HCCL_SUCCESS;
}
HcclResult CollNativeExecutorBase::GetDevNumInlocalPod(u32& devNumInlocalPod)
{
    (void)devNumInlocalPod;
    return HCCL_SUCCESS;
}

HcclResult CollNativeExecutorBase::SetOpCache(
    const AivOpArgs& opArgs, const AivTopoArgs& topoArgs, const AivResourceArgs& resourceArgs,
    const AivAlgArgs& algArgs, ExtraArgs& extraArgs, AivProfilingInfo& aivProfilingInfo, bool isA3CrossNode)
{
    cacheInfo_.opArgs = opArgs;
    cacheInfo_.topoArgs = topoArgs;
    cacheInfo_.resourceArgs = resourceArgs;
    cacheInfo_.algArgs = algArgs;
    cacheInfo_.profilingInfo = aivProfilingInfo;
    cacheInfo_.extraArgs = extraArgs;
    cacheInfo_.isUseCache = true;

    if (isA3CrossNode) {
        u8 buffersOutSize = 2 * sizeof(void*);
        CHK_SAFETY_FUNC_RET(memcpy_s(cacheInfo_.buffersIn, sizeof(void*), resourceArgs.buffersIn, sizeof(void*)));
        CHK_SAFETY_FUNC_RET(memcpy_s(cacheInfo_.buffersOut, buffersOutSize, resourceArgs.buffersOut, buffersOutSize));
    } else {
        u64 bufferInfoSize = sizeof(void*) * topoArgs.rankSize;
        CHK_SAFETY_FUNC_RET(memcpy_s(cacheInfo_.buffersIn, bufferInfoSize, resourceArgs.buffersIn, bufferInfoSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(cacheInfo_.buffersOut, bufferInfoSize, resourceArgs.buffersOut, bufferInfoSize));
    }

    HCCL_INFO(
        "[CollNativeExecutorBase][SetOpCache] cmdType:%d, count:%llu, dataType:%d, op:%d, "
        "rank:%u, rankSize:%u, serverNum:%u, isA3CrossNode:%d, buffersIn:%p, buffersOut:%p",
        opArgs.cmdType, opArgs.count, opArgs.dataType, opArgs.op, topoArgs.rank, topoArgs.rankSize, topoArgs.serverNum,
        isA3CrossNode, cacheInfo_.buffersIn, cacheInfo_.buffersOut);

    return HCCL_SUCCESS;
}

void CollNativeExecutorBase::SetWorkflowMode(HcclWorkflowMode workflowMode) { workflowMode_ = workflowMode; }

HcclWorkflowMode CollNativeExecutorBase::GetExecutorWorkflowMode() const { return workflowMode_; }
} // namespace hccl
