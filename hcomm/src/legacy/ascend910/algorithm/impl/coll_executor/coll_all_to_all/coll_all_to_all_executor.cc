/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_executor.h"
#include "device_capacity.h"

namespace hccl {

CollAlltoAllExecutor::CollAlltoAllExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollNativeExecutorBase(dispatcher, topoMatcher)
{}

HcclResult CollAlltoAllExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;
    AlltoAllVParam_ = param;
    ExecMem execMem;
    execMem.count = 0;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    HcclResult ret = HCCL_SUCCESS;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;

        auto opMeta = GetOpMeta(param.opType, algRes.paramInputMem.size()); // override
        CHK_RET(InitTask(dispatcher_, param.stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
        bool massTasks = HasMassTasks(allMeshAggregationSendRecvInfo_);
        if (massTasks) {
            CHK_RET(SetNormalMode(dispatcher_));
        }
        ret = KernelRun(param, execMem);
    } else {
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.paramOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        ret = KernelRun(param, execMem);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlltoAllExecutor][Orchestrate]errNo[0x%016llx]executor run failed", HCCL_ERROR_CODE(ret)),
        ret);

    // Enforce task launch at the end of Orchestrate
    // 注意: 不要删除这里的强制launch, 否则会导致aicpu cache功能问题
    HCCL_INFO("%s: enforce task launch at the end of Orchestrate", __func__);
    CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));

    HCCL_INFO(
        "tag[%s], AlltoAll executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllExecutor::GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo)
{
    algResResp_ = &algRes;
    SubCommInfo levelCommInfo = {};
    AdjInfo nslbAdjInfo = {};
    u32 devNumInlocalPod = INVALID_VALUE_RANKSIZE;

    if (Getlevel1CommRank(levelCommInfo) != HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }
    u32 localRank = levelCommInfo.localRank;
    u32 localRankSize = levelCommInfo.localRankSize;

    std::unique_ptr<AlgTemplateBase> levelTempAlg;
    if (SelectTempAlg(levelTempAlg, localRankSize) != HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }
    GetDevNumInlocalPod(devNumInlocalPod);
    if (devNumInlocalPod == INVALID_VALUE_RANKSIZE) {
        HCCL_INFO("[GetAdjInfo-NSLB-DP] devNumInlocalPod == INVALID_VALUE_RANKSIZE.");
        return HCCL_SUCCESS;
    }

    nslbAdjInfo.dstRankNum = devNumInlocalPod;
    CHK_RET(levelTempAlg->GetNslbAdjInfo(localRank, localRankSize, levelCommInfo.links, nslbAdjInfo));

    adjInfo.dstRankNum = nslbAdjInfo.dstRankNum;
    HCCL_INFO("[GetAdjInfo-NSLB-DP] adjInfo.dstRankNum[%u].", adjInfo.dstRankNum);

    for (size_t i = 0; i < nslbAdjInfo.nsAdjInfo.size(); i++) {
        NslbDpAdjInfo dpAdjInfo = {};
        dpAdjInfo.dstLocalRankId = nslbAdjInfo.nsAdjInfo[i].dstLocalRankId;
        dpAdjInfo.phaseId = nslbAdjInfo.nsAdjInfo[i].phaseId;
        dpAdjInfo.rev = 0;
        adjInfo.nsAdjInfo.push_back(dpAdjInfo);
        HCCL_INFO(
            "[NSLB-DP]GetAdjInfo dstLocalRankId[%u], phaseId[%u].", nslbAdjInfo.nsAdjInfo[i].dstLocalRankId,
            nslbAdjInfo.nsAdjInfo[i].phaseId);
    }
    return HCCL_SUCCESS;
}

// override----------------------资源计算接口----------------------
HcclResult CollAlltoAllExecutor::CalcResRequest(const OpParam& param, AlgResourceRequest& resourceRequest)
{
    (void)ParseParam(param);

    u64 scratchMemSize = 0U;
    u32 streamNum = 0U;
    u32 notifyNum = 0U;
    u64 aivBufferRequest = 0U;
    std::vector<LevelNSubCommTransport> opTransport{
        std::vector<LevelNSubCommTransport>(static_cast<u32>(COMM_LEVEL_RESERVED))};

    // AICPU aicpuUnfold展开模式下临时强制OP_BASE，使整个资源计算路径与AICPU侧一致
    // CalcScratchMemSize走OP_BASE分支正确计算scratch
    // CalcCommInfo走OP_BASE分支设outputMemType为CCL_OUTPUT而非SCRATCH
    // 避免transport因scratch未分配而拿到nullptr
    // AIV executor有独立的资源计算逻辑，不需要force OP_BASE
    const bool needForceOpBase = param.aicpuUnfoldMode && !param.isZeroCopy && !desc_.isAivMode;
    HCCL_INFO(
        "[CollAlltoAllExecutor][CalcResRequest] aicpuUnfoldMode[%d] isZeroCopy[%d] "
        "needForceOpBase[%d] workflowMode[%d] tag[%s]",
        param.aicpuUnfoldMode, param.isZeroCopy, needForceOpBase, workflowMode_, param.tag.c_str());
    const HcclWorkflowMode savedWorkflowMode = workflowMode_;
    if (needForceOpBase) {
        HCCL_INFO(
            "[CollAlltoAllExecutor][CalcResRequest] aicpuUnfoldMode force OpBase, "
            "originalWorkflowMode[%d], tag[%s]",
            savedWorkflowMode, param.tag.c_str());
        workflowMode_ = HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    }

    CHK_RET(CalcScratchMemSize(scratchMemSize));
    CHK_RET(CalcStreamNum(streamNum));
    CHK_RET(CalcNotifyNum(streamNum, notifyNum));
    CHK_RET(CalcAivBufferRequest(aivBufferRequest));
    CHK_RET(CalcCommInfo(opTransport));

    if (needForceOpBase) {
        HCCL_DEBUG(
            "[CollAlltoAllExecutor][CalcResRequest] restore workflowMode "
            "after resource calc, scratchMemSize[%llu]",
            scratchMemSize);
        workflowMode_ = savedWorkflowMode;
    }

    CHK_RET(BuildResourceRequest(scratchMemSize, streamNum, notifyNum, aivBufferRequest, opTransport, resourceRequest));
    HCCL_INFO(
        "[CollAlltoAllExecutor][%s] streamNum[%u], notifyNum[%u], scratchMemSize[%llu], aivBufferRequest[%llu]",
        __func__, resourceRequest.streamNum, resourceRequest.notifyNum, resourceRequest.scratchMemSize,
        resourceRequest.aivBufferRequest);
    // 打印建链诉求
    for (u32 levelIndex = 0; levelIndex < COMM_LEVEL_RESERVED; levelIndex++) {
        LevelNSubCommTransport& levelTransport = resourceRequest.opTransport[levelIndex];
        u32 ringSize = levelTransport.size();
        for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
            SingleSubCommTransport& subCommTransport = levelTransport[ringIndex];
            u32 rankSize = subCommTransport.transportRequests.size();
            for (u32 rankIndex = 0; rankIndex < rankSize; rankIndex++) {
                if (subCommTransport.transportRequests[rankIndex].isValid == true) {
                    HCCL_INFO(
                        "[CollAlltoAllExecutor][CalcResRequest]"
                        "levelIndex[%u], ringIndex[%u], rankIndex[%u], userRank[%u], remoteRank[%u] "
                        "isUsedRdma[%d]",
                        levelIndex, ringIndex, rankIndex, subCommTransport.transportRequests[rankIndex].localUserRank,
                        subCommTransport.transportRequests[rankIndex].remoteUserRank,
                        subCommTransport.transportRequests[rankIndex].isUsedRdma);
                }
            }
        }
    }
    CHK_RET(CheckNeedCreateVirtualLinks(resourceRequest));
    HCCL_DEBUG("[%s] process success", __func__);
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllExecutor::CheckNeedCreateVirtualLinks([[maybe_unused]] AlgResourceRequest& resourceRequest)
{
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllExecutor::SetExecutorExtraInfo(
    const std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, u64 cclbufferSize)
{
    allMeshAggregationSendRecvInfo_.clear();
    allMeshAggregationSendRecvInfo_ = allMeshAggregationSendRecvInfo;
    UpdateAlltoAllZCopyMode(allMeshAggregationSendRecvInfo_, cclbufferSize);
    HCCL_DEBUG("[%s] allMeshAggregationSendRecvInfo_ size[%u]", __func__, allMeshAggregationSendRecvInfo_.size());

    return HCCL_SUCCESS;
}

void CollAlltoAllExecutor::UpdateAlltoAllZCopyMode(
    std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, u64 cclbufferSize)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u64 maxSendSize = 0;
        u64 maxRecvSize = 0;
        for (auto& sendRecvInfo : allMeshAggregationSendRecvInfo) {
            for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
                u64 curSendSize = sendRecvInfo.sendLength[i] + sendRecvInfo.sendOffset[i];
                maxSendSize = std::max(maxSendSize, curSendSize);
                u64 curRecvSize = sendRecvInfo.recvLength[i] + sendRecvInfo.recvOffset[i];
                maxRecvSize = std::max(maxRecvSize, curRecvSize);
            }
        }
        bool isAlltoAllZCopyMode = (maxSendSize <= cclbufferSize) && (maxRecvSize <= cclbufferSize);
        if (isAlltoAllZCopyMode) {
            isAlltoAllZCopyMode_ = true;
        }
        HCCL_INFO(
            "[CollAlltoAllExecutor][UpdateAlltoAllZCopyMode] maxSendSize[%llu], maxRecvSize[%llu], "
            "cclBufferSize[%llu]",
            maxSendSize, maxRecvSize, cclbufferSize);
    } else {
        // 图模式走ZCopy实现
        isAlltoAllZCopyMode_ = true;
    }
    HCCL_DEBUG("UpdateAlltoAllZCopyMode isAlltoAllZCopyMode_[%d]", isAlltoAllZCopyMode_);
}

void CollAlltoAllExecutor::CalcIntraMeshAggregationSendInfo(
    const AlltoAllUserRankInfo& userRankInfo, const SendRecvInfo& mySendRecvInfo,
    const std::vector<SendRecvInfo>& myMeshAggregationSendRecvInfo, u32 rankInMeshAggregation, u32 infoIndex,
    OneSendRecvAddrInfo& curSendInfo, u32 meshAggregationRankSize, const bool& isSingleMesh) const
{
    if (infoIndex >= mySendRecvInfo.sendOffset.size() || infoIndex >= mySendRecvInfo.sendLength.size()) {
        HCCL_ERROR("[CalcIntraMeshAggregationSendInfo] Invalid infoIndex[%u]", infoIndex);
        return;
    }
    curSendInfo.localOffset = mySendRecvInfo.sendOffset[infoIndex];
    curSendInfo.localLength = mySendRecvInfo.sendLength[infoIndex];
    u64 remoteOffset = 0;

    if (isSingleMesh) {
        remoteOffset = myMeshAggregationSendRecvInfo[infoIndex].recvOffset[userRankInfo.userRank];
    } else {
        for (u32 j = infoIndex % meshAggregationRankSize; j <= infoIndex; j += meshAggregationRankSize) {
            for (u32 k = 0; k < meshAggregationRankSize; k++) {
                if (j == infoIndex && k == rankInMeshAggregation) {
                    break;
                }
                if (k < myMeshAggregationSendRecvInfo.size()
                    && j < myMeshAggregationSendRecvInfo[k].sendLength.size()) {
                    remoteOffset += myMeshAggregationSendRecvInfo[k].sendLength[j];
                } else {
                    HCCL_ERROR(
                        "[CalcIntraMeshAggregationSendInfo] invalid MeshAggregationSendRecvInfo size[%zu]",
                        myMeshAggregationSendRecvInfo.size());
                    return;
                }
            }
        }
    }

    curSendInfo.remoteOffset = remoteOffset;
    curSendInfo.remoteLength = curSendInfo.localLength;
    HCCL_DEBUG(
        "[CalcIntraMeshAggregationSendInfo] localOffset[%llu], localLength[%llu], "
        "remoteOffset[%llu], remoteLength[%llu]",
        curSendInfo.localOffset, curSendInfo.localLength, curSendInfo.remoteOffset, curSendInfo.remoteLength);
}

void CollAlltoAllExecutor::CalcIntraMeshAggregationRecvInfoInMeshAggregation(
    u32 rankIndex, u32 infoIndex, const std::vector<SendRecvInfo>& myMeshAggregationSendRecvInfo, u64& localOffset,
    u32& offsetCounter, u64& localLength, u64& remoteOffset, u32 meshAggregationRankSize) const
{
    // 这里的判断在外部已经保证了，为了应对coverity sc
    if (myMeshAggregationSendRecvInfo.size() < meshAggregationRankSize) {
        HCCL_ERROR(
            "[CalcIntraMeshAggregationSendInfo] Invalid myMeshAggregationSendRecvInfo[%zu]",
            myMeshAggregationSendRecvInfo.size());
        return;
    }
    if (myMeshAggregationSendRecvInfo[0].sendLength.size() == 0
        || myMeshAggregationSendRecvInfo[0].sendOffset.size() == 0) {
        HCCL_ERROR(
            "[CalcIntraMeshAggregationSendInfo] Invalid sendLength size[%zu] or sendOffset size[%zu]",
            myMeshAggregationSendRecvInfo[0].sendLength.size(), myMeshAggregationSendRecvInfo[0].sendOffset.size());
        return;
    }
    for (u32 k = 0; k < meshAggregationRankSize; k++) {
        if (infoIndex == 0) {
            localOffset = 0;
            localLength = myMeshAggregationSendRecvInfo[k].sendLength[rankIndex];
            remoteOffset = myMeshAggregationSendRecvInfo[k].sendOffset[rankIndex];
            break;
        }

        localOffset += myMeshAggregationSendRecvInfo[k].sendLength[rankIndex];
        offsetCounter++;
        if (offsetCounter == infoIndex) {
            if (k == meshAggregationRankSize - 1) {
                localLength = myMeshAggregationSendRecvInfo[0].sendLength[rankIndex + meshAggregationRankSize];
                remoteOffset = myMeshAggregationSendRecvInfo[0].sendOffset[rankIndex + meshAggregationRankSize];
            } else {
                localLength = myMeshAggregationSendRecvInfo[k + 1].sendLength[rankIndex];
                remoteOffset = myMeshAggregationSendRecvInfo[k + 1].sendOffset[rankIndex];
            }
            break;
        }
    }
    HCCL_DEBUG("[%s] process success", __func__);
}

void CollAlltoAllExecutor::CalcIntraMeshAggregationRecvInfo(
    const AlltoAllUserRankInfo& userRankInfo, const std::vector<SendRecvInfo>& myMeshAggregationSendRecvInfo,
    u32 infoIndex, OneSendRecvAddrInfo& curRecvInfo, u32 meshAggregationRankSize, const bool& isSingleMesh)
{
    u64 localOffset = 0, localLength = 0, remoteLength = 0, remoteOffset = 0;
    u32 offsetCounter = 0;

    if (isSingleMesh) {
        localOffset = myMeshAggregationSendRecvInfo[userRankInfo.userRank].recvOffset[infoIndex];
        localLength = myMeshAggregationSendRecvInfo[userRankInfo.userRank].recvLength[infoIndex];
        remoteLength = myMeshAggregationSendRecvInfo[infoIndex].sendLength[userRankInfo.userRank];
        remoteOffset = myMeshAggregationSendRecvInfo[infoIndex].sendOffset[userRankInfo.userRank];
    } else {
        for (u32 j = userRankInfo.userRank % meshAggregationRankSize; j < userRankInfo.userRankSize;
             j += meshAggregationRankSize) {
            CalcIntraMeshAggregationRecvInfoInMeshAggregation(
                j, infoIndex, myMeshAggregationSendRecvInfo, localOffset, offsetCounter, localLength, remoteOffset,
                meshAggregationRankSize);
            if (offsetCounter == infoIndex || infoIndex == 0) {
                break;
            }
        }
        remoteLength = localLength;
    }
    curRecvInfo.localOffset = localOffset;
    curRecvInfo.localLength = localLength;

    curRecvInfo.remoteOffset = remoteOffset;
    curRecvInfo.remoteLength = remoteLength;
    HCCL_DEBUG(
        "[CalcIntraMeshAggregationRecvInfo] localOffset[%llu], localLength[%llu], "
        "remoteOffset[%llu], remoteLength[%llu]",
        localOffset, localLength, remoteOffset, remoteLength);
}

void CollAlltoAllExecutor::CalcIntraMeshAggregationAlltoAllMemInfo(
    const AlltoAllUserRankInfo& userRankInfo, const std::vector<SendRecvInfo>& allSendRecvInfo,
    std::map<u32, std::list<OneSendRecvAddrInfo>>& sendAddrInfosIntra,
    std::map<u32, std::list<OneSendRecvAddrInfo>>& recvAddrInfosIntra, u32 meshAggregationRankSize,
    const bool& isSingleMesh)
{
    sendAddrInfosIntra.clear();
    recvAddrInfosIntra.clear();
    if (allSendRecvInfo.size() != userRankInfo.userRankSize) {
        HCCL_ERROR(
            "Invalid All send recv info size[%zu], should be[%u]", allSendRecvInfo.size(), userRankInfo.userRankSize);
        return;
    }
    SendRecvInfo mySendRecvInfo = allSendRecvInfo[userRankInfo.userRank];
    u32 rankInMeshAggregation = userRankInfo.userRank % meshAggregationRankSize;
    u32 cluserIndex = userRankInfo.userRank / meshAggregationRankSize;
    auto itBegin = allSendRecvInfo.begin();
    auto itEnd = allSendRecvInfo.begin();
    std::advance(itBegin, cluserIndex * meshAggregationRankSize);
    std::advance(itEnd, (cluserIndex + 1) * meshAggregationRankSize);
    std::vector<SendRecvInfo> myMeshAggregationSendRecvInfo(itBegin, itEnd);

    for (u32 i = 0; i < userRankInfo.userRankSize; i++) {
        // sendInfo 的计算
        OneSendRecvAddrInfo curSendInfo;
        u32 remoteRankInMeshAggregation = i % meshAggregationRankSize;
        CalcIntraMeshAggregationSendInfo(
            userRankInfo, mySendRecvInfo, myMeshAggregationSendRecvInfo, rankInMeshAggregation, i, curSendInfo,
            meshAggregationRankSize, isSingleMesh);
        sendAddrInfosIntra[remoteRankInMeshAggregation].push_back(curSendInfo);

        // recvInfo 的计算
        OneSendRecvAddrInfo curRecvInfo;
        CalcIntraMeshAggregationRecvInfo(
            userRankInfo, myMeshAggregationSendRecvInfo, i, curRecvInfo, meshAggregationRankSize, isSingleMesh);
        recvAddrInfosIntra[remoteRankInMeshAggregation].push_back(curRecvInfo);
    }
}

HcclOpMetaInfo CollAlltoAllExecutor::GetOpMeta(HcclCMDType opType, const u64 size)
{
    bool hugeData = size > SDMA_SEND_MAX_SIZE;
    HcclOpMetaInfoDef opMeta;

    if (isAlltoAllZCopyMode_) {
        /* zcopy拆分4GB以上SDMA任务前，准备好子图不复用标志 */
        if (opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
            opMeta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::ZCOPY, size, hugeData);
        } else {
            opMeta = HcclOpMetaInfo::GetOneForAllToAllVC(CopyPattern::ZCOPY, size, hugeData);
        }
    } else {
        /* bcopy每次重新生成子图 */
        if (opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
            opMeta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::BCOPY, size, false);
        } else {
            opMeta = HcclOpMetaInfo::GetOneForAllToAllVC(CopyPattern::BCOPY, size, false);
        }
    }

    return opMeta;
}

u64 CollAlltoAllExecutor::CalAlltoAllVScratchMemSize(u64& workSpaceMemSize)
{
    u64 scratchMemSize = 0U;
    if (workSpaceMemSize == 0) {
        scratchMemSize = TINY_MEM_SIZE;
        HCCL_DEBUG("[CalAlltoAllVScratchMemSize] workSpaceMemSize==0, use TINY_MEM_SIZE[%llu]", TINY_MEM_SIZE);
    } else {
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            scratchMemSize = std::max(std::max(workSpaceMemSize, inCCLbufferSize_), TINY_MEM_SIZE);
            HCCL_DEBUG(
                "[CalAlltoAllVScratchMemSize] OpBase mode, workSpaceMemSize[%llu], "
                "inCCLbufferSize_[%llu], scratchMemSize[%llu]",
                workSpaceMemSize, inCCLbufferSize_, scratchMemSize);
        } else {
            scratchMemSize = workSpaceMemSize;
            HCCL_DEBUG(
                "[CalAlltoAllVScratchMemSize] non-OpBase mode, workSpaceMemSize[%llu], "
                "scratchMemSize[%llu]",
                workSpaceMemSize, scratchMemSize);
        }
    }
    return scratchMemSize;
}

bool CollAlltoAllExecutor::HasMassTasks(std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo)
{
    if (isAlltoAllZCopyMode_) {
        return false;
    }

    u64 maxSendTimes = 0;
    u64 maxRecvTimes = 0;
    const u64 cclBufferSize = algResResp_->cclInputMem.size();
    for (auto& sendRecvInfo : allMeshAggregationSendRecvInfo) {
        u64 sendTimes = 0;
        u64 recvTimes = 0;
        for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
            sendTimes += (sendRecvInfo.sendLength[i] + cclBufferSize - 1) / cclBufferSize;
            recvTimes += (sendRecvInfo.recvLength[i] + cclBufferSize - 1) / cclBufferSize;
        }
        maxSendTimes = (maxSendTimes > sendTimes) ? maxSendTimes : sendTimes;
        maxRecvTimes = (maxRecvTimes > recvTimes) ? maxRecvTimes : recvTimes;
    }
    const u64 massThreshold = 65535; //  65535: 单个ffts+任务中，最多承载64K个task
    const u64 maxTasksPerStep = 10;  // BCOPY中每次和远端通信最多消耗task数
    const u64 maxTasksBaseCost = 50; // BCOPY中除每步和远端通信外，最多消耗的task数
    u64 maxTasks = (maxSendTimes + maxRecvTimes) * maxTasksPerStep + maxTasksBaseCost;
    HCCL_DEBUG(
        "[AlltoAll] bcopy maxSendTimes[%llu], maxRecvTimes[%llu], maxTasks[%llu], hasMassTask[%u]", maxSendTimes,
        maxRecvTimes, maxTasks, (maxTasks > massThreshold));
    return (maxTasks > massThreshold);
}

HcclResult CollAlltoAllExecutor::SetVirtualDispatcher(const HcclDispatcher virtualDispatcher)
{
    vDispatcher_ = virtualDispatcher;
    return HCCL_SUCCESS;
}

HcclResult
CollAlltoAllExecutor::CheckNeedRecreateComm([[maybe_unused]] u64 lastScratchMemSize, bool& needRecreateAlltoallComm)
{
    needRecreateAlltoallComm = false;
    return HCCL_SUCCESS;
}

HcclResult
CollAlltoAllExecutor::RunAlltoAllTemplate(const std::unique_ptr<AlgTemplateBase>& executor, const SubCommInfo& commInfo)
{
    HcclResult ret = executor->RunAsync(commInfo.localRank, commInfo.localRankSize, commInfo.links);
    CHK_PRT_RET(
        ret == HCCL_E_AGAIN,
        HCCL_WARNING("[CollAlltoAllExecutor][RunAlltoAllTemplate]"
                     "group has been destroyed. Break!"),
        ret);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAlltoAllExecutor][RunAlltoAllTemplate]run executor rank[%u] rank size[%u] failed", commInfo.localRank,
            commInfo.localRankSize),
        ret);
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllExecutor::RunAlltoAllVTemplateStaged(
    const std::unique_ptr<AlgTemplateBase>& executor, const SubCommInfo& commInfo)
{
    HcclResult ret = executor->RunAsync(commInfo.localRank, commInfo.localRankSize, commInfo.links);
    CHK_PRT_RET(
        ret == HCCL_E_AGAIN,
        HCCL_WARNING("[CollAlltoAllExecutor][RunAlltoAllVTemplateStaged]"
                     "group has been destroyed. Break!"),
        ret);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAlltoAllExecutor][RunAlltoAllVTemplateStaged]run executor rank[%u] rank size[%u] failed",
            commInfo.localRank, commInfo.localRankSize),
        ret);
    return HCCL_SUCCESS;
}

// deprecated
HcclResult CollAlltoAllExecutor::RunTemplateWithVirtualLink(
    const std::unique_ptr<AlgTemplateBase>& executor, const SubCommInfo& commInfo)
{
    HcclResult ret = executor->RunAsync(commInfo.localRank, commInfo.localRankSize, commInfo.virtualLinks);
    CHK_PRT_RET(
        ret == HCCL_E_AGAIN,
        HCCL_WARNING("[CollAlltoAllExecutor][RunTemplateWithVirtualLink]"
                     "group has been destroyed. Break!"),
        ret);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAlltoAllExecutor][RunTemplateWithVirtualLink]run executor rank[%u] rank size[%u] failed",
            commInfo.localRank, commInfo.localRankSize),
        ret);
    return HCCL_SUCCESS;
}
void CollAlltoAllExecutor::SetWaitFlagTimeoutSec(u32 waitFlagTimeoutSec) { timeOut_ = waitFlagTimeoutSec; }
} // namespace hccl
