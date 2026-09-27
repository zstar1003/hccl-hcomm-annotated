/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_v_staged_executor.h"
namespace hccl {

CollRunAlltoAllVStaged::CollRunAlltoAllVStaged(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{}

HcclResult CollRunAlltoAllVStaged::ParallelTaskLoaderProcess(
    const std::string& tag, Stream& stream, SubCommInfo& level0CommInfo, std::vector<Stream>& ringStreams) const
{
    (void)tag;
    u32 streamIndex;
    std::vector<Stream*> streamsPtr;
    streamsPtr.resize(ringStreams.size() + 1);

    for (streamIndex = 0; streamIndex < ringStreams.size(); streamIndex++) { // slaveStreams
        streamsPtr[streamIndex] = &ringStreams[streamIndex];
    }
    streamsPtr[streamIndex] = &stream;

    HCCL_INFO("[ParallelTaskLoaderProcess]main stream[%p], streams size[%zu]", stream.ptr(), streamsPtr.size());

    // 准备多线程启动参数
    CHK_RET(parallelTaskLoader_->Prepare(streamsPtr, level0CommInfo));

    // 启动多线程处理
    CHK_RET(parallelTaskLoader_->StartTaskLoad());

    // 等待多线程处理结果
    CHK_RET(parallelTaskLoader_->WaitTaskLoadFinish());

    // 销毁通信域
    CHK_RET(parallelTaskLoader_->ClearTagCommInfo());
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVStaged::CalcStreamNum(u32& streamNum)
{
    streamNum = 0U;
    if (FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(
            topoAttr_.deviceType, topoAttr_.userRankSize, topoAttr_.useSuperPodMode,
            topoMatcher_->GetAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL))) {
        streamNum = topoAttr_.meshAggregationRankSize - 1;
    } else {
        if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE || isAlltoAllZCopyMode_) {
            std::vector<HcclAlgoType> algoTypeArr = topoMatcher_->GetAlgoConfig();
            if ((algoTypeArr[HCCL_ALGO_LEVEL_0] != HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE
                 || algoTypeArr[HCCL_ALGO_LEVEL_1] != HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE)
                && const_cast<HcclTopoInfo&>(topoAttr_)
                           .pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)]
                       == 0
                && topoAttr_.meshAggregationRankSize != 1) {
                streamNum = topoAttr_.meshAggregationRankSize - MINORS_NUM_TWO;
            }
        }
    }

    HCCL_INFO("[CollRunAlltoAllVStaged][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

void CollRunAlltoAllVStaged::CalcWorkSpaceMemSize(
    const AlltoAllUserRankInfo& userRankInfo, const std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo,
    u64& workspaceMemSize, u32 meshAggregationRankSize) const
{
    if (allMeshAggregationSendRecvInfo.size() % meshAggregationRankSize != 0
        || allMeshAggregationSendRecvInfo.size() == 0) {
        workspaceMemSize = 0;
        HCCL_ERROR("Invalid Send Recv Info Size[%zu]", allMeshAggregationSendRecvInfo.size());
        return;
    }
    workspaceMemSize = 0;
    u32 meshAggregationIndex = userRankInfo.userRank / meshAggregationRankSize;
    u32 meshAggregationRankBegin = meshAggregationIndex * meshAggregationRankSize;
    for (u32 infoIndex = userRankInfo.userRank % meshAggregationRankSize; infoIndex < userRankInfo.userRankSize;
         infoIndex += meshAggregationRankSize) {
        for (u32 k = meshAggregationRankBegin; k < meshAggregationRankBegin + meshAggregationRankSize; k++) {
            workspaceMemSize += allMeshAggregationSendRecvInfo[k].sendLength[infoIndex];
        }
    }
    HCCL_INFO("[AlltoAllVStagedCalculator][CalcWorkSpaceMemSize] workspaceMemSize[%llu]", workspaceMemSize);
}

HcclResult CollRunAlltoAllVStaged::CalcScratchMemSize(u64& scratchMemSize)
{
    scratchMemSize = 0U;
    u64 maxWorkSpaceMemSize = 0;
    AlltoAllUserRankInfo tmpUserRankInfo;
    tmpUserRankInfo.userRankSize = topoAttr_.userRankSize;
    tmpUserRankInfo.userRank = INVALID_VALUE_RANKID;

    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u64 workSpaceMemSize = 0;
        tmpUserRankInfo.userRank = topoAttr_.userRank;
        CalcWorkSpaceMemSize(
            tmpUserRankInfo, allMeshAggregationSendRecvInfo_, workSpaceMemSize, topoAttr_.meshAggregationRankSize);
        scratchMemSize = CalAlltoAllVScratchMemSize(workSpaceMemSize);
    } else {
        if (isAlltoAllZCopyMode_ && topoAttr_.isSingleMeshAggregation) {
            scratchMemSize = 0;
            HCCL_INFO("[CollRunAlltoAllVStaged][CalcScratchMemSize] scratchMemSize[%llu]", scratchMemSize);
            return HCCL_SUCCESS;
        }
        for (u32 rank = 0; rank < topoAttr_.userRankSize; rank++) {
            u64 workSpaceMemSize = 0;
            tmpUserRankInfo.userRank = rank;
            CalcWorkSpaceMemSize(
                tmpUserRankInfo, allMeshAggregationSendRecvInfo_, workSpaceMemSize, topoAttr_.meshAggregationRankSize);
            maxWorkSpaceMemSize = std::max(workSpaceMemSize, maxWorkSpaceMemSize);
        }
        scratchMemSize = CalAlltoAllVScratchMemSize(maxWorkSpaceMemSize);
        HCCL_DEBUG(
            "[CollRunAlltoAllVStaged][CalcScratchMemSize] OpBase branch, "
            "maxWorkSpaceMemSize[%llu], scratchMemSize[%llu]",
            maxWorkSpaceMemSize, scratchMemSize);
    }

    HCCL_INFO(
        "[CollRunAlltoAllVStaged][CalcScratchMemSize] workflowMode[%d], scratchMemSize[%llu]", workflowMode_,
        scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVStaged::CheckNeedRecreateComm(u64 lastScratchMemSize, bool& needRecreateAlltoallComm)
{
    u64 tmpScratchMemSize = 0;
    CHK_RET(CalcScratchMemSize(tmpScratchMemSize));
    needRecreateAlltoallComm = ((lastScratchMemSize < tmpScratchMemSize) ? (true) : (false));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVStaged::CheckNeedCreateVirtualLinks(AlgResourceRequest& resourceRequest)
{
    bool alltoallMeshReadOnly = FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(
        topoAttr_.deviceType, topoAttr_.userRankSize, topoAttr_.useSuperPodMode,
        topoMatcher_->GetAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL));
    HCCL_DEBUG(
        "[CollRunAlltoAllVStaged][CheckNeedCreateVirtualLinks] AllToAllVMeshReadOnly[%d],"
        "resourceRequest.streamNum[%u], GetExternalInputHcclEnableFfts()[%d], isAlltoAllZCopyMode_[%d]",
        alltoallMeshReadOnly, resourceRequest.streamNum, GetExternalInputHcclEnableFfts(), isAlltoAllZCopyMode_);
    if (!alltoallMeshReadOnly && (resourceRequest.streamNum != 0) && (!GetExternalInputHcclEnableFfts())
        && isAlltoAllZCopyMode_) {
        for (auto& levelNSubCommTransport : resourceRequest.opTransport) {
            for (auto& singleSubCommTransport : levelNSubCommTransport) {
                singleSubCommTransport.needVirtualLink = true;
                HCCL_INFO("[CollRunAlltoAllVStaged][CheckNeedCreateVirtualLinks] needVirtualLink is true");
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVStaged::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_MESH_L0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_MESH_L0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVStaged::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_MESH_L1, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_MESH_L1], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVStaged::CalcLevel2CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVStaged::CalStagedAlltoallVCommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    (void)inputType;
    (void)outputType;
    // 将网卡初始化判断，提到上层调用，减少无必要的循环依赖。
    bool alltoallMeshReadOnly = FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(
        topoAttr_.deviceType, topoAttr_.userRankSize, topoAttr_.useSuperPodMode,
        topoMatcher_->GetAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL));

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !isAlltoAllZCopyMode_) { // 单算子 && BCopy模式
        HCCL_INFO("cal comm in opbase and Bcopy mode");
        CHK_RET(CalcLevel0CommInfo(TransportMemType::CCL_INPUT, TransportMemType::CCL_OUTPUT, opTransport));
        CHK_RET(CalcLevel1CommInfo(TransportMemType::CCL_INPUT, TransportMemType::CCL_OUTPUT, opTransport));
        CHK_RET(CalcLevel2CommInfo(TransportMemType::CCL_INPUT, TransportMemType::CCL_OUTPUT, opTransport));
    } else if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isAlltoAllZCopyMode_) { // 单算子 &&
                                                                                                        // ZCopy模式
        HCCL_INFO("cal comm in opbase and Zcopy mode");
        if (topoAttr_.isSingleMeshAggregation) {
            CHK_RET(CalcLevel0CommInfo(TransportMemType::CCL_INPUT, TransportMemType::CCL_OUTPUT, opTransport));
        } else {
            CHK_RET(CalcLevel0CommInfo(
                TransportMemType::CCL_INPUT,
                (alltoallMeshReadOnly ? TransportMemType::CCL_OUTPUT : TransportMemType::SCRATCH), opTransport));
            CHK_RET(CalcLevel1CommInfo(TransportMemType::SCRATCH, TransportMemType::CCL_OUTPUT, opTransport));
        }
        CHK_RET(CalcLevel2CommInfo(TransportMemType::CCL_INPUT, TransportMemType::CCL_OUTPUT, opTransport));
    } else {
        HCCL_INFO("cal comm in graph mode");
        CHK_RET(CalcLevel0CommInfo(TransportMemType::PARAM_INPUT, TransportMemType::SCRATCH, opTransport));
        CHK_RET(CalcLevel1CommInfo(TransportMemType::SCRATCH, TransportMemType::PARAM_OUTPUT, opTransport));
        CHK_RET(CalcLevel2CommInfo(TransportMemType::PARAM_INPUT, TransportMemType::PARAM_OUTPUT, opTransport));
    }
    HCCL_DEBUG("[CollRunAlltoAllVStaged][CalStagedAlltoallVCommInfo] ends");
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVStaged::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;

    CHK_RET(CalStagedAlltoallVCommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVStaged::PrepareAlltoAllVStaged1(
    DeviceMem& sendBuf, DeviceMem& recvBuf, DeviceMem& scratchMem,
    std::map<u32, std::list<OneSendRecvAddrInfo>>& sendAddrInfosIntra,
    std::map<u32, std::list<OneSendRecvAddrInfo>>& recvAddrInfosIntra, Stream& stream, const std::string& tag,
    std::unique_ptr<AlgTemplateBase>& alltoallLevel0, ExecMem& execMem)
{
    (void)tag;
    // opbase BCopy 不支持fullmesh算法，因此不必做算法选择
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && !isAlltoAllZCopyMode_) { // 单算子 && Buffer拷贝模式
        HCCL_INFO("Running AllToAllV Staged Pairwise intra Server");
        alltoallLevel0 = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_2_ALL_V_STAGED_PAIRWISE, dispatcher_);
        CHK_SMART_PTR_NULL(alltoallLevel0);
        CHK_RET(alltoallLevel0->Prepare(
            sendBuf, scratchMem, execMem.inputMem, execMem.outputMem, sendAddrInfosIntra, recvAddrInfosIntra,
            isAlltoAllZCopyMode_, stream));
    } else {
        bool isOpBaseZCopy = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isAlltoAllZCopyMode_;
        DeviceMem inBuf = (isOpBaseZCopy) ? execMem.inputMem : sendBuf;
        // 单MeshAggregation下, 分级算法不做第二级, 结果输出到outCCLbuffer_
        DeviceMem outBuf = (isOpBaseZCopy && topoAttr_.isSingleMeshAggregation) ? recvBuf : scratchMem;
        // opbase ZCopy 与 graph，除input buffer差异外，其余行为应保持一致
        if (isOpBaseZCopy) { // 单算子 && ZCopy模式
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.inputMem, sendBuf, stream));
        }
        std::vector<HcclAlgoType> algoTypeArr = topoMatcher_->GetAlgoConfig();
        // 互联场景, alltoall暂不支持走fullmesh+pairwise
        if ((algoTypeArr[HCCL_ALGO_LEVEL_0] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE
             && algoTypeArr[HCCL_ALGO_LEVEL_1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE)
            || const_cast<HcclTopoInfo&>(topoAttr_).pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)]
                   != 0
            || topoAttr_.meshAggregationRankSize == 1) {
            HCCL_INFO("Running AllToAllV Staged Pairwise intra Server");
            alltoallLevel0 = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_2_ALL_V_STAGED_PAIRWISE, dispatcher_);
            CHK_SMART_PTR_NULL(alltoallLevel0);
            CHK_RET(alltoallLevel0->Prepare(
                inBuf, outBuf, sendAddrInfosIntra, recvAddrInfosIntra, isAlltoAllZCopyMode_, stream));
        } else {
            HCCL_INFO("Running AllToAllV Staged Mesh intra Server");
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
                CHK_RET(ActiveSlaveStreams(AlltoAllVParam_.stream));
            }
            if (GetExternalInputHcclEnableFfts() || algResResp_->slaveStreams.size() == 0) {
                alltoallLevel0 = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_2_ALL_V_STAGED_MESH, dispatcher_);
            } else {
                alltoallLevel0 = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_2_ALL_V_STAGED_MESH, vDispatcher_);
            }
            CHK_SMART_PTR_NULL(alltoallLevel0);
            CHK_RET(alltoallLevel0->Prepare(
                inBuf, outBuf, sendAddrInfosIntra, recvAddrInfosIntra, isAlltoAllZCopyMode_, topoAttr_.userRank, stream,
                algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux));
        }
    }
    return HCCL_SUCCESS;
}

void CollRunAlltoAllVStaged::CalcInterMeshAggregationRecvRemoteOffset(
    const AlltoAllUserRankInfo& userRankInfo, const std::vector<SendRecvInfo>& allSendRecvInfo, u32 index,
    u64& remoteOffset, u32 meshAggregationRankSize) const
{
    // 对于stage1 来说，相当于是从rand index 发送给 userRankInfo.userRank, 然后计算这种情况下的stage1 的接收偏移
    remoteOffset = 0;
    u32 anchoruserRank_ = index;
    u32 anchorIndex = userRankInfo.userRank;
    u32 beginIndex = anchorIndex % meshAggregationRankSize;
    u32 beginRank = anchoruserRank_ / meshAggregationRankSize * meshAggregationRankSize;
    bool getAnchor = false;
    for (index = beginIndex; index <= anchorIndex; index += meshAggregationRankSize) {
        for (u32 rank = beginRank; rank < beginRank + meshAggregationRankSize; rank++) {
            if (index == anchorIndex && rank == anchoruserRank_) {
                getAnchor = true;
                break;
            }
            remoteOffset += allSendRecvInfo[rank].sendLength[index];
        }
        if (getAnchor) {
            break;
        }
    }
}

void CollRunAlltoAllVStaged::CalcInterMeshAggregationAlltoAllMemInfo(
    const AlltoAllUserRankInfo& userRankInfo, const std::vector<SendRecvInfo>& allSendRecvInfo,
    std::map<u32, std::list<OneSendRecvAddrInfo>>& sendAddrInfosInter,
    std::map<u32, std::list<OneSendRecvAddrInfo>>& recvAddrInfosInter, u32 meshAggregationRankSize)
{
    sendAddrInfosInter.clear();
    recvAddrInfosInter.clear();

    u64 localOffsetMarker = 0;
    for (u32 toRank = 0; toRank < userRankInfo.userRankSize; toRank++) {
        u32 myRank = userRankInfo.userRank;
        u32 myMeshAggregationRankBegin = myRank / meshAggregationRankSize * meshAggregationRankSize;
        u32 myMeshAggregationRankEnd = myMeshAggregationRankBegin + meshAggregationRankSize;

        for (u32 myMeshAggregationRank = myMeshAggregationRankBegin; myMeshAggregationRank < myMeshAggregationRankEnd;
             myMeshAggregationRank++) {
            if (toRank % meshAggregationRankSize == myRank % meshAggregationRankSize) {
                OneSendRecvAddrInfo sendAddrInfo;
                sendAddrInfo.localLength = allSendRecvInfo[myMeshAggregationRank].sendLength[toRank];
                sendAddrInfo.localOffset = localOffsetMarker;
                localOffsetMarker += sendAddrInfo.localLength;
                sendAddrInfo.remoteOffset = allSendRecvInfo[toRank].recvOffset[myMeshAggregationRank];
                sendAddrInfo.remoteLength = allSendRecvInfo[toRank].recvLength[myMeshAggregationRank];
                u32 remoteRankInter = toRank / meshAggregationRankSize;
                sendAddrInfosInter[remoteRankInter].push_back(sendAddrInfo);
                HCCL_DEBUG(
                    "[CalcInterMeshAggregationAlltoAllMemInfo] sendAddrInfo localOffset[%llu], "
                    "localLength[%llu], remoteOffset[%llu], remoteLength[%llu]",
                    sendAddrInfo.localOffset, sendAddrInfo.localLength, sendAddrInfo.remoteOffset,
                    sendAddrInfo.remoteLength);
            }
        }
    }

    //  构造接收数据结构
    for (u32 index = 0; index < userRankInfo.userRankSize; index++) {
        OneSendRecvAddrInfo recvAddrInfo;
        u32 meshAggregationIndex = index / meshAggregationRankSize;

        recvAddrInfo.localOffset = allSendRecvInfo[userRankInfo.userRank].recvOffset[index];
        recvAddrInfo.localLength = allSendRecvInfo[userRankInfo.userRank].recvLength[index];
        // index 是 从那个rank 来的
        recvAddrInfo.remoteLength = allSendRecvInfo[index].sendLength[userRankInfo.userRank];
        u64 remoteOffset = 0;
        CalcInterMeshAggregationRecvRemoteOffset(
            userRankInfo, allSendRecvInfo, index, remoteOffset, meshAggregationRankSize);

        recvAddrInfo.remoteOffset = remoteOffset;
        recvAddrInfosInter[meshAggregationIndex].push_back(recvAddrInfo);
        HCCL_DEBUG(
            "[CalcInterMeshAggregationAlltoAllMemInfo] recvAddrInfo localOffset[%llu], "
            "localLength[%llu], remoteOffset[%llu], remoteLength[%llu]",
            recvAddrInfo.localOffset, recvAddrInfo.localLength, recvAddrInfo.remoteOffset, recvAddrInfo.remoteLength);
    }
}

HcclResult CollRunAlltoAllVStaged::PrepareAlltoAllVStaged2(
    DeviceMem& recvBuf, DeviceMem& scratchMem, std::map<u32, std::list<OneSendRecvAddrInfo>>& sendAddrInfosInter,
    std::map<u32, std::list<OneSendRecvAddrInfo>>& recvAddrInfosInter, Stream& stream, const std::string& tag,
    std::unique_ptr<AlgTemplateBase>& alltoallLevel1, ExecMem& execMem) const
{
    (void)tag;
    alltoallLevel1 = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_2_ALL_V_STAGED_PAIRWISE, dispatcher_);
    CHK_SMART_PTR_NULL(alltoallLevel1);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !isAlltoAllZCopyMode_) { // 单算子 && BCopy模式
        CHK_RET(alltoallLevel1->Prepare(
            scratchMem, recvBuf, execMem.inputMem, execMem.outputMem, sendAddrInfosInter, recvAddrInfosInter,
            isAlltoAllZCopyMode_, stream));
    } else if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isAlltoAllZCopyMode_) { // 单算子 &&
                                                                                                        // ZCopy模式
        CHK_RET(alltoallLevel1->Prepare(
            scratchMem, execMem.outputMem, execMem.inputMem, execMem.outputMem, sendAddrInfosInter, recvAddrInfosInter,
            isAlltoAllZCopyMode_, stream));
    } else {
        CHK_RET(alltoallLevel1->Prepare(
            scratchMem, recvBuf, sendAddrInfosInter, recvAddrInfosInter, isAlltoAllZCopyMode_, stream));
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVStaged::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollRunAlltoAllVStaged][KernelRun] AllToAllV staged starts");
    CHK_PRT_RET(
        topoAttr_.userRankSize % topoAttr_.meshAggregationRankSize != 0,
        HCCL_ERROR(
            "userRankSize[%u] is not an Integer multiple of MeshAggregation Dev Num[%u]", topoAttr_.userRankSize,
            topoAttr_.meshAggregationRankSize),
        HCCL_E_PARA);

    AlltoAllUserRankInfo userRankInfo;
    userRankInfo.userRank = topoAttr_.userRank;
    userRankInfo.userRankSize = topoAttr_.userRankSize;
    bool alltoallMeshReadOnly = FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(
        topoAttr_.deviceType, topoAttr_.userRankSize, topoAttr_.useSuperPodMode,
        topoMatcher_->GetAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL));

    std::map<u32, std::list<OneSendRecvAddrInfo>> sendAddrInfosIntra;
    std::map<u32, std::list<OneSendRecvAddrInfo>> recvAddrInfosIntra;
    bool isSingleMesh = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isAlltoAllZCopyMode_
                        && topoAttr_.isSingleMeshAggregation;
    CalcIntraMeshAggregationAlltoAllMemInfo(
        userRankInfo, allMeshAggregationSendRecvInfo_, sendAddrInfosIntra, recvAddrInfosIntra,
        topoAttr_.meshAggregationRankSize, isSingleMesh);

    CHK_RET(CheckCommSize(COMM_MESH_L0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_MESH_L0, COMM_INDEX_0);

    std::unique_ptr<AlgTemplateBase> alltoallLevel0 = nullptr;
    CHK_RET(PrepareAlltoAllVStaged1(
        algResResp_->paramInputMem, algResResp_->paramOutputMem, execMem.scratchMem, sendAddrInfosIntra,
        recvAddrInfosIntra, const_cast<Stream&>(param.stream), tag_, alltoallLevel0, execMem));
    if ((algResResp_->slaveStreams.size() != 0) && (!GetExternalInputHcclEnableFfts()) && isAlltoAllZCopyMode_) {
        HCCL_INFO("[AlltoAllOperator][RunAlltoAllVStaged] staged 0 use parallel multi-thread delivery of tasks");
        CHK_RET(RunTemplateWithVirtualLink(alltoallLevel0, level0CommInfo));
        // 多流场景下，并行多线程下发task处理
        CHK_RET(ParallelTaskLoaderProcess(
            tag_, const_cast<Stream&>(param.stream), level0CommInfo, algResResp_->slaveStreams));
    } else {
        CHK_RET(RunAlltoAllVTemplateStaged(alltoallLevel0, level0CommInfo));
    }

    HCCL_INFO("[hcclImpl][RunAlltoAllVStaged] stage0 run success!");

    std::map<u32, std::list<OneSendRecvAddrInfo>> sendAddrInfosInter;
    std::map<u32, std::list<OneSendRecvAddrInfo>> recvAddrInfosInter;
    CalcInterMeshAggregationAlltoAllMemInfo(
        userRankInfo, allMeshAggregationSendRecvInfo_, sendAddrInfosInter, recvAddrInfosInter,
        topoAttr_.meshAggregationRankSize);

    if (((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isAlltoAllZCopyMode_)
         || alltoallMeshReadOnly)
        && topoAttr_.isSingleMeshAggregation) {
        HCCL_DEBUG("we don't need to do stage 2 when there is only one mesh aggregation");
        // we don't needtodo stage 2 when there is only one mesh aggregation
    } else {
        HCCL_INFO("[hcclImpl][RunAlltoAllVStaged] stage1 run starts!");
        CHK_RET(CheckCommSize(COMM_MESH_L1, COMM_INDEX_0 + 1));
        SubCommInfo level1CommInfo = GetSubCommInfo(COMM_MESH_L1, COMM_INDEX_0);
        std::unique_ptr<AlgTemplateBase> alltoallLevel1 = nullptr;
        CHK_RET(PrepareAlltoAllVStaged2(
            algResResp_->paramOutputMem, execMem.scratchMem, sendAddrInfosInter, recvAddrInfosInter,
            const_cast<Stream&>(param.stream), tag_, alltoallLevel1, execMem));
        CHK_RET(RunAlltoAllVTemplateStaged(alltoallLevel1, level1CommInfo));
    }

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isAlltoAllZCopyMode_
        && !topoAttr_.isSingleMeshAggregation) {
        DeviceMem srcMem = (execMem.outputMem).range(0, algResResp_->paramOutputMem.size());
        CHK_RET(
            HcclD2DMemcpyAsync(dispatcher_, algResResp_->paramOutputMem, srcMem, const_cast<Stream&>(param.stream)));
    }

    HCCL_INFO("[CollRunAlltoAllVStaged][kernelRun] AllToAllV staged ends");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("RunAlltoAllVStaged", AlltoAllVStaged, CollRunAlltoAllVStaged);
} // namespace hccl
