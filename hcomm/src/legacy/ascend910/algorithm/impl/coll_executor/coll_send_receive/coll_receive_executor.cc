/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_receive_executor.h"

namespace hccl {

CollReceiveExecutor::CollReceiveExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollNativeExecutorBase(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
}

HcclResult CollReceiveExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    HcclResult ret = HCCL_SUCCESS;
    // 图模式场景下不需要Loop
    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        DeviceMem outputMem = algRes.paramOutputMem;
        ret = RunTemplate(param, outputMem);
    } else {
        ret = RunLoop(param, algRes);
    }

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReceiveExecutor][Orchestrate]errNo[0x%016llx]recv executor kernel run failed", HCCL_ERROR_CODE(ret)),
        ret);

    HCCL_INFO(
        "tag[%s], Receive Executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollReceiveExecutor::GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo)
{
    (void)algRes;
    (void)adjInfo;
    return HCCL_SUCCESS;
}

HcclResult CollReceiveExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_OUTPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollRecvExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(), inputType,
        outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReceiveExecutor::CalcP2PCommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport,
    u32 srcRank)
{
    HCCL_INFO("[CollRecvExecutor][CalcLevel0CommInfo]tag[%s] start", tag_.c_str());
    CommParaInfo commP2P(COMM_COMBINE, CommType::COMM_TAG_P2P);
    commP2P.peerUserRank = srcRank;
    CHK_RET(CalcCommPlaneInfo(tag_, commP2P, opTransport[COMM_COMBINE], inputType, outputType));
    HCCL_INFO("[CollRecvExecutor][CalcLevel0CommInfo]tag[%s] Calc RingComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollReceiveExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport, u32 srcRank)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CalcTransportMemType(inputType, outputType);
    CalcP2PCommInfo(inputType, outputType, opTransport, srcRank);
    return HCCL_SUCCESS;
}

HcclResult CollReceiveExecutor::CalcResRequest(const OpParam& param, AlgResourceRequest& resourceRequest)
{
    ParseParam(param);

    u64 scratchMemSize = 0U;
    u32 streamNum = 0U;
    u32 notifyNum = 0U;
    u64 aivBufferRequest = 0U;
    std::vector<LevelNSubCommTransport> opTransport{
        std::vector<LevelNSubCommTransport>(static_cast<u32>(COMM_LEVEL_RESERVED))};

    CHK_RET(CalcCommInfo(opTransport, param.srcRank));

    CHK_RET(BuildResourceRequest(scratchMemSize, streamNum, notifyNum, aivBufferRequest, opTransport, resourceRequest));
    HCCL_INFO(
        "streamNum[%u], notifyNum[%u], scratchMemSize[%llu], aivBufferRequest[%llu]", resourceRequest.streamNum,
        resourceRequest.notifyNum, resourceRequest.scratchMemSize, resourceRequest.aivBufferRequest);
    // 打印建链诉求
    PrintTransportRequest(resourceRequest);
    return HCCL_SUCCESS;
}

HcclResult CollReceiveExecutor::RunLoop(OpParam& param, AlgResourceResponse& algRes)
{
    HcclResult ret;

    u64 commOutputSize = algRes.cclOutputMem.size();

    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];

    auto meta = HcclOpMetaInfo::GetOneForReceive();
    u8* curOutputPtr = static_cast<u8*>(param.outputPtr);
    CHK_PTR_NULL(curOutputPtr);

    u64 outputOffset = 0;
    u64 countLeft = param.DataDes.count;
    while (countLeft > 0) {
        CHK_RET(InitTask(dispatcher_, param.stream, meta.isEnableCache, meta.GetCacheKey()));
        curOutputPtr += outputOffset;
        HCCL_DEBUG("RecvOutPlace:outputOffset[%llu]", outputOffset);
        u64 curCount = ((countLeft * unitSize) > commOutputSize) ? (commOutputSize / unitSize) : countLeft;
        u64 curSize = curCount * unitSize; // 单位 byte
        HCCL_DEBUG("RecvOutPlace:curOutputPtr[%p], curCount[%llu], curSize[%llu]", curOutputPtr, curCount, curSize);

        if (topoAttr_.deviceType != DevType::DEV_TYPE_910_93 || topoAttr_.isDiffDeviceType
            || (topoAttr_.superPodNum > 1
                || (topoAttr_.moduleNum > 1 && static_cast<bool>(topoMatcher_->GetExternalInputInterHccsDisable())))) {
            // 非A3场景不做DMA消减；A3的RDMA场景，也不做DMA消减
            DeviceMem outCommMem(algRes.cclOutputMem.ptr(), curSize);
            DeviceMem outMem(curOutputPtr, curSize);
            ret = RunTemplate(param, outCommMem);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "errNo[0x%016llx] RecvOutPlace: recv error, tag[%s], ptr[%p], count[%llu], dataType[%d]",
                    HCCL_ERROR_CODE(ret), param.tag.c_str(), curOutputPtr, curCount, param.DataDes.dataType),
                ret);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outMem, outCommMem, param.stream));
            HCCL_DEBUG("[CollReceiveExecutor][RunLoop]copy from ccl output to user output.");
        } else {
            // A3的SDMA场景，recv端做DMA消减
            DeviceMem outMem(curOutputPtr, curSize);
            ret = RunTemplate(param, outMem);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "errNo[0x%016llx] RecvOutPlace: recv error, tag[%s], ptr[%p], count[%llu], dataType[%d]",
                    HCCL_ERROR_CODE(ret), param.tag.c_str(), curOutputPtr, curCount, param.DataDes.dataType),
                ret);
            HCCL_DEBUG("[CollReceiveExecutor][RunLoop]copy from ccl input to user output.");
        }

        CHK_PRT_RET((curCount == 0), HCCL_ERROR("In OP_BASE curCount is zero"), HCCL_E_PARA);
        countLeft -= curCount;
        outputOffset = curSize;

        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }
    return HCCL_SUCCESS;
}

HcclResult CollReceiveExecutor::RunTemplate(const OpParam& param, DeviceMem& outputMem)
{
    SubCommInfo commInfo = GetSubCommInfo(COMM_COMBINE, 0);
    if (commInfo.links.size() == 0) {
        HCCL_ERROR("[CollReceiveExecutor]links size is 0");
    }
    LINK transportLink = commInfo.links[0];

    SendReceive ReceiveExecutor(dispatcher_, transportLink);
    CHK_RET(ReceiveExecutor.ReceivePrepare(outputMem, param.srcRank, param.stream));
    CHK_RET(ReceiveExecutor.RegisterProfiler(0, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(ReceiveExecutor.ReceiveRunAsync());

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReceiveExecutor", Receive, CollReceiveExecutor);

} // namespace hccl
