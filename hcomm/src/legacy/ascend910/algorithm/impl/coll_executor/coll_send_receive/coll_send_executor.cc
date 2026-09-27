/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_send_executor.h"

namespace hccl {

CollSendExecutor::CollSendExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollNativeExecutorBase(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
}

HcclResult CollSendExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    HcclResult ret = HCCL_SUCCESS;
    // 图模式场景下不需要Loop
    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        DeviceMem InputMem = algRes.paramInputMem;
        ret = RunTemplate(param, InputMem);
    } else {
        ret = RunLoop(param, algRes);
    }

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollSendExecutor][Orchestrate]errNo[0x%016llx]send executor kernel run failed", HCCL_ERROR_CODE(ret)),
        ret);

    HCCL_INFO(
        "tag[%s] Send Executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollSendExecutor::GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo)
{
    algResResp_ = &algRes;

    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_COMBINE, 0);
    if (level1CommInfo.links.size() == 0) {
        HCCL_ERROR("[CollSendExecutor]links size is 0");
    }
    u32 localRank = level1CommInfo.localRank;
    u32 localRankSize = level1CommInfo.localRankSize;

    if (localRankSize == 1) {
        return HCCL_SUCCESS;
    }
    u32 ringNextRank = (localRank + 1) % localRankSize;

    adjInfo.dstRankNum = 1;
    NslbDpAdjInfo adjInfoStep = {};
    adjInfoStep.dstLocalRankId = ringNextRank;
    adjInfoStep.phaseId = 1;
    adjInfoStep.rev = 0;
    adjInfo.nsAdjInfo.push_back(adjInfoStep);

    HCCL_INFO("[NSLB-DP]GetAdjInfo localRank[%u], phaseId[%u].", localRank, ringNextRank);
    return HCCL_SUCCESS;
}

HcclResult CollSendExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_INPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollSendExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(), inputType,
        outputType);
    return HCCL_SUCCESS;
}

HcclResult CollSendExecutor::CalcP2PCommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport,
    u32 dstRank)
{
    HCCL_INFO("[CollSendExecutor][CalcLevel0CommInfo]tag[%s] start", tag_.c_str());
    CommParaInfo commP2P(COMM_COMBINE, CommType::COMM_TAG_P2P);
    commP2P.peerUserRank = dstRank;
    CHK_RET(CalcCommPlaneInfo(tag_, commP2P, opTransport[COMM_COMBINE], inputType, outputType));
    HCCL_INFO("[CollSendExecutor][CalcLevel0CommInfo]tag[%s] Calc P2PComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollSendExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport, u32 dstRank)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CalcTransportMemType(inputType, outputType);
    CHK_RET(CalcP2PCommInfo(inputType, outputType, opTransport, dstRank));
    return HCCL_SUCCESS;
}

HcclResult CollSendExecutor::CalcResRequest(const OpParam& param, AlgResourceRequest& resourceRequest)
{
    ParseParam(param);

    u64 scratchMemSize = 0U;
    u32 streamNum = 0U;
    u32 notifyNum = 0U;
    u64 aivBufferRequest = 0U;
    std::vector<LevelNSubCommTransport> opTransport{
        std::vector<LevelNSubCommTransport>(static_cast<u32>(COMM_LEVEL_RESERVED))};

    CHK_RET(CalcCommInfo(opTransport, param.dstRank));

    CHK_RET(BuildResourceRequest(scratchMemSize, streamNum, notifyNum, aivBufferRequest, opTransport, resourceRequest));
    HCCL_INFO(
        "streamNum[%u], notifyNum[%u], scratchMemSize[%llu], aivBufferRequest[%llu]", resourceRequest.streamNum,
        resourceRequest.notifyNum, resourceRequest.scratchMemSize, resourceRequest.aivBufferRequest);
    // 打印建链诉求
    PrintTransportRequest(resourceRequest);
    return HCCL_SUCCESS;
}

HcclResult CollSendExecutor::RunLoop(OpParam& param, AlgResourceResponse& algRes)
{
    HcclResult ret;

    u64 commInputSize = algRes.cclInputMem.size();

    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    auto meta = HcclOpMetaInfo::GetOneForSend();
    u8* curInputPtr = static_cast<u8*>(param.inputPtr);
    CHK_PTR_NULL(curInputPtr);

    u64 inputOffset = 0;
    u64 countLeft = param.DataDes.count;
    while (countLeft > 0) {
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), meta.isEnableCache, meta.GetCacheKey()));
        curInputPtr += inputOffset;

        HCCL_DEBUG("SendOutPlace:inputOffset[%llu]", inputOffset);
        u64 curCount = ((countLeft * unitSize) > commInputSize) ? (commInputSize / unitSize) : countLeft;
        u64 curSize = curCount * unitSize; // 单位 byte

        HCCL_DEBUG("SendOutPlace:curInputPtr[%p], curCount[%llu], curSize[%llu]", curInputPtr, curCount, curSize);

        // 非A3场景，或A3的SDMA消减场景，send端不消减
        DeviceMem inCommMem(algRes.cclInputMem.ptr(), curSize);
        DeviceMem inMem(curInputPtr, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCommMem, inMem, const_cast<Stream&>(param.stream)));
        HCCL_INFO(
            "[CollSendExecutor] inCommMem[%p]size[%lld], inMem[%p]size[%lld]", inCommMem.ptr(), inCommMem.size(),
            inMem.ptr(), inMem.size());
        ret = RunTemplate(param, inCommMem);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "errNo[0x%016llx] SendOutPlace: send error, tag[%s], ptr[%p], count[%llu], dataType[%d]",
                HCCL_ERROR_CODE(ret), param.tag.c_str(), curInputPtr, curCount, param.DataDes.dataType),
            ret);
        HCCL_DEBUG("[CollSendExecutor][RunLoop]copy from user input to ccl input.");

        CHK_PRT_RET((curCount == 0), HCCL_ERROR("In OP_BASE curCount is zero"), HCCL_E_PARA);
        countLeft -= curCount;
        inputOffset = curSize;

        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }
    return HCCL_SUCCESS;
}

HcclResult CollSendExecutor::RunTemplate(const OpParam& param, DeviceMem& inputMem)
{
    SubCommInfo commInfo = GetSubCommInfo(COMM_COMBINE, 0);
    if (commInfo.links.size() == 0) {
        HCCL_ERROR("[CollSendExecutor]links size is 0");
    }
    LINK transportLink = commInfo.links[0];

    SendReceive sendExecutor(dispatcher_, transportLink);
    CHK_RET(sendExecutor.SendPrepare(inputMem, param.dstRank, param.stream));
    CHK_RET(sendExecutor.RegisterProfiler(0, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(sendExecutor.SendRunAsync());

    return HCCL_SUCCESS;
}

REGISTER_EXEC("SendExecutor", Send, CollSendExecutor);

} // namespace hccl
