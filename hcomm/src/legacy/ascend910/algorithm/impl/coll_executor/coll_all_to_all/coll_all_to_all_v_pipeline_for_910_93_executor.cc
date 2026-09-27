/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_v_pipeline_for_910_93_executor.h"

namespace hccl {

CollAlltoAllVPipelineFor91093::CollAlltoAllVPipelineFor91093(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{}

HcclResult
CollAlltoAllVPipelineFor91093::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;
    HCCL_INFO(
        "[CollAlltoAllVPipelineFor91093][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllVPipelineFor91093::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    // level0 + level1 - Mesh建链
    CommParaInfo commParaCombineL1(COMM_COMBINE_L1, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaCombineL1, opTransport[COMM_COMBINE_L1], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllVPipelineFor91093::CalcLevel2CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    // level2 - Mesh建链
    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllVPipelineFor91093::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllVPipelineFor91093::CalcStreamNum(u32& streamNum)
{
    const u32 level0StreamNum = std::min(topoAttr_.deviceNumPerAggregation - 1, DEVICE_EIGHT);
    const u32 level2StreamNum = 1;                 // 先固定1条RDMA 流
    streamNum = level0StreamNum + level2StreamNum; // 最大流数量为9条, RDMA 1条, SDMA 最大8条
    HCCL_INFO(
        "[CollAlltoAllVPipelineFor91093]tag[%s] level0StreamNum[%u], level2StreamNum[%u], streamNum[%u]", tag_.c_str(),
        level0StreamNum, level2StreamNum, streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllVPipelineFor91093::CalLocalSendRecvInfo(const OpParam& param, SendRecvInfo& info)
{
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        CalcA2ASendRecvInfo(param, info);
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        CalcA2AvSendRecvInfo(param, info);
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        CalcA2AvcSendRecvInfo(param, info);
    } else {
        HCCL_ERROR("[CollAlltoAllVPipelineFor91093] get invalid optype.");
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllVPipelineFor91093::CalcA2ASendRecvInfo(const OpParam& param, SendRecvInfo& info)
{
    const u32 userRankSize = topoAttr_.userRankSize;
    info.sendCounts.resize(userRankSize);
    info.sendDispls.resize(userRankSize);
    info.recvCounts.resize(userRankSize);
    info.recvDispls.resize(userRankSize);
    u64 sdispl = 0, rdispl = 0;

    for (u32 i = 0; i < userRankSize; i++) {
        info.sendCounts[i] = param.All2AllDataDes.sendCount;
        info.recvCounts[i] = param.All2AllDataDes.sendCount;
        info.sendDispls[i] = sdispl;
        info.recvDispls[i] = rdispl;
        sdispl += param.All2AllDataDes.sendCount;
        rdispl += param.All2AllDataDes.sendCount;
    }
    if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {
        for (u32 i = 0; i < userRankSize; i++) {
            HCCL_DEBUG(
                "CalcA2ASendRecvInfo sendCounts[%d]=%ld, recvCounts[%d]=%ld, sdispls[%d]=%ld, rdispls[%d]=%ld", i,
                info.sendCounts[i], i, info.recvCounts[i], i, info.sendDispls[i], i, info.recvDispls[i]);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllVPipelineFor91093::CalcA2AvSendRecvInfo(const OpParam& param, SendRecvInfo& info)
{
    const bool hasRecvInfo = param.All2AllDataDes.recvCounts != nullptr;
    const u32 userRankSize = topoAttr_.userRankSize;

    info.sendCounts.resize(userRankSize);
    info.sendDispls.resize(userRankSize);

    if (hasRecvInfo) {
        info.recvCounts.resize(userRankSize);
        info.recvDispls.resize(userRankSize);
    }

    for (u32 i = 0; i < userRankSize; ++i) {
        info.sendCounts[i] = *(static_cast<const u64*>(param.All2AllDataDes.sendCounts) + i);
        info.sendDispls[i] = *(static_cast<const u64*>(param.All2AllDataDes.sdispls) + i);

        if (hasRecvInfo) {
            info.recvCounts[i] = *(static_cast<const u64*>(param.All2AllDataDes.recvCounts) + i);
            info.recvDispls[i] = *(static_cast<const u64*>(param.All2AllDataDes.rdispls) + i);
        }
    }

    if (UNLIKELY(HcclCheckLogLevel(DLOG_INFO))) {
        for (u32 i = 0; i < userRankSize; i++) {
            HCCL_DEBUG(
                "CalcA2AvSendRecvInfo sendCounts[%d]=%ld, recvCounts[%d]=%ld, sdispls[%d]=%ld, rdispls[%d]=%ld", i,
                info.sendCounts[i], i, info.recvCounts[i], i, info.sendDispls[i], i, info.recvDispls[i]);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllVPipelineFor91093::CalcA2AvcSendRecvInfo(const OpParam& param, SendRecvInfo& info)
{
    const u32 userRankSize = topoAttr_.userRankSize;
    const u32 userRank = topoAttr_.userRank;
    info.sendCounts.resize(userRankSize);
    info.sendDispls.resize(userRankSize);
    info.recvCounts.resize(userRankSize);
    info.recvDispls.resize(userRankSize);
    u64 sdispl = 0, rdispl = 0;

    u64* sendCountMatrix = static_cast<u64*>(param.All2AllDataDes.sendCountMatrix);
    for (u32 i = 0; i < userRankSize; i++) {
        info.sendCounts[i] = *(sendCountMatrix + userRank * userRankSize + i);
        info.recvCounts[i] = *(sendCountMatrix + userRank + userRankSize * i);
        info.sendDispls[i] = sdispl;
        info.recvDispls[i] = rdispl;
        sdispl += *(sendCountMatrix + userRank * userRankSize + i);
        rdispl += *(sendCountMatrix + userRank + userRankSize * i);
    }
    if (UNLIKELY(HcclCheckLogLevel(DLOG_INFO))) {
        for (u32 i = 0; i < userRankSize; i++) {
            HCCL_DEBUG(
                "CalcA2AvcSendRecvInfo sendCounts[%d]=%ld, recvCounts[%d]=%ld, sdispls[%d]=%ld, rdispls[%d]=%ld", i,
                info.sendCounts[i], i, info.recvCounts[i], i, info.sendDispls[i], i, info.recvDispls[i]);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllVPipelineFor91093::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;
    tag_ = param.tag;
    algResResp_ = &algRes;
    AlltoAllVParam_ = param;
    ExecMem execMem;
    execMem.count = 0;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    execMem.inputMem = algRes.cclInputMem;
    execMem.outputMem = algRes.cclOutputMem;
    ret = KernelRun(param, execMem);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAlltoAllVPipelineFor91093][Orchestrate]errNo[0x%016llx]executor run failed", HCCL_ERROR_CODE(ret)),
        ret);

    HCCL_INFO(
        "tag[%s], CollAlltoAllVPipelineFor91093 tempAlg orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllVPipelineFor91093::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAlltoAllVPipelineFor91093][KernelRun] AllToAllV npu direct start.");
    // 获取通信域
    SubCommInfo level1CommInfo;
    SubCommInfo level2CommInfo;
    CHK_RET(CheckCommSize(COMM_COMBINE_L1, COMM_INDEX_0 + 1));
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    level1CommInfo = GetSubCommInfo(COMM_COMBINE_L1, COMM_INDEX_0);
    level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    // 执行
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_2_ALL_V_CONTINUOUS_PIPELINE, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);

    SendRecvInfo sendRecvInfo;
    CHK_RET(CalLocalSendRecvInfo(param, sendRecvInfo));

    A2aPipelineMemory a2aPipelineMemory;
    a2aPipelineMemory.userInput = algResResp_->paramInputMem;
    a2aPipelineMemory.userOutput = algResResp_->paramOutputMem;
    a2aPipelineMemory.cclInBuffer = execMem.inputMem;
    a2aPipelineMemory.cclOutBuffer = execMem.outputMem;

    HCCL_INFO(
        "[CollAlltoAllVPipelineFor91093] Memory info[addr, size]: userInput[%p, %llu], userOutput[%p, %llu], "
        "cclInBuffer[%p, %llu] ,cclOutBuffer[%p, %llu].",
        a2aPipelineMemory.userInput.ptr(), a2aPipelineMemory.userInput.size(), a2aPipelineMemory.userOutput.ptr(),
        a2aPipelineMemory.userOutput.size(), a2aPipelineMemory.cclInBuffer.ptr(), a2aPipelineMemory.cclInBuffer.size(),
        a2aPipelineMemory.cclOutBuffer.ptr(), a2aPipelineMemory.cclOutBuffer.size());

#ifndef OPEN_HCCL_TEST
    std::vector<SendRecvInfo> sendRecvInfoList{sendRecvInfo};
#else
    // 适配算法检查器，传入全局的info
    std::vector<SendRecvInfo> sendRecvInfoList = allMeshAggregationSendRecvInfo_;
#endif

    CHK_RET(tempAlg->Prepare(
        topoAttr_.userRank, a2aPipelineMemory, level1CommInfo, level2CommInfo, param.stream, algResResp_->slaveStreams,
        algResResp_->notifiesMain, algResResp_->notifiesAux, sendRecvInfoList, param.All2AllDataDes.sendType,
        workflowMode_, timeOut_));

    CHK_RET(tempAlg->RunAsync());

    HCCL_INFO("[CollAlltoAllVPipelineFor91093] executor run success.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("RunAlltoAllVPipelineFor91093", AlltoAllVPipelineFor91093, CollAlltoAllVPipelineFor91093);
} // namespace hccl
