/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_v_continuous_pipeline_executor.h"

namespace hccl {

CollAlltoAllVContinuousPipeline::CollAlltoAllVContinuousPipeline(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{}

template <typename T>
static void PrintListHelper(const T* listPtr, const size_t length, const std::string& prefix)
{
    std::ostringstream oss;
    oss << "[ ";
    for (size_t i = 0; i < length; ++i) {
        oss << listPtr[i] << " ";
    }
    oss << "]";
    HCCL_DEBUG("%s: %s", prefix.c_str(), oss.str().c_str());
}

static void PrintCountsAndDispls(const u32 length, const OpParam& param)
{
    // 打印counts和displs
    const u64* sendCountsPtr = static_cast<const u64*>(param.All2AllDataDes.sendCounts);
    const u64* sendDisplsPtr = static_cast<const u64*>(param.All2AllDataDes.sdispls);

    PrintListHelper(sendCountsPtr, length, "[PrintCountsAndDispls] sendCounts");
    PrintListHelper(sendDisplsPtr, length, "[PrintCountsAndDispls] sendDispls");
    if (param.All2AllDataDes.recvCounts != nullptr) {
        const u64* recvCountsPtr = static_cast<const u64*>(param.All2AllDataDes.recvCounts);
        const u64* recvDisplsPtr = static_cast<const u64*>(param.All2AllDataDes.rdispls);
        PrintListHelper(recvCountsPtr, length, "[PrintCountsAndDispls] recvCounts");
        PrintListHelper(recvDisplsPtr, length, "[PrintCountsAndDispls] recvDispls");
    }
}

HcclResult CollAlltoAllVContinuousPipeline::CalcStreamNum(u32& streamNum)
{
    const u32 level0StreamNum = topoAttr_.deviceNumPerAggregation - 1;
    const u32 level1StreamNum = 1; // 先固定1条

    streamNum = level0StreamNum + level1StreamNum;
    HCCL_INFO(
        "[CollAlltoAllVContinuousPipeline]tag[%s] level0StreamNum[%u], level1StreamNum[%u], streamNum[%u]",
        tag_.c_str(), level0StreamNum, level1StreamNum, streamNum);
    return HCCL_SUCCESS;
}

HcclResult
CollAlltoAllVContinuousPipeline::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;

    HCCL_INFO(
        "[CollAlltoAllVContinuousPipeline][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllVContinuousPipeline::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;

    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllVContinuousPipeline::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    // level0 - Mesh建链
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllVContinuousPipeline::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    // level1 - Mesh建链
    CommParaInfo commParaLevel1(COMM_LEVEL1, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel1, opTransport[COMM_LEVEL1], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllVContinuousPipeline::FillLocalSendRecvInfo(const OpParam& param, SendRecvInfo& info)
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

    if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {
        PrintCountsAndDispls(userRankSize, param);
    }

    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllVContinuousPipeline::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
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
            "[CollAlltoAllVContinuousPipeline][Orchestrate]errNo[0x%016llx]executor run failed", HCCL_ERROR_CODE(ret)),
        ret);

    HCCL_INFO(
        "tag[%s], CollAlltoAllVContinuousPipeline tempAlg orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllVContinuousPipeline::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAlltoAllVContinuousPipeline][KernelRun] AllToAllV npu direct start.");

    // 获取通信域
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    const u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    // 执行
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_2_ALL_V_CONTINUOUS_PIPELINE, dispatcher_);

    CHK_SMART_PTR_NULL(tempAlg);

    SendRecvInfo sendRecvInfo;
    CHK_RET(FillLocalSendRecvInfo(param, sendRecvInfo));

    A2aPipelineMemory a2aPipelineMemory;
    a2aPipelineMemory.userInput = algResResp_->paramInputMem;
    a2aPipelineMemory.userOutput = algResResp_->paramOutputMem;
    a2aPipelineMemory.cclInBuffer = execMem.inputMem;
    a2aPipelineMemory.cclOutBuffer = execMem.outputMem;

    HCCL_INFO(
        "[CollAlltoAllVContinuousPipeline] Memory info[addr, size]: userInput[%p, %llu], userOutput[%p, %llu], "
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
        topoAttr_.userRank, a2aPipelineMemory, level0CommInfo, level1CommInfo, param.stream, algResResp_->slaveStreams,
        algResResp_->notifiesMain, algResResp_->notifiesAux, sendRecvInfoList, param.All2AllDataDes.sendType,
        workflowMode_, timeOut_));

    CHK_RET(tempAlg->RunAsync());

    HCCL_INFO("[CollAlltoAllVContinuousPipeline] executor run success.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("RunAlltoAllVContinuousPipeline", AlltoAllVContinuousPipeline, CollAlltoAllVContinuousPipeline);
} // namespace hccl
