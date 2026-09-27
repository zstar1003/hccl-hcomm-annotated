/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_direct_fullmesh_aiv_executor.h"

namespace hccl {

CollAlltoAllDirectFullmeshAIVExecutor::CollAlltoAllDirectFullmeshAIVExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{
    desc_.isAivMode = true;
}

HcclResult CollAlltoAllDirectFullmeshAIVExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = 0; // AIV通信不需要申请从流
    HCCL_INFO("[CollAlltoAllDirectFullmeshAIVExecutor][%s] tag[%s] streamNum[%u].", __func__, tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllDirectFullmeshAIVExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllDirectFullmeshAIVExecutor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;
    HCCL_INFO(
        "[CollAlltoAllDirectFullmeshAIVExecutor][%s] tag[%s] inputType[%d], outputType[%d].", __func__, tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

// level0-level1 打平fullmesh
// RDMA链路
HcclResult CollAlltoAllDirectFullmeshAIVExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commCombinePara(COMM_COMBINE_ORDER, CommType::COMM_TAG_MESH);
    commCombinePara.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commCombinePara, opTransport[COMM_COMBINE_ORDER], inputType, outputType));

    LevelNSubCommTransport& commTransportLevel0 = opTransport[COMM_COMBINE_ORDER];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        for (auto& transportRequest : commTransportLevel0[subCommIndex].transportRequests) {
            transportRequest.isUsedRdma = true;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllDirectFullmeshAIVExecutor::PrepareCommInfoToDevice(AlgResourceResponse& algResource)
{
    HCCL_INFO("[CollAlltoAllDirectFullmeshAIVExecutor][PrepareCommInfoToDevice]"
              "alltoall aiv copy comm info to device.");
    CHK_RET(CopyAivCommInfoToDevice(COMM_COMBINE_ORDER, COMM_INDEX_0, algResource));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllDirectFullmeshAIVExecutor::CopyAivCommInfoToDevice(
    const CommPlane levelIndex, const u32 subLevelIndex, AlgResourceResponse& algResource)
{
    algResResp_ = &algResource;
    CHK_RET(CheckCommSize(levelIndex, subLevelIndex + 1));
    SubCommInfo commInfo = GetSubCommInfo(levelIndex, subLevelIndex);
    u32 localRank = commInfo.localRank;
    u32 localRankSize = commInfo.localRankSize;
    void* buffersInOut[MAX_RANK_SIZE_RDMA * 2] = {};

    for (u32 i = 0; i < localRankSize; i++) {
        u32 idx = (i << 1);
        if (i != localRank) {
            CHK_RET(commInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(buffersInOut[idx])));
            CHK_RET(commInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(buffersInOut[idx + 1])));
        } else {
            buffersInOut[idx] = algResource.cclInputMem.ptr();
            buffersInOut[idx + 1] = algResource.cclOutputMem.ptr();
        }
    }
    const u32 bufferNum = 2;
    CHK_RET(hrtMemSyncCopy(
        algResource.aivCommInfoMem.ptr(), sizeof(u64) * localRankSize * bufferNum, buffersInOut,
        sizeof(u64) * localRankSize * bufferNum, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllDirectFullmeshAIVExecutor::CalNumBlocks(
    u32& numBlocks, [[maybe_unused]] u32 rankSize, [[maybe_unused]] u64 dataSize, [[maybe_unused]] HcclCMDType cmdType)
{
    numBlocks = topoAttr_.moduleNum; // 默认情况使用serverNum个AIV
    CHK_PRT_RET(
        numBlocks_ < numBlocks,
        HCCL_ERROR(
            "[CollAlltoAllDirectFullmeshAIVExecutor][%s]aivCore[%u] is invalid, at least need[%u].", __func__,
            numBlocks_, numBlocks),
        HCCL_E_PARA);

    HCCL_INFO(
        "[CollAlltoAllDirectFullmeshAIVExecutor][%s] numBlocks is set to [%u], limit[%u], recommended[%u]", __func__,
        numBlocks, numBlocks_, numBlocks);
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllDirectFullmeshAIVExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    HcclResult ret = HCCL_SUCCESS;
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    execMem.inputMem = algRes.cclInputMem;
    execMem.outputMem = algRes.cclOutputMem;
    ret = KernelRun(param, execMem);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAlltoAllDirectFullmeshAIVExecutor][%s]errNo[0x%016llx] tag[%s] executor kernel run failed", __func__,
            HCCL_ERROR_CODE(ret), param.tag.c_str()),
        ret);

    HCCL_INFO(
        "tag[%s], AlltoAll executor orchestrate success, take time [%lld]us", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllDirectFullmeshAIVExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAlltoAllDirectFullmeshAIVExecutor][KernelRun]alltoall aiv enter.");

    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);

    void* buffersIn[MAX_RANK_SIZE];
    void* buffersOut[MAX_RANK_SIZE];

    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG(
        "[CollAlltoAllDirectFullmeshAIVExecutor][%s] userRank [%u] localRank [%u]", __func__, topoAttr_.userRank,
        localRank);

    buffersIn[0] = execMem.inputMem.ptr();
    buffersOut[0] = execMem.outputMem.ptr();
    constexpr u32 BUFFER_IDX_ONE = 1;
    buffersOut[BUFFER_IDX_ONE] = algResResp_->aivCommInfoMem.ptr(); // 交叉存储cclIn和cclOut

    u64 sendCount = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix));
    AivOpArgs opArgs{
        HcclCMDType::HCCL_CMD_ALLTOALL,
        execMem.inputPtr,
        execMem.outputPtr,
        sendCount,
        param.All2AllDataDes.sendType,
        HCCL_REDUCE_RESERVED,
        0,
        true};
    AivTopoArgs topoArgs{localRank, localRankSize, MAX_RANK_SIZE, 0, topoAttr_.serverNum, topoAttr_.deviceType};
    topoArgs.identify = algoAttr_.identifier;
    AivResourceArgs resourceArgs{param.tag, param.stream.ptr(), buffersIn, buffersOut, execMem.inputMem.size(),
                                 0,         param.aivTag};
    AivAlgArgs algArgs{0};
    algArgs.execTimeOut = topoMatcher_->GetExecTimeOutConfig();
    algArgs.execTimeOutSet = true;
    algArgs.isNpuDirectRoce = true;
    algArgs.rmaInfo = reinterpret_cast<uintptr_t>(rmaInfo_);
    struct AivProfilingInfo aivProfilingInfo;
    aivProfilingInfo.counter = opCounter_;

    u32 numBlocks;
    CHK_RET(CalNumBlocks(numBlocks, localRankSize));
    numBlocks_ = numBlocks;
    resourceArgs.numBlocks = numBlocks_;
    HcclResult ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlltoAllDirectFullmeshAIVExecutor][%s]alltoall aiv failed, return[%d]", __func__, ret), ret);

    HCCL_INFO("[CollAlltoAllDirectFullmeshAIVExecutor][%s]alltoall aiv run success.", __func__);
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AlltoAllDirectFullmeshAIVExecutor", AlltoAllDirectFullmeshAIV, CollAlltoAllDirectFullmeshAIVExecutor);

} // namespace hccl
