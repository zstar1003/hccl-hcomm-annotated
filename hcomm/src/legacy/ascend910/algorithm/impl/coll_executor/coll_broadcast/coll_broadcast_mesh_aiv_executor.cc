/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_broadcast_mesh_aiv_executor.h"

namespace hccl {

CollBroadcastMeshAivExecutor::CollBroadcastMeshAivExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
    desc_.isAivMode = true;
}

HcclResult CollBroadcastMeshAivExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = 0; // AIV通信不需要申请从流
    HCCL_INFO("[CollBroadcastMeshAivExecutor][CalcStreamNum] tag[%s] streamNum[%u].", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshAivExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType;
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::AIV_INPUT;
    } else {
        inputType = TransportMemType::CCL_INPUT;
    }
    TransportMemType outputType = TransportMemType::AIV_OUTPUT;
    HCCL_INFO(
        "[CollBroadcastMeshAivExecutor][CalcTransportMemType] tag[%s] inputType[%d],"
        " outputType[%d].",
        tag_.c_str(), inputType, outputType);
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshAivExecutor::CalNumBlocks(
    u32& numBlocks, u32 rankSize, [[maybe_unused]] u64 dataSize, [[maybe_unused]] HcclCMDType cmdType)
{
    numBlocks = rankSize; // 默认情况使用rankSize个AIV
    CHK_PRT_RET(
        numBlocks_ < numBlocks,
        HCCL_ERROR(
            "[CollBroadcastMeshAivExecutor][CalNumBlocks]aivCore[%u] is invalid, at least need [%u].", numBlocks_,
            numBlocks),
        HCCL_E_PARA);

    HCCL_INFO(
        "[CollBroadcastMeshAivExecutor][CalNumBlocks] numBlocks is set to [%u], limit[%u], recommended[%u]", numBlocks,
        numBlocks_, numBlocks);
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshAivExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshAivExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    HcclResult ret = HCCL_SUCCESS;
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.inputPtr; // broadcast使用一块内存

    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.inputMem = algRes.aivInputMem;
    } else {
        execMem.inputMem = algRes.cclInputMem;
    }
    execMem.outputMem = algRes.aivOutputMem;

    ret = KernelRun(param, execMem);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollBroadcastMeshAivExecutor][Orchestrate]errNo[0x%016llx] tag[%s] executor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()),
        ret);

    HCCL_INFO(
        "tag[%s], Broadcast executor orchestrate success, take time [%lld]us", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshAivExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_INFO("[CollBroadcastMeshAivExecutor][KernelRun]broadcast aiv enter.");

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    void* buffersIn[MAX_RANK_SIZE];
    void* buffersOut[MAX_RANK_SIZE];

    u32 rootRank = 0;
    HcclResult ret = GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, param.root, rootRank);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[BroadCastOperator][CollBroadcastMeshAivExecutor]invalid root[%u] to get userrank", param.root),
        ret);

    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG("[CollBroadcastMeshAivExecutor][KernelRun] userRank [%u] localRank [%u]", topoAttr_.userRank, localRank);
    for (u32 i = 0; i < localRankSize; i++) {
        if (i != localRank) {
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(buffersIn[i])));
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(buffersOut[i])));
        } else {
            buffersIn[i] = execMem.inputMem.ptr(); // ccl buffer
            buffersOut[i] = execMem.outputMem.ptr();
        }
    }

    bool isOpbase = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    AivOpArgs opArgs{HcclCMDType::HCCL_CMD_BROADCAST, execMem.inputPtr, execMem.outputPtr, execMem.count,
                     param.DataDes.dataType,          param.reduceType, rootRank,          isOpbase};
    AivTopoArgs topoArgs{localRank, localRankSize};
    topoArgs.identify = algoAttr_.identifier;
    u32 numBlocks;
    CHK_PRT_RET(
        CalNumBlocks(numBlocks, localRankSize) != HCCL_SUCCESS, HCCL_ERROR("[%s] CalNumBlocks failed", __func__),
        HCCL_E_PARA);
    numBlocks_ = numBlocks;
    AivResourceArgs resourceArgs{param.tag,  param.stream.ptr(), buffersIn, buffersOut, execMem.inputMem.size(),
                                 numBlocks_, param.aivTag};
    AivAlgArgs algArgs{};
    algArgs.execTimeOut = topoMatcher_->GetExecTimeOutConfig();
    algArgs.execTimeOutSet = true;
    struct AivProfilingInfo aivProfilingInfo;
    aivProfilingInfo.counter = opCounter_;
    ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo); // 执行kernelLaunch
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollBroadcastMeshAivExecutor][KernelRun]broadcast aiv failed, return[%d]", ret), ret);

    ExtraArgs extraArgs;
    CHK_RET(SetOpCache(opArgs, topoArgs, resourceArgs, algArgs, extraArgs, aivProfilingInfo, false));

    HCCL_INFO("[CollBroadcastMeshAivExecutor][KernelRun]broadcast aiv run success.");

    return HCCL_SUCCESS;
}

REGISTER_EXEC("BroadcastMeshAivExecutor", BroadcastMeshAiv, CollBroadcastMeshAivExecutor);

} // namespace hccl
