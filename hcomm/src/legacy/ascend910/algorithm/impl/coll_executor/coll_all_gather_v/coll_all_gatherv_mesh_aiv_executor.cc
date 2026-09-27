/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gatherv_mesh_aiv_executor.h"

namespace hccl {

AllGatherVMeshAivExecutor::AllGatherVMeshAivExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherVExecutor(dispatcher, topoMatcher)
{
    desc_.isAivMode = true;
}

HcclResult AllGatherVMeshAivExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult AllGatherVMeshAivExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType)
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::AIV_OUTPUT;
    HCCL_INFO(
        "[AllGatherVMeshAivExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d].", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult AllGatherVMeshAivExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult AllGatherVMeshAivExecutor::CalNumBlocks(
    u32& numBlocks, u32 rankSize, [[maybe_unused]] u64 dataSize, [[maybe_unused]] HcclCMDType cmdType)
{
    numBlocks = rankSize + 1; // 单机场景，单算子AllGather大数据使用(rankSize + 1)个aiv
    u32 bestNumBlocks = numBlocks;

    CHK_PRT_RET(
        numBlocks_ < numBlocks,
        HCCL_WARNING(
            "[AllGatherVMeshAivExecutor][CalNumBlocks]aivCore[%u] is invalid, at least need [%u].", numBlocks_,
            numBlocks),
        HCCL_E_PARA);

    HCCL_INFO(
        "[AllGatherVMeshAivExecutor][CalNumBlocks] numBlocks is set to [%u], limit[%u], recommended[%u]", numBlocks,
        numBlocks_, bestNumBlocks);
    return HCCL_SUCCESS;
}

HcclResult AllGatherVMeshAivExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    HcclResult ret = HCCL_SUCCESS;
    ExecMem execMem;

    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    execMem.count = (static_cast<const u64*>(param.VDataDes.counts))[topoAttr_.userRank];

    execMem.inputMem = algRes.cclInputMem;
    execMem.outputMem = algRes.aivOutputMem;
    ret = KernelRun(param, execMem);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllGatherVMeshAivExecutor][Orchestrate]errNo[0x%016llx] tag[%s] executor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()),
        ret);

    HCCL_INFO(
        "tag[%s], AllGatherVMeshAivExecutor orchestrate success, take time [%lld]us", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult AllGatherVMeshAivExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_INFO("[AllGatherVMeshAivExecutor][KernelRun]AllGatherV aiv enter.");

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    void* buffersIn[MAX_RANK_SIZE];
    void* buffersOut[MAX_RANK_SIZE];

    u32 localRank = outerCommInfo.localRank;
    u32 localRankSize = outerCommInfo.localRankSize;
    HCCL_DEBUG("[AllGatherVMeshAivExecutor][KernelRun] userRank [%u] localRank [%u]", topoAttr_.userRank, localRank);

    ExtraArgs extraArgs;
    for (u32 i = 0; i < localRankSize; i++) {
        extraArgs.recvCounts[i] = *(static_cast<const u64*>(param.VDataDes.counts) + i);
        extraArgs.recvDispls[i] = *(static_cast<const u64*>(param.VDataDes.displs) + i);
        if (i != localRank) {
            CHK_RET(outerCommInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(buffersIn[i])));
            CHK_RET(outerCommInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(buffersOut[i])));
            HCCL_DEBUG("[AllGatherVMeshAivExecutor][KernelRun] localRank [%u]", localRank);
        } else {
            buffersIn[i] = execMem.inputMem.ptr();
            buffersOut[i] = execMem.outputMem.ptr();
        }
        if (extraArgs.recvCounts[i] > extraArgs.maxCount) {
            extraArgs.maxCount = extraArgs.recvCounts[i];
        }
    }

    bool isOpbase = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HCCL_DEBUG("[AllGatherVMeshAivExecutor][KernelRun] isOpbase [%d]", isOpbase);
    AivOpArgs opArgs{HcclCMDType::HCCL_CMD_ALLGATHER_V, execMem.inputPtr, execMem.outputPtr, extraArgs.maxCount,
                     param.VDataDes.dataType,           param.reduceType, param.root,        isOpbase};
    AivTopoArgs topoArgs{localRank, localRankSize};
    u32 numBlocks;
    CHK_PRT_RET(
        CalNumBlocks(numBlocks, localRankSize) != HCCL_SUCCESS, HCCL_ERROR("[%s] CalNumBlocks failed", __func__),
        HCCL_E_PARA);
    numBlocks_ = numBlocks;
    topoArgs.identify = algoAttr_.identifier;
    HCCL_DEBUG("[AllGatherVMeshAivExecutor][KernelRun] numBlocks_ [%u]", numBlocks_);
    AivResourceArgs resourceArgs{param.tag,  param.stream.ptr(), buffersIn, buffersOut, execMem.inputMem.size(),
                                 numBlocks_, param.aivTag};
    AivAlgArgs algArgs{};
    algArgs.execTimeOut = topoMatcher_->GetExecTimeOutConfig();
    algArgs.execTimeOutSet = true;
    struct AivProfilingInfo aivProfilingInfo;
    aivProfilingInfo.counter = opCounter_;

    HcclResult ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, extraArgs, aivProfilingInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherVMeshAivExecutor][KernelRun]AllGatherV aiv failed, return[%d]", ret),
        ret);

    HCCL_INFO("[AllGatherVMeshAivExecutor][KernelRun]AllGatherV aiv run success.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherVMeshAivExecutor", AllGatherVAiv, AllGatherVMeshAivExecutor);

} // namespace hccl
