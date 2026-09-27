/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_broadcast_mesh_aiv_for_910_93_executor.h"

namespace hccl {

constexpr u32 BUFFER_IDX_ONE = 1;

CollBroadcastMeshAivFor91093Executor::CollBroadcastMeshAivFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
    desc_.isAivMode = true;
    desc_.isAivCrossNode = true;
}

HcclResult CollBroadcastMeshAivFor91093Executor::CalcStreamNum(u32& streamNum)
{
    streamNum = 0; // AIV通信不需要申请从流
    HCCL_INFO("[CollBroadcastMeshAivFor91093Executor][CalcStreamNum] tag[%s] streamNum[%u].", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshAivFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::CCL_INPUT;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        inputType = TransportMemType::SCRATCH;
    }
    TransportMemType outputType = TransportMemType::AIV_OUTPUT;

    HCCL_INFO(
        "[CollBroadcastMeshAivFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d].",
        tag_.c_str(), inputType, outputType);
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshAivFor91093Executor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commCombinePara(COMM_COMBINE_ORDER, CommType::COMM_TAG_MESH);
    commCombinePara.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commCombinePara, opTransport[COMM_COMBINE_ORDER], inputType, outputType));

    LevelNSubCommTransport& commTransportLevel0 = opTransport[COMM_COMBINE_ORDER];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        for (auto& transportRequest : commTransportLevel0[subCommIndex].transportRequests) {
            transportRequest.isUsedRdma = false;
        }
    }
    return HCCL_SUCCESS;
}
void CollBroadcastMeshAivFor91093Executor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    totalSize_ = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
}

HcclResult CollBroadcastMeshAivFor91093Executor::CalcScratchMemSize(u64& scratchMemSize)
{
    scratchMemSize = 0;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        scratchMemSize = totalSize_;
    }
    HCCL_INFO("[%s]tag[%s] scratchMemSize[%llu]", __func__, tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshAivFor91093Executor::CalNumBlocks(
    u32& numBlocks, u32 rankSize, [[maybe_unused]] u64 dataSize, [[maybe_unused]] HcclCMDType cmdType)
{
    // Step1. 计算最佳块维度
    const u32 bestNumBlock = std::min(rankSize - 1, MAX_NUM_BLOCKS);
    const u32 minNumBlock = (rankSize + MAX_TARGET_NUM - 1) / MAX_TARGET_NUM;

    // 参数校验
    if (numBlocks_ < NUM_BLOCKS_FACTOR_TWO) {
        HCCL_ERROR(
            "[CollBroadcastMeshAivFor91093Executor][CalNumBlocks] aivCore[%u] is invalid, at least need 2", numBlocks_);
        return HCCL_E_PARA;
    }

    // 选择合适的块数量
    numBlocks = numBlocks_ < bestNumBlock ? numBlocks_ : bestNumBlock;

    // 检查最小块数量限制
    if (numBlocks_ < minNumBlock) {
        HCCL_ERROR(
            "[CollBroadcastMeshAivFor91093Executor][CalNumBlocks] limit[%u], at least[%u]", numBlocks_, minNumBlock);
        return HCCL_E_PARA;
    }

    // 记录日志
    HCCL_INFO(
        "[CollBroadcastMeshAivFor91093Executor][CalNumBlocks] numBlock is set to [%u], limit[%u], best[%u]", numBlocks,
        numBlocks_, bestNumBlock);
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshAivFor91093Executor::PrepareCommInfoToDevice(AlgResourceResponse& algResource)
{
    HCCL_INFO("[CollBroadcastMeshAivFor91093Executor][PrepareCommInfoToDevice]Broadcast aiv copy comm info to device.");
    CHK_RET(CopyAivCommInfoToDevice(COMM_COMBINE_ORDER, COMM_INDEX_0, algResource));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshAivFor91093Executor::CopyAivCommInfoToDevice(
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
            buffersInOut[idx] = isOpbaseMode ? algResource.cclInputMem.ptr() : algResource.scratchMem.ptr();
            buffersInOut[idx + 1] = algResource.aivOutputMem.ptr();
        }
    }
    constexpr u32 bufferNum = 2;
    CHK_RET(hrtMemSyncCopy(
        algResource.aivCommInfoMem.ptr(), sizeof(u64) * localRankSize * bufferNum, buffersInOut,
        sizeof(u64) * localRankSize * bufferNum, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshAivFor91093Executor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.inputPtr; // broadcast使用一块内存

    execMem.inputMem
        = (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ? algRes.scratchMem : algRes.cclInputMem);
    execMem.outputMem = algRes.aivOutputMem;

    HcclResult ret = KernelRun(param, execMem);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollBroadcastMeshAivFor91093Executor][Orchestrate]errNo[0x%016llx] tag[%s] executor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()),
        ret);

    HCCL_INFO(
        "tag[%s], Broadcast executor orchestrate success, take time [%lld]us", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshAivFor91093Executor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_INFO("[CollBroadcastMeshAivFor91093Executor][KernelRun]broadcast aiv enter.");

    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);

    void* buffersIn[MAX_RANK_SIZE];
    void* buffersOut[MAX_RANK_SIZE];

    u32 rootRank = 0;
    HcclResult ret = GetRankByUserRank(COMM_COMBINE_ORDER, COMM_INDEX_0, param.root, rootRank);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[BroadCastOperator][CollBroadcastMeshAivFor91093Executor]invalid root[%u] to get userrank", param.root),
        ret);

    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG(
        "[CollBroadcastMeshAivFor91093Executor][KernelRun] userRank [%u] localRank [%u]", topoAttr_.userRank,
        localRank);

    buffersIn[0] = execMem.inputMem.ptr();
    buffersOut[0] = execMem.outputMem.ptr();
    buffersOut[BUFFER_IDX_ONE] = algResResp_->aivCommInfoMem.ptr();

    bool isOpbase = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    AivOpArgs opArgs{HcclCMDType::HCCL_CMD_BROADCAST, execMem.inputPtr, execMem.outputPtr, execMem.count,
                     param.DataDes.dataType,          param.reduceType, rootRank,          isOpbase};
    AivTopoArgs topoArgs{localRank,           localRankSize,        MAX_RANK_SIZE,       0,
                         topoAttr_.serverNum, topoAttr_.deviceType, algoAttr_.identifier};

    u32 numBlocks;
    CHK_RET(CalNumBlocks(numBlocks, localRankSize));
    numBlocks_ = numBlocks;

    AivResourceArgs resourceArgs{param.tag,  param.stream.ptr(), buffersIn, buffersOut, execMem.inputMem.size(),
                                 numBlocks_, param.aivTag};
    AivAlgArgs algArgs{};
    algArgs.execTimeOut = topoMatcher_->GetExecTimeOutConfig();
    algArgs.execTimeOutSet = true;
    algArgs.argsType = KernelArgsType::ARGS_TYPE_SIMPLE;

    struct AivProfilingInfo aivProfilingInfo;
    aivProfilingInfo.counter = opCounter_;
    ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollBroadcastMeshAivFor91093Executor][KernelRun]broadcast aiv failed, return[%d]", ret), ret);

    ExtraArgs extraArgs;
    CHK_RET(SetOpCache(opArgs, topoArgs, resourceArgs, algArgs, extraArgs, aivProfilingInfo, true));

    HCCL_INFO("[CollBroadcastMeshAivFor91093Executor][KernelRun]broadcast aiv run success.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("BroadcastMeshAivFor91093Executor", BroadcastMeshAivFor91093, CollBroadcastMeshAivFor91093Executor);

} // namespace hccl
