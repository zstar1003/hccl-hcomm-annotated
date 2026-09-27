/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_mesh_aiv_for_910_93_executor.h"

namespace hccl {
CollAllGatherMeshAivFor91093Executor::CollAllGatherMeshAivFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
    desc_.isAivMode = true;
    desc_.isAivCrossNode = true;
}

HcclResult CollAllGatherMeshAivFor91093Executor::CalcStreamNum(u32& streamNum)
{
    streamNum = 0; // AIV通信不需要申请从流
    HCCL_INFO("[CollAllGatherMeshAivFor91093Executor][CalcStreamNum] tag[%s] streamNum[%u].", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshAivFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshAivFor91093Executor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType
        = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ? TransportMemType::CCL_INPUT :
                                                                           TransportMemType::PARAM_INPUT);
    outputType = TransportMemType::AIV_OUTPUT;
    HCCL_INFO(
        "[CollAllGatherMeshAivFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d],"
        " outputType[%d].",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshAivFor91093Executor::CalcLevel0CommInfo(
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

HcclResult CollAllGatherMeshAivFor91093Executor::CalNumBlocks(
    u32& numBlocks, u32 rankSize, u64 dataSize, [[maybe_unused]] HcclCMDType cmdType)
{
    u32 minNumBlocks = (rankSize + MAX_TARGET_NUM - 1) / MAX_TARGET_NUM;
    numBlocks = rankSize; // 默认情况使用rankSize个AIV

    // A3超节点内多机场景
    // 多核并行优化
    if (rankSize > HALF_MAX_NUM_BLOCKS || dataSize < AIV_A3_CROSSNODE_TINY_SIZE) {
        numBlocks = rankSize;
    } else if (rankSize > ONE_THIRD_MAX_NUM_BLOCKS || dataSize < AIV_A3_CROSSNODE_SMALL_SIZE) {
        numBlocks = rankSize * NUM_BLOCKS_FACTOR_TWO;
    } else if (rankSize > ONE_FOURTH_MAX_NUM_BLOCKS) {
        numBlocks = rankSize * NUM_BLOCKS_FACTOR_THREE;
    } else {
        numBlocks = rankSize * NUM_BLOCKS_FACTOR_FOUR;
    }
    HCCL_DEBUG("[CollAllGatherMeshAivFor91093Executor][CalNumBlocks]aivCore at least is [%u]", minNumBlocks);
    u32 bestNumBlocks = numBlocks;
    numBlocks
        = numBlocks_ < rankSize ? numBlocks_ : (numBlocks_ < numBlocks ? numBlocks_ / rankSize * rankSize : numBlocks);

    CHK_PRT_RET(
        numBlocks < minNumBlocks,
        HCCL_ERROR(
            "[CollAllGatherMeshAivFor91093Executor][CalNumBlocks]aivCore[%u] is invalid, at least need [%u].",
            numBlocks, minNumBlocks),
        HCCL_E_PARA);

    HCCL_INFO(
        "[CollAllGatherMeshAivFor91093Executor][CalNumBlocks] numBlocks is set to [%u], limit[%u], recommended[%u].",
        numBlocks, numBlocks_, bestNumBlocks);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshAivFor91093Executor::GetAivExecParam(
    const OpParam& param, AlgResourceResponse& algRes, AivSuperKernelArgs& args)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    HcclResult ret = HCCL_SUCCESS;
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    execMem.inputMem = algRes.paramInputMem;
    execMem.outputMem = algRes.aivOutputMem;
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);

    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG(
        "[CollAllGatherMeshAivFor91093Executor][GetAivExecParam] userRank [%d] localRank [%d]", topoAttr_.userRank,
        localRank);

    args.buffersIn[0] = execMem.inputMem.ptr();
    args.buffersOut[0] = execMem.outputMem.ptr();
    constexpr u32 BUFFER_IDX_ONE = 1;
    args.buffersOut[BUFFER_IDX_ONE] = algRes.aivCommInfoMem.ptr(); // 通信域信息

    args.rank = localRank;
    args.rankSize = localRankSize;
    args.len = execMem.count;
    args.dataType = param.DataDes.dataType;
    args.unitSize = SIZE_TABLE[param.DataDes.dataType];
    args.reduceOp = param.reduceType;
    args.devType = static_cast<u32>(topoAttr_.deviceType);
    HCCL_INFO(
        "[CollAllGatherMeshAivFor91093Executor][GetAivExecParam], rank[%llu], rankSize[%llu], "
        "len[%llu],datatype[%llu], op[%llu]",
        args.rank, args.rankSize, args.len, args.dataType, args.reduceOp);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAllGatherMeshAivFor91093Executor][Orchestrate]errNo[0x%016llx] tag[%s] executor kernel "
            "run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()),
        ret);

    HCCL_INFO(
        "tag[%s], AllGather executor getalgexecparam success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshAivFor91093Executor::PrepareCommInfoToDevice(AlgResourceResponse& algResource)
{
    HCCL_INFO("[CollAllGatherMeshAivFor91093Executor][PrepareCommInfoToDevice]AllGather aiv copy comm info to device.");
    CHK_RET(CopyAivCommInfoToDevice(COMM_COMBINE_ORDER, COMM_INDEX_0, algResource));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshAivFor91093Executor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    execMem.inputMem
        = (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ? algRes.paramInputMem : algRes.cclInputMem);
    execMem.outputMem = algRes.aivOutputMem;
    HcclResult ret = KernelRun(param, execMem);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAllGatherMeshAivFor91093Executor][Orchestrate]errNo[0x%016llx] tag[%s] executor kernel "
            "run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()),
        ret);

    HCCL_INFO(
        "tag[%s], AllGather executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMeshAivFor91093Executor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherMeshAivFor91093Executor][KernelRun]AllGather aiv enter.");

    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);

    void* buffersIn[MAX_RANK_SIZE];
    void* buffersOut[MAX_RANK_SIZE];

    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG(
        "[CollAllGatherMeshAivFor91093Executor][KernelRun] userRank [%d] localRank [%d].", topoAttr_.userRank,
        localRank);

    buffersIn[0] = execMem.inputMem.ptr();
    buffersOut[0] = execMem.outputMem.ptr();
    constexpr u32 BUFFER_IDX_ONE = 1;
    buffersOut[BUFFER_IDX_ONE] = algResResp_->aivCommInfoMem.ptr(); // 通信域信息

    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    AivOpArgs opArgs{
        HcclCMDType::HCCL_CMD_ALLGATHER,
        execMem.inputPtr,
        execMem.outputPtr,
        execMem.count,
        param.DataDes.dataType,
        param.reduceType,
        0,
        isOpbase};

    AivTopoArgs topoArgs{localRank,           localRankSize,        MAX_RANK_SIZE,       0,
                         topoAttr_.serverNum, topoAttr_.deviceType, algoAttr_.identifier};
    u32 numBlocks;
    CHK_PRT_RET(
        CalNumBlocks(numBlocks, localRankSize, opArgs.count * SIZE_TABLE[opArgs.dataType]) != HCCL_SUCCESS,
        HCCL_ERROR("[%s] CalNumBlocks failed", __func__), HCCL_E_PARA);
    numBlocks_ = numBlocks;
    AivResourceArgs resourceArgs{param.tag,  param.stream.ptr(), buffersIn, buffersOut, execMem.inputMem.size(),
                                 numBlocks_, param.aivTag};
    AivAlgArgs algArgs{};
    algArgs.argsType = KernelArgsType::ARGS_TYPE_SIMPLE;
    if (numBlocks_ >= localRankSize) {
        algArgs.step = localRankSize;
    } else {
        algArgs.step = numBlocks_;
    }
    algArgs.execTimeOut = topoMatcher_->GetExecTimeOutConfig();
    algArgs.execTimeOutSet = true;
    struct AivProfilingInfo aivProfilingInfo;
    aivProfilingInfo.counter = opCounter_;

    HcclResult ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherMeshAivFor91093Executor][KernelRun]AllGather aiv failed, return[%d]", ret), ret);

    ExtraArgs extraArgs;
    CHK_RET(SetOpCache(opArgs, topoArgs, resourceArgs, algArgs, extraArgs, aivProfilingInfo, true));

    HCCL_INFO("[CollAllGatherMeshAivFor91093Executor][KernelRun]AllGather aiv run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherMeshAivFor91093Executor", AllGatherMeshAivFor91093, CollAllGatherMeshAivFor91093Executor);

} // namespace hccl
