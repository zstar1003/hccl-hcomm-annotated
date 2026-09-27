/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_mesh_aiv_for_910_93_executor.h"
#include <algorithm>

namespace hccl {

CollAlltoAllMeshAivFor91093Executor::CollAlltoAllMeshAivFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{
    desc_.isAivMode = true;
    desc_.isAivCrossNode = true;
}

HcclResult CollAlltoAllMeshAivFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    // level0 - level1 全连接通信域
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivFor91093Executor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::AIV_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::AIV_OUTPUT;
    }
    HCCL_INFO(
        "[CollAlltoAllMeshAivFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d].",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

// level0-level1 打平fullmesh 超节点内建SDMA链路
HcclResult CollAlltoAllMeshAivFor91093Executor::CalcLevel0CommInfo(
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

HcclResult CollAlltoAllMeshAivFor91093Executor::CalNumBlocks(
    u32& numBlocks, u32 rankSize, [[maybe_unused]] u64 dataSize, [[maybe_unused]] HcclCMDType cmdType)
{
    // A3超节点内多机场景，numBlocks_需要为偶数
    numBlocks = (rankSize < MAX_NUM_BLOCKS ? rankSize + rankSize % NUM_BLOCKS_FACTOR_TWO : MAX_NUM_BLOCKS);
    u32 bestNumBlocks = numBlocks;
    u32 minNumBlocks = std::max((rankSize + MAX_TARGET_NUM - 1) / MAX_TARGET_NUM, NUM_BLOCKS_FACTOR_TWO);

    if (numBlocks_ < numBlocks) {
        numBlocks = numBlocks_ / NUM_BLOCKS_FACTOR_TWO * NUM_BLOCKS_FACTOR_TWO;
        minNumBlocks = (minNumBlocks + NUM_BLOCKS_FACTOR_TWO - 1) / NUM_BLOCKS_FACTOR_TWO * NUM_BLOCKS_FACTOR_TWO;
    }

    CHK_PRT_RET(
        numBlocks < minNumBlocks,
        HCCL_ERROR(
            "[CollAlltoAllMeshAivFor91093Executor][CalNumBlocks]aivCore[%u] is invalid, at least need [%u].",
            numBlocks_, minNumBlocks),
        HCCL_E_PARA);

    HCCL_INFO(
        "[CollAlltoAllMeshAivFor91093Executor][CalNumBlocks] numBlocks is set to [%u], limit[%u], recommended[%u]",
        numBlocks, numBlocks_, bestNumBlocks);
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivFor91093Executor::GetAivExecParam(
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
        "[CollAlltoAllMeshAivFor91093Executor][GetAivExecParam] userRank [%u] localRank [%u]", topoAttr_.userRank,
        localRank);

    args.buffersIn[0] = execMem.inputMem.ptr();
    args.buffersOut[0] = execMem.outputMem.ptr();
    constexpr u32 BUFFER_IDX_ONE = 1;
    args.buffersOut[BUFFER_IDX_ONE] = algRes.aivCommInfoMem.ptr(); // 通信域信息

    args.rank = localRank;
    args.rankSize = localRankSize;
    args.len = param.All2AllDataDes.sendCount;
    args.dataType = param.All2AllDataDes.sendType;
    args.unitSize = SIZE_TABLE[param.All2AllDataDes.sendType];
    args.reduceOp = param.reduceType;
    args.devType = static_cast<u32>(topoAttr_.deviceType);
    HCCL_INFO(
        "[CollAlltoAllMeshAivFor91093Executor][GetAivExecParam], rank[%llu], rankSize[%llu], "
        "len[%llu],datatype[%llu], op[%llu]",
        args.rank, args.rankSize, args.len, args.dataType, args.reduceOp);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAlltoAllMeshAivFor91093Executor][Orchestrate]errNo[0x%016llx] tag[%s] executor kernel "
            "run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()),
        ret);

    HCCL_INFO(
        "tag[%s], AlltoAll executor getalgexecparam success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivFor91093Executor::PrepareCommInfoToDevice(AlgResourceResponse& algResource)
{
    HCCL_INFO("[CollAlltoAllMeshAivFor91093Executor][PrepareCommInfoToDevice]AllToAll aiv copy comm info to device.");
    CHK_RET(CopyAivCommInfoToDevice(COMM_COMBINE_ORDER, COMM_INDEX_0, algResource));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivFor91093Executor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
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
            "[CollAlltoAllMeshAivFor91093Executor][Orchestrate]errNo[0x%016llx] tag[%s] "
            "executor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()),
        ret);

    HCCL_INFO(
        "tag[%s], AlltoAll executor orchestrate success, take time [%lld]us", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivFor91093Executor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAlltoAllMeshAivFor91093Executor][KernelRun]AllToAll aiv enter.");

    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);

    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG(
        "[CollAlltoAllMeshAivFor91093Executor][KernelRun] userRank [%u] localRank [%u] localRankSize[%u]",
        topoAttr_.userRank, localRank, localRankSize);

    HcclResult ret;
    bool isOpbase = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    void* buffersIn[MAX_RANK_SIZE] = {};
    void* buffersOut[MAX_RANK_SIZE] = {};
    buffersIn[0] = execMem.inputMem.ptr();
    buffersOut[0] = execMem.outputMem.ptr();
    constexpr u32 BUFFER_IDX_ONE = 1;
    buffersOut[BUFFER_IDX_ONE] = algResResp_->aivCommInfoMem.ptr(); // 通信域信息

    AivTopoArgs topoArgs{localRank, localRankSize, MAX_RANK_SIZE, 0, topoAttr_.serverNum, topoAttr_.deviceType};
    u32 numBlocks;
    CHK_PRT_RET(
        CalNumBlocks(numBlocks, localRankSize) != HCCL_SUCCESS, HCCL_ERROR("[%s] CalNumBlocks failed", __func__),
        HCCL_E_PARA);
    numBlocks_ = numBlocks;
    topoArgs.identify = algoAttr_.identifier;
    AivResourceArgs resourceArgs{param.tag,  param.stream.ptr(), buffersIn, buffersOut, execMem.inputMem.size(),
                                 numBlocks_, param.aivTag};
    AivAlgArgs algArgs{};
    algArgs.execTimeOut = topoMatcher_->GetExecTimeOutConfig();
    algArgs.execTimeOutSet = true;
    AivProfilingInfo aivProfilingInfo;
    aivProfilingInfo.counter = opCounter_;
    AivOpArgs opArgs{
        HcclCMDType::HCCL_CMD_ALLTOALL,
        execMem.inputPtr,
        execMem.outputPtr,
        param.All2AllDataDes.sendCount,
        param.All2AllDataDes.sendType,
        HCCL_REDUCE_RESERVED,
        0,
        isOpbase};

    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        // 兜底算法，单机也可能走这里
        constexpr u32 TWO_SERVER_NUM = 2;
        if (topoArgs.serverNum == 1) {
            topoArgs.serverNum = TWO_SERVER_NUM;
        }
        ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo);
    } else {
        algArgs.argsType = KernelArgsType::ARGS_TYPE_SUPERPOD;
        ExtraArgsV2 extraArgs;
        if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
            for (u32 i = 0; i < localRankSize; i++) {
                extraArgs.sendCounts[i]
                    = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + localRank * localRankSize + i);
                extraArgs.recvCounts[i]
                    = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + i * localRankSize + localRank);
                if (i == 0) {
                    extraArgs.sendDispls[i] = 0;
                    extraArgs.recvDispls[i] = 0;
                } else {
                    extraArgs.sendDispls[i] = extraArgs.sendDispls[i - 1] + extraArgs.sendCounts[i - 1];
                    extraArgs.recvDispls[i] = extraArgs.recvDispls[i - 1] + extraArgs.recvCounts[i - 1];
                }
            }
        } else {
            for (u32 i = 0; i < localRankSize; i++) {
                extraArgs.sendCounts[i] = *(static_cast<const u64*>(param.All2AllDataDes.sendCounts) + i);
                extraArgs.recvCounts[i] = *(static_cast<const u64*>(param.All2AllDataDes.recvCounts) + i);
                extraArgs.sendDispls[i] = *(static_cast<const u64*>(param.All2AllDataDes.sdispls) + i);
                extraArgs.recvDispls[i] = *(static_cast<const u64*>(param.All2AllDataDes.rdispls) + i);
            }
        }
        opArgs.cmdType = HcclCMDType::HCCL_CMD_ALLTOALLV;
        opArgs.dataType = param.All2AllDataDes.sendType;
        ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, extraArgs, aivProfilingInfo);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlltoAllMeshAivFor91093Executor][KernelRun]AllToAll aiv failed, return[%d]", ret), ret);

    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        ExtraArgs extraArgs;
        CHK_RET(SetOpCache(opArgs, topoArgs, resourceArgs, algArgs, extraArgs, aivProfilingInfo, true));
    }

    HCCL_INFO("[CollAlltoAllMeshAivFor91093Executor][KernelRun]AllToAll aiv run success.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AlltoAllMeshAivFor91093Executor", AlltoAllMeshAivFor91093, CollAlltoAllMeshAivFor91093Executor);

} // namespace hccl
