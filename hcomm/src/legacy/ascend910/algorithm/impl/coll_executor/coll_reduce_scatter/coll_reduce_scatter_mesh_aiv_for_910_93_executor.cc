/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_mesh_aiv_for_910_93_executor.h"
#include <algorithm>

namespace hccl {

CollReduceScatterMeshAivFor91093Executor::CollReduceScatterMeshAivFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
    desc_.isAivMode = true;
    desc_.isAivCrossNode = true;
    desc_.deterministic = 1;
}

HcclResult CollReduceScatterMeshAivFor91093Executor::CalcStreamNum(u32& streamNum)
{
    streamNum = 0; // AIV通信不需要申请从流
    HCCL_INFO(
        "[CollReduceScatterMeshAivFor91093Executor][CalcStreamNum] tag[%s] streamNum[%u].", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshAivFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshAivFor91093Executor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        if (topoMatcher_->GetDeterministicConfig() != DETERMINISTIC_DISABLE) {
            // 图模式确定性
            inputType = TransportMemType::SCRATCH;
        } else {
            inputType = TransportMemType::PARAM_INPUT;
        }
    } else {
        inputType = TransportMemType::CCL_INPUT;
    }

    outputType = TransportMemType::AIV_OUTPUT;
    HCCL_INFO(
        "[CollReduceScatterMeshAivFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d],"
        " outputType[%d].",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshAivFor91093Executor::CalcLevel0CommInfo(
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

void CollReduceScatterMeshAivFor91093Executor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;

    u64 inputSize = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];

    totalSize_ = BUFFER_DIVIDE * inputSize * topoAttr_.userRankSize;
}

HcclResult CollReduceScatterMeshAivFor91093Executor::CalcScratchMemSize(u64& scratchMemSize)
{
    scratchMemSize = 0;
    // 确定性时图模式需要计算scratchMem
    if (topoMatcher_->GetDeterministicConfig() != DETERMINISTIC_DISABLE
        && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        scratchMemSize = totalSize_;
    }
    HCCL_INFO("[%s]tag[%s] scratchMemSize[%llu]", __func__, tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshAivFor91093Executor::CalNumBlocksDeter(
    u32& numBlocks, u32 rankSize, u64 dataSize, HcclCMDType cmdType) const
{
    (void)dataSize;
    (void)cmdType;
    // Step1. Calculate the best block dimension
    u32 bestNumBlocks = (rankSize < MAX_NUM_BLOCKS ? rankSize : MAX_NUM_BLOCKS);
    u32 minNumBlocks = std::max((rankSize + MAX_TARGET_NUM - 1) / MAX_TARGET_NUM, NUM_BLOCKS_FACTOR_TWO);

    // Step2. Compare User Given numBlocks_ with bestNumBlocks
    numBlocks = bestNumBlocks;
    if (numBlocks_ < numBlocks) {
        numBlocks = numBlocks_ / NUM_BLOCKS_FACTOR_TWO * NUM_BLOCKS_FACTOR_TWO;
        minNumBlocks = (minNumBlocks + NUM_BLOCKS_FACTOR_TWO - 1) / NUM_BLOCKS_FACTOR_TWO * NUM_BLOCKS_FACTOR_TWO;
    }

    CHK_PRT_RET(
        numBlocks < minNumBlocks,
        HCCL_ERROR(
            "[CollReduceScatterMeshAivFor91093Executor][CalNumBlocks]aivCore[%u] is invalid, at least need [%u].",
            numBlocks_, minNumBlocks),
        HCCL_E_PARA);

    HCCL_INFO(
        "[CollReduceScatterMeshAivFor91093Executor][CalNumBlocks] numBlocks is set to [%u], limit[%u], recommended[%u]",
        numBlocks, numBlocks_, bestNumBlocks);
    return HCCL_SUCCESS;
}

HcclResult
CollReduceScatterMeshAivFor91093Executor::CalNumBlocks(u32& numBlocks, u32 rankSize, u64 dataSize, HcclCMDType cmdType)
{
    if (topoMatcher_->GetDeterministicConfig() != DETERMINISTIC_DISABLE) {
        return CalNumBlocksDeter(numBlocks, rankSize, dataSize, cmdType);
    }

    numBlocks = rankSize; // 默认情况使用rankSize个AIV

    u32 minNumBlocks = (rankSize + MAX_TARGET_NUM - 1) / MAX_TARGET_NUM;

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

    u32 bestNumBlocks = numBlocks;
    numBlocks
        = numBlocks_ < rankSize ? numBlocks_ : (numBlocks_ < numBlocks ? numBlocks_ / rankSize * rankSize : numBlocks);

    CHK_PRT_RET(
        numBlocks < minNumBlocks,
        HCCL_ERROR(
            "[CollReduceScatterMeshAivFor91093Executor][CalNumBlocks]aivCore[%u] is invalid, at least need [%u].",
            numBlocks_, minNumBlocks),
        HCCL_E_PARA);

    HCCL_INFO(
        "[CollReduceScatterMeshAivFor91093Executor][CalNumBlocks] numBlocks is set to [%u], limit[%u], recommended[%u]",
        numBlocks, numBlocks_, bestNumBlocks);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshAivFor91093Executor::CopyAivCommInfoToDevice(
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
            if (isOpbaseMode) {
                buffersInOut[idx] = algResource.cclInputMem.ptr();
            } else {
                if (topoMatcher_->GetDeterministicConfig() != DETERMINISTIC_DISABLE) {
                    buffersInOut[idx] = algResource.scratchMem.ptr();
                } else {
                    buffersInOut[idx] = algResource.paramInputMem.ptr();
                }
            }
            buffersInOut[idx + 1] = algResource.aivOutputMem.ptr();
        }
    }
    const u32 bufferNum = 2;
    CHK_RET(hrtMemSyncCopy(
        algResource.aivCommInfoMem.ptr(), sizeof(u64) * localRankSize * bufferNum, buffersInOut,
        sizeof(u64) * localRankSize * bufferNum, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshAivFor91093Executor::PrepareCommInfoToDevice(AlgResourceResponse& algResource)
{
    HCCL_INFO("[CollReduceScatterMeshAivFor91093Executor][PrepareCommInfoToDevice]"
              "ReduceScatter aiv copy comm info to device.");
    CHK_RET(CopyAivCommInfoToDevice(COMM_COMBINE_ORDER, COMM_INDEX_0, algResource));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshAivFor91093Executor::GetAivExecParam(
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

    execMem.inputMem = algRes.aivInputMem;
    if (topoMatcher_->GetDeterministicConfig() != DETERMINISTIC_DISABLE) {
        // 图模式确定性
        execMem.inputMem = algRes.scratchMem;
    } else {
        execMem.inputMem = algRes.paramInputMem;
    }
    execMem.outputMem = algRes.aivOutputMem;
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);

    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG(
        "[CollReduceScatterMeshAivFor91093Executor][GetAivExecParam] userRank [%d] localRank [%d]", topoAttr_.userRank,
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
        "SPK [CollReduceScatterMeshAivFor91093Executor][GetAivExecParam], rank[%llu], rankSize[%llu], "
        "len[%llu],datatype[%llu], op[%llu]",
        args.rank, args.rankSize, args.len, args.dataType, args.reduceOp);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceScatterMeshAivFor91093Executor][Orchestrate]errNo[0x%016llx] tag[%s] executor kernel "
            "run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()),
        ret);

    HCCL_INFO(
        "tag[%s], ReduceScatter executor getalgexecparam success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshAivFor91093Executor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        if (topoMatcher_->GetDeterministicConfig() != DETERMINISTIC_DISABLE) {
            // 图模式确定性
            execMem.inputMem = algRes.scratchMem;
        } else {
            execMem.inputMem = algRes.paramInputMem;
        }
    } else {
        execMem.inputMem = algRes.cclInputMem;
    }

    execMem.outputMem = algRes.aivOutputMem;
    HcclResult ret = KernelRun(param, execMem);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceScatterMeshAivFor91093Executor][Orchestrate]errNo[0x%016llx] tag[%s] executor kernel "
            "run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()),
        ret);

    HCCL_INFO(
        "tag[%s], ReduceScatter executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshAivFor91093Executor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] ReduceScatter aiv enter.", __func__);

    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);

    void* buffersIn[MAX_RANK_SIZE];
    void* buffersOut[MAX_RANK_SIZE];

    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG(
        "[CollReduceScatterMeshAivFor91093Executor][KernelRun] userRank [%d] localRank [%d].", topoAttr_.userRank,
        localRank);

    buffersIn[0] = execMem.inputMem.ptr();
    buffersOut[0] = execMem.outputMem.ptr();
    constexpr u32 BUFFER_IDX_ONE = 1;
    buffersOut[BUFFER_IDX_ONE] = algResResp_->aivCommInfoMem.ptr(); // 通信域信息

    bool isOpbase = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;

    AivOpArgs opArgs{
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER,
        execMem.inputPtr,
        execMem.outputPtr,
        execMem.count,
        param.DataDes.dataType,
        param.reduceType,
        0,
        isOpbase};
    AivTopoArgs topoArgs{localRank, localRankSize, MAX_RANK_SIZE, 0, topoAttr_.serverNum, topoAttr_.deviceType};
    topoArgs.identify = algoAttr_.identifier;
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
    if (topoMatcher_->GetDeterministicConfig() != DETERMINISTIC_DISABLE) {
        algArgs.deterministic = 1;
    }
    algArgs.execTimeOut = topoMatcher_->GetExecTimeOutConfig();
    algArgs.execTimeOutSet = true;
    struct AivProfilingInfo aivProfilingInfo;
    aivProfilingInfo.counter = opCounter_;

    HcclResult ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollReduceScatterMeshAivFor91093Executor][KernelRun]ReduceScatter aiv failed, return[%d]", ret),
        ret);

    ExtraArgs extraArgs;
    CHK_RET(SetOpCache(opArgs, topoArgs, resourceArgs, algArgs, extraArgs, aivProfilingInfo, true));

    HCCL_INFO("[CollReduceScatterMeshAivFor91093Executor][KernelRun]ReduceScatter aiv run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC(
    "ReduceScatterMeshAivFor91093Executor", ReduceScatterMeshAivFor91093, CollReduceScatterMeshAivFor91093Executor);

} // namespace hccl
