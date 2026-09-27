/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_aiv_deter_small_executor.h"

namespace hccl {
CollReduceScatterAivDeterSmallExecutor::CollReduceScatterAivDeterSmallExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
    desc_.isAivMode = true;
}

HcclResult CollReduceScatterAivDeterSmallExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = 0; // AIV通信不需要申请从流
    HCCL_INFO(
        "[CollReduceScatterAivDeterSmallExecutor][CalcStreamNum] tag[%s] streamNum[%u].", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterAivDeterSmallExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterAivDeterSmallExecutor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType = TransportMemType::AIV_INPUT;
    outputType = TransportMemType::AIV_OUTPUT;
    HCCL_INFO(
        "[CollReduceScatterAivDeterSmallExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d].",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterAivDeterSmallExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterAivDeterSmallExecutor::CalNumBlocks(
    u32& numBlocks, u32 rankSize, u64 dataSize, [[maybe_unused]] HcclCMDType cmdType)
{
    numBlocks = rankSize; // 默认情况使用rankSize个AIV

    numBlocks = NUM_BLOCKS_FACTOR_TWO * rankSize; // 小数据量使用2倍rankSize的AIV core
    if ((rankSize * dataSize) >= AIV_REDUCE_SCATTER_DETER_SMALL_SIZE) {
        numBlocks = NUM_BLOCKS_FACTOR_THREE * rankSize;
    }

    u32 bestNumBlocks = numBlocks;
    CHK_PRT_RET(
        numBlocks_ < rankSize,
        HCCL_WARNING(
            "[CollReduceScatterAivDeterSmallExecutor][CalNumBlocks]aivCore[%u] is invalid, at least need [%u].",
            numBlocks_, rankSize),
        HCCL_E_PARA);
    if (numBlocks_ < numBlocks) {
        numBlocks = numBlocks_ / rankSize * rankSize;
    }

    HCCL_INFO(
        "[CollReduceScatterAivDeterSmallExecutor][CalNumBlocks] numBlocks is set to [%u], limit[%u], recommended[%u]",
        numBlocks, numBlocks_, bestNumBlocks);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterAivDeterSmallExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
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
    execMem.outputMem = algRes.aivOutputMem;

    ret = KernelRun(param, execMem);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceScatterAivDeterSmallExecutor][Orchestrate]errNo[0x%016llx] tag[%s] executor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()),
        ret);

    HCCL_INFO(
        "tag[%s], ReduceScatter executor orchestrate success, take time [%lld]us", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterAivDeterSmallExecutor::GetAivExecParam(
    const OpParam& param, AlgResourceResponse& algRes, AivSuperKernelArgs& args)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    // 单算子大数据量
    execMem.inputMem = algRes.paramInputMem;
    execMem.outputMem = algRes.aivOutputMem;

    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG(
        "[CollReduceScatterAivDeterSmallExecutor][GetAivExecParam] userRank [%d] localRank [%d]", topoAttr_.userRank,
        localRank);

    for (u32 i = 0; i < localRankSize; i++) {
        if (i != localRank) {
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(args.buffersIn[i])));
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(args.buffersOut[i])));
        } else {
            args.buffersIn[i] = execMem.inputMem.ptr();
            args.buffersOut[i] = execMem.outputMem.ptr();
        }
    }

    HCCL_INFO(
        "[CollReduceScatterAivDeterSmallExecutor] buffersIn [%p] [%p] [%p] [%p] "
        "buffersOut [%p] [%p] [%p] [%p]",
        args.buffersIn[0], args.buffersIn[1], args.buffersIn[2], args.buffersIn[3], args.buffersOut[0],
        args.buffersOut[1], args.buffersOut[2], args.buffersOut[3]);
    args.rank = localRank;
    args.rankSize = localRankSize;
    args.len = execMem.count;
    args.dataType = param.DataDes.dataType;
    args.unitSize = SIZE_TABLE[param.DataDes.dataType];
    args.reduceOp = param.reduceType;
    args.devType = static_cast<u32>(topoAttr_.deviceType);
    HCCL_INFO(
        "[CollReduceScatterAivDeterSmallExecutor][GetAivExecParam], rank[%llu], rankSize[%llu], "
        "len[%llu],datatype[%llu], op[%llu]",
        args.rank, args.rankSize, args.len, args.dataType, args.reduceOp);

    HCCL_INFO(
        "tag[%s], ReduceScatter executor getalgexecparam success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterAivDeterSmallExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_INFO("[CollReduceScatterAivDeterSmallExecutor][KernelRun]ReduceScatter aiv enter.");

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    void* buffersIn[MAX_RANK_SIZE];
    void* buffersOut[MAX_RANK_SIZE];

    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG(
        "[CollReduceScatterAivDeterSmallExecutor][KernelRun] userRank [%u] localRank [%u]", topoAttr_.userRank,
        localRank);

    for (u32 i = 0; i < localRankSize; i++) {
        if (i != localRank) {
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(buffersIn[i])));
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(buffersOut[i])));
        } else {
            buffersIn[i] = execMem.inputMem.ptr();
            buffersOut[i] = execMem.outputMem.ptr();
        }
    }

    bool isOpbase = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    AivOpArgs opArgs{
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER,
        execMem.inputPtr,
        execMem.outputPtr,
        execMem.count,
        param.DataDes.dataType,
        param.reduceType,
        0,
        isOpbase};
    AivTopoArgs topoArgs{localRank, localRankSize, MAX_RANK_SIZE, 0, 1, topoAttr_.deviceType};
    topoArgs.identify = algoAttr_.identifier;

    u64 dataSize = SIZE_TABLE[param.DataDes.dataType] * execMem.count;
    u32 numBlocks;
    CHK_PRT_RET(
        CalNumBlocks(numBlocks, localRankSize, dataSize) != HCCL_SUCCESS,
        HCCL_ERROR("[%s] CalNumBlocks failed", __func__), HCCL_E_PARA);
    numBlocks_ = numBlocks;
    AivResourceArgs resourceArgs{param.tag,  param.stream.ptr(), buffersIn, buffersOut, execMem.inputMem.size(),
                                 numBlocks_, param.aivTag};
    AivAlgArgs algArgs{};
    algArgs.deterministic = 1;
    algArgs.execTimeOut = topoMatcher_->GetExecTimeOutConfig();
    algArgs.execTimeOutSet = true;
    struct AivProfilingInfo aivProfilingInfo;
    aivProfilingInfo.counter = opCounter_;
    HCCL_INFO(
        "[CollReduceScatterAivDeterSmallExecutor][KernelRun]ReduceScatter bufferin[%d] bufferout[%d]",
        execMem.inputMem.size(), execMem.outputMem.size());

    HcclResult ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollReduceScatterAivDeterSmallExecutor][KernelRun]ReduceScatter aiv failed, return[%d]", ret),
        ret);

    ExtraArgs extraArgs;
    CHK_RET(SetOpCache(opArgs, topoArgs, resourceArgs, algArgs, extraArgs, aivProfilingInfo, false));

    HCCL_INFO("[CollReduceScatterAivDeterSmallExecutor][KernelRun]ReduceScatter aiv run success.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterAivDeterSmallExecutor", ReduceScatterAivDeterSmall, CollReduceScatterAivDeterSmallExecutor);

} // namespace hccl
