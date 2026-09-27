/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_mesh_aiv_executor.h"

namespace hccl {

CollAlltoAllMeshAivExecutor::CollAlltoAllMeshAivExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{
    desc_.isAivMode = true;
}

HcclResult CollAlltoAllMeshAivExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = 0; // AIV通信不需要申请从流
    HCCL_INFO("[CollAlltoAllMeshAivExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::AIV_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::AIV_OUTPUT;
    }
    HCCL_INFO(
        "[CollAlltoAllMeshAivExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_MESH_L0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_MESH_L0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivExecutor::CalNumBlocks(u32& numBlocks, u32 rankSize, u64 dataSize, HcclCMDType cmdType)
{
    numBlocks = rankSize; // 默认情况使用rankSize个AIV

    bool isOpBase = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    if (cmdType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93 && !isOpBase) {
            numBlocks = rankSize * NUM_BLOCKS_FOUR_PER_RANK_A3 > MAX_NUM_BLOCKS ?
                            rankSize * NUM_BLOCKS_THREE_PER_RANK_A3 :
                            rankSize * NUM_BLOCKS_FOUR_PER_RANK_A3;
        } else if (isOpBase && dataSize >= AIV_ALL_TO_ALL_BIG_SIZE) {
            numBlocks = NUM_BLOCKS_FACTOR_TWO * rankSize; // 单机场景，单算子AlltoAll使用2倍 rankSize个aiv
        }
    } else if (cmdType == HcclCMDType::HCCL_CMD_ALLTOALLVC || cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93
            && ((isOpBase && cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV)
                || (!isOpBase && cmdType == HcclCMDType::HCCL_CMD_ALLTOALLVC))) {
            // A3单机单算子场景，numBlocks_为3倍或者4倍的ranksize
            numBlocks = rankSize * NUM_BLOCKS_FOUR_PER_RANK_A3 > MAX_NUM_BLOCKS ?
                            rankSize * NUM_BLOCKS_THREE_PER_RANK_A3 :
                            rankSize * NUM_BLOCKS_FOUR_PER_RANK_A3;
        } else if (isOpBase) {
            numBlocks = NUM_BLOCKS_FACTOR_TWO * rankSize; // 单机场景，单算子AlltoAll使用2倍 rankSize个aiv
        }
    }

    u32 bestNumBlocks = numBlocks;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910B
        || (isOpBase && topoAttr_.deviceType == DevType::DEV_TYPE_910_93 && cmdType != HcclCMDType::HCCL_CMD_ALLTOALLV)
        || (!isOpBase && topoAttr_.deviceType == DevType::DEV_TYPE_910_93
            && cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV)) {
        CHK_PRT_RET(
            numBlocks_ < numBlocks,
            HCCL_WARNING(
                "[CollAlltoAllMeshAivExecutor][CalNumBlocks]aivCore[%u] is invalid, at least need [%u].", numBlocks_,
                numBlocks),
            HCCL_E_PARA);
    } else if (
        (isOpBase && topoAttr_.deviceType == DevType::DEV_TYPE_910_93 && cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV)
        || (!isOpBase && topoAttr_.deviceType == DevType::DEV_TYPE_910_93
            && cmdType != HcclCMDType::HCCL_CMD_ALLTOALLV)) {
        CHK_PRT_RET(
            numBlocks_ < rankSize,
            HCCL_WARNING(
                "[CollAlltoAllMeshAivExecutor][CalNumBlocks]aivCore[%u] is invalid, at least need [%u].", numBlocks_,
                rankSize),
            HCCL_E_PARA);
        if (numBlocks_ < numBlocks) {
            numBlocks = numBlocks_ / rankSize * rankSize;
        }
    }

    HCCL_INFO(
        "[CollAlltoAllMeshAivExecutor][CalNumBlocks] numBlocks is set to [%u], limit[%u], recommended[%u]", numBlocks,
        numBlocks_, bestNumBlocks);
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    HcclResult ret = HCCL_SUCCESS;
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.aivOutputMem; // 存放flag
        ret = KernelRun(param, execMem);
    } else {
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.aivOutputMem;
        ret = KernelRun(param, execMem);
    }

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAlltoAllMeshAivExecutor][Orchestrate]errNo[0x%016llx] tag[%s] executor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()),
        ret);

    HCCL_INFO(
        "tag[%s], AlltoAll executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult
CollAlltoAllMeshAivExecutor::GetAdjInfo([[maybe_unused]] AlgResourceResponse& algRes, [[maybe_unused]] AdjInfo& adjInfo)
{
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivExecutor::GetAivExecParam(
    const OpParam& param, AlgResourceResponse& algRes, AivSuperKernelArgs& args)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    HcclResult ret = HCCL_SUCCESS;
    ExecMem execMem;
    execMem.count = param.All2AllDataDes.sendCount;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    // 单算子大数据量
    execMem.inputMem = algRes.paramInputMem;
    execMem.outputMem = algRes.aivOutputMem;

    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_MESH_L0, COMM_INDEX_0);

    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG(
        "[CollAlltoAllMeshAivExecutor][GetAivExecParam] userRank [%d] localRank [%d]", topoAttr_.userRank, localRank);

    for (u32 i = 0; i < localRankSize; i++) {
        if (i != localRank) {
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(args.buffersIn[i])));
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(args.buffersOut[i])));
        } else {
            args.buffersIn[i] = execMem.inputMem.ptr();
            args.buffersOut[i] = execMem.outputMem.ptr();
        }
    }
    args.rank = localRank;
    args.rankSize = localRankSize;
    args.len = execMem.count;
    args.dataType = param.All2AllDataDes.sendType;
    args.unitSize = SIZE_TABLE[param.All2AllDataDes.sendType];
    args.devType = static_cast<u32>(topoAttr_.deviceType);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAlltoAllMeshAivExecutor][Orchestrate]errNo[0x%016llx] tag[%s] executor kernel "
            "run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()),
        ret);

    HCCL_INFO(
        "tag[%s], AlltoAll executor getalgexecparam success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAlltoAllMeshAivExecutor][KernelRun]AllToAll aiv enter");

    CHK_RET(CheckCommSize(COMM_MESH_L0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_MESH_L0, COMM_INDEX_0);

    void* buffersIn[MAX_RANK_SIZE];
    void* buffersOut[MAX_RANK_SIZE];

    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG("[CollAlltoAllMeshAivExecutor][KernelRun] userRank [%u] localRank [%u]", topoAttr_.userRank, localRank);

    for (u32 i = 0; i < localRankSize; i++) {
        if (i != localRank) {
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(buffersIn[i])));
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(buffersOut[i])));
        } else {
            buffersIn[i] = execMem.inputMem.ptr();
            buffersOut[i] = execMem.outputMem.ptr();
        }
    }

    ExtraArgs extraArgs;
    bool isOpbase = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HcclResult ret;
    u64 dataSize
        = (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL ?
               param.All2AllDataDes.sendCount * SIZE_TABLE[param.All2AllDataDes.sendType] :
               0);

    AivOpArgs opArgs{
        HcclCMDType::HCCL_CMD_ALLTOALL,
        execMem.inputPtr,
        execMem.outputPtr,
        0,
        param.All2AllDataDes.sendType,
        HCCL_REDUCE_RESERVED,
        0,
        isOpbase};
    AivTopoArgs topoArgs{localRank, localRankSize, MAX_RANK_SIZE, 0, topoAttr_.serverNum, topoAttr_.deviceType};
    topoArgs.identify = algoAttr_.identifier;
    AivResourceArgs resourceArgs{param.tag, param.stream.ptr(), buffersIn, buffersOut, execMem.inputMem.size(),
                                 0,         param.aivTag};
    AivAlgArgs algArgs{};
    algArgs.execTimeOut = topoMatcher_->GetExecTimeOutConfig();
    algArgs.execTimeOutSet = true;
    struct AivProfilingInfo aivProfilingInfo;
    aivProfilingInfo.counter = opCounter_;

    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL
        && ((isOpbase && dataSize < AIV_ALL_TO_ALL_BIG_SIZE)
            || (!isOpbase && topoAttr_.deviceType == DevType::DEV_TYPE_910_93))) {
        opArgs.count = param.All2AllDataDes.sendCount;
        u32 numBlocks;
        CHK_PRT_RET(
            CalNumBlocks(numBlocks, localRankSize, dataSize, opArgs.cmdType) != HCCL_SUCCESS,
            HCCL_ERROR("[%s] CalNumBlocks failed", __func__), HCCL_E_PARA);
        numBlocks_ = numBlocks;
        resourceArgs.numBlocks = numBlocks_;
        ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo);
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC || param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        for (u32 i = 0; i < localRankSize; i++) {
            u64 rankCount = 0;
            for (u32 j = 0; j < localRankSize; j++) {
                u64 curSendCount
                    = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + i * localRankSize + j);
                extraArgs.sendCountMatrix[i * localRankSize + j] = curSendCount;
                rankCount += curSendCount;
            }
            if (rankCount > extraArgs.maxCount) {
                extraArgs.maxCount = rankCount;
            }
        }
        opArgs.count = extraArgs.maxCount;
        opArgs.cmdType = HcclCMDType::HCCL_CMD_ALLTOALLVC;
        u32 numBlocks;
        CHK_PRT_RET(
            CalNumBlocks(numBlocks, localRankSize, dataSize, opArgs.cmdType) != HCCL_SUCCESS,
            HCCL_ERROR("[%s] CalNumBlocks failed", __func__), HCCL_E_PARA);
        numBlocks_ = numBlocks;
        resourceArgs.numBlocks = numBlocks_;
        ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, extraArgs, aivProfilingInfo);
    } else {
        for (u32 i = 0; i < localRankSize; i++) {
            extraArgs.sendCounts[i] = *(static_cast<const u64*>(param.All2AllDataDes.sendCounts) + i);
            extraArgs.sendDispls[i] = *(static_cast<const u64*>(param.All2AllDataDes.sdispls) + i);
            extraArgs.recvCounts[i] = *(static_cast<const u64*>(param.All2AllDataDes.recvCounts) + i);
            extraArgs.recvDispls[i] = *(static_cast<const u64*>(param.All2AllDataDes.rdispls) + i);
        }

        opArgs.cmdType = HcclCMDType::HCCL_CMD_ALLTOALLV;
        u32 numBlocks;
        CHK_PRT_RET(
            CalNumBlocks(numBlocks, localRankSize, dataSize, opArgs.cmdType) != HCCL_SUCCESS,
            HCCL_ERROR("[%s] CalNumBlocks failed", __func__), HCCL_E_PARA);
        numBlocks_ = numBlocks;
        resourceArgs.numBlocks = numBlocks_;
        ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, extraArgs, aivProfilingInfo);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[CollAlltoAllMeshAivExecutor][KernelRun]AllToAll aiv failed, return[%d]", ret),
        ret);

    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL && isOpbase) {
        CHK_RET(SetOpCache(opArgs, topoArgs, resourceArgs, algArgs, extraArgs, aivProfilingInfo, false));
    }

    HCCL_INFO("[CollAlltoAllMeshAivExecutor][KernelRun]AllToAll aiv run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AlltoAllMeshAivExecutor", AlltoAllMeshAiv, CollAlltoAllMeshAivExecutor);

} // namespace hccl
