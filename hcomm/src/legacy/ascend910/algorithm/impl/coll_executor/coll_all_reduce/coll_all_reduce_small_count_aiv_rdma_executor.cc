/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_small_count_aiv_rdma_executor.h"
#include "sender.h"
#include "reducer.h"

namespace hccl {
constexpr s32 INTRA_RS_STEP = 0;
constexpr s32 INTER_AR_STEP = 1;
constexpr s32 INTRA_AG_STEP = 2;
constexpr u32 A_X_AGGR_SIZE = 2;
constexpr u32 A_X_SIZE = 16;
constexpr u64 HALF_OFFSET = 16 * 1024 * 1024;

u64 CollAllReduceSmallCountAivRdmaExecutor::allreduceSmallDataAivRdmaCount_ = 0;

CollAllReduceSmallCountAivRdmaExecutor::CollAllReduceSmallCountAivRdmaExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
    desc_.isAivMode = true;
    desc_.aivTagNum = AIV_A2_ALL_REDUCE_RDMA_KERNEL_NUM;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation > 1U ? topoAttr_.deviceNumPerAggregation - 1U : 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollAllReduceSmallCountAivRdmaExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));

    // aiv+rdma小数据量在server间使用HD通信域，并在多机A+X场景下当未设置使用RDMA时，默认使用PCIE
    bool isSingleAX = topoAttr_.serverNum == 1 && topoAttr_.moduleNum == 2; // A+X单机跨module
    if (topoMatcher_->GetExternalInputIntraRoceSwitch() == 0 && isSingleAX) {
        std::vector<SingleSubCommTransport>& commTransportLevel1 = opTransport[COMM_LEVEL1];
        for (u32 ringIndex = 0; ringIndex < commTransportLevel1.size(); ringIndex++) {
            for (auto& transportRequest : commTransportLevel1[ringIndex].transportRequests) {
                transportRequest.isUsedRdma = false;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    // 小数据量：使用AIVIN+AIVOUT，标记区在AIVOUT，RS前从inputPtr到AIVIN做本地拷贝
    inputType = TransportMemType::AIV_INPUT;
    outputType = TransportMemType::AIV_OUTPUT;
    HCCL_INFO(
        "[CollAllReduceSmallCountAivRdmaExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL1, CommType::COMM_TAG_HALVING_DOUBLING);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL1], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::CalNumBlocks(
    u32& numBlocks, u32 rankSize, [[maybe_unused]] u64 dataSize, [[maybe_unused]] HcclCMDType cmdType)
{
    numBlocks = rankSize; // 默认情况使用rankSize个AIV
    u32 bestNumBlocks = numBlocks;

    CHK_PRT_RET(
        numBlocks_ < numBlocks,
        HCCL_WARNING(
            "[CollAllReduceSmallCountAivRdmaExecutor][CalNumBlocks]aivCore[%u] is invalid, at least need [%u].",
            numBlocks_, numBlocks),
        HCCL_E_PARA);

    HCCL_INFO(
        "[CollAllReduceSmallCountAivRdmaExecutor][CalNumBlocks] numBlocks is set to [%u], limit[%u], recommended[%u]",
        numBlocks, numBlocks_, bestNumBlocks);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    allreduceSmallDataAivRdmaCount_ += 1;
    HCCL_INFO(
        "[CollAllReduceSmallCountAivRdmaExecutor][Orchestrate] AllreduceSmallCountAivRdma has been called [%llu].",
        allreduceSmallDataAivRdmaCount_);
    tag_ = param.tag;
    algResResp_ = &algRes;

    // 小数据量：使用AIVIN+AIVOUT，标记区在AIVOUT
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    execMem.inputMem = algRes.aivInputMem;
    execMem.outputMem = algRes.aivOutputMem;
    HcclResult ret = KernelRun(param, execMem);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAllReduceSmallCountAivRdmaExecutor]errNo[0x%016llx] tag[%s] executor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()),
        ret);

    HCCL_INFO(
        "tag[%s], AllReduce executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::GetAdjInfo(
    [[maybe_unused]] AlgResourceResponse& algRes, [[maybe_unused]] AdjInfo& adjInfo)
{
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::InterServerHDOneshot(
    const OpParam& param, ExecMem& execMem, u32& outputOffset, u64 sliceCount, u32 dbOffset, u32 interRankSize,
    u32 interRankId, bool isOpbase, std::vector<LINK>& interLinks)
{
    (void)isOpbase;
    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
    HCCL_INFO("[CollAllReduceSmallCountAivRdmaExecutor][InterServerHDOneshot]reduceAttr is [%llu].", reduceAttr);
    std::unique_ptr<Sender> senderInfo;
    std::unique_ptr<Reducer> reducerInfo;
    senderInfo.reset(new (std::nothrow) Sender(param.DataDes.dataType, param.reduceType, reduceAttr));
    CHK_SMART_PTR_NULL(senderInfo);
    reducerInfo.reset(new (std::nothrow) Reducer(param.DataDes.dataType, param.reduceType, reduceAttr));
    CHK_SMART_PTR_NULL(reducerInfo);
    u32 hdStepNum = static_cast<u32>(log2(interRankSize));
    HCCL_INFO(
        "[CollAllReduceSmallCountAivRdmaExecutor] Find interlink type for cross-aggregation link[%d].",
        interLinks[(interRankId + 1) % A_X_AGGR_SIZE + interRankId - interRankId % A_X_AGGR_SIZE]->GetLinkType());
    u32 sliceSize = sliceCount * SIZE_TABLE[param.DataDes.dataType];
    auto opMeta = HcclOpMetaInfo::GetOneForAllReduce(
        0, param.DataDes.dataType, ReduceType::INLINE_REDUCE, true, 0, false, hccl::CopyPattern::BCOPY, 1, true);
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
    for (u32 step = 1; step <= hdStepNum; step++) {
        u32 peerMask = 1 << (step - 1);
        u32 peer = interRankId ^ peerMask;
        HCCL_INFO("[CollAllReduceSmallCountAivRdmaExecutor][InterServerHDOneshot] Step %u, peer %u.", step, peer);
        u32 sliceForReadOffset = HCCL_SMALL_COUNT_2_MB + (step - 1) * sliceSize + dbOffset;
        u32 sliceForWriteOffset = HCCL_SMALL_COUNT_2_MB + step * sliceSize + dbOffset;
        DeviceMem src = execMem.inputMem.range(sliceForReadOffset, sliceSize);
        DeviceMem dst = execMem.inputMem.range(sliceForWriteOffset, sliceSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, const_cast<Stream&>(param.stream)));
        interLinks[peer]->TxAck(const_cast<Stream&>(param.stream));
        interLinks[peer]->RxAck(const_cast<Stream&>(param.stream));
        if (interLinks[peer]->IsSupportTransportWithReduce()
            && ((interLinks[peer]->GetLinkType() == LinkType::LINK_STANDARD_ROCE)
                || static_cast<bool>((RDMA_REDUCE_BITMASK & reduceAttr)))) {
            HCCL_INFO("[CollAllReduceSmallCountAivRdmaExecutor][InterServerHDOneshot] inter use RDMA");
            CHK_RET(senderInfo->run(
                interLinks[peer], sliceForWriteOffset, src, const_cast<Stream&>(param.stream), UserMemType::INPUT_MEM));
            CHK_RET(reducerInfo->run(
                dispatcher_, interLinks[peer], 0, src, src, src, const_cast<Stream&>(param.stream),
                DstMemType::RESULT_INPUT_MEM, UserMemType::INPUT_MEM));
        } else if (interLinks[peer]->IsSpInlineReduce() && static_cast<bool>((INLINE_REDUCE_BITMASK & reduceAttr))) {
            HCCL_INFO("[CollAllReduceSmallCountAivRdmaExecutor][InterServerHDOneshot] inter use SDMA");
            CHK_RET(senderInfo->run(
                interLinks[peer], sliceForWriteOffset, src, const_cast<Stream&>(param.stream), UserMemType::INPUT_MEM));
            CHK_RET(reducerInfo->run(
                dispatcher_, interLinks[peer], sliceForReadOffset, dst, dst, dst, const_cast<Stream&>(param.stream),
                DstMemType::RESULT_INPUT_MEM, UserMemType::INPUT_MEM));
        } else {
            CHK_RET(interLinks[peer]->TxAsync(
                UserMemType::INPUT_MEM, HALF_OFFSET + sliceForWriteOffset, src.ptr(), src.size(),
                const_cast<Stream&>(param.stream)));
            DeviceMem localSrc = execMem.inputMem.range(HALF_OFFSET + sliceForWriteOffset, sliceSize);
            CHK_RET(reducerInfo->run(
                dispatcher_, interLinks[peer], 0, localSrc, dst, src, const_cast<Stream&>(param.stream),
                DstMemType::RESULT_INPUT_MEM, UserMemType::INPUT_MEM));
        }
    }
    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));
    outputOffset = HCCL_SMALL_COUNT_2_MB + hdStepNum * sliceSize;
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountAivRdmaExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_INFO("[CollAllReduceSmallCountAivRdmaExecutor][KernelRun]AllReduce aiv enter");
    HcclWorkflowMode workflow = workflowMode_;
    bool isOpbase = (workflow == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    // 获取通信域信息
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    // 数据准备，按照server内rankSize切片
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];
    u64 totalSize = param.DataDes.count * perDataSize;
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    u32 sliceNum = level0CommInfo.localRankSize;
    CHK_RET(PrepareSliceDataWithAlignSize(totalSize, sliceNum, 0, dataSegsSlice, perDataSize));
    CHK_PRT_RET(
        commIndex >= dataSegsSlice.size(),
        HCCL_ERROR(
            "[CollAllReduceSmallCountAivRdmaExecutor][Run]commIndex[%u] >= dataSegsSlice size[%zu]", commIndex,
            dataSegsSlice.size()),
        HCCL_E_INTERNAL);
    std::vector<hccl::LINK> intraLinks = level0CommInfo.links;
    std::vector<hccl::LINK> interLinks = level1CommInfo.links;
    u32 intraRankSize = level0CommInfo.localRankSize;
    u32 interRankSize = level1CommInfo.localRankSize;
    u32 intraRankId = level0CommInfo.localRank;
    u32 interRankId = level1CommInfo.localRank;

    // reduce scatter via AIV
    void* dataBuffers[MAX_RANK_SIZE];
    void* flagBuffers[MAX_RANK_SIZE]; // 标记区的具体偏移在kernel中决定
    CHK_RET(PrepareAivBuffers(
        intraRankSize, intraRankId, 0, execMem.inputMem, execMem.inputMem, intraLinks, dataBuffers, flagBuffers,
        UserMemType::INPUT_MEM, UserMemType::INPUT_MEM, 0, HCCL_MID_COUNT_32_MB));
    // RS总数据量最大1m，rs的结果存储到2m处
    void* rsOutput = static_cast<u8*>(execMem.inputMem.ptr()) + HCCL_SMALL_COUNT_2_MB;

    AivOpArgs opArgs{HcclCMDType::HCCL_CMD_ALLREDUCE, execMem.inputPtr, rsOutput, execMem.count,
                     param.DataDes.dataType,          param.reduceType, 0,        isOpbase};
    AivTopoArgs topoArgs{intraRankId, intraRankSize, topoAttr_.isDiffDeviceModule ? topoAttr_.devicePhyId : A_X_SIZE};
    topoArgs.identify = algoAttr_.identifier;
    u32 numBlocks;
    CHK_PRT_RET(
        CalNumBlocks(numBlocks, intraRankSize) != HCCL_SUCCESS, HCCL_ERROR("[%s] CalNumBlocks failed", __func__),
        HCCL_E_PARA);
    numBlocks_ = numBlocks;
    AivResourceArgs resourceArgs{param.tag,  param.stream.ptr(), dataBuffers, flagBuffers, execMem.inputMem.size(),
                                 numBlocks_, param.aivTag};
    AivAlgArgs algArgs{INTRA_RS_STEP, true};
    algArgs.execTimeOut = topoMatcher_->GetExecTimeOutConfig();
    algArgs.execTimeOutSet = true;
    struct AivProfilingInfo aivProfilingInfo;
    aivProfilingInfo.counter = opCounter_;

    CHK_RET(ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo));

    // use hd algo
    u32 arOutputOffset = 0; // 跨机allreduce的结果的位置，相对于inputMem的偏移
    CHK_RET(InterServerHDOneshot(
        param, execMem, arOutputOffset, dataSegsSlice[commIndex].size / perDataSize, 0, interRankSize, interRankId,
        isOpbase, interLinks));
    void* arOutput = static_cast<u8*>(execMem.inputMem.ptr()) + arOutputOffset;

    // AllGather via AIV
    CHK_RET(PrepareAivBuffers(
        intraRankSize, intraRankId, 0, execMem.inputMem, execMem.inputMem, intraLinks, dataBuffers, flagBuffers,
        UserMemType::INPUT_MEM, UserMemType::INPUT_MEM, HCCL_SMALL_COUNT_8_MB, HCCL_MID_COUNT_32_MB));

    opArgs.input = arOutput;
    opArgs.output = execMem.outputPtr;
    resourceArgs.buffersIn = dataBuffers;
    resourceArgs.buffersOut = flagBuffers;
    resourceArgs.aivTag = GetNextAivTag(resourceArgs.aivTag);
    algArgs.step = INTRA_AG_STEP;

    CHK_RET(ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo));

    HCCL_INFO("[CollAllReduceSmallCountAivRdmaExecutor][KernelRun]AllReduce aiv run success.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceSmallCountAivRdmaExecutor", AllReduceSmallCountAivRdma, CollAllReduceSmallCountAivRdmaExecutor);

} // namespace hccl
