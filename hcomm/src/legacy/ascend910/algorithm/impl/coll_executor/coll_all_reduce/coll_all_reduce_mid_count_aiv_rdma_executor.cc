/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_mid_count_aiv_rdma_executor.h"

namespace hccl {
constexpr s32 INTRA_RS_STEP = 0;
constexpr s32 INTRA_AG_STEP = 2;

CollAllReduceMidCountAivRdmaExecutor::CollAllReduceMidCountAivRdmaExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
    desc_.isAivMode = true;
    desc_.aivTagNum = AIV_A2_ALL_REDUCE_RDMA_KERNEL_NUM;
}

HcclResult CollAllReduceMidCountAivRdmaExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation > 1U ? topoAttr_.deviceNumPerAggregation - 1U : 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollAllReduceMidCountAivRdmaExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMidCountAivRdmaExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMidCountAivRdmaExecutor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    // 中数据量：使用AIVIN，标记区在AIVIN末尾，单算子模式用CCLOUT，图模式用USEROUT
    inputType = TransportMemType::AIV_INPUT;
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollAllReduceMidCountAivRdmaExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMidCountAivRdmaExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMidCountAivRdmaExecutor::CalNumBlocks(
    u32& numBlocks, u32 rankSize, [[maybe_unused]] u64 dataSize, [[maybe_unused]] HcclCMDType cmdType)
{
    numBlocks = rankSize; // 默认情况使用rankSize个AIV
    u32 bestNumBlocks = numBlocks;

    CHK_PRT_RET(
        numBlocks_ < numBlocks,
        HCCL_WARNING(
            "[CollAllReduceMidCountAivRdmaExecutor][CalNumBlocks]aivCore[%u] is invalid, at least need [%u].",
            numBlocks_, numBlocks),
        HCCL_E_PARA);

    HCCL_INFO(
        "[CollAllReduceMidCountAivRdmaExecutor][CalNumBlocks] numBlocks is set to [%u], limit[%u], recommended[%u]",
        numBlocks, numBlocks_, bestNumBlocks);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMidCountAivRdmaExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    // 中数据量：使用AIVIN，标记区在AIVIN末尾，单算子模式用CCLOUT，图模式用USEROUT
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.inputMem = algRes.aivInputMem;
    execMem.outputPtr = param.outputPtr;
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.outputMem = algRes.cclOutputMem;
    } else {
        execMem.outputMem = algRes.paramOutputMem;
    }
    HcclResult ret = KernelRun(param, execMem);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAllReduceMidCountAivRdmaExecutor]errNo[0x%016llx] tag[%s] executor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()),
        ret);

    HCCL_INFO(
        "tag[%s], AllReduce executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMidCountAivRdmaExecutor::GetAdjInfo(
    [[maybe_unused]] AlgResourceResponse& algRes, [[maybe_unused]] AdjInfo& adjInfo)
{
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMidCountAivRdmaExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllReduceMidCountAivRdmaExecutor][KernelRun]AllReduce aiv enter");
    HcclWorkflowMode workflow = workflowMode_;
    bool isOpbase = (workflow == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    CHK_RET(ActiveSlaveStreams(param.stream));

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
    CHK_RET(PrepareSliceDataWithAlignSize(totalSize, sliceNum, 0, dataSegsSlice, HCCL_ALIGN_COUNT_32_B));
    CHK_PRT_RET(
        commIndex >= dataSegsSlice.size(),
        HCCL_ERROR(
            "[CollAllReduceMidCountAivRdmaExecutor][Run]commIndex[%u] >= dataSegsSlice size[%zu]", commIndex,
            dataSegsSlice.size()),
        HCCL_E_INTERNAL);
    std::vector<hccl::LINK> intraLinks = level0CommInfo.links;
    std::vector<hccl::LINK> interLinks = level1CommInfo.links;
    u32 intraRankSize = level0CommInfo.localRankSize;
    u32 intraRankId = level0CommInfo.localRank;

    // reduce scatter阶段，inputMem0-31m做数据区，32M开始后的1M做标记区
    void* dataBuffers[MAX_RANK_SIZE];
    void* flagBuffers[MAX_RANK_SIZE]; // 标记区的具体偏移在kernel中决定
    CHK_RET(PrepareAivBuffers(
        intraRankSize, intraRankId, 0, execMem.inputMem, execMem.inputMem, intraLinks, dataBuffers, flagBuffers,
        UserMemType::INPUT_MEM, UserMemType::INPUT_MEM, 0, HCCL_MID_COUNT_32_MB));
    // 先做本地拷贝到AIVIN再跨片拷贝；output统一为allreduceInput的位置，即buffer中原位

    AivOpArgs opArgs{HcclCMDType::HCCL_CMD_ALLREDUCE, execMem.inputPtr, nullptr, execMem.count,
                     param.DataDes.dataType,          param.reduceType, 0,       isOpbase};
    AivTopoArgs topoArgs{intraRankId, intraRankSize};
    u32 numBlocks;
    CHK_PRT_RET(
        CalNumBlocks(numBlocks, intraRankSize) != HCCL_SUCCESS, HCCL_ERROR("[%s] CalNumBlocks failed", __func__),
        HCCL_E_PARA);
    numBlocks_ = numBlocks;
    topoArgs.identify = algoAttr_.identifier;
    AivResourceArgs resourceArgs{param.tag,  param.stream.ptr(), dataBuffers, flagBuffers, execMem.inputMem.size(),
                                 numBlocks_, param.aivTag};
    AivAlgArgs algArgs{INTRA_RS_STEP, false};
    algArgs.execTimeOut = topoMatcher_->GetExecTimeOutConfig();
    algArgs.execTimeOutSet = true;
    struct AivProfilingInfo aivProfilingInfo;
    aivProfilingInfo.counter = opCounter_;

    CHK_RET(ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo));

    // allreduce 阶段
    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    DeviceMem allreduceInput = execMem.inputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(allreduceInput);
    DeviceMem allreduceOutput = execMem.outputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(allreduceOutput);

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
    auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    auto opMeta = HcclOpMetaInfo::GetOneForAllReduce(
        autoSelectedAlgTypeLevel1, param.DataDes.dataType, ReduceType::INLINE_REDUCE, IsAllReduceSmallData(totalSize),
        1, false, hccl::CopyPattern::BCOPY, 1, true);
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));

    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
        HCCL_INFO("AllReduce mesh: using ring algo inter-server.");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        u64 curSize = execMem.count * perDataSize; // 单位 byte
        HCCL_DEBUG(
            "AllReduce mesh: curSize[%llu] deviceNumPerAggregation[%u] commLevel0Size[%u]", curSize,
            topoAttr_.deviceNumPerAggregation, level0CommInfo.localRankSize);
        if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_ALLREDUCE_SMALL_SIZE) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_REDUCE_NHR_ONESHOT, dispatcher_);
        } else {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR, dispatcher_);
        }
        HCCL_INFO("AllReduce mesh: using nhr algo inter-server.");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
        level1TempAlg->CloseBarrier();
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR_V1, dispatcher_);
        HCCL_INFO("AllReduce mesh: using nhr_v1 algo inter-server.");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NB, dispatcher_);
        HCCL_INFO("AllReduce mesh: using nb algo inter-server.");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
    } else {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_REDUCE_RECURSIVE_HALVING_DOUBLING, dispatcher_);
        HCCL_INFO("AllReduce mesh: using Recursive halving-doubling algo inter-server.");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
    }
    CHK_SMART_PTR_NULL(level1TempAlg);

    u32 rankSize = level1CommInfo.localRankSize;
    u64 hdCount = dataSegsSlice[commIndex].size / perDataSize;
    CHK_RET(level1TempAlg->Prepare(
        allreduceInput, allreduceOutput, allreduceOutput, hdCount, param.DataDes.dataType, param.stream,
        param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), dataSegsSlice[commIndex].offset));

    CHK_RET(level1TempAlg->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
        param.stream));
    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    HCCL_INFO("[CollAllReduceMidCountAivRdmaExecutor] rdma stage run success.");
    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));

    // allgather阶段，outputMem做数据区，32M开始后的1M做标记区
    CHK_RET(PrepareAivBuffers(
        intraRankSize, intraRankId, 0, execMem.outputMem, execMem.inputMem, intraLinks, dataBuffers, flagBuffers,
        UserMemType::OUTPUT_MEM, UserMemType::INPUT_MEM, 0, HCCL_MID_COUNT_32_MB));
    // 输入统一为allreduceOutput的位置，各卡不同；单算子模式需要outputAddr，先做本地拷贝再跨片拷贝；图模式结果直接放在CCL
    // Out中

    opArgs.input = nullptr;
    opArgs.output = execMem.outputPtr;
    resourceArgs.buffersIn = dataBuffers;
    resourceArgs.buffersOut = flagBuffers;
    resourceArgs.aivTag = GetNextAivTag(resourceArgs.aivTag);
    algArgs.step = INTRA_AG_STEP;

    CHK_RET(ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo));

    HCCL_INFO("[CollAllReduceMidCountAivRdmaExecutor][KernelRun]AllReduce aiv run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceMidCountAivRdmaExecutor", AllReduceMidCountAivRdma, CollAllReduceMidCountAivRdmaExecutor);

} // namespace hccl
