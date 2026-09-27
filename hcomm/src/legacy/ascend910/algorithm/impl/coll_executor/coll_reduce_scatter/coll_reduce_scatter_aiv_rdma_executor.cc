/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_aiv_rdma_executor.h"
#include "alg_template_register.h"

namespace hccl {
constexpr u32 A_X_SIZE = 16;

CollReduceScatterAivRdmaExecutor::CollReduceScatterAivRdmaExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
    desc_.isAivMode = true;
}

void CollReduceScatterAivRdmaExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    root_ = param.root;
    opType_ = param.opType;
    // 记录图模式总数据量
    totalSize_ = topoAttr_.userRankSize * param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
}

HcclResult CollReduceScatterAivRdmaExecutor::CalcScratchMemSize(u64& scratchMemSize)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        scratchMemSize = 0U;
    } else {
        scratchMemSize = totalSize_;
    }
    HCCL_INFO(
        "[CollReduceScatterAivRdmaExecutor][CalcScratchMemSize] tag[%s] scratchMemSize[%llu]", tag_.c_str(),
        scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterAivRdmaExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult
CollReduceScatterAivRdmaExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    // 使用AIVIN，标记区在AIVIN末尾，单算子模式用CCLOUT，图模式用USEROUT
    inputType = TransportMemType::AIV_INPUT;
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        outputType = TransportMemType::SCRATCH;
    }

    HCCL_INFO(
        "[CollReduceScatterAivRdmaExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterAivRdmaExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterAivRdmaExecutor::CalNumBlocks(
    u32& numBlocks, u32 rankSize, [[maybe_unused]] u64 dataSize, [[maybe_unused]] HcclCMDType cmdType)
{
    numBlocks = rankSize; // 多机场景，单算子ReduceScatter使用rankSize个aiv
    u32 bestNumBlocks = numBlocks;

    CHK_PRT_RET(
        numBlocks_ < numBlocks,
        HCCL_WARNING(
            "[CollReduceScatterAivRdmaExecutor][CalNumBlocks]aivCore[%u] is invalid, at least need [%u].", numBlocks_,
            numBlocks),
        HCCL_E_PARA);

    HCCL_INFO(
        "[CollReduceScatterAivRdmaExecutor][CalNumBlocks] numBlocks is set to [%u], limit[%u], recommended[%u]",
        numBlocks, numBlocks_, bestNumBlocks);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterAivRdmaExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HCCL_INFO("[CollReduceScatterAivRdmaExecutor][Orchestrate]start");

    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    // OutputMem单算子模式用CCLOUT，图模式用USEROUT
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    execMem.inputMem = algRes.aivInputMem;
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.cclOutputMem;
    } else {
        execMem.outputMem = algRes.paramOutputMem;
        execMem.scratchMem = algRes.scratchMem;
    }
    HcclResult ret = KernelRun(param, execMem);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceScatterAivRdmaExecutor]errNo[0x%016llx] tag[%s] executor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()),
        ret);

    HCCL_INFO(
        "tag[%s], ReduceScatter executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterAivRdmaExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterAivRdmaExecutor][KernelRun]ReduceScatter aiv enter");

    HcclWorkflowMode workflow = workflowMode_;
    bool isOpbase = (workflow == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    // 获取通信域信息
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = outerCommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    /*  第一步 节点内重排序RS */
    // 数据准备，按照server内rankSize切片
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];
    u64 perRankSize = param.DataDes.count * perDataSize;
    std::vector<hccl::LINK> intraLinks = outerCommInfo.links; // 机间
    std::vector<hccl::LINK> interLinks = innerCommInfo.links; // 机内
    u32 intraRankSize = outerCommInfo.localRankSize;
    u32 intraRankId = outerCommInfo.localRank;

    // reduce scatter阶段，inputMem0-31m做数据区，32M开始后的1M做标记区
    void* dataBuffers[MAX_RANK_SIZE];
    void* flagBuffers[MAX_RANK_SIZE]; // 标记区的具体偏移在kernel中决定
    CHK_RET(PrepareAivBuffers(
        intraRankSize, intraRankId, 0, execMem.inputMem, execMem.inputMem, intraLinks, dataBuffers, flagBuffers,
        UserMemType::INPUT_MEM, UserMemType::INPUT_MEM, 0, HCCL_MID_COUNT_32_MB));

    u32 serverNum = innerCommInfo.localRankSize;
    // 先做本地拷贝到AIVIN再跨片拷贝；output统一为reduceScatterInput的位置，即buffer中原位
    AivOpArgs opArgs{
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER,
        execMem.inputPtr,
        execMem.outputPtr,
        execMem.count,
        param.DataDes.dataType,
        param.reduceType,
        0,
        isOpbase};
    AivTopoArgs topoArgs{
        intraRankId,
        intraRankSize,
        topoAttr_.isDiffDeviceModule ? topoAttr_.devicePhyId : A_X_SIZE,
        0,
        serverNum,
        topoAttr_.deviceType,
        algoAttr_.identifier};
    u32 numBlocks;
    CHK_PRT_RET(
        CalNumBlocks(numBlocks, intraRankSize) != HCCL_SUCCESS, HCCL_ERROR("[%s] CalNumBlocks failed", __func__),
        HCCL_E_PARA);
    numBlocks_ = numBlocks;
    AivResourceArgs resourceArgs{param.tag,  param.stream.ptr(), dataBuffers, flagBuffers, execMem.inputMem.size(),
                                 numBlocks_, param.aivTag};
    AivAlgArgs algArgs{0};
    algArgs.execTimeOut = topoMatcher_->GetExecTimeOutConfig();
    algArgs.execTimeOutSet = true;
    struct AivProfilingInfo aivProfilingInfo;
    aivProfilingInfo.counter = opCounter_;

    CHK_RET(ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo));
    /*  第二步  节点间RS */
    auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    ReduceType reduceType
        = ((param.reduceType != HCCL_REDUCE_PROD) && (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)) ?
              ReduceType::INLINE_REDUCE :
              ReduceType::TBE_REDUCE;
    auto opMeta = HcclOpMetaInfo::GetOneForReduceScatter(
        autoSelectedAlgTypeLevel1, param.DataDes.dataType, reduceType, false, false, CopyPattern::BCOPY, false, 0,
        true);
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));

    u32 innerRankSize = innerCommInfo.localRankSize;
    DeviceMem inputMem = execMem.inputMem;
    if (innerRankSize > 1) {
        // execMem.inputMem要改成按平面制定的初始位置
        u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<ExecutorBase> innerExecutor;
        std::vector<Slice> dataSegsSlice;
        dataSegsSlice.resize(innerRankSize);
        for (u32 i = 0; i < innerRankSize; i++) {
            dataSegsSlice[i].size = perRankSize;
            dataSegsSlice[i].offset = (commIndex * innerRankSize + i) * perRankSize;
        }
        u64 count = param.DataDes.count;
        u64 baseOffset = 0;
        DeviceMem inputMem = execMem.inputMem;
        DeviceMem scratchMem = execMem.scratchMem;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            innerExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
            CHK_SMART_PTR_NULL(innerExecutor);
            CHK_RET(innerExecutor->Prepare(reduceAttr));
            HCCL_INFO("ReduceScatter mesh: using ring algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            innerExecutor
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
            CHK_SMART_PTR_NULL(innerExecutor);
            CHK_RET(innerExecutor->Prepare(reduceAttr, false));
            innerExecutor->CloseBarrier();
            HCCL_INFO("ReduceScatter mesh: using nhr algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            innerExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_NHR_V1, dispatcher_);
            CHK_SMART_PTR_NULL(innerExecutor);
            CHK_RET(innerExecutor->Prepare(reduceAttr));
            HCCL_INFO("ReduceScatter mesh: using nhr_v1 algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            innerExecutor
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
            CHK_SMART_PTR_NULL(innerExecutor);
            CHK_RET(innerExecutor->Prepare(reduceAttr));
            HCCL_INFO("ReduceScatter mesh: using nonuniform-bruck algo inter-server.");
        } else {
            count = count * innerRankSize;
            baseOffset = commIndex * innerRankSize * perRankSize;
            inputMem = execMem.inputMem.range(commIndex * innerRankSize * perRankSize, perRankSize * innerRankSize);
            scratchMem = execMem.scratchMem.range(commIndex * innerRankSize * perRankSize, perRankSize * innerRankSize);
            innerExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RECURSIVE_HD, dispatcher_);
            CHK_SMART_PTR_NULL(innerExecutor);
            CHK_RET(innerExecutor->Prepare(reduceAttr));
            HCCL_INFO("ReduceScatter mesh: using halving-doubling algo inter-server.");
        }
        CHK_RET(innerExecutor->Prepare(
            inputMem, inputMem, scratchMem, count, param.DataDes.dataType, param.stream, param.reduceType,
            LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, baseOffset));
        CHK_RET(innerExecutor->RegisterProfiler(
            (innerRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + innerCommInfo.localRank, PROF_STAGE_0,
            HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(innerExecutor, innerCommInfo));
        HCCL_INFO("[CollReduceScatterAivRdmaExecutor] rdma stage run success.");
    }
    /*  第三步 最后D2D拷贝 */

    // 如果使用CCL buffer，需要将CCL buffer in中的结果拷贝到user buffer out
    DeviceMem srcMem
        = execMem.inputMem.range(perRankSize * (commIndex * serverNum + innerCommInfo.localRank), perRankSize);
    DeviceMem dstMem = DeviceMem::create(execMem.outputPtr, perRankSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));

    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));

    HCCL_INFO("[CollReduceScatterAivRdmaExecutor][KernelRun]ReduceScatter aiv run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterAivRdmaExecutor", ReduceScatterAivRdma, CollReduceScatterAivRdmaExecutor);

} // namespace hccl
