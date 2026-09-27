/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_v_2level_pipeline_excecutor.h"
namespace hccl {

CollRunAlltoAllVTwoLevelPipeline::CollRunAlltoAllVTwoLevelPipeline(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{}

// 计算 alltoall pipeline 910B 的两级流水算法本卡需要的 scratch 大小(图模式需要)
u64 CollRunAlltoAllVTwoLevelPipeline::GetAlltoall2LevelPipelineScratchSize910B(
    u32 rank, std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo)
{
    u64 userRankSize = allMeshAggregationSendRecvInfo.size();
    u64 maxBlockSize = 0;
    u64 maxScratchSize = 0;
    const SendRecvInfo& info = allMeshAggregationSendRecvInfo[rank];
    for (u64 i = 0; i < userRankSize; i++) {
        maxBlockSize = std::max(maxBlockSize, info.sendLength[i]);
        maxBlockSize = std::max(maxBlockSize, info.recvLength[i]);
        maxScratchSize = std::max(maxScratchSize, info.sendOffset[i] + info.sendLength[i]);
        maxScratchSize = std::max(maxScratchSize, info.recvOffset[i] + info.recvLength[i]);
    }
    maxScratchSize = std::max(maxBlockSize * userRankSize, maxScratchSize);
    return maxScratchSize;
}

// 计算 alltoall pipeline 910B 的两级流水算法所有卡需要的 scratch 大小的最大值(单算子模式需要)
u64 CollRunAlltoAllVTwoLevelPipeline::GetAlltoall2LevelPipelineMaxScratchSize910B(
    std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo)
{
    u64 maxScratchSize = 0;
    for (u32 rank = 0, userRankSize = allMeshAggregationSendRecvInfo.size(); rank < userRankSize; rank++) {
        u64 currRankScratchSize = GetAlltoall2LevelPipelineScratchSize910B(rank, allMeshAggregationSendRecvInfo);
        maxScratchSize = (currRankScratchSize > maxScratchSize ? currRankScratchSize : maxScratchSize);
    }
    return maxScratchSize;
}

HcclResult CollRunAlltoAllVTwoLevelPipeline::CalcScratchMemSize(u64& scratchMemSize)
{
    scratchMemSize = 0U;
    u64 tmpMemSize = 0U;
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB
        && topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        // 图模式才需要申请 scratch 在此只计算scratchMem size
        tmpMemSize = GetAlltoall2LevelPipelineMaxScratchSize910B(allMeshAggregationSendRecvInfo_);
    }
    scratchMemSize = CalAlltoAllVScratchMemSize(tmpMemSize);
    HCCL_INFO(
        "[CollRunAlltoAllVTwoLevelPipeline][CalcScratchMemSize] tag_[%s] scratchMemSize[%llu]", tag_.c_str(),
        scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVTwoLevelPipeline::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation + 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollRunAlltoAllVTwoLevelPipeline][CalcStreamNum] tag_[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVTwoLevelPipeline::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_MESH_L0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_MESH_L0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVTwoLevelPipeline::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel1(COMM_MESH_L1, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel1, opTransport[COMM_MESH_L1], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVTwoLevelPipeline::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;

    CHK_RET(CalNoScratchAlltoallCommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVTwoLevelPipeline::CalNoScratchAlltoallCommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    (void)inputType;
    (void)outputType;
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        || topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        CHK_RET(CalcLevel0CommInfo(TransportMemType::CCL_OUTPUT, TransportMemType::CCL_OUTPUT, opTransport));
        CHK_RET(CalcLevel1CommInfo(TransportMemType::CCL_INPUT, TransportMemType::CCL_OUTPUT, opTransport));
    } else {
        CHK_RET(CalcLevel0CommInfo(TransportMemType::SCRATCH, TransportMemType::CCL_OUTPUT, opTransport));
        CHK_RET(CalcLevel1CommInfo(TransportMemType::CCL_INPUT, TransportMemType::SCRATCH, opTransport));
    }

    return HCCL_SUCCESS;
}

HcclOpMetaInfo CollRunAlltoAllVTwoLevelPipeline::GetOpMeta(HcclCMDType opType, const u64 size)
{
    (void)opType;
    (void)size;
    bool hugeData = (isAlltoAllZCopyMode_) ? (algResResp_->paramInputMem.size() > SDMA_SEND_MAX_SIZE) : (false);
    bool alltoallPingPong
        = ((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
            || topoAttr_.deviceType == DevType::DEV_TYPE_910_93)
           && !static_cast<bool>(topoAttr_.multiModuleDiffDeviceNumMode)
           && GetAlltoall2LevelPipelineMaxScratchSize910B(allMeshAggregationSendRecvInfo_)
                  > algResResp_->cclInputMem.size());
    HcclOpMetaInfo opMeta;
    if (AlltoAllVParam_.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        opMeta = HcclOpMetaInfo::GetOneForAllToAllV(
            (isAlltoAllZCopyMode_ ? CopyPattern::ZCOPY : CopyPattern::BCOPY), algResResp_->paramInputMem.size(),
            hugeData || alltoallPingPong);
    } else {
        opMeta = HcclOpMetaInfo::GetOneForAllToAllV(
            (isAlltoAllZCopyMode_ ? CopyPattern::ZCOPY : CopyPattern::BCOPY), algResResp_->paramInputMem.size(),
            hugeData || alltoallPingPong);
    }
    HCCL_DEBUG("[CollRunAlltoAllVTwoLevelPipeline][GetOpMeta] Get OpMeta for AllToAllV pipeline success.");
    return opMeta;
}

HcclResult CollRunAlltoAllVTwoLevelPipeline::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollRunAlltoAllVTwoLevelPipeline][KernelRun] AllToAllV two level pipeline start");

    bool cclEnough = true;
    if ((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
         || topoAttr_.deviceType == DevType::DEV_TYPE_910_93)
        && GetAlltoall2LevelPipelineMaxScratchSize910B(allMeshAggregationSendRecvInfo_) > execMem.inputMem.size()) {
        cclEnough = false;
    }
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[CollRunAlltoAllVTwoLevelPipeline][KernelRun] AllToAllV pipeline run %s algo",
        cclEnough ? "cclEnough" : "ping pong");
    A2aPipelineMemory a2aPipelineMemory;
    a2aPipelineMemory.userInput = algResResp_->paramInputMem;
    a2aPipelineMemory.userOutput = algResResp_->paramOutputMem;
    // 具体传入 A2aPipelineMemory 对象的 alltoall pipeline executor 会根据图模式还是单算子模式
    // 选择使用 ccl 还是 scratch，不会访问空指针
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        || topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        a2aPipelineMemory.cclInBuffer = execMem.inputMem;
        a2aPipelineMemory.cclOutBuffer = execMem.outputMem;
    } else {
        a2aPipelineMemory.scratchMem = execMem.scratchMem;
    }
    std::unique_ptr<AlgTemplateBase> alltoallPipe = nullptr;
    if (cclEnough) {
        alltoallPipe = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_2_ALL_PIPELINE_MESH_PAIRWISE_CCL_ENOUGH, dispatcher_);
    } else {
        alltoallPipe = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_2_ALL_PIPELINE_MESH_PAIRWISE_PING_PONG, dispatcher_);
    }

    CHK_RET(CheckCommSize(COMM_MESH_L0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_MESH_L0, COMM_INDEX_0);
    CHK_RET(CheckCommSize(COMM_MESH_L1, COMM_INDEX_0 + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_MESH_L1, COMM_INDEX_0);

    CHK_SMART_PTR_NULL(alltoallPipe);
    CHK_RET(alltoallPipe->Prepare(
        topoAttr_.userRank, a2aPipelineMemory, level0CommInfo, level1CommInfo, const_cast<Stream&>(param.stream),
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, allMeshAggregationSendRecvInfo_,
        workflowMode_));
    CHK_RET(alltoallPipe->RunAsync());
    HCCL_INFO("[CollRunAlltoAllVTwoLevelPipeline][kernelRun] AllToAllV two level pipeline exec end");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("RunAlltoAllVTwoLevelPipeline", AlltoAllVTwoLevelPipeline, CollRunAlltoAllVTwoLevelPipeline);
} // namespace hccl
