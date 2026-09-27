/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_mesh_graph_executor.h"

namespace hccl {

CollReduceScatterMeshGraphExecutor::CollReduceScatterMeshGraphExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

void CollReduceScatterMeshGraphExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;

    // 910B 图模式非确定计算，inlineReduce使能，MESH拓扑场景下，创建一个mesh平面
    bool isInlineReduce
        = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    meshSinglePlane_ = (topoAttr_.deviceType == DevType::DEV_TYPE_910B)
                       && topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_DISABLE && isInlineReduce
                       && (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    // 是否需要scratch memory
    scratchMemFlag_ = true;

    // 记录图模式总数据量
    totalSize_ = topoAttr_.userRankSize * param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
}

HcclResult CollReduceScatterMeshGraphExecutor::CalcScratchMemSize(u64& scratchMemSize)
{
    if (scratchMemFlag_) {
        scratchMemSize = totalSize_;
    } else {
        scratchMemSize = 0U;
    }
    HCCL_INFO(
        "[CollReduceScatterMeshGraphExecutor][CalcScratchMemSize] tag[%s] scratchMemSize[%llu]", tag_.c_str(),
        scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshGraphExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation > 1U ? topoAttr_.deviceNumPerAggregation - 1U : 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollReduceScatterMeshGraphExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshGraphExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshGraphExecutor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType = TransportMemType::SCRATCH;
    outputType = TransportMemType::PARAM_INPUT;

    HCCL_INFO(
        "[CollReduceScatterMeshGraphExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshGraphExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = meshSinglePlane_;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

bool CollReduceScatterMeshGraphExecutor::IsHugeData(const u64 curSize, [[maybe_unused]] OpParam* param)
{
    bool hugeData = (curSize * topoAttr_.userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE)
                    || (curSize > SDMA_SEND_MAX_SIZE);
    return hugeData;
}

HcclResult CollReduceScatterMeshGraphExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[CollReduceScatterMeshGraphExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);

    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];
    u64 singleRankDataSize = execMem.count * perDataSize;

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank; // 找到rank所在的节点间平面
    u32 level0RankSize = level0CommInfo.localRankSize;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    u32 serverIndex = level1CommInfo.localRank;
    u32 level1RankSize = level1CommInfo.localRankSize;
    CHK_RET(ActiveSlaveStreams(param.stream));

    /* ******************第1步: input to scratch *******************************/
    HCCL_INFO(
        "[CollReduceScatterMeshGraphExecutor][KernelRun] userRank[%u], level0RankSize[%u], level1RankSize[%u]",
        topoAttr_.userRank, level0RankSize, level1RankSize);
    for (u32 inputSliceId = 0; inputSliceId < topoAttr_.userRankSize; inputSliceId++) {
        u32 dstServerId = inputSliceId / topoAttr_.deviceNumPerAggregation;
        u32 dstLocalRank = inputSliceId % topoAttr_.deviceNumPerAggregation;
        u32 dstSliceId = dstLocalRank * topoAttr_.moduleNum + dstServerId;

        u64 srcInputOffset = inputSliceId * singleRankDataSize;
        u64 dstScratchOffset = dstSliceId * singleRankDataSize;

        DeviceMem srcInputMem = execMem.inputMem.range(srcInputOffset, singleRankDataSize);
        CHK_SMART_PTR_NULL(srcInputMem);
        DeviceMem dstScratchMem = execMem.scratchMem.range(dstScratchOffset, singleRankDataSize);
        CHK_SMART_PTR_NULL(dstScratchMem);

        HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, dstScratchMem, srcInputMem, const_cast<Stream&>(param.stream));
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[CollReduceScatterMeshGraphExecutor][KernelRun] rank[%u] slice[%u] to slice[%u] failed",
                topoAttr_.userRank, inputSliceId, dstSliceId),
            ret);
    }

    /* ******************第2步: intranode *******************************/
    u32 sliceNum = level0CommInfo.localRankSize;
    // 根据数据量算每个环上数据的偏移和大小，把做完hd的slice均分成RankSize份
    std::vector<Slice> dataSegsSlice;
    u32 level0ReduceCount = execMem.count * level1RankSize;
    CHK_RET(PrepareReduceScatterSliceData(level0ReduceCount, perDataSize, sliceNum, dataSegsSlice));

    // 每个server分配的slice大小
    u64 serverSliceSize = execMem.inputMem.size();
    // 每个服务器对应的偏移
    u64 serverSliceOffset = 0;

    HCCL_DEBUG(
        "inputMem.size=%llu, level0CommInfo.localRankSize=%u, serverSliceSize=%llu, serverSliceOffset=%llu "
        "commIndex=%u level1CommInfo.localRank=%u",
        execMem.inputMem.size(), level0CommInfo.localRankSize, serverSliceSize, serverSliceOffset, commIndex,
        level1CommInfo.localRank);

    DeviceMem reduceScatterMeshInput = execMem.scratchMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(reduceScatterMeshInput);
    DeviceMem reduceScatterMeshOutput = execMem.inputMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(reduceScatterMeshOutput);

    HcomCollOpInfo* opInfoPtr = nullptr;

    if (topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_DISABLE
        && (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)
        && (topoAttr_.deviceType == DevType::DEV_TYPE_910B && param.reduceType != HCCL_REDUCE_PROD)) {
        CHK_RET(MultiStreamReduceScatterMeshAtomic(
            param.tag, reduceScatterMeshInput, reduceScatterMeshOutput, // 非确定性
            level0ReduceCount, param.DataDes.dataType, param.reduceType, dataSegsSlice,
            const_cast<Stream&>(param.stream), COMM_LEVEL0, serverSliceOffset, opInfoPtr));
    } else {
        std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
        // mesh算法stream数量为rank数减1
        CHK_RET(AlgTemplateBase::PrepareSliceMeshStreams(dataSegsSlice, sliceNum - 1, multiStreamSlice));
        CHK_RET(MultiStreamReduceScatterMesh(
            param.tag, reduceScatterMeshInput, reduceScatterMeshOutput, // 确定性
            level0ReduceCount, param.DataDes.dataType, param.reduceType, multiStreamSlice,
            const_cast<Stream&>(param.stream), COMM_LEVEL0, serverSliceOffset));
    }

    /* ******************第3步: internode *******************************/

    if (level1RankSize > 1) {
        u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<AlgTemplateBase> level1TempAlg;
        u64 ringCount = execMem.count;
        u64 level1SliceSize = execMem.inputMem.size() / level0RankSize;
        // 每个服务器对应的偏移
        u64 level1SliceOffset = commIndex * level1SliceSize;
        DeviceMem level1ReduceScatterInput = execMem.scratchMem.range(level1SliceOffset, level1SliceSize);
        CHK_SMART_PTR_NULL(level1ReduceScatterInput);
        DeviceMem level1ReduceScatterScratch = execMem.inputMem.range(level1SliceOffset, level1SliceSize);
        CHK_SMART_PTR_NULL(level1ReduceScatterScratch);
        HCCL_INFO(
            "[CollReduceScatterMeshGraphExecutor][KernelRun] rank[%u] level 1 sliceSize[%llu] sliceOffset[%llu]",
            topoAttr_.userRank, level1SliceSize, level1SliceOffset);
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
            HCCL_INFO("ReduceScatter mesh: using ring algo inter-server.");
            CHK_RET(level1TempAlg->Prepare(
                level1ReduceScatterInput, level1ReduceScatterInput, level1ReduceScatterScratch, ringCount,
                param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0),
                level1SliceOffset));
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
            HCCL_INFO("ReduceScatter mesh: using nhr algo inter-server.");
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr, false));
            CHK_RET(level1TempAlg->Prepare(
                level1ReduceScatterInput, level1ReduceScatterInput, level1ReduceScatterScratch, ringCount,
                param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0),
                level1SliceOffset));
            level1TempAlg->CloseBarrier();
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_NHR_V1, dispatcher_);
            HCCL_INFO("ReduceScatter mesh: using nhr_v1 algo inter-server.");
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
            CHK_RET(level1TempAlg->Prepare(
                level1ReduceScatterInput, level1ReduceScatterInput, level1ReduceScatterScratch, ringCount,
                param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0),
                level1SliceOffset));
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
            HCCL_INFO("ReduceScatter mesh: using nonuniform-bruck algo inter-server.");
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
            CHK_RET(level1TempAlg->Prepare(
                level1ReduceScatterInput, level1ReduceScatterInput, level1ReduceScatterScratch, ringCount,
                param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0),
                level1SliceOffset));
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RECURSIVE_HD, dispatcher_);
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
            HCCL_INFO(
                "ReduceScatter mesh: algo is [%s] using halving-doubling algo inter-server.",
                (HCCL_ALGO_LEVEL1_MAP.at(algType_.algoLevel1)).c_str());
            u64 inputDataCount = level1SliceSize / perDataSize; // count是output的数据个数
            CHK_RET(level1TempAlg->Prepare(
                level1ReduceScatterInput, level1ReduceScatterInput, level1ReduceScatterScratch, inputDataCount,
                param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0),
                level1SliceOffset));
        }
        CHK_RET(level1TempAlg->RegisterProfiler(
            (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_1,
            HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    }

    /* *******************第4步: 节点内reducescatter ******************************************/
    u32 transposeRankId = commIndex * level1RankSize + serverIndex;
    u64 finalOffset = transposeRankId * singleRankDataSize;
    DeviceMem srcScratchMem = execMem.scratchMem.range(finalOffset, singleRankDataSize);
    CHK_SMART_PTR_NULL(srcScratchMem);
    HcclResult ret
        = HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcScratchMem, const_cast<Stream&>(param.stream));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceScatterMeshGraphExecutor][KernelRun] rank[%u] memcpy failed, offset[%llu], size[%llu]",
            topoAttr_.userRank, finalOffset, singleRankDataSize),
        ret);

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterMeshGraphExecutor", ReduceScatterMeshGraph, CollReduceScatterMeshGraphExecutor);
} // namespace hccl
