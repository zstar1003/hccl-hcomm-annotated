/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_mesh_dma_elimination.h"

namespace hccl {

CollReduceScatterMeshDmaEliminationExecutor::CollReduceScatterMeshDmaEliminationExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
    CCLMemSlice_ = false;
}

void CollReduceScatterMeshDmaEliminationExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
}

HcclResult CollReduceScatterMeshDmaEliminationExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO(
        "[CollReduceScatterMeshDmaEliminationExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshDmaEliminationExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshDmaEliminationExecutor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollReduceScatterMeshDmaEliminationExecutor][CalcTransportMemType] tag[%s] inputType[%d],"
        " outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshDmaEliminationExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollReduceScatterMeshDmaEliminationExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，放开ranksize限制
    u64 maxCountPerLoop = inCCLbufferSize_ / unitSize;
    return maxCountPerLoop;
}

bool CollReduceScatterMeshDmaEliminationExecutor::IsHugeData(const u64 curSize, [[maybe_unused]] OpParam* param)
{
    // 只有server内通信，多QP哈希散列下不刷新子图
    bool hugeData = curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

bool CollReduceScatterMeshDmaEliminationExecutor::IsSmallData(const u64 totalSize, [[maybe_unused]] const u64 curSize)
{
    bool smallData = totalSize <= HCCL_SMALL_COUNT_32_KB;
    return smallData;
}

HcclResult CollReduceScatterMeshDmaEliminationExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[CollReduceScatterMeshDmaEliminationExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u32 commIndex = level0CommInfo.localRank; // 找到rank所在的节点间平面

    /* *******************节点内reducescatter ******************************************/
    CHK_RET(ActiveSlaveStreams(param.stream));

    u32 sliceNum = level0CommInfo.localRankSize;
    // 根据数据量算每个环上数据的偏移和大小，把做完hd的slice均分成RankSize份
    std::vector<Slice> dataSegsSlice;

    HCCL_DEBUG(
        "inputMem.size()=%llu, level0CommInfo.localRankSize=%u, commIndex=%u", execMem.inputMem.size(),
        level0CommInfo.localRankSize, commIndex);

    HcomCollOpInfo* opInfoPtr = nullptr;
    HcomCollOpInfo opInfo = {"",
                             execMem.inputPtr,
                             execMem.outputPtr,
                             param.DataDes.count,
                             param.DataDes.dataType,
                             param.root,
                             param.reduceType,
                             0};
    if (DMAReduceFlag_) {
        opInfoPtr = &opInfo;
    }

    if (topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_DISABLE
        && (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)
        && (topoAttr_.deviceType == DevType::DEV_TYPE_910B && param.reduceType != HCCL_REDUCE_PROD)) {
        CHK_RET(MultiStreamReduceScatterMeshAtomic(
            param.tag, execMem.inputMem, execMem.scratchMem, execMem.count, param.DataDes.dataType, param.reduceType,
            dataSegsSlice, const_cast<Stream&>(param.stream), COMM_LEVEL0, 0, opInfoPtr));
    } else {
        std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
        // mesh算法stream数量为rank数减1
        CHK_RET(AlgTemplateBase::PrepareSliceMeshStreams(dataSegsSlice, sliceNum - 1, multiStreamSlice));
        CHK_RET(MultiStreamReduceScatterMesh(
            param.tag, execMem.inputMem, execMem.scratchMem, execMem.count, param.DataDes.dataType, param.reduceType,
            multiStreamSlice, param.stream, COMM_LEVEL0, 0));
    }

    return HCCL_SUCCESS;
}

REGISTER_EXEC(
    "ReduceScatterMeshDmaEliminationExecutor", ReduceScatterMeshDmaElimination,
    CollReduceScatterMeshDmaEliminationExecutor);
} // namespace hccl
