/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_v_mesh_executor.h"
#include "alg_template_register.h"

namespace hccl {

CollReduceScatterVMeshExecutor::CollReduceScatterVMeshExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterVExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

void CollReduceScatterVMeshExecutor::ParseParam(const OpParam& param)
{
    // 910B 图模式非确定计算，inlineReduce使能，MESH拓扑场景下，创建一个mesh平面
    bool isInlineReduce
        = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.VDataDes.dataType, param.reduceType);
    meshSinglePlane_ = (topoAttr_.deviceType == DevType::DEV_TYPE_910B)
                       && topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_DISABLE && isInlineReduce
                       && (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);

    // 记录图模式总数据量
    u64 totalSize = 0;
    const u64* counts = static_cast<const u64*>(param.VDataDes.counts);
    for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
        totalSize += counts[i] * SIZE_TABLE[param.VDataDes.dataType];
    }
    totalSize_ = totalSize;
    scratchMemFlag_ = false; // mesh算法不需要使用scratchMem
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
}

HcclResult CollReduceScatterVMeshExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation > 1U ? topoAttr_.deviceNumPerAggregation - 1U : 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollReduceScatterVMeshExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult
CollReduceScatterVMeshExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType = TransportMemType::PARAM_INPUT;
    outputType = TransportMemType::PARAM_OUTPUT;

    HCCL_INFO(
        "[CollReduceScatterVMeshExecutor][CalcTransportMemType] tag[%s] inputType[%d],"
        " outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = meshSinglePlane_;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterVMeshExecutor] ReduceScatterV mesh run");
    HcclDataType dataType = param.VDataDes.dataType;
    const u32 unitSize = SIZE_TABLE[dataType];

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo subCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 rankSize = subCommInfo.localRankSize;

    CHK_RET(ActiveSlaveStreams(param.stream));

    /* *******************节点内reducescatter ******************************************/
    // reduce_scatter_v 计算slice，数据分成ranksize份，每份的起始偏移和大小
    std::vector<Slice> inputSlices;
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    const auto displs = static_cast<u64*>(param.VDataDes.displs);
    for (u32 rank = 0; rank < rankSize; ++rank) {
        Slice userslice;
        userslice.offset = displs[rank] * unitSize;
        userslice.size = counts[rank] * unitSize;
        inputSlices.emplace_back(std::move(userslice));
    }

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, dataType, param.reduceType);
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_MESH_ATOMIC, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);

    CHK_RET(tempAlg->Prepare(
        execMem.inputMem, execMem.outputMem, execMem.scratchMem, execMem.count, dataType, param.stream,
        param.reduceType, LEVEL0_BRIDGE_RANK_ID, inputSlices, 0, reduceAttr, algResResp_->slaveStreams,
        algResResp_->notifiesMain, algResResp_->notifiesAux, topoAttr_.userRank));

    CHK_RET(tempAlg->RegisterProfiler(
        (subCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + subCommInfo.localRank, PROF_STAGE_0,
        HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(tempAlg, subCommInfo));

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterVMeshExecutor", ReduceScatterVMesh, CollReduceScatterVMeshExecutor);
} // namespace hccl
