/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "Coll_reduce_scatter_v_mesh_opbase_pipeline_executor.h"

namespace hccl {

CollReduceScatterVMeshOpbasePipelineExecutor::CollReduceScatterVMeshOpbasePipelineExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterVExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
    CCLMemSlice_ = false;
}

void CollReduceScatterVMeshOpbasePipelineExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
}

HcclResult CollReduceScatterVMeshOpbasePipelineExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation + 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO(
        "[CollReduceScatterVMeshOpbasePipelineExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbasePipelineExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbasePipelineExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaInfo.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

// PipeLine模式下使用Ring算法
HcclResult CollReduceScatterVMeshOpbasePipelineExecutor::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL1, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL1], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbasePipelineExecutor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;
    HCCL_INFO(
        "[CollReduceScatterVMeshOpbasePipelineExecutor][CalcTransportMemType] tag[%s] inputType[%d],"
        " outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

bool CollReduceScatterVMeshOpbasePipelineExecutor::IsHugeData(const u64 curSize, [[maybe_unused]] const OpParam& param)
{
    bool hugeData = curSize > RDMA_SEND_MAX_SIZE || curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

HcclResult CollReduceScatterVMeshOpbasePipelineExecutor::Getlevel1CommRank(SubCommInfo& level1CommInfo)
{
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u32 commIndex = level0CommInfo.localRank; // 找到rank所在的节点间平面
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));

    level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbasePipelineExecutor::SelectTempAlg(
    std::unique_ptr<AlgTemplateBase>& level1TempAlg, u32 level1RankSize)
{
    if (level1RankSize > 1) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCESCATTER_PIPELINE, dispatcher_);
        CHK_SMART_PTR_NULL(level1TempAlg);
    }
    return HCCL_SUCCESS;
}

u64 CollReduceScatterVMeshOpbasePipelineExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，放开ranksize限制
    u64 maxCountPerLoop = ((inCCLbufferSize_ / (HCCL_MIN_SLICE_ALIGN_910B * PIPELINE_DEPTH)) * HCCL_MIN_SLICE_ALIGN_910B
                           - HCCL_MIN_SLICE_ALIGN_910B)
                          / unitSize;
    HCCL_INFO(
        "[CollReduceScatterVMeshOpbasePipelineExecutor][CalcLoopMaxCount] maxCountPerLoop[%llu]", maxCountPerLoop);
    return maxCountPerLoop;
}

HcclResult CollReduceScatterVMeshOpbasePipelineExecutor::CalcCurCountsAndCurDispls(
    const u64 maxTotalCount, std::vector<u64>& countsLeft, std::vector<u64>& displs, std::vector<u64>& curCounts,
    std::vector<u64>& curDispls, bool& finished)
{
    finished = true;
    curCounts.resize(countsLeft.size(), 0);
    curDispls.resize(displs.size(), 0);

    // 先设置本轮的displacements，等于入参displs
    std::copy(displs.begin(), displs.end(), curDispls.begin());
    // 分配好每个rank的counts
    for (auto i = 0U; i < countsLeft.size(); ++i) {
        const auto curCount = countsLeft[i] < maxTotalCount ? countsLeft[i] : maxTotalCount;
        curCounts[i] = curCount;
        countsLeft[i] -= curCount;
        displs[i] += curCount;

        if (countsLeft[i] != 0) {
            finished = false;
        }
    }
    HCCL_INFO("[%s] Calc CurCountsAndCurDispls finish.", __func__);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbasePipelineExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterVMeshOpbasePipelineExecutor] reducescatterv pipeline run");
    HcclDataType dataType = param.VDataDes.dataType;
    const u32 unitSize = SIZE_TABLE[dataType];

    // 先获取 comm level0 \ comm level1 的value
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    std::vector<Slice> inputSlices;
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    const auto displs = static_cast<u64*>(param.VDataDes.displs);
    for (u32 rank = 0; rank < topoAttr_.userRankSize; ++rank) {
        Slice userslice;
        userslice.offset = displs[rank] * unitSize;
        userslice.size = counts[rank] * unitSize;
        inputSlices.emplace_back(std::move(userslice));
    }

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.inputMem, dataType, param.reduceType);

    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_V_PIPELINE, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);

    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, 0, dataType, param.root, param.reduceType, 0};

    Stream stream = param.stream;
    CHK_RET(tempAlg->Prepare(
        &opInfo, execMem.inputMem, execMem.inputMem.size(), inputSlices, level0CommInfo, level1CommInfo, stream,
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, reduceAttr));
    CHK_RET(tempAlg->RunAsync());

    return HCCL_SUCCESS;
}

REGISTER_EXEC(
    "ReduceScatterVMeshOpbasePipelineExecutor", ReduceScatterVMeshOpbasePipeline,
    CollReduceScatterVMeshOpbasePipelineExecutor);

} // namespace hccl
