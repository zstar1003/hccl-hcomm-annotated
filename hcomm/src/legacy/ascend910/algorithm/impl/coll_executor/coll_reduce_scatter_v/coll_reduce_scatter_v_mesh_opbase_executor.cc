/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_v_mesh_opbase_executor.h"
#include <numeric>
namespace hccl {

CollReduceScatterVMeshOpbaseExecutor::CollReduceScatterVMeshOpbaseExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterVExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
    CCLMemSlice_ = false;
}

void CollReduceScatterVMeshOpbaseExecutor::ParseParam(const OpParam& param)
{
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
    DMAReduceFlag_ = topoAttr_.moduleNum > 1 ? false : true;
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollReduceScatterVMeshOpbaseExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::CalcTransportMemType(
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
        "[CollReduceScatterVMeshOpbaseExecutor][CalcTransportMemType] tag[%s] inputType[%d],"
        " outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

bool CollReduceScatterVMeshOpbaseExecutor::IsHugeData(const u64 curSize, const OpParam& param)
{
    const auto* countsPtr = static_cast<const u64*>(param.VDataDes.counts);
    u64 totalCounts = std::accumulate(countsPtr, countsPtr + topoAttr_.userRankSize, 0ULL);
    return (totalCounts * SIZE_TABLE[param.VDataDes.dataType] > RDMA_SEND_MAX_SIZE) || (curSize > SDMA_SEND_MAX_SIZE);
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::CalcCurCountsAndCurDisplsSingleModule(
    const u64 maxTotalCount, std::vector<u64>& countsLeft, std::vector<u64>& displs, std::vector<u64>& curCounts,
    std::vector<u64>& curDispls, bool& finished) const
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
    HCCL_INFO("[%s] Calc CurCountsAndCurDispls for SingleModule finish.", __func__);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::CalcCurCountsAndCurDisplsMultiModule(
    const u64 maxTotalCount, std::vector<u64>& countsLeft, std::vector<u64>& displs, std::vector<u64>& curCounts,
    std::vector<u64>& curDispls, bool& finished) const
{
    curCounts = std::vector<u64>(countsLeft.size(), 0);
    curDispls = std::vector<u64>(displs.size(), 0);
    auto allocatableCount = maxTotalCount;

    // 先设置本轮的displacements，等于入参displs
    std::copy(displs.begin(), displs.end(), curDispls.begin());

    // 分配本轮的counts，如果CCLbuffer空间还没完全利用，则再进行分配
    while (allocatableCount > 0) {
        // 计算现在还有几个rank还有数据需要去通信(countsLeft不为0)
        const auto nonZeroCount = std::count_if(countsLeft.begin(), countsLeft.end(), [](const u64 count) {
            return count != 0;
        });
        if (nonZeroCount == 0) {
            finished = true;
            HCCL_INFO("[%s] Calc CurCountsAndCurDispls for multiModule finish.", __func__);
            return HCCL_SUCCESS;
        }
        // 计算每个rank可以分到多少count
        const auto perRankCount = allocatableCount / nonZeroCount;
        if (perRankCount == 0) {
            break;
        }
        HCCL_DEBUG("[CollReduceScatterVMeshOpbaseExecutor]Calc for perRankCount start");
        for (auto i = 0U; i < countsLeft.size(); ++i) {
            const auto curCount = countsLeft[i] < perRankCount ? countsLeft[i] : perRankCount;
            allocatableCount -= curCount;
            curCounts[i] += curCount;
            countsLeft[i] -= curCount;
            displs[i] += curCount;
        }
    }
    // 特殊情况下，allocatableCount 刚好使用完毕时，不仅如此while循环，导致RunLoop额外循环一次
    const auto nonZeroCount = std::count_if(countsLeft.begin(), countsLeft.end(), [](const u64 count) {
        return count != 0;
    });
    if (nonZeroCount == 0) {
        finished = true;
    }
    HCCL_INFO("[%s] Calc CurCountsAndCurDispls for multiModule finish.", __func__);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::CalcCurCountsAndCurDispls(
    const u64 maxTotalCount, std::vector<u64>& countsLeft, std::vector<u64>& displs, std::vector<u64>& curCounts,
    std::vector<u64>& curDispls, bool& finished)
{
    if (topoAttr_.moduleNum > 1) {
        CHK_RET(
            CalcCurCountsAndCurDisplsMultiModule(maxTotalCount, countsLeft, displs, curCounts, curDispls, finished));
    } else {
        CHK_RET(
            CalcCurCountsAndCurDisplsSingleModule(maxTotalCount, countsLeft, displs, curCounts, curDispls, finished));
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::RunReduceScattervLevel0SingleModule(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level0CommInfo)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterVMeshOpbaseExecutor] Run ReduceScatterV Level0 SingleModule ");
    HcclDataType dataType = param.VDataDes.dataType;
    const u32 unitSize = SIZE_TABLE[dataType];
    u32 level0RankSize = level0CommInfo.localRankSize;

    /* *******************节点内reducescatter ******************************************/
    // reduce_scatter_v 计算slice，数据分成ranksize份，每份的起始偏移和大小
    std::vector<Slice> inputSlices;
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    const auto displs = static_cast<u64*>(param.VDataDes.displs);
    for (u32 rankId = 0; rankId < level0RankSize; ++rankId) {
        Slice userslice;
        userslice.offset = displs[rankId] * unitSize;
        userslice.size = counts[rankId] * unitSize;
        inputSlices.emplace_back(std::move(userslice));
    }

    HcomCollOpInfo* opInfoPtr = nullptr;
    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, 0, dataType, param.root, param.reduceType, 0};
    if (DMAReduceFlag_) {
        opInfoPtr = &opInfo;
    }

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, dataType, param.reduceType);
    std::unique_ptr<AlgTemplateBase> TempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_MESH_DIRECT, dispatcher_);
    CHK_SMART_PTR_NULL(TempAlg);

    CHK_RET(TempAlg->Prepare(
        execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count, dataType, param.stream, param.reduceType,
        LEVEL0_BRIDGE_RANK_ID, inputSlices, 0, reduceAttr, algResResp_->slaveStreams, algResResp_->notifiesMain,
        algResResp_->notifiesAux, topoAttr_.userRank, opInfoPtr));

    CHK_RET(TempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_0,
        HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(TempAlg, level0CommInfo));

    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::RunReduceScattervLevel0(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level0CommInfo)
{
    HcclDataType dataType = param.VDataDes.dataType;
    const u32 unitSize = SIZE_TABLE[dataType];
    u32 level0rankSize = level0CommInfo.localRankSize;
    u32 commIndex = level0CommInfo.localRank; // 找到rank所在的节点间平面
    /* *******************节点内reducescatter ******************************************/

    std::vector<Slice> inputSlices;
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    u64 offset = 0;

    for (u32 moduleId = 0; moduleId < topoAttr_.moduleNum; moduleId++) {
        for (u32 rankId = 0; rankId < level0rankSize; ++rankId) {
            if (topoAttr_.userRank / level0rankSize == moduleId) {
                Slice userslice;
                userslice.size = counts[rankId + moduleId * level0rankSize] * unitSize;
                userslice.offset = offset * unitSize;
                inputSlices.emplace_back(std::move(userslice));
            }
            offset += counts[rankId + moduleId * level0rankSize];
        }
    }

    HcomCollOpInfo* opInfoPtr = nullptr;
    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, 0, dataType, param.root, param.reduceType, 0};
    if (DMAReduceFlag_) {
        opInfoPtr = &opInfo;
    }

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, dataType, param.reduceType);
    std::unique_ptr<AlgTemplateBase> TempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_MESH_ATOMIC, dispatcher_);
    CHK_SMART_PTR_NULL(TempAlg);

    CHK_RET(TempAlg->Prepare(
        execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count, dataType, param.stream, param.reduceType,
        LEVEL0_BRIDGE_RANK_ID, inputSlices, 0, reduceAttr, algResResp_->slaveStreams, algResResp_->notifiesMain,
        algResResp_->notifiesAux, topoAttr_.userRank, opInfoPtr));

    CHK_RET(TempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_0,
        HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(TempAlg, level0CommInfo));

    // 机间reduceScatter 结果 搬运到 cclout
    DeviceMem srcMem = execMem.inputMem.range(inputSlices[commIndex].offset, inputSlices[commIndex].size);
    CHK_SMART_PTR_NULL(srcMem);
    Stream stream = param.stream;
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, stream));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::CalReduceScatterVSliceData(
    const OpParam& param, u32 level0RankSize, u32 level1RankSize, std::vector<Slice>& dataSlices) const
{
    HcclDataType dataType = param.VDataDes.dataType;
    u32 unitSize = SIZE_TABLE[dataType];
    std::vector<Slice> slices;
    const auto curCounts = static_cast<u64*>(param.VDataDes.counts);
    u64 offset = 0;
    for (u32 moduleId = 0; moduleId < level1RankSize; moduleId++) {
        u64 size = 0;
        for (u32 rankid = 0; rankid < level0RankSize; rankid++) {
            size += curCounts[rankid + moduleId * level0RankSize];
        }
        Slice slice;
        slice.size = size * unitSize;
        slice.offset = offset * unitSize;
        slices.emplace_back(std::move(slice));
        offset += size;
    }
    dataSlices = std::move(slices);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::RunReduceScattervLevel1(
    const OpParam& param, ExecMem& execMem, const SubCommInfo& level0CommInfo)
{
    u32 commIndex = level0CommInfo.localRank; // 找到rank所在的节点间平面
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    HcclDataType dataType = param.VDataDes.dataType;

    u32 level0RankSize = level0CommInfo.localRankSize;
    u32 level1RankSize = level1CommInfo.localRankSize;
    /* ******************第一步: 机间reducescatter *******************************/
    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, dataType, param.reduceType);
    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
        HCCL_INFO("reducescatterv mesh: using ring algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
        HCCL_INFO("reducescatterv mesh: using nonuniform-bruck algo inter-server.");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
    } else {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
        HCCL_INFO("reducescatterv mesh: using nhr algo inter-server.");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr, false));
        level1TempAlg->CloseBarrier();
    }

    std::vector<Slice> slices;
    CalReduceScatterVSliceData(param, level0RankSize, level1RankSize, slices);

    CHK_RET(level1TempAlg->Prepare(
        execMem.inputMem, execMem.inputMem, execMem.scratchMem, 0, dataType, param.stream, param.reduceType,
        LEVEL0_BRIDGE_RANK_ID, slices));

    CHK_RET(level1TempAlg->RegisterProfiler(
        (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_0,
        HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterVMeshOpbaseExecutor] reducescatterv mesh run");
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    if (topoAttr_.moduleNum > 1) {
        CHK_RET(RunReduceScattervLevel1(param, execMem, level0CommInfo));
        CHK_RET(RunReduceScattervLevel0(param, execMem, level0CommInfo));
    } else {
        CHK_RET(RunReduceScattervLevel0SingleModule(param, execMem, level0CommInfo));
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterVMeshOpbaseExecutor", ReduceScatterVMeshOpbase, CollReduceScatterVMeshOpbaseExecutor);
} // namespace hccl
