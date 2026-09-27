/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_v_mesh_executor.h"

#include <algorithm>
#include <numeric>

namespace hccl {
CollAllGatherVMeshExecutor::CollAllGatherVMeshExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherVExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = (topoAttr_.moduleNum <= 1);
}

HcclResult CollAllGatherVMeshExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0;
    if (topoAttr_.moduleNum > 1) {
        totalStreamNum = topoAttr_.deviceNumPerAggregation > 1U ? topoAttr_.deviceNumPerAggregation - 1U : 1U;
    } else {
        totalStreamNum = topoAttr_.deviceNumPerAggregation;
    }
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollAllGatherVMeshExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherVMeshExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    if (topoAttr_.moduleNum > 1) {
        CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    }

    return HCCL_SUCCESS;
}

HcclResult
CollAllGatherVMeshExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollAllGatherVMeshExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherVMeshExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherVMeshExecutor::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollAllGatherVMeshExecutor][CalcLevel1CommInfo]tag[%s] start", tag_.c_str());
    CommParaInfo commParaLevel1(COMM_LEVEL1, CommType::COMM_TAG_MAX);
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING
        || (topoAttr_.isDiffDeviceModule && topoAttr_.serverNum == 1)) {
        commParaLevel1.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[CollAllGatherVMeshExecutor][CalcLevel1CommInfo]tag[%s] Calc RingCommInfo", tag_.c_str());
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        HCCL_INFO("[CollAllGatherVMeshExecutor][CalcLevel1CommInfo]tag[%s] Calc NBCommInfo", tag_.c_str());
    } else {
        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[CollAllGatherVMeshExecutor][CalcLevel1CommInfo]tag[%s] Calc NHRCommInfo", tag_.c_str());
    }
    commParaLevel1.forceRdma = false;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel1, opTransport[commParaLevel1.commPlane], inputType, outputType));

    HCCL_INFO("[CollAllGatherVMeshExecutor][COMM_LEVEL1]tag[%s] Calc CommInfo Finish", tag_.c_str());

    return HCCL_SUCCESS;
}

u64 CollAllGatherVMeshExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop;
    if (topoAttr_.moduleNum > 1) {
        maxCountPerLoop = cclBuffSize / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / unitSize;
    } else {
        maxCountPerLoop
            = (cclBuffSize - HCCL_MIN_SLICE_ALIGN_910B) / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / unitSize;
    }

    return maxCountPerLoop;
}

bool CollAllGatherVMeshExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize * topoAttr_.userRankSize > RDMA_SEND_MAX_SIZE || curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

HcclResult CollAllGatherVMeshExecutor::RunLevel0(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherVMeshExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    u32 perDataSize = SIZE_TABLE[param.VDataDes.dataType];
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    u32 serverIndex = level1CommInfo.localRank;
    u32 rankBaseOffset = serverIndex * level0CommInfo.localRankSize;
    u64 countBaseOffset = std::accumulate(counts, counts + rankBaseOffset, 0ULL);
    // level0的rank 0在整个通信域中的偏移
    u64 baseOffset = countBaseOffset * perDataSize;
    // allgatherv 计算slice，数据分成ranksize份，每份的起始偏移和大小
    std::vector<Slice> outputSlices;
    u64 outputMemSize = 0;
    for (u32 rank = rankBaseOffset; rank < rankBaseOffset + level0CommInfo.localRankSize; ++rank) {
        Slice userslice;
        userslice.offset = outputMemSize;
        userslice.size = counts[rank] * perDataSize;
        outputSlices.emplace_back(std::move(userslice));
        outputMemSize += userslice.size;
    }

    u64 inputMemSize = outputSlices[level0CommInfo.localRank].size;
    u64 level0Offset = outputSlices[level0CommInfo.localRank].offset;
    DeviceMem srcMem = execMem.inputMem.range(0, inputMemSize);
    DeviceMem dstMem = execMem.outputMem.range(baseOffset + level0Offset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);
    Stream stream = param.stream;
    // 将数据从input内存拷贝到output内存的对应位置
    HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAllGatherVMeshExecutor][KernelRun]all gatherV mesh memcpy Failed, Offset[%llu], Size[%llu].",
            level0Offset, inputMemSize),
        ret);

    CHK_RET(ActiveSlaveStreams(param.stream));

    //  抽取当前用于多环all gather 的output内存数据
    DeviceMem currentOutputMem = execMem.outputMem.range(baseOffset, outputMemSize);
    CHK_SMART_PTR_NULL(currentOutputMem);

    std::unique_ptr<AlgTemplateBase> level0TempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_MESH_ATOMIC, dispatcher_);
    CHK_SMART_PTR_NULL(level0TempAlg);
    CHK_RET(level0TempAlg->Prepare(
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, topoAttr_.userRank, nullptr,
        level0CommInfo.localRank, level0CommInfo.localRankSize));
    CHK_RET(level0TempAlg->Prepare(
        currentOutputMem, currentOutputMem, execMem.inputMem, execMem.count, param.VDataDes.dataType, param.stream,
        HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, outputSlices, baseOffset));
    u32 rankSize = level0CommInfo.localRankSize;
    CHK_RET(level0TempAlg->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
        param.stream));
    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    HCCL_INFO("[CollAllGatherVMeshExecutor][RunLevel0] level 0 for A2 run success");
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherVMeshExecutor::RunLevel1(
    const OpParam& param, ExecMem& execMem, const SubCommInfo& level0CommInfo, SubCommInfo& level1CommInfo) const
{
    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING
        || (topoAttr_.isDiffDeviceModule && topoAttr_.serverNum == 1)) {
        // 1-单server-SDMA
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
        HCCL_INFO("allgatherv mesh: using ring algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
        HCCL_INFO("allgatherv mesh: using nonuniform-bruck algo inter-server.");
    } else {
        // 使用nhr作为兜底算法
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
        level1TempAlg->CloseBarrier();
        HCCL_INFO("allgatherv mesh: using nhr algo inter-server.");
    }
    CHK_SMART_PTR_NULL(level1TempAlg);

    u32 perDataSize = SIZE_TABLE[param.VDataDes.dataType];
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    // allgatherv 计算slice，数据分成level1 ranksize份，每份的起始偏移和大小
    std::vector<Slice> outputSlices;
    u64 outputMemSize = 0;
    for (u32 rankLevel1 = 0; rankLevel1 < level1CommInfo.localRankSize; ++rankLevel1) {
        Slice userslice;
        // 计算偏移值
        u64 countLevel0 = std::accumulate(
            counts + rankLevel1 * level0CommInfo.localRankSize,
            counts + (rankLevel1 + 1) * level0CommInfo.localRankSize, 0ULL);
        userslice.offset = outputMemSize;
        userslice.size = countLevel0 * perDataSize;
        outputSlices.emplace_back(std::move(userslice));
        outputMemSize += userslice.size;
    }

    //  此处虽然带入inputMem作为scratch mem, 但inputMem 不能被使用
    CHK_RET(level1TempAlg->Prepare(
        execMem.outputMem, execMem.outputMem, execMem.inputMem,
        outputSlices[level1CommInfo.localRank].size / perDataSize, param.VDataDes.dataType, param.stream,
        HcclReduceOp::HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, outputSlices, 0));

    u32 rankSize = level1CommInfo.localRankSize;
    CHK_RET(level1TempAlg->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET,
        param.stream));

    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    HCCL_INFO("[CollAllGatherVMeshExecutor][RunLevel1] level 1 for A2 run success");

    return HCCL_SUCCESS;
}

HcclResult CollAllGatherVMeshExecutor::RunSingleMesh(const OpParam& param, ExecMem& execMem)
{
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    dataType = param.VDataDes.dataType;
    const u32 unitSize = SIZE_TABLE[dataType];

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 rankSize = level0CommInfo.localRankSize;

    // DMA消减后仅使用ccl out通信，ccl out根据实际使用大小重新申请内存空间
    u64 inputMemSize = execMem.inputMem.size();
    u64 baseOffset = 0;
    DeviceMem curOutputMem = execMem.outputMem.range(baseOffset, inputMemSize);
    CHK_SMART_PTR_NULL(curOutputMem);

    // allgatherv 计算slice，数据分成ranksize份，每份的起始偏移和大小
    std::vector<Slice> outputSlices;
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    const auto displs = static_cast<u64*>(param.VDataDes.displs);
    for (u32 rank = 0; rank < rankSize; ++rank) {
        Slice userslice;
        userslice.offset = displs[rank] * unitSize;
        userslice.size = counts[rank] * unitSize;
        outputSlices.emplace_back(std::move(userslice));
    }

    // DMA消减场景，打包opInfo
    HcomCollOpInfo opInfo
        = {"", execMem.inputPtr, execMem.outputPtr, execMem.count, dataType, param.root, param.reduceType, 0};

    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_MESH_DIRECT, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, topoAttr_.userRank, &opInfo,
        level0CommInfo.localRank, level0CommInfo.localRankSize));

    CHK_RET(tempAlg->Prepare(
        curOutputMem, curOutputMem, execMem.inputMem, execMem.count, dataType, param.stream, HCCL_REDUCE_RESERVED,
        LEVEL0_BRIDGE_RANK_ID, outputSlices, baseOffset));

    CHK_RET(tempAlg->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
        param.stream));

    CHK_RET(RunTemplate(tempAlg, level0CommInfo));

    HCCL_INFO("[CollAllGatherVMeshExecutor][RunSingleMesh] single mesh for A2 run success");
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherVMeshExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherVMeshExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    if (topoAttr_.moduleNum > 1) {
        // 获取子通信域信息
        CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
        SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
        CHK_RET(CheckCommSize(COMM_LEVEL1, level0CommInfo.localRank + 1));
        SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, level0CommInfo.localRank);

        CHK_RET(RunLevel0(param, execMem, level0CommInfo, level1CommInfo));
        CHK_RET(RunLevel1(param, execMem, level0CommInfo, level1CommInfo));
    } else {
        CHK_RET(RunSingleMesh(param, execMem));
    }

    return HCCL_SUCCESS;
}

HcclResult CollAllGatherVMeshExecutor::CalcCurCountsAndCurDisplsMultiModule(
    const u64 maxTotalCount, std::vector<u64>& countsLeft, std::vector<u64>& displs, std::vector<u64>& curCounts,
    std::vector<u64>& curDispls, bool& finished) const
{
    curCounts = std::vector<u64>(countsLeft.size(), 0);
    curDispls = std::vector<u64>(displs.size(), 0);
    auto allocatableCount = maxTotalCount;

    HCCL_DEBUG("CalcCurCountsAndCurDisplsMultiModule begin");
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
            HCCL_INFO("[%s] Calc CurCountsAndCurDispls for multiModule finish", __func__);
            return HCCL_SUCCESS;
        }
        // 计算每个rank可以分到多少count
        const auto perRankCount = allocatableCount / nonZeroCount;
        if (perRankCount == 0) {
            break;
        }
        HCCL_DEBUG("[%s] Calc CurCountsAndCurDispls for perRankCount finish", __func__);
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

HcclResult CollAllGatherVMeshExecutor::CalcCurCountsAndCurDisplsSingleModule(
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

    return HCCL_SUCCESS;
}

HcclResult CollAllGatherVMeshExecutor::CalcCurCountsAndCurDispls(
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

REGISTER_EXEC("AllGatherVMeshExecutor", AllGatherVMesh, CollAllGatherVMeshExecutor);
} // namespace hccl
