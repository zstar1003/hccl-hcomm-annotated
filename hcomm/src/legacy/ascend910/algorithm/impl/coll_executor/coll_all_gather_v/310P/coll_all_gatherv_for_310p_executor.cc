/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gatherv_for_310p_executor.h"

#include <algorithm>

namespace hccl {
CollAllGatherVFor310PExecutor::CollAllGatherVFor310PExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherVExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
}

HcclResult CollAllGatherVFor310PExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = DMAReduceFlag_ ? 1 : 0;
    HCCL_INFO("[CollAllGatherVFor310PExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);

    return HCCL_SUCCESS;
}

HcclResult CollAllGatherVFor310PExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult
CollAllGatherVFor310PExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollAllGatherVFor310PExecutor][CalcTransportMemType]"
        "tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherVFor310PExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollAllGatherVFor310PExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = cclBuffSize / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / unitSize;

    HCCL_INFO(
        "[CollAllGatherVExecutor][CalcLoopMaxCount]"
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.",
        maxCountPerLoop);
    return maxCountPerLoop;
}

bool CollAllGatherVFor310PExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize > HCCL_SMALL_COUNT_4_MB;
    return hugeData;
}

HcclResult CollAllGatherVFor310PExecutor::CalcCurCountsAndCurDispls(
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

    return HCCL_SUCCESS;
}

HcclResult CollAllGatherVFor310PExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherVFor310PExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    dataType = param.VDataDes.dataType;
    const u32 unitSize = SIZE_TABLE[dataType];

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u32 rankSize = outerCommInfo.localRankSize;

    // allgatherv 计算slice，从外部传入
    std::vector<Slice> inputSlices;
    std::vector<Slice> outputSlices;
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    const auto displs = static_cast<u64*>(param.VDataDes.displs);
    u64 cclOffset = 0;

    for (u32 rank = 0; rank < rankSize; ++rank) {
        Slice cclslice;
        cclslice.offset = cclOffset;
        cclslice.size = counts[rank] * unitSize;
        cclOffset += cclslice.size;
        inputSlices.emplace_back(std::move(cclslice));

        Slice userslice;
        userslice.offset = displs[rank] * unitSize;
        userslice.size = counts[rank] * unitSize;
        outputSlices.emplace_back(std::move(userslice));
    }

    HcomCollOpInfo opInfo
        = {"", execMem.inputPtr, execMem.outputPtr, 0, param.VDataDes.dataType, param.root, param.reduceType, 0};

    std::unique_ptr<AlgTemplateBase> tempAlg;
    if (!IsHugeData(cclOffset)) {
        tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_RING_DIRECT, dispatcher_);
        CHK_SMART_PTR_NULL(tempAlg);
        CHK_RET(tempAlg->Prepare(&opInfo, topoAttr_.userRank, outputSlices));
    } else {
        std::vector<u32> rankOrder(rankSize, 0);
        tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT, dispatcher_);
        CHK_SMART_PTR_NULL(tempAlg);
        CHK_RET(tempAlg->Prepare(
            &opInfo, topoAttr_.userRank, algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
            rankOrder, outputSlices));
    }

    CHK_RET(tempAlg->Prepare(
        execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count, dataType, param.stream, param.reduceType,
        0, inputSlices));

    CHK_RET(tempAlg->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerCommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
        param.stream));

    CHK_RET(RunTemplate(tempAlg, outerCommInfo));

    HCCL_INFO("AllGatherV for 310P run success");

    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherVFor310PExecutor", AllGatherVFor310P, CollAllGatherVFor310PExecutor);

} // namespace hccl
