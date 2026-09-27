/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_v_for_310p_ring_executor.h"

#include <algorithm>

namespace hccl {
CollReduceScatterVFor310PRingExecutor::CollReduceScatterVFor310PRingExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterVExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
    CCLMemSlice_ = true;
}

HcclResult CollReduceScatterVFor310PRingExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = DMAReduceFlag_ ? 1 : 0;
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVFor310PRingExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVFor310PRingExecutor::CalcTransportMemType(
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
        "[CollReduceScatterVFor310PRingExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVFor310PRingExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollReduceScatterVFor310PRingExecutor][CalcLevel0CommInfo]tag[%s] start", tag_.c_str());
    CommParaInfo commParaInfo(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollReduceScatterVFor310PRingExecutor][CalcLevel0CommInfo]tag[%s] Calc RingComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVFor310PRingExecutor::CalcCurCountsAndCurDispls(
    const u64 maxTotalCount, std::vector<u64>& countsLeft, std::vector<u64>& displs, std::vector<u64>& curCounts,
    std::vector<u64>& curDispls, bool& finished)
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
            break;
        } else {
            // 计算每个rank可以分到多少count
            const auto perRankCount = allocatableCount / nonZeroCount;
            if (perRankCount == 0) {
                break;
            }
            // 分配好每个rank的counts
            for (auto i = 0U; i < countsLeft.size(); ++i) {
                const auto curCount = countsLeft[i] < perRankCount ? countsLeft[i] : perRankCount;
                allocatableCount -= curCount;
                curCounts[i] += curCount;
                countsLeft[i] -= curCount;
                displs[i] += curCount;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVFor310PRingExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterVFor310PRingExecutor][KernelRun] 310p aiv ReduceScatterV start");
    const auto* displsPtr = static_cast<const u64*>(param.VDataDes.displs);
    HcclDataType dataType = param.VDataDes.dataType;
    const u32 unitSize = SIZE_TABLE[dataType];

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    const u32 level0RankSize = outerCommInfo.localRankSize;

    bool isInlineReduce
        = IsSupportSDMAReduce(execMem.inputMem.ptr(), execMem.outputMem.ptr(), dataType, param.reduceType);

    u64 reduceAttr = 0;
    if (isInlineReduce) {
        SalSetBitOne(reduceAttr, ATTR_POS_INLINE_REDUCE);
    } else {
        HCCL_ERROR("[CollReduceScatterVFor310PRingExecutor][KernelRun] ReduceScatterV only support InlineReduce!");

        return HCCL_E_NOT_SUPPORT;
    }

    // 根据counts和displace计算每个rank的数据范围
    // 两个slices: dataSlices里的offset是cclBuffer范围内的偏移，就地计算得出
    // outputSlices里的offset是user output的偏移，使用传入的displs算得
    std::vector<Slice> dataSlices;
    std::vector<Slice> outputSlices;
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    auto displace = 0ULL;
    for (auto rank = 0U; rank < level0RankSize; ++rank) {
        Slice slice;
        slice.offset = displace * unitSize;
        slice.size = counts[rank] * unitSize;
        dataSlices.emplace_back(slice);

        slice.offset = displsPtr[rank] * unitSize;
        outputSlices.emplace_back(std::move(slice));

        displace += counts[rank];
    }

    // opInfo这里主要填对inputPtr和outputPtr就好
    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, 0, dataType, 0, param.reduceType, 0};
    HcomCollOpInfo* opInfoPtr = nullptr;
    if (DMAReduceFlag_) {
        opInfoPtr = &opInfo;
    }

    std::vector<u32> rankOrder(level0RankSize, 0);
    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_RING_DIRECT, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(
        reduceAttr, opInfoPtr, topoAttr_.userRank, algResResp_->slaveStreams, algResResp_->notifiesMain,
        algResResp_->notifiesAux, rankOrder, outputSlices, true));

    CHK_RET(tempAlg->Prepare(
        execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count, dataType, param.stream, param.reduceType,
        0, dataSlices));

    CHK_RET(tempAlg->RegisterProfiler(
        (outerCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerCommInfo.localRank, PROF_STAGE_0,
        HCCL_EXEC_STEP_NOT_SET, param.stream));

    // 执行Ring算法
    CHK_RET(RunTemplate(tempAlg, outerCommInfo));

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterVFor310PRing", ReduceScatterVFor310PRing, CollReduceScatterVFor310PRingExecutor);
} // namespace hccl
