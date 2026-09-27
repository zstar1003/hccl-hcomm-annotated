/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_aligned_all_gather_double_ring_for_910_93_executor.h"
#include "hccl_types.h"

namespace hccl {
CollAlignedAllGatherDoubleRingFor91093Executor::CollAlignedAllGatherDoubleRingFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherRingFor91093Executor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
}

HcclResult CollAlignedAllGatherDoubleRingFor91093Executor::RunIntraSeverAllGather(
    const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count, const HcclDataType& dataType,
    const std::vector<std::vector<Slice>>& multRingsSliceZero, const Stream& stream, s32 profStage,
    const u64 baseOffset, const HcomCollOpInfo* opInfo, const std::vector<std::vector<Slice>>& multRingsUserMemSlice)
{
    CHK_RET(DoubleRingAllGather(
        tag, inputMem, outputMem, count, dataType, multRingsSliceZero, stream, profStage, baseOffset, opInfo,
        multRingsUserMemSlice));
    return HCCL_SUCCESS;
}

HcclResult CollAlignedAllGatherDoubleRingFor91093Executor::DoubleRingAllGather(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream, s32 profStage, const u64 baseOffset,
    const HcomCollOpInfo* opInfo, const std::vector<std::vector<Slice>> multRingsUserMemSlice)
{
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[CollAlignedAllGatherDoubleRingFor91093Executor]userRank[%u], count[%llu]", topoAttr_.userRank,
        count);

    (void)tag;
    HCCL_INFO("[CollAlignedAllGatherDoubleRingFor91093Executor][DoubleRingAllGather] DoubleRingAllGather starts");
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));
    // 拿到ring环映射关系
    SubCommInfo level0ZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder
        = GetRingsOrderByTopoType(level0ZeroCommInfo.localRankSize, topoType_, nicList);
    // 生成两个ring上的userMemOut_上对应的slices
    std::vector<std::vector<Slice>> userMemOutputSlicesOfDoubleRing;
    CHK_RET(CollectMultiRingsUserMemSlices(
        ringNum, dataType, opInfo, multRingsSliceZero, multiRingsOrder, multRingsUserMemSlice,
        userMemOutputSlicesOfDoubleRing));
    // 生成两个ring上的rankOrder
    std::vector<std::vector<u32>> rankOrders;
    CHK_RET(CollectMultiRingsRankOrder(ringNum, multiRingsOrder, rankOrders));
    // 初始化executor
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALIGNED_ALL_GATHER_DOUBLE_RING, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALIGNED_ALL_GATHER_DOUBLE_RING in COMM_LEVEL0", __func__);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(
        const_cast<HcomCollOpInfo*>(opInfo), topoAttr_.userRank, algResResp_->slaveStreams, algResResp_->notifiesMain,
        algResResp_->notifiesAux, rankOrders, userMemOutputSlicesOfDoubleRing));

    ret = tempAlg->Prepare(
        outputMem, outputMem, inputMem, count, dataType, stream, multRingsSliceZero, HCCL_REDUCE_RESERVED,
        LEVEL0_BRIDGE_RANK_ID, baseOffset);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAlignedAllGatherDoubleRingFor91093Executor][DoubleRingAllGather]Double ring "
            "AllGather failed, return[%d]",
            ret),
        ret);
    u32 ringIndexOp = COMM_INDEX_0;
    u32 rankSize = level0ZeroCommInfo.localRankSize;
    ret = tempAlg->RegisterProfiler(
        ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
            + level0ZeroCommInfo.localRank,
        profStage, HCCL_EXEC_STEP_NOT_SET, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAlignedAllGatherDoubleRingFor91093Executor][DoubleRingAllGather]Double ring "
            "AllGather failed, return[%d]",
            ret),
        ret);

    // 空拷贝用于后续操作附着
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    ret = RunTemplate(tempAlg, level0ZeroCommInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAlignedAllGatherDoubleRingFor91093Executor][DoubleRingAllGather] Double ring "
            "AllGather failed, return[%d]",
            ret),
        ret);
    // 添加空task,保证执行时不乱序
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

REGISTER_EXEC(
    "AlignedAllGatherDoubleRingFor91093Executor", AlignedAllGatherDoubleRingFor91093,
    CollAlignedAllGatherDoubleRingFor91093Executor);

} // namespace hccl
