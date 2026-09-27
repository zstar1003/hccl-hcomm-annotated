/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_semi_ring_executor.h"

namespace hccl {

CollReduceScatterSemiRingExecutor::CollReduceScatterSemiRingExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterRingFor91093Executor(dispatcher, topoMatcher)
{}

HcclResult CollReduceScatterSemiRingExecutor::CalcNotifyNum(u32 streamNum, u32& notifyNum)
{
    // notify数量是从流的两倍 + 新增带notifyId的notify资源
    notifyNum = 2U * streamNum + (topoAttr_.deviceNumPerAggregation + 4U);
    HCCL_INFO("[CollReduceScatterSemiRingExecutor][CalcNotifyNum]tag[%s] notifyNum_ is [%u]", tag_.c_str(), notifyNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterSemiRingExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE + 1U;

    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollReduceScatterSemiRingExecutor][CalcStreamNum] tag[%s] streamNum_[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

void CollReduceScatterSemiRingExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;

    // 是否需要scratch memory
    if (isSupportSDMAReduce_ && topoAttr_.serverNum == 1) {
        scratchMemFlag_ = false;
    } else {
        scratchMemFlag_ = true;
    }

    // 记录图模式总数据量
    const HcclDataType dataType = param.GetDataType();
    totalSize_ = CalcTotalCount(param) * SIZE_TABLE[dataType];
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
}

HcclResult CollReduceScatterSemiRingExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));

    LevelNSubCommTransport& commTransportLevel0 = opTransport[COMM_LEVEL0];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        for (auto& transportRequest : commTransportLevel0[subCommIndex].transportRequests) {
            transportRequest.notifyNum = topoAttr_.deviceNumPerAggregation + 4U; // 只传递额外的notify个数
            HCCL_INFO(
                "[CollReduceScatterSemiRingExecutor][CalcLevel0CommInfo] set extra notifyNum[%u]",
                transportRequest.notifyNum);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterSemiRingExecutor::DoubleRingMidCountReduceScatter(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream,
    s32 profStage, const u64 baseOffset, const HcomCollOpInfo* opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice, const bool retryEnable)
{
    (void)tag;
    (void)baseOffset;
    (void)opInfo;
    (void)multRingsSliceZero;
    (void)retryEnable;
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterSemiRingExecutor] DoubleRingMidCountReduceScatter starts.");

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    // 此处计算reduceAttr计算outputmem使用scratchmem
    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);
    // 执行
    std::unique_ptr<AlgTemplateBase> executor = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_UNIFIED_MARCH, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_UNIFIED_MARCH in COMM_LEVEL0", __func__);
    CHK_SMART_PTR_NULL(executor);

    CHK_RET(executor->Prepare(
        stream, level0CommInfo, algResResp_->paramInputMem, algResResp_->paramOutputMem, inputMem, outputMem, count,
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, dataType, reductionOp,
        multRingsUserMemSlice, reduceAttr));

    HcclResult ret = executor->RegisterProfiler(
        ((COMM_INDEX_0 + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID)
            + (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
        profStage, HCCL_EXEC_STEP_NOT_SET, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceScatterSemiRingExecutor][DoubleRingMidCountReduceScatter]"
            "Double ring ReduceScatter failed,return[%d]",
            ret),
        ret);

    CHK_RET(executor->RunAsync());

    HCCL_INFO("[CollReduceScatterSemiRingExecutor] ReduceScatter double ring level1 run success");
    return ret;
}

HcclResult CollReduceScatterSemiRingExecutor::RunIntraSeverReduceScatter(
    const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count, const HcclDataType& dataType,
    const HcclReduceOp& reductionOp, const std::vector<std::vector<Slice>>& multRingsSliceZero, const Stream& stream,
    s32 profStage, const u64 baseOffset, const HcomCollOpInfo* opInfo,
    const std::vector<std::vector<Slice>>& multRingsUserMemSlice, const bool retryEnable)
{
    CHK_RET(DoubleRingMidCountReduceScatter(
        tag, inputMem, outputMem, count, dataType, reductionOp, multRingsSliceZero, stream, profStage, baseOffset,
        opInfo, multRingsUserMemSlice, retryEnable));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterSemiRingExecutor", ReduceScatterDoubleRingMidCount, CollReduceScatterSemiRingExecutor);

} // namespace hccl
