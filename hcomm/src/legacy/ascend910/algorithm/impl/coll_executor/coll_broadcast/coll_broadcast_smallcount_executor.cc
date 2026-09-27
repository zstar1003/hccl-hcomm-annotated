/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_broadcast_smallcount_executor.h"
#include <cmath>

namespace hccl {

CollBroadcastSmallCountExecutor::CollBroadcastSmallCountExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        scratchMemFlag_ = true;
    }
}

void CollBroadcastSmallCountExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    root_ = param.root;
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
    totalSize_ = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
}

HcclResult CollBroadcastSmallCountExecutor::CalcScratchMemSize(u64& scratchMemSize)
{
    const u32 base = 2;
    scratchMemSize = 0U;
    if (scratchMemFlag_) {
        scratchMemSize = static_cast<u64>(log2(base * topoAttr_.userRankSize - 1)) - 1;
        scratchMemSize *= totalSize_;
    }
    HCCL_INFO(
        "[CollBroadcastSmallCountExecutor][CalcScratchMemSize] tag[%s] scratchMemSize[%llu]", tag_.c_str(),
        scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastSmallCountExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = 1;
    HCCL_INFO("[CollBroadcastSmallCountExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastSmallCountExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastSmallCountExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastSmallCountExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollBroadcastSmallCountExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    std::vector<Slice> dataSegsSlice;

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    CHK_RET(ActiveSlaveStreams(param.stream));
    HcomCollOpInfo opInfoPtr
        = {"", execMem.inputPtr, nullptr, param.DataDes.count, param.DataDes.dataType, param.root, param.reduceType, 0};

    std::unique_ptr<AlgTemplateBase> level0TempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_HD, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_HD in COMM_LEVEL0", __func__);
    CHK_SMART_PTR_NULL(level0TempAlg);

    u32 rootRank = 0;
    HcclResult ret = GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, param.root, rootRank);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollBroadcastSmallCountExecutor][KernelRun]invalid root [%u] to get userrank", param.root), ret);

    CHK_RET(level0TempAlg->Prepare(
        execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.stream,
        HCCL_REDUCE_RESERVED, rootRank, algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        level0CommInfo.localRank, &opInfoPtr));

    CHK_RET(level0TempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_1,
        HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    HCCL_INFO("broadcast small count executor run success.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("BroadCastSmallCountExecutor", BroadcastSmallCount, CollBroadcastSmallCountExecutor);

} // namespace hccl
