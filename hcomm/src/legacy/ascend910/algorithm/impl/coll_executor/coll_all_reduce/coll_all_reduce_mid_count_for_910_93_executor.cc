/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_mid_count_for_910_93_executor.h"

namespace hccl {
CollAllReduceMidCountFor91093Executor::CollAllReduceMidCountFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
    desc_.level1SupportedAlgos = {
        AlgTypeLevel1::ALG_LEVEL1_NHR,
    };
    desc_.level2SupportedAlgos = {
        AlgTypeLevel2::ALG_LEVEL2_NHR,
    };
}

HcclResult CollAllReduceMidCountFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMidCountFor91093Executor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        HCCL_ERROR("AllReduceMidCountFor91093Executor do not support offload mode");
        return HCCL_E_UNAVAIL;
    }
    HCCL_INFO(
        "[CollAllReduceMidCountFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMidCountFor91093Executor::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaCombineL1(COMM_COMBINE_L1, CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaCombineL1, opTransport[COMM_COMBINE_L1], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMidCountFor91093Executor::CalcLevel2CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollAllReduceMidCountFor91093Executor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = cclBuffSize / HCCL_MIN_SLICE_ALIGN_910_93 / unitSize * HCCL_MIN_SLICE_ALIGN_910_93;
    return maxCountPerLoop;
}

HcclResult CollAllReduceMidCountFor91093Executor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] The MidCountFor91093Executor starts, topoType_[%u]", __func__, topoType_);

    u32 unitSize = 0;
    CHK_RET(SalGetDataTypeSize(param.GetDataType(), unitSize));

    // 获取 level1 打平级通信域
    CHK_RET(CheckCommSize(COMM_COMBINE_L1, COMM_INDEX_0 + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_COMBINE_L1, COMM_INDEX_0);

    // 获取 level2 级通信域
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level2RankSize = level2CommInfo.localRankSize;
    u64 inputMemSize = execMem.count * unitSize;
    const u32 SINGLERANK = 1;

    if (DMAReduceFlag_) {
        DeviceMem srcMem = DeviceMem::create(static_cast<u8*>(execMem.inputPtr), inputMemSize);
        DeviceMem dstMem = execMem.inputMem.range(0, inputMemSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
        HCCL_DEBUG("copy from user in to ccl in.");
    }

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);

    // step1:  run nhr ont shot in level1
    if (level1RankSize > SINGLERANK) {
        std::unique_ptr<AlgTemplateBase> level1tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_REDUCE_NHR_ONESHOT, dispatcher_);
        CHK_SMART_PTR_NULL(level1tempAlg);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NHR_ONESHOT in COMM_COMBINE_L1/COMM_LEVEL2", __func__);
        HCCL_INFO("AllReduce mid count: using nhr algo intra-server.");

        CHK_RET(level1tempAlg->Prepare(reduceAttr));
        level1tempAlg->CloseBarrier();

        CHK_RET(level1tempAlg->Prepare(
            execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.stream,
            param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), 0));

        CHK_RET(level1tempAlg->RegisterProfiler(
            (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_0,
            HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level1tempAlg, level1CommInfo));
    }

    // 数据回拷
    if (level1RankSize > SINGLERANK && level2RankSize > SINGLERANK) {
        DeviceMem srcMem = execMem.outputMem.range(0, inputMemSize);
        DeviceMem dstMem = execMem.inputMem.range(0, inputMemSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    }

    // step2:  run nhr ont shot in level2
    if (level2RankSize > SINGLERANK) {
        std::unique_ptr<AlgTemplateBase> level2tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_REDUCE_NHR_ONESHOT, dispatcher_);
        CHK_SMART_PTR_NULL(level2tempAlg);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NHR_ONESHOT in COMM_COMBINE_L1/COMM_LEVEL2", __func__);
        HCCL_INFO("AllReduce mid count: using nhr algo intra-server.");

        CHK_RET(level2tempAlg->Prepare(reduceAttr));
        level2tempAlg->CloseBarrier();

        CHK_RET(level2tempAlg->Prepare(
            execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.stream,
            param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), 0));

        CHK_RET(level2tempAlg->RegisterProfiler(
            (level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank, PROF_STAGE_1,
            HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level2tempAlg, level2CommInfo));
    }

    if (DMAReduceFlag_) {
        DeviceMem srcMem = execMem.outputMem.range(0, inputMemSize);
        DeviceMem dstMem = DeviceMem::create(static_cast<u8*>(execMem.outputPtr), inputMemSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
        HCCL_DEBUG("copy from ccl out to user out.");
    }

    HCCL_INFO("AllReduce mid count run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceMidCountFor91093Executor", AllReduceMidCountFor91093, CollAllReduceMidCountFor91093Executor);
} // namespace hccl
