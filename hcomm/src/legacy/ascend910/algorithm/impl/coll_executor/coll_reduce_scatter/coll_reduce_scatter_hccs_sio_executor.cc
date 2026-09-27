/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_hccs_sio_executor.h"
#include "alg_template_register.h"

namespace hccl {

CollReduceScatterHccsSioExecutor::CollReduceScatterHccsSioExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    desc_.isZeroCopy = false;
    desc_.deterministic = 1;
    DMAReduceFlag_ = true;
    CCLMemSlice_ = false;
}

u64 CollReduceScatterHccsSioExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = inCCLbufferSize_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / unitSize;
    HCCL_INFO(
        "[CollReduceScatterHccsSioExecutor][CalcLoopMaxCount]"
        "using default maxCountPerLoop[%llu] as CCLBuffSize / (userRankSize * unitSize).",
        maxCountPerLoop);
    return maxCountPerLoop;
}

HcclResult CollReduceScatterHccsSioExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation + 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollReduceScatterHccsSioExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterHccsSioExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult
CollReduceScatterHccsSioExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollReduceScatterHccsSioExecutor][CalcTransportMemType] tag[%s] inputType[%d],"
        " outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterHccsSioExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_HCCS_PLUS_SIO);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    LevelNSubCommTransport& commTransportLevel0 = opTransport[COMM_LEVEL0];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        for (auto& transportRequest :
             commTransportLevel0[subCommIndex].transportRequests) { // 两个transport分别设置ishccs为false、true
            // 根据子通信索引设置isHccs的值
            transportRequest.linkType = (subCommIndex == 0) ? TransportLinkType::SIO : TransportLinkType::HCCS;
            HCCL_INFO(
                "[CollReduceScatterHccsSioExecutor][CalcLevel0CommInfo] set extra notifyNum[%u]",
                transportRequest.notifyNum);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterHccsSioExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] userRank[%u] starts.", __func__, topoAttr_.userRank);
    HcclDataType dataType = param.DataDes.dataType;
    HcomCollOpInfo opInfo = {"",
                             execMem.inputPtr,
                             execMem.outputPtr,
                             param.DataDes.count,
                             param.DataDes.dataType,
                             param.root,
                             param.reduceType,
                             0};

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo subCommInfoHccs = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_1 + 1));
    SubCommInfo subCommInfoSio = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_1);

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, dataType, param.reduceType);
    std::unique_ptr<AlgTemplateBase> TempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_HCCS_SIO, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_HCCS_SIO in COMM_LEVEL0", __func__);
    CHK_SMART_PTR_NULL(TempAlg);

    CHK_RET(TempAlg->Prepare(
        execMem.inputMem, execMem.outputMem, execMem.scratchMem, execMem.count, dataType, param.stream,
        param.reduceType, LEVEL0_BRIDGE_RANK_ID, 0, reduceAttr, algResResp_->slaveStreams, algResResp_->notifiesMain,
        algResResp_->notifiesAux, topoAttr_.userRank, subCommInfoHccs, subCommInfoSio, &opInfo));

    CHK_RET(TempAlg->RegisterProfiler(
        (subCommInfoHccs.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + subCommInfoHccs.localRank, PROF_STAGE_0,
        HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(TempAlg, subCommInfoHccs));

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterHccsSioExecutor", ReduceScatterHccsSio, CollReduceScatterHccsSioExecutor);
} // namespace hccl
