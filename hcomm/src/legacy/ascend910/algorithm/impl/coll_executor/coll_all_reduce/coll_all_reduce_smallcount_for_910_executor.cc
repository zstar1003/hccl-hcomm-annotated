/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_smallcount_for_910_executor.h"

namespace hccl {
CollAllReduceSmallCountFor910Executor::CollAllReduceSmallCountFor910Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
}

HcclResult CollAllReduceSmallCountFor910Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;

    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountFor910Executor::CalcTransportMemType(
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
        "[CollAllReduceSmallCountFor910Executor][CalcTransportMemType]"
        "tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountFor910Executor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollAllReduceSmallCountFor910Executor][CalcLevel0CommInfo]tag[%s] start", tag_.c_str());

    if (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_HD) {
        CommParaInfo commParaInfo(COMM_LEVEL0, CommType::COMM_TAG_HALVING_DOUBLING);
        CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL0], inputType, outputType));
    } else {
        HCCL_ERROR("unsupported algType %d plus %d", algType_.algoLevel0, algType_.algoLevel1);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[CollAllReduceSmallCountFor910Executor][CalcLevel0CommInfo]tag[%s] Calc RingComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountFor910Executor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    CHK_PRT_RET(
        workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE,
        HCCL_ERROR("[CollAllReduceSmallCountFor910Executor][Orchestrate] only support op base mode."),
        HCCL_E_NOT_SUPPORT);
    ParseParam(param);
    algResResp_ = &algRes;
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    execMem.inputMem = algRes.cclInputMem;
    execMem.outputMem = algRes.cclOutputMem;
    execMem.scratchMem = algRes.scratchMem;

    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 totalSize = execMem.count * unitSize; // 单位：字节

    DeviceMem inMem(execMem.inputPtr, totalSize);
    DeviceMem inCommMem = execMem.inputMem.range(0, totalSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCommMem, inMem, param.stream));
    HCCL_DEBUG("[CollAllReduceSmallCountFor910Executor][RunLoop]copy from user in to ccl out.");
    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAllReduceSmallCountFor910Executor][Orchestrate]errNo[0x%016llx]executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    DeviceMem outMem(execMem.outputPtr, totalSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outMem, inCommMem, param.stream));
    HCCL_DEBUG("[CollAllReduceSmallCountFor910Executor][RunLoop]copy from ccl out to usr out.");

    HCCL_INFO(
        "tag[%s], AllReduce executor orchestrate success, take time [%lld]us", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceSmallCountFor910Executor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[CollAllReduceSmallCountFor910Executor][KernelRun] userRank[%u] starts", topoAttr_.userRank);

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u64 reduceAttr = 0;
    std::unique_ptr<AlgTemplateBase> tempAlg;
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_REDUCE_DOUBLING_LOCAL_REDUCE, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(reduceAttr));

    CHK_RET(tempAlg->Prepare(
        execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.stream,
        param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), 0));

    CHK_RET(tempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_0,
        HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(tempAlg, level0CommInfo));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceSmallCountFor910", AllReduceSmallCountFor910, CollAllReduceSmallCountFor910Executor);
} // namespace hccl
