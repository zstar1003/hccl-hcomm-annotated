/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_single_rank_executor.h"

namespace hccl {
CollAlltoAllSingleRankExecutor::CollAlltoAllSingleRankExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{}

HcclResult CollAlltoAllSingleRankExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;
    tag_ = param.tag;
    algResResp_ = &algRes;
    AlltoAllVParam_ = param;
    ExecMem execMem;
    execMem.count = 0;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    ret = KernelRun(param, execMem);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAlltoAllSingleRankExecutor][Orchestrate]errNo[0x%016llx]executor run failed", HCCL_ERROR_CODE(ret)),
        ret);

    HCCL_INFO(
        "tag[%s], AlltoAllSingleRankExecutor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllSingleRankExecutor::GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo)
{
    (void)algRes;
    (void)adjInfo;
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllSingleRankExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAlltoAllSingleRankExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    u64 sendCount = 0;
    u64 totalSize = 0;
    u64 unitSize = SIZE_TABLE[param.All2AllDataDes.sendType];
    CopyPattern copyPattern = (execMem.inputPtr == execMem.outputPtr) ? CopyPattern::ZCOPY : CopyPattern::BCOPY;
    HcclOpMetaInfo opMeta;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        sendCount = *(static_cast<u64*>(param.All2AllDataDes.sendCountMatrix));
        totalSize = sendCount * unitSize;
        opMeta = HcclOpMetaInfo::GetOneForAllToAllVC(copyPattern, totalSize, totalSize > SDMA_SEND_MAX_SIZE);
    } else {
        sendCount = *(static_cast<u64*>(param.All2AllDataDes.sendCounts));
        totalSize = sendCount * unitSize;
        opMeta = HcclOpMetaInfo::GetOneForAllToAllVC(copyPattern, totalSize, totalSize > SDMA_SEND_MAX_SIZE);
    }
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));

    if (execMem.inputPtr != execMem.outputPtr) {
        DeviceMem srcMem(execMem.inputPtr, totalSize);
        DeviceMem dstMem(execMem.outputPtr, totalSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    }
    CHK_RET(LaunchTaskExtend(dispatcher_, const_cast<Stream&>(param.stream), algResResp_->slaveStreams));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("RunAlltoAllSingleExecutor", AlltoAllSingleRank, CollAlltoAllSingleRankExecutor);

} // namespace hccl
