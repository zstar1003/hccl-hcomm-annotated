/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_single_rank_executor.h"

namespace hccl {
CollAllGatherSingleRankExecutor::CollAllGatherSingleRankExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{}

HcclResult CollAllGatherSingleRankExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherSingleRankExecutor][KernelRun]AllGather single rank");
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    auto originalAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    bool hugeData = (execMem.count * unitSize) > SDMA_SEND_MAX_SIZE;
    if (execMem.inputPtr == execMem.outputPtr) {
        // 通过CopyPattern字段区分不同的子图
        auto opMeta = HcclOpMetaInfo::GetOneForAllGather(originalAlgTypeLevel1, hugeData, false, CopyPattern::ZCOPY);
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
    } else {
        auto opMeta = HcclOpMetaInfo::GetOneForAllGather(originalAlgTypeLevel1, hugeData, false, CopyPattern::BCOPY);
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
        // ranksize = 1; input、output地址不同，input->output
        DeviceMem srcMem(execMem.inputPtr, execMem.count * unitSize);
        DeviceMem dstMem(execMem.outputPtr, execMem.count * unitSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    }
    CHK_RET(LaunchTaskExtend(dispatcher_, const_cast<Stream&>(param.stream), algResResp_->slaveStreams));

    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherSingleExecutor", AllGatherSingleRank, CollAllGatherSingleRankExecutor);

} // namespace hccl
