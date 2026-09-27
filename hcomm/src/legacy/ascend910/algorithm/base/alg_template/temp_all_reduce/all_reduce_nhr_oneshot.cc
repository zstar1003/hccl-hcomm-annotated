/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_reduce_nhr_oneshot.h"
#include "alg_template_register.h"
#include "reduce_nhr_oneshot.h"
#include "broadcast_nhr_oneshot.h"

namespace hccl {
AllReduceNHROneshot::AllReduceNHROneshot(const HcclDispatcher dispatcher) : NHRBase(dispatcher) {}

AllReduceNHROneshot::~AllReduceNHROneshot() {}

HcclResult AllReduceNHROneshot::Prepare(u64 reduceAttrBitMap, [[maybe_unused]] HcomCollOpInfo* opInfo)
{
    reduceAttr_ = reduceAttrBitMap;
    return HCCL_SUCCESS;
}

HcclResult AllReduceNHROneshot::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 基本的检查
    CHK_RET(SimpleCheck(rank, rankSize, links));
    HCCL_INFO(
        "[AllReduceNHROneshot][RunAsync] run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank,
        rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    HcclResult ret = HCCL_SUCCESS;
    // 如果ranksize为1, inline reduce和普通跨片reduce操作一致，从input->output
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceNHROneshot][RunAsync] rank[%u] memcpy async failed", rank),
                ret);
        }

        return ret;
    }

    // 先执行1-reduce
    ret = RunReduceOneshot(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceNHROneshot][RunAsync] rank[%u] count[%llu] failed in "
            "1-reduce step",
            rank, count_),
        ret);

    // 再执行1-bcast
    ret = RunBroadcastOneshot(rank, rankSize, links);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AllReduceNHROneshot][RunAsync] rank[%u] count[%llu] failed in "
            "1-bcast step",
            rank, count_),
        ret);

    HCCL_INFO("[AllReduceNHROneshot][RunAsync] finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceNHROneshot::SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK>& links)
{
    // 判断stream, dispatcher是否为空
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());

    // 检查memory
    CHK_PRT_RET(
        !outputMem_ || !inputMem_,
        HCCL_ERROR("[AllReduceNHROneshot][SimpleCheck] rank[%u] inputmem or outputmem is null", rank), HCCL_E_PTR);

    // 判断links数量是否正确
    CHK_PRT_RET(
        links.size() < rankSize,
        HCCL_ERROR(
            "[AllReduceNHROneshot][SimpleCheck] rank[%u] link size[%llu] is "
            "less than rank size[%u]",
            rank, links.size(), rankSize),
        HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

HcclResult AllReduceNHROneshot::RunReduceOneshot(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    std::unique_ptr<AlgTemplateBase> tempAlg;
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCE_NHR_ONE_SHOT, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(reduceAttr_));
    if (!barrierSwitchOn_) {
        tempAlg->CloseBarrier();
    }
    HCCL_INFO(
        "[AllReduceNHROneshot][RunReduceOneshot] 1-reduce tempAlg rank[%u] inputMem[%p] outputMem[%p] "
        "mem_size[%llu] count[%llu] planeID:[%d]",
        rank, inputMem_.ptr(), outputMem_.ptr(), outputMem_.size(), count_, profilerInput_.planeID);
    CHK_RET(tempAlg->Prepare(
        inputMem_, inputMem_, outputMem_, count_, dataType_, stream_, reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(tempAlg->RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    return tempAlg->RunAsync(rank, rankSize, links);
}

HcclResult AllReduceNHROneshot::RunBroadcastOneshot(u32 rank, u32 rankSize, const std::vector<LINK>& links)
{
    BroadcastNHROneshot tempAlg(dispatcher_);
    HCCL_INFO(
        "[AllReduceNHROneshot][RunBroadcastOneshot] 1-broadcast tempAlg rank[%u] inputMem[%p] outputMem[%p] "
        "mem_size[%llu] count[%llu] planeID:[%d]",
        rank, inputMem_.ptr(), outputMem_.ptr(), outputMem_.size(), count_, profilerInput_.planeID);
    CHK_RET(tempAlg.Prepare(
        inputMem_, outputMem_, outputMem_, count_, dataType_, stream_, reductionOp_, root_, slices_, baseOffset_));

    CHK_RET(tempAlg.RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step, stream_));

    return tempAlg.RunAsyncForAllReduce(rank, rankSize, links);
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_NHR_ONESHOT, AllReduceNHROneshot);
} // namespace hccl
