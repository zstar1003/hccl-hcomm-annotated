/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include "mc2_handler.h"

namespace hccl {

Mc2HandlerPub::Mc2HandlerPub()
{
    // mc2TurnNum_ 用于提供turnNum的地址
    std::iota(mc2TurnNum_, mc2TurnNum_ + MC2_MAX_TURN * MC2_MAX_RANK_NUM, 0);
}

HcclResult
Mc2HandlerPub::Mc2WaitValue(HcclDispatcher dispatcherPtr, hccl::Stream& stream, Mc2Handler* mc2Handler, u32 step) const
{
    if (mc2Handler == nullptr) {
        HCCL_INFO("[Mc2HandlerPub][Mc2WaitValue] mc2Handler is NULL.");
        return HCCL_SUCCESS;
    } else if (mc2Handler->repeatCnt < 1 || step >= mc2Handler->rankSize) {
        // repeatCnt 至少为1
        // step 校验 范围为[0, rankSize-1]
        HCCL_ERROR("[Mc2HandlerPub][Mc2WaitValue] error parameter, repeatCnt %u, step %u", mc2Handler->repeatCnt, step);
        return HCCL_E_PARA;
    } else if (mc2Handler->stepSize == 0 || (step + 1) % mc2Handler->stepSize != 0) {
        HCCL_INFO(
            "[Mc2HandlerPub][Mc2WaitValue] no need to add wait task, step %u, stepSize %u.", step,
            mc2Handler->stepSize);
        return HCCL_SUCCESS;
    }
    // 第 repeatCnt轮通信，turnNum范围为[(repeatCnt -1)*rankSize + 1, repeatCnt*rankSize]
    // 计算当前轮次的turnNum值
    u32 turnNum = (mc2Handler->repeatCnt - 1) * (mc2Handler->rankSize) + (step + 1);
    HCCL_INFO(
        "[Mc2HandlerPub][Mc2WaitValue] repeatCnt %u, rankSize %u, step %u, turnNum %u", mc2Handler->repeatCnt,
        mc2Handler->rankSize, step, turnNum);
    u64 valueAddr = mc2Handler->valueAddr + sizeof(uint32_t) * turnNum;
    CHK_RET(HcclDispatcherWaitValue(dispatcherPtr, stream, mc2Handler->commitAddr, valueAddr, false));
    return HCCL_SUCCESS;
}

HcclResult Mc2HandlerPub::Mc2WriteValue(HcclDispatcher dispatcherPtr, hccl::Stream& stream, Mc2Handler* mc2Handler)
{
    if (mc2Handler == nullptr) {
        HCCL_INFO("[Mc2HandlerPub][Mc2WriteValue] mc2Handler is NULL.");
        return HCCL_SUCCESS;
    }
    turnNumForWrite_++;
    u32 turnNum = (mc2Handler->repeatCnt - 1) * (mc2Handler->rankSize) + turnNumForWrite_;
    HCCL_INFO("[Mc2HandlerPub][Mc2WriteValue] turnNumForWrite %u, turnNum %u.", turnNumForWrite_, turnNum);
    u64 valueAddr = mc2Handler->valueAddr + sizeof(uint32_t) * turnNum;
    CHK_RET(HcclDispatcherWriteValue(dispatcherPtr, stream, mc2Handler->finishAddr, valueAddr));
    return HCCL_SUCCESS;
}
} // namespace hccl
