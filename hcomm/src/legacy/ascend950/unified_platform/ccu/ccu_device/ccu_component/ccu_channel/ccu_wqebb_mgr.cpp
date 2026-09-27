/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_wqebb_mgr_legacy.h"
#include "ccu_res_specs_legacy.h"

namespace Hccl {

// 计算 num 的以 2 为底的对数后移位，相当于向上最接近的 2 的整数次幂
static uint32_t RoundUpToNextPowerOfTwo(uint32_t num)
{
    if (num == 0) {
        return 1;
    }

    num--;
    num |= num >> 1;
    num |= num >> SHIFT_2BITS;
    num |= num >> SHIFT_4BITS;
    num |= num >> SHIFT_8BITS;
    num |= num >> SHIFT_16BITS;
    return num + 1;
}

static uint32_t GetWqeBBReqSizeBySqSize(uint32_t sqSize)
{
    if (sqSize < CCU_MIN_SQ_DEPTH) {
        HCCL_WARNING(
            "[CcuJettyCtxMgr][%s] sqSize[%u] is too small, reset to [%u].", __func__, sqSize, CCU_MIN_SQ_DEPTH);
        return CCU_MIN_SQ_DEPTH;
    }

    // WQE basic block 的 size 必须是 2 的整数次幂，向上取整
    uint32_t wqeBBReqNum = RoundUpToNextPowerOfTwo(sqSize);
    if (wqeBBReqNum != sqSize) {
        HCCL_WARNING(
            "[CcuJettyCtxMgr][%s] sqSize[%u] is not power of 2, reset to [%u].", __func__, sqSize, wqeBBReqNum);
    }

    if (sqSize > CCU_MAX_SQ_DEPTH) {
        HCCL_WARNING(
            "[CcuJettyCtxMgr][%s] sqSize[%u] is too large, reset to [%u].", __func__, sqSize, CCU_MAX_SQ_DEPTH);
        return CCU_MAX_SQ_DEPTH;
    }

    return wqeBBReqNum;
}

CcuWqeBBMgr::CcuWqeBBMgr(const int32_t devLogicId, const uint8_t dieId) : devLogicId(devLogicId), dieId(dieId)
{
    uint32_t wqeBBNum = 0; // 获取失败或为0场景，分配将按资源不足操作
    (void)CcuResSpecifications::GetInstance(devLogicId).GetWqeBBNum(dieId, wqeBBNum);
    idAllocator = std::make_unique<CcuResIdAllocator>(wqeBBNum);
}

HcclResult CcuWqeBBMgr::Alloc(const uint32_t sqSize, ResInfo& wqeBBInfo)
{
    uint32_t wqeBBReqSize = GetWqeBBReqSizeBySqSize(sqSize);
    vector<ResInfo> resInfo;
    auto ret = idAllocator->Alloc(wqeBBReqSize, true, resInfo, "ResType::WqeBB"); // wqebb资源要求连续
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_WARNING("[CcuWqeBBMgr][%s] failed, sqSize[%u], wqeBBSize[%u]", __func__, sqSize, wqeBBReqSize);
        return ret;
    }

    wqeBBInfo = resInfo[0]; // 分配连续资源包含1个元素
    return ret;
}

HcclResult CcuWqeBBMgr::Release(const ResInfo& wqeBBInfo)
{
    auto ret = idAllocator->Release(wqeBBInfo.startId, wqeBBInfo.num);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR(
            "[CcuWqeBBMgr][%s] failed, wqe basic block resource info[%s]", __func__, wqeBBInfo.Describe().c_str()),
        ret);

    return HcclResult::HCCL_SUCCESS;
}

}; // namespace Hccl
