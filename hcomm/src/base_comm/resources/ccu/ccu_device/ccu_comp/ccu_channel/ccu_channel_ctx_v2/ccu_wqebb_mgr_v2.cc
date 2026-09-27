/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_wqebb_mgr_v2.h"

#include "ccu_res_specs.h"

namespace hcomm {

HcclResult CcuWqeBBMgrV2::Init()
{
    uint32_t wqeBBNum = 0; // 获取失败或为0场景，分配将按资源不足操作
    uint32_t jettyNum = 0;
    uint32_t blockNum = 0;
    (void)CcuResSpecifications::GetInstance(userDevId_).GetWqeBBNum(dieId_, wqeBBNum);
    (void)CcuResSpecifications::GetInstance(userDevId_).GetJettyNum(dieId_, jettyNum);
    if (jettyNum == 0) {
        HCCL_WARNING("[CcuWqeBBMgrV2][%s] failed, jettyNum is [%u].", __func__, jettyNum);
        return HcclResult::HCCL_E_UNAVAIL;
    }

    blockNum = wqeBBNum / CCU_V2_FIXED_SQ_SIZE;
    allocatedWqeBBBlocks_.resize(blockNum, false);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuWqeBBMgrV2::Alloc(const WqeBBReq& wqeBBReq, ResInfo& wqeBBInfo)
{
    wqeBBInfo.startId = wqeBBReq.jettyCtxStartId * CCU_V2_FIXED_SQ_SIZE;
    wqeBBInfo.num = CCU_V2_FIXED_SQ_SIZE;
    // 检查是否可用，防止重新分配
    if (wqeBBReq.jettyCtxStartId >= allocatedWqeBBBlocks_.size()) {
        HCCL_ERROR(
            "[CcuWqeBBMgrV2][%s] failed, jetty ctx start id[%u] is not expected, "
            "more than specNum[%zu].",
            __func__, wqeBBReq.jettyCtxStartId, allocatedWqeBBBlocks_.size());
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (allocatedWqeBBBlocks_[wqeBBReq.jettyCtxStartId]) {
        HCCL_ERROR(
            "[CcuWqeBBMgrV2][%s] failed, the jetty ctx id[%u] has been allocated.", __func__, wqeBBReq.jettyCtxStartId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    allocatedWqeBBBlocks_[wqeBBReq.jettyCtxStartId] = true;
    HCCL_INFO(
        "[CcuWqeBBMgrV2][%s] jettyCtxStartId[%u], wqeBBInfo.startId[%u], wqeBBInfo.num[%u]", __func__,
        wqeBBReq.jettyCtxStartId, wqeBBInfo.startId, wqeBBInfo.num);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuWqeBBMgrV2::Release(const ResInfo& wqeBBInfo)
{
    auto startId = wqeBBInfo.startId / CCU_V2_FIXED_SQ_SIZE;
    if (startId >= allocatedWqeBBBlocks_.size()) {
        HCCL_ERROR(
            "[CcuWqeBBMgrV2][%s] failed, jetty ctx start id[%u] is not expected, "
            "more than specNum[%zu].",
            __func__, startId, allocatedWqeBBBlocks_.size());
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (!allocatedWqeBBBlocks_[startId]) {
        HCCL_ERROR(
            "[CcuWqeBBMgrV2][%s] failed, jetty ctx start id[%u] is not allocated.", __func__, startId,
            allocatedWqeBBBlocks_.size());
        return HcclResult::HCCL_E_PARA;
    }

    allocatedWqeBBBlocks_[startId] = false;
    return HcclResult::HCCL_SUCCESS;
}

}; // namespace hcomm
