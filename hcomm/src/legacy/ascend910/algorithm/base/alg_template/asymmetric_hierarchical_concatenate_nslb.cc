/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_ahc_base_pub.h"
#include <iostream>
#include <fstream>

namespace hccl {

HcclResult
CommAHCBaseInfo::GetNBNslbDstRanks(const u32 rank, const std::vector<u32> commGroups, std::vector<u32>& dstRanks) const
{
    CHK_PRT_RET(
        rank >= commGroups.size(),
        HCCL_ERROR(
            "[CalcNBTransportReq][CalcDstRanks] rank [%u] exceed commGroups Size [%u] error", rank, commGroups.size()),
        HCCL_E_INTERNAL);

    for (auto i = 0; static_cast<u32>(1 << i) < commGroups.size(); ++i) {
        // 正方向第2^i个节点的rank号
        const u32 targetRankPos = static_cast<u32>(rank + (1 << i)) % commGroups.size();
        dstRanks.push_back(commGroups[targetRankPos]);

        // 反方向第2^i个节点的rank号
        const u32 targetRankNeg = static_cast<u32>(rank + commGroups.size() - (1 << i)) % commGroups.size();

        HCCL_DEBUG(
            "[CalcNBTransportReq][CalcDstRanks] local rank[%u], remote rank[%u]", commGroups[rank],
            commGroups[targetRankNeg]);

        dstRanks.push_back(commGroups[targetRankNeg]);
    }

    return HCCL_SUCCESS;
}

HcclResult
CommAHCBaseInfo::GetNHRNslbDstRanks(const u32 rank, const std::vector<u32> commGroups, std::vector<u32>& dstRanks) const
{
    CHK_PRT_RET(
        rank >= commGroups.size(),
        HCCL_ERROR(
            "[CalcNHRTransportReq][CalcDstRanks] rank [%u] exceed commGroups Size [%u] error", rank, commGroups.size()),
        HCCL_E_INTERNAL);

    for (auto i = 0; static_cast<u32>(1 << i) < commGroups.size(); ++i) {
        // 正方向第2^i个节点的rank号
        const u32 targetRankPos = static_cast<u32>(rank + (1 << i)) % commGroups.size();
        dstRanks.push_back(commGroups[targetRankPos]);

        // 反方向第2^i个节点的rank号
        const u32 targetRankNeg = static_cast<u32>(rank + commGroups.size() - (1 << i)) % commGroups.size();

        HCCL_DEBUG(
            "[CalcNHRTransportReq][CalcDstRanks] local rank[%u], remote rank[%u]", commGroups[rank],
            commGroups[targetRankNeg]);

        dstRanks.push_back(commGroups[targetRankNeg]);
    }

    return HCCL_SUCCESS;
}

HcclResult CommAHCBaseInfo::GetRingNslbDstRanks(
    const u32 rank, const std::vector<u32> commGroups, std::vector<u32>& dstRanks) const
{
    CHK_PRT_RET(
        rank >= commGroups.size(),
        HCCL_ERROR(
            "[CalcRingTransportReq][CalcDstRanks] rank [%u] exceed commGroups Size [%u] error", rank,
            commGroups.size()),
        HCCL_E_INTERNAL);

    // 正方向下一个节点的rank号
    const u32 targetRankPos = static_cast<u32>(rank + 1) % commGroups.size();
    dstRanks.push_back(commGroups[targetRankPos]);

    // 反方向下一个节点的rank号
    const u32 targetRankNeg = static_cast<u32>(rank + commGroups.size() - 1) % commGroups.size();

    HCCL_DEBUG(
        "[CalcRingTransportReq][CalcDstRanks] local rank[%u], remote rank[%u]", commGroups[rank],
        commGroups[targetRankNeg]);

    dstRanks.push_back(commGroups[targetRankNeg]);

    return HCCL_SUCCESS;
}

HcclResult CommAHCBaseInfo::GetDstRanksByType(
    AHCTemplateType type, const u32 rank, const std::vector<u32> commGroups, std::vector<u32>& dstRanks)
{
    if (type == AHCTemplateType::AHC_TEMPLATE_NB) {
        CHK_RET(GetRingNslbDstRanks(rank, commGroups, dstRanks));
    }
    if (type == AHCTemplateType::AHC_TEMPLATE_NHR) {
        CHK_RET(GetRingNslbDstRanks(rank, commGroups, dstRanks));
    }
    if (type == AHCTemplateType::AHC_TEMPLATE_NHR) {
        CHK_RET(GetRingNslbDstRanks(rank, commGroups, dstRanks));
    }
    return HCCL_SUCCESS;
}

} // namespace hccl
