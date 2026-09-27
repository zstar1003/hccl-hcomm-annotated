/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_channel_ctx_mgr.h"

#include "ccu_res_specs.h"

namespace hcomm {

static void DumpJettyCtxInfo(const JettyInfo& info)
{
    HCCL_RUN_INFO(
        "[CcuChannelCtxMgr][%s] local jetty context id[%u], ta jetty id[%u], "
        "ta jetty type[%s], sq depth[%u], wqe basic block start id[%u].",
        __func__, info.jettyCtxId, info.taJettyId, info.jettyType.Describe().c_str(), info.sqDepth, info.wqeBBStartId);

    if (info.jettyType == CcuJettyType::CCUM_CACHED_JETTY) {
        HCCL_RUN_INFO(
            "[CcuChannelCtxMgr][%s] sq buffer va[%llu], sq buffer size[%u].", __func__, info.sqBufVa, info.sqBufSize);
    }
}

void DumpChannelResInfo(const uint32_t feId, const ChannelInfo& info)
{
    const auto& jettyInfos = info.jettyInfos;
    HCCL_RUN_INFO(
        "[CcuChannelCtxMgr][%s]: fe id[%u], channel id[%u], die id[%u], "
        "used jetty num[%zu].",
        __func__, feId, info.channelId, info.dieId, jettyInfos.size());

    for (const auto& jettyInfo : jettyInfos) {
        DumpJettyCtxInfo(jettyInfo);
    }
}

bool IsEidEmpty(const uint8_t (&eidRaw)[URMA_EID_LEN])
{
    for (uint32_t i = 0; i < URMA_EID_LEN; i++) {
        if (eidRaw[i] != 0) {
            return false;
        }
    }

    return true;
}

bool CcuChannelCtxMgr::CheckIfChannelAllocated(const uint32_t channelId) const
{
    const uint32_t strategy = channelResInfos_.size();
    if (channelId >= strategy) {
        HCCL_ERROR(
            "[CcuChannelCtxMgr][%s] failed, channelId[%u] is invalid, "
            "should be less than the channel strategy[%u], userDevId[%d], dieId[%u].",
            __func__, channelId, strategy, userDevId_, dieId_);
        return false;
    }

    if (!channelResInfos_[channelId].allocated) {
        HCCL_ERROR(
            "[CcuChannelCtxMgr][%s] failed, channelId[%u] has not been "
            "allocated yet, userDevId[%d], dieId[%u].",
            __func__, channelId, userDevId_, dieId_);
        return false;
    }

    return true;
}

}; // namespace hcomm
