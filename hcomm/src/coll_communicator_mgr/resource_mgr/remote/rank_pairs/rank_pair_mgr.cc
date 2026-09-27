/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_pair_mgr.h"

namespace hccl {

HcclResult RankPairMgr::Get(RankIdPair rankIdPair, RankPair*& out)
{
    std::unique_lock<std::shared_mutex> lock(mapMtx_);
    auto iterPtr = rankPairMap_.find(rankIdPair);
    if (iterPtr != rankPairMap_.end()) {
        out = iterPtr->second.get();
        return HCCL_SUCCESS;
    }

    std::unique_ptr<RankPair> rankPair = nullptr;
    EXCEPTION_CATCH((rankPair = std::make_unique<RankPair>(rankIdPair, rankIpPortMap_)), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(rankPair);
    CHK_RET(rankPair->Init());

    out = rankPair.get();
    rankPairMap_.emplace(rankIdPair, std::move(rankPair));

    return HCCL_SUCCESS;
}

ChannelTable RankPairMgr::GetChannelTable()
{
    std::shared_lock<std::shared_mutex> lock(mapMtx_);
    ChannelTable channelTable;
    for (const auto& rankPair : rankPairMap_) {
        channelTable[rankPair.first] = rankPair.second->GetEpChannelMap();
    }
    return channelTable;
}

HcclResult RankPairMgr::Find(RankIdPair rankIdPair, RankPair*& out)
{
    std::shared_lock<std::shared_mutex> lock(mapMtx_);
    auto iterPtr = rankPairMap_.find(rankIdPair);
    if (iterPtr != rankPairMap_.end()) {
        out = iterPtr->second.get();
        return HCCL_SUCCESS;
    }
    return HCCL_E_NOT_FOUND;
}

} // namespace hccl
