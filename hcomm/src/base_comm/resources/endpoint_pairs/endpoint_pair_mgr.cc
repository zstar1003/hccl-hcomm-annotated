/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "endpoint_pair_mgr.h"

namespace hcomm {

HcclResult EndpointPairMgr::Get(
    const EndpointDescPair& endpointDescPair, EndpointPair*& out, const Hccl::RankIpPortMapPtr& rankIpPortMap)
{
    {
        std::lock_guard<std::mutex> lock(mapMtx_);
        if (endpointPairMap_.find(endpointDescPair) != endpointPairMap_.end()) {
            out = endpointPairMap_[endpointDescPair].get();
            return HCCL_SUCCESS;
        }
    }

    std::unique_ptr<EndpointPair> endpointPair = nullptr;
    EXCEPTION_CATCH(
        (endpointPair = std::make_unique<EndpointPair>(endpointDescPair.first, endpointDescPair.second, rankIpPortMap)),
        return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(endpointPair);
    CHK_RET(endpointPair->Init());

    // 二次确认，防止并发修改
    {
        std::lock_guard<std::mutex> lock(mapMtx_);
        if (endpointPairMap_.find(endpointDescPair) != endpointPairMap_.end()) {
            out = endpointPairMap_[endpointDescPair].get();
            return HCCL_SUCCESS;
        }
        out = endpointPair.get();
        endpointPairMap_.emplace(endpointDescPair, std::move(endpointPair));
    }

    return HCCL_SUCCESS;
}

EpChannelMap EndpointPairMgr::GetEpChannelMap()
{
    std::lock_guard<std::mutex> lock(mapMtx_);
    EpChannelMap epChannelMap;
    for (const auto& endpointPair : endpointPairMap_) {
        auto channelList = endpointPair.second->GetChannelHandles();
        epChannelMap[endpointPair.first] = channelList;
    }
    return epChannelMap;
}

} // namespace hcomm
