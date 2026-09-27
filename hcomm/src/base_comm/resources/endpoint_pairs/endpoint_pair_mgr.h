/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENDPOINT_PAIR_MGR_H
#define ENDPOINT_PAIR_MGR_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <mutex>
#include "endpoint_pair.h"

namespace hcomm {

using EpChannelMap = std::unordered_map<EndpointDescPair, std::unordered_map<CommEngine, std::vector<ChannelHandle>>>;
class EndpointPairMgr {
public:
    EndpointPairMgr() {};

    ~EndpointPairMgr() = default;

    HcclResult
    Get(const EndpointDescPair& endpointDescPair, EndpointPair*& out, const Hccl::RankIpPortMapPtr& rankIpPortMap);
    EpChannelMap GetEpChannelMap();

private:
    std::mutex mapMtx_;
    std::unordered_map<EndpointDescPair, std::unique_ptr<EndpointPair>> endpointPairMap_{};
};

} // namespace hcomm

#endif // ENDPOINT_PAIR_MGR_H
