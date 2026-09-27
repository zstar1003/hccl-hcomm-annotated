/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LLT_HCCL_STUB_RANK_GRAPH_H
#define LLT_HCCL_STUB_RANK_GRAPH_H

#include <vector>
#include <hccl_types.h>
#include "rank_gph.h"

namespace hccl {

class RankGraphStub {
public:
    explicit RankGraphStub() = default;
    ~RankGraphStub() = default;
    std::shared_ptr<Hccl::RankGraph> Create2PGraph();
    std::shared_ptr<Hccl::RankGraph> Create1PGraph();

private:
    std::shared_ptr<Hccl::NetInstance::Peer>
    InitPeer(Hccl::RankId rankId, Hccl::LocalId localId, Hccl::DeviceId deviceId, uint32_t listenPort);
    std::shared_ptr<Hccl::NetInstance> InitNetInstance(uint32_t netLayer, std::string id);
    std::shared_ptr<Hccl::NetInstance::ConnInterface> InitConnInterface(Hccl::IpAddress addr);
};

} // namespace hccl

#endif
