/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_STUB_TRANSPORT_H
#define TESTS_STUB_TRANSPORT_H

#include "transport_pub.h"
#include "coll_alg_param.h"

namespace hccl {
class TransportCompared {
public:
    void Describe() { HCCL_DEBUG("TODO"); };
    u32 GetTransportId() { return transportId_; }
    HcclResult SetTransportId(u32 transportId)
    {
        transportId_ = transportId;
        return HCCL_SUCCESS;
    }

    RankId localRank = 0;
    RankId remoteRank = 0;
    bool isValid = false;
    u32 transportId_;
    bool isCompared = false;
    TransportMemType inputMemType = TransportMemType::RESERVED;
    TransportMemType outputMemType = TransportMemType::RESERVED;
    TransportMemType remoteinputMemType = TransportMemType::RESERVED;
    TransportMemType remoteoutputMemType = TransportMemType::RESERVED;
};
} // namespace hccl

#endif
