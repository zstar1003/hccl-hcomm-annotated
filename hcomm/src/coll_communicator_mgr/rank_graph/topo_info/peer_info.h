/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NEW_PEER_INFO_H
#define NEW_PEER_INFO_H

#include "types.h"
#include "binary_stream.h"
#include "nlohmann/json.hpp"

namespace Hccl {

constexpr u32 MAX_PEER_LOCAL_ID = 512;
class PeerInfo {
public:
    PeerInfo() {};
    ~PeerInfo() {};
    u32 localId{0};
    void Deserialize(const nlohmann::json& peerInfoJson);
    std::string Describe() const;
    explicit PeerInfo(BinaryStream& binaryStream);
    void GetBinStream(BinaryStream& binaryStream) const;
};
} // namespace Hccl

#endif // NEW_PEER_INFO_H
