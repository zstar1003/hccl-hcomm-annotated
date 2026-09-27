/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NEW_TOPO_INFO_H
#define NEW_TOPO_INFO_H

#include <map>
#include <vector>
#include <unordered_set>

#include "peer_info.h"
#include "edge_info.h"

namespace Hccl {

constexpr u32 MAX_PEER_COUNT = 513;
class TopoInfo {
public:
    TopoInfo() {};
    std::string version;
    u32 peerCount{0};
    u32 edgeCount{0};
    std::vector<PeerInfo> peers;
    std::map<u32, std::vector<EdgeInfo>> edges;
    void Deserialize(const nlohmann::json& topoInfoJson);
    std::string Describe() const;
    void Dump() const;
    TopoInfo(BinaryStream& binaryStream);
    void GetBinStream(BinaryStream& binaryStream) const;

private:
    void DeserializePeers(const nlohmann::json& topoInfoJson);
    void DeserializeEdges(const nlohmann::json& topoInfoJson);
    void VerifyEdges(EdgeInfo& edge);
    unordered_set<u32> idSet;
};
} // namespace Hccl
#endif // NEW_TOPO_INFO_H
