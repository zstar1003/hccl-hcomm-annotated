/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EDGE_INFO_H
#define EDGE_INFO_H

#include <string>
#include <unordered_map>
#include <set>
#include "nlohmann/json.hpp"
#include "types.h"
#include "topo_common_types.h"
#include "ip_address.h"

namespace Hccl {

constexpr unsigned int MAX_VALUE_LEVEL = 7;
constexpr u32 PORT_MAX_LENGTH = 32;
constexpr u32 MAX_PORTS_SIZE = 64;

class EdgeInfo {
public:
    EdgeInfo() {};
    EdgeInfo(BinaryStream& binaryStream);
    ~EdgeInfo() {};

    u32 netLayer{0};
    LinkType linkType;
    TopoType topoType{TopoType::CLOS};
    u32 topoInstId{0};
    std::set<LinkProtocol> protocols;

    u32 localA{0};
    std::set<std::string> localAPorts;
    u32 localB{0};
    std::set<std::string> localBPorts;
    AddrPosition position;

    bool operator==(const EdgeInfo& other) const;
    void Deserialize(const nlohmann::json& edgeInfoJson);
    LinkProtocol GetLinkProtocol(std::string str) const;
    TopoType GetTopoType(std::string topoTypeStr) const;
    LinkType GetLinkType(std::string linkTypeStr) const;
    AddrPosition GetAddrPosition(std::string str) const;
    std::string Describe() const;
    void GetBinStream(BinaryStream& binaryStream) const;

private:
    const static std::unordered_map<std::string, LinkProtocol> strToLinkProtocol;
    const static std::unordered_map<std::string, TopoType> strToTopoType;
    const static std::unordered_map<std::string, LinkType> strToLinkType;
    const static std::unordered_map<std::string, AddrPosition> strToAddrPosition;
    bool CompareEndpoints(const EdgeInfo& other) const;
    void DeserializeProtocol(const nlohmann::json& edgeInfoJson);
    void DeserializeEndpoint(const nlohmann::json& edgeInfoJson);
    void DeserializePort(const nlohmann::json& edgeInfoJson, std::string propName, std::set<std::string>& ports);
    std::string DescribePorts(std::set<std::string> ports) const;
};
} // namespace Hccl

#endif // EDGE_INFO_H
