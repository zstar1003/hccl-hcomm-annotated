/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANK_LEVEL_INFO_H
#define RANK_LEVEL_INFO_H

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include <regex>
#include "topo_common_types.h"
#include "address_info.h"

namespace Hccl {
constexpr unsigned int MIN_VALUE_NETLAYER = 0;
constexpr unsigned int MAX_VALUE_NETLAYER = 7;
constexpr unsigned int MIN_VALUE_NETID = 1;
constexpr unsigned int MAX_VALUE_NETID = 1024;
constexpr unsigned int MIN_VALUE_RANKADDR_SIZE = 0;
constexpr unsigned int MAX_VALUE_RANKADDR_SIZE = 24;
constexpr unsigned int MIN_VALUE_U32 = 0;

class RankLevelInfo {
public:
    RankLevelInfo() {};
    u32 netLayer{0};
    std::string netInstId;
    NetType netType{NetType::CLOS};
    std::string netAttr;
    std::vector<AddressInfo> rankAddrs;
    std::string Describe() const;
    std::map<std::string, std::vector<IpAddress>> portAddrMap;
    void Deserialize(const nlohmann::json& rankLevelInfoJson);
    explicit RankLevelInfo(BinaryStream& binaStream);
    void GetBinStream(BinaryStream& binaStream) const;

private:
    static const std::unordered_map<std::string, NetType> strToNetType;

    static bool IsStringInNetType(std::string str) { return strToNetType.count(str) > 0; }
};

} // namespace Hccl

#endif // RANK_LEVEL_INFO_H
