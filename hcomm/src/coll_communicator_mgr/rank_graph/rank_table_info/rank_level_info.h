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
#include "rank_table_source.h"
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
    // 无UB场景兜底标记：由rank_info_detect_client插入兜底level0时置true（JSON字段pcie_fallback），
    // 表示该层为合成的PCIe兜底层而非真实UB mesh层；正常层缺省false
    bool pcieFallback{false};
    std::vector<AddressInfo> rankAddrs;
    std::string Describe() const;
    std::map<std::string, std::vector<IpAddress>> portAddrMap;
    void Deserialize(const nlohmann::json& rankLevelInfoJson, RankTableSource source = RankTableSource::RANKTABLE);
    explicit RankLevelInfo(BinaryStream& binaStream);
    void GetBinStream(BinaryStream& binaStream) const;

private:
    static const std::unordered_map<std::string, NetType> strToNetType;
    void DeserializeNetLayerInfo(const nlohmann::json& rankLevelInfoJson, RankTableSource source);
    void DeserializeRankAddrs(const nlohmann::json& rankLevelInfoJson, RankTableSource source);
    void BuildPortAddrMap();
    static bool IsStringInNetType(std::string str) { return strToNetType.count(str) > 0; }
};

} // namespace Hccl

#endif // RANK_LEVEL_INFO_H
