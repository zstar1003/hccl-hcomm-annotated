/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADDRESS_INFO_H
#define ADDRESS_INFO_H

#include <set>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"
#include "ip_address.h"
#include "topo_common_types.h"
#include "rank_table_source.h"

namespace Hccl {

constexpr unsigned int MIN_VALUE_PLANEID = 0;
constexpr unsigned int MAX_VALUE_PLANEID = 1024;
constexpr unsigned int MIN_VALUE_PORT = 1;
constexpr unsigned int MAX_VALUE_PORT = 16;
constexpr unsigned int MIN_VALUE_PORT_LENGTH = 1;
constexpr unsigned int MAX_VALUE_PORT_LENGTH = 32;
constexpr unsigned int MIN_VALUE_ADDR_LENGRH = 1;
constexpr unsigned int MAX_VALUE_ADDR_LENGRH = 256;
constexpr unsigned int MAX_VALUE_BACKUP_ADDR_SIZE = 16;

class AddressInfo {
public:
    AddressInfo() {};
    ~AddressInfo() {};

    IpAddress addr;
    std::vector<IpAddress> backupAddrs;
    u32 socketPort_{0}; // socket监听端口
    AddrType addrType;
    std::set<std::string> ports; // 网口标记
    std::string planeId{"0"};
    void Deserialize(const nlohmann::json& addressInfoJson, RankTableSource source = RankTableSource::RANKTABLE);
    explicit AddressInfo(BinaryStream& binStream);
    void GetBinStream(BinaryStream& binStream) const;
    std::string Describe() const;
    static void ParseAddrByType(const std::string& addrType, const std::string& address, IpAddress& ipAddress);
    static void ParseBackupAddrs(
        const nlohmann::json& backupAddrJson, const std::string& addrType, std::vector<IpAddress>& backupAddrs);

private:
    static constexpr int MAX_DISPLAY_LEN = 128;
    static const std::unordered_map<std::string, AddrType> strToAddrType;
    void EidToAddr(std::string str);
    void IPV4ToAddr(std::string str);
    void IPV6ToAddr(std::string str);
    void DeserializeBackupAddrs(const nlohmann::json& addressInfoJson, const std::string& addrTypeStr);
    void DeserializeAddrTypeAndAddr(const nlohmann::json& addressInfoJson, RankTableSource source);
    void DeserializePorts(const nlohmann::json& addressInfoJson, RankTableSource source);
    static bool IsStringInAddrType(std::string str) { return strToAddrType.count(str) > 0; }
};

} // namespace Hccl

#endif // ADDRESS_INFO_H
