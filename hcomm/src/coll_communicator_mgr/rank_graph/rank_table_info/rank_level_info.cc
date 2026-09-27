/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_level_info.h"
#include <sstream>
#include <string>
#include <unordered_map>
#include "json_parser.h"
#include "invalid_params_exception.h"
#include "exception_util.h"
#include "adapter_error_manager_pub.h"
#include "rank_table_report_macro.h"

namespace Hccl {
using namespace std;

const unordered_map<string, NetType> RankLevelInfo::strToNetType = (unordered_map<string, NetType>{
    {"1DMESH", NetType::MESH_1D},
    {"2DMESH", NetType::MESH_2D},
    {"A3_SERVER", NetType::A3_SERVER},
    {"A2_AX_SERVER", NetType::A2_AX_SERVER},
    {"TOPO_FILE_DESC", NetType::TOPO_FILE_DESC},
    {"CLOS", NetType::CLOS}});

void RankLevelInfo::DeserializeNetLayerInfo(const nlohmann::json& rankLevelInfoJson, RankTableSource source)
{
    std::string msgNetlayer = "error occurs when parser object of propName \"net_layer\"";
    std::string msgNetinstid = "error occurs when parser object of propName \"net_instance_id\"";
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgNetlayer, (netLayer = GetJsonPropertyUInt(rankLevelInfoJson, "net_layer")),
        rankLevelInfoJson, "net_layer", "0 ~ UINT32_MAX", source);
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgNetinstid, (netInstId = GetJsonProperty(rankLevelInfoJson, "net_instance_id")),
        rankLevelInfoJson, "net_instance_id", "non-empty string", source);

    if (netLayer > MAX_VALUE_NETLAYER) {
        THROW<InvalidParamsException>(
            StringFormat("netLayer[%u] out of range [%u] to [%u]", netLayer, MIN_VALUE_U32, MAX_VALUE_NETLAYER));
    }
    if (netInstId.length() < MIN_VALUE_NETID || netInstId.length() > MAX_VALUE_NETID) {
        THROW<InvalidParamsException>(StringFormat(
            "netInstId length[%zu] out of range [%u] to [%u]", netInstId.length(), MIN_VALUE_NETID, MAX_VALUE_NETID));
    }

    netAttr = rankLevelInfoJson.value<std::string>("net_attr", "");
    // 兜底level0的显式标记，正常层无此字段缺省false
    pcieFallback = rankLevelInfoJson.value<bool>("pcie_fallback", false);

    if (rankLevelInfoJson.contains("net_type")) {
        string netTypeStr;
        std::string msgNettype = "error occurs when parser object of propName \"net_type\"";
        TRY_CATCH_THROW_REPORT(
            InvalidParamsException, msgNettype, (netTypeStr = GetJsonProperty(rankLevelInfoJson, "net_type")),
            rankLevelInfoJson, "net_type", "non-empty string", source);
        if (!IsStringInNetType(netTypeStr)) {
            THROW<InvalidParamsException>(StringFormat("[RankLevelInfo::%s] failed with Invalid netType. ", __func__));
        }
        netType = strToNetType.at(netTypeStr);
    }
}

void RankLevelInfo::DeserializeRankAddrs(const nlohmann::json& rankLevelInfoJson, RankTableSource source)
{
    nlohmann::json rank_addrs;
    std::string msgAddrs = "error occurs when parser object of propName \"rank_addrs\"";
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgAddrs, (GetJsonPropertyList(rankLevelInfoJson, "rank_addr_list", rank_addrs)),
        rankLevelInfoJson, "rank_addr_list", "array", source);
    for (auto& addr : rank_addrs) {
        AddressInfo addressInfo;
        addressInfo.Deserialize(addr, source);
        rankAddrs.emplace_back(addressInfo);
    }
    if (rankAddrs.size() > MAX_VALUE_RANKADDR_SIZE) {
        THROW<InvalidParamsException>(StringFormat(
            "rank_addr_list [%u] out of range [%u] to [%u]", rankAddrs.size(), MIN_VALUE_RANKADDR_SIZE,
            MAX_VALUE_RANKADDR_SIZE));
    }
}

void RankLevelInfo::BuildPortAddrMap()
{
    for (auto& rankAddr : rankAddrs) {
        IpAddress ipAddress = rankAddr.addr;
        for (auto& port : rankAddr.ports) {
            if (portAddrMap.find(port) != portAddrMap.end()) {
                portAddrMap[port].push_back(ipAddress);
            } else {
                std::vector<IpAddress> ipAddrList;
                ipAddrList.push_back(ipAddress);
                portAddrMap[port] = ipAddrList;
            }
        }
    }
}

void RankLevelInfo::Deserialize(const nlohmann::json& rankLevelInfoJson, RankTableSource source)
{
    DeserializeNetLayerInfo(rankLevelInfoJson, source);
    DeserializeRankAddrs(rankLevelInfoJson, source);
    BuildPortAddrMap();
}

string RankLevelInfo::Describe() const
{
    return StringFormat(
        "RankLevelInfo[net_layer=%u, net_instance_id=%s, netType=%s, rankAddrs size=%d]", netLayer, netInstId.c_str(),
        netType.Describe().c_str(), rankAddrs.size());
}

RankLevelInfo::RankLevelInfo(BinaryStream& binStream)
{
    binStream >> netLayer >> netInstId >> netAttr;
    u32 netTypeInt{0};
    binStream >> netTypeInt;
    netType = static_cast<NetType::Value>(netTypeInt);
    u32 pcieFallbackInt{0};
    binStream >> pcieFallbackInt;
    pcieFallback = (pcieFallbackInt != 0);
    size_t addrSize{0};
    binStream >> addrSize;
    HCCL_INFO(
        "[%s] net_layer[%u] net_instance_id[%s] netType[%s] addrs size[%u]", __func__, netLayer, netInstId.c_str(),
        netType.Describe().c_str(), rankAddrs.size());
    for (u32 i = 0; i < addrSize; i++) {
        AddressInfo addressInfo(binStream);
        rankAddrs.emplace_back(addressInfo);
    }

    for (auto& rankAddr : rankAddrs) {
        IpAddress ipAddress = rankAddr.addr;
        for (auto& port : rankAddr.ports) {
            if (portAddrMap.find(port) != portAddrMap.end()) {
                portAddrMap[port].push_back(ipAddress);
            } else {
                std::vector<IpAddress> ipAddrList;
                ipAddrList.push_back(ipAddress);
                portAddrMap[port] = ipAddrList;
            }
        }
    }
}

void RankLevelInfo::GetBinStream(BinaryStream& binStream) const
{
    binStream << netLayer << netInstId << netAttr << static_cast<u32>(netType);
    binStream << static_cast<u32>(pcieFallback);
    binStream << rankAddrs.size();
    HCCL_INFO(
        "[%s] net_layer[%u] net_instance_id[%s] netType[%s] addrs size[%u]", __func__, netLayer, netInstId.c_str(),
        netType.Describe().c_str(), rankAddrs.size());
    if (rankAddrs.size() == 0) {
        return;
    }
    for (auto& rankAddr : rankAddrs) {
        rankAddr.GetBinStream(binStream);
    }
}
} // namespace Hccl
