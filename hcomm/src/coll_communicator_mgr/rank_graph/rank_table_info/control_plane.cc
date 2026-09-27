/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include "json_parser.h"
#include "invalid_params_exception.h"
#include "exception_util.h"
#include "control_plane.h"
#include "topo_common_types.h"
#include "adapter_error_manager_pub.h"
#include "rank_table_report_macro.h"
namespace Hccl {
using namespace std;

const unordered_map<string, AddrType> ControlPlane::strToAddrType
    = (unordered_map<string, AddrType>{{"EID", AddrType::EID}, {"IPV4", AddrType::IPV4}, {"IPV6", AddrType::IPV6}});

void ControlPlane::Deserialize(const nlohmann::json& controlPlaneJson, RankTableSource source)
{
    string addrTypeStr;
    std::string msgAddrtype = "error occurs when parser object of propName \"addr_type\"";

    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgAddrtype, (addrTypeStr = GetJsonProperty(controlPlaneJson, "addr_type")),
        controlPlaneJson, "addr_type", "non-empty string", source);

    if (!IsStringInAddrType(addrTypeStr)) {
        THROW<InvalidParamsException>(StringFormat("[ControlPlane::%s] failed with Invalid addrType. ", __func__));
    }
    addrType = strToAddrType.at(addrTypeStr);

    std::string address;
    std::string msgAddr = "error occurs when parser object of propName \"addr\"";
    const int MAX_DISPLAY_LEN = 128;
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgAddr, (address = GetJsonProperty(controlPlaneJson, "addr")), controlPlaneJson,
        "addr", "non-empty string", source);

    if (address.length() < MIN_VALUE_ADDR || address.length() > MAX_VALUE_ADDR) {
        THROW<InvalidParamsException>(StringFormat(
            "addr [%.*s] length is out of range [%u] to [%u]", MAX_DISPLAY_LEN, address.c_str(), MIN_VALUE_ADDR,
            MAX_VALUE_ADDR));
    }

    if (addrTypeStr == "IPV4") {
        IPV4ToAddr(address);
    } else if (addrTypeStr == "IPV6") {
        IPV6ToAddr(address);
    } else if (addrTypeStr == "EID") {
        EidToAddr(address);
    }

    std::string msgListenport = "error occurs when parser object of propName \"listen_port\"";
    TRY_CATCH_THROW_REPORT(
        InvalidParamsException, msgListenport, (listenPort = GetJsonPropertyUInt(controlPlaneJson, "listen_port")),
        controlPlaneJson, "listen_port", "0 ~ UINT32_MAX", source);

    if (listenPort < MIN_VALUE_LISTENPORT || listenPort > MAX_VALUE_LISTENPORT) {
        THROW<InvalidParamsException>(StringFormat(
            "listen_port [%u] length is out of range [%u] to [%u]", listenPort, MIN_VALUE_LISTENPORT,
            MAX_VALUE_LISTENPORT));
    }
}

void ControlPlane::EidToAddr(std::string address)
{
    if (address.length() != URMA_EID_LEN * URMA_EID_NUM_TWO) {
        THROW<InvalidParamsException>(
            StringFormat("[ControlPlane::%s] failed with rankAddrs : error in length. ", __func__));
    } else if (!IpAddress::IsEID(address)) {
        THROW<InvalidParamsException>(
            StringFormat("[ControlPlane::%s] failed with rankAddrs : error in format. ", __func__));
    }
    Eid eid = IpAddress::StrToEID(address);
    IpAddress ipAddress0(eid);
    addr = ipAddress0;
}

void ControlPlane::IPV4ToAddr(std::string address)
{
    s32 ipFamily = AF_INET;

    if (IpAddress::IsIPv4(address)) {
        ipFamily = AF_INET;
    } else {
        THROW<InvalidParamsException>(StringFormat("[ControlPlane::%s] failed with addrs is error. ", __func__));
    }
    IpAddress ipAddress0(address, ipFamily);
    addr = ipAddress0;
}

void ControlPlane::IPV6ToAddr(std::string address)
{
    s32 ipFamily = AF_INET6;

    if (IpAddress::IsIPv6(address)) {
        ipFamily = AF_INET6;
    } else {
        THROW<InvalidParamsException>(StringFormat("[ControlPlane::%s] failed with addr is error. ", __func__));
    }
    IpAddress ipAddress0(address, ipFamily);
    addr = ipAddress0;
}

std::string ControlPlane::Describe() const
{
    return StringFormat(
        "ControlPlane[addrType=%s, addr=%s, listenPort=%u]", addrType.Describe().c_str(), addr.Describe().c_str(),
        listenPort);
}

void ControlPlane::GetBinStream(BinaryStream& binStream) const
{
    binStream << static_cast<u32>(addrType) << listenPort;
    addr.GetBinStream(binStream);
}

ControlPlane::ControlPlane(BinaryStream& binStream)
{
    u32 addrTypeInt{0};
    binStream >> addrTypeInt;
    addrType = static_cast<AddrType::Value>(addrTypeInt);
    binStream >> listenPort;
    IpAddress address(binStream);
    addr = address;
}

} // namespace Hccl
