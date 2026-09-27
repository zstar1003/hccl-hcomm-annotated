/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_addr_logger.h"
#include "hccl_comm_pub.h"
#include <arpa/inet.h>
#include <array>
#include <securec.h>

namespace hcomm {
namespace logger {

    std::string CommAddrLogger::GetTypeString(CommAddrType type)
    {
        switch (type) {
            case COMM_ADDR_TYPE_IP_V4:
                return "IPv4";
            case COMM_ADDR_TYPE_IP_V6:
                return "IPv6";
            case COMM_ADDR_TYPE_ID:
                return "ID";
            case COMM_ADDR_TYPE_EID:
                return "EID";
            default:
                return "Unknown(" + std::to_string(type) + ")";
        }
    }

    std::string CommAddrLogger::ConvertIPv4(const struct in_addr& addr)
    {
        std::array<char, INET_ADDRSTRLEN> buffer;
        const char* result = inet_ntop(AF_INET, &addr, buffer.data(), buffer.size());
        if (result == nullptr) {
            HCCL_ERROR("[%s] inet_ntop failed for IPv4 address", __func__);
            return "conversion failed";
        }
        return std::string(result);
    }

    std::string CommAddrLogger::ConvertIPv6(const struct in6_addr& addr6)
    {
        std::array<char, INET6_ADDRSTRLEN> buffer;
        const char* result = inet_ntop(AF_INET6, &addr6, buffer.data(), buffer.size());
        if (result == nullptr) {
            HCCL_ERROR("[%s] inet_ntop failed for IPv6 address", __func__);
            return "conversion failed";
        }
        return std::string(result);
    }

    std::string CommAddrLogger::ConvertID(uint32_t id) { return "id:0x" + std::to_string(id); }

    std::string CommAddrLogger::ConvertEID(const uint8_t eid[16])
    {
        // 模仿 Eid::Describe 的格式
        // 输出：eid[xxxxxxxxxxxxxxxx:xxxxxxxxxxxxxxxx]
        // 其中前 8 字节是 subnetPrefix，后 8 字节是 interfaceId（网络字节序）

        // 从字节数组中提取两个 64 位整数（网络字节序）
        uint64_t subnetPrefix = 0;
        uint64_t interfaceId = 0;

        // 使用 memcpy 避免对齐问题
        (void)memcpy_s(&subnetPrefix, sizeof(subnetPrefix), eid, sizeof(subnetPrefix));
        (void)memcpy_s(&interfaceId, sizeof(interfaceId), eid + 8, sizeof(interfaceId));

        // 转换字节序（网络字节序 -> 主机字节序）
        subnetPrefix = be64toh(subnetPrefix);
        interfaceId = be64toh(interfaceId);

        char buffer[64];
        int32_t ret = snprintf_s(
            buffer, sizeof(buffer), sizeof(buffer) - 1, "eid[%016llx:%016llx]",
            static_cast<unsigned long long>(subnetPrefix), static_cast<unsigned long long>(interfaceId));
        if (ret < 0) {
            HCCL_ERROR("[%s] snprintf_s failed for EID", __func__);
            return "conversion failed";
        }
        return std::string(buffer);
    }

    std::string CommAddrLogger::ToString(const CommAddr& commAddr)
    {
        // 模仿 IpAddress::Describe 的格式
        // 输出：IpAddress[eid[xxxxxxxxxxxxxxxx:xxxxxxxxxxxxxxxx], AF=v4/v6, addr=xxx.xxx.xxx.xxx, scopeId=0x0]
        std::string desc = "IpAddress[";

        switch (commAddr.type) {
            case COMM_ADDR_TYPE_IP_V4:
                desc += ConvertEID(commAddr.eid) + ", AF=v4, addr=" + ConvertIPv4(commAddr.addr) + "]";
                break;
            case COMM_ADDR_TYPE_IP_V6:
                desc += ConvertEID(commAddr.eid) + ", AF=v6, addr=" + ConvertIPv6(commAddr.addr6) + ", scopeId=0x0]";
                break;
            case COMM_ADDR_TYPE_ID:
                desc += "id:" + ConvertID(commAddr.id) + "]";
                break;
            case COMM_ADDR_TYPE_EID:
                desc += ConvertEID(commAddr.eid) + "]";
                break;
            default:
                desc += "Unknown]";
                break;
        }

        return desc;
    }

    void CommAddrLogger::Print(uint32_t idx, const char* endpointName, const CommAddr& commAddr)
    {
        std::string addrStr = ToString(commAddr);
        HCCL_INFO(
            "[%s] channelDescs[%u] %s commAddr: type[%d], addr[%s]", __func__, idx, endpointName, commAddr.type,
            addrStr.c_str());
    }

} // namespace logger
} // namespace hcomm
