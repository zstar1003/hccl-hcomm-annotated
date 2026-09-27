/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <sstream>
#include <iomanip>
#include "hccl_ip_address.h"
#include <regex>
#include <log.h>
namespace hccl {
constexpr int MIN_IPV6_COLON_COUNT = 2;             // IPv6地址中至少包含的冒号个数
constexpr uint32_t URMA_EID_HEX_CHARS_PER_BYTE = 2; // EID十六进制字符串中每字节占用的字符数
constexpr uint32_t EID_DESCRIBE_HEX_WIDTH = 16;     // Eid::Describe中64位段对应的十六进制显示宽度

HcclResult HcclIpAddress::SetBianryAddress(s32 family, const union HcclInAddr& address)
{
    char buf[IP_ADDRESS_BUFFER_LEN] = {0};
    if (inet_ntop(family, &address, buf, sizeof(buf)) == nullptr) {
        if (family == AF_INET) {
            HCCL_ERROR("ip addr[0x%08x] is invalid IPv4 address.", address.addr.s_addr);
        } else {
            HCCL_ERROR(
                "ip addr[%08x %08x %08x %08x] is invalid IPv6 address.",
                address.addr6.s6_addr32[0],  // 打印ipv6地址中的 word 0
                address.addr6.s6_addr32[1],  // 打印ipv6地址中的 word 1
                address.addr6.s6_addr32[2],  // 打印ipv6地址中的 word 2
                address.addr6.s6_addr32[3]); // 打印ipv6地址中的 word 3
        }
        return HCCL_E_PARA;
    } else {
        this->family = family;
        this->binaryAddr = address;
        this->readableIP = buf;
        this->readableAddr = this->readableIP;
        return HCCL_SUCCESS;
    }
}

HcclResult HcclIpAddress::SetReadableAddress(const std::string& address)
{
    CHK_PRT_RET(address.empty(), HCCL_ERROR("ip addr is null."), HCCL_E_PARA);

    std::size_t found = address.find("%");
    if ((found == 0) || (found == (address.length() - 1))) {
        HCCL_ERROR("addr[%s] is invalid.", address.c_str());
        return HCCL_E_PARA;
    }
    std::string ipStr = address.substr(0, found);
    int cnt = std::count(ipStr.begin(), ipStr.end(), ':');
    if (cnt >= MIN_IPV6_COLON_COUNT) { // ipv6地址中至少有2个":"
        if (inet_pton(AF_INET6, ipStr.c_str(), &binaryAddr.addr6) <= 0) {
            HCCL_ERROR("ip addr[%s] is invalid IPv6 address.", ipStr.c_str());
            binaryAddr.addr6.s6_addr32[0] = 0; // 清空ipv6地址中的 word 0
            binaryAddr.addr6.s6_addr32[1] = 0; // 清空ipv6地址中的 word 1
            binaryAddr.addr6.s6_addr32[2] = 0; // 清空ipv6地址中的 word 2
            binaryAddr.addr6.s6_addr32[3] = 0; // 清空ipv6地址中的 word 3
            clear();
            return HCCL_E_PARA;
        }
        this->family = AF_INET6;
    } else {
        if (inet_pton(AF_INET, ipStr.c_str(), &binaryAddr.addr) <= 0) {
            HCCL_ERROR("ip addr[%s] is invalid IPv4 address.", ipStr.c_str());
            clear();
            return HCCL_E_PARA;
        }
        this->family = AF_INET;
    }
    if (found != std::string::npos) {
        this->ifname = address.substr(found + 1);
    }
    this->readableIP = ipStr;
    this->readableAddr = address;
    return HCCL_SUCCESS;
}

HcclResult HcclIpAddress::SetIfName(const std::string& name)
{
    CHK_PRT_RET(name.empty(), HCCL_ERROR("if name is null."), HCCL_E_PARA);

    std::size_t found = readableAddr.find("%");
    if (found == std::string::npos) {
        ifname = name;
        readableAddr.append("%");
        readableAddr.append(ifname);
    } else {
        HCCL_ERROR("addr[%s] ifname has existed.", readableAddr.c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

std::string HcclIpAddress::Describe() const
{
    std::ostringstream oss;
    oss << "IpAddress[" << eid.Describe() << ",";

    if (family == AF_INET) {
        oss << "AF=v4,addr=" << GetIpStr() << "]";
    } else {
        oss << "AF=v6,addr=" << GetIpStr() << ", scopeId=0x" << std::hex << scopeID << "]";
    }
    return oss.str();
}

HcclIpAddress::HcclIpAddress(const Eid& eidInput)
{
    for (uint32_t i = 0; i < URMA_EID_LEN; i++) {
        eid.raw[i] = eidInput.raw[i];
    }

    HCCL_INFO("[IpAddress] %s", eid.Describe().c_str());
    // IPoURMA适配后，使用EID初始化时转为ipv6建链
    this->family = AF_INET6;
    (void)memcpy_s(binaryAddr.addr6.s6_addr, sizeof(eid.raw), eid.raw, sizeof(eid.raw));
    (void)SetBianryAddress(family, binaryAddr);
}

bool HcclIpAddress::IsEID(const std::string& str)
{
    if (str.length() == URMA_EID_LEN * URMA_EID_NUM_TWO) {
        std::regex hexCharsRegex("[0-9a-fA-F]+");
        return std::regex_match(str, hexCharsRegex);
    }
    return false;
}

Eid HcclIpAddress::StrToEID(const std::string& str)
{
    Eid tmpeEid{};
    const int Base = 16;
    for (size_t i = 0; i < URMA_EID_LEN; ++i) {
        std::string byteString = str.substr(i * URMA_EID_HEX_CHARS_PER_BYTE, URMA_EID_HEX_CHARS_PER_BYTE);
        tmpeEid.raw[i] = static_cast<uint8_t>(std::stoi(byteString, nullptr, Base));
    }
    return tmpeEid;
}
std::string HcclIpAddress::GetIpStr() const
{
    const void* src = nullptr;
    if (family == AF_INET) {
        src = &binaryAddr.addr;
    } else if (family == AF_INET6) {
        src = &binaryAddr.addr6;
    }
    char dst[INET6_ADDRSTRLEN];
    const char* res = inet_ntop(family, src, dst, INET6_ADDRSTRLEN);
    if (res == nullptr) {
        // 转换失败处理：返回空字符串或抛异常
        return ""; // 示例
    }
    return dst;
}

bool HcclIpAddress::IsIPv6(const std::string& str)
{
    std::regex ipv6Pattern(
        R"(^([\da-fA-F]{1,4}:){6}((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$|^::([\da-fA-F]{1,4}:){0,4}((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$|^([\da-fA-F]{1,4}:):([\da-fA-F]{1,4}:){0,3}((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$|^([\da-fA-F]{1,4}:){2}:([\da-fA-F]{1,4}:){0,2}((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$|^([\da-fA-F]{1,4}:){3}:([\da-fA-F]{1,4}:){0,1}((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$|^([\da-fA-F]{1,4}:){4}:((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$|^([\da-fA-F]{1,4}:){7}[\da-fA-F]{1,4}$|^:((:[\da-fA-F]{1,4}){1,6}|:)$|^[\da-fA-F]{1,4}:((:[\da-fA-F]{1,4}){1,5}|:)$|^([\da-fA-F]{1,4}:){2}((:[\da-fA-F]{1,4}){1,4}|:)$|^([\da-fA-F]{1,4}:){3}((:[\da-fA-F]{1,4}){1,3}|:)$|^([\da-fA-F]{1,4}:){4}((:[\da-fA-F]{1,4}){1,2}|:)$|^([\da-fA-F]{1,4}:){5}:([\da-fA-F]{1,4})?$|^([\da-fA-F]{1,4}:){6}:$)");
    return regex_match(str, ipv6Pattern);
}

/*All the five types of IPV4 addresses,ABCDE,can be identified.
    A: 1.0.0.1 - 126.255.255.254
    B: 128.0.0.1 - 191.255.255.254
    C: 192.0.0.1 - 223.255.255.254
    D: 224.0.0.1 - 239.255.255.254
    E: 240.0.0.1 - 255.255.255.254
    127.x.x.x is reserved address for loopback test.
    0.0.0.0 can only be used as the source address.
    255.255.255.255 is broadcast address.
*/
bool HcclIpAddress::IsIPv4(const std::string& str)
{
    // 快速长度检查
    size_t len = str.length();
    if (len < MIN_IPV4_LEN || len > MAX_IPV4_LEN) {
        return false;
    }

    uint32_t num = 0;
    uint32_t dotCount = 0;
    bool hasDigit = false;

    for (size_t i = 0; i < len; ++i) {
        char c = str[i];

        if (c >= '0' && c <= '9') {
            // 检查前导零
            if (!hasDigit && c == '0' && i + 1 < len && str[i + 1] != '.') {
                return false;
            }

            num = num * BASE + (c - '0');
            hasDigit = true;

            if (num > MAX_IPV4_SEGMENT_VALUE) {
                return false;
            }
        } else if (c == '.') {
            // 检查点号位置和数字有效性
            if (!hasDigit || dotCount >= MAX_DOT_COUNT || i == 0 || i == len - 1) {
                return false;
            }

            dotCount++;
            num = 0;
            hasDigit = false;
        } else {
            return false;
        }
    }

    return dotCount == MAX_DOT_COUNT && hasDigit;
}

std::string Eid::Describe() const
{
    std::ostringstream oss;
    oss << "eid[" << std::hex << std::setw(EID_DESCRIBE_HEX_WIDTH) << std::setfill('0') << be64toh(in6.subnetPrefix)
        << ":" << std::hex << std::setw(EID_DESCRIBE_HEX_WIDTH) << std::setfill('0') << be64toh(in6.interfaceId) << "]";
    return oss.str();
}

} // namespace hccl
