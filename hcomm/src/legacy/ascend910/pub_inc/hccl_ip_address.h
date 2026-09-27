/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_IP_ADDRESS_H
#define HCCL_IP_ADDRESS_H

#include <securec.h>
#include <arpa/inet.h>
#include <hccl/base.h>
#include <string>
#include <vector>

namespace hccl {
constexpr u32 IP_ADDRESS_BUFFER_LEN = 64;

struct HcclSocketInfo {
    void* socketHandle; /**< socket handle */
    void* fdHandle;     /**< fd handle */
};

constexpr uint32_t URMA_EID_LEN = 16;
constexpr uint32_t URMA_EID_NUM_TWO = 2;
constexpr uint32_t MAX_IPV4_LEN = 15;            // 最大IPv4地址长度
constexpr uint32_t MIN_IPV4_LEN = 7;             // 最小IPv4地址长度
constexpr uint32_t BASE = 10;                    // 进制基数
constexpr uint32_t MAX_DOT_COUNT = 3;            // IPv4地址.分割符的最大个数
constexpr uint32_t MAX_IPV4_SEGMENT_VALUE = 255; // 每个段的最大值
constexpr uint32_t URMA_EID_IPV4_PREFIX = 0x0;

union Eid {
    uint8_t raw[URMA_EID_LEN]{0};
    struct {
        uint64_t reserved;
        uint32_t prefix;
        uint32_t addr;
    } in4;
    struct {
        uint64_t subnetPrefix;
        uint64_t interfaceId;
    } in6;

    std::string Describe() const;

    bool operator==(const Eid& other) const { return memcmp(raw, other.raw, URMA_EID_LEN) == 0; }

    bool operator<(const Eid& other) const { return memcmp(raw, other.raw, URMA_EID_LEN) < 0; }
};

union HcclInAddr {
    struct in_addr addr;
    struct in6_addr addr6;
};

class HcclIpAddress {
public:
    explicit HcclIpAddress()
    {
        scopeID = 0;
        family = AF_INET;
        binaryAddr.addr.s_addr = 0;
        readableIP = "0.0.0.0";
        readableAddr = readableIP;
    }
    explicit HcclIpAddress(const Eid& eidInput);

    explicit HcclIpAddress(u32 address)
    {
        union HcclInAddr ipAddr;
        ipAddr.addr.s_addr = address;
        (void)SetBianryAddress(AF_INET, ipAddr);
    }
    explicit HcclIpAddress(s32 family, const union HcclInAddr& address) { (void)SetBianryAddress(family, address); }
    explicit HcclIpAddress(const struct in_addr& address)
    {
        union HcclInAddr ipAddr;
        ipAddr.addr = address;
        (void)SetBianryAddress(AF_INET, ipAddr);
    }
    explicit HcclIpAddress(const struct in6_addr& address)
    {
        union HcclInAddr ipAddr;
        ipAddr.addr6 = address;
        (void)SetBianryAddress(AF_INET6, ipAddr);
    }
    explicit HcclIpAddress(const std::string& address) { (void)SetReadableAddress(address); }
    ~HcclIpAddress() {}

    static bool IsIPv6(const std::string& str);
    static bool IsIPv4(const std::string& str);
    static bool IsEID(const std::string& str);
    static Eid StrToEID(const std::string& str);
    std::string GetIpStr() const;
    Eid GetEid() const { return eid; }
    std::string Describe() const;

    std::string GetIfName() const { return ifname; }
    HcclResult SetScopeID(s32 scopeID)
    {
        this->scopeID = scopeID;
        return HCCL_SUCCESS;
    }
    s32 GetScopeID() const { return scopeID; }
    s32 GetFamily() const { return family; }

    const char* GetReadableIP() const
    {
        // return "IP address (string)"
        return readableIP.c_str();
    }
    const char* GetReadableAddress() const
    {
        // return "IP address (string) % ifname"
        return readableAddr.c_str();
    }
    union HcclInAddr GetBinaryAddress() const { return binaryAddr; }
    bool IsIPv6() const { return (family == AF_INET6); }
    void clear()
    {
        family = AF_INET;
        scopeID = 0;
        binaryAddr.addr.s_addr = 0;
        readableAddr.clear();
        readableIP.clear();
        ifname.clear();
    }
    bool IsInvalid() const { return ((family == AF_INET) && (binaryAddr.addr.s_addr == 0)); }
    bool operator==(const HcclIpAddress& that) const
    {
        if (this->family != that.family) {
            return false;
        }
        if (this->family == AF_INET) {
            return (this->binaryAddr.addr.s_addr == that.binaryAddr.addr.s_addr);
        } else {
            if (memcmp(&this->binaryAddr.addr6, &that.binaryAddr.addr6, sizeof(this->binaryAddr.addr6)) != 0) {
                return false;
            } else {
                if (this->ifname.empty() || that.ifname.empty()) {
                    return true;
                } else {
                    return (this->ifname == that.ifname);
                }
            }
        }
    }

    bool operator!=(const HcclIpAddress& that) const { return !(*this == that); }
    bool operator<(const HcclIpAddress& that) const
    {
        if (this->family < that.family) {
            return true;
        }
        if (this->family > that.family) {
            return false;
        }
        return (this->family == AF_INET) ? (this->binaryAddr.addr.s_addr < that.binaryAddr.addr.s_addr) :
                                           (this->readableAddr < that.readableAddr);
    }

    HcclResult SetReadableAddress(const std::string& address);
    HcclResult SetIfName(const std::string& name);

private:
    HcclResult SetBianryAddress(s32 family, const union HcclInAddr& address);

    union HcclInAddr binaryAddr {}; // 二进制IP地址
    std::string readableAddr{};     // 字符串IP地址 + % + 网卡名
    std::string readableIP{};       // 字符串IP地址
    std::string ifname{};           // 网卡名
    s32 family{};
    s32 scopeID{};
    Eid eid{};
};
} // namespace hccl
#endif // HCCL_IP_ADDRESS_H
