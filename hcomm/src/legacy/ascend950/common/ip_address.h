/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_IP_ADDRESS_H
#define HCCLV2_IP_ADDRESS_H

#include <arpa/inet.h>
#include <cstring>
#include <string>
#include <vector>

#include "hccl/base.h"
#include "string_util.h"
#include "not_support_exception.h"
#include "exception_util.h"
#include "invalid_params_exception.h"
#include "hash_utils.h"
#include "binary_stream.h"
#include "internal_exception.h"

namespace Hccl {
using namespace std;

constexpr uint32_t URMA_EID_LEN = 16;
constexpr uint32_t URMA_EID_NUM_TWO = 2;
constexpr uint32_t MAX_IPV4_LEN = 15;            // 最大IPv4地址长度
constexpr uint32_t MIN_IPV4_LEN = 7;             // 最小IPv4地址长度
constexpr uint32_t BASE = 10;                    // 进制基数
constexpr uint32_t MAX_DOT_COUNT = 3;            // IPv4地址.分割符的最大个数
constexpr uint32_t MAX_IPV4_SEGMENT_VALUE = 255; // 每个段的最大值
constexpr uint32_t URMA_EID_IPV4_PREFIX = 0x0;
constexpr uint32_t EID_WORD_SHIFT_0 = 48; // EID 64位值中第0个16位字的右移位数
constexpr uint32_t EID_WORD_SHIFT_1 = 32; // EID 64位值中第1个16位字的右移位数
constexpr uint32_t EID_WORD_SHIFT_2 = 16; // EID 64位值中第2个16位字的右移位数

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

    string Describe() const
    {
        uint64_t subnet = be64toh(in6.subnetPrefix);
        uint64_t ifId = be64toh(in6.interfaceId);
        return StringFormat(
            "%04llx:%04llx:%04llx:%04llx:%04llx:%04llx:%04llx:%04llx",
            static_cast<unsigned long long>((subnet >> EID_WORD_SHIFT_0) & 0xFFFF),
            static_cast<unsigned long long>((subnet >> EID_WORD_SHIFT_1) & 0xFFFF),
            static_cast<unsigned long long>((subnet >> EID_WORD_SHIFT_2) & 0xFFFF),
            static_cast<unsigned long long>(subnet & 0xFFFF),
            static_cast<unsigned long long>((ifId >> EID_WORD_SHIFT_0) & 0xFFFF),
            static_cast<unsigned long long>((ifId >> EID_WORD_SHIFT_1) & 0xFFFF),
            static_cast<unsigned long long>((ifId >> EID_WORD_SHIFT_2) & 0xFFFF),
            static_cast<unsigned long long>(ifId & 0xFFFF));
    }

    bool operator==(const Eid& that) const { return memcmp(&raw, &that.raw, sizeof(raw)) == 0; }

    bool operator<(const Eid& that) const { return memcmp(&raw, &that.raw, sizeof(raw)) < 0; }

    Eid Reversed() const
    {
        Eid eidOut;
        for (uint32_t i = 0; i < URMA_EID_LEN; i++) {
            eidOut.raw[i] = raw[URMA_EID_LEN - i - 1];
        }
        return eidOut;
    }
};

union BinaryAddr {
    struct in_addr addr;
    struct in6_addr addr6;
};
class IpAddress {
public:
    IpAddress()
    {
        scopeID_ = 0;
        family_ = AF_INET;
        binaryAddr_.addr.s_addr = 0;
    }

    explicit IpAddress(const string& ip, s32 family = AF_INET) : family_(family) { InitBinaryAddr(ip); }

    explicit IpAddress(const union BinaryAddr& ip, s32 family, const uint8_t* eid) : family_(family)
    {
        binaryAddr_ = ip;
        if (eid != nullptr) {
            // 安全复制原始EID
            s32 sRet = memcpy_s(eid_.raw, sizeof(eid_.raw), eid, URMA_EID_LEN);
            if (sRet != 0) {
                THROW<InternalException>("[IpAddress]memcpy_s failed when setting original EID");
            }
        }
    }

    explicit IpAddress(const union BinaryAddr& ip, s32 family, s32 scopeID = 0) : family_(family), scopeID_(scopeID)
    {
        binaryAddr_ = ip;
        // 区分ipv4和ipv6转eid
        if (family_ == AF_INET6) {
            s32 sRet
                = memcpy_s(eid_.raw, sizeof(eid_.raw), binaryAddr_.addr6.s6_addr, sizeof(binaryAddr_.addr6.s6_addr));
            if (sRet != 0) {
                THROW<InternalException>("[IpAddress]memcpy_s failed");
            }
        } else {
            ipv4AddrToEid(binaryAddr_.addr.s_addr);
        }
    }

    explicit IpAddress(u32 address)
    {
        struct in_addr addr {
            address
        };
        family_ = AF_INET;
        binaryAddr_.addr = addr;
        ipv4AddrToEid(address);
    }

    explicit IpAddress(std::vector<char>& uniqueId) // 基于序列化数据得到IpAddress
    {
        char dst[INET6_ADDRSTRLEN]{0};
        BinaryStream binaryStream(uniqueId);
        binaryStream >> family_;
        binaryStream >> scopeID_;
        binaryStream >> dst;

        std::string ip = dst;
        InitBinaryAddr(ip);
        binaryStream >> eid_.raw; // 恢复eid.raw，覆盖eid
    }
    explicit IpAddress(const Eid& eidInput)
    {
        for (uint32_t i = 0; i < URMA_EID_LEN; i++) {
            eid_.raw[i] = eidInput.raw[i];
        }
        HCCL_INFO("[IpAddress] %s", eid_.Describe().c_str());
        // IPoURMA适配后，使用EID初始化时转为ipv6建链
        family_ = AF_INET6;
        (void)memcpy_s(binaryAddr_.addr6.s6_addr, sizeof(eid_.raw), eid_.raw, sizeof(eid_.raw));
    }

    std::vector<char> GetUniqueId() const // 获取序列化数据
    {
        std::string ipStr = GetIpStr();
        char dst[INET6_ADDRSTRLEN]{0};
        int sret = strcpy_s(dst, sizeof(dst), ipStr.data());
        if (sret != 0) {
            auto msg = StringFormat(
                "[Get][UniqueId]errNo[0x%016llx] memory copy failed. ret[%d]",
                HCOM_ERROR_CODE(HcclResult::HCCL_E_MEMORY), sret);
            THROW<InternalException>(msg);
        }
        BinaryStream binaryStream;
        binaryStream << family_;
        binaryStream << scopeID_;
        binaryStream << dst;
        binaryStream << eid_.raw; // 保存eid.raw
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void SetScopeID(s32 scope) { this->scopeID_ = scope; }

    s32 GetScopeID() const { return scopeID_; }

    s32 GetFamily() const { return family_; }

    union BinaryAddr GetBinaryAddress() const { return binaryAddr_; }

    bool IsIPv6() const { return (family_ == AF_INET6); }

    /*The following IPV6 formats can be verified:
        dotted quad at the end, multiple zeroes collapsed: fe80::204:61ff:254.157.241.86
        collapse multiple zeroes to :: in the IPv6 address: fe80::204:61ff:fe9d:f156
        full form of IPv6: fe80:0000:0000:0000:0204:61ff:fe9d:f156
        drop leading zeroes, IPv4 dotted quad at the end: fe80:0:0:0:0204:61ff:254.157.241.86
        drop leading zeroes: fe80:0:0:0:204:61ff:fe9d:f156
        IPv4 dotted quad at the end: fe80:0000:0000:0000:0204:61ff:254.157.241.86
        global unicast prefix: 2001::
        link-local prefix: fe80::
        localhost: ::1
    */
    static bool IsIPv6(const string& str)
    {
        if (str.find('\0') != string::npos) {
            return false;
        }
        struct in6_addr ipv6Addr {};
        return inet_pton(AF_INET6, str.c_str(), &ipv6Addr) == 1;
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
    static bool IsIPv4(const std::string& str)
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

    static bool IsEID(const string& str)
    {
        if (str.length() != URMA_EID_LEN * URMA_EID_NUM_TWO) {
            return false;
        }

        for (char ch : str) {
            const bool isDigit = ch >= '0' && ch <= '9';
            const bool isLowerHex = ch >= 'a' && ch <= 'f';
            const bool isUpperHex = ch >= 'A' && ch <= 'F';
            if (!isDigit && !isLowerHex && !isUpperHex) {
                return false;
            }
        }
        return true;
    }

    static Eid StrToEID(const string& str)
    {
        Eid tmpeEid{};
        const int Base = 16;
        for (size_t i = 0; i < URMA_EID_LEN; ++i) {
            std::string byteString = str.substr(i * 2, 2);
            tmpeEid.raw[i] = static_cast<uint8_t>(std::stoi(byteString, nullptr, Base));
        }
        return tmpeEid;
    }

    string GetIpStr() const
    {
        const void* src = nullptr;
        if (family_ == AF_INET) {
            src = &binaryAddr_.addr;
        } else if (family_ == AF_INET6) {
            src = &binaryAddr_.addr6;
        } else {
            THROW<NotSupportException>(StringFormat("Unsupported Address Family: %d", family_));
        }
        char dst[INET6_ADDRSTRLEN];
        const char* res = inet_ntop(family_, src, dst, INET6_ADDRSTRLEN);
        if (res == nullptr) {
            THROW<InvalidParamsException>("Invalid Binary Network Address");
        }
        return dst;
    }

    Eid GetEid() const { return eid_; }

    void SetEid(Eid eid) { eid_ = eid; }

    Eid GetReverseEid() const { return eid_.Reversed(); }

    string Describe() const
    {
        string desc = StringFormat("IpAddress[%s, ", eid_.Describe().c_str());

        if (family_ == AF_INET) {
            desc += StringFormat("AF=IPv4, addr=%s]", GetIpStr().c_str());
        } else {
            desc += StringFormat("AF=IPv6, addr=%s, scopeId=0x%x]", GetIpStr().c_str(), scopeID_);
        }
        return desc;
    }

    bool operator==(const IpAddress& that) const
    {
        if (this->family_ != that.family_) {
            return false;
        }
        if (memcmp(&this->eid_.raw, &that.eid_.raw, sizeof(this->eid_.raw)) != 0) {
            return false;
        }
        return true;
    }

    bool operator<(const IpAddress& that) const
    {
        if (this->family_ < that.family_) {
            return true;
        }
        if (that.family_ < this->family_) {
            return false;
        }
        if (memcmp(&this->eid_.raw, &that.eid_.raw, sizeof(this->eid_.raw)) < 0) {
            return true;
        }
        return false;
    }

    explicit IpAddress(BinaryStream& binaryStream) // 基于序列化数据得到IpAddress
    {
        binaryStream >> family_ >> scopeID_;
        // 打印family_、scopeID_
        HCCL_INFO("[IpAddress::%s] family_[%d], scopeID_[%d]", __func__, family_, scopeID_);
        char dst[INET6_ADDRSTRLEN]{0};
        binaryStream >> dst;
        std::string ip = dst;
        // 打印ip
        HCCL_INFO("[IpAddress::%s] ip_[%s]", __func__, ip.c_str());
        InitBinaryAddr(ip);
        binaryStream >> eid_.raw; // 恢复eid.raw，覆盖eid
    }

    void GetBinStream(BinaryStream& binaryStream) const
    {
        std::string ipStr = GetIpStr();
        char dst[INET6_ADDRSTRLEN]{0};
        int sret = strcpy_s(dst, sizeof(dst), ipStr.data());
        if (sret != 0) {
            auto msg = StringFormat(
                "[Get][UniqueId]errNo[0x%016llx] memory copy failed. ret[%d]",
                HCOM_ERROR_CODE(HcclResult::HCCL_E_MEMORY), sret);
            THROW<InternalException>(msg);
        }
        binaryStream << family_ << scopeID_ << dst;
        binaryStream << eid_.raw; // 保存eid.raw
    }

    bool IsInvalid() const { return ((family_ == AF_INET) && (binaryAddr_.addr.s_addr == 0)); }

private:
    union BinaryAddr binaryAddr_ {}; // 二进制IP地址
    s32 family_{AF_INET};
    s32 scopeID_{0};
    Eid eid_{};
    void InitBinaryAddr(const string& ip)
    {
        void* dst;
        int cnt = std::count(ip.begin(), ip.end(), ':');
        if (cnt >= 2) { // ipv6地址中至少有2个":"
            family_ = AF_INET6;
            dst = &binaryAddr_.addr6;
        } else {
            family_ = AF_INET;
            dst = &binaryAddr_.addr;
        }
        int res = inet_pton(family_, ip.c_str(), dst);
        if (res == -1) {
            THROW<NotSupportException>(StringFormat("Unsupported Address Family: %d", family_));
        } else if (res == 0) {
            THROW<InvalidParamsException>(StringFormat("Invalid Network Address: %s", ip.c_str()));
        }

        if (family_ == AF_INET6) {
            s32 sRet
                = memcpy_s(eid_.raw, sizeof(eid_.raw), binaryAddr_.addr6.s6_addr, sizeof(binaryAddr_.addr6.s6_addr));
            if (sRet != 0) {
                THROW<InternalException>("[InitBinaryAddr]memcpy_s failed");
            }
        } else {
            ipv4AddrToEid(binaryAddr_.addr.s_addr);
        }
    }

    void ipv4AddrToEid(const uint32_t& inAddr)
    {
        eid_.in4.reserved = 0;
        eid_.in4.prefix = URMA_EID_IPV4_PREFIX;
        eid_.in4.addr = inAddr;
    }
};
} // namespace Hccl

namespace std {

template <>
class equal_to<Hccl::Eid> {
public:
    bool operator()(const Hccl::Eid& p1, const Hccl::Eid& p2) const { return p1 == p2; }
};

template <>
class hash<Hccl::Eid> {
public:
    size_t operator()(const Hccl::Eid& eid) const
    {
        auto subnetPrefixHash = hash<uint64_t>{}(be64toh(eid.in6.subnetPrefix));
        auto interfaceIdHash = hash<uint64_t>{}(be64toh(eid.in6.interfaceId));
        return Hccl::HashCombine({subnetPrefixHash, interfaceIdHash});
    }
};

template <>
class equal_to<Hccl::IpAddress> {
public:
    bool operator()(const Hccl::IpAddress& p1, const Hccl::IpAddress& p2) const { return p1 == p2; }
};

template <>
class hash<Hccl::IpAddress> {
public:
    size_t operator()(const Hccl::IpAddress& ip) const
    {
        auto scopeIDHash = hash<s32>{}(ip.GetScopeID());
        auto familyHash = hash<s32>{}(ip.GetFamily());
        auto addrHash = hash<size_t>{}(ip.GetBinaryAddress().addr.s_addr); // Ipv4地址hash
        auto eidSubnetPrefix = hash<uint64_t>{}(ip.GetEid().in6.subnetPrefix);
        auto eidInterfaceId = hash<uint64_t>{}(ip.GetEid().in6.interfaceId);

        return Hccl::HashCombine({scopeIDHash, familyHash, addrHash, eidSubnetPrefix, eidInterfaceId});
    }
};
} // namespace std

#endif // HCCLV2_IP_ADDRESS_H
