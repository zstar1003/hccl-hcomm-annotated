/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_IP_ADDRESS_H
#define SIM_IP_ADDRESS_H

#include <arpa/inet.h>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include "hccl/base.h"
#include "sim_common_defs.h"

namespace HcclSim {
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

    explicit IpAddress(const union BinaryAddr& ip, s32 family, s32 scopeID = 0) : family_(family), scopeID_(scopeID)
    {
        binaryAddr_ = ip;
        // 区分ipv4和ipv6转eid
        if (family_ == AF_INET6) {
            (void)memcpy(eid_.raw, binaryAddr_.addr6.s6_addr, sizeof(binaryAddr_.addr6.s6_addr));
        } else {
            ipv4AddrToEid(binaryAddr_.addr.s_addr);
        }
    }

    explicit IpAddress(u32 ipAddr)
    {
        struct in_addr addr {
            ipAddr
        };
        family_ = AF_INET;
        binaryAddr_.addr = addr;
        ipv4AddrToEid(ipAddr);
    }

    explicit IpAddress(const Eid& eidInput)
    {
        for (uint32_t i = 0; i < URMA_EID_LEN; i++) {
            eid_.raw[i] = eidInput.raw[i];
        }
        // IPoURMA适配后，使用EID初始化时转为ipv6建链
        family_ = AF_INET6;
        (void)memcpy(binaryAddr_.addr6.s6_addr, eid_.raw, sizeof(eid_.raw));
    }

    explicit IpAddress(const std::string& ip, s32 family = AF_INET) : family_(family)
    {
        void* dst;
        if (family != AF_INET && family != AF_INET6) {
            throw std::invalid_argument("[IpAddress] Unsupported Address Family");
        }
        if (family == AF_INET6) {
            dst = &binaryAddr_.addr6;
        } else {
            dst = &binaryAddr_.addr;
        }
        int res = inet_pton(family, ip.c_str(), dst);
        if (res == -1) {
            throw std::invalid_argument("[IpAddress] Unsupported Address Family");
        } else if (res == 0) {
            throw std::invalid_argument("[IpAddress] Invalid Network Address");
        }
        if (family_ == AF_INET6) {
            (void)memcpy(eid_.raw, binaryAddr_.addr6.s6_addr, sizeof(binaryAddr_.addr6.s6_addr));
        } else {
            ipv4AddrToEid(binaryAddr_.addr.s_addr);
        }
    }

    union BinaryAddr GetBinaryAddress() const { return binaryAddr_; }

    Eid GetEid() const { return eid_; }

    std::string EidToHexString() const
    {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (uint32_t i = 0; i < URMA_EID_LEN; i++) {
            oss << std::setw(2) << static_cast<uint32_t>(eid_.raw[i]);
        }
        return oss.str();
    }

    std::string GetIpStr() const
    {
        const void* src = nullptr;
        if (family_ == AF_INET) {
            src = &binaryAddr_.addr;
        } else if (family_ == AF_INET6) {
            src = &binaryAddr_.addr6;
        } else {
            throw std::invalid_argument("[IpAddress] Unsupported Address Family");
        }
        char dst[INET6_ADDRSTRLEN];
        const char* res = inet_ntop(family_, src, dst, INET6_ADDRSTRLEN);
        if (res == nullptr) {
            throw std::invalid_argument("[IpAddress] Invalid Binary Network Address");
        }
        return dst;
    }

    auto GetFamily() const { return family_; }

private:
    union BinaryAddr binaryAddr_ {}; // 二进制IP地址
    s32 family_{AF_INET};
    s32 scopeID_{0};
    Eid eid_{};

    void ipv4AddrToEid(const uint32_t& inAddr)
    {
        eid_.in4.reserved = 0;
        eid_.in4.prefix = URMA_EID_IPV4_PREFIX;
        eid_.in4.addr = inAddr;
    }
};
} // namespace HcclSim

#endif // SIM_IP_ADDRESS_H
