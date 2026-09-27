/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INNER_NET_DEVICE_H
#define HCCLV2_INNER_NET_DEVICE_H

#include <mutex>
#include <unordered_map>
#include <functional>
#include <tuple>
#include <type_traits>
#include "port.h"
#include "ip_address.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"
#include "buffer_key.h"
#include "tokenInfo_manager.h"

namespace Hccl {
struct NetDevInfo {
    bool operator==(const NetDevInfo& rhs) const
    {
        return rankId == rhs.rankId && type == rhs.type && devId == rhs.devId && addr == rhs.addr
               && protoType == rhs.protoType;
    }

    bool operator!=(const NetDevInfo& rhs) const { return !(rhs == *this); }

    bool operator<(const NetDevInfo& rhs) const
    {
        if (rankId < rhs.rankId) {
            return true;
        }
        if (rhs.rankId < rankId) {
            return false;
        }
        if (type < rhs.type) {
            return true;
        }
        if (rhs.type < type)
            return false;
        if (addr < rhs.addr) {
            return true;
        }
        if (rhs.addr < addr) {
            return false;
        }
        return devId < rhs.devId;
    }

    bool operator>(const NetDevInfo& rhs) const { return rhs < *this; }

    bool operator<=(const NetDevInfo& rhs) const { return !(rhs < *this); }

    bool operator>=(const NetDevInfo& rhs) const { return !(*this < rhs); }

    RankId rankId;
    PortDeploymentType type;
    LinkProtoType protoType;
    u32 devId;
    IpAddress addr;
};

class InnerNetDev {
public:
    InnerNetDev(const NetDevInfo& info);
    ~InnerNetDev();
    // RdmaHandle 读写函数
    RdmaHandle getRdmaHandle() const { return rdmaHandle_; }
    void setRdmaHandle(const RdmaHandle& handle) { rdmaHandle_ = handle; }

    // HrtUbJfcMode 读写函数
    HrtUbJfcMode getUbMode() const { return ubMode_; }
    void setUbMode(HrtUbJfcMode mode) { ubMode_ = mode; }

    // JfcHandle 读写函数
    JfcHandle getUbJfcHandle(HrtUbJfcMode jfcMode);
    void setUbJfcHandle(const JfcHandle& handle) { ubJfcHandle_ = handle; }

    // dieId_ 读写函数
    uint32_t getDieId() const { return dieId_; }
    void setDieId(uint32_t id) { dieId_ = id; }

    // funcId_ 读写函数
    uint32_t getFuncId() const { return funcId_; }
    void setFuncId(uint32_t id) { funcId_ = id; }

    // TokenIdHandle 读写函数
    TokenIdHandle getTokenHandle() const { return tokenHandle_; }
    void setTokenHandle(const TokenIdHandle& handle) { tokenHandle_ = handle; }

    // tokenId_ 读写函数
    uint32_t getTokenId() const { return tokenId_; }
    void setTokenId(uint32_t id) { tokenId_ = id; }

    bool GetIsValid() const { return isValid_; }

    std::pair<TokenIdHandle, uint32_t> getTokenIdInfo(const BufferKey<uintptr_t, u64>& bufKey);

    void putTokenIdInfo(const BufferKey<uintptr_t, u64>& bufKey, TokenIdHandle tokenIdHandle);

private:
    RdmaHandle rdmaHandle_{nullptr};
    HrtUbJfcMode ubMode_;
    JfcHandle ubJfcHandle_{0};
    uint32_t dieId_{0};
    uint32_t funcId_{0};
    TokenIdHandle tokenHandle_{0};
    uint32_t tokenId_{0};
    LinkProtoType localProto_;
    HrtNetworkMode netMode_{HrtNetworkMode::HDC};

    std::unique_ptr<TokenInfoManager> tokenInfoManager_{nullptr};
    bool isValid_{true};
};

} // namespace Hccl

namespace std {
template <>
struct hash<Hccl::PortDeploymentType> {
    size_t operator()(const Hccl::PortDeploymentType& val) const
    {
        // 直接指定底层类型为 int（如果确认枚举底层是 int）
        return hash<int>()(static_cast<int>(val));
    }
};

template <>
struct hash<Hccl::LinkProtoType> {
    size_t operator()(const Hccl::LinkProtoType& val) const { return hash<int>()(static_cast<int>(val)); }
};

const u64 HASH_NET_DEV = 0x9e3779b9;
const u32 HASH_SEED_LEFT_BIT = 6;
const u32 HASH_SEED_RIGHT_BIT = 2;

template <>
struct hash<Hccl::NetDevInfo> {
    size_t operator()(const Hccl::NetDevInfo& obj) const
    {
        // 组合各个成员的哈希值
        size_t hash1 = hash<Hccl::RankId>()(obj.rankId);
        size_t hash2 = hash<Hccl::PortDeploymentType>()(obj.type);
        size_t hash3 = hash<Hccl::LinkProtoType>()(obj.protoType);
        size_t hash4 = hash<u32>()(obj.devId);
        size_t hash5 = hash<Hccl::IpAddress>()(obj.addr); // 假设IpAddress有可用的hash

        // 改进的哈希组合方式，减少冲突
        size_t seed = 0;
        seed ^= hash1 + HASH_NET_DEV + (seed << HASH_SEED_LEFT_BIT) + (seed >> HASH_SEED_RIGHT_BIT);
        seed ^= hash2 + HASH_NET_DEV + (seed << HASH_SEED_LEFT_BIT) + (seed >> HASH_SEED_RIGHT_BIT);
        seed ^= hash3 + HASH_NET_DEV + (seed << HASH_SEED_LEFT_BIT) + (seed >> HASH_SEED_RIGHT_BIT);
        seed ^= hash4 + HASH_NET_DEV + (seed << HASH_SEED_LEFT_BIT) + (seed >> HASH_SEED_RIGHT_BIT);
        seed ^= hash5 + HASH_NET_DEV + (seed << HASH_SEED_LEFT_BIT) + (seed >> HASH_SEED_RIGHT_BIT);
        return seed;
    }
};
} // namespace std

#endif // HCCLV2_INNER_NET_DEVICE_H
