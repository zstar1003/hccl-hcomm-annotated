/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENDPOINT_MGR_H
#define ENDPOINT_MGR_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include "endpoint.h"
#include "../endpoint_pairs/endpoint_pair.h"
#include "hccl_mem_defs.h"

namespace hccl {

/**
 * @brief endpoint 粒度的 tag→handle 映射。
 *        持有 EndpointHandle，析构时自动解注册全部 tag 对应内存。
 */
class TaggedMemMap {
public:
    explicit TaggedMemMap(EndpointHandle handle) : handle_(handle) {}
    ~TaggedMemMap();

    TaggedMemMap(const TaggedMemMap&) = delete;
    TaggedMemMap& operator=(const TaggedMemMap&) = delete;
    TaggedMemMap(TaggedMemMap&&) = default;
    TaggedMemMap& operator=(TaggedMemMap&&) = default;

    uint64_t GetVersion() const { return version_; }
    void SetVersion(uint64_t ver) { version_ = ver; }

    MemHandle FindHandle(const std::string& tag) const;
    bool HasTag(const std::string& tag) const;
    void EmplaceHandle(const std::string& tag, MemHandle handle);
    MemHandle RemoveTag(const std::string& tag);

private:
    EndpointHandle handle_;
    uint64_t version_{0};
    std::unordered_map<std::string, MemHandle> tagToHandle_;
};

/**
 * @note 职责：Endpoint管理器，支持不同类型的Endpoint的创建和销毁管理。
 */
class EndpointMgr {
public:
    EndpointMgr() {};
    ~EndpointMgr();

    // 获取端点
    HcclResult Get(EndpointDesc epDesc, EndpointHandle& handle);

    // 获取端点（按 sharedQueueTag 区分）：共享 jetty 场景下不同 tag 创建独立 Endpoint，
    // 实现不同 tag 隔离底层 jetty 资源。tag 为空时退化为 Get（兼容非共享路径）。
    HcclResult GetWithTag(EndpointDesc epDesc, const std::string& sharedQueueTag, EndpointHandle& handle);

    // 注册内存到端点，若 commMemsVersion 与上次注册时一致则跳过
    HcclResult RegisterMemory(
        EndpointHandle epHandle, const std::vector<std::string>& memTag, const std::vector<HcclMem>& memVec,
        uint64_t commMemsVersion);

    // 查询指定 endpoint 下若干 tag 对应的 MemHandle
    HcclResult GetMemHandlesByTags(
        EndpointHandle epHandle, const std::vector<std::string>& memTags, std::vector<MemHandle>& memHandleVec);

    // 从所有 endpoint 中删除指定 tag 并调 HcommMemUnreg
    HcclResult UnregMemByTag(const std::string& tag);

private:
    bool IsDescExist(EndpointDesc epDesc);

    // 共享 jetty 场景按 sharedQueueTag 区分的 Endpoint 映射 key： (EndpointDesc, tag)。
    // 字段级 hash/compare，规避 EndpointDesc padding 字段未初始化导致的误判（与 EndpointDescPairHash 同思路）。
    struct EndpointDescTagKey {
        EndpointDesc desc;
        std::string tag;
    };
    struct EndpointDescTagHash {
        std::size_t operator()(const EndpointDescTagKey& k) const noexcept
        {
            std::string buf;
            buf.append(reinterpret_cast<const char*>(&k.desc.protocol), sizeof(k.desc.protocol));
            buf.append(reinterpret_cast<const char*>(&k.desc.commAddr.type), sizeof(k.desc.commAddr.type));
            buf.append(reinterpret_cast<const char*>(k.desc.commAddr.raws), sizeof(k.desc.commAddr.raws));
            buf.append(reinterpret_cast<const char*>(&k.desc.loc.locType), sizeof(k.desc.loc.locType));
            buf.append(reinterpret_cast<const char*>(k.desc.loc.raws), sizeof(k.desc.loc.raws));
            buf.append(k.tag);
            return std::hash<std::string>{}(buf);
        }
    };
    struct EndpointDescTagEqual {
        bool operator()(const EndpointDescTagKey& a, const EndpointDescTagKey& b) const noexcept
        {
            return a.tag == b.tag && a.desc.protocol == b.desc.protocol && a.desc.commAddr.type == b.desc.commAddr.type
                   && std::memcmp(a.desc.commAddr.raws, b.desc.commAddr.raws, sizeof(a.desc.commAddr.raws)) == 0
                   && a.desc.loc.locType == b.desc.loc.locType
                   && std::memcmp(a.desc.loc.raws, b.desc.loc.raws, sizeof(a.desc.loc.raws)) == 0;
        }
    };

    // 共享 jetty 场景按 sharedQueueTag 区分的 Endpoint 映射。
    // 同一 EndpointDesc + 不同 tag → 不同 EndpointHandle → 不同底层 jetty 资源。
    // tag 为空时不进入此 map，退化为 endpointMap_ 行为（兼容非共享路径）。
    std::unordered_map<EndpointDescTagKey, EndpointHandle, EndpointDescTagHash, EndpointDescTagEqual>
        taggedEndpointMap_{};

    std::unordered_map<EndpointDesc, EndpointHandle> endpointMap_{};
    std::unordered_map<EndpointHandle, TaggedMemMap> endpointTagMemMap_{};
    std::mutex mutex_{};
};

} // namespace hccl

#endif // ENDPOINT_MGR_H
