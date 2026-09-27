/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PROC_REGED_MEM_MGR_CACHE_H
#define PROC_REGED_MEM_MGR_CACHE_H

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "reged_mem_mgr.h"
#include "ip_address.h"
#include "port.h"
#include "hcomm_res_defs.h"

namespace hcomm {

struct MemMgrCacheKey {
    u32 devPhyId{0};
    CommProtocol protocol{COMM_PROTOCOL_ROCE};
    Hccl::IpAddress ip{};
    Hccl::PortDeploymentType portType{Hccl::PortDeploymentType::HOST_NET};

    bool operator==(const MemMgrCacheKey& other) const
    {
        return devPhyId == other.devPhyId && protocol == other.protocol && ip == other.ip && portType == other.portType;
    }
};

struct MemMgrCacheKeyHash {
    size_t operator()(const MemMgrCacheKey& k) const
    {
        return Hccl::HashCombine({
            std::hash<u32>{}(k.devPhyId),
            std::hash<int>{}(static_cast<int>(k.protocol)),
            std::hash<Hccl::IpAddress>{}(k.ip),
            std::hash<int>{}(static_cast<int>(k.portType)),
        });
    }
};

// 由 endpointDesc.loc.locType 推导 PortDeploymentType
inline Hccl::PortDeploymentType LocTypeToPortType(EndpointLocType locType)
{
    return (locType == ENDPOINT_LOC_TYPE_DEVICE) ? Hccl::PortDeploymentType::DEV_NET :
                                                   Hccl::PortDeploymentType::HOST_NET;
}

struct MemMgrEntry {
    std::shared_ptr<RegedMemMgr> mgrPtr{nullptr};
    u64 refCount{0};
};

/**
 * @note 进程级 RegedMemMgr 复用缓存。同一网卡跨 EndpointHandle 复用实例，跳过冗余硬件注册。
 *
 * 仍是单例。构造 private、禁止拷贝，GetHolder() 里 static shared_ptr 只 new 一次，
 * 之后每次调用返回同一对象的 shared_ptr 拷贝。变的是生命周期，不是实例个数。
 * 旧写法是 Meyers 单例（static T + GetInstance 返回 T&），寿命绑在静态析构上；
 * 静态对象先构造的晚析构。GetHolder 往往比其它静态对象更晚才第一次调用，
 * 退出时自己的 static 会先拆。Endpoint 多持有一份，就能活过那些先构造的静态对象的析构。
 *
 * 用法：
 * 1. Init：把 GetHolder() 存进成员，再用这份指针 GetOrCreate。不要把返回值当临时量用完即弃。
 * 2. Destroy / 析构：走持有的指针 Release，再 reset。
 * 3. 析构路径不要再调 GetHolder()。函数内那份 static shared_ptr 拆掉后，入口已悬空。
 */
class ProcRegedMemMgrCache {
public:
    static std::shared_ptr<ProcRegedMemMgrCache> GetHolder();

    // hit: refCount++ 返已有 shared_ptr; miss: 调 creator() 建实例 insert refCount=1
    std::shared_ptr<RegedMemMgr>
    GetOrCreate(const MemMgrCacheKey& key, std::function<std::shared_ptr<RegedMemMgr>()> creator);

    // refCount--, 归 0 则 erase cacheMap_ 条目
    void Release(const MemMgrCacheKey& key);

    ProcRegedMemMgrCache(const ProcRegedMemMgrCache&) = delete;
    ProcRegedMemMgrCache& operator=(const ProcRegedMemMgrCache&) = delete;
    // shared_ptr 默认删除器在类外 delete，析构必须可访问；构造仍 private，外部不能直接 new。
    ~ProcRegedMemMgrCache() = default;

private:
    ProcRegedMemMgrCache() = default;

    std::mutex mtx_;
    std::unordered_map<MemMgrCacheKey, MemMgrEntry, MemMgrCacheKeyHash> cacheMap_;
};

} // namespace hcomm

#endif // PROC_REGED_MEM_MGR_CACHE_H
