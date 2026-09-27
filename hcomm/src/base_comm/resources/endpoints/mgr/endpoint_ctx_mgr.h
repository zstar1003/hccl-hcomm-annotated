/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_ENDPOINT_CTX_MGR_H
#define HCOMM_ENDPOINT_CTX_MGR_H

#include <memory>
#include <mutex>
#include <unordered_map>
#include "port.h"
#include "hcomm_res_defs.h"
#include "rdma_handle_manager.h"

namespace hcomm {

/**
 * @note 职责：EndpointCtx 去重 key
 */
struct EndpointCtxKey {
    u32 devPhyId{0};
    CommProtocol protocol{COMM_PROTOCOL_RESERVED};
    EndpointLocType locType{ENDPOINT_LOC_TYPE_RESERVED};
    Hccl::IpAddress ip{}; // HOST 侧为原始 IP；DEVICE 侧 UB 为 EID 地址；UBOE 入参为原始 IPv4、缓存键为转换后 EID
    bool operator==(const EndpointCtxKey& o) const
    {
        return devPhyId == o.devPhyId && protocol == o.protocol && locType == o.locType && ip == o.ip;
    }
};

struct EndpointCtxKeyHash {
    size_t operator()(const EndpointCtxKey& k) const
    {
        return Hccl::HashCombine({
            std::hash<u32>{}(k.devPhyId),
            std::hash<int>{}(static_cast<int>(k.protocol)),
            std::hash<int>{}(static_cast<int>(k.locType)),
            std::hash<Hccl::IpAddress>{}(k.ip),
        });
    }
};

/**
 * @note 职责：保存 ctxHandle（RdmaHandle），以 EndpointCtxKey 为去重 key。
 *       无 refCount（由 shared_ptr 自动管理）。多个 endpoint 实例可共享同一 EndpointCtx。
 */
struct EndpointCtx {
    void* ctxHandle{nullptr};
    EndpointCtxKey key;
};

/**
 * @note 职责：EndpointCtx 去重缓存管理层（key→ctxHandle_），仅管理 endpointCtxMap_。
 *       endpoint 句柄表（handle→Endpoint）由同目录 EndpointMgr 管理，两者各持一把锁。
 *       作为 HcommBaseResMgr 成员 per-device 持有（同 devPhyId 的 ctx 天然隔离）：
 *       经 HcommResMgr::GetInstance().GetDeviceResMgr(devPhyId).GetEndpointCtxMgr() 访问；
 *       进程销毁兜底：~HcommResMgr() 调用 DeInit()，遍历 endpointCtxMap_ 强制释放残留。
 */
class EndpointCtxMgr {
public:
    EndpointCtxMgr() = default;
    ~EndpointCtxMgr() = default; // 兜底由 ~HcommResMgr() 调 DeInit() 承担（成员化后析构序确定）

    EndpointCtxMgr(const EndpointCtxMgr&) = delete;
    EndpointCtxMgr& operator=(const EndpointCtxMgr&) = delete;

    // --- EndpointCtx 去重（RdmaHandleManager 调用收口于此）---
    // 统一获取入口，按 key.locType 分支（key 内含 locType，与去重 key 同源）：
    // - ENDPOINT_LOC_TYPE_HOST：走 RdmaHandleManager::GetByAddr（PortDeploymentType 固定 HOST_NET），
    //   承载迁移前 cpu_roce/cpu_urma 语义（cpu_urma 为 RDMA→RDMA 链路的 UB PEER 模式）
    // - ENDPOINT_LOC_TYPE_DEVICE：走 RdmaHandleManager::GetByIp(devPhyId, key.ip)（HDC 模式），
    //   承载迁移前 urma/ub_rtp 语义；isUboeIp=true 表示 key.ip 为 UBOE 原始 IPv4，
    //   先 UboeIpv4ToEid 转换为 EID 再查（转换后 EID 进缓存键，与 rdmaHandleMap 键同构）
    HcclResult Acquire(const EndpointCtxKey& key, bool isUboeIp, std::shared_ptr<EndpointCtx>& endpointCtx);

    // 调用方须先释放（reset）自身持有的 EndpointCtx 引用再调用——use_count 判定依赖此前提；
    // 仅剩 map 引用时移除条目，条目析构在锁外执行
    void Release(const EndpointCtxKey& key);

    // 进程销毁兜底：遍历 endpointCtxMap_ 强制释放残留
    void DeInit();

private:
    // 缓存命中校验，调用方须持有 mtx_
    bool FindValidCachedLocked(
        Hccl::RdmaHandleManager& rdmaHandleMgr, const EndpointCtxKey& key, std::shared_ptr<EndpointCtx>& endpointCtx);
    // 按 locType 分发底层句柄查询，调用方须持有 mtx_
    HcclResult
    AcquireHandleLocked(Hccl::RdmaHandleManager& rdmaHandleMgr, const EndpointCtxKey& key, void*& ctxHandle) const;

    std::mutex mtx_; // 仅护 endpointCtxMap_；与 EndpointMgr::mtx_ 无嵌套（锁外析构纪律保证无反向持锁）
    std::unordered_map<EndpointCtxKey, std::shared_ptr<EndpointCtx>, EndpointCtxKeyHash> endpointCtxMap_;
};

} // namespace hcomm

#endif // HCOMM_ENDPOINT_CTX_MGR_H
