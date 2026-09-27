/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_ENDPOINT_MGR_H
#define HCOMM_ENDPOINT_MGR_H

#include <memory>
#include <mutex>
#include <unordered_map>
#include "endpoint.h"
#include "hcomm_res_defs.h"

namespace hcomm {

/**
 * @note 职责：收编 HcommEndpointMap（handle→Endpoint）的全局句柄表管理层，仅管理 endpointMap_。
 *       EndpointCtx 去重缓存（key→ctxHandle_）已拆至同目录 EndpointCtxMgr（HcommBaseResMgr 成员，per-device）。
 *       作为 HcommResMgr 的成员：进程内全局唯一实例，
 *       经 HcommResMgr::GetInstance().GetEndpointMgr() 直接链路访问；
 *       进程销毁兜底：~HcommResMgr() 调用 DeInit()，遍历 endpointMap_ 强制释放残留。
 */
class EndpointMgr {
public:
    EndpointMgr() = default;
    ~EndpointMgr() = default; // 兜底由 ~HcommResMgr() 调 DeInit() 承担（成员化后析构序确定）

    EndpointMgr(const EndpointMgr&) = delete;
    EndpointMgr& operator=(const EndpointMgr&) = delete;

    // --- 收编 HcommEndpointMap（语义不变，不做去重）---
    void Add(EndpointHandle handle, std::unique_ptr<Endpoint> endpoint);
    bool Remove(EndpointHandle handle);
    bool Update(EndpointHandle handle, std::unique_ptr<Endpoint> newEndpoint);
    Endpoint* Get(EndpointHandle handle);

    // 进程销毁兜底：遍历 endpointMap_ 强制释放残留（析构链中子类会回调各设备 EndpointCtxMgr 释放 ctx）
    void DeInit();

private:
    std::mutex mtx_; // 仅护 endpointMap_；Endpoint 析构须在锁外（析构链回调 EndpointCtxMgr 重新拿 ctx 锁）
    std::unordered_map<EndpointHandle, std::unique_ptr<Endpoint>> endpointMap_; // 收编 HcommEndpointMap
};

} // namespace hcomm

#endif // HCOMM_ENDPOINT_MGR_H
