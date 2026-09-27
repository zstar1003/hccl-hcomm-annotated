/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROCE_ENDPOINT_H
#define ROCE_ENDPOINT_H

#include <optional>
#include <memory>
#include <mutex>
#include <vector>
#include <string>
#include "endpoint.h"
#include "server_socket_context/host_server_socket_context.h"
#include "roce_reged_mem_mgr.h"
#include "externalinput_pub.h"

namespace hcomm {
/**
 * @note 职责：Host CPU通信引擎+RoCE协议的通信设备Endpoint，管理通信设备上下文，以及设备上的注册内存。
 */
class CpuRoceEndpoint : public Endpoint {
public:
    explicit CpuRoceEndpoint(const EndpointDesc& endpointDesc);

    ~CpuRoceEndpoint() noexcept override;

    HcclResult Init() override;

    RegedMemMgr* GetRegedMemMgr() override { return regedMemMgr_.get(); }
    void* GetRdmaHandle() override { return ctxHandle_; }
    bool IsCtxHandleValid() const override;
    ServerSocketContext* GetServerSocketContext() override
    {
        return serverSocketContext_.has_value() ? &serverSocketContext_.value() : nullptr;
    }

    struct Capabilities {
        uint64_t maxMsgSize{0};
        int lbMax{0};
        // 按需扩展
    };
    HcclResult GetCapabilities(Capabilities& caps);

private:
    HcclResult ReleaseEndpointCtx();
    HcclResult AttachCache(const MemMgrCacheKey& key, const std::function<std::shared_ptr<RegedMemMgr>()>& creator);
    HcclResult ReleaseCache();

    void* ctxHandle_{nullptr};
    std::shared_ptr<EndpointCtx> endpointCtx_{};
    std::shared_ptr<RoceRegedMemMgr> regedMemMgr_{};
    std::optional<HostServerSocketContext> serverSocketContext_{};
    MemMgrCacheKey cacheKey_{};
    std::shared_ptr<ProcRegedMemMgrCache> cacheKeepAlive_{};
    Capabilities capabilities_{};
    bool isCapabilitiesAvailable_{false};
};
} // namespace hcomm
#endif // ROCE_ENDPOINT_H
