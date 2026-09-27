/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CPU_URMA_ENDPOINT_H
#define CPU_URMA_ENDPOINT_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

#include "endpoint.h"
#include "comm_queue_context/jetty_context.h"
#include "server_socket_context/host_server_socket_context.h"
#include "ub_reged_mem_mgr.h"
#include "ccu_channel_ctx_pool.h"
#include "externalinput_pub.h"

namespace hcomm {
class CpuUrmaEndpoint : public Endpoint {
public:
    explicit CpuUrmaEndpoint(const EndpointDesc& endpointDesc);
    ~CpuUrmaEndpoint() noexcept override;

    HcclResult Init() override;

    RegedMemMgr* GetRegedMemMgr() override { return regedMemMgr_.get(); }
    void* GetRdmaHandle() override { return ctxHandle_; }
    bool IsCtxHandleValid() const override;
    // 共享 Jetty 上下文访问入口：返回 CommQueueContext 基类视图，调用方按需 downcast JettyContext
    CommQueueContext* GetCommQueueContext() override;
    ServerSocketContext* GetServerSocketContext() override
    {
        return serverSocketContext_.has_value() ? &serverSocketContext_.value() : nullptr;
    }

private:
    HcclResult ReleaseEndpointCtx();
    HcclResult AttachCache(const MemMgrCacheKey& key, const std::function<std::shared_ptr<RegedMemMgr>()>& creator);
    HcclResult ReleaseCache();

    void* ctxHandle_{nullptr};
    std::shared_ptr<EndpointCtx> endpointCtx_{};
    std::shared_ptr<UbRegedMemMgr> regedMemMgr_{};
    std::optional<HostServerSocketContext> serverSocketContext_{};
    MemMgrCacheKey cacheKey_{};
    std::shared_ptr<ProcRegedMemMgrCache> cacheKeepAlive_{};
    std::unique_ptr<JettyContext> jettyContext_{nullptr};
    std::once_flag jettyContextOnce_;
};
} // namespace hcomm

#endif // CPU_URMA_ENDPOINT_H
