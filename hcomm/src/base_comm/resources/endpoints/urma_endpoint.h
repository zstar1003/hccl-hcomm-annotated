/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef URMA_ENDPOINT_H
#define URMA_ENDPOINT_H

#include <memory>
#include <mutex>
#include <optional>
#include <vector>
#include <string>
#include "endpoint.h"
#include "comm_queue_context/jetty_context.h"
#include "server_socket_context/device_server_socket_context.h"
#include "ub_reged_mem_mgr.h"
#include "ccu_channel_ctx_pool.h"
#include "socket/socket.h"
#include "externalinput_pub.h"

namespace hcomm {
/**
 * @note 职责：DEVICE 侧 UB 协议通信设备 Endpoint，管理通信设备上下文，以及设备上的注册内存。
 */
class UrmaEndpoint : public Endpoint {
public:
    explicit UrmaEndpoint(const EndpointDesc& endpointDesc);
    ~UrmaEndpoint() noexcept override;

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

    CcuChannelCtxPool* GetCcuChannelCtxPool();

    // UB 异步事件获取为 UrmaEndpoint 自有方法（基类不再提供默认实现）
    HcclResult GetAsyncEvents(uint32_t devPhyId, struct AsyncEvent events[], uint32_t& num);

private:
    HcclResult ReleaseEndpointCtx();
    HcclResult AttachCache(const MemMgrCacheKey& key, const std::function<std::shared_ptr<RegedMemMgr>()>& creator);
    HcclResult ReleaseCache();

    void* ctxHandle_{nullptr};
    std::shared_ptr<EndpointCtx> endpointCtx_{};
    std::shared_ptr<UbRegedMemMgr> regedMemMgr_{};
    std::optional<DeviceServerSocketContext> serverSocketContext_{};
    MemMgrCacheKey cacheKey_{};
    std::shared_ptr<ProcRegedMemMgrCache> cacheKeepAlive_{};
    std::unique_ptr<JettyContext> jettyContext_{nullptr};
    std::once_flag jettyContextOnce_;
    std::unique_ptr<CcuChannelCtxPool> ccuChannelCtxPool_{nullptr};
};
} // namespace hcomm

#endif // URMA_ENDPOINT_H
