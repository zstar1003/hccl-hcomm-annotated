/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UB_RTP_ENDPOINT_H
#define UB_RTP_ENDPOINT_H

#include <memory>
#include <mutex>
#include "endpoint.h"
#include "comm_queue_context/jetty_context.h"
#include "server_socket_context/ub_rtp_uboe_server_socket_context.h"
#include "ub_reged_mem_mgr.h"

namespace hcomm {
/**
 * @note 职责：AICPU通信引擎+UB_RTP协议的通信设备Endpoint，管理通信设备上下文，以及设备上的注册内存。
 *       UB_RTP的Init直接使用EID地址（不做IP→EID转换），rdmaHandle通过EID直接获取。
 */
class UbRtpEndpoint : public Endpoint {
public:
    explicit UbRtpEndpoint(const EndpointDesc& endpointDesc);
    ~UbRtpEndpoint() noexcept override;

    HcclResult Init() override;
    RegedMemMgr* GetRegedMemMgr() override { return regedMemMgr_.get(); }
    void* GetRdmaHandle() override { return ctxHandle_; }
    bool IsCtxHandleValid() const override;
    CommQueueContext* GetCommQueueContext() override;
    ServerSocketContext* GetServerSocketContext() override { return &serverSocketContext_; }

private:
    HcclResult ReleaseEndpointCtx();

    void* ctxHandle_{nullptr};
    std::shared_ptr<EndpointCtx> endpointCtx_{};
    std::shared_ptr<UbRegedMemMgr> regedMemMgr_{};
    UbRtpUboeServerSocketContext serverSocketContext_{}; // no-op 监听语义
    std::unique_ptr<JettyContext> jettyContext_{nullptr};
    std::once_flag jettyContextOnce_;
};

} // namespace hcomm

#endif // UB_RTP_ENDPOINT_H
