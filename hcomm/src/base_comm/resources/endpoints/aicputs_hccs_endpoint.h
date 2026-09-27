/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPUTS_HCCS_ENDPOINT_H
#define AICPUTS_HCCS_ENDPOINT_H

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include "endpoint.h"
#include "server_socket_context/aicpu_ts_hccs_server_socket_context.h"
#include "hccs_reged_mem_mgr.h"
#include "hccl_ip_address.h"
#include "remote_ipc_rma_buffer.h"

namespace hcomm {
constexpr uint32_t AICPU_CHANNEL_DEFAULT_PORT = 16666;

/**
 * @note 职责：AICPU_TS通信引擎+HCCS协议的通信设备EndPoint，管理通信设备上下文，以及设备上的注册内存。
 */
class AicpuTsHccsEndpoint : public Endpoint {
public:
    explicit AicpuTsHccsEndpoint(const EndpointDesc& endpointDesc);

    ~AicpuTsHccsEndpoint() override;

    HcclResult Init() override;

    RegedMemMgr* GetRegedMemMgr() override { return regedMemMgr_.get(); }
    void* GetRdmaHandle() override { return nullptr; }
    bool IsCtxHandleValid() const override { return false; }

    // 监听方法由 AicpuTsHccsServerSocketContext 承载（Init 内 emplace 后才有值）
    ServerSocketContext* GetServerSocketContext() override
    {
        return serverSocketContext_.has_value() ? &serverSocketContext_.value() : nullptr;
    }

private:
    std::shared_ptr<HccsRegedMemMgr> regedMemMgr_{};
    hccl::HcclIpAddress devIpAddr_;
    HcclNetDevCtx netDevCtx_{nullptr};
    u32 serverPort_{AICPU_CHANNEL_DEFAULT_PORT};
    // Init 内 locType 检查通过后 emplace（延迟构造）
    std::optional<AicpuTsHccsServerSocketContext> serverSocketContext_{};
};
} // namespace hcomm
#endif // AICPUTS_HCCS_ENDPOINT_H
