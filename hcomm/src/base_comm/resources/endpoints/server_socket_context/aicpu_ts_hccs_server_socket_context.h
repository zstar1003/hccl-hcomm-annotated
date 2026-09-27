/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_HCCS_SERVER_SOCKET_CONTEXT_H
#define AICPU_TS_HCCS_SERVER_SOCKET_CONTEXT_H

#include <cstdint>
#include "port.h"
#include "server_socket_context.h"

namespace hcomm {

// ---- GlobalNetDevMgr 路径（ServerInit/ServerDeInit，从 AicpuTsHccsEndpoint 迁入）----

// AICPU_TS HCCS 专用：调 GlobalNetDevMgr::ServerInit/ServerDeInit，无端口分配、无 socket 实例
class AicpuTsHccsServerSocketContext : public ServerSocketContext {
public:
    AicpuTsHccsServerSocketContext(uint32_t devPhyId, uint32_t serverPort);
    ~AicpuTsHccsServerSocketContext() override;                // serverListened_ 时 ServerDeInit 兜底（幂等）
    HcclResult ServerSocketListen(uint32_t port) override;     // 调 ServerInit(serverPort_)，port 未使用
    HcclResult ServerSocketStopListen(uint32_t port) override; // 调 ServerDeInit(port)，serverListened_ 判断
    // ServerSocketGetListenPort 迁移前未覆写，继承基类默认 HCCL_E_NOT_SUPPORT

private:
    uint32_t devPhyId_;   // 构造时从 endpointDesc_.loc.device.devPhyId 传入
    uint32_t serverPort_; // 构造时传入（AICPU_CHANNEL_DEFAULT_PORT）
    bool serverListened_{false};
};

} // namespace hcomm

#endif // AICPU_TS_HCCS_SERVER_SOCKET_CONTEXT_H
