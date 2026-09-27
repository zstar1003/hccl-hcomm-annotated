/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UB_RTP_UBOE_SERVER_SOCKET_CONTEXT_H
#define UB_RTP_UBOE_SERVER_SOCKET_CONTEXT_H

#include <cstdint>
#include "log.h"
#include "server_socket_context.h"

namespace hcomm {

// Uboe/UbRtp 专用：监听操作为无操作成功（迁移前 UboeUbRtpEndpointHelper 的 no-op 语义，仅 HCCL_INFO）；
// ServerSocketGetListenPort 不覆写，继承基类默认 HCCL_E_NOT_SUPPORT（与迁移前一致）。
// UbMem 不持有 context（C 适配层判空返回 NOT_SUPPORT，即其迁移前行为）。
class UbRtpUboeServerSocketContext : public ServerSocketContext {
public:
    UbRtpUboeServerSocketContext() = default;
    HcclResult ServerSocketListen([[maybe_unused]] uint32_t port) override
    {
        HCCL_INFO("[%s] server socket listen is not supported, no-op success", __func__);
        return HCCL_SUCCESS;
    }
    HcclResult ServerSocketStopListen([[maybe_unused]] uint32_t port) override
    {
        HCCL_INFO("[%s] server socket stop listen is not supported, no-op success", __func__);
        return HCCL_SUCCESS;
    }
};

} // namespace hcomm

#endif // UB_RTP_UBOE_SERVER_SOCKET_CONTEXT_H
