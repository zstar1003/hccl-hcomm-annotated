/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_SOCKET_AGENT_H
#define HCCLV2_SOCKET_AGENT_H

#include "hccl/base.h"
#include "socket.h"

namespace Hccl {

class SocketAgent {
public:
    explicit SocketAgent(Socket* socket) : socket_(socket) {}

    void SendMsg(const void* data, u64 dataLen);
    bool RecvMsg(void* msg, u64& revMsgLen);

private:
    Socket* socket_{nullptr};
};

} // namespace Hccl
#endif // HCCLV2_SOCKET_AGENT_H
