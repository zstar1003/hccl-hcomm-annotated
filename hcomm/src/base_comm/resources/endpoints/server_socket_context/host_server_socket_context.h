/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HOST_SERVER_SOCKET_CONTEXT_H
#define HOST_SERVER_SOCKET_CONTEXT_H

#include <cstdint>
#include <mutex>
#include "port.h"
#include "hcomm_res_defs.h"
#include "socket/socket.h"
#include "externalinput_pub.h"
#include "server_socket_context.h"

namespace hcomm {

// ---- ServerSocketManager 路径（原基类公共成员 protoType_/portMutex_/dynamicPort_ 下放至子类）----

// HOST 侧 Endpoint 使用：devPhyId 经 hrtGetDevice + hrtGetDevicePhyIdByIndex 获取，NicType=HOST_NIC_TYPE
class HostServerSocketContext : public ServerSocketContext {
public:
    HostServerSocketContext(Hccl::ConnectProtoType protoType, const CommAddr& commAddr);
    ~HostServerSocketContext() override; // 析构期内虚表仍指向本类，显式调非虚停止监听实现
    HcclResult ServerSocketListen(uint32_t port) override;
    HcclResult ServerSocketStopListen(uint32_t port) override;
    HcclResult ServerSocketGetListenPort(uint32_t* port) override;

private:
    HcclResult ServerSocketStopListenImpl(uint32_t port);
    // devPhyId 在每个方法内部调 hrtGetDevice + hrtGetDevicePhyIdByIndex 获取
    Hccl::ConnectProtoType protoType_;
    std::mutex portMutex_;
    uint32_t dynamicPort_{HCCL_INVALID_PORT};
    CommAddr commAddr_; // 构造时从 endpointDesc.commAddr 传入，方法内经 CommAddrToIpAddress 转换
};

} // namespace hcomm

#endif // HOST_SERVER_SOCKET_CONTEXT_H
