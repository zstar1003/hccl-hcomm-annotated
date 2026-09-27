/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEVICE_SERVER_SOCKET_CONTEXT_H
#define DEVICE_SERVER_SOCKET_CONTEXT_H

#include <cstdint>
#include <mutex>
#include "port.h"
#include "externalinput_pub.h"
#include "hcomm_res_defs.h"
#include "server_socket_context.h"

namespace hcomm {

// DEVICE 侧 Endpoint 使用：devPhyId/locType 构造时传入，NicType=DEVICE_NIC_TYPE，有 locType 前置检查
class DeviceServerSocketContext : public ServerSocketContext {
public:
    DeviceServerSocketContext(
        Hccl::ConnectProtoType protoType, uint32_t devPhyId, EndpointLocType locType, const CommAddr& commAddr);
    ~DeviceServerSocketContext() override; // 析构期内虚表仍指向本类，显式调非虚停止监听实现
    HcclResult ServerSocketListen(uint32_t port) override;
    HcclResult ServerSocketStopListen(uint32_t port) override;
    HcclResult ServerSocketGetListenPort(uint32_t* port) override;

private:
    HcclResult ServerSocketStopListenImpl(uint32_t port);
    Hccl::ConnectProtoType protoType_;
    uint32_t devPhyId_;       // 构造时从 endpointDesc_.loc.device.devPhyId 传入
    EndpointLocType locType_; // 构造时传入，用于前置检查
    std::mutex portMutex_;
    uint32_t dynamicPort_{HCCL_INVALID_PORT};
    CommAddr commAddr_; // 构造时从 endpointDesc.commAddr 传入，方法内经 CommAddrToIpAddress 转换
};

} // namespace hcomm

#endif // DEVICE_SERVER_SOCKET_CONTEXT_H
