/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SERVER_SOCKET_CONTEXT_H
#define SERVER_SOCKET_CONTEXT_H

#include <cstdint>
#include "port.h"

namespace hcomm {

/**
 * @note 职责：封装 ServerSocketListen/ServerSocketStopListen/ServerSocketGetListenPort 的纯虚接口。
 *       接口形态与迁移前 Endpoint 基类一致：仅传入 port，不传 ipAddr；Listen 纯虚；
 *       StopListen/GetListenPort 虚默认返回 HCCL_E_NOT_SUPPORT（迁移前未覆写的子类继承此行为，
 *       基类默认实现不检查入参）。基类不持有数据成员：需要按地址监听的子类（Host/Device）自行持有
 *       commAddr（构造时从 endpointDesc.commAddr 传入），方法内经 CommAddrToIpAddress 转换为
 *       IpAddress 使用（与迁移前一致）；不使用地址的子类不持有。
 */
class ServerSocketContext {
public:
    virtual ~ServerSocketContext() = default;

    virtual HcclResult ServerSocketListen(uint32_t port) = 0;
    virtual HcclResult ServerSocketStopListen([[maybe_unused]] uint32_t port) { return HCCL_E_NOT_SUPPORT; }
    virtual HcclResult ServerSocketGetListenPort([[maybe_unused]] uint32_t* port) { return HCCL_E_NOT_SUPPORT; }
};

} // namespace hcomm

#endif // SERVER_SOCKET_CONTEXT_H
