/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONN_INTERFACE_H
#define CONN_INTERFACE_H

#include <string>
#include "topo_common_types.h"
#include "ip_address.h"

namespace Hccl {

class ConnInterface {
public:
    // 使用地址信息、位置信息、链路类型、链路协议构造接口
    ConnInterface(
        const IpAddress inputAddr, const AddrPosition inputPos, const LinkType inputLinkType,
        const LinkProtocol inputLinkProtocol);
    IpAddress GetAddr() const;
    AddrPosition GetPos() const;
    LinkType GetLinkType() const;
    LinkProtocol GetLinkProtocol() const;
    std::string Describe() const;
    bool operator==(const ConnInterface& rhs) const;
    bool operator!=(const ConnInterface& rhs) const;

private:
    IpAddress addr{};
    AddrPosition pos{};
    LinkType linkType{};
    LinkProtocol linkProtocol{};
};
} // namespace Hccl

#endif // CONN_INTERFACE_H
