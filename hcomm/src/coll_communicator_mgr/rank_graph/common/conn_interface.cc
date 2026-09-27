/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conn_interface.h"
#include "string_util.h"

namespace Hccl {

ConnInterface::ConnInterface(
    const IpAddress inputAddr, const AddrPosition inputPos, const LinkType inputLinkType,
    const LinkProtocol inputLinkProtocol)
    : addr(inputAddr),
      pos(inputPos),
      linkType(inputLinkType),
      linkProtocol(inputLinkProtocol)
{}

IpAddress ConnInterface::GetAddr() const { return addr; }

AddrPosition ConnInterface::GetPos() const { return pos; }

LinkType ConnInterface::GetLinkType() const { return linkType; }

LinkProtocol ConnInterface::GetLinkProtocol() const { return linkProtocol; }

std::string ConnInterface::Describe() const
{
    return StringFormat("ConnInterface[addr=%s, pos=%s]", addr.Describe().c_str(), pos.Describe().c_str());
}

bool ConnInterface::operator==(const ConnInterface& rhs) const
{
    return addr == rhs.addr && pos == rhs.pos && linkType == rhs.linkType && linkProtocol == rhs.linkProtocol;
}

bool ConnInterface::operator!=(const ConnInterface& rhs) const { return !(rhs == *this); }
} // namespace Hccl
