/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONTROL_PLANE_H
#define CONTROL_PLANE_H

#include <string>
#include "nlohmann/json.hpp"
#include "ip_address.h"
#include "nlohmann/json.hpp"
#include "topo_common_types.h"

namespace Hccl {
constexpr unsigned int MIN_VALUE_LISTENPORT = 1;
constexpr unsigned int MAX_VALUE_LISTENPORT = 65536;
constexpr unsigned int MIN_VALUE_ADDR = 1;
constexpr unsigned int MAX_VALUE_ADDR = 256;
class ControlPlane {
public:
    ControlPlane() {};
    AddrType addrType;
    IpAddress addr;
    u32 listenPort{0};
    void Deserialize(const nlohmann::json& controlPlaneJson);
    void GetBinStream(BinaryStream& binaryStream) const;
    explicit ControlPlane(BinaryStream& binStream);
    std::string Describe() const;

private:
    static const std::unordered_map<std::string, AddrType> strToAddrType;
    void EidToAddr(std::string str);
    void IPV4ToAddr(std::string str);
    void IPV6ToAddr(std::string str);

    static bool IsStringInAddrType(std::string str) { return strToAddrType.count(str) > 0; }
};

} // namespace Hccl

#endif // CONTROL_PLANE_H
