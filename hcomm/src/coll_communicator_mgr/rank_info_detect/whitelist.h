/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_WHITELIST_H
#define HCCLV2_WHITELIST_H

#include <mutex>
#include "nlohmann/json.hpp"
#include "ip_address.h"

namespace Hccl {

MAKE_ENUM(WhiteListType, HCCL_WHITELIST_HOST, HCCL_WHITELIST_RESERVED)

class Whitelist {
public:
    static Whitelist& GetInstance();
    void LoadConfigFile(const std::string& realName);
    void GetHostWhiteList(std::vector<IpAddress>& whiteList);
    nlohmann::json GetHostIp(nlohmann::json fileContent) const;

private:
    ~Whitelist();
    std::map<WhiteListType, std::vector<IpAddress>> whiteLists;
    std::mutex whiteListsMutex;
};
} // namespace Hccl

#endif // HCCLV2_WHITELIST_H
