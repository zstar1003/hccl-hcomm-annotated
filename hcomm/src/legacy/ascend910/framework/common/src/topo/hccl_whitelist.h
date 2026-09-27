/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_WHITELIST_H
#define HCCL_WHITELIST_H

#include <mutex>
#include <map>
#include <vector>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "sal_pub.h"

namespace hccl {
enum class HcclWhiteListType { HCCL_WHITELIST_HOST = 0, HCCL_WHITELIST_RESERVED };

class HcclWhitelist {
public:
    static HcclWhitelist& GetInstance();
    HcclResult LoadConfigFile(const std::string& realName);
    HcclResult GetHostWhiteList(std::vector<HcclIpAddress>& whiteList);

private:
    HcclWhitelist();
    ~HcclWhitelist();
    std::map<HcclWhiteListType, std::vector<HcclIpAddress>> whiteLists_;
    std::mutex whiteListsMutex_;
};
} // namespace hccl

#endif // HCCL_WHITELIST_H
