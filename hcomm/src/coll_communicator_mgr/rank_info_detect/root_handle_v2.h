/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ROOT_HANDLE_V2_H
#define HCCL_ROOT_HANDLE_V2_H

#include "orion_adapter_hccp.h"
#include "ip_address.h"

namespace Hccl {

// HcclRootHandleV2 定义
constexpr u32 IP_ADDRESS_BUFFER_LEN = 64;
constexpr u32 ROOTINFO_IDENTIFIER_MAX_LENGTH = 128;
const std::string RANK_INFO_DETECT_TAG = "rank_info_detect_default_tag";

using HcclRootHandleV2 = struct HcclRootHandleDefV2 {
    char ip[IP_ADDRESS_BUFFER_LEN];
    u32 listenPort{HCCL_INVALID_PORT};
    HrtNetworkMode netMode{HrtNetworkMode::HDC};
    char identifier[ROOTINFO_IDENTIFIER_MAX_LENGTH];
};

// buffer大小
constexpr u32 MAX_BUFFER_LEN = 100 * 1024 * 1024; // 100M

constexpr u32 ONE_MILLISECOND_OF_USLEEP = 1000;

} // namespace Hccl

#endif // HCCL_ROOT_HANDLE_V2_H
