/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_COMMON_INNER_H
#define RS_COMMON_INNER_H

#include "hccp_common.h"

#define RS_MAX_IP_LEN 64     // IP地址(IPv4：点分十进制，IPv6 十六进制字符串)最大长度
#define IPV6_S6_ADDR_SIZE 16 // IPv6 have 16 u6_addr8
#define RS_MAX_DEV_NUM 64

struct RsIpAddrInfo {
    uint32_t family;
    union HccpIpAddr binAddr;
    char readAddr[RS_MAX_IP_LEN];
};

#endif // RS_COMMON_INNER_H