/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NETWORK_COMMON_H
#define NETWORK_COMMON_H

#define USER_NAME_PATH_LEN 64
#define NET_INVALID_PORT 0xFF
#define NET_INVALID_GW 0xFF
#define NET_THREE_VALUE 3
#define NET_PHY_ID_MAX 16
#define BUF_LEN 256

static inline int DlRetConvert(int ret)
{
    return ret > 0 ? -ret : ret;
}

int NetCommGetSelfHome(char *homePath, unsigned int pathLen);
int NetGetGatewayAddress(unsigned int phyId, unsigned int *gtwAddr);

#endif