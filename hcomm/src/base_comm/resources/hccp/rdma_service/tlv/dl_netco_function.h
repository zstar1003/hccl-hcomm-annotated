/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DL_NETCO_FUNCTION_H
#define DL_NETCO_FUNCTION_H

#include "netco_api.h"

struct RsNetcoOps {
    void *(*rsNetcoInit)(int epollfd, NetCoIpPortArg ipPortArg);
    void (*rsNetcoDeinit)(void *co);
    unsigned int (*rsNetcoEventDispatch)(void *co, int fd, unsigned int curEvents);
    int (*rsNetcoTblAddUpd)(void *netcoHandle, unsigned int type, char *data, unsigned int dataLen);
};

void RsNslbApiDeinit(void);
int RsNslbApiInit(void);
void *RsNetcoInit(int epollfd, NetCoIpPortArg ipPortArg);
void RsNetcoDeinit(void *co);
unsigned int RsNetcoEventDispatch(void *co, int fd, unsigned int curEvents);
int RsNetcoTblAddUpd(void *netcoHandle, unsigned int type, char *data, unsigned int dataLen);
#endif // DL_NETCO_FUNCTION_H
