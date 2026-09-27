/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NETCO_API_H
#define NETCO_API_H

typedef struct TagNetCoIpPortArg {
    unsigned int localIp;
    unsigned int gatewayIp;
    unsigned short localNetPort;
    unsigned short gatewayPort;
    unsigned short listenPort;
} NetCoIpPortArg;

#ifdef CA_CONFIG_LLT
void *NetCoInitFactory(int epollfd, NetCoIpPortArg ipPortArg);
void NetCoDestruct(void *co);
unsigned int NetCoFdEventDispatch(void *co, int fd, unsigned int curEvents);
int NetCoTblAddUpd(void *netcoHandle, unsigned int type, char *data, unsigned int data_len);
#endif
#endif // NETCO_API_H
