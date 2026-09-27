/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NET_CO_H
#define NET_CO_H

#ifdef __cplusplus
extern "C" {
#endif
#define TYPE_TBL_COMM_INFO (1001)
#define TYPE_TBL_OPER (1002)
#define TYPE_TBL_ADJ (1003)
#define TYPE_TBL_RANK (1004)
#define TYPE_TBL_RANK_DIST (1005)
#define TYPE_TBL_ROOT_RANK (1006)

typedef struct TagNetCoIpPortArg {
    unsigned int localIp;
    unsigned int gatewayIp;
    unsigned short localNetPort;
    unsigned short gatewayPort;
    unsigned short listenPort;
} NetCoIpPortArg;

__attribute__((visibility("default"))) void *NetCoInitFactory(int epollfd, NetCoIpPortArg ipPortArg);

__attribute__((visibility("default"))) void NetCoDestruct(void *co);

__attribute__((visibility("default"))) unsigned int NetCoFdEventDispatch(void *co, int fd, unsigned int curEvents);

__attribute__((visibility("default"))) int NetCoTblAddUpd(void *netcoHandle, unsigned int type, char *data,
    unsigned int data_len);

#ifdef __cplusplus
}
#endif

#endif // NET_CO_H