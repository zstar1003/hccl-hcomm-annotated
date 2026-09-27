/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_URL_H
#define BKF_URL_H

#include "bkf_comm.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* common */
/*
1. 目前支持下面类型地址：
  1) v8 dms pid/cid方式： dms://0x80860472
  2) tcp方式: tcp://127.0.0.1:2021
2. 地址支持字串方式的构造；部分支持输入数据(易用性)
3. 只在边界处set/读取真实的地址，过程中url尽量赋值透传。
*/

enum {
    BKF_URL_TYPE_INVALID,
    BKF_URL_TYPE_DMS,
    BKF_URL_TYPE_TCP,
    BKF_URL_TYPE_MESH,
    BKF_URL_TYPE_V8TLS,
    BKF_URL_TYPE_V8TCP,
    BKF_URL_TYPE_LETCP,
    BKF_URL_TYPE_CNT
};
#define BKF_URL_TYPE_IS_VALID(type) (((type) > BKF_URL_TYPE_INVALID) && ((type) < BKF_URL_TYPE_CNT))
#define BKF_URL_TYPE_IS_TCP(type)                                                                                      \
    ((type) == BKF_URL_TYPE_TCP || (type) == BKF_URL_TYPE_V8TLS || (type) == BKF_URL_TYPE_V8TCP ||                     \
        (type) == BKF_URL_TYPE_LETCP)

#define BKF_URL_STR_LEN_MAX (255)
#define BKF_URL_uint32_t_HEX_STR_BUF_LEN_MIN (12)
#define BKF_URL_uint32_t_DEC_STR_BUF_LEN_MIN (12)
#define BKF_URL_DMS_STR_BUF_LEN_MIN (17)
#define BKF_URL_TCP_STR_BUF_LEN_MIN (28)
#define BKF_URL_MESH_STR_BUF_LEN_MIN (29)
#define BKF_URL_V8TLS_STR_BUF_LEN_MIN (30)
#define BKF_URL_V8TCP_STR_BUF_LEN_MIN (30)
#define BKF_URL_LETCP_STR_BUF_LEN_MIN (30)

typedef struct tagBkfUrlIp {
    uint32_t addrH; /* 主机序 */
    uint16_t port;
    uint8_t pad1[2];
} BkfUrlIp;
typedef struct tagBkfUrl {
    uint8_t sign;
    uint8_t signH;
    uint8_t type;
    uint8_t pad1[1];
    union {
        uint32_t dmsId;
        BkfUrlIp ip;
    };
} BkfUrl;

uint32_t BkfUrlInit(BkfUrl *url);
uint32_t BkfUrlSet(BkfUrl *url, char *urlStr);
uint32_t BkfUrlDmsSet(BkfUrl *url, uint32_t dmsId);
uint32_t BkfUrlTcpSet(BkfUrl *url, uint32_t addrH, uint16_t port);
uint32_t BkfUrlMeshSet(BkfUrl *url, uint32_t addrH, uint16_t port);
uint32_t BkfUrlTlsSet(BkfUrl *url, uint32_t addrH, uint16_t port);
uint32_t BkfUrlV8TcpSet(BkfUrl *url, uint32_t addrH, uint16_t port);
uint32_t BkfUrlLeTcpSet(BkfUrl *url, uint32_t addrH, uint16_t port);
BOOL BkfUrlIsValid(BkfUrl *url);
int32_t BkfUrlCmp(BkfUrl *url1Input, BkfUrl *url2InDs);
char *BkfUrlGetStr(BkfUrl *url, uint8_t *buf, int32_t bufLen);
uint32_t BkfUrlH2N(BkfUrl *urlH, BkfUrl *urlN);
uint32_t BkfUrlN2H(BkfUrl *urlN, BkfUrl *urlH);

#define BKF_URL_IS_EQUAL(url1, url2) (BkfUrlCmp((url1), (url2)) == 0)

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
