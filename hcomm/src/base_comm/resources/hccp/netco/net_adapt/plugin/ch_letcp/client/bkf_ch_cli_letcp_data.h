/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_CH_CLI_LETCP_DATA_H
#define BKF_CH_CLI_LETCP_DATA_H

#include "bkf_comm.h"
#include "bkf_assert.h"
#include "bkf_ch_cli_adef.h"
#include "bkf_ch_cli_conn_id_base.h"
#include "v_avll.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma pack(4)

#define BKF_CH_CLI_LETCP_APP_DATA_2CH_MSG_HEAD(appData)                                                                \
    (((uintptr_t)(appData) >= sizeof(BkfChCliLetcpMsgHead))                                                            \
            ? (BkfChCliLetcpMsgHead *)((uint8_t *)(appData) - sizeof(BkfChCliLetcpMsgHead))                            \
            : VOS_NULL)

#define BKF_CH_CLI_LETCP_SIGN (0xb189)
struct tagBkfChCli {
    uint16_t sign;
    uint8_t hasEnable : 1;
    uint8_t selfCidHasSet : 1;
    uint8_t rsv1 : 6;
    uint8_t pad1[1];
    BkfChCliInitArg argInit;
    char *name;
    BkfLog *log;
    BkfChCliEnableArg argEnable;
    BkfUrl selfUrl;
    BkfChCliConnId *connIdCache;
    AVLL_TREE connSet; /* 连接远端 */
};

#define BKF_CH_CLI_LETCP_CONN_ID_SIGN (0xb190)
struct tagBkfChCliConnId {
    BkfChCliConnIdBase base;
    uint16_t sign;
    uint8_t connFdAttachOk : 1;
    uint8_t lockDel : 1;
    uint8_t softDel : 1;
    uint8_t rsv1 : 5;
    uint8_t tcpState;
    BkfUrl keyUrlSer; /* serUrl + localurl */
    BkfUrl keyUrlCli; /* localurl */
    BkfChCli *ch;
    AVLL_NODE avlNode;
    int32_t connFd;
    BkfTmrId *tmrIdWriteErr;
};
BKF_STATIC_ASSERT(BKF_MBR_IS_FIRST(BkfChCliConnId, base));

#define BKF_CH_CLI_LETCP_MSG_SIGN (0xb191)
typedef struct tagBkfChCliLetcpMsgHead {
    uint16_t sign;
    uint8_t pad1[2];
    uint8_t data[0];
} BkfChCliLetcpMsgHead;

BkfChCli *BkfChCliLetcpDataInit(BkfChCliInitArg *arg);
void BkfChCliLetcpDataUninit(BkfChCli *ch);

BkfChCliLetcpMsgHead *BkfChCliLetcpMsgMalloc(BkfChCli *ch, int32_t dataBufLen);
void BkfChCliLetcpMsgFree(BkfChCli *ch, BkfChCliLetcpMsgHead *chMsgHead);

BkfChCliConnId *BkfChCliLetcpNewConnId(BkfChCli *ch, BkfUrl *urlServer, BkfUrl *urlSelf);
BkfChCliConnId *BkfChCliLetcpFindConnId(BkfChCli *ch, BkfUrl *urlServer, BkfUrl *urlself);
BkfChCliConnId *BkfChCliLetcpFindNextConnId(BkfChCli *ch, BkfUrl *urlServer, BkfUrl *urlself);
void BkfChCliLetcpFreeConnId(BkfChCliConnId *connId);
BkfChCliConnId *BkfChCliLetcpGetFirstConnId(BkfChCli *ch, void **itorOutOrNull);
BkfChCliConnId *BkfChCliLetcpGetNextConnId(BkfChCli *ch, void **itorInOut);

uint32_t BkfChCliLetcpStartConnOnceTmrWriteErr(BkfChCliConnId *connId, F_BKF_TMR_TIMEOUT_PROC proc,
    uint32_t intervalMs);
void BkfChCliLetcpStopConnTmrWriteErr(BkfChCliConnId *connId);

#pragma pack()
#ifdef __cplusplus
}
#endif

#endif