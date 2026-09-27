/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_CH_SER_LETCP_DATA_H
#define BKF_CH_SER_LETCP_DATA_H

#include "bkf_comm.h"
#include "bkf_assert.h"
#include "bkf_ch_ser_adef.h"
#include "bkf_ch_ser_conn_id_base.h"
#include "v_avll.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma pack(4)
typedef struct tagBkfChSerLetcpLsn BkfChSerLetcpLsn;

#define BKF_CH_SER_LETCP_SIGN (0xa189)
struct tagBkfChSer {
    uint16_t sign;
    uint8_t hasEnable : 1;
    uint8_t rsv1 : 7;
    uint8_t pad1[1];
    BkfChSerInitArg argInit;
    char *name;
    BkfLog *log;
    BkfChSerEnableArg argEnable;
    BkfChSerLetcpLsn *lsnCache;
    AVLL_TREE lsnSet;
};

#define BKF_CH_SER_LETCP_LSN_SIGN (0xa190)
struct tagBkfChSerLetcpLsn {
    uint16_t sign;
    uint8_t lsnFdAttachOk : 1;
    uint8_t lsnBindOk : 1;
    uint8_t lsnOk : 1;
    uint8_t rsv1 : 5;
    uint8_t pad1[1];
    BkfChSer *ch;
    AVLL_NODE avlNode;
    BkfUrl keyUrlSelf;
    int32_t lsnFd;
    BkfChSerConnId *connIdCache;
    AVLL_TREE connIdSet;
    BkfTmrId *tmrIdReLsn;
};

#define BKF_CH_SER_LETCP_CONN_ID_SIGN (0xa191)
struct tagBkfChSerConnId {
    BkfChSerConnIdBase base;
    uint16_t sign;
    uint8_t connFdAttachOk : 1;
    uint8_t lockDel : 1;
    uint8_t softDel : 1;
    uint8_t rsv1 : 5;
    uint8_t tcpState;
    BkfChSerLetcpLsn *lsn;
    AVLL_NODE avlNode;
    int32_t keyConnFd;
    BkfUrl urlCli;
    BkfUrl urlLocal; /* 本地tcp上送的地址和端口号 */
    BkfTmrId *tmrIdWriteErr;
};
BKF_STATIC_ASSERT(BKF_MBR_IS_FIRST(BkfChSerConnId, base));

typedef struct tagBkfChSerLetcpMsgHead {
    uint16_t sign;
    uint8_t pad1[2];
    uint8_t data[0];
} BkfChSerLetcpMsgHead;
#define BKF_CH_SER_LETCP_MSG_SIGN (0xa189)
#define BKF_CH_SER_LETCP_APP_DATA_2CH_MSG_HEAD(appData)                                                                \
    (((uintptr_t)(appData) >= sizeof(BkfChSerLetcpMsgHead))                                                            \
            ? (BkfChSerLetcpMsgHead *)((uint8_t *)(appData) - sizeof(BkfChSerLetcpMsgHead))                            \
            : VOS_NULL)

/* func */
BkfChSer *BkfChSerLetcpDataInit(BkfChSerInitArg *arg);
void BkfChSerLetcpDataUninit(BkfChSer *ch);

BkfChSerLetcpLsn *BkfChSerLetcpAddLsn(BkfChSer *ch, BkfUrl *urlSelf);
void BkfChSerLetcpDelLsn(BkfChSer *ch, BkfChSerLetcpLsn *lsn);
void BkfChSerLetcpDelAllLsn(BkfChSer *ch);
BkfChSerLetcpLsn *BkfChSerLetcpFindLsn(BkfChSer *ch, BkfUrl *urlSelf);
BkfChSerLetcpLsn *BkfChSerLetcpFindNextLsn(BkfChSer *ch, BkfUrl *urlSelf);
BkfChSerLetcpLsn *BkfChSerLetcpGetFirstLsn(BkfChSer *ch, void **itorOutOrNull);
BkfChSerLetcpLsn *BkfChSerLetcpGetNextLsn(BkfChSer *ch, void **itorInOut);

BkfChSerConnId *BkfChSerLetcpAddConnId(BkfChSer *ch, BkfChSerLetcpLsn *lsn, int32_t connFd, BkfUrl *urlCli);
void BkfChSerLetcpDelConnId(BkfChSer *ch, BkfChSerConnId *connId);
void BkfChSerLetcpDelAllConnId(BkfChSer *ch, BkfChSerLetcpLsn *lsn);
BkfChSerConnId *BkfChSerLetcpFindConnId(BkfChSer *ch, BkfChSerLetcpLsn *lsn, int32_t connFd);
BkfChSerConnId *BkfChSerLetcpFindNextConnId(BkfChSer *ch, BkfChSerLetcpLsn *lsn, int32_t connFd);
BkfChSerConnId *BkfChSerLetcpGetFirstConnId(BkfChSer *ch, BkfChSerLetcpLsn *lsn, void **itorOutOrNull);
BkfChSerConnId *BkfChSerLetcpGetNextConnId(BkfChSer *ch, BkfChSerLetcpLsn *lsn, void **itorInOut);

uint32_t BkfChSerLetcpStartConnOnceTmrWriteErr(BkfChSer *ch, BkfChSerConnId *connId, F_BKF_TMR_TIMEOUT_PROC proc,
    uint32_t intervalMs);
void BkfChSerLetcpStopConnTmrWriteErr(BkfChSer *ch, BkfChSerConnId *connId);

BkfChSerLetcpMsgHead *BkfChSerLetcpMsgMalloc(BkfChSer *ch, int32_t dataBufLen);
void BkfChSerLetcpMsgFree(BkfChSer *ch, BkfChSerLetcpMsgHead *chMsgHead);
uint32_t BkfChSerLetcpStartOnceTmrReListen(BkfChSer *ch, BkfChSerLetcpLsn *lsn, F_BKF_TMR_TIMEOUT_PROC proc,
    uint32_t intervalMs);
void BkfChSerLetcpStopTmrReListen(BkfChSer *ch, BkfChSerLetcpLsn *lsn);

#pragma pack()
#if __cplusplus
}
#endif

#endif
