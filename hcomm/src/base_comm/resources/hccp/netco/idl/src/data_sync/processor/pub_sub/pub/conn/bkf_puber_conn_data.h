/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_PUBER_CONN_DATA_H
#define BKF_PUBER_CONN_DATA_H

#include "bkf_puber_conn.h"
#include "bkf_puber_table_type.h"
#include "bkf_msg.h"
#include "bkf_puber_sess.h"
#include "bkf_fsm.h"
#include "bkf_bufq.h"
#include "v_avll.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(4)
enum {
    BKF_PUBER_CONN_STATE_WAIT_HELLO,
    BKF_PUBER_CONN_STATE_WAIT_SEND_HELLO_ACK,
    BKF_PUBER_CONN_STATE_CONN,
    /* 扩展其他定义 */
    BKF_PUBER_CONN_STATE_CNT
};

enum {
    BKF_PUBER_CONN_EVT_HELLO,
    BKF_PUBER_CONN_EVT_UNDERLAY_UNBLOCK,
    BKF_PUBER_CONN_EVT_JOB,
    BKF_PUBER_CONN_EVT_TMR,
    /* 扩展其他定义 */
    BKF_PUBER_CONN_EVT_CNT
};

#define BKF_PUBER_CONN_NEED_DELETE 100

struct TagBkfPuberConnMng {
    BkfPuberInitArg *argInit;
    char *name;
    BkfLog *log;
    uint8_t dispInitOk : 1;
    uint8_t rsv : 7;
    uint8_t pad1[3];
    BkfPuberTableTypeMng *tableTypeMng;
    BkfFsmTmpl *connFsmTmpl;
    int32_t connCnt;
    AVLL_TREE connSet;
};

typedef struct {
    BkfPuberConnMng *connMng;
    AVLL_NODE avlNode;
    BkfChSerConnId *connId; /* key */
    char suberName[BKF_TLV_NAME_LEN_MAX + 1];
    BkfBufq *rcvDataBuf;
    BkfBufq *lastSendLeftDataBuf;
    BkfJobId *jobId;
    uint8_t lockDel : 1; /* 防止conn处理过程中删除，后续处理跑飞 */
    uint8_t rsv1 : 7;
    uint8_t pad1[3];
    BkfTmrId *tmrId;
    BkfFsm fsm;
    int32_t slowSchedGenProcCntItor;
    BkfPuberSessMng *sessMng;
    BkfUrl cliUrl;
    BkfUrl localUrl;
    BkfUrl cliLsnUrl;
} BkfPuberConn;
#pragma pack()

BkfPuberConnMng *BkfPuberConnDataInit(BkfPuberInitArg *arg, BkfPuberTableTypeMng *tableTypeMng);
void BkfPuberConnDataUninit(BkfPuberConnMng *connMng);
BkfPuberConn *BkfPuberConnAdd(BkfPuberConnMng *connMng, BkfChSerConnId *connId,
    F_BKF_PUBER_SESS_TRIG_SCHED_SELF trigSchedSess, F_BKF_PUBER_SESS_TRIG_SLOW_SCHED_SELF trigSlowSchedSess);
void BkfPuberConnDel(BkfPuberConn *conn, BOOL ntfCh);
void BkfPuberConnDelAll(BkfPuberConnMng *connMng);
BkfPuberConn *BkfPuberConnFind(BkfPuberConnMng *connMng, BkfChSerConnId *connId);
BkfPuberConn *BkfPuberConnFindNext(BkfPuberConnMng *connMng, BkfChSerConnId *connId);
BkfPuberConn *BkfPuberConnGetFirst(BkfPuberConnMng *connMng, void **itorOutOrNull);
BkfPuberConn *BkfPuberConnGetNext(BkfPuberConnMng *connMng, void **itorInOut);

uint32_t BkfPuberConnStartJob(BkfPuberConn *conn);
void BkfPuberConnStopJob(BkfPuberConn *conn);
uint32_t BkfPuberConnStartTmr(BkfPuberConn *conn, F_BKF_TMR_TIMEOUT_PROC proc, uint32_t intervalMs);
void BkfPuberConnStopTmr(BkfPuberConn *conn);

static inline void BkfPuberConnLockDel(BkfPuberConn *conn)
{
    conn->lockDel = VOS_TRUE;
}

static inline void BkfPuberConnUnlockDel(BkfPuberConn *conn)
{
    conn->lockDel = VOS_FALSE;
}

static inline BOOL BkfPuberConnIsLockDel(BkfPuberConn *conn)
{
    return conn->lockDel;
}

#ifdef __cplusplus
}
#endif

#endif
