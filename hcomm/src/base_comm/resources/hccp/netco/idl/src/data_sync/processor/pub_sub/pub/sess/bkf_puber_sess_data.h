/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_PUBER_SESS_DATA_H
#define BKF_PUBER_SESS_DATA_H

#include "bkf_puber_sess.h"
#include "bkf_dc_itor.h"
#include "bkf_fsm.h"
#include "bkf_dl.h"
#include "v_avll.h"
#include "bkf_log.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(4)
enum {
    /* 下面的顺序，和调度的先后顺序有关, 并需要参考sched和slow sched的宏定义 */
    BKF_PUBER_SESS_STATE_WAIT_SUB,
    BKF_PUBER_SESS_STATE_WAIT_SEND_SUB_ACK,
    BKF_PUBER_SESS_STATE_WAIT_MAY_SEND_BATCH_BEGIN,
    BKF_PUBER_SESS_STATE_WAIT_SEND_BATCH_BEGIN,
    BKF_PUBER_SESS_STATE_WAIT_SEND_BATCH_END,
    BKF_PUBER_SESS_STATE_WAIT_SEND_DEL_SUB_NTF,
    BKF_PUBER_SESS_STATE_BATCH_DATA,
    BKF_PUBER_SESS_STATE_REAL_DATA,
    BKF_PUBER_SESS_STATE_IDLE,
    BKF_PUBER_SESS_STATE_SLOW_BATCH_DATA,
    BKF_PUBER_SESS_STATE_REAL_SLOW_BATCH_DATA,
    /* 扩展其他定义 */
    BKF_PUBER_SESS_STATE_CNT
};

enum {
    BKF_PUBER_SESS_EVT_SUB,
    BKF_PUBER_SESS_EVT_UNSUB,
    BKF_PUBER_SESS_EVT_TUPLE_CHG,
    BKF_PUBER_SESS_EVT_TABLE_CPLT,
    BKF_PUBER_SESS_EVT_TABLE_REL,
    BKF_PUBER_SESS_EVT_SCHED,
    BKF_PUBER_SESS_EVT_SLOW_SCHED,

    BKF_PUBER_SESS_EVT_CNT
};

#define BKF_PUBER_SESS_NEED_DELETE 250

struct TagBkfPuberSessMng {
    BkfPuberInitArg *argInit;
    char *name;
    BkfLog *log;
    BkfPuberTableTypeMng *tableTypeMng;
    F_BKF_PUBER_SESS_TRIG_SCHED_SELF trigSchedSelf;
    F_BKF_PUBER_SESS_TRIG_SLOW_SCHED_SELF trigSlowSchedSelf;
    void *cookieInit;
    BkfFsmTmpl *sessFsmTmpl;
    uint32_t batchTimeoutChkFlag; /* 平滑超时检查定时器取值：0-1 */
    AVLL_TREE sessSet;
    BkfDl sessSetByState[BKF_PUBER_SESS_STATE_CNT];
};

typedef struct {
    uint16_t tableTypeId;
    uint8_t pad1[2];
    F_BKF_CMP sliceKeyCmp;
    uint8_t *sliceKey;
} BkfPuberSessKey;

typedef struct {
    BkfPuberSessMng *sessMng;
    AVLL_NODE avlNode;
    BkfDlNode dlStateNode;
    uint64_t subTransNum;
    BkfFsm fsm;
    BkfPuberSessKey key;
    BkfDcTupleKeyItor *itorBatchData;
    BkfDcTupleSeqItor *itorRealData;
    uint8_t subWithVerify : 1;
    uint8_t subWithNeedTblCplt : 1;
    uint8_t rsv1 : 6;
    uint8_t pad1[3];
    BkfTmrId *batchChkTmr;
    uint8_t sliceKey[0];
} BkfPuberSess;
#pragma pack()

BkfPuberSessMng *BkfPuberSessDataInit(BkfPuberInitArg *arg, BkfPuberTableTypeMng *tableTypeMng,
    F_BKF_PUBER_SESS_TRIG_SCHED_SELF trigSchedSess, F_BKF_PUBER_SESS_TRIG_SLOW_SCHED_SELF trigSlowSchedSess,
    void *cookie);
void BkfPuberSessDataUninit(BkfPuberSessMng *sessMng);
BkfPuberSess *BkfPuberSessAdd(BkfPuberSessMng *sessMng, uint8_t *sliceKey, uint16_t tableTypeId,
    BkfDcTupleItorVTbl *vTbl);
void BkfPuberSessDel(BkfPuberSess *sess);
void BkfPuberSessDelAll(BkfPuberSessMng *sessMng);
BkfPuberSess *BkfPuberSessFind(BkfPuberSessMng *sessMng, uint8_t *sliceKey, uint16_t tableTypeId);
BkfPuberSess *BkfPuberSessFindNext(BkfPuberSessMng *sessMng, uint8_t *sliceKey, uint16_t tableTypeId);
BkfPuberSess *BkfPuberSessGetFirst(BkfPuberSessMng *sessMng, void **itorOutOrNull);
BkfPuberSess *BkfPuberSessGetNext(BkfPuberSessMng *sessMng, void **itorInOut);
BkfPuberSess *BkfPuberSessGetFirstByState(BkfPuberSessMng *sessMng, uint8_t sessState, void **itorOutOrNull);
uint32_t BkfPuberSessMoveFirst2LastByState(BkfPuberSessMng *sessMng, uint8_t sessState);

static inline BOOL BkfPuberSessStateIsValid(uint8_t state)
{
    return (state < BKF_PUBER_SESS_STATE_CNT);
}

#ifdef __cplusplus
}
#endif

#endif
