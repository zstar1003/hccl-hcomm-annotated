/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_SUBER_SUB_SESS_DATA_H
#define BKF_SUBER_SUB_SESS_DATA_H

#include "bkf_suber_sess.h"
#include "bkf_log.h"
#include "v_avll.h"
#include "bkf_dl.h"
#include "bkf_suber_data.h"
#include "bkf_tmr.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)

struct tagBkfSuberSessMng {
    BkfSuberEnv *env;
    BkfLog *log;
    uint32_t sessCnt;
    AVLL_TREE sessSet;
    void *cookie;
    F_BKF_SUBER_SESS_TRIG_SCHED_SELF tirgSchedSelf;
    BkfDl subSess;
    BkfDl unSubSess;
    BkfDl verifySubSess;
    BkfUrl pubUrl;
    BkfUrl locUrl;
    BkfSuberDataMng *dataMng;
    BkfFsmTmpl *sessFsmTmpl;
    uint32_t batchTimeoutChkFlag; /* 平滑超时检查定时器取值：0-1 */
    char sessFsmTmplName[BKF_NAME_LEN_MAX + 1];
    BkfTmrId *reSubTmrId;
    uint32_t disconnReason;
    uint8_t deCongest : 1; /* 解流控标记 */
    uint8_t pad : 7;
    uint8_t pad1[3];
};

typedef struct tagBkfSuberSessKey {
    uint16_t tableTypeId;
    uint8_t pad1[2];
    F_BKF_CMP sliceKeyCmp;
    void *sliceKey;
} BkfSuberSessKey;

struct tagBkfSuberSess {
    BkfSuberSessKey key;
    BkfSuberSessMng *sessMng;
    AVLL_NODE avlNode;
    BkfDlNode dlStateNode;
    BkfTmrId *batchChkTmr;
    uint64_t seq;
    BkfFsm fsm;
    uint8_t isVerify : 1;
    uint8_t pad : 7;
    uint8_t sliceKey[0];
};
BkfSuberSessMng *BkfSuberSessMngDataInit(BkfSuberSessMngInitArg *initArg);
void BkfSuberSessMngDataUnInit(BkfSuberSessMng *sessMng);

BkfSuberSess *BkfSuberSessDataAdd(BkfSuberSessMng *sessMng, void *sliceKey, uint16_t tableTypeId);
void BkfSuberSessDataDel(BkfSuberSess *sess);
void BkfSuberSessDataDelByKey(BkfSuberSessMng *sessMng, void *sliceKey, uint16_t tableTypeId);
BkfSuberSess *BkfSuberSessDataFind(BkfSuberSessMng *sessMng, void *sliceKey, uint16_t tableTypeId);
BkfSuberSess *BkfSuberSessDataFindNext(BkfSuberSessMng *sessMng, void *sliceKey, uint16_t tableTypeId);
BkfSuberSess *BkfSuberSessDataGetFirst(BkfSuberSessMng *sessMng, void **itorOutOrNull);
BkfSuberSess *BkfSuberSessDataGetNext(BkfSuberSessMng *sessMng, void **itorInOut);
void BkfSuberSessDataDelAll(BkfSuberSessMng *sessMng);

#define BKF_SUBER_SESS_RESUB_TIMER_INTERVAL (5000)
uint32_t BkfSuberSessCreateReSubTmr(BkfSuberSessMng *sessMng);
void BkfSuberSessDeleteReSubTmr(BkfSuberSessMng *sessMng);
uint32_t BkfSubSessReSubTmrSvcProc(void *paramTmrStart, void *noUse);

#pragma pack()
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif