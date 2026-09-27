/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_SUBSCRIBER_ENV_H
#define BKF_SUBSCRIBER_ENV_H

#include "bkf_comm.h"
#include "bkf_assert.h"
#include "bkf_mem.h"
#include "bkf_disp.h"
#include "bkf_log.h"
#include "bkf_job.h"
#include "bkf_fsm.h"
#include "bkf_xmap.h"
#include "bkf_tmr.h"
#include "bkf_pfm.h"
#include "bkf_bas_type_mthd.h"
#include "bkf_ch_cli.h"
#include "bkf_subscriber.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#pragma pack(4)

/* common */
typedef struct tagBkfSuberSliceVTbl {
    uint16_t keyLen;
    uint8_t pad1[2];
    F_BKF_CMP keyCmp;
    F_BKF_GET_STR keyGetStrOrNull;
    F_BKF_DO keyCodec;
} BkfSuberSliceVTbl;

typedef struct tagBkfSuberEnv {
    char *name;
    char *svcName;
    uint16_t idlVersionMajor;
    uint16_t idlVersionMinor;
    BOOL dbgOn;
    uint8_t subMod;
    uint8_t pad[3];
    BkfMemMng *memMng;
    BkfDisp *disp;
    BkfLog *log;
    BkfPfm *pfm;
    BkfJobMng *jobMng;
    BkfXMap *xmap;
    uint32_t jobTypeId1;
    uint32_t jobTypeId2;
    uint32_t jobPrioH;
    uint32_t jobPrioL;
    uint64_t seeds; /* 全局事务号 */
    uint32_t msgLenMax;
    BkfSysLogMng *sysLogMng;
    BkfTmrMng *tmrMng;
    BkfChCliMng *chCliMng;
    BkfUrl selfUrl;
    BkfSuberSliceVTbl sliceVTbl;
    BkfUrl lsnUrl; /* 本设备作为puber时候的监听地址 */
    void *cookie;
    F_SUBER_ON_CONNECT onConn;
    F_SUBER_ON_DISCONNECT onDisConn;
} BkfSuberEnv;

#define BKF_SUBER_DISP_SLICELEN (128)
#define BKF_SUBER_SLICE_KEY_LEN_MAX (BKF_1K / 4)
char *BkfSuberGetSliceKeyStr(BkfSuberEnv *env, void *sliceKey, uint8_t *buf, int32_t bufLen);
#pragma pack()

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
