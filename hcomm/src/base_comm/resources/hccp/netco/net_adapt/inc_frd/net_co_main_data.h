/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NET_CO_MAIN_DATA_H
#define NET_CO_MAIN_DATA_H

#include "net_co_main.h"
#include "bkf_mem.h"
#include "bkf_xmap.h"
#include "bkf_disp.h"
#include "bkf_log_cnt.h"
#include "bkf_log.h"
#include "bkf_pfm.h"
#include "wfw_mux.h"
#include "bkf_tmr.h"
#include "bkf_job.h"
#include "bkf_sys_log.h"
#include "wfw_linux_comm.h"

#ifdef __cplusplus
extern "C" {
#endif
#define NET_CO_LOG_MOD_DO(log, fmt, ...)                                                                               \
    do {                                                                                                               \
        int depth_ = WfwGetFuncCallStackDepth();                                                                       \
        BKF_LOG_DEBUG((log), "[modDo_@_%03d][%s]" fmt, depth_, __func__, ##__VA_ARGS__);                               \
    } while (0)

#define NET_CO_JOB_PRIO_DFT 6
enum {
    NET_CO_JOB_TYPE_IN_SUBER,
    NET_CO_JOB_TYPE_OUT_DC,
    NET_CO_JOB_TYPE_OUT_PUBER,

    NET_CO_JOB_TYPE_CNT
};

#pragma pack(4)
typedef struct TagNetCoIn NetCoIn;
typedef struct TagNetCoOut NetCoOut;
struct TagNetCo {
    NetCoInitArg argInit;
    char *name;
    BOOL dbgOn;
    uint32_t selfCid;
    BkfMemMng *memMng;
    BkfXMap *xMap;
    BkfDisp *disp;
    BkfLogCnt *logCnt;
    BkfLog *log;
    char *logFilePathName;
    BOOL pfmOn;
    BkfPfm *pfm;
    WfwMux *mux;
    void *muxAdpee;
    BkfTmrMng *tmrMng;
    void *tmrMngAdpee;
    BkfTmrId *tmrIdDispOut;
    BkfJobMng *jobMng;
    void *jobMngAdpee;
    BkfSysLogMng *sysLogMng;

    NetCoIn *in;
    NetCoOut *out;
    uint8_t rsv1[0x10];
};
#pragma pack()

#ifdef __cplusplus
}
#endif

#endif
