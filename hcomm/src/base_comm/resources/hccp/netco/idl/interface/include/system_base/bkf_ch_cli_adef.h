/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_CH_CLI_ADEF_H
#define BKF_CH_CLI_ADEF_H

#include "bkf_comm.h"
#include "bkf_mem.h"
#include "bkf_disp.h"
#include "bkf_log.h"
#include "bkf_pfm.h"
#include "bkf_sys_log.h"
#include "bkf_log_cnt.h"
#include "bkf_xmap.h"
#include "bkf_tmr.h"
#include "bkf_job.h"
#include "bkf_url.h"
#include "wfw_mux.h"
#include "bkf_cer_arg.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* common */
typedef struct tagBkfChCliMng BkfChCliMng;
typedef struct tagBkfChCliConnId BkfChCliConnId;

/* init */
typedef struct tagBkfChCliMngInitArg {
    char *name;
    BOOL dbgOn;
    BkfMemMng *memMng;
    BkfDisp *disp;
    BkfLog *log;
    BkfPfm *pfm;
    BkfSysLogMng *sysLogMng;
    BkfLogCnt *logCnt;
    BkfXMap *xMap; /* v8tls传输层时，携带fd分发xmap句柄 */
    BkfTmrMng *tmrMng;
    BkfJobMng *jobMng;
    uint32_t jobTypeId;
    uint32_t jobPrio;
    uint32_t dmsCliQueId;
    uint8_t dmsIntfId;
    uint8_t dmsSubIntfId;
    uint8_t pad1[2];
    WfwMux *mux;  /* 和mesh/tcp等传输层配套使用 */
    void *sslHnd; /* sslLib全局句柄 */
    uint8_t rsv1[0x10];
} BkfChCliMngInitArg;

#define BKF_CH_CLI_MALLOC_DATA_BUF_LEN_MAX (0xffff - 0x40)

/* enable */
typedef void (*F_BKF_CH_CLI_ON_RCV_DATA)(void *cookieEnable, BkfChCliConnId *connId, void *data, int32_t dataLen);
typedef void (*F_BKF_CH_CLI_ON_RCV_DATA_EVENT)(void *cookieEnable, BkfChCliConnId *connId);
typedef void (*F_BKF_CH_CLI_ON_UNBLOCK)(void *cookieEnable, BkfChCliConnId *connId);
typedef void (*F_BKF_CH_CLI_ON_DISCONN)(void *cookieEnable, BkfChCliConnId *connId);
typedef void (*F_BKF_CH_CLI_ON_DISCONNEX)(void *cookieEnable, BkfChCliConnId *connId, BOOL isPeerFin);
typedef void (*F_BKF_CH_CLI_ON_CONN)(void *cookieEnable, BkfChCliConnId *connId, uint16_t srcPort);

typedef struct tagBkfChCliEnableArg {
    void *cookie;
    F_BKF_CH_CLI_ON_RCV_DATA onRcvData;
    F_BKF_CH_CLI_ON_RCV_DATA_EVENT onRcvDataEvent;
    F_BKF_CH_CLI_ON_UNBLOCK onUnblock;
    F_BKF_CH_CLI_ON_DISCONN onDisconn;
    F_BKF_CH_CLI_ON_DISCONNEX onDisconnEx;
    F_BKF_CH_CLI_ON_CONN onConn;
    uint8_t rsv1[0x10];
} BkfChCliEnableArg;

/* regType */
typedef struct tagBkfChCli BkfChCli;
typedef struct tagBkfChCliInitArg BkfChCliInitArg;

typedef BkfChCli *(*F_BKF_CH_CLI_INIT)(BkfChCliInitArg *arg);
typedef void (*F_BKF_CH_CLI_UNINIT)(BkfChCli *ch);
typedef uint32_t (*F_BKF_CH_CLI_ENABLE)(BkfChCli *ch, BkfChCliEnableArg *arg);
typedef uint32_t (*F_BKF_CH_CLI_SET_SELF_CID)(BkfChCli *ch, uint32_t selfCid);
typedef uint32_t (*F_BKF_CH_CLI_SET_SELF_URL)(BkfChCli *ch, BkfUrl *selfUrl);

typedef BkfChCliConnId *(*F_BKF_CH_CLI_CONN)(BkfChCli *ch, BkfUrl *urlServer);
typedef BkfChCliConnId *(*F_BKF_CH_CLI_CONNEX)(BkfChCli *ch, BkfUrl *urlServer, BkfUrl *urlClient);

typedef void (*F_BKF_CH_CLI_DISCONN)(BkfChCli *ch, BkfChCliConnId *connId);
typedef void (*F_BKF_CH_CLI_CONN_TIMOUT)(BkfChCli *ch, BkfUrl *urlServer);

typedef void *(*F_BKF_CH_CLI_MALLOC_DATA_BUF)(BkfChCliConnId *connId, int32_t dataBufLen);
typedef void (*F_BKF_CH_CLI_FREE_DATA_BUF)(BkfChCliConnId *connId, void *dataBuf);
typedef int32_t (*F_BKF_CH_CLI_READ)(BkfChCliConnId *connId, void *dataBuf, int32_t bufLen);
typedef int32_t (*F_BKF_CH_CLI_WRITE)(BkfChCliConnId *connId, void *dataBuf, int32_t dataLen);
typedef uint32_t (*F_BKF_CH_CLI_SET_CERA)(BkfChCli *ch, BkfCera *cera);
typedef uint32_t (*F_BKF_CH_CLI_UNSET_CERA)(BkfChCli *ch);

typedef struct tagBkfChCliTypeVTbl {
    char *name;
    uint8_t typeId; /* = url type */
    uint8_t pad1[3];

    F_BKF_CH_CLI_INIT init;
    F_BKF_CH_CLI_UNINIT uninit;
    F_BKF_CH_CLI_ENABLE enable;
    F_BKF_CH_CLI_SET_SELF_CID setSelfCid;
    F_BKF_CH_CLI_SET_SELF_URL setSelfUrl;
    F_BKF_CH_CLI_CONN conn;
    F_BKF_CH_CLI_DISCONN disConn;
    F_BKF_CH_CLI_CONN_TIMOUT connTimeOut;
    F_BKF_CH_CLI_MALLOC_DATA_BUF mallocDataBuf;
    F_BKF_CH_CLI_FREE_DATA_BUF freeDataBuf;
    F_BKF_CH_CLI_READ read;
    F_BKF_CH_CLI_WRITE write;
    F_BKF_CH_CLI_CONNEX connEx;
    F_BKF_CH_CLI_SET_CERA setCera;
    F_BKF_CH_CLI_UNSET_CERA unSetCera;
    uint8_t rsv1[0x10];
} BkfChCliTypeVTbl;

struct tagBkfChCliInitArg {
    BkfChCliMngInitArg *base;
    char *name;
    uint8_t rsv1[0x10];
};

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
