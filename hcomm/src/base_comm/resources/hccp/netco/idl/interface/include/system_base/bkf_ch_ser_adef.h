/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_CH_SER_ADEF_H
#define BKF_CH_SER_ADEF_H

#include "bkf_comm.h"
#include "bkf_mem.h"
#include "bkf_disp.h"
#include "bkf_log.h"
#include "bkf_pfm.h"
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
typedef struct tagBkfChSerMng BkfChSerMng;
typedef struct tagBkfChSer BkfChSer;
typedef struct tagBkfChSerConnId BkfChSerConnId;

/* init */
typedef struct tagBkfChSerMngInitArg {
    char *name;
    BOOL dbgOn;
    BkfMemMng *memMng;
    BkfDisp *disp;
    BkfLogCnt *logCnt;
    BkfLog *log;
    BkfPfm *pfm;
    BkfXMap *xMap; /* v8tls场景输入fdXmap */
    BkfTmrMng *tmrMng;
    BkfJobMng *jobMng;

    /* 下面一些参数最好能放在子类中。因为现有代码已经调用，并且后续扩展也不频繁，容忍 */
    uint32_t jobTypeId;    /* 用于dms */
    uint32_t jobPrio;      /* 用于dms。优先级低于puber，尽量使得ACK等信息和数据一并发送 */
    int32_t dmsCliQueId;   /* 用于dms */
    uint8_t dmsMsgIntf;    /* 用于dms */
    uint8_t dmsMsgSubIntf; /* 用于dms */
    uint8_t pad1[2];
    WfwMux *mux;  /* 用于mesh/tcp */
    void *sslHnd; /* sslLib全局句柄 */
    uint8_t rsv1[0x10];
} BkfChSerMngInitArg;

/* regType */
typedef struct tagBkfChSerInitArg BkfChSerInitArg;
typedef struct tagBkfChSerEnableArg BkfChSerEnableArg;

#define BKF_CH_SER_MALLOC_DATA_BUF_LEN_MAX (0xffff - 0x40)

typedef BkfChSer *(*F_BKF_CH_SER_INIT)(BkfChSerInitArg *arg);
typedef void (*F_BKF_CH_SER_UNINIT)(BkfChSer *ch);
typedef uint32_t (*F_BKF_CH_SER_ENABLE)(BkfChSer *ch, BkfChSerEnableArg *arg);
typedef uint32_t (*F_BKF_CH_SER_SET_SELF_CID)(BkfChSer *ch, uint32_t selfCid);
typedef uint32_t (*F_BKF_CH_SER_LISTEN)(BkfChSer *ch, BkfUrl *urlSelf);
typedef void (*F_BKF_CH_SER_UNLISTEN)(BkfChSer *ch, BkfUrl *urlSelf);
typedef void *(*F_BKF_CH_SER_MALLOC_DATA_BUF)(BkfChSerConnId *connId, int32_t dataBufLen);
typedef void (*F_BKF_CH_SER_FREE_DATA_BUF)(BkfChSerConnId *connId, void *dataBuf);
typedef int32_t (*F_BKF_CH_SER_READ)(BkfChSerConnId *connId, void *dataBuf, int32_t bufLen);
typedef int32_t (*F_BKF_CH_SER_WRITE)(BkfChSerConnId *connId, void *dataBuf, int32_t dataLen);
typedef void (*F_BKF_CH_SER_CLOSE)(BkfChSerConnId *connId);
typedef uint32_t (*F_BKF_CH_SER_SET_CERA)(BkfChSer *ch, BkfCera *cera);
typedef uint32_t (*F_BKF_CH_SER_UNSET_CERA)(BkfChSer *ch);

typedef struct tagBkfChSerTypeVTbl {
    char *name;
    uint8_t typeId; /* = url type */
    uint8_t pad1[3];

    F_BKF_CH_SER_INIT init;
    F_BKF_CH_SER_UNINIT uninit;
    F_BKF_CH_SER_ENABLE enable;
    F_BKF_CH_SER_SET_SELF_CID setSelfCid; /* 不必须， 可以为空 */
    F_BKF_CH_SER_LISTEN listen;
    F_BKF_CH_SER_UNLISTEN unlisten;
    F_BKF_CH_SER_MALLOC_DATA_BUF mallocDataBuf;
    F_BKF_CH_SER_FREE_DATA_BUF freeDataBuf;
    F_BKF_CH_SER_READ read; /* 不是所有都支持读模式，可以为空 */
    F_BKF_CH_SER_WRITE write;
    F_BKF_CH_SER_CLOSE close;
    F_BKF_CH_SER_SET_CERA setCera;
    F_BKF_CH_SER_UNSET_CERA unSetCera;
    uint8_t rsv1[0x10];
} BkfChSerTypeVTbl;

struct tagBkfChSerInitArg {
    BkfChSerMngInitArg *base;
    char *name;
    uint8_t rsv1[0x10];
};

/* enable */
typedef BOOL (*F_BKF_CH_SER_ON_MAY_ACCEPT)(void *cookieEnable, BkfUrl *urlCli);
typedef void (*F_BKF_CH_SER_ON_CONN)(void *cookieEnable, BkfChSerConnId *connId, BkfUrl *urlCli);
typedef void (*F_BKF_CH_SER_ON_RCV_DATA)(void *cookieEnable, BkfChSerConnId *connId, void *data, int32_t dataLen);
typedef void (*F_BKF_CH_SER_ON_RCV_DATA_EVENT)(void *cookieEnable, BkfChSerConnId *connId);
typedef void (*F_BKF_CH_SER_ON_UNBLOCK)(void *cookieEnable, BkfChSerConnId *connId);
typedef void (*F_BKF_CH_SER_ON_DISCONN)(void *cookieEnable, BkfChSerConnId *connId);
typedef void (*F_BKF_CH_SER_ON_CONNEX)(void *cookieEnable, BkfChSerConnId *connId, BkfUrl *urlCli, BkfUrl *localUrl);

struct tagBkfChSerEnableArg {
    void *cookie;
    F_BKF_CH_SER_ON_MAY_ACCEPT onMayAccept;
    F_BKF_CH_SER_ON_CONN onConn;
    F_BKF_CH_SER_ON_RCV_DATA onRcvData;            /* 推数据模式 */
    F_BKF_CH_SER_ON_RCV_DATA_EVENT onRcvDataEvent; /* 拉/读数据模式 */
    F_BKF_CH_SER_ON_UNBLOCK onUnblock;
    F_BKF_CH_SER_ON_DISCONN onDisconn;
    F_BKF_CH_SER_ON_CONNEX onConnEx; /* 扩展api，增加本地收到上送的url */
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
