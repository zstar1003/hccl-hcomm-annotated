/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdbool.h>
#include <sys/epoll.h>
#include "securec.h"
#include "wfw_mux.h"
#include "bkf_mem.h"
#include "bkf_tmr.h"

#include "wfw_twl_timer_by_linux.h"
#include "udf_twl_timer.h"
#include "udf_base_timer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BKF_ADPT_TWL_CTL_MAGIC (0xaebc9867)
#pragma pack(4)

/* 用于保存BKF相关信息和Twl timer信息之间的转换，作为适配BKF后接口对外统一控制块 */
typedef struct {
    uint32_t magicNum;

    /* BKF初始化信息 */
    void *appHandle;
    uint32_t appCid;
    BkfMemMng *memMng;       /* 内存管理库句柄 */
    WfwMux *mux;             /* 多路复用句柄 */
    unsigned char dbgSwitch; /* （必选参数）是否开启诊断 */
    unsigned char pad[3];    /* （填充参数）无需设置 */
    BkfDisp *disp;           /* （必选参数）诊断显示库句柄 */
    BkfLogCnt *logCnt;       /* （必选参数）logCnt库句柄 */
    BkfLog *log;             /* （必选参数）日志库句柄 */
    char name[UDF_TWL_TMR_NAME_LEN];

    /* Timer内部信息 */
    UdfTwlTimerHandle timerHandle;
    TimerFdEventCB timerFdProc;
    void *timerEventParam;
} BkfTwlAdpCtrl;

#pragma pack()

/* 适配BKF-> Twl_timer event处理回调 */
void BkfAdptTimerFdEventProc(int fd, uint32_t curEvents, void *cookieAttachFd)
{
    if (cookieAttachFd == NULL) {
        return;
    }

    BkfTwlAdpCtrl *adptCtrl = (BkfTwlAdpCtrl *)cookieAttachFd;
    if (adptCtrl->timerFdProc == NULL) {
        return;
    }

    struct epoll_event event;
    (void)memset_s(&event, sizeof(event), 0, sizeof(event));
    event.events = curEvents;
    event.data.fd = fd;

    adptCtrl->timerFdProc(&event, adptCtrl->timerEventParam);
    return;
}

/* 适配Twl_timer -> BKF 注册timer fd回调 */
uint32_t BkfAdptTwlTimerEpollCtl(void *appHandle, uint32_t cid, int fd, UdfTimerFdParams *timerFdParam)
{
    (void)cid;
    (void)fd;
    if ((appHandle == NULL) || (timerFdParam == NULL)) {
        return 0;
    }

    BkfTwlAdpCtrl *adptCtrl = (BkfTwlAdpCtrl *)appHandle;
    adptCtrl->timerFdProc = timerFdParam->timerFdProc;
    adptCtrl->timerEventParam = timerFdParam->timerEventParam;
    if (timerFdParam->operate == EPOLL_CTL_ADD) {
        return (uint32_t)WfwMuxAttachFd(adptCtrl->mux, timerFdParam->event.data.fd, timerFdParam->event.events,
            BkfAdptTimerFdEventProc, adptCtrl, NULL);
    } else if (timerFdParam->operate == EPOLL_CTL_DEL) {
        return (uint32_t)WfwMuxDetachFd(adptCtrl->mux, timerFdParam->event.data.fd);
    } else {
        return (uint32_t)0;
    }
}

void *BkfAdptTwlTimerMalloc(void *handle, size_t size, const char *name, uint32_t line)
{
    if (handle == NULL) {
        return NULL;
    }

    BkfTwlAdpCtrl *adptCtrl = (BkfTwlAdpCtrl *)handle;
    if (adptCtrl->magicNum != BKF_ADPT_TWL_CTL_MAGIC) {
        return NULL;
    }

    return BkfMalloc(adptCtrl->memMng, size, name, line);
}

void BkfAdptTwlTimerFree(void *handle, void *ptr, const char *name, uint32_t line)
{
    if (handle == NULL) {
        return;
    }

    BkfTwlAdpCtrl *adptCtrl = (BkfTwlAdpCtrl *)handle;
    if (adptCtrl->magicNum != BKF_ADPT_TWL_CTL_MAGIC) {
        return;
    }

    return BkfFree(adptCtrl->memMng, ptr, name, line);
}

bool BkfAdptCheckInitArg(BkfTwlTmrInitArg *initArg)
{
    if (initArg == NULL) {
        return false;
    }
    if (initArg->memMng == NULL) {
        return false;
    }
    if (initArg->mux == NULL) {
        return false;
    }

    if (initArg->appHandle == NULL) {
        return false;
    }

    if (initArg->log == NULL) {
        return false;
    }

    return true;
}

/* 适配BKF -> Twl_timer 初始化时间轮定时器 */
void *BkfAdptTwlTimerInit(BkfTwlTmrInitArg *initArg)
{
    if (!BkfAdptCheckInitArg(initArg)) {
        return NULL;
    }

    BKF_LOG_DEBUG(initArg->log, "BkfTwlTimer  init start.\n");
    BkfTwlAdpCtrl *adptCtrl = BkfMalloc(initArg->memMng, sizeof(BkfTwlAdpCtrl), TIMER_FNAME, TIMER_FLINE);
    if (adptCtrl == NULL) {
        return NULL;
    }
    (void)memset_s(adptCtrl, sizeof(BkfTwlAdpCtrl), 0, sizeof(BkfTwlAdpCtrl));
    adptCtrl->magicNum = BKF_ADPT_TWL_CTL_MAGIC;
    adptCtrl->memMng = initArg->memMng;
    adptCtrl->mux = initArg->mux;
    adptCtrl->appHandle = initArg->appHandle;
    adptCtrl->appCid = initArg->appCid;
    adptCtrl->dbgSwitch = initArg->dbgSwitch;
    adptCtrl->disp = initArg->disp;
    adptCtrl->logCnt = initArg->logCnt;
    adptCtrl->log = initArg->log;
    (void)memcpy_s(adptCtrl->name, UDF_TWL_TMR_NAME_LEN, initArg->name, BKF_TWL_TMR_NAME_LEN);
    adptCtrl->name[UDF_TWL_TMR_NAME_LEN - 1] = '\0';

    UdfTimerInitParam innerInit;
    (void)memset_s(&innerInit, sizeof(UdfTimerInitParam), 0, sizeof(UdfTimerInitParam));
    innerInit.appHandle = (void *)adptCtrl;
    innerInit.appCid = initArg->appCid;
    innerInit.memHandle = (void *)adptCtrl;
    innerInit.epollCtlCB = BkfAdptTwlTimerEpollCtl;
    innerInit.freeCB = BkfAdptTwlTimerFree;
    innerInit.mallocCB = BkfAdptTwlTimerMalloc;
    (void)memcpy_s(innerInit.timerInstName, UDF_TWL_TMR_NAME_LEN, adptCtrl->name, UDF_TWL_TMR_NAME_LEN);
    innerInit.timerInstName[UDF_TWL_TMR_NAME_LEN - 1] = '\0';

    uint32_t ret = UdfTwlTimerCreate(&innerInit, &adptCtrl->timerHandle);
    if (ret != 0) {
        BkfFree(initArg->memMng, adptCtrl, TIMER_FNAME, TIMER_FLINE);
        adptCtrl = NULL;
    }

    return (BkfTwlTmrHandle)adptCtrl;
}

void *BkfAdptTwlTimerStartTimer(BkfTwlTmrHandle twlWfwAdapt, F_BKF_TMR_TIMEOUT_PROC proc, UdfTimerMode tmrMode,
    uint32_t intervalMs, void *param)
{
    if (twlWfwAdapt == NULL) {
        return NULL;
    }

    BkfTwlAdpCtrl *adptCtrl = (BkfTwlAdpCtrl *)twlWfwAdapt;
    if (adptCtrl->magicNum != BKF_ADPT_TWL_CTL_MAGIC) {
        return NULL;
    }
    BKF_LOG_DEBUG(adptCtrl->log, "BkfTwlTimer StartTimer, mode = %d, intervalMs = %d.\n", tmrMode, intervalMs);

    /* 创建定时器实例 */
    UdfTmrInstParam timerParam;
    (void)memset_s(&timerParam, sizeof(UdfTmrInstParam), 0, sizeof(UdfTmrInstParam));
    timerParam.interval = intervalMs;
    timerParam.callBack = proc;
    timerParam.usrData = param;
    timerParam.tmrMode = tmrMode;

    UdfTmrInstHandle timerId = NULL;
    uint32_t ret = UdfTimerInstCreate(adptCtrl->timerHandle, &timerParam, &timerId);
    if (ret != 0) {
        return NULL;
    }

    return timerId;
}

void *BkfAdptTwlTimerStartLoopTimer(BkfTwlTmrHandle twlWfwAdapt, F_BKF_TMR_TIMEOUT_PROC proc, uint32_t intervalMs,
    void *param)
{
    return BkfAdptTwlTimerStartTimer(twlWfwAdapt, proc, UDF_TIMER_MODE_PERIOD, intervalMs, param);
}

void *BkfAdptTwlTimerStartOnceTimer(BkfTwlTmrHandle twlWfwAdapt, F_BKF_TMR_TIMEOUT_PROC proc, uint32_t intervalMs,
    void *param)
{
    return BkfAdptTwlTimerStartTimer(twlWfwAdapt, proc, UDF_TIMER_MODE_ONE_SHOT, intervalMs, param);
}

void BkfAdptTwlTimerStopTimer(BkfTwlTmrHandle twlWfwAdapt, UdfTmrInstHandle timerId)
{
    if (twlWfwAdapt == NULL) {
        return;
    }
    BkfTwlAdpCtrl *adptCtrl = (BkfTwlAdpCtrl *)twlWfwAdapt;
    if (adptCtrl->magicNum != BKF_ADPT_TWL_CTL_MAGIC) {
        return;
    }

    BKF_LOG_DEBUG(adptCtrl->log, "BkfTwlTimer Stop Timer,  timerId = 0x%x.\n", BKF_MASK_ADDR(timerId));
    UdfTimerInstDelete(timerId);
    return;
}

uint32_t BkfAdptTwlTimerRefreshTimer(BkfTwlTmrHandle twlWfwAdapt, BkfTmrId *timerId, uint32_t newIntervalMs)
{
    if (twlWfwAdapt == NULL) {
        return -1;
    }
    BkfTwlAdpCtrl *adptCtrl = (BkfTwlAdpCtrl *)twlWfwAdapt;
    if (adptCtrl->magicNum != BKF_ADPT_TWL_CTL_MAGIC) {
        return -1;
    }
    BKF_LOG_DEBUG(adptCtrl->log, "BkfTwlTimer RefreshTimer: newIntervalMs = %u,  timerId = 0x%x.\n", newIntervalMs,
        BKF_MASK_ADDR(timerId));

    if (timerId == NULL) {
        return -1;
    }

    return UdfTimerInstResize((UdfTmrInstHandle)timerId, newIntervalMs);
}

void *BkfAdptTwlTimerGetTimerUsrData(BkfTwlTmrHandle twlWfwAdapt, BkfTmrId *timerId)
{
    if (twlWfwAdapt == NULL) {
        return NULL;
    }

    BkfTwlAdpCtrl *adptCtrl = (BkfTwlAdpCtrl *)twlWfwAdapt;
    if (adptCtrl->magicNum != BKF_ADPT_TWL_CTL_MAGIC) {
        return NULL;
    }

    if (timerId == NULL) {
        return NULL;
    }

    return UdfTwlTimerGetInstUsrData((UdfTmrInstHandle)timerId);
}

/* 构造BKF BkfTmrMng，挂接相关的虚接口 */
BkfTmrMng *BkfAdptTwlTimerInitTmrMng(BkfTwlTmrHandle twlWfwAdapt)
{
    if (twlWfwAdapt == NULL) {
        return NULL;
    }

    BkfTwlAdpCtrl *adptCtrl = (BkfTwlAdpCtrl *)twlWfwAdapt;
    if (adptCtrl->magicNum != BKF_ADPT_TWL_CTL_MAGIC) {
        return NULL;
    }
    BKF_LOG_DEBUG(adptCtrl->log, "BkfTwlTimer Init TmrMng\n");

    BkfITmr iTmr;
    iTmr.name = adptCtrl->name;
    iTmr.memMng = adptCtrl->memMng;
    iTmr.cookie = adptCtrl;
    iTmr.startLoop = BkfAdptTwlTimerStartLoopTimer;
    iTmr.startOnce = BkfAdptTwlTimerStartOnceTimer;
    iTmr.stop = BkfAdptTwlTimerStopTimer;
    iTmr.refreshOrNull = BkfAdptTwlTimerRefreshTimer;
    iTmr.getParamStartOrNull = BkfAdptTwlTimerGetTimerUsrData;

    return BkfTmrInit(&iTmr);
}

/* 析构BKF BkfTmrMng */
void BkfAdptTwlTimerUnInit(BkfTwlTmrHandle twlWfwAdapt)
{
    if (twlWfwAdapt == NULL) {
        return;
    }
    BkfTwlAdpCtrl *adptCtrl = (BkfTwlAdpCtrl *)twlWfwAdapt;
    if (adptCtrl->magicNum != BKF_ADPT_TWL_CTL_MAGIC) {
        return;
    }

    BKF_LOG_DEBUG(adptCtrl->log, "BkfTwlTimer UnInit\n");
    UdfTwlTimerDestroy(adptCtrl->timerHandle);
    BkfFree(adptCtrl->memMng, adptCtrl, TIMER_FNAME, TIMER_FLINE);
    return;
}

/* 析构BKF BkfTmrMng */
void BkfAdptBkfTmrMngUnInit(BkfTmrMng *bkfTmrMng)
{
    BkfTmrUninit(bkfTmrMng);
    return;
}

uint32_t BkfAdptTwlGetTimerRunningCount(BkfTwlTmrHandle twlWfwAdapt)
{
    if (twlWfwAdapt == NULL) {
        return 0;
    }
    BkfTwlAdpCtrl *adptCtrl = (BkfTwlAdpCtrl *)twlWfwAdapt;
    if (adptCtrl->magicNum != BKF_ADPT_TWL_CTL_MAGIC) {
        return 0;
    }

    return UdfTwlTimerRunningInstGetCnt(adptCtrl->timerHandle);
}

#ifdef __cplusplus
}
#endif
