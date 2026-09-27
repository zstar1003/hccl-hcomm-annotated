/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((muxLe)->log)
#define BKF_MOD_NAME ((muxLe)->name)
#include <sys/epoll.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include "wfw_mux_base.h"
#include "wfw_linux_comm.h"
#include "bkf_str.h"
#include "bkf_assert.h"
#include "securec.h"
#include "wfw_mux_linux_epoll.h"

#ifdef __cplusplus
extern "C" {
#endif
#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
#define WFW_MUX_LE_SIGN (0xab91)
struct tagWfwMuxLe {
    uint16_t sign;
    uint8_t muxBaseInitOk : 1;
    uint8_t stopLoopRun : 1;
    uint8_t rsv1 : 6;
    uint8_t pad1[1];
    WfwMuxLeInitArg argInit;
    char *name;
    BkfLog *log;
    WfwMuxBase muxBase;
    int epFd;
    int trigFd;
    char muxName[BKF_NAME_LEN_MAX + 1];
    struct epoll_event *leEvs;
    WfwMuxBaseEv baseEvs[0];
};

#define WFW_MUX_LE_FD_SIGN (0xab93)
typedef struct tagWfwMuxLeFd {
    uint16_t sign;
    uint8_t baseFdInitOk : 1;
    uint8_t fdAdd2EpollOk : 1;
    uint8_t rsv1 : 6;
    uint8_t pad1[1];
    WfwMuxLe *muxLe;
    WfwMuxBaseFd baseFd;
} WfwMuxLeFd;

/* on msg  & proc */
STATIC uint32_t WfwMuxLeInitTrig(WfwMuxLe *muxLe);
STATIC void WfwMuxLeUninitTrig(WfwMuxLe *muxLe);
STATIC void WfwMuxLeTrigStopLoopRun(WfwMuxLe *muxLe);

/* data op */
STATIC WfwMuxLeFd *WfwMuxLeAddFd(WfwMuxLe *muxLe, int fd, uint32_t interestedEvents, F_WFW_MUX_FD_PROC proc,
    void *cookie, BOOL notLogEvt);
STATIC uint32_t WfwMuxLeUpdFd(WfwMuxLeFd *leFd, uint32_t interestedEvents, F_WFW_MUX_FD_PROC proc, void *cookie);
STATIC void WfwMuxLeDelFd(WfwMuxLeFd *leFd);
STATIC void WfwMuxLeDelAllFd(WfwMuxLe *muxLe);
STATIC WfwMuxLeFd *WfwMuxLeFindFd(WfwMuxLe *muxLe, int fd);
STATIC WfwMuxLeFd *WfwMuxLeGetFirstFd(WfwMuxLe *muxLe, void **itorOutOrNull);
STATIC WfwMuxLeFd *WfwMuxLeGetNextFd(WfwMuxLe *muxLe, void **itorInOut);

#pragma pack()

#endif

#if BKF_BLOCK("公有函数定义")
STATIC uint32_t WfwMuxLeInitMuxBase(WfwMuxLe *muxLe)
{
    WfwMuxBaseInitArg arg = {.name = muxLe->name,
        .memMng = muxLe->argInit.memMng,
        .disp = muxLe->argInit.disp,
        .logCnt = muxLe->argInit.logCnt,
        .log = muxLe->log,
        .pfm = muxLe->argInit.pfm,
        .eventsAlwaysReport = EPOLLERR | EPOLLHUP};
    uint32_t ret = WfwMuxBaseInit(&muxLe->muxBase, &arg);
    muxLe->muxBaseInitOk = BKF_COND_2BIT_FIELD(ret == BKF_OK);
    return ret;
}
STATIC void WfwMuxLeDoUninit(WfwMuxLe *muxLe)
{
    WfwMuxLeUninitTrig(muxLe);
    if (muxLe->muxBaseInitOk) {
        WfwMuxLeDelAllFd(muxLe);
        WfwMuxBaseUninit(&muxLe->muxBase);
    }

    if (!muxLe->argInit.injectEpoll && muxLe->epFd != -1) {
        (void)close(muxLe->epFd);
        muxLe->epFd = -1;
    }
    BkfStrDel(muxLe->name);
    BKF_SIGN_CLR(muxLe->sign);
    BKF_FREE(muxLe->argInit.memMng, muxLe);
}
WfwMuxLe *WfwMuxLeInit(WfwMuxLeInitArg *arg)
{
    BOOL paramIsInvalid = (arg == VOS_NULL) || (arg->name == VOS_NULL) || (arg->memMng == VOS_NULL) ||
                          (arg->disp == VOS_NULL) || (arg->logCnt == VOS_NULL) || (arg->log == VOS_NULL) ||
                          (arg->pfm == VOS_NULL) || (arg->onceProcEvCntMax <= 0);
    if (paramIsInvalid) {
        return VOS_NULL;
    }

    uint32_t len = sizeof(WfwMuxLe) +
                   (sizeof(WfwMuxBaseEv) + sizeof(struct epoll_event)) * ((uint32_t)arg->onceProcEvCntMax);
    WfwMuxLe *muxLe = BKF_MALLOC(arg->memMng, len);
    if (muxLe == VOS_NULL) {
        return VOS_NULL;
    }
    (void)memset_s(muxLe, len, 0, len);
    muxLe->epFd = -1;
    muxLe->trigFd = -1;
    muxLe->leEvs = (struct epoll_event *)(&muxLe->baseEvs[arg->onceProcEvCntMax]);
    muxLe->argInit = *arg;
    muxLe->name = BkfStrNew(arg->memMng, "%s", arg->name);
    muxLe->argInit.name = muxLe->name;
    muxLe->log = arg->log;

    uint32_t ret = WfwMuxLeInitMuxBase(muxLe);
    if (ret != BKF_OK) {
        WfwMuxLeDoUninit(muxLe);
        return VOS_NULL;
    }
    if (arg->injectEpoll) {
        muxLe->epFd = arg->injectEpollFd;
    } else {
        muxLe->epFd = epoll_create(1);
    }
    if (muxLe->epFd == -1) {
        BKF_ASSERT(0);
        WfwMuxLeDoUninit(muxLe);
        return VOS_NULL;
    }

    ret = WfwMuxLeInitTrig(muxLe);
    if (ret != BKF_OK) {
        WfwMuxLeDoUninit(muxLe);
        return VOS_NULL;
    }

    BKF_SIGN_SET(muxLe->sign, WFW_MUX_LE_SIGN);
    BKF_LOG_DEBUG(BKF_LOG_HND, "muxLe(%#x), epollFd %d init ok\n", BKF_MASK_ADDR(muxLe), muxLe->epFd);
    return muxLe;
}

void WfwMuxLeUninit(WfwMuxLe *muxLe)
{
    if ((muxLe == VOS_NULL) || !BKF_SIGN_IS_VALID(muxLe->sign, WFW_MUX_LE_SIGN)) {
        return;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "muxLe(%#x), 2uninit\n", BKF_MASK_ADDR(muxLe));

    WfwMuxLeDoUninit(muxLe);
}

static inline void WfwMuxLeOnFdLog(WfwMuxLe *muxLe, int fd, uint32_t interestedEvents, F_WFW_MUX_FD_PROC fdProc,
    void *fdCookie, WfwArgForMux *argForMuxOrNull, char *opTxt)
{
    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND,
        "muxLe(%#x), fd(%d)/op(%s)/interestedEvents(%#x), fdProc(%#x)/fdCookie(%#x), "
        "argForMuxOrNull(%#x), %s\n",
        BKF_MASK_ADDR(muxLe), fd, opTxt, interestedEvents, BKF_MASK_ADDR(fdProc), BKF_MASK_ADDR(fdCookie),
        BKF_MASK_ADDR(argForMuxOrNull), WfwGetBackTraceStr(buf, sizeof(buf)));
}
STATIC int WfwMuxLeOnAttachFd(WfwMuxLe *muxLe, int fd, uint32_t interestedEvents, F_WFW_MUX_FD_PROC fdProc,
    void *fdCookie, WfwArgForMux *argForMuxOrNull)
{
    if (muxLe == VOS_NULL || !BKF_SIGN_IS_VALID(muxLe->sign, WFW_MUX_LE_SIGN) || (fdProc == VOS_NULL)) {
        BKF_ASSERT(0);
        return -1;
    }
    WfwMuxLeOnFdLog(muxLe, fd, interestedEvents, fdProc, fdCookie, argForMuxOrNull, "attach");

    WfwMuxLeFd *leFd = WfwMuxLeFindFd(muxLe, fd);
    if (leFd != VOS_NULL) {
        BKF_ASSERT(0); /* 重复attach */
        return -1;
    }
    BOOL notLogEvt = (argForMuxOrNull != VOS_NULL) ? argForMuxOrNull->notLogEvt : VOS_FALSE;
    leFd = WfwMuxLeAddFd(muxLe, fd, interestedEvents, fdProc, fdCookie, notLogEvt);
    return (leFd != VOS_NULL) ? 0 : -1;
}

STATIC int WfwMuxLeOnReattachFd(WfwMuxLe *muxLe, int fd, uint32_t interestedEvents, F_WFW_MUX_FD_PROC fdProc,
    void *fdCookie, WfwArgForMux *argForMuxOrNull)
{
    if (muxLe == VOS_NULL || !BKF_SIGN_IS_VALID(muxLe->sign, WFW_MUX_LE_SIGN) || (fdProc == VOS_NULL)) {
        BKF_ASSERT(0);
        return -1;
    }
    WfwMuxLeOnFdLog(muxLe, fd, interestedEvents, fdProc, fdCookie, argForMuxOrNull, "reAttach");

    WfwMuxLeFd *leFd = WfwMuxLeFindFd(muxLe, fd);
    if (leFd == VOS_NULL) {
        BKF_ASSERT(0); /* reattach，leFd一定存在 */
        return -1;
    }
    uint32_t ret = WfwMuxLeUpdFd(leFd, interestedEvents, fdProc, fdCookie);
    return (ret == BKF_OK) ? 0 : -1;
}

STATIC int WfwMuxLeOnDetachFd(WfwMuxLe *muxLe, int fd)
{
    if (muxLe == VOS_NULL || !BKF_SIGN_IS_VALID(muxLe->sign, WFW_MUX_LE_SIGN)) {
        BKF_ASSERT(0);
        return -1;
    }
    WfwMuxLeOnFdLog(muxLe, fd, 0, VOS_NULL, VOS_NULL, VOS_NULL, "detach");

    WfwMuxLeFd *leFd = WfwMuxLeFindFd(muxLe, fd);
    if (leFd == VOS_NULL) {
        return -1;
    }
    WfwMuxLeDelFd(leFd);
    return 0;
}

STATIC void WfwMuxLeProcEv(WfwMuxLe *muxLe, int evCnt)
{
    int i;
    int evCntReal = 0;
    for (i = 0; i < evCnt; i++) {
        WfwMuxLeFd *leFd = muxLe->leEvs[i].data.ptr;
        if (!BKF_SIGN_IS_VALID(leFd->sign, WFW_MUX_LE_FD_SIGN)) {
            BKF_LOG_ERROR(BKF_LOG_HND, "muxLe(%#x), leFd(%#x)\n", BKF_MASK_ADDR(muxLe), BKF_MASK_ADDR(leFd));
            BKF_ASSERT(!muxLe->argInit.dbgOn);
            continue;
        }
        muxLe->baseEvs[evCntReal].baseFd = &leFd->baseFd;
        muxLe->baseEvs[evCntReal].curEvents = muxLe->leEvs[i].events;
        evCntReal++;
    }
    WfwMuxBaseDispatch(&muxLe->muxBase, muxLe->baseEvs, evCntReal);
}
STATIC void *WfwMuxLeOnLoopRun(WfwMuxLe *muxLe)
{
    if (muxLe == VOS_NULL || !BKF_SIGN_IS_VALID(muxLe->sign, WFW_MUX_LE_SIGN)) {
        BKF_ASSERT(0);
        return VOS_NULL;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "muxLe(%#x)\n", BKF_MASK_ADDR(muxLe));

    for (;;) {
        int evCnt = epoll_wait(muxLe->epFd, muxLe->leEvs, muxLe->argInit.onceProcEvCntMax, -1);
        if ((evCnt >= 0) && (evCnt <= muxLe->argInit.onceProcEvCntMax)) {
            WfwMuxLeProcEv(muxLe, evCnt);
            if (muxLe->stopLoopRun) {
                BKF_LOG_DEBUG(BKF_LOG_HND, "muxLe(%#x), stopLoopRun(%u)\n", BKF_MASK_ADDR(muxLe), muxLe->stopLoopRun);
                return muxLe->argInit.cookie;
            }
        } else if (WFW_RET_RETRY_NOW(evCnt, errno)) {
            continue;
        } else {
            BKF_LOG_ERROR(BKF_LOG_HND, "muxLe(%#x), evCnt(%d)/errno(%d), ng\n", BKF_MASK_ADDR(muxLe), evCnt, errno);
            BKF_ASSERT(0);
            /* 继续 */
        }
    }
}

STATIC void WfwMuxLeOnStopLoopRun(WfwMuxLe *muxLe)
{
    if (muxLe == VOS_NULL || !BKF_SIGN_IS_VALID(muxLe->sign, WFW_MUX_LE_SIGN)) {
        BKF_ASSERT(0);
        return;
    }

    WfwMuxLeTrigStopLoopRun(muxLe);
}

STATIC F_WFW_MUX_FD_PROC WfwMuxLeGetFdProc(WfwMuxLe *muxLe, int fd, void **fdCookieOut, uint32_t *interestedEventsOut)
{
    if (muxLe == VOS_NULL || !BKF_SIGN_IS_VALID(muxLe->sign, WFW_MUX_LE_SIGN) || (fdCookieOut == VOS_NULL) ||
        (interestedEventsOut == VOS_NULL)) {
        BKF_ASSERT(0);
        return VOS_NULL;
    }

    WfwMuxLeFd *leFd = WfwMuxLeFindFd(muxLe, fd);
    if (leFd != VOS_NULL) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "muxLe(%#x), fd(%d) match\n", BKF_MASK_ADDR(muxLe), fd);
        *fdCookieOut = leFd->baseFd.cookie;
        *interestedEventsOut = leFd->baseFd.interestedEvents;
        return leFd->baseFd.proc;
    } else {
        *fdCookieOut = VOS_NULL;
        *interestedEventsOut = 0;
        return VOS_NULL;
    }
}

WfwMuxInitArg *WfwMuxLeBuildMuxInitArg(WfwMuxLe *muxLe, WfwMuxInitArg *temp)
{
    if ((muxLe == VOS_NULL) || (temp == VOS_NULL)) {
        return VOS_NULL;
    }

    (void)memset_s(temp, sizeof(WfwMuxInitArg), 0, sizeof(WfwMuxInitArg));
    int32_t err = snprintf_truncated_s(muxLe->muxName, sizeof(muxLe->muxName), "%s_mux", muxLe->name);
    if (err <= 0) {
        return VOS_NULL;
    }
    temp->name = muxLe->muxName;
    temp->memMng = muxLe->argInit.memMng;
    temp->cookie = muxLe;
    temp->attachFd = (F_WFW_IMUX_ATTACH_FD)WfwMuxLeOnAttachFd;
    temp->reattachFd = (F_WFW_IMUX_REATTACH_FD)WfwMuxLeOnReattachFd;
    temp->detachFd = (F_WFW_IMUX_DETACH_FD)WfwMuxLeOnDetachFd;
    temp->loopRunOrNull = (F_WFW_IMUX_LOOP_RUN)WfwMuxLeOnLoopRun;
    temp->stopLoopRunOrNull = (F_WFW_IMUX_STOP_LOOP_RUN)WfwMuxLeOnStopLoopRun;
    temp->getFdProcOrNull = (F_WFW_IMUX_GET_FD_PROC)WfwMuxLeGetFdProc;
    return temp;
}

#endif

#if BKF_BLOCK("私有函数定义")
/* on msg  & proc */
STATIC void WfwMuxLeOnTrigFd(int fd, uint32_t curEvents, WfwMuxLe *muxLe)
{
    (void)curEvents;
    WfwMuxBaseProcTrigFdEvent(fd);
    muxLe->stopLoopRun = VOS_TRUE;
}
STATIC uint32_t WfwMuxLeInitTrig(WfwMuxLe *muxLe)
{
    int fd = WfwMuxBaseInitTrigFd();
    if (fd == -1) {
        return BKF_ERR;
    }

    WfwMuxLeFd *leFd = WfwMuxLeAddFd(muxLe, fd, EPOLLIN, (F_WFW_MUX_FD_PROC)WfwMuxLeOnTrigFd, muxLe, VOS_FALSE);
    if (leFd == VOS_NULL) {
        WfwMuxBaseUninitTrigFd(fd);
        return BKF_ERR;
    }

    muxLe->trigFd = fd;
    return BKF_OK;
}

STATIC void WfwMuxLeUninitTrig(WfwMuxLe *muxLe)
{
    if (muxLe->trigFd != -1) {
        WfwMuxLeFd *leFd = WfwMuxLeFindFd(muxLe, muxLe->trigFd);
        muxLe->trigFd = -1;
        if (leFd == VOS_NULL) {
            BKF_ASSERT(0);
            return;
        }

        WfwMuxLeDelFd(leFd);
    }
}

STATIC void WfwMuxLeTrigStopLoopRun(WfwMuxLe *muxLe)
{
    WfwMuxBaseStartTrig(muxLe->trigFd);
}

/* data op */
STATIC WfwMuxLeFd *WfwMuxLeAddFd(WfwMuxLe *muxLe, int fd, uint32_t interestedEvents, F_WFW_MUX_FD_PROC proc,
    void *cookie, BOOL notLogEvt)
{
    uint32_t len = sizeof(WfwMuxLeFd);
    WfwMuxLeFd *leFd = BKF_MALLOC(muxLe->argInit.memMng, len);
    if (leFd == VOS_NULL) {
        return VOS_NULL;
    }
    (void)memset_s(leFd, len, 0, len);
    leFd->muxLe = muxLe;

    uint32_t ret = WfwMuxBaseInitFd(&muxLe->muxBase, &leFd->baseFd, fd, interestedEvents, proc, cookie, notLogEvt);
    if (ret != BKF_OK) {
        WfwMuxLeDelFd(leFd);
        return VOS_NULL;
    }
    leFd->baseFdInitOk = VOS_TRUE;

    struct epoll_event ev;
    (void)memset_s(&ev, sizeof(struct epoll_event), 0, sizeof(struct epoll_event));
    ev.events = interestedEvents;
    if (muxLe->argInit.injectEpoll) {
        ev.data.fd = fd;
    } else {
        ev.data.ptr = leFd;
    }
    int ret2 = epoll_ctl(muxLe->epFd, EPOLL_CTL_ADD, fd, &ev);
    if (ret2 == -1) {
        BKF_ASSERT(!muxLe->argInit.dbgOn);
        WfwMuxLeDelFd(leFd);
        return VOS_NULL;
    }
    leFd->fdAdd2EpollOk = VOS_TRUE;

    BKF_SIGN_SET(leFd->sign, WFW_MUX_LE_FD_SIGN);
    return leFd;
}

STATIC uint32_t WfwMuxLeUpdFd(WfwMuxLeFd *leFd, uint32_t interestedEvents, F_WFW_MUX_FD_PROC proc, void *cookie)
{
    WfwMuxLe *muxLe = leFd->muxLe;
    WfwMuxBaseFd *baseFd = &leFd->baseFd;
    struct epoll_event ev;
    (void)memset_s(&ev, sizeof(struct epoll_event), 0, sizeof(struct epoll_event));
    ev.events = interestedEvents;
    if (muxLe->argInit.injectEpoll) {
        ev.data.fd = baseFd->keyFd;
    } else {
        ev.data.ptr = leFd;
    }
    int ret = epoll_ctl(muxLe->epFd, EPOLL_CTL_MOD, baseFd->keyFd, &ev);
    if (ret == -1) {
        BKF_ASSERT(!muxLe->argInit.dbgOn);
        return BKF_ERR;
    }

    baseFd->interestedEvents = interestedEvents;
    baseFd->proc = proc;
    baseFd->cookie = cookie;
    leFd->fdAdd2EpollOk = VOS_TRUE;
    return BKF_OK;
}

STATIC void WfwMuxLeDelFd(WfwMuxLeFd *leFd)
{
    WfwMuxLe *muxLe = leFd->muxLe;
    WfwMuxBaseFd *baseFd = &leFd->baseFd;
    if (leFd->fdAdd2EpollOk) {
        int ret = epoll_ctl(muxLe->epFd, EPOLL_CTL_DEL, baseFd->keyFd, VOS_NULL);
        if (ret == -1) {
            BKF_ASSERT(!muxLe->argInit.dbgOn);
        }
    }
    if (leFd->baseFdInitOk) {
        WfwMuxBaseUninitFd(baseFd);
    }
    BKF_SIGN_CLR(leFd->sign);
    BKF_FREE(muxLe->argInit.memMng, leFd);
}

STATIC void WfwMuxLeDelAllFd(WfwMuxLe *muxLe)
{
    WfwMuxLeFd *leFd = VOS_NULL;
    void *itor = VOS_NULL;
    for (leFd = WfwMuxLeGetFirstFd(muxLe, &itor); leFd != VOS_NULL; leFd = WfwMuxLeGetNextFd(muxLe, &itor)) {
        (void)WfwMuxLeDelFd(leFd);
    }
}

STATIC WfwMuxLeFd *WfwMuxLeFindFd(WfwMuxLe *muxLe, int fd)
{
    WfwMuxBaseFd *baseFd = WfwMuxBaseFindFd(&muxLe->muxBase, fd);
    return (baseFd != VOS_NULL) ? BKF_MBR_PARENT(WfwMuxLeFd, baseFd, baseFd) : VOS_NULL;
}

STATIC WfwMuxLeFd *WfwMuxLeGetFirstFd(WfwMuxLe *muxLe, void **itorOutOrNull)
{
    WfwMuxBaseFd *baseFd = WfwMuxBaseGetFirstFd(&muxLe->muxBase, itorOutOrNull);
    return (baseFd != VOS_NULL) ? BKF_MBR_PARENT(WfwMuxLeFd, baseFd, baseFd) : VOS_NULL;
}

STATIC WfwMuxLeFd *WfwMuxLeGetNextFd(WfwMuxLe *muxLe, void **itorInOut)
{
    WfwMuxBaseFd *baseFd = WfwMuxBaseGetNextFd(&muxLe->muxBase, itorInOut);
    return (baseFd != VOS_NULL) ? BKF_MBR_PARENT(WfwMuxLeFd, baseFd, baseFd) : VOS_NULL;
}

#endif

#ifdef __cplusplus
}
#endif
