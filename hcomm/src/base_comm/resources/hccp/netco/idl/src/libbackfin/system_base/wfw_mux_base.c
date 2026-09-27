/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((muxBase)->argInit.log)
#define BKF_MOD_NAME ((muxBase)->argInit.name)
#define BKF_LINE (__LINE__ + 10000)

#include <sys/timerfd.h>
#include <unistd.h>
#include <stdint.h>
#include <pthread.h>
#include "wfw_linux_comm.h"
#include "bkf_bas_type_mthd.h"
#include "bkf_assert.h"
#include "securec.h"
#include "wfw_mux_base.h"

#ifdef __cplusplus
extern "C" {
#endif
#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
#define WFW_MUX_BASE_SIGN (0xaa98)
#define WFW_MUX_BASE_FD_SIGN (0xaa99)

static inline BOOL WfwMuxBaseIsInvalid(WfwMuxBase *muxBase)
{
    return (muxBase == VOS_NULL) || !BKF_SIGN_IS_VALID(muxBase->sign, WFW_MUX_BASE_SIGN);
}

static inline BOOL WfwMuxBaseFdIsInvalid(WfwMuxBaseFd *baseFd)
{
    return (baseFd == VOS_NULL) || !BKF_SIGN_IS_VALID(baseFd->sign, WFW_MUX_BASE_FD_SIGN);
}

static inline uint8_t WfwMuxBaseGetFdCacheIdx(int fd)
{
    return (BKF_GET_U32_FOLD8_VAL(fd) & (WFW_MUX_BASE_FD_CACHE_SIZE - 1));
}

struct TagWfwMuxBaseDispatchCtx {
    WfwMuxBaseEv *evs; /* [min, max] */
    int idxMin;
    int idxMax;
};

#pragma pack()
#endif

#if BKF_BLOCK("公有函数定义")
STATIC int32_t WfwMuxBaseCmpFd(const int *key1Input, const int *key2InDs)
{
    return BKF_CMP_X(*key1Input, *key2InDs);
}
uint32_t WfwMuxBaseInit(WfwMuxBase *muxBase, WfwMuxBaseInitArg *arg)
{
    BOOL paramIsInvalid = (muxBase == VOS_NULL) || (arg == VOS_NULL) || (arg->name == VOS_NULL) ||
                          (arg->memMng == VOS_NULL) || (arg->disp == VOS_NULL) || (arg->logCnt == VOS_NULL) ||
                          (arg->pfm == VOS_NULL);
    if (paramIsInvalid) {
        return BKF_ERR;
    }

    (void)memset_s(muxBase, sizeof(WfwMuxBase), 0, sizeof(WfwMuxBase));
    muxBase->argInit = *arg;
    VOS_AVLL_INIT_TREE(muxBase->fdSet, (AVLL_COMPARE)WfwMuxBaseCmpFd, BKF_OFFSET(WfwMuxBaseFd, keyFd),
        BKF_OFFSET(WfwMuxBaseFd, setNode));
    BKF_SIGN_SET(muxBase->sign, WFW_MUX_BASE_SIGN);
    return BKF_OK;
}

void WfwMuxBaseUninit(WfwMuxBase *muxBase)
{
    if (WfwMuxBaseIsInvalid(muxBase)) {
        return;
    }

    WfwMuxBaseFd *baseFd = VOS_NULL;
    void *itor = VOS_NULL;
    for (baseFd = WfwMuxBaseGetFirstFd(muxBase, &itor); baseFd != VOS_NULL;
         baseFd = WfwMuxBaseGetNextFd(muxBase, &itor)) {
        BKF_ASSERT(0); /* 上层忘记释放，断言 */
        WfwMuxBaseUninitFd(baseFd);
    }
    BKF_SIGN_CLR(muxBase->sign);
}

uint32_t WfwMuxBaseInitFd(WfwMuxBase *muxBase, WfwMuxBaseFd *baseFd, int fd, uint32_t interestedEvents,
    F_WFW_MUX_FD_PROC proc, void *cookie, BOOL notLogEvt)
{
    BOOL paramIsInvalid = WfwMuxBaseIsInvalid(muxBase) || (baseFd == VOS_NULL) || (proc == VOS_NULL);
    if (paramIsInvalid) {
        return BKF_ERR;
    }

    (void)memset_s(baseFd, sizeof(WfwMuxBaseFd), 0, sizeof(WfwMuxBaseFd));
    baseFd->notLogEvt = BKF_COND_2BIT_FIELD(notLogEvt);
    baseFd->muxBase = muxBase;
    VOS_AVLL_INIT_NODE(baseFd->setNode);
    baseFd->keyFd = fd;
    baseFd->interestedEvents = interestedEvents;
    baseFd->proc = proc;
    baseFd->cookie = cookie;
    if (!VOS_AVLL_INSERT(muxBase->fdSet, baseFd->setNode)) {
        return BKF_ERR;
    }

    BKF_SIGN_SET(baseFd->sign, WFW_MUX_BASE_FD_SIGN);
#ifndef BKF_CUT_AVL_CACHE
    uint8_t idx = WfwMuxBaseGetFdCacheIdx(fd);
    muxBase->fdCache[idx] = baseFd;
#endif
    return BKF_OK;
}

STATIC void WfwMuxBaseDelEvInDispatchCtx(WfwMuxBase *muxBase, int fdNeedDelEv)
{
    WfwMuxBaseDispatchCtx *ctx = muxBase->dispatchCtx;
    if (ctx == VOS_NULL) {
        return;
    }

    int chkIdx = ctx->idxMin;
    while ((chkIdx >= 0) && (chkIdx <= ctx->idxMax)) {
        WfwMuxBaseFd *chkBaseFd = ctx->evs[chkIdx].baseFd;
        if (WfwMuxBaseFdIsInvalid(chkBaseFd)) {
            BKF_ASSERT(0);
            chkIdx++;
            continue;
        }

        BOOL needDel = (chkBaseFd->keyFd == fdNeedDelEv);
        if (!needDel) {
            chkIdx++;
            continue;
        }

        BKF_LOG_DEBUG(BKF_LOG_HND, "muxBase(%#x), fdNeedDelEv(%d)/chkIdx(%d)\n", BKF_MASK_ADDR(muxBase), fdNeedDelEv,
            chkIdx);
        /*
        删除数组当前位置数据，数组数据整体往前替换一格
        [chkIdx, idxMax] <== [chkIdx + 1, idxMax]
        */
        size_t destSize = sizeof(WfwMuxBaseEv) * ((uint32_t)(ctx->idxMax - chkIdx + 1));
        size_t srcSize = sizeof(WfwMuxBaseEv) * ((uint32_t)(ctx->idxMax - chkIdx));
        errno_t err = memmove_s(&ctx->evs[chkIdx], destSize, &ctx->evs[chkIdx + 1], srcSize);
        if (err != EOK) {
            BKF_ASSERT(0);
            /* 继续 */
            chkIdx++;
            continue;
        }
        ctx->idxMax--;
    }
}
void WfwMuxBaseUninitFd(WfwMuxBaseFd *baseFd)
{
    if (WfwMuxBaseFdIsInvalid(baseFd)) {
        return;
    }

    WfwMuxBase *muxBase = baseFd->muxBase;
    WfwMuxBaseDelEvInDispatchCtx(muxBase, baseFd->keyFd);
#ifndef BKF_CUT_AVL_CACHE
    uint8_t idx = WfwMuxBaseGetFdCacheIdx(baseFd->keyFd);
    if (muxBase->fdCache[idx] == baseFd) {
        muxBase->fdCache[idx] = VOS_NULL;
    }
#endif
    VOS_AVLL_DELETE(baseFd->muxBase->fdSet, baseFd->setNode);
    BKF_SIGN_CLR(baseFd->sign);
}

WfwMuxBaseFd *WfwMuxBaseFindFd(WfwMuxBase *muxBase, int fd)
{
    if (WfwMuxBaseIsInvalid(muxBase)) {
        return VOS_NULL;
    }
#ifndef BKF_CUT_AVL_CACHE
    uint8_t idx = WfwMuxBaseGetFdCacheIdx(fd);
    BOOL hit = (muxBase->fdCache[idx] != VOS_NULL) && (muxBase->fdCache[idx]->keyFd == fd);
    if (hit) {
        return muxBase->fdCache[idx];
    }
#endif
    WfwMuxBaseFd *baseFd = VOS_AVLL_FIND(muxBase->fdSet, &fd);
    if (baseFd == VOS_NULL) {
        return VOS_NULL;
    }
#ifndef BKF_CUT_AVL_CACHE
    muxBase->fdCache[idx] = baseFd;
#endif
    return baseFd;
}

WfwMuxBaseFd *WfwMuxBaseGetFirstFd(WfwMuxBase *muxBase, void **itorOutOrNull)
{
    if (WfwMuxBaseIsInvalid(muxBase)) {
        return VOS_NULL;
    }

    WfwMuxBaseFd *baseFd = VOS_AVLL_FIRST(muxBase->fdSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (baseFd != VOS_NULL) ? VOS_AVLL_NEXT(muxBase->fdSet, baseFd->setNode) : VOS_NULL;
    }
    return baseFd;
}

WfwMuxBaseFd *WfwMuxBaseGetNextFd(WfwMuxBase *muxBase, void **itorInOut)
{
    if (WfwMuxBaseIsInvalid(muxBase)) {
        return VOS_NULL;
    }

    WfwMuxBaseFd *baseFd = (*itorInOut);
    *itorInOut = (baseFd != VOS_NULL) ? VOS_AVLL_NEXT(muxBase->fdSet, baseFd->setNode) : VOS_NULL;
    return baseFd;
}

STATIC void WfwMuxBaseProcOneEv(WfwMuxBase *muxBase, WfwMuxBaseEv *ev)
{
    uint32_t curEvents = ev->curEvents;
    WfwMuxBaseFd *baseFd = ev->baseFd;
    if (baseFd == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "baseFd(%#x), ng\n", BKF_MASK_ADDR(baseFd));
        BKF_ASSERT(0);
        return;
    }
    if (WfwMuxBaseFdIsInvalid(baseFd)) {
        BKF_LOG_ERROR(BKF_LOG_HND, "baseFd(%#x, %#x), ng\n", BKF_MASK_ADDR(baseFd), (baseFd)->sign);
        BKF_ASSERT(0);
        return;
    }
    if (!baseFd->notLogEvt) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "muxBase(%#x), baseFd(%#x)/fd(%d)/curEvents(%#x)\n", BKF_MASK_ADDR(muxBase),
            BKF_MASK_ADDR(baseFd), baseFd->keyFd, curEvents);
    }

    /* 过滤事件 */
    curEvents &= (baseFd->interestedEvents | muxBase->argInit.eventsAlwaysReport);
    /* 回调 必须 放在最后一行 */
    baseFd->proc(baseFd->keyFd, curEvents, baseFd->cookie);
    return;
}
void WfwMuxBaseDispatch(WfwMuxBase *muxBase, WfwMuxBaseEv *evs, int evCnt)
{
    BOOL paramIsInvalid = WfwMuxBaseIsInvalid(muxBase) || (evs == VOS_NULL) || (evCnt < 0);
    if (paramIsInvalid) {
        return;
    }
    /* 对log稍作过滤 */
    BOOL notLog = (evCnt == 1) && evs->baseFd->notLogEvt;
    if (!notLog) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "muxBase(%#x), evs(%#x)/evCnt(%d)\n", BKF_MASK_ADDR(muxBase), BKF_MASK_ADDR(evs),
            evCnt);
    }
    WfwMuxBaseDispatchCtx ctx = {.evs = evs, .idxMin = 0, .idxMax = (evCnt - 1)};
    muxBase->dispatchCtx = &ctx;
    WFW_MUX_DISPATCH_DOBEFORE(muxBase);
    while (ctx.idxMin <= ctx.idxMax) {
        WfwMuxBaseEv *ev = &(ctx.evs[ctx.idxMin]);
        ctx.idxMin++;
        WfwMuxBaseProcOneEv(muxBase, ev);
        ev = VOS_NULL; /* 处理完毕后 ev 可能无效，赋空防止误用 */
    }
    muxBase->dispatchCtx = VOS_NULL;
    WFW_MUX_DISPATCH_DOAFTER(muxBase);
}

int WfwMuxBaseInitTrigFd(void)
{
    int fd = timerfd_create(CLOCK_MONOTONIC, 0);
    if (fd == -1) {
        return -1;
    }

    int ret = WfwSetFdNonBlock(fd);
    if (ret == -1) {
        (void)close(fd);
        return -1;
    }

    struct itimerspec newValue;
    (void)memset_s(&newValue, sizeof(struct itimerspec), 0, sizeof(struct itimerspec));
    ret = timerfd_settime(fd, 0, &newValue, NULL);
    if (ret == -1) {
        (void)close(fd);
        return -1;
    }

    return fd;
}

void WfwMuxBaseUninitTrigFd(int fd)
{
    (void)close(fd);
}

void WfwMuxBaseStartTrig(int fd)
{
    struct itimerspec newValue;
    (void)memset_s(&newValue, sizeof(struct itimerspec), 0, sizeof(struct itimerspec));
    newValue.it_value.tv_sec = 0;
    newValue.it_value.tv_nsec = 1;
    newValue.it_interval.tv_sec = 0;
    newValue.it_interval.tv_nsec = 0;
    int ret = timerfd_settime(fd, 0, &newValue, NULL);
    if (ret == -1) {
        BKF_ASSERT(0);
    }
}

void WfwMuxBaseProcTrigFdEvent(int fd)
{
    uint64_t temp;
    ssize_t rlen = read(fd, &temp, sizeof(temp));
    if (rlen != (ssize_t)sizeof(temp)) {
        /* 继续 */
    }
}

#endif

#ifdef __cplusplus
}
#endif
