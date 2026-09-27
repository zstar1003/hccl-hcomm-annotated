/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef WFW_MUX_BASE_H
#define WFW_MUX_BASE_H

#include "wfw_mux.h"
#include "bkf_disp.h"
#include "bkf_log_cnt.h"
#include "bkf_log.h"
#include "bkf_pfm.h"
#include "v_avll.h"

/*
权衡，采用暴露结构，内嵌节点的方式
同时，不暴露到interface仓中
*/

#ifdef __cplusplus
extern "C" {
#endif
#pragma pack(4)

/* common */
typedef struct TagWfwMuxBaseInitArg {
    char *name;
    BkfMemMng *memMng;
    BkfDisp *disp;
    BkfLogCnt *logCnt;
    BkfLog *log;
    BkfPfm *pfm;
    uint32_t eventsAlwaysReport; /* 即使app不设置，os也会上报的事件 */
    void *cookie;
    F_WFW_MUX_FD_DOBEFORE beforeProc;
    F_WFW_MUX_FD_DOAFTER afterProc;
    uint8_t rsv1[0x10];
} WfwMuxBaseInitArg;

typedef struct tagWfwMuxBaseFd WfwMuxBaseFd;
#define WFW_MUX_BASE_FD_CACHE_SIZE 0x80
typedef struct TagWfwMuxBaseDispatchCtx WfwMuxBaseDispatchCtx;
typedef struct TagWfwMuxBase {
    uint16_t sign;
    uint8_t pad1[2];
    WfwMuxBaseInitArg argInit;
    AVLL_TREE fdSet;
#ifndef BKF_CUT_AVL_CACHE
    WfwMuxBaseFd *fdCache[WFW_MUX_BASE_FD_CACHE_SIZE];
#endif
    WfwMuxBaseDispatchCtx *dispatchCtx;
    uint8_t rsv1[0x10];
} WfwMuxBase;

struct tagWfwMuxBaseFd {
    uint16_t sign;
    uint8_t notLogEvt : 1;
    uint8_t rsv1 : 7;
    uint8_t pad1[1];
    WfwMuxBase *muxBase;
    AVLL_NODE setNode;
    int keyFd;
    uint32_t interestedEvents;
    F_WFW_MUX_FD_PROC proc;
    void *cookie;
};

typedef struct tagWfwMuxBaseEv {
    /* 取最小信息，curEvents + ptr */
    uint32_t curEvents;
    WfwMuxBaseFd *baseFd;
} WfwMuxBaseEv;

#define WFW_MUX_DISPATCH_DOBEFORE(muxBase)                                                                             \
    do {                                                                                                               \
        if ((muxBase)->argInit.beforeProc) {                                                                           \
            (muxBase)->argInit.beforeProc((muxBase)->argInit.cookie);                                                  \
        }                                                                                                              \
    } while (0)

#define WFW_MUX_DISPATCH_DOAFTER(muxBase)                                                                              \
    do {                                                                                                               \
        if ((muxBase)->argInit.afterProc) {                                                                            \
            (muxBase)->argInit.afterProc((muxBase)->argInit.cookie);                                                   \
        }                                                                                                              \
    } while (0)

uint32_t WfwMuxBaseInit(WfwMuxBase *muxBase, WfwMuxBaseInitArg *arg);
void WfwMuxBaseUninit(WfwMuxBase *muxBase);

uint32_t WfwMuxBaseInitFd(WfwMuxBase *muxBase, WfwMuxBaseFd *baseFd, int fd, uint32_t interestedEvents,
    F_WFW_MUX_FD_PROC proc, void *cookie, BOOL notLogEvt);
void WfwMuxBaseUninitFd(WfwMuxBaseFd *baseFd);
WfwMuxBaseFd *WfwMuxBaseFindFd(WfwMuxBase *muxBase, int fd);
WfwMuxBaseFd *WfwMuxBaseGetFirstFd(WfwMuxBase *muxBase, void **itorOutOrNull);
WfwMuxBaseFd *WfwMuxBaseGetNextFd(WfwMuxBase *muxBase, void **itorInOut);
void WfwMuxBaseDispatch(WfwMuxBase *muxBase, WfwMuxBaseEv *evs, int evCnt);

/* 下面几个函数必须线程安全 */
int WfwMuxBaseInitTrigFd(void);
void WfwMuxBaseUninitTrigFd(int fd);
void WfwMuxBaseStartTrig(int fd);
void WfwMuxBaseProcTrigFdEvent(int fd);

#pragma pack()
#ifdef __cplusplus
}
#endif

#endif
