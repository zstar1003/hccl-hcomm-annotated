/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef WFW_MUX_H
#define WFW_MUX_H

#include "bkf_comm.h"
#include "bkf_mem.h"

/*
1. 层次关系为：
   ------------------------------------
   |app                               |
   ------------------------------------
   |imux(linux_epoll / uni_epoll ...) |
   ------------------------------------
   |mux                               |
   ------------------------------------

2. mux目前类型：
1) linux epoll
2) uni epoll
3) mesh epoll

3. 有如下的约束：
1) 三种epoll的event不冲突
2) 使用某个epoll特有的event，需要明确调用点在此种epoll的上下文
*/

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* common */
typedef struct tagWfwMux WfwMux;

/* init */
typedef void (*F_WFW_MUX_FD_PROC)(int fd, uint32_t curEvents, void *cookieAttachFd);

/*
1. 不同系统封装的多路复用器，除了类似linux的参数，还有一些其他的特定的参数。
2. 如果使用通用的 void* ，因为没有类型，需要强转，反而会引起混乱和误用
   因此，在 此处/基类 的地方定下 子类 的参数。架构上不是很好，容忍
*/
typedef struct tagWfwArgForMux WfwArgForMux;

/*
1. attach 对应 add; reattach 对应 mod; detach 对应 del
2. 如果只是fd相关的事件修改，需要调用reattach，而不能调用detach+attach。否则会有事件会掉
3. reattach也会传入cookie。一般来说，这个值和attach时相等。
   如果不相等，需要权衡好，新cookie指向对象是否能处理缓冲中已经存在的旧事件
4. 某些场景，app在线程处理函数中调用get接口获取回调，调用即可。
*/
typedef int (*F_WFW_IMUX_ATTACH_FD)(void *cookieInit, int fd, uint32_t interestedEvents, F_WFW_MUX_FD_PROC fdProc,
    void *fdCookie, WfwArgForMux *argForMuxOrNull);
typedef int (*F_WFW_IMUX_REATTACH_FD)(void *cookieInit, int fd, uint32_t interestedEvents, F_WFW_MUX_FD_PROC fdProc,
    void *fdCookie, WfwArgForMux *argForMuxOrNull);
typedef int (*F_WFW_IMUX_DETACH_FD)(void *cookieInit, int fd);
typedef void *(*F_WFW_IMUX_LOOP_RUN)(void *cookieInit);
typedef void (*F_WFW_IMUX_STOP_LOOP_RUN)(void *cookieInit);
typedef F_WFW_MUX_FD_PROC (*F_WFW_IMUX_GET_FD_PROC)(void *cookieInit, int fd, void **fdCookieOut,
    uint32_t *interestedEventsOut);
typedef struct tagWfwMuxInitArg {
    char *name;
    BkfMemMng *memMng;
    void *cookie;
    F_WFW_IMUX_ATTACH_FD attachFd;
    F_WFW_IMUX_REATTACH_FD reattachFd;
    F_WFW_IMUX_DETACH_FD detachFd;
    F_WFW_IMUX_LOOP_RUN loopRunOrNull;
    F_WFW_IMUX_STOP_LOOP_RUN stopLoopRunOrNull;
    F_WFW_IMUX_GET_FD_PROC getFdProcOrNull;
    uint8_t rsv1[0x10];
} WfwMuxInitArg;

/* attach/reattach */
struct tagWfwArgForMux {
    uint32_t selfCid; /* 代表一个不可重入的调度对象，即组件实例 */
    BOOL isMeshFd;    /* 指明fd的类型。因为mesh实现原因而输入 */
    BOOL notLogEvt;   /* 不记录event */
    uint8_t rsv1[0x10];
};

typedef void (*F_WFW_MUX_FD_DOBEFORE)(void *cookie);
typedef void (*F_WFW_MUX_FD_DOAFTER)(void *cookie);
/* func */
WfwMux *WfwMuxInit(WfwMuxInitArg *arg);
void WfwMuxUninit(WfwMux *mux);

int WfwMuxAttachFd(WfwMux *mux, int fd, uint32_t interestedEvents, F_WFW_MUX_FD_PROC fdProc, void *fdCookie,
    WfwArgForMux *argForMuxOrNull);
int WfwMuxReattachFd(WfwMux *mux, int fd, uint32_t interestedEvents, F_WFW_MUX_FD_PROC fdProc, void *fdCookie,
    WfwArgForMux *argForMuxOrNull);
int WfwMuxDetachFd(WfwMux *mux, int fd);
void *WfwMuxLoopRun(WfwMux *mux);
void WfwMuxStopLoopRun(WfwMux *mux);
F_WFW_MUX_FD_PROC WfwMuxGetFdProc(WfwMux *mux, int fd, void **fdCookieOut, uint32_t *interestedEventsOut);

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
