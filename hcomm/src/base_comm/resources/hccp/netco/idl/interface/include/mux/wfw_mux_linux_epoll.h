/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef WFW_MUX_LINUX_EPOLL_H
#define WFW_MUX_LINUX_EPOLL_H

#include "wfw_mux.h"
#include "bkf_disp.h"
#include "bkf_log_cnt.h"
#include "bkf_log.h"
#include "bkf_pfm.h"

/*
1. 一个linux epoll对象只和一个线程关联
*/

#ifdef __cplusplus
extern "C" {
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* common */
/**
 * @brief WfwMuxLe库句柄，封装linux epoll
 */
typedef struct tagWfwMuxLe WfwMuxLe;

/* init */
/**
 * @brief WfwMuxLe库初始化参数
 */
typedef struct tagWfwMuxLeInitArg {
    char *name;
    BOOL dbgOn;
    BkfMemMng *memMng;
    BkfDisp *disp;
    BkfLogCnt *logCnt;
    BkfLog *log;
    BkfPfm *pfm;
    void *cookie;
    int onceProcEvCntMax;
    BOOL injectEpoll;
    int injectEpollFd;
    uint8_t rsv1[0x10];
} WfwMuxLeInitArg;

/* func */
/**
 * @brief WfwMuxLe库初始化
 *
 * @param[in] *arg 参数
 * @return WfwMuxLe库句柄
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
WfwMuxLe *WfwMuxLeInit(WfwMuxLeInitArg *arg);

/**
 * @brief WfwMuxLe库反初始化
 *
 * @param[in] *muxLe WfwMuxLe库句柄
 * @return none
 */
void WfwMuxLeUninit(WfwMuxLe *muxLe);

/* loopRun返回cookieInit */
/**
 * @brief 构造linux epoll对应的mux虚表
 *
 * @param[in] *muxLe WfwMuxLe库句柄
 * @param[in] *temp 注入虚表结构体指针，堆变量或栈变量均可
 * @return 构造结果
 *   @retval VOS_NULL 失败
 *   @retval 其他 成功
 */
WfwMuxInitArg *WfwMuxLeBuildMuxInitArg(WfwMuxLe *muxLe, WfwMuxInitArg *temp);

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
}
#endif

#endif
