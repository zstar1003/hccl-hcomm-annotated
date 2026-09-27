/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_BASE_TIMER_H
#define UDF_BASE_TIMER_H

#include <stdint.h>
#include <sys/timerfd.h>

#include "vos_list.h"
#include "udf_twl_timer.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(4)

typedef VOS_LIST_HEAD_S VOS_LIST_NODE_S;

/* 基础定时器模块通用返回值 */
#define BASE_TIMER_ERRNO_OK 0
#define BASE_TIMER_ERRNO_ERR ((uint32_t)(-1))
#define BASE_TIMER_ERR_BASE (0x100)

/* 0x03016901 属性句柄非法 */
#define BASE_TIMER_ERRNO_ATTR_INVALID (BASE_TIMER_ERR_BASE + 0x1)

/* 0x03016902 定时器句柄非法 */
#define BASE_TIMER_ERRNO_HANDLE_INVALID (BASE_TIMER_ERR_BASE + 0x2)

/* 0x03016903 入参非法 */
#define BASE_TIMER_ERRNO_PARAM_INVALID (BASE_TIMER_ERR_BASE + 0x3)

/* 0x03016904 用法不支持 */
#define BASE_TIMER_ERRNO_NOT_SUPPORT (BASE_TIMER_ERR_BASE + 0x4)

/* 0x03016905 魔术字被破坏 */
#define BASE_TIMER_ERRNO_MAGIC_DAMAGED (BASE_TIMER_ERR_BASE + 0x5)

/* 0x03016906 用户回调执行失败 */
#define BASE_TIMER_ERRNO_USR_FAILED (BASE_TIMER_ERR_BASE + 0x6)

/* 0x03016907 系统定时器创建失败 */
#define BASE_TIMER_ERRNO_TIMER_CREATE_FAILED (BASE_TIMER_ERR_BASE + 0x7)

/* 0x03016908 系统定时器获取时间失败 */
#define BASE_TIMER_ERRNO_TIMER_GET_FAILED (BASE_TIMER_ERR_BASE + 0x8)

/* 0x03016909 系统定时器设置时间失败 */
#define BASE_TIMER_ERRNO_TIMER_SET_FAILED (BASE_TIMER_ERR_BASE + 0x9)

/* 0x0301690a 系统定时器读取时间失败 */
#define BASE_TIMER_ERRNO_TIMER_READ_FAILED (BASE_TIMER_ERR_BASE + 0xa)
#define BASE_TIMER_ERRNO_INVAL (BASE_TIMER_ERR_BASE + 0xb)
#define BASE_TIMER_ERRNO_NOMEM (BASE_TIMER_ERR_BASE + 0xc)

#define BASE_TIMER_NAME_LEN_MAX 16
#define BASE_TIMER_LAST_PROC_TIME_LEN 32

#define BASE_TIMER_DEL_TAG_FALSE 0
#define BASE_TIMER_DEL_TAG_TRUE 1

typedef void *BaseTimerAttr;

typedef void *BaseTimerHandle;

typedef uint32_t (*BaseTimerProc)(BaseTimerHandle handle, uint64_t rdVal, void *param);

typedef void (*BaseTimerParamDestruct)(void *param);

/* 定时器用户回调 */
typedef struct {
    BaseTimerProc proc;
    void *param;
} BaseTimerUsrCb;

/* 定时器模式 */
typedef enum { BASE_TIMER_MODE_PERIOD, BASE_TIMER_MODE_ONE_SHOT, BASE_TIMER_MODE_BUTT } BaseTimerMode;

/* 基础定时器可选属性 */
typedef enum {
    BASE_TIMER_ATTR_MODE,                /* 定时器模式，BaseTimerMode */
    BASE_TIMER_ATTR_NAME,                /* 定时器名称，char* */
    BASE_TIMER_ATTR_USRPARAM_DESTRUCTOR, /* 定时器用户参数析构回调，BaseTimerParamDestruct */
    BASE_TIMER_ATTR_BUTT
} BaseTimerAttrType;

typedef enum {
    BASE_TIMER_STATUS_STOPED, /* 停止中 */
    BASE_TIMER_STATUS_ACTIVE, /* 运行中 */
    BASE_TIMER_STATUS_BUTT
} BaseTimerStatus;

typedef struct {
    char tmrName[BASE_TIMER_NAME_LEN_MAX];
    BaseTimerMode mode;
    BaseTimerParamDestruct destructor;
} BaseTimerParam;

typedef struct {
    uint64_t timeOutCount;
    BaseTimerStatus status;
    uint32_t maxProcTime;
    char lastTime[BASE_TIMER_LAST_PROC_TIME_LEN];
} BaseTimerMntInfo;

typedef struct {
    void *appHandle;
    uint32_t appCid;
    void *memHandle;
    EpollCtrCallBack epollCtlCB;
    TimerMallocCB mallocCB;
    TimerFreeCB freeCB;
} TimerAppInfo;

typedef struct {
    uint32_t magic;
    int32_t tmrFd;
    BaseTimerUsrCb usrCb;
    uint32_t timeoutMs;
    uint32_t delTag;
    BaseTimerMntInfo mntInfo;
    BaseTimerParam *optParam;
    TimerAppInfo appInfo;
} BaseTimerInfo;

#define TIMER_FNAME (__FUNCTION__)
#define TIMER_FLINE (__LINE__)
#define Timer_MemAlloc(d_appInfo, d_size)                                                                              \
    (d_appInfo)->mallocCB((d_appInfo)->appHandle, (d_size), TIMER_FNAME, TIMER_FLINE)
#define Timer_MemFree(d_appInfo, d_pointer)                                                                            \
    (d_appInfo)->freeCB((d_appInfo)->appHandle, (d_pointer), TIMER_FNAME, TIMER_FLINE)
void BASE_TimerCopyAppInfo(TimerAppInfo *dest, TimerAppInfo *source);

BaseTimerAttr BASE_TimerCreateAttr(TimerAppInfo *appInfo);

void BASE_TimerDestroyAttr(TimerAppInfo *appInfo, BaseTimerAttr attr);

uint32_t BASE_TimerSetAttr(BaseTimerAttr attr, BaseTimerAttrType type, void *value, size_t len);

/* 定时器创建：创建成功或者失败都会尝试销毁attr，即此接口调用后无需用户手动销毁attr */
BaseTimerHandle BASE_TimerCreate(TimerAppInfo *appInfo, BaseTimerUsrCb *usrCb, uint32_t timeoutMs, BaseTimerAttr attr);

void BASE_TimerDestroy(BaseTimerHandle handle);

uint32_t BASE_TimerStart(BaseTimerHandle handle);

uint32_t BASE_TimerStop(BaseTimerHandle handle);

uint32_t BASE_TimerResize(BaseTimerHandle handle, uint32_t timeoutMs);

uint32_t BASE_TimerGetRemainTime(BaseTimerHandle handle, uint32_t *remainMs);

void BaseTimerDestroy(BaseTimerInfo *tmrInfo);
#pragma pack()

#ifdef __cplusplus
}
#endif

#endif
